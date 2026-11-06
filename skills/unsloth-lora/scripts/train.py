#!/usr/bin/env python
"""Fine-tune an instruct model on a chat-format dataset with Unsloth (QLoRA), loss on the answers only.

Run it with the Python of the Unsloth environment:

    python train.py --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit --data data --out outputs/my-3b --max-steps 2
    python train.py --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit --data data --out outputs/my-3b
    python train.py ... --resume          # continue from the latest checkpoint after a crash

Everything that decides what the model learns is printed instead of hidden: the first example exactly as the
model sees it, token length statistics (with a warning when examples would be truncated), and the tokens that
receive loss. ``<out>/run.json`` records the run (arguments, data checksums, losses) and the ``chat`` section a server
needs to render prompts like training did (see ``references/run-json.md``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Unsloth patches transformers/trl when it is imported, so it has to come before them.
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

# isort: split

import torch  # noqa: E402
from chat_format import FORMATS, chat_record, detect  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_examples(path: Path) -> list[list[dict]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="base model, e.g. unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit")
    ap.add_argument("--data", type=Path, required=True, help="folder with train.jsonl and val.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="run folder: checkpoints/, adapter/, run.json")
    ap.add_argument("--format", choices=sorted(FORMATS), help="chat format (default: detected from the template)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (alpha is set equal to it)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100, help="checkpoint every this many steps (last 2 kept)")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in <out>/checkpoints")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-response-only", action="store_true", help="also train on the prompt tokens")
    args = ap.parse_args()

    for name in ("train.jsonl", "val.jsonl"):
        if not (args.data / name).exists():
            print(f"missing {args.data / name}", file=sys.stderr)
            return 2

    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    def render(msgs: list[dict]) -> str:
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    train_msgs, val_msgs = read_examples(args.data / "train.jsonl"), read_examples(args.data / "val.jsonl")
    train_text, val_text = [render(m) for m in train_msgs], [render(m) for m in val_msgs]
    fmt = FORMATS[args.format] if args.format else detect(train_text[0])
    chat = chat_record(fmt, render, train_msgs)

    print("\n=== first training example, exactly as the model will see it ===")
    print(train_text[0])
    print("=== end of example ===\n")
    print(f"chat format {fmt.name}; system messages in the data: {chat['system_in_data']}")
    print(f"system turn a server has to send: {chat['system']!r}")
    if chat["system_in_data"] == "mixed":
        print(
            "WARNING: the examples use different system messages, so there is no fixed system turn and every "
            "request has to send the right one",
            file=sys.stderr,
        )
        if chat["template_rewrites_system"]:
            print(
                "WARNING: this chat template rewrites system messages, so a request's own system message will not "
                "look like training. Use one shared system message or none.",
                file=sys.stderr,
            )
    if chat["empty_think"]:
        print(
            "WARNING: the chat template puts an empty <think></think> block in front of the answer, the model will "
            "learn to start its answers with it and the app using the model has to strip it",
            file=sys.stderr,
        )
    if chat["system"] is not None and chat["system"] != chat["system"].strip():
        print(
            "WARNING: the system turn starts or ends with whitespace, check that the server keeps it",
            file=sys.stderr,
        )

    lengths = sorted(len(ids) for ids in tokenizer(train_text, add_special_tokens=False)["input_ids"])
    p99 = lengths[int(len(lengths) * 0.99)]
    print(
        f"{len(train_text)} train / {len(val_text)} validation examples; tokens per example: "
        f"median {lengths[len(lengths) // 2]}, p99 {p99}, max {lengths[-1]}"
    )
    if lengths[-1] > args.max_seq_length:
        print(
            f"WARNING: {sum(n > args.max_seq_length for n in lengths)} examples are longer than --max-seq-length "
            f"{args.max_seq_length} and will be truncated",
            file=sys.stderr,
        )

    bf16 = is_bfloat16_supported()
    config = SFTConfig(
        output_dir=str(args.out / "checkpoints"),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        dataset_num_proc=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=0.03,  # a float below 1 means 3% of all steps
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=Dataset.from_dict({"text": train_text}),
        eval_dataset=Dataset.from_dict({"text": val_text}),
    )
    if not args.no_response_only:
        trainer = train_on_responses_only(trainer, instruction_part=fmt.prefix("user"), response_part=fmt.reply_prefix)
    first = trainer.train_dataset[0]
    if "labels" in first:
        # labels are -100 wherever the model is not trained
        supervised = [t for t, label in zip(first["input_ids"], first["labels"], strict=True) if label != -100]
        print(f"=== the {len(supervised)} of {len(first['input_ids'])} tokens of the first example with loss ===")
        print(tokenizer.decode(supervised))
        print("=== end ===\n")
        if not supervised:
            print("no token receives loss, the response marker does not match the template", file=sys.stderr)
            return 1

    result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    metrics = trainer.evaluate()
    print(f"\ntrain loss {result.training_loss:.4f}   validation loss {metrics['eval_loss']:.4f}")
    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_reserved() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"peak GPU memory reserved: {peak:.2f} GiB of {total:.2f} GiB")

    adapter = args.out / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    run = {
        "base_model": args.model,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data": {name: sha256(args.data / name) for name in ("train.jsonl", "val.jsonl")},
        "examples": {"train": len(train_text), "val": len(val_text)},
        "train_loss": result.training_loss,
        "eval_loss": metrics["eval_loss"],
        "steps": result.global_step,
        "peak_gpu_gib": peak,
        "chat": chat,
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"saved the LoRA adapter to {adapter} and the run record to {args.out / 'run.json'}")
    print(f"\nNext: python export_gguf.py --run {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
