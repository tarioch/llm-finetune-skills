#!/usr/bin/env python
"""Write release notes for a model from its ``run.json`` and the evaluation report.

The notes say what the model was trained from (base model, data checksums, settings, losses), which file was
published (GGUF quantization, size, checksum) and how it compared (the Markdown ``evaluate.py --report`` wrote), so
a published model can be traced back to its data.

    python release_notes.py --run outputs/my-3b --eval outputs/my-3b/eval.md --name me/my-model:3b
    python release_notes.py --run outputs/my-3b -o outputs/my-3b/RELEASE.md

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SETTINGS = ("epochs", "max_steps", "max_seq_length", "batch_size", "grad_accum", "lr", "rank", "seed")


def _number(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def notes(run: dict, evaluation: str | None = None, name: str | None = None) -> str:
    out = [f"# {name}" if name else "# Release notes", ""]
    out += [f"Fine-tuned from `{run['base_model']}` with a LoRA.", ""]

    out += ["## Training", ""]
    examples = run.get("examples", {})
    if examples:
        out.append("- Examples: " + ", ".join(f"{count} {split}" for split, count in examples.items()))
    args = run.get("args", {})
    settings = [f"{key.replace('_', ' ')} {_number(args[key])}" for key in SETTINGS if key in args]
    settings = [s for s in settings if s != "max steps -1"]
    if settings:
        out.append("- Settings: " + ", ".join(settings))
    for key, label in (("steps", "Steps"), ("train_loss", "Training loss"), ("eval_loss", "Validation loss")):
        if run.get(key) is not None:
            out.append(f"- {label}: {_number(run[key])}")
    if run.get("peak_gpu_gib") is not None:
        out.append(f"- Peak GPU memory: {_number(run['peak_gpu_gib'])} GiB")
    out.append("")

    if run.get("data"):
        out += ["## Data", "", "| File | SHA-256 |", "| --- | --- |"]
        out += [f"| `{file}` | `{digest}` |" for file, digest in run["data"].items()]
        out.append("")

    gguf = run.get("gguf")
    if gguf:
        out += ["## Model file", ""]
        out.append(f"- `{Path(gguf['path']).name}`, {gguf['quant'].upper()}")
        if gguf.get("bytes"):
            out.append(f"- Size: {gguf['bytes'] / 2**30:.2f} GiB ({gguf['bytes']} bytes)")
        if gguf.get("sha256"):
            out.append(f"- SHA-256: `{gguf['sha256']}`")
        out.append("")

    chat = run.get("chat", {})
    usage = []
    if chat.get("system") is None and chat.get("system_in_data") == "mixed":
        usage.append("Send a system message with every request: training used several different ones.")
    if chat.get("empty_think"):
        usage.append("Answers start with an empty `<think></think>` block, strip it in the application.")
    if usage:
        out += ["## Using it", "", *(f"- {line}" for line in usage), ""]

    if evaluation:
        out += ["## Evaluation", "", _demote_headings(evaluation.strip()), ""]
    return "\n".join(out)


def _demote_headings(markdown: str) -> str:
    """The report's own headings go one level below the notes' ``## Evaluation``."""
    return "\n".join("#" + line if line.startswith("#") else line for line in markdown.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run folder with run.json")
    ap.add_argument("--eval", type=Path, help="the Markdown report of evaluate.py --report")
    ap.add_argument("--name", help="the published model name, used as the title")
    ap.add_argument("-o", "--out", type=Path, help="write to this file instead of printing")
    args = ap.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    evaluation = args.eval.read_text(encoding="utf-8") if args.eval else None
    text = notes(run, evaluation, args.name)
    if args.out:
        args.out.write_bytes(text.encode("utf-8"))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
