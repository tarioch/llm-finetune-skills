---
name: unsloth-ollama
description: Fine-tune an instruct LLM with a LoRA on a local NVIDIA GPU using Unsloth, export it to GGUF, package it for Ollama with a Modelfile that matches training, compare it against other models, and push it to ollama.com. Use for training, retraining, comparing base models or publishing a fine-tuned model to Ollama, given a chat-format JSONL dataset.
---

# Unsloth LoRA to Ollama

Train a LoRA on a chat-format dataset with Unsloth on a local GPU, turn it into an Ollama model that sees exactly
what training saw, check it against the previous model, and publish it. The dataset itself is the user's
(generating it is problem-specific); this skill starts from `train.jsonl` and `val.jsonl`.

All scripts are in `scripts/` next to this file. `check_data.py`, `make_modelfile.py` and `evaluate.py` need only
the standard library; `preflight.py`, `train.py` and `export_gguf.py` run with the Python of the Unsloth
environment. Run them by path, from the folder where the run should live (`outputs/` by default).

## Before you start, find out

- The dataset folder, and an evaluation set that is not part of training (ask for one; if there is none, propose
  holding out examples before training, never evaluate on `val.jsonl` examples the user tuned against).
- The base model(s), as Unsloth 4-bit names (e.g. `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`). Several base
  models means one run folder each, the same steps for each, and one comparison at the end.
- The machine: training needs a CUDA GPU and the Unsloth environment's Python. If you run on another computer
  (remote session), confirm which Python is the Unsloth one before anything else.
- The Ollama name to publish under (`<user>/<model>`) and the tags in use. Ask before pushing anything.

## Steps

### 1. Check the data

```bash
python scripts/check_data.py data/ --eval eval.jsonl
```

Must print `ok`. It fails on broken lines, conversations that don't end with an assistant answer, and validation or
evaluation examples that are also in training. It warns when a question repeats with a different answer: show the
user the example, it can be legitimate (the answer depends on context) or a contradiction in the data. Note how system messages are used ("no system messages", one
shared system message, or mixed): it decides the Modelfile's `SYSTEM`, see `references/chat-formats.md`.

### 2. Preflight the machine

```bash
python scripts/preflight.py --need-gib 8 --out outputs/<run>
```

Free GPU memory matters more than total: games, a browser with GPU acceleration, or a model Ollama still holds
(`ollama ps`, unload with `ollama stop <model>`) can leave too little. If preflight lists other GPU users, tell the
user and let them decide, don't kill processes. Roughly: a 3B model in 4-bit trains in 6 to 8 GiB at sequence
length 1024 and batch size 4, a 7B in 10 to 12 GiB; lower `--batch-size` and raise `--grad-accum` to fit.

If an import fails, the environment is broken, see `references/troubleshooting.md` before reinstalling anything.

### 3. Smoke run

```bash
python scripts/train.py --model <base> --data data --out outputs/<run> --max-steps 2
```

Read the output, don't just check the exit code:

- the first example as rendered: the system line at the top is what the model will learn to expect,
- `chat format ...; system messages in the data: ...; template default: ...`,
- the tokens that receive loss: only the answer, not the question,
- token lengths: a truncation warning means raising `--max-seq-length` or dropping examples,
- the `Modelfile SYSTEM` line and any warnings about it, and the empty `<think></think>` warning of thinking
  templates (Qwen3, Qwen3.5): tell the user that the app will have to strip that block from answers.

### 4. Full run

```bash
python scripts/train.py --model <base> --data data --out outputs/<run>
```

Runs for a while (minutes for a few thousand examples on a 3B model, longer for 7B). Run it in the background with
the output going to a log file and check the log, instead of holding a session open. After a crash use `--resume`.
Report train and validation loss and the peak GPU memory. A validation loss far above the training loss means
overfitting: fewer epochs or more data.

### 5. Export and package

```bash
python scripts/export_gguf.py --run outputs/<run>
ollama create <name>:<tag> -f outputs/<run>/Modelfile
```

`export_gguf.py` writes the GGUF and a Modelfile generated from `run.json`, never write one by hand. Try one
question with `ollama run <name>:<tag> "<question>"` to see that the answer has the trained shape.

### 6. Compare

```bash
python scripts/evaluate.py --eval eval.jsonl --model <name>:<new> --model <name>:<previous> --out outputs/<run>/eval.jsonl
```

Use a domain check (`--check checks.py:function`) when exact match is too strict, e.g. execute a generated query
and compare the results. Show the user the scores and a few examples where the models disagree, and let them decide
what ships. With several base models, this is where the size/quality trade-off gets decided.

### 7. Publish (only after the user says so)

```bash
ollama cp <name>:<new> <user>/<model>:<tag>
ollama push <user>/<model>:<tag>
```

Pushing needs the machine's Ollama key added to the user's ollama.com account. Moving tags like `latest` are a
copy plus push of the same model. `ollama push` does not set the model page's readme, that's pasted on ollama.com.
Write release notes from `run.json` (base model, data checksums, steps, losses) and the evaluation result, so the
model can be traced back to the data it was trained on.

### 8. Check it where it's used

Ask the app that uses the model one of the evaluation questions. Ollama's OpenAI-compatible `/v1` endpoint ignores
the Modelfile's `temperature`, so clients have to send it with each request.

## Rules

- Never change the dataset to make the evaluation look better, and never evaluate on training data.
- Keep the Modelfile generated: a template or system message that differs from training changes the model's
  behaviour without any error. Keep it LF-only (`.gitattributes`: `Modelfile text eol=lf`).
- Ask before pushing to ollama.com, moving a tag, or deleting run folders and models.
- Don't install or upgrade packages in the Unsloth environment without asking: its torch/CUDA combination is fragile.
