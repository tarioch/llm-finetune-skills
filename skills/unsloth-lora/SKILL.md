---
name: unsloth-lora
description: Fine-tune an instruct LLM with a LoRA on a local NVIDIA GPU using Unsloth, from a chat-format JSONL dataset, and export it as a GGUF together with a run.json recording what a server needs to prompt it like training did. Use for training or retraining a model, or trying several base models.
---

# Unsloth LoRA training

Train a LoRA on a chat-format dataset with Unsloth on a local GPU and export the result as a GGUF. The dataset is
the user's (producing it is problem-specific); this skill starts from `train.jsonl` and `val.jsonl`. Serving the
model is a separate step: the run folder this skill leaves (GGUF plus `run.json`) is what a serving skill such as
`ollama-model` takes.

All scripts are in `scripts/` next to this file. `check_data.py` needs only the standard library; `preflight.py`,
`train.py` and `export_gguf.py` run with the Python of the Unsloth environment. Run them by path, from the folder
where the runs should live, not from a git checkout: Unsloth writes an `unsloth_compiled_cache/` folder into the
current directory.

## Before you start, find out

- The dataset folder, and the evaluation set used later to compare models: it must not be part of training. If
  there is none, propose holding out examples before training.
- The base model(s), as Unsloth 4-bit names (e.g. `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`). Several base
  models means one run folder each and the same steps for each.
- The machine: training needs a CUDA GPU and the Unsloth environment's Python. If you run on another computer
  (remote session), confirm which Python is the Unsloth one before anything else.

## Steps

### 1. Check the data

```bash
python scripts/check_data.py data/ --eval eval.jsonl
```

Must print `ok`. It fails on broken lines, conversations that don't end with an assistant answer, and validation or
evaluation examples that are also in training. It warns when a question repeats with a different answer: show the
user the example, it can be legitimate (the answer depends on context) or a contradiction in the data. Once the user
says such repeats are not intended, use `--strict` so they fail the check. It also
reports how system messages are used: none, one shared, or mixed (see `references/chat-formats.md`).

### 2. Preflight the machine

```bash
python scripts/preflight.py --need-gib 8 --out outputs/<run>
```

Free GPU memory matters more than total: games, a browser with GPU acceleration, or a model a local server still
holds (`ollama ps`, unload with `ollama stop <model>`) can leave too little. If preflight lists other GPU users,
tell the user and let them decide, don't kill processes. Roughly: a 3B model in 4-bit trains in 6 to 8 GiB at
sequence length 1024 and batch size 4, a 7B in 10 to 12 GiB; lower `--batch-size` and raise `--grad-accum` to fit.

If an import fails, the environment is broken, see `references/troubleshooting.md` before reinstalling anything.

### 3. Smoke run

```bash
python scripts/train.py --model <base> --data data --out outputs/<run> --max-steps 2
```

Read the output, don't just check the exit code:

- the first example as rendered: the system line at the top is what the model will learn to expect,
- the tokens that receive loss: only the answer, not the question,
- token lengths: a truncation warning means raising `--max-seq-length` or dropping examples,
- the system turn a server has to send, and the warnings about it. A thinking template (Qwen3, Qwen3.5) puts an
  empty `<think></think>` block in front of every trained answer: tell the user that the app will get it in
  answers and has to strip it.

### 4. Full run

```bash
python scripts/train.py --model <base> --data data --out outputs/<run>
```

Runs for a while (minutes for a few thousand examples on a 3B model, longer for 7B). Run it in the background with
the output going to a log file, instead of holding a session open, and check it with
`python scripts/train_status.py outputs/<run>`: step, losses, the step with the lowest validation loss and the time
left, as of the last checkpoint. After a crash use `--resume`.
Report train and validation loss and the peak GPU memory. A validation loss far above the training loss means
overfitting: fewer epochs or more data.

### 5. Export

```bash
python scripts/export_gguf.py --run outputs/<run>
```

Writes the GGUF and records it in `run.json` with its size and checksum. It removes the merged 16-bit model
(several GB, `--keep-merged` keeps it) and the Modelfile Unsloth writes next to the GGUF: that one uses the stock
template, not the layout training used, so don't follow Unsloth's hint to `ollama create` from it. If export prints
a `[CRITICAL]` warning about llama.cpp's converter, see `references/troubleshooting.md`. The run folder is now
ready for serving: continue with the
`ollama-model` skill, or hand `run.json`'s `chat` section to whatever serves the model
(`references/run-json.md`).

## Rules

- Never change the dataset to make an evaluation look better, and never evaluate on training data.
- Don't install or upgrade packages in the Unsloth environment without asking: its torch/CUDA combination is fragile.
- Ask before deleting run folders or checkpoints.
