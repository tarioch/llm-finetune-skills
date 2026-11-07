---
name: unsloth-lora
description: Fine-tune an instruct LLM with a LoRA on a local NVIDIA GPU using Unsloth, from a chat-format JSONL dataset, and export it as a GGUF together with a run.json recording what a server needs to prompt it like training did. Use for training or retraining a model, or trying several base models.
allowed-tools: Bash(${CLAUDE_SKILL_DIR}/scripts/run *), Bash(nvidia-smi), Bash(ollama ps), Bash(ollama stop *)
---

# Unsloth LoRA training

Train a LoRA on a chat-format dataset with Unsloth on a local GPU and export the result as a GGUF. The dataset is
the user's (producing it is problem-specific); this skill starts from `train.jsonl` and `val.jsonl`. Serving the
model is a separate step: the run folder this skill leaves (GGUF plus `run.json`) is what a serving skill such as
`ollama-model` takes.

Run every script through the launcher, with the Bash tool: `${CLAUDE_SKILL_DIR}/scripts/run <script.py> [args]`.
The skill allows exactly that command, so it runs in auto mode; calling a script another way gets refused there. If
a launcher command is refused after a new message from the user, invoke this skill again. The launcher runs
`check_data.py` with any Python 3 and `preflight.py`, `train.py` and `export_gguf.py` with the Unsloth
environment's Python, in the parent folder of their `--out`/`--run` folder (Unsloth writes an
`unsloth_compiled_cache/` folder into the current directory). Give every path as an absolute path: commands don't
share a working directory.

## Before you start, find out

- The dataset folder, and the evaluation set used later to compare models: it must not be part of training. If
  there is none, propose holding out examples before training.
- The base model(s), as Unsloth 4-bit names (e.g. `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`). Several base
  models means one run folder each and the same steps for each.
- The machine: training needs a CUDA GPU and the Unsloth environment's Python. Ask the user for that Python's path
  if `UNSLOTH_PYTHON` isn't set and `~/.config/llm-finetune/unsloth-python` doesn't exist, and write the path as the
  first line of that file (with their OK): the launcher reads it from there.
- The runs folder, where the run folders go, outside any git checkout. `<runs>` below is its absolute path.

## Steps

### 1. Check the data

```bash
${CLAUDE_SKILL_DIR}/scripts/run check_data.py <data> --eval <eval.jsonl>
```

Must print `ok`. It fails on broken lines, conversations that don't end with an assistant answer, and validation or
evaluation prompts that are also in training, with the same or a different answer: the model has seen them, so they
measure recall. The fix is to take them out of validation or evaluation (or fix how the data is split), never to
change training to make the check pass. It also
reports how system messages are used: none, one shared, or mixed (see `references/chat-formats.md`).

### 2. Preflight the machine

```bash
${CLAUDE_SKILL_DIR}/scripts/run preflight.py --params-b <billions of parameters> --out <runs>/<run>
```

Free GPU memory matters more than total: games, a browser with GPU acceleration, or a model a local server still
holds (`ollama ps`, unload with `ollama stop <model>`) can leave too little. If preflight lists other GPU users,
tell the user and let them decide, don't kill processes. With `--params-b` it estimates the need (a 3B model in
4-bit peaked at 6.6 GiB at sequence length 1024 and batch size 4, a 7B at 8.1 GiB) and, when the free memory is
less, prints the `--batch-size`/`--grad-accum` pair to use instead. It is an estimate from two runs: for other
sizes or sequence lengths pass `--need-gib` and watch the peak the smoke run reports.

If an import fails, the environment is broken, see `references/troubleshooting.md` before reinstalling anything.

### 3. Smoke run

```bash
${CLAUDE_SKILL_DIR}/scripts/run train.py --model <base> --data <data> --out <runs>/<run>-smoke --max-steps 2
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
${CLAUDE_SKILL_DIR}/scripts/run train.py --model <base> --data <data> --out <runs>/<run>
```

Runs for a while (minutes for a few thousand examples on a 3B model, longer for 7B). Start it with the Bash tool's
background option and the output going to a log file in `<runs>`, instead of holding a session open, and check it
with `${CLAUDE_SKILL_DIR}/scripts/run train_status.py <runs>/<run>`: step, losses, the step with the lowest
validation loss and the time left, as of the last checkpoint. After a crash use `--resume`.
Report train and validation loss and the peak GPU memory. A validation loss far above the training loss means
overfitting: fewer epochs or more data.

### 5. Export

```bash
${CLAUDE_SKILL_DIR}/scripts/run export_gguf.py --run <runs>/<run>
```

Takes a few minutes for a 3B model and more than ten for a 7B one (merging, converting, quantizing), which is longer
than the Bash tool lets a foreground command run: start it in the background like the training, with the output in a
log file in `<runs>`, and wait for it to finish. It writes the GGUF to `<run>/gguf/` and records it in `run.json`
with its size and checksum. It removes the merged
16-bit model (several GB, `--keep-merged` keeps it, and then the GGUF stays in `<run>/gguf_gguf/`)
and the Modelfile Unsloth writes next to the GGUF: that one uses the stock template, not the layout training used,
so don't follow Unsloth's hint to `ollama create` from it. Expect a `[CRITICAL]` warning about llama.cpp's
converter on every export; tell the user what it means (`references/troubleshooting.md`), it doesn't stop the
export. The run folder is now ready for serving: continue with the `ollama-model` skill, or hand `run.json`'s
`chat` section to whatever serves the model (`references/run-json.md`).

### 6. Clean up (ask first)

Once the model is exported and evaluated, offer to delete what is only needed to resume or to check the setup:
`<run>/checkpoints/` (for `--resume`, several GB for a 7B model) and the `<run>-smoke` folder. Say how much space
each takes and delete only what the user agrees to. Keep `adapter/` and `run.json`: they are small, and they are
what a new export and the release notes need.

## Rules

- Never change the dataset to make an evaluation look better, and never evaluate on training data.
- Don't install or upgrade packages in the Unsloth environment without asking: its torch/CUDA combination is fragile.
- Ask before deleting run folders or checkpoints.
