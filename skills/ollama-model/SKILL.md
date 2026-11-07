---
name: ollama-model
description: Package a fine-tuned GGUF as an Ollama model with a Modelfile that prompts it exactly like training did, compare it with other Ollama models on a held-out set, and publish it to ollama.com with tags. Use after training a model (e.g. with the unsloth-lora skill), to compare model versions or sizes, or to push a model to Ollama.
allowed-tools: Bash(python ${CLAUDE_SKILL_DIR}/scripts/modelfile.py *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/modelfile.py *) Bash(python ${CLAUDE_SKILL_DIR}/scripts/evaluate.py *) Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py *) Bash(uv run python ${CLAUDE_SKILL_DIR}/scripts/evaluate.py *) Bash(ollama create *) Bash(ollama run *) Bash(ollama stop *) Bash(ollama pull *) Bash(ollama ps) Bash(ollama list)
---

# Ollama model from a fine-tuned GGUF

Turn a GGUF into an Ollama model that sees exactly the prompt layout it was trained on, check it against the
previous model, and publish it. The input is a run folder with a GGUF and a `run.json` whose `chat` section records
the training layout (written by the `unsloth-lora` skill; for a GGUF from elsewhere, write it by hand following
`references/run-json.md`).

The scripts in `scripts/` next to this file need only the standard library and a running Ollama.

Run them in the Bash tool in the form the steps show (`python ${CLAUDE_SKILL_DIR}/scripts/<script>.py`, `python3`
works too) and use absolute paths: this skill pre-approves those commands and the `ollama` commands of steps 1 and 2
(`allowed-tools`), so they run without a permission prompt, also in auto mode. `ollama cp`, `ollama push` and
`ollama rm` are left out on purpose and keep asking. The approval holds for the turn the skill is invoked in: when
the user has sent a message since, invoke the skill again first.

## Before you start, find out

- The run folder(s), and the evaluation set: held out from training, same format as the training data.
- The model it should beat (usually the version currently published) and the Ollama name to publish under
  (`<user>/<model>`) with the tags in use. Ask before pushing anything.

## Steps

### 1. Write the Modelfile and create the model

```bash
python ${CLAUDE_SKILL_DIR}/scripts/modelfile.py --run <runs>/<run>
ollama create <name>:<tag> -f <runs>/<run>/Modelfile
```

The Modelfile is generated from `run.json`, never write one by hand: a template or system message that differs
from training changes the answers without any error (`references/modelfile.md`). Its `FROM` is relative to the
Modelfile, so it can be committed without a machine path. Pass on the warnings it prints,
in particular that the app has to strip an empty `<think></think>` block. Try one question with
`ollama run <name>:<tag> "<question>"` to see that the answer has the trained shape.

### 2. Compare

```bash
python ${CLAUDE_SKILL_DIR}/scripts/evaluate.py --eval /abs/eval.jsonl --model <name>:<new> --model <name>:<previous> --out <runs>/<run>/eval.jsonl
```

Use a domain check (`--check checks.py:function`) when exact match is too strict, e.g. execute a generated query
and compare the results. When the check needs the project's own packages, run evaluate.py in the project's
environment, in one command: `cd <project> && uv run python ${CLAUDE_SKILL_DIR}/scripts/evaluate.py ...`. Show the
user the scores and a few examples where the models disagree, and let them decide what ships. With several base models, this is where the size/quality trade-off gets decided. The last model stays
loaded in Ollama for a few minutes: `ollama stop <model>` before training again, so its GPU memory is free.

### 3. Publish (only after the user says so)

```bash
ollama cp <name>:<new> <user>/<model>:<tag>
ollama push <user>/<model>:<tag>
```

Pushing needs the machine's Ollama key added to the user's ollama.com account. Moving tags like `latest` are a
copy plus push of the same model. `ollama push` does not set the model page's readme, that's pasted on ollama.com.
Write release notes from `run.json` (base model, data checksums, steps, losses) and the evaluation result, so the
model can be traced back to the data it was trained on.

### 4. Check it where it's used

Ask the app that uses the model one of the evaluation questions. Ollama's OpenAI-compatible `/v1` endpoint ignores
the Modelfile's `temperature`, so clients have to send it with each request.

## Rules

- Keep the Modelfile generated and LF-only (`.gitattributes`: `Modelfile text eol=lf`).
- Ask before pushing to ollama.com, moving a tag, or removing models (`ollama rm`).
