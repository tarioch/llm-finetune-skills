# llm-finetune-skills

[Claude](https://claude.com/claude-code) skills for fine-tuning open LLMs on your own hardware and serving them.

| Skill | What it does |
|---|---|
| [`unsloth-ollama`](skills/unsloth-ollama/SKILL.md) | Trains a LoRA with [Unsloth](https://unsloth.ai) on a local NVIDIA GPU from a chat-format JSONL dataset, exports a GGUF, writes an [Ollama](https://ollama.com) Modelfile that matches training, compares the new model with others on a held-out set, and publishes it. |

Producing the training data is specific to each problem and stays in your own project. The skills start from
`train.jsonl` and `val.jsonl`, one conversation per line:

```json
{"messages": [{"role": "user", "content": "How much did I spend on food in 2024?"}, {"role": "assistant", "content": "SELECT ..."}]}
```

## What the unsloth-ollama skill takes care of

1. **Data check** (`check_data.py`): structure, alternating turns, questions shared between training and validation
   or evaluation, how system messages are used.
2. **Preflight** (`preflight.py`): free GPU memory (and who else uses it), the training environment, disk space.
3. **Training** (`train.py`): QLoRA with loss on the answers only. Prints the first example as the model sees it,
   the tokens that get loss and token lengths. Writes `run.json` with arguments, data checksums and losses.
4. **Export** (`export_gguf.py`, `make_modelfile.py`): GGUF via Unsloth/llama.cpp and a generated Modelfile whose
   template, system message and stop tokens agree with training.
5. **Evaluation** (`evaluate.py`): several Ollama models side by side on a held-out set, with exact match or your
   own check function.
6. **Publishing**: tags, `ollama push`, release notes from `run.json`.

Supported chat formats: ChatML (Qwen2.5, all sizes). Adding one is described in
[references/chat-formats.md](skills/unsloth-ollama/references/chat-formats.md).

## Install

In Claude Code, add this repository as a plugin marketplace and install the plugin:

```text
/plugin marketplace add tarioch/llm-finetune-skills
/plugin install llm-finetune@llm-finetune-skills
```

Or copy `skills/unsloth-ollama` into `~/.claude/skills/` (personal) or `.claude/skills/` (one project). For
claude.ai, zip the `skills/unsloth-ollama` folder and upload it under Settings, Capabilities, Skills.

## Requirements

- An NVIDIA GPU and a Python environment with Unsloth installed (see the
  [Unsloth installation docs](https://docs.unsloth.ai/get-started/installing-+-updating)). Only `preflight.py`,
  `train.py` and `export_gguf.py` need it; the other scripts use the standard library.
- Ollama, for packaging, evaluating and publishing.

## License

[MIT](LICENSE). `tests/fixtures/qwen2.5.jinja` is Qwen2.5's chat template, from the Qwen2.5 model repositories
(Apache 2.0), used to test that the generated Modelfile matches it.
