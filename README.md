# llm-finetune-skills

[Claude](https://claude.com/claude-code) skills for fine-tuning open LLMs on your own hardware and serving them.

| Skill | What it does |
|---|---|
| [`unsloth-lora`](skills/unsloth-lora/SKILL.md) | Trains a LoRA with [Unsloth](https://unsloth.ai) on a local NVIDIA GPU from a chat-format JSONL dataset and exports a GGUF, with a `run.json` recording how training laid out prompts. |
| [`ollama-model`](skills/ollama-model/SKILL.md) | Turns a GGUF and its `run.json` into an [Ollama](https://ollama.com) model whose Modelfile matches training, compares it with other models on a held-out set, and publishes it. |

The two work on their own: a GGUF trained elsewhere can go to Ollama (write its `run.json` by hand), and a model
trained with Unsloth can be served by something else. The handoff between them is the run folder, described in
[run-json.md](skills/unsloth-lora/references/run-json.md).

Producing the training data is specific to each problem and stays in your own project. Training starts from
`train.jsonl` and `val.jsonl`, one conversation per line:

```json
{"messages": [{"role": "user", "content": "How much did I spend on food in 2024?"}, {"role": "assistant", "content": "SELECT ..."}]}
```

## unsloth-lora

1. **Data check** (`check_data.py`): structure, alternating turns, examples shared between training and validation
   or evaluation, how system messages are used.
2. **Preflight** (`preflight.py`): free GPU memory (and who else uses it), the training environment, disk space.
3. **Training** (`train.py`): QLoRA with loss on the answers only. Prints the first example as the model sees it,
   the tokens that get loss and token lengths. Writes `run.json` with arguments, data checksums, losses and the
   chat layout.
4. **Export** (`export_gguf.py`): GGUF via Unsloth/llama.cpp.

Supported chat format: ChatML, the `<|im_start|>`/`<|im_end|>` format of Qwen2.5, Qwen3, Qwen3.5, SmolLM3 and many
others, checked against their real templates ([tests/fixtures/chat-templates](tests/fixtures/chat-templates/README.md)).
Other formats (Llama 3, Gemma, Mistral, ...) need an entry in `chat_format.py`, see
[chat-formats.md](skills/unsloth-lora/references/chat-formats.md).

## ollama-model

1. **Modelfile** (`modelfile.py`): template, system message and stop strings built from `run.json`, so Ollama sends
   the model exactly the prompt layout it was trained on.
2. **Evaluation** (`evaluate.py`): several Ollama models side by side on a held-out set, with exact match or your
   own check function.
3. **Publishing**: tags, `ollama push`, release notes from `run.json`.

## Install

In Claude Code, add this repository as a plugin marketplace and install the plugin, which brings both skills:

```text
/plugin marketplace add tarioch/llm-finetune-skills
/plugin install llm-finetune@llm-finetune-skills
```

Or copy a skill folder from `skills/` into `~/.claude/skills/` (personal) or `.claude/skills/` (one project). For
claude.ai, zip a skill folder and upload it under Settings, Capabilities, Skills.

Install the plugin yourself: in auto mode, Claude Code refuses to add a marketplace or install a plugin on Claude's
own initiative.

### Permissions

Each skill pre-approves the commands it runs (`allowed-tools` in its SKILL.md), so they run without a permission
prompt, also in auto mode, where Claude Code would otherwise refuse to run scripts it didn't write:

- `unsloth-lora`: its launcher `scripts/run <script>` (which runs that script with the Unsloth Python),
  `nvidia-smi`, `ollama ps` and `ollama stop`.
- `ollama-model`: `python`/`python3` and `uv run python` on `modelfile.py` and `evaluate.py`, and `ollama create`,
  `run`, `stop`, `pull`, `ps` and `list`. Publishing (`ollama cp`, `push`) and `ollama rm` still ask.

The approval holds for the turn in which the skill is invoked, so Claude invokes the skill again for a step that
comes after a message of yours.

## Requirements

- `unsloth-lora`: an NVIDIA GPU and a Python environment with Unsloth installed (see the
  [Unsloth installation docs](https://docs.unsloth.ai/get-started/installing-+-updating)). `check_data.py` needs
  only the standard library. The scripts run through `scripts/run`, a Bash script (on Windows, Claude Code's Git
  Bash), which takes the Unsloth Python from `$UNSLOTH_PYTHON` or the first line of
  `~/.config/llm-finetune/unsloth-python`; Claude asks for the path and writes that file when neither is set.
- `ollama-model`: Ollama. The scripts need only the standard library.

## License

[MIT](LICENSE). The chat templates in `tests/fixtures/chat-templates/` belong to their models (Apache 2.0), see
[their README](tests/fixtures/chat-templates/README.md).
