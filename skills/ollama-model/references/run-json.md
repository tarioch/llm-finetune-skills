# run.json

`train.py` writes `<run>/run.json`, `export_gguf.py` adds the GGUF. It is the handoff to whatever serves the model:
the `chat` section says how training laid out a conversation, so a server can build exactly the same prompt.
This file is the same in the `unsloth-lora` and `ollama-model` skills.

```json
{
  "base_model": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
  "args": {"epochs": 2.0, "max_seq_length": 1024, "rank": 32, "...": "every train.py argument"},
  "data": {"train.jsonl": "<sha256>", "val.jsonl": "<sha256>"},
  "examples": {"train": 6321, "val": 263},
  "train_loss": 0.21,
  "eval_loss": 0.24,
  "steps": 792,
  "peak_gpu_gib": 7.5,
  "chat": {
    "format": "chatml",
    "turn_prefix": "<|im_start|>{role}\n",
    "turn_suffix": "<|im_end|>\n",
    "reply_prefix": "<|im_start|>assistant\n",
    "stop": ["<|im_end|>"],
    "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "system_in_data": "none",
    "template_rewrites_system": false,
    "empty_think": false
  },
  "gguf": {
    "path": "gguf_gguf/qwen2.5-coder-3b-instruct.Q4_K_M.gguf",
    "quant": "q4_k_m",
    "bytes": 1929935584,
    "sha256": "<sha256 of the GGUF>"
  }
}
```

## The chat section

A prompt is every message written as `turn_prefix` (with `{role}` replaced by the role) + content + `turn_suffix`,
followed by `reply_prefix`, after which the model writes its answer and ends it with one of `stop`.

| Field | Meaning |
|---|---|
| `format` | name of the family's layout in the trainer (informational) |
| `turn_prefix`, `turn_suffix`, `reply_prefix`, `stop` | the layout above, exact strings |
| `system` | the system turn's content exactly as training rendered it, to send when a request has none; `null` when there is none to send |
| `system_in_data` | `none` (the data had no system messages, `system` is the template's own default or `null`), `same` (one shared message, `system` is it as the template rendered it), or `mixed` (`system` is `null`, every request has to bring its own) |
| `template_rewrites_system` | the chat template changes a given system message (e.g. SmolLM3 wraps it), so with `mixed` a request's own message won't look like training |
| `empty_think` | the template put an empty `<think></think>` block in front of every trained answer, so the model writes one and the app has to strip it |

The GGUF's `path` is relative to the run folder, or absolute when the GGUF was written elsewhere. `bytes` and
`sha256` identify the exact file, for release notes.
