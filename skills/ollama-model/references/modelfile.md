# The Modelfile: keeping Ollama and training in agreement

A fine-tuned model has learned one exact text layout. If Ollama renders a conversation differently, nothing fails,
the answers just get worse. `scripts/modelfile.py` builds the Modelfile from the `chat` section of `run.json`
(`run-json.md`), so the layout comes from what training recorded, not from knowledge of the model family.

## What goes where

| `run.json` `chat` | Modelfile |
|---|---|
| `turn_prefix`, `turn_suffix`, `reply_prefix` | `TEMPLATE`: each of `.Messages` in the training layout, then the reply prefix |
| `system` | `SYSTEM`, left out when it is `null` |
| `stop` | `PARAMETER stop` |

`SYSTEM` is the system turn exactly as the chat template rendered it in training: the template's default when the
data had no system messages (Qwen2.5 has one, Qwen3 none), the data's shared message as the template rewrote it
(SmolLM3 wraps it into a metadata block), or nothing when the data mixed system messages.

## Pitfalls, each one seen in practice

- **System prompt twice**: Ollama passes the Modelfile's `SYSTEM` both as the first message and as `.System`. A
  template that prints both feeds it twice. The generated template reads only `.Messages`.
- **CRLF**: Ollama keeps a `\r` inside a template or system prompt. A Modelfile checked out with CRLF on Windows
  changes the tokens the model sees. `modelfile.py` writes bytes; pin committed ones with `.gitattributes`.
- **BOS token**: llama.cpp adds the BOS token itself, so the template must not print it (e.g. Llama 3's
  `<|begin_of_text|>`); `run.json`'s markers don't contain it.
- **Empty think block**: a model trained from a thinking template (Qwen3, Qwen3.5) starts its answers with an
  empty `<think></think>` block (`empty_think` in `run.json`). Ollama doesn't strip it with this template, the app
  has to; `evaluate.py` strips it before checking.
- **Temperature**: the Modelfile's `PARAMETER temperature` applies to Ollama's native API, the OpenAI-compatible
  `/v1` endpoint ignores it.

## Troubleshooting

- **Answers differ from what training produced**: regenerate the Modelfile with `modelfile.py`, check its line
  endings, and check the request's temperature.
- **`ollama push` is refused**: the machine's public key (`~/.ollama/id_ed25519.pub`) has to be added to the
  ollama.com account, and the model name has to start with that account's namespace (`ollama cp` to rename).
