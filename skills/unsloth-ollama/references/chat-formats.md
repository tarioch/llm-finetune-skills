# Chat formats: keeping Ollama and training in agreement

A fine-tuned model has learned one exact text layout. If Ollama renders a conversation differently, nothing fails,
the answers just get worse. `scripts/chat_formats.py` holds what has to agree for each supported family.

## What has to match

| Training (Unsloth, the model's chat template) | Ollama (Modelfile) |
|---|---|
| the chat template's turn markers | `TEMPLATE`, built from `.Messages` |
| the system message the data has, or the template's default when it has none | `SYSTEM` |
| the end-of-turn token | `PARAMETER stop` |

How the Modelfile's `SYSTEM` is chosen (`chat_formats.modelfile_system`): it is the system turn exactly as the chat
template wrote it in training, because the Ollama template writes the system message unchanged.

- **No system messages in the data**: many chat templates (Qwen2.5 among them) insert a default system message on
  their own, so the model was trained with it. The Modelfile bakes that same text in. Others (Qwen3) insert none.
- **One shared system message**: the Modelfile bakes it in as the template wrote it. Some templates rewrite it,
  SmolLM3 for example wraps it into a metadata block, and the Modelfile then holds the whole block.
- **Mixed**: no `SYSTEM`, every caller has to send the right system message itself. `train.py` warns, and warns
  again when the template rewrites system messages, because then a caller's message can't match training.

## Pitfalls, each one seen in practice

- **System prompt twice**: Ollama passes the Modelfile's `SYSTEM` both as the first message and as `.System`. A
  template that prints both feeds it twice. The templates here read only `.Messages`.
- **CRLF**: Ollama keeps a `\r` inside a template or system prompt. A Modelfile checked out with CRLF on Windows
  changes the tokens the model sees. Write it as bytes, pin it with `.gitattributes`.
- **Template defaults differ per family**: some chat templates write dates or knowledge cutoffs into their default
  system message. `train.py` prints the detected default, check it is what you want baked in.
- **BOS token**: llama.cpp adds the BOS token itself, so an Ollama template must not print it (e.g. Llama 3's
  `<|begin_of_text|>`).
- **Temperature**: the Modelfile's `PARAMETER temperature` applies to Ollama's native API, the OpenAI-compatible
  `/v1` endpoint ignores it.

## Adding a family

Only ChatML is included so far. The tests check it against the real templates of Qwen2.5, Qwen3, Qwen3.5 and SmolLM3
(`tests/unsloth-ollama/scripts/fixtures/`), a trained model has been checked for Qwen2.5. A new ChatML model needs only its template added
to the fixtures. Thinking models (Qwen3, Qwen3.5) render an empty `<think>\n\n</think>\n\n` in front of the last answer of
every training conversation, so the fine-tuned model learns to start its answers with it. The prompt still matches,
but the empty block comes back as part of the answer: `evaluate.py` strips it, and the app using the model has to
strip it too (`train.py` warns when it sees one in the first example).

1. Render a conversation with the model's real tokenizer:
   `tokenizer.apply_chat_template([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
   tokenize=False)`, and one with a system message.
2. Add a `ChatFormat` to `FORMATS` with the user, assistant and system markers, the end-of-turn token, an Ollama
   template that produces the same text from `.Messages`, and the stop strings.
3. Add the model's template to `tests/unsloth-ollama/scripts/fixtures/` and to `TEMPLATES` in
   `tests/unsloth-ollama/scripts/test_chat_formats.py`, which checks
   that the Ollama prompt is exactly the training text, with and without a system message and over two turns.
4. Train a 2-step smoke run and check the "tokens that receive loss" output: only the answer.
