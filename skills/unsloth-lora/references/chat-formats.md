# Chat formats

`train.py` needs to know how the base model's chat template writes a turn: the loss is computed only after the
marker that starts the assistant's reply, and `run.json` records the layout so a server can reproduce it
(`run-json.md`). `scripts/chat_format.py` holds one `ChatFormat` per family:

- `turn_prefix` (with `{role}`) and `turn_suffix` around each message,
- `reply_prefix`, which starts the assistant's reply,
- `stop`, the strings that end a reply.

Only ChatML (`<|im_start|>{role}\n` … `<|im_end|>\n`) is included so far. The repository's tests check it against
the real templates of Qwen2.5, Qwen3, Qwen3.5 and SmolLM3, and a trained model has been checked for Qwen2.5.

## System messages

The system turn a server has to send is taken from the template's own rendering, not from the data:

- **No system messages in the data**: many templates (Qwen2.5 among them) insert a default system message, so the
  model was trained with it. Others (Qwen3) insert none. Some write dates or knowledge cutoffs into the default,
  check that `train.py`'s printed system turn is what you want served.
- **One shared system message**: recorded as the template rendered it. Some templates rewrite it, SmolLM3 for
  example wraps it into a metadata block, and the whole block is what the server has to send.
- **Mixed**: nothing is recorded, every request has to bring its own. With a template that rewrites system
  messages that can't match training, `train.py` warns: use one shared message or none.

## Thinking templates

Qwen3 and Qwen3.5 render an empty `<think>\n\n</think>\n\n` in front of the last answer of every training
conversation, so the fine-tuned model learns to start its answers with it. `train.py` warns and `run.json` records
`empty_think`; the app using the model has to strip the block.

## Adding a family

1. Render a conversation with the model's real tokenizer:
   `tokenizer.apply_chat_template([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
   tokenize=False)`, and one with a system message.
2. Add a `ChatFormat` to `FORMATS` with its turn prefix, turn suffix, reply prefix and stop strings.
3. Add the model's template to the repository's `tests/fixtures/chat-templates/` and to `TEMPLATES` in
   `tests/test_handoff.py`, which checks that the prompt a server builds from `run.json` is exactly the training
   text, with and without a system message and over two turns.
4. Train a 2-step smoke run and check the "tokens that receive loss" output: only the answer.
