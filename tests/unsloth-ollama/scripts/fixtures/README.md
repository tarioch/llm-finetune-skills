# Chat template fixtures

Real chat templates, used to check that the Ollama template and the Modelfile's `SYSTEM` reproduce what training
renders. Each is the `chat_template` of the model's tokenizer, unchanged except for line endings and a final newline.

| File | Model | Source | License |
|---|---|---|---|
| `Qwen2.5-Coder-Instruct.jinja` | Qwen2.5-Coder Instruct (same template for all sizes) | saved by Unsloth with a trained adapter | Apache 2.0 |
| `Qwen3.jinja` | Qwen/Qwen3-0.6B | llama.cpp `models/templates/Qwen-Qwen3-0.6B.jinja` at 669204ee | Apache 2.0 |
| `Qwen3.5.jinja` | Qwen3.5-4B | llama.cpp `models/templates/Qwen3.5-4B.jinja` at 669204ee | Apache 2.0 |
| `SmolLM3.jinja` | HuggingFaceTB/SmolLM3-3B | llama.cpp `models/templates/HuggingFaceTB-SmolLM3-3B.jinja` at 669204ee | Apache 2.0 |

To add one, copy the template from the model's `tokenizer_config.json` (`chat_template`) or its
`chat_template.jinja`, or from llama.cpp's `models/templates/`, and add it to `TEMPLATES` in
`tests/unsloth-ollama/scripts/test_chat_formats.py` with the default system message it inserts.
