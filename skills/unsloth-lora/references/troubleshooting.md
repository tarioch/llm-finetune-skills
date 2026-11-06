# Troubleshooting

## The environment

- **Python of the environment is gone** (`No Python at ...`, or `python` from the environment exits at once): a
  virtual environment points at its base interpreter in `pyvenv.cfg` (`home = ...`). If that interpreter was
  uninstalled or upgraded, install the same minor version (for example `uv python install 3.13`), back up
  `pyvenv.cfg` and point `home` at the new interpreter's folder. Reinstalling Unsloth is the last resort, its
  torch/CUDA combination is hard to get right again.
- **torch does not see the GPU**: a CPU-only torch wheel got installed, often by a later `pip install` of
  something that depends on torch. Check `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- **Import of unsloth fails after an upgrade**: pin back to the versions that worked (keep a
  `pip freeze > freeze.txt` of a working environment).

## Training

- **CUDA out of memory**: check `nvidia-smi` for other users of the GPU first (games, Ollama holding a model:
  `ollama ps`). Then lower `--batch-size` and raise `--grad-accum` by the same factor (same effective batch), then
  `--max-seq-length` if examples allow it.
- **No token receives loss / loss is 0**: the response marker doesn't match the chat template, the format detection
  picked the wrong family or the family is missing. See `chat-formats.md`.
- **Loss goes to near zero quickly, validation loss rises**: overfitting or duplicated data. Fewer epochs, check
  `check_data.py`'s overlap output.
- **Very slow steps**: on Windows, the first steps compile kernels. If it stays slow, the GPU may be shared with a
  display-heavy process, or memory spills into shared system memory (Windows does that instead of failing), check
  "Shared GPU memory" in Task Manager.

## Harmless notices

- `torchcodec`/FFmpeg warnings when Unsloth is imported: audio support is off, text training doesn't need it.
- `The tokenizer has new PAD/BOS/EOS tokens ... {'bos_token_id': None}`: Qwen models have no BOS token, Unsloth
  only reports aligning the config with the tokenizer.
- Deprecation warnings from transformers (`use_return_dict`, `has_cuda` and similar).

## Export

- **Unsloth reports `[CRITICAL] Harvests environment variables/secrets AND makes network calls`** about llama.cpp's
  `gguf-py/gguf/utility.py`: that is llama.cpp's code for reading safetensors from Hugging Face, which sends
  `HF_TOKEN` if one is set. The export doesn't need it (it converts local files). Tell the user; to use a converter
  they reviewed, point `UNSLOTH_LLAMA_CPP_SCRIPTS_DIR` at it.

- **llama.cpp missing or build fails**: Unsloth builds llama.cpp on first export, which needs network access and a
  C/C++ toolchain. Point `UNSLOTH_LLAMA_CPP_PATH` at a prebuilt copy instead.
- **Disk full during export**: the merge writes 16-bit weights first (about 2 bytes per parameter, so ~15 GB for a
  7B model) before quantizing.
