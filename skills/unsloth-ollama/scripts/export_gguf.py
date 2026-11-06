#!/usr/bin/env python
"""Merge the LoRA adapter of a ``train.py`` run into its base model, export a GGUF and write a matching Modelfile.

Uses Unsloth's ``save_pretrained_gguf``: it merges into 16-bit weights, then converts and quantizes with llama.cpp
(a prebuilt copy at ``~/.unsloth/llama.cpp`` or ``UNSLOTH_LLAMA_CPP_PATH``; Unsloth builds one if none is found,
which needs network access and a C/C++ toolchain). Needs disk for the 16-bit merge on top of the GGUF itself.

Run it with the Python of the Unsloth environment:

    python export_gguf.py --run outputs/my-3b                  # -> outputs/my-3b/gguf/*.gguf and Modelfile
    python export_gguf.py --run outputs/my-3b --quant q5_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Unsloth patches transformers/trl when it is imported, so it has to come before them.
from unsloth import FastLanguageModel  # noqa: E402

# isort: split

from make_modelfile import write_modelfile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run folder train.py wrote (adapter/, run.json)")
    ap.add_argument("--quant", default="q4_k_m", help="llama.cpp quantization method (default: %(default)s)")
    args = ap.parse_args()

    adapter, run_file = args.run / "adapter", args.run / "run.json"
    if not adapter.exists() or not run_file.exists():
        print(f"{args.run} has no adapter/ and run.json from train.py", file=sys.stderr)
        return 2
    run = json.loads(run_file.read_text(encoding="utf-8"))
    out = args.run / "gguf"

    print(f"loading {adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(adapter), max_seq_length=int(run["args"]["max_seq_length"]), dtype=None, load_in_4bit=True
    )
    print(f"merging and exporting a {args.quant} GGUF to {out} ...")
    result = model.save_pretrained_gguf(str(out), tokenizer, quantization_method=args.quant)

    gguf_files = [Path(p) for p in result.get("gguf_files", [])] if isinstance(result, dict) else []
    gguf_files = gguf_files or sorted(out.glob("*.gguf"))
    if not gguf_files:
        print("export finished but no .gguf file was found, check the output above", file=sys.stderr)
        return 1
    gguf = gguf_files[0]
    print(f"\nwrote {gguf} ({gguf.stat().st_size / 2**30:.2f} GiB)")

    modelfile = write_modelfile(run, gguf.resolve(), args.run / "Modelfile")
    run["gguf"] = {"file": gguf.name, "quant": args.quant}
    run_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {modelfile}\n\nNext:\n"
        f'  ollama create <name>:<tag> -f "{modelfile}"\n'
        "  python evaluate.py --eval eval.jsonl --model <name>:<tag> --model <previous model>"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
