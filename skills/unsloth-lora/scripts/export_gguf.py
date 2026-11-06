#!/usr/bin/env python
"""Merge the LoRA adapter of a ``train.py`` run into its base model and export it as a GGUF.

Uses Unsloth's ``save_pretrained_gguf``: it merges into 16-bit weights, then converts and quantizes with llama.cpp
(a prebuilt copy at ``~/.unsloth/llama.cpp`` or ``UNSLOTH_LLAMA_CPP_PATH``; Unsloth builds one if none is found,
which needs network access and a C/C++ toolchain). Needs disk for the 16-bit merge on top of the GGUF itself.

Run it with the Python of the Unsloth environment:

    python export_gguf.py --run outputs/my-3b                  # -> outputs/my-3b/gguf/*.gguf, recorded in run.json
    python export_gguf.py --run outputs/my-3b --quant q5_k_m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Unsloth patches transformers/trl when it is imported, so it has to come before them.
from unsloth import FastLanguageModel  # noqa: E402


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

    # relative to the run folder when it is inside it (some Unsloth versions write to <out>_gguf/ instead)
    gguf, run_dir = gguf.resolve(), args.run.resolve()
    path = gguf.relative_to(run_dir).as_posix() if gguf.is_relative_to(run_dir) else str(gguf)
    run["gguf"] = {"path": path, "quant": args.quant}
    run_file.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(
        f"recorded it in {run_file}\n\nNext: package it for a server, e.g. the ollama-model skill:\n"
        f"  python modelfile.py --run {args.run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
