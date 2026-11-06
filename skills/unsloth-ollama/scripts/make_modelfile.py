#!/usr/bin/env python
"""Write the Ollama Modelfile for a ``train.py`` run, so Ollama renders prompts exactly like training did.

``export_gguf.py`` calls this already; run it by hand for a GGUF exported some other way, or to change a parameter:

    python make_modelfile.py --run outputs/my-3b --gguf outputs/my-3b/gguf/model.Q4_K_M.gguf
    python make_modelfile.py --run outputs/my-3b --gguf model.gguf --temperature 0.7 -o Modelfile

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_formats import FORMATS, render_modelfile  # noqa: E402


def write_modelfile(run: dict, gguf: Path, out: Path, temperature: float | None = 0.0) -> Path:
    text = render_modelfile(str(gguf), FORMATS[run["chat_format"]], run["modelfile_system"], temperature)
    out.write_bytes(text.encode("utf-8"))  # bytes: no CRLF translation on Windows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run folder train.py wrote (run.json)")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, help="default: <run>/Modelfile")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="baked-in default (%(default)s). Ollama's OpenAI-compatible /v1 ignores it: send it with each request",
    )
    args = ap.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    out = write_modelfile(run, args.gguf.resolve(), args.out or args.run / "Modelfile", args.temperature)
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
