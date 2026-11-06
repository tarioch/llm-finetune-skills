#!/usr/bin/env python
"""Check the training machine before a run: GPU memory actually free, the training environment importable, disk.

Run it with the Python of the Unsloth environment (the one ``train.py`` will use):

    python preflight.py                     # GPU, environment, disk in the current folder
    python preflight.py --need-gib 8 --out outputs/my-run

Exits non-zero when training would fail or crawl: too little free GPU memory (other processes, often games or a
second model loaded in Ollama, are listed), Unsloth or torch not importable, CUDA not available, too little disk.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def parse_csv(text: str) -> list[list[str]]:
    return [[c.strip() for c in row] for row in csv.reader(text.strip().splitlines()) if row]


def gpus() -> list[dict]:
    """Free and total memory per GPU, in MiB, from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        {"index": i, "name": name, "used": int(used), "total": int(total)} for i, name, used, total in parse_csv(out)
    ]


def gpu_processes() -> list[tuple[str, str]]:
    """(process name, MiB) of compute processes; graphics processes such as games only show in plain nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [(row[0], row[1]) for row in parse_csv(out) if len(row) == 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need-gib", type=float, default=8.0, help="free GPU memory the run needs (default %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("."), help="where checkpoints and the GGUF will be written")
    ap.add_argument("--need-disk-gib", type=float, default=30.0, help="free disk at --out (default %(default)s)")
    args = ap.parse_args()
    problems = []

    try:
        found = gpus()
    except (OSError, subprocess.CalledProcessError) as e:
        found = []
        problems.append(f"nvidia-smi failed ({e}), no NVIDIA GPU or driver?")
    for g in found:
        free = (g["total"] - g["used"]) / 1024
        print(f"GPU {g['index']} {g['name']}: {free:.1f} of {g['total'] / 1024:.1f} GiB free")
    if found and max((g["total"] - g["used"]) / 1024 for g in found) < args.need_gib:
        problems.append(f"less than {args.need_gib:g} GiB GPU memory free")
        for name, mib in gpu_processes():
            print(f"  using the GPU: {name} ({mib} MiB)")
        print("  (games and other graphics processes only show in plain `nvidia-smi`; `ollama ps` lists loaded models)")

    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    for module in ("torch", "unsloth", "trl", "datasets"):
        try:
            mod = importlib.import_module(module)
            print(f"{module} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001 - any import failure means the environment is broken
            problems.append(f"cannot import {module}: {e}")
    torch = sys.modules.get("torch")
    if torch is not None and not torch.cuda.is_available():
        problems.append("torch does not see a CUDA device (CPU-only torch build?)")

    args.out.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.out).free / 2**30
    print(f"disk free at {args.out.resolve()}: {disk:.0f} GiB")
    if disk < args.need_disk_gib:
        problems.append(
            f"less than {args.need_disk_gib:g} GiB disk free (the merge for GGUF export needs 16-bit weights)"
        )

    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    print("ok" if not problems else "not ready")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
