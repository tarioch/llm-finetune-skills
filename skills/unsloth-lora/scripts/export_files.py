"""The file handling around a GGUF export, kept out of ``export_gguf.py`` so it can be tested without a GPU.

Unsloth's ``save_pretrained_gguf(out)`` writes the merged 16-bit model to ``out`` and, depending on the version,
the GGUF to ``out`` or to ``out_gguf`` next to it, together with a Modelfile of its own. That Modelfile uses the base
model's stock template, not the layout training used, so it is removed: ``ollama-model``'s ``modelfile.py`` writes
the right one from ``run.json``.

Standard library only.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def find_ggufs(out: Path, reported: list[str]) -> list[Path]:
    """The GGUF files of an export: what Unsloth reported, else what is in ``out`` or ``out_gguf``."""
    found = [Path(p) for p in reported]
    for folder in (out, out.with_name(out.name + "_gguf")):
        found = found or sorted(folder.glob("*.gguf"))
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finish(run_dir: Path, out: Path, gguf: Path, quant: str, keep_merged: bool = False) -> tuple[dict, list[str]]:
    """Record the GGUF for run.json and tidy up the export; returns the record and what was done, for printing."""
    notes = []
    gguf, run_dir, out = gguf.resolve(), run_dir.resolve(), out.resolve()
    stray = gguf.parent / "Modelfile"
    if stray.exists():
        stray.unlink()
        notes.append(f"removed {stray}: Unsloth's own Modelfile uses the stock template, not the trained layout")
    # the merged 16-bit weights, several GB, are only needed for the conversion
    if not keep_merged and out.is_dir() and out != gguf.parent and not any(out.glob("*.gguf")):
        size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        shutil.rmtree(out)
        notes.append(f"removed the merged 16-bit model in {out} ({size / 2**30:.1f} GiB), --keep-merged keeps it")
    path = gguf.relative_to(run_dir).as_posix() if gguf.is_relative_to(run_dir) else str(gguf)
    record = {"path": path, "quant": quant, "bytes": gguf.stat().st_size, "sha256": sha256(gguf)}
    return record, notes
