#!/usr/bin/env python
"""Show how far a training run is, from what ``train.py`` leaves in the run folder.

Reads the newest ``<run>/checkpoints/checkpoint-*/trainer_state.json`` (written at every checkpoint, so it is as
recent as the last ``--save-steps``) and ``<run>/run.json`` once the run has finished. Prints the step, the latest
training and validation loss, the step with the lowest validation loss, and an estimate of the time left from the
times the checkpoints were written. Use it to check a run in the background instead of reading its whole log.

    python train_status.py outputs/my-3b

Standard library only, so it runs with any Python while the training holds the GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

CHECKPOINT = re.compile(r"checkpoint-(\d+)$")


def checkpoints(run: Path) -> list[tuple[int, Path]]:
    """``(step, folder)`` for each checkpoint, oldest first."""
    found = []
    for folder in (run / "checkpoints").glob("checkpoint-*"):
        match = CHECKPOINT.search(folder.name)
        if match and (folder / "trainer_state.json").exists():
            found.append((int(match[1]), folder))
    return sorted(found)


def status(run: Path, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    record = run / "run.json"
    if record.exists():
        done = json.loads(record.read_text(encoding="utf-8"))
        return {
            "state": "finished",
            "step": done.get("steps"),
            "train_loss": done.get("train_loss"),
            "eval_loss": done.get("eval_loss"),
            "exported": "gguf" in done,
        }
    found = checkpoints(run)
    if not found:
        return {"state": "no checkpoint yet"}
    step, folder = found[-1]
    state = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
    history = state.get("log_history", [])
    train = [h for h in history if "loss" in h]
    evals = [h for h in history if "eval_loss" in h]
    best = min(evals, key=lambda h: h["eval_loss"], default=None)
    result = {
        "state": "running or stopped",
        "step": state.get("global_step", step),
        "max_steps": state.get("max_steps"),
        "train_loss": train[-1]["loss"] if train else None,
        "eval_loss": evals[-1]["eval_loss"] if evals else None,
        "best_eval": (best["step"], best["eval_loss"]) if best else None,
        "last_checkpoint_age": now - (folder / "trainer_state.json").stat().st_mtime,
    }
    if len(found) > 1 and result["max_steps"]:
        (old_step, old), (new_step, new) = found[-2], found[-1]
        seconds = (new / "trainer_state.json").stat().st_mtime - (old / "trainer_state.json").stat().st_mtime
        if new_step > old_step and seconds > 0:
            per_step = seconds / (new_step - old_step)
            result["seconds_per_step"] = per_step
            result["seconds_left"] = max(result["max_steps"] - result["step"], 0) * per_step
    return result


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def describe(s: dict) -> str:
    if s["state"] == "no checkpoint yet":
        return "no checkpoint yet: the run hasn't reached its first --save-steps, or hasn't started"
    if s["state"] == "finished":
        losses = [
            f"{label} {s[key]:.4f}"
            for key, label in (("train_loss", "train loss"), ("eval_loss", "validation loss"))
            if s[key] is not None
        ]
        return ", ".join(
            [f"finished after {s['step']} steps", *losses, "exported" if s["exported"] else "not exported yet"]
        )
    parts = [f"step {s['step']}/{s['max_steps']}" if s["max_steps"] else f"step {s['step']}"]
    if s["train_loss"] is not None:
        parts.append(f"train loss {s['train_loss']:.4f}")
    if s["eval_loss"] is not None:
        parts.append(f"validation loss {s['eval_loss']:.4f}")
    if s["best_eval"]:
        parts.append(f"lowest validation loss {s['best_eval'][1]:.4f} at step {s['best_eval'][0]}")
    if "seconds_left" in s:
        parts.append(f"about {_minutes(s['seconds_left'])} left ({s['seconds_per_step']:.1f} s per step)")
    parts.append(f"last checkpoint {_minutes(s['last_checkpoint_age'])} ago")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run folder given to train.py --out")
    args = ap.parse_args()
    if not args.run.is_dir():
        print(f"no run folder {args.run}", file=sys.stderr)
        return 2
    print(describe(status(args.run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
