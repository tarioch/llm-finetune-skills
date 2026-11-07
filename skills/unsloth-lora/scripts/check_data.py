#!/usr/bin/env python
"""Check a chat-format training set before spending GPU time on it.

Expects ``train.jsonl`` and ``val.jsonl``, one conversation per line: ``{"messages": [{"role": ..., "content": ...},
...]}`` with roles ``system`` (optional, first only), ``user`` and ``assistant``, ending with an assistant turn.

Fails on structural errors and on validation or evaluation prompts that are also in training, with the same or a
different answer: the model has seen that prompt, so its loss or score there measures recall, not generalization.
The fix is to take them out of validation or evaluation, training keeps its data. Prints what else matters for
training: counts, how system messages are used, the longest examples in characters (``train.py`` reports the exact
token counts).

    python check_data.py data/            # a folder with train.jsonl and val.jsonl
    python check_data.py --eval eval.jsonl data/   # also check an evaluation set for overlap with training

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_format import data_system  # noqa: E402

ROLES = {"system", "user", "assistant"}


def load(path: Path) -> tuple[list[list[dict]], list[str]]:
    """The conversations in a JSONL file and the structural errors found, as ``file:line: problem``."""
    conversations, errors = [], []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path.name}:{n}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: not JSON ({e})")
                continue
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                errors.append(f"{where}: no 'messages' list")
                continue
            problems = check_messages(messages)
            errors += [f"{where}: {p}" for p in problems]
            if not problems:
                conversations.append(messages)
    return conversations, errors


def check_messages(messages: list) -> list[str]:
    problems = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            problems.append(f"message {i} needs a role ({', '.join(sorted(ROLES))}) and string content")
            continue
        if not m["content"].strip():
            problems.append(f"message {i} ({m['role']}) is empty")
        if m["role"] == "system" and i != 0:
            problems.append(f"system message at position {i}, only the first message can be one")
    if problems:
        return problems
    turns = [m["role"] for m in messages if m["role"] != "system"]
    if not turns or turns[-1] != "assistant":
        problems.append("does not end with an assistant message")
    if any(a == b for a, b in zip(turns, turns[1:], strict=False)):
        problems.append("user and assistant turns do not alternate")
    if turns and turns[0] != "user":
        problems.append("the first non-system message is not from the user")
    return problems


def prompt_key(messages: list[dict]) -> str:
    """Everything before the final answer, which is what must not repeat between training and validation."""
    return json.dumps([(m["role"], m["content"].strip()) for m in messages[:-1]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="folder with train.jsonl and val.jsonl")
    ap.add_argument("--eval", type=Path, help="an evaluation set (same format) to check for overlap with training")
    args = ap.parse_args()

    sets: dict[str, list[list[dict]]] = {}
    errors: list[str] = []
    paths = {"train": args.data / "train.jsonl", "val": args.data / "val.jsonl"}
    if args.eval:
        paths["eval"] = args.eval
    for name, path in paths.items():
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        sets[name], errs = load(path)
        errors += errs
        if not sets[name] and not errs:
            errors.append(f"{path} is empty")

    train_answers: dict[str, set[str]] = {}
    for c in sets.get("train", []):
        train_answers.setdefault(prompt_key(c), set()).add(c[-1]["content"].strip())
    for name in ("val", "eval"):
        same, other = [], []
        for c in sets.get(name, []):
            answers = train_answers.get(prompt_key(c))
            if answers is not None:
                (same if c[-1]["content"].strip() in answers else other).append(c[-2]["content"])
        if same or other:
            example = (same or other)[0]
            errors.append(
                f"{len(same) + len(other)} {name} prompts are also in training ({len(same)} with the same answer, "
                f"{len(other)} with a different one), take them out of {name}: e.g. {example[:100]!r}"
            )

    for name, conversations in sets.items():
        usage, text = data_system(conversations)
        system = {"none": "no system messages", "mixed": "different system messages"}.get(usage, f"system: {text!r}")
        longest = max((sum(len(m["content"]) for m in c) for c in conversations), default=0)
        turns = max((sum(m["role"] == "user" for m in c) for c in conversations), default=0)
        print(f"{name}: {len(conversations)} conversations, {system}, up to {turns} turns, longest {longest} chars")
    if "train" in sets and "val" in sets and data_system(sets["train"]) != data_system(sets["val"]):
        errors.append("train and val use system messages differently")

    for e in errors[:50]:
        print(f"ERROR {e}", file=sys.stderr)
    if len(errors) > 50:
        print(f"... and {len(errors) - 50} more errors", file=sys.stderr)
    print("ok" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
