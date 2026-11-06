#!/usr/bin/env python
"""Compare Ollama models on a held-out evaluation set, side by side.

The evaluation set has the training format: each line's ``messages`` ends with the expected assistant answer, the
messages before it are sent to each model. An answer is right when the check says so: by default the answer and
the expectation are equal after whitespace normalization; ``--check file.py:function`` plugs in a domain check
(``function(prompt_messages, expected, answer) -> bool``), e.g. run the generated query and compare the rows.

    python evaluate.py --eval eval.jsonl --model my-model:3b --model my-model:7b --model my-model:previous
    python evaluate.py --eval eval.jsonl --model my-model:3b --check checks.py:same_rows --out results.jsonl

Uses Ollama's native ``/api/chat`` with temperature 0 sent in every request. Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

Check = Callable[[list[dict], str, str], bool]


def normalized_equal(prompt: list[dict], expected: str, answer: str) -> bool:
    return " ".join(expected.split()) == " ".join(answer.split())


def load_check(spec: str | None) -> Check:
    if not spec:
        return normalized_equal
    path, _, name = spec.rpartition(":")
    if not path or not name:
        raise SystemExit(f"--check needs file.py:function, got {spec!r}")
    module_spec = importlib.util.spec_from_file_location("user_check", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)


def ollama_chat(host: str, model: str, messages: list[dict], timeout: float) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}).encode()
    request = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - the user's Ollama host
        return json.load(response)["message"]["content"]


def evaluate(
    examples: list[list[dict]], models: list[str], ask: Callable[[str, list[dict]], str], check: Check
) -> list[dict]:
    results = []
    for i, conversation in enumerate(examples):
        prompt, expected = conversation[:-1], conversation[-1]["content"]
        for model in models:
            start = time.monotonic()
            try:
                answer, error = ask(model, prompt), None
                ok = bool(check(prompt, expected, answer))
            except Exception as e:  # noqa: BLE001 - one failing example must not end the comparison
                answer, error, ok = "", f"{type(e).__name__}: {e}", False
            results.append(
                {
                    "example": i,
                    "model": model,
                    "ok": ok,
                    "seconds": round(time.monotonic() - start, 2),
                    "question": prompt[-1]["content"],
                    "expected": expected,
                    "answer": answer,
                    "error": error,
                }
            )
    return results


def summary(results: list[dict], models: list[str]) -> str:
    lines = []
    for model in models:
        mine = [r for r in results if r["model"] == model]
        right = sum(r["ok"] for r in mine)
        seconds = sum(r["seconds"] for r in mine) / max(len(mine), 1)
        lines.append(f"{model}: {right}/{len(mine)} right, {seconds:.1f} s per answer")
    by_example: dict[int, dict[str, bool]] = {}
    for r in results:
        by_example.setdefault(r["example"], {})[r["model"]] = r["ok"]
    differ = [i for i, oks in by_example.items() if len(set(oks.values())) > 1]
    if len(models) > 1:
        lines.append(f"{len(differ)} examples where the models disagree: {differ[:20]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True, help="JSONL, each conversation ends with the expected answer")
    ap.add_argument("--model", action="append", required=True, help="an Ollama model, repeat to compare")
    ap.add_argument("--check", help="file.py:function deciding whether an answer is right")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, help="only the first N examples")
    ap.add_argument("--out", type=Path, help="write every answer as JSONL")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        examples = [json.loads(line)["messages"] for line in f if line.strip()][: args.limit]
    results = evaluate(
        examples,
        args.model,
        lambda model, messages: ollama_chat(args.host, model, messages, args.timeout),
        load_check(args.check),
    )
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    print(summary(results, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
