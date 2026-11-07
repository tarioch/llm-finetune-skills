#!/usr/bin/env python
"""Compare Ollama models on a held-out evaluation set, side by side.

The evaluation set has the training format: each line's ``messages`` ends with the expected assistant answer, the
messages before it are sent to each model. An answer is right when the check says so: by default the answer and
the expectation are equal after whitespace normalization; ``--check file.py:function`` plugs in a domain check
(``function(prompt_messages, expected, answer) -> bool``), e.g. run the generated query and compare the rows.

    python evaluate.py --eval eval.jsonl --model my-model:3b --model my-model:7b --model my-model:previous
    python evaluate.py --eval eval.jsonl --model my-model:3b --check checks.py:same_rows --out results.jsonl
    python evaluate.py --eval eval.jsonl --model my-model:new --model my-model:old --report outputs/new/eval.md

Uses Ollama's native ``/api/chat`` with temperature 0 sent in every request. An empty ``<think></think>`` block at
the start of an answer (models fine-tuned from a thinking template learn to write one) is removed before the check.
Per model it reports the score, the median time per answer, the generation speed and the size Ollama lists. With
several models, each one is compared with the first: on how many examples only one of the two is right, which says
more on a small set than two scores. ``--report`` writes the same as Markdown, with the first examples where the
models disagree, for the user and for ``release_notes.py``. Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

Check = Callable[[list[dict], str, str], bool]
EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*")


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


def ollama_chat(host: str, model: str, messages: list[dict], timeout: float) -> dict:
    """The answer and Ollama's generation counters: ``tokens`` generated in ``token_seconds``."""
    body = json.dumps({"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}).encode()
    request = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - the user's Ollama host
        reply = json.load(response)
    return {
        "content": reply["message"]["content"],
        "tokens": reply.get("eval_count"),
        "token_seconds": reply["eval_duration"] / 1e9 if reply.get("eval_duration") else None,
    }


def ollama_sizes(host: str, timeout: float) -> dict[str, int]:
    """Bytes per model as ``/api/tags`` lists them; empty when that fails, the size is only informational."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:  # noqa: S310
            return {m["name"]: m["size"] for m in json.load(response)["models"]}
    except (OSError, ValueError, KeyError):
        return {}


Answer = str | dict


def checked(check: Check, prompt: list[dict], expected: str, answer: str) -> bool:
    """Run the check, then put back ``sys.stdout`` and ``sys.stderr`` if it replaced them.

    Libraries a check uses can wrap the streams on every call: TatSu (beanquery's parser) calls ``colorama.init()``
    on every parse, which on Windows wraps ``sys.stdout`` once more each time. After a few hundred answers, printing
    the summary recursed through all the wrappers and the run died with a RecursionError, the summary unprinted.
    """
    streams = sys.stdout, sys.stderr
    try:
        return bool(check(prompt, expected, answer))
    finally:
        sys.stdout, sys.stderr = streams


def evaluate(
    examples: list[list[dict]], models: list[str], ask: Callable[[str, list[dict]], Answer], check: Check
) -> list[dict]:
    """Asks one model all examples before the next model: Ollama keeps one model loaded instead of swapping per answer.

    The results are ordered by example, then model, as given."""
    results = []
    for model in models:
        for i, conversation in enumerate(examples):
            prompt, expected = conversation[:-1], conversation[-1]["content"]
            start = time.monotonic()
            tokens = token_seconds = None
            try:
                reply = ask(model, prompt)
                if isinstance(reply, dict):
                    reply, tokens, token_seconds = reply["content"], reply.get("tokens"), reply.get("token_seconds")
                answer, error = EMPTY_THINK.sub("", reply), None
                ok = checked(check, prompt, expected, answer)
            except Exception as e:  # noqa: BLE001 - one failing example must not end the comparison
                answer, error, ok = "", f"{type(e).__name__}: {e}", False
            results.append(
                {
                    "example": i,
                    "model": model,
                    "ok": ok,
                    "seconds": round(time.monotonic() - start, 2),
                    "tokens": tokens,
                    "token_seconds": token_seconds,
                    "question": prompt[-1]["content"],
                    "expected": expected,
                    "answer": answer,
                    "error": error,
                }
            )
    order = {model: n for n, model in enumerate(models)}
    return sorted(results, key=lambda r: (r["example"], order[r["model"]]))


def _size(sizes: dict[str, int], model: str) -> int | None:
    """Ollama lists a model given without a tag as ``<model>:latest``."""
    return sizes.get(model, sizes.get(f"{model}:latest") if ":" not in model else None)


def model_stats(results: list[dict], models: list[str], sizes: dict[str, int] | None = None) -> list[dict]:
    stats = []
    for model in models:
        mine = [r for r in results if r["model"] == model]
        timed = [r for r in mine if r.get("tokens") and r.get("token_seconds")]
        stats.append(
            {
                "model": model,
                "right": sum(r["ok"] for r in mine),
                "total": len(mine),
                "errors": sum(r["error"] is not None for r in mine),
                "median_seconds": statistics.median(r["seconds"] for r in mine) if mine else None,
                "tokens_per_second": (
                    sum(r["tokens"] for r in timed) / sum(r["token_seconds"] for r in timed) if timed else None
                ),
                "bytes": _size(sizes or {}, model),
            }
        )
    return stats


def paired(results: list[dict], models: list[str]) -> list[dict]:
    """Each further model against the first: examples only the first got right, and only the other got right."""
    oks: dict[tuple[int, str], bool] = {(r["example"], r["model"]): r["ok"] for r in results}
    examples = sorted({r["example"] for r in results})
    first = models[0]
    return [
        {
            "model": other,
            "against": first,
            "only_first": [i for i in examples if oks.get((i, first)) and not oks.get((i, other))],
            "only_other": [i for i in examples if oks.get((i, other)) and not oks.get((i, first))],
        }
        for other in models[1:]
    ]


def disagreements(results: list[dict]) -> list[int]:
    by_example: dict[int, set[bool]] = {}
    for r in results:
        by_example.setdefault(r["example"], set()).add(r["ok"])
    return [i for i, oks in sorted(by_example.items()) if len(oks) > 1]


def _gib(size: int | None) -> str:
    return f"{size / 2**30:.1f} GiB" if size else "?"


def _stat_line(s: dict) -> str:
    speed = f", {s['tokens_per_second']:.0f} tokens/s" if s["tokens_per_second"] else ""
    errors = f", {s['errors']} errors" if s["errors"] else ""
    seconds = f"{s['median_seconds']:.1f} s" if s["median_seconds"] is not None else "?"
    return (
        f"{s['model']}: {s['right']}/{s['total']} right{errors}, median {seconds} per answer{speed}, {_gib(s['bytes'])}"
    )


def summary(results: list[dict], models: list[str], sizes: dict[str, int] | None = None) -> str:
    lines = [_stat_line(s) for s in model_stats(results, models, sizes)]
    for p in paired(results, models):
        lines.append(
            f"{p['model']} vs {p['against']}: right only with {p['model']} on {len(p['only_other'])}, "
            f"only with {p['against']} on {len(p['only_first'])}"
        )
    if len(models) > 1:
        differ = disagreements(results)
        lines.append(f"{len(differ)} examples where the models disagree: {differ[:20]}")
    return "\n".join(lines)


def _cell(text: str, width: int = 200) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def report(results: list[dict], models: list[str], sizes: dict[str, int] | None = None, examples: int = 5) -> str:
    """Markdown: a table per model, the paired comparison, and the first examples where the models disagree."""
    out = [
        "| Model | Right | Share | Median s per answer | Tokens/s | Size |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in model_stats(results, models, sizes):
        share = f"{s['right'] / s['total']:.0%}" if s["total"] else "?"
        speed = f"{s['tokens_per_second']:.0f}" if s["tokens_per_second"] else "?"
        seconds = f"{s['median_seconds']:.1f}" if s["median_seconds"] is not None else "?"
        right = f"{s['right']}/{s['total']}" + (f" ({s['errors']} errors)" if s["errors"] else "")
        out.append(f"| `{s['model']}` | {right} | {share} | {seconds} | {speed} | {_gib(s['bytes'])} |")
    comparisons = paired(results, models)
    if comparisons:
        out.append("")
        for p in comparisons:
            out.append(
                f"- `{p['model']}` against `{p['against']}`: right only with `{p['model']}` on "
                f"{len(p['only_other'])} examples, only with `{p['against']}` on {len(p['only_first'])}."
            )
    differ = disagreements(results)
    if differ:
        by_key = {(r["example"], r["model"]): r for r in results}
        out += ["", f"## Where the models disagree ({len(differ)} examples, first {min(examples, len(differ))})"]
        for i in differ[:examples]:
            first = by_key[(i, models[0])]
            out += ["", f"**{_cell(first['question'])}**", "", f"- expected: `{_cell(first['expected'])}`"]
            for model in models:
                r = by_key.get((i, model))
                if r is not None:
                    mark = "right" if r["ok"] else "wrong"
                    out.append(f"- `{model}` ({mark}): `{_cell(r['answer'] or r['error'] or '')}`")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True, help="JSONL, each conversation ends with the expected answer")
    ap.add_argument("--model", action="append", required=True, help="an Ollama model, repeat to compare")
    ap.add_argument("--check", help="file.py:function deciding whether an answer is right")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, help="only the first N examples")
    ap.add_argument("--out", type=Path, help="write every answer as JSONL")
    ap.add_argument("--report", type=Path, help="write the comparison as Markdown (for the user and release notes)")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        examples = [json.loads(line)["messages"] for line in f if line.strip()][: args.limit]
    results = evaluate(
        examples,
        args.model,
        lambda model, messages: ollama_chat(args.host, model, messages, args.timeout),
        load_check(args.check),
    )
    sizes = ollama_sizes(args.host, args.timeout)
    # the summary first and flushed: the files are on disk either way, the console may not survive interpreter exit
    print(summary(results, args.model, sizes), flush=True)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")
    if args.report:
        args.report.write_bytes(report(results, args.model, sizes).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
