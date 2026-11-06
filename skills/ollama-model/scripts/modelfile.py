#!/usr/bin/env python
"""Write an Ollama Modelfile that renders prompts exactly like the model was trained.

Reads the ``chat`` section of a ``run.json`` (see ``references/run-json.md``): the turn markers, stop strings and
system turn that training used. The prompt template is built from those markers, so it works for any family the
trainer knows.

    python modelfile.py --run outputs/my-3b                         # GGUF from run.json, -> outputs/my-3b/Modelfile
    python modelfile.py --run outputs/my-3b --gguf other.gguf --temperature 0.7 -o Modelfile

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REQUIRED = ("turn_prefix", "turn_suffix", "reply_prefix", "stop", "system")


def check_chat(chat: dict) -> None:
    missing = [k for k in REQUIRED if k not in chat]
    if missing:
        raise ValueError(f"run.json's chat section misses {', '.join(missing)}")
    parts = [chat["turn_prefix"], chat["turn_suffix"], chat["reply_prefix"], *chat["stop"]]
    if "{role}" not in chat["turn_prefix"] or any("{{" in p or "}}" in p for p in parts):
        raise ValueError("turn_prefix needs {role}, and no marker may contain {{ or }}")
    for name, value in (("system message", chat["system"]), *(("marker", p) for p in parts)):
        if value is not None and ('"""' in value or "\r" in value):
            raise ValueError(f"a {name} contains a triple quote or CR, which a Modelfile cannot hold")


def template(chat: dict) -> str:
    """The Ollama TEMPLATE: every message in the training layout, then the start of the reply.

    It reads only ``.Messages``: Ollama passes the Modelfile's SYSTEM as the first message and also as ``.System``,
    so a template using both would send the system prompt twice.
    """
    prefix = chat["turn_prefix"].replace("{role}", "{{ .Role }}")
    return (
        f"{{{{- range .Messages }}}}{prefix}{{{{ .Content }}}}{chat['turn_suffix']}{{{{ end }}}}{chat['reply_prefix']}"
    )


def render_modelfile(gguf: str, chat: dict, temperature: float | None = 0.0) -> str:
    """Modelfile text, LF line endings: Ollama keeps a stray CR inside a template or system prompt."""
    check_chat(chat)
    lines = [f"FROM {gguf}", f'TEMPLATE """{template(chat)}"""']
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature:g}")
    lines += [f'PARAMETER stop "{s}"' for s in chat["stop"]]
    if chat["system"] is not None:
        lines.append(f'SYSTEM """{chat["system"]}"""')
    return "\n".join(lines) + "\n"


_TEMPLATE = re.compile(r"\{\{- range \.Messages \}\}(?P<body>.*?)\{\{ end \}\}(?P<suffix>.*)", re.DOTALL)


def render_prompt(modelfile_template: str, messages: list[dict], system: str | None) -> str:
    """What Ollama renders from a template built by :func:`template` for a request (for the tests).

    Interprets the template text instead of re-implementing it, but only the subset :func:`template` writes. Ollama
    puts the Modelfile's SYSTEM first unless the request has its own system message.
    """
    match = _TEMPLATE.fullmatch(modelfile_template)
    if match is None or "{{" in match["suffix"]:
        raise ValueError("the template uses more than this renderer understands")
    if system is not None and not (messages and messages[0]["role"] == "system"):
        messages = [{"role": "system", "content": system}, *messages]
    body = match["body"]
    text = "".join(body.replace("{{ .Role }}", m["role"]).replace("{{ .Content }}", m["content"]) for m in messages)
    return text + match["suffix"]


def from_path(gguf: Path, modelfile_dir: Path) -> str:
    """The FROM path: relative to the Modelfile's folder (Ollama resolves it from there), so no machine path ends up
    in a Modelfile that gets committed; absolute only when there is no relative path (another Windows drive)."""
    gguf, modelfile_dir = gguf.resolve(), modelfile_dir.resolve()
    try:
        relative = Path(os.path.relpath(gguf, modelfile_dir)).as_posix()
    except ValueError:
        return str(gguf)
    return relative if relative.startswith("../") else f"./{relative}"


def write_modelfile(chat: dict, gguf: Path, out: Path, temperature: float | None = 0.0) -> Path:
    text = render_modelfile(from_path(gguf, out.parent), chat, temperature)
    out.write_bytes(text.encode("utf-8"))  # bytes: no CRLF on Windows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run folder with run.json")
    ap.add_argument("--gguf", type=Path, help="default: the GGUF recorded in run.json")
    ap.add_argument("-o", "--out", type=Path, help="default: <run>/Modelfile")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="baked-in default (%(default)s). Ollama's OpenAI-compatible /v1 ignores it: send it with each request",
    )
    args = ap.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    if "chat" not in run:
        print("run.json has no chat section, see references/run-json.md", file=sys.stderr)
        return 2
    gguf = args.gguf or (args.run / run["gguf"]["path"] if "gguf" in run else None)
    if gguf is None:
        print("no --gguf given and run.json records none", file=sys.stderr)
        return 2
    out = write_modelfile(run["chat"], gguf, args.out or args.run / "Modelfile", args.temperature)
    print(out.read_text(encoding="utf-8"))
    for warning in warnings(run["chat"]):
        print(f"WARNING {warning}", file=sys.stderr)
    return 0


def warnings(chat: dict) -> list[str]:
    found = []
    if chat["system"] is None and chat.get("system_in_data") == "mixed":
        found.append("no SYSTEM: the training data mixes system messages, every request has to send the right one")
        if chat.get("template_rewrites_system"):
            found.append("the training template rewrote system messages, so a request's own one won't match training")
    if chat["system"] is not None and chat["system"] != chat["system"].strip():
        found.append("SYSTEM starts or ends with whitespace, check that Ollama keeps it")
    if chat.get("empty_think"):
        found.append("the model starts its answers with an empty <think></think> block, the app has to strip it")
    return found


if __name__ == "__main__":
    sys.exit(main())
