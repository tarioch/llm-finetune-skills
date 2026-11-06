"""Chat formats: what training and Ollama have to agree on for one model family.

A fine-tuned model only behaves like it did in training if Ollama renders a conversation into exactly the same text
the trainer saw. For each family this module knows

* the markers that start a user and an assistant turn (the trainer computes the loss only after the assistant
  marker),
* how a system turn is written (to find the default system message a chat template inserts on its own),
* the Ollama ``TEMPLATE`` that reproduces the training text from ``.Messages``, and the stop strings.

No third-party imports: the scripts that use it run in the Unsloth environment, the tests run without it.
To support another family, add a :class:`ChatFormat` to ``FORMATS`` and a test that renders a conversation with the
family's real chat template (see ``references/chat-formats.md``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

SENTINEL = "SENTINEL-QUESTION-7f3a"
SYSTEM_SENTINEL = "SENTINEL-SYSTEM-7f3a"


@dataclass(frozen=True)
class ChatFormat:
    name: str
    instruction_part: str  # text that starts a user turn
    response_part: str  # text that starts an assistant turn, the loss starts after it
    system_part: str  # text that starts a system turn
    end_of_turn: str
    ollama_template: str  # Go template, reads only .Messages (Ollama passes SYSTEM as the first message)
    stop: tuple[str, ...]


CHATML = ChatFormat(
    name="chatml",
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
    system_part="<|im_start|>system\n",
    end_of_turn="<|im_end|>",
    ollama_template=(
        "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"
    ),
    stop=("<|im_end|>",),
)

# ChatML: Qwen2.5, Qwen3, Qwen3.5, SmolLM3 and others. The templates checked are in the repository's
# tests/unsloth-ollama/scripts/test_chat_formats.py.
FORMATS = {f.name: f for f in (CHATML,)}


def detect(rendered: str) -> ChatFormat:
    """The format whose user and assistant markers both appear in a rendered conversation."""
    for fmt in FORMATS.values():
        if fmt.instruction_part in rendered and fmt.response_part in rendered:
            return fmt
    raise ValueError(
        "unknown chat format, add it to chat_formats.FORMATS (see references/chat-formats.md). Rendered example:\n"
        + rendered
    )


Render = Callable[[list[dict]], str]


def system_turn(fmt: ChatFormat, render: Render, system: str | None = None) -> str | None:
    """The system turn's text as the chat template writes it, or None when it writes none.

    ``render`` renders messages with the model's chat template (``tokenizer.apply_chat_template`` with
    ``tokenize=False``). With ``system=None`` this is the default the template inserts on its own (Qwen2.5 has one,
    Qwen3 has none). Some templates also rewrite a given system message, e.g. SmolLM3 wraps it into a larger block.
    """
    messages = [{"role": "system", "content": system}] if system is not None else []
    text = render([*messages, {"role": "user", "content": SENTINEL}])
    head = text[: text.index(SENTINEL)]
    start = head.find(fmt.system_part)
    if start < 0:
        return None
    start += len(fmt.system_part)
    return head[start : head.index(fmt.end_of_turn, start)]


def default_system(fmt: ChatFormat, render: Render) -> str | None:
    """The system message the chat template inserts into a conversation that has none, or None."""
    return system_turn(fmt, render)


def data_system(conversations: list[list[dict]]) -> tuple[str, str | None]:
    """How the training data uses system messages: ("none", None), ("same", text) or ("mixed", None)."""
    systems = {c[0]["content"] if c and c[0]["role"] == "system" else None for c in conversations}
    if systems == {None}:
        return "none", None
    if len(systems) == 1:
        return "same", systems.pop()
    return "mixed", None


def modelfile_system(fmt: ChatFormat, render: Render, usage: str, text: str | None) -> str | None:
    """The SYSTEM line the Modelfile needs so that Ollama sends what training saw.

    The Ollama template reads only ``.Messages`` and writes the system message as it is, so it has to be the system
    turn exactly as the chat template wrote it in training: its default when the data has none, or the data's own
    message after any rewriting the template does.
    """
    if usage == "mixed":
        return None  # every caller has to send its own system message
    return system_turn(fmt, render, text if usage == "same" else None)


def rewrites_system(fmt: ChatFormat, render: Render) -> bool:
    """Whether the template changes a given system message (then a caller's own one won't match training)."""
    return system_turn(fmt, render, SYSTEM_SENTINEL) != SYSTEM_SENTINEL


def render_modelfile(gguf: str, fmt: ChatFormat, system: str | None, temperature: float | None = 0.0) -> str:
    """Modelfile text, LF line endings: Ollama keeps a stray CR inside a template or system prompt."""
    for name, value in (("system message", system), ("template", fmt.ollama_template)):
        if value is not None and ('"""' in value or "\r" in value):
            raise ValueError(f"the {name} contains a triple quote or CR, which a Modelfile cannot hold")
    lines = [f"FROM {gguf}", f'TEMPLATE """{fmt.ollama_template}"""']
    if temperature is not None:
        lines.append(f"PARAMETER temperature {temperature:g}")
    lines += [f'PARAMETER stop "{s}"' for s in fmt.stop]
    if system is not None:
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"


_TEMPLATE = re.compile(r"\{\{- range \.Messages \}\}(?P<body>.*?)\{\{ end \}\}(?P<suffix>.*)", re.DOTALL)


def render_like_ollama(fmt: ChatFormat, messages: list[dict], system: str | None) -> str:
    """What Ollama renders from the format's template for a request (used by the tests to compare with training).

    Interprets the template itself instead of re-implementing the format, but only the subset the templates here
    use: one ``range .Messages`` loop with ``.Role`` and ``.Content``, then a fixed suffix. Ollama puts the
    Modelfile's SYSTEM first unless the request has its own system message.
    """
    match = _TEMPLATE.fullmatch(fmt.ollama_template)
    if match is None or "{{" in match["suffix"]:
        raise ValueError(f"{fmt.name}: the template uses more than this renderer understands")
    if system is not None and not (messages and messages[0]["role"] == "system"):
        messages = [{"role": "system", "content": system}, *messages]
    body = match["body"]
    text = "".join(body.replace("{{ .Role }}", m["role"]).replace("{{ .Content }}", m["content"]) for m in messages)
    return text + match["suffix"]
