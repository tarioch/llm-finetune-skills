"""The chat format of a model family, and the record of it a server needs to reproduce what training saw.

A fine-tuned model only behaves like it did in training if the server renders a conversation into exactly the same
text. ``train.py`` writes :func:`chat_record` into ``run.json`` (see ``references/run-json.md``): the turn markers,
the stop strings and the system turn exactly as the chat template rendered it. A serving skill builds its prompt
template from that record, without knowing the family.

For each family this module knows how a turn is written: ``turn_prefix`` (with ``{role}``) and ``turn_suffix``
around each message's content, and ``reply_prefix``, which starts the assistant's reply. The trainer computes the
loss only after ``reply_prefix``.

No third-party imports: ``train.py`` uses it in the Unsloth environment, the tests run without it.
To support another family, add a :class:`ChatFormat` to ``FORMATS`` and its real chat template to the tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

SENTINEL = "SENTINEL-QUESTION-7f3a"
SYSTEM_SENTINEL = "SENTINEL-SYSTEM-7f3a"

Render = Callable[[list[dict]], str]


@dataclass(frozen=True)
class ChatFormat:
    name: str
    turn_prefix: str  # starts a message, {role} is the role
    turn_suffix: str  # ends a message
    reply_prefix: str  # starts the assistant's reply, the loss starts after it
    stop: tuple[str, ...]

    def prefix(self, role: str) -> str:
        return self.turn_prefix.replace("{role}", role)


# ChatML: Qwen2.5, Qwen3, Qwen3.5, SmolLM3 and many others
CHATML = ChatFormat(
    name="chatml",
    turn_prefix="<|im_start|>{role}\n",
    turn_suffix="<|im_end|>\n",
    reply_prefix="<|im_start|>assistant\n",
    stop=("<|im_end|>",),
)

FORMATS = {f.name: f for f in (CHATML,)}


def detect(rendered: str) -> ChatFormat:
    """The format whose user and assistant markers both appear in a rendered conversation."""
    for fmt in FORMATS.values():
        if fmt.prefix("user") in rendered and fmt.reply_prefix in rendered:
            return fmt
    raise ValueError("unknown chat format, add it to chat_format.FORMATS. Rendered example:\n" + rendered)


def system_turn(fmt: ChatFormat, render: Render, system: str | None = None) -> str | None:
    """The system turn's text as the chat template writes it, or None when it writes none.

    ``render`` renders messages with the model's chat template (``tokenizer.apply_chat_template`` with
    ``tokenize=False``). With ``system=None`` this is the default the template inserts on its own (Qwen2.5 has one,
    Qwen3 has none). Some templates also rewrite a given system message, e.g. SmolLM3 wraps it into a larger block.
    """
    messages = [{"role": "system", "content": system}] if system is not None else []
    text = render([*messages, {"role": "user", "content": SENTINEL}])
    head = text[: text.index(SENTINEL)]
    marker = fmt.prefix("system")
    start = head.find(marker)
    if start < 0:
        return None
    start += len(marker)
    return head[start : head.index(fmt.turn_suffix, start)]


def data_system(conversations: list[list[dict]]) -> tuple[str, str | None]:
    """How the training data uses system messages: ("none", None), ("same", text) or ("mixed", None)."""
    systems = {c[0]["content"] if c and c[0]["role"] == "system" else None for c in conversations}
    if systems == {None}:
        return "none", None
    if len(systems) == 1:
        return "same", systems.pop()
    return "mixed", None


def rewrites_system(fmt: ChatFormat, render: Render) -> bool:
    """Whether the template changes a given system message (then a caller's own one won't match training)."""
    return system_turn(fmt, render, SYSTEM_SENTINEL) != SYSTEM_SENTINEL


def empty_think(fmt: ChatFormat, rendered: str) -> bool:
    """Whether the template puts an empty think block in front of the trained answer (Qwen3, Qwen3.5)."""
    start = rendered.rfind(fmt.reply_prefix) + len(fmt.reply_prefix)
    return rendered.startswith("<think>", start)


def chat_record(fmt: ChatFormat, render: Render, conversations: list[list[dict]]) -> dict:
    """The ``chat`` section of run.json: what a server needs to render prompts exactly like training did.

    ``system`` is the system turn as the chat template wrote it in training (its default when the data has none,
    the data's own message after any rewriting otherwise), or None when the data mixes system messages: then every
    caller has to send its own.
    """
    usage, text = data_system(conversations)
    system = None if usage == "mixed" else system_turn(fmt, render, text if usage == "same" else None)
    return {
        "format": fmt.name,
        "turn_prefix": fmt.turn_prefix,
        "turn_suffix": fmt.turn_suffix,
        "reply_prefix": fmt.reply_prefix,
        "stop": list(fmt.stop),
        "system": system,
        "system_in_data": usage,
        "template_rewrites_system": rewrites_system(fmt, render),
        "empty_think": empty_think(fmt, render(conversations[0])),
    }
