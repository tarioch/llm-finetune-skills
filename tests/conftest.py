import datetime
import json
from pathlib import Path

import jinja2
import pytest
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

CHAT_TEMPLATES = Path(__file__).parent / "fixtures" / "chat-templates"


def load_chat_template(name):
    """A renderer like tokenizer.apply_chat_template(messages, tokenize=False), for a real chat template.

    Uses the same Jinja setup as transformers: sandboxed, trim_blocks, lstrip_blocks, loop controls.
    """
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.date(2026, 1, 1).strftime(fmt)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii
    )
    template = env.from_string((CHAT_TEMPLATES / name).read_text(encoding="utf-8"))

    def render(messages, add_generation_prompt=False):
        return template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, tools=None, bos_token="", eos_token=""
        )

    return render


@pytest.fixture(scope="session")
def chat_template():
    """Load a real chat template from fixtures/chat-templates by file name."""
    return load_chat_template
