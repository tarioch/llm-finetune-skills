import datetime
import json
import re
from pathlib import Path

import jinja2
import pytest
from chat_formats import (
    CHATML,
    SYSTEM_SENTINEL,
    data_system,
    default_system,
    detect,
    modelfile_system,
    render_like_ollama,
    render_modelfile,
    rewrites_system,
    system_turn,
)
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXTURES = Path(__file__).parent / "fixtures"
QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
QA = [{"role": "user", "content": "How much?"}, {"role": "assistant", "content": "SELECT 1"}]
SYSTEM_QA = [{"role": "system", "content": "Answer in BQL."}, *QA]
TWO_TURNS = [*QA, {"role": "user", "content": "And in 2024?"}, {"role": "assistant", "content": "SELECT 2"}]


def load_template(name):
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
    template = env.from_string((FIXTURES / name).read_text(encoding="utf-8"))

    def render(messages, add_generation_prompt=False):
        return template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, tools=None, bos_token="", eos_token=""
        )

    return render


# real chat templates (see fixtures/README.md) and the default system message each one inserts
TEMPLATES = {
    "Qwen2.5-Coder-Instruct.jinja": QWEN_DEFAULT,
    "Qwen3.jinja": None,
    "Qwen3.5.jinja": None,
    "SmolLM3.jinja": "## Metadata\n\nKnowledge Cutoff Date: June 2025\nReasoning Mode: /think\n\n",
}


@pytest.fixture(params=sorted(TEMPLATES), scope="module")
def template(request):
    return request.param, load_template(request.param)


def test_detect_chatml(template):
    _, render = template
    assert detect(render(QA)) is CHATML


def test_detect_unknown():
    with pytest.raises(ValueError, match="unknown chat format"):
        detect("[INST] hi [/INST]")


def test_default_system(template):
    name, render = template
    expected = TEMPLATES[name]
    found = default_system(CHATML, render)
    if expected is None:
        assert found is None
    else:
        assert found.startswith(expected)


def test_no_system_turn():
    assert system_turn(CHATML, lambda m: f"<|im_start|>user\n{m[-1]['content']}<|im_end|>\n") is None


def test_rewrites_system():
    assert not rewrites_system(CHATML, load_template("Qwen2.5-Coder-Instruct.jinja"))
    assert rewrites_system(CHATML, load_template("SmolLM3.jinja"))  # wraps it into its metadata block
    assert SYSTEM_SENTINEL in system_turn(CHATML, load_template("SmolLM3.jinja"), SYSTEM_SENTINEL)


@pytest.mark.parametrize(
    "conversations, expected",
    [
        ([QA, QA], ("none", None)),
        ([SYSTEM_QA] * 2, ("same", "Answer in BQL.")),
        ([SYSTEM_QA, QA], ("mixed", None)),
    ],
)
def test_data_system(conversations, expected):
    assert data_system(conversations) == expected


@pytest.mark.parametrize("conversation", [QA, SYSTEM_QA, TWO_TURNS], ids=["plain", "system", "two-turns"])
def test_ollama_prompt_matches_training(template, conversation):
    """Ollama's prompt for a request is exactly the training text up to where the trained answer starts.

    The client sends the conversation without the answer and without the system message, which the Modelfile's
    SYSTEM provides. Thinking templates (Qwen3, Qwen3.5) put an empty think block in front of the last answer: it
    belongs to what the model learns to write, not to the prompt.
    """
    _, render = template
    usage, text = data_system([conversation])
    system = modelfile_system(CHATML, render, usage, text)
    request = [m for m in conversation[:-1] if m["role"] != "system"]
    prompt = render_like_ollama(CHATML, request, system)
    trained = render(conversation)
    assert trained.startswith(prompt)
    answer = re.sub(r"^<think>\s*</think>\s*", "", trained[len(prompt) :])  # an empty thinking block is trained
    assert answer.startswith(conversation[-1]["content"])


def test_render_modelfile():
    text = render_modelfile("model.gguf", CHATML, QWEN_DEFAULT)
    assert text == (
        "FROM model.gguf\n"
        'TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n"""\n'
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
        f'SYSTEM """{QWEN_DEFAULT}"""\n'
    )
    assert "\r" not in text


def test_render_modelfile_without_system_or_temperature():
    text = render_modelfile("m.gguf", CHATML, None, temperature=None)
    assert "SYSTEM" not in text and "temperature" not in text


@pytest.mark.parametrize("system", ['say """hi"""', "line\r\n"])
def test_render_modelfile_rejects_unrepresentable_system(system):
    with pytest.raises(ValueError):
        render_modelfile("m.gguf", CHATML, system)
