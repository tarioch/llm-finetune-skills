from pathlib import Path

import jinja2
import pytest
from chat_formats import (
    CHATML,
    data_system,
    default_system,
    detect,
    modelfile_system,
    render_like_ollama,
    render_modelfile,
)

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@pytest.fixture(scope="module")
def qwen_render():
    """Renders like tokenizer.apply_chat_template(messages, tokenize=False) with Qwen2.5's real chat template."""
    source = (Path(__file__).parent / "fixtures" / "qwen2.5.jinja").read_text(encoding="utf-8")
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(source)
    return lambda messages, add_generation_prompt=False: template.render(
        messages=messages, add_generation_prompt=add_generation_prompt, tools=None
    )


QA = [{"role": "user", "content": "How much?"}, {"role": "assistant", "content": "SELECT 1"}]


def test_detect_chatml(qwen_render):
    assert detect(qwen_render(QA)) is CHATML


def test_detect_unknown():
    with pytest.raises(ValueError, match="unknown chat format"):
        detect("[INST] hi [/INST]")


def test_qwen_default_system(qwen_render):
    assert default_system(CHATML, qwen_render) == QWEN_DEFAULT


def test_no_default_system():
    assert default_system(CHATML, lambda m: f"<|im_start|>user\n{m[0]['content']}<|im_end|>\n") is None


@pytest.mark.parametrize(
    "conversations, expected",
    [
        ([QA, QA], ("none", None)),
        ([[{"role": "system", "content": "S"}, *QA]] * 2, ("same", "S")),
        ([[{"role": "system", "content": "S"}, *QA], QA], ("mixed", None)),
    ],
)
def test_data_system(conversations, expected):
    assert data_system(conversations) == expected


@pytest.mark.parametrize("data_messages", [QA, [{"role": "system", "content": "Answer in BQL."}, *QA]])
def test_ollama_prompt_matches_training(qwen_render, data_messages):
    """The prompt Ollama builds from the Modelfile is exactly the text training saw before the answer."""
    usage, text = data_system([data_messages])
    system = modelfile_system(default_system(CHATML, qwen_render), usage, text)
    request = [m for m in data_messages[:-1] if m["role"] != "system"]  # a client sends only the question
    trained = qwen_render(data_messages[:-1], add_generation_prompt=True)
    assert render_like_ollama(CHATML, request, system) == trained


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
