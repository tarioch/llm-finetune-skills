import pytest
from chat_format import (
    CHATML,
    SYSTEM_SENTINEL,
    chat_record,
    data_system,
    detect,
    empty_think,
    rewrites_system,
    system_turn,
)

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
QA = [{"role": "user", "content": "How much?"}, {"role": "assistant", "content": "SELECT 1"}]
SYSTEM_QA = [{"role": "system", "content": "Answer in BQL."}, *QA]

# real chat templates and the start of the default system message each one inserts
TEMPLATES = {
    "Qwen2.5-Coder-Instruct.jinja": QWEN_DEFAULT,
    "Qwen3.jinja": None,
    "Qwen3.5.jinja": None,
    "SmolLM3.jinja": "## Metadata\n\nKnowledge Cutoff Date: June 2025\nReasoning Mode: /think\n\n",
}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_detect_chatml(chat_template, name):
    assert detect(chat_template(name)(QA)) is CHATML


def test_detect_unknown():
    with pytest.raises(ValueError, match="unknown chat format"):
        detect("[INST] hi [/INST]")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_default_system(chat_template, name):
    found = system_turn(CHATML, chat_template(name))
    if TEMPLATES[name] is None:
        assert found is None
    else:
        assert found.startswith(TEMPLATES[name])


def test_no_system_turn():
    assert system_turn(CHATML, lambda m: f"<|im_start|>user\n{m[-1]['content']}<|im_end|>\n") is None


def test_rewrites_system(chat_template):
    assert not rewrites_system(CHATML, chat_template("Qwen2.5-Coder-Instruct.jinja"))
    smollm = chat_template("SmolLM3.jinja")
    assert rewrites_system(CHATML, smollm)  # wraps it into its metadata block
    assert SYSTEM_SENTINEL in system_turn(CHATML, smollm, SYSTEM_SENTINEL)


@pytest.mark.parametrize(
    "name, expected", [("Qwen2.5-Coder-Instruct.jinja", False), ("Qwen3.jinja", True), ("Qwen3.5.jinja", True)]
)
def test_empty_think(chat_template, name, expected):
    assert empty_think(CHATML, chat_template(name)(QA)) is expected


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


def test_chat_record(chat_template):
    record = chat_record(CHATML, chat_template("Qwen2.5-Coder-Instruct.jinja"), [QA])
    assert record == {
        "format": "chatml",
        "turn_prefix": "<|im_start|>{role}\n",
        "turn_suffix": "<|im_end|>\n",
        "reply_prefix": "<|im_start|>assistant\n",
        "stop": ["<|im_end|>"],
        "system": QWEN_DEFAULT,
        "system_in_data": "none",
        "template_rewrites_system": False,
        "empty_think": False,
    }


def test_chat_record_mixed_system(chat_template):
    record = chat_record(CHATML, chat_template("SmolLM3.jinja"), [SYSTEM_QA, QA])
    assert record["system"] is None and record["system_in_data"] == "mixed" and record["template_rewrites_system"]
