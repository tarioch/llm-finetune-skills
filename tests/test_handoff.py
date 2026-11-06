"""The two skills together: the prompt Ollama builds from run.json is exactly what training rendered.

``unsloth-lora`` records the chat layout from the real chat template, ``ollama-model`` builds its template from that
record. Both are loaded by file path, as each skill only knows its own scripts.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parent.parent / "skills"


def load(skill, script):
    name = f"{skill}_{script}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, SKILLS / skill / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


chat_format = load("unsloth-lora", "chat_format")
modelfile = load("ollama-model", "modelfile")

QA = [{"role": "user", "content": "How much?"}, {"role": "assistant", "content": "SELECT 1"}]
SYSTEM_QA = [{"role": "system", "content": "Answer in BQL."}, *QA]
TWO_TURNS = [*QA, {"role": "user", "content": "And in 2024?"}, {"role": "assistant", "content": "SELECT 2"}]
TEMPLATES = ["Qwen2.5-Coder-Instruct.jinja", "Qwen3.jinja", "Qwen3.5.jinja", "SmolLM3.jinja"]


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("conversation", [QA, SYSTEM_QA, TWO_TURNS], ids=["plain", "system", "two-turns"])
def test_ollama_prompt_matches_training(chat_template, name, conversation):
    """The client sends the conversation without the answer and without the system message (SYSTEM provides it).

    Thinking templates (Qwen3, Qwen3.5) put an empty think block in front of the last answer: it belongs to what the
    model learns to write, not to the prompt.
    """
    render = chat_template(name)
    chat = chat_format.chat_record(chat_format.detect(render(conversation)), render, [conversation])
    request = [m for m in conversation[:-1] if m["role"] != "system"]
    prompt = modelfile.render_prompt(modelfile.template(chat), request, chat["system"])
    trained = render(conversation)
    assert trained.startswith(prompt)
    answer = re.sub(r"^<think>\s*</think>\s*", "", trained[len(prompt) :])
    assert answer.startswith(conversation[-1]["content"])
    modelfile.check_chat(chat)  # and the Modelfile can hold it


def test_run_json_docs_are_the_same():
    first, second = (SKILLS / s / "references" / "run-json.md" for s in ("unsloth-lora", "ollama-model"))
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
