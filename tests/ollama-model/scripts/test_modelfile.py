import json

import modelfile
import pytest

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHAT = {
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


def test_render_modelfile():
    assert modelfile.render_modelfile("model.gguf", CHAT) == (
        "FROM model.gguf\n"
        'TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n"""\n'
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
        f'SYSTEM """{QWEN_DEFAULT}"""\n'
    )


def test_render_modelfile_without_system_or_temperature():
    text = modelfile.render_modelfile("m.gguf", {**CHAT, "system": None}, temperature=None)
    assert "SYSTEM" not in text and "temperature" not in text


@pytest.mark.parametrize(
    "change",
    [
        {"system": 'say """hi"""'},
        {"system": "line\r\n"},
        {"turn_suffix": "{{ .Content }}"},
        {"turn_prefix": "<|im_start|>"},
    ],
)
def test_rejects_what_a_modelfile_cannot_hold(change):
    with pytest.raises(ValueError):
        modelfile.render_modelfile("m.gguf", {**CHAT, **change})


def test_rejects_incomplete_chat_section():
    with pytest.raises(ValueError, match="misses reply_prefix"):
        modelfile.render_modelfile("m.gguf", {k: v for k, v in CHAT.items() if k != "reply_prefix"})


def test_render_prompt():
    template = modelfile.template(CHAT)
    prompt = modelfile.render_prompt(template, [{"role": "user", "content": "Q"}], "S")
    assert prompt == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
    own = modelfile.render_prompt(
        template, [{"role": "system", "content": "Own"}, {"role": "user", "content": "Q"}], "S"
    )
    assert own.startswith("<|im_start|>system\nOwn<|im_end|>\n")


def test_main_writes_lf_only_modelfile(tmp_path, monkeypatch, capsys):
    (tmp_path / "run.json").write_text(
        json.dumps({"chat": {**CHAT, "empty_think": True}, "gguf": {"path": "gguf/m.gguf", "quant": "q4_k_m"}})
    )
    monkeypatch.setattr("sys.argv", ["modelfile.py", "--run", str(tmp_path)])
    assert modelfile.main() == 0
    data = (tmp_path / "Modelfile").read_bytes()
    assert b"\r" not in data
    assert data.startswith(b"FROM ./gguf/m.gguf\n")
    assert "empty <think></think> block" in capsys.readouterr().err


def test_main_needs_a_chat_section(tmp_path, monkeypatch):
    (tmp_path / "run.json").write_text("{}")
    monkeypatch.setattr("sys.argv", ["modelfile.py", "--run", str(tmp_path), "--gguf", "m.gguf"])
    assert modelfile.main() == 2


def test_from_path_is_relative_to_the_modelfile(tmp_path):
    assert modelfile.from_path(tmp_path / "run" / "gguf" / "m.gguf", tmp_path / "run") == "./gguf/m.gguf"
    assert modelfile.from_path(tmp_path / "other" / "m.gguf", tmp_path / "run") == "../other/m.gguf"
