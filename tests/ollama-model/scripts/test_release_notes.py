import json
import subprocess
import sys
from pathlib import Path

import release_notes

RUN = {
    "base_model": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    "args": {"epochs": 2.0, "max_steps": -1, "max_seq_length": 1024, "rank": 32, "lr": 0.0002, "data": "data"},
    "data": {"train.jsonl": "a" * 64, "val.jsonl": "b" * 64},
    "examples": {"train": 6321, "val": 263},
    "train_loss": 0.2134,
    "eval_loss": 0.24,
    "steps": 792,
    "peak_gpu_gib": 7.5,
    "chat": {"system": "You are a helpful assistant.", "system_in_data": "none", "empty_think": False},
    "gguf": {"path": "gguf/model.Q4_K_M.gguf", "quant": "q4_k_m", "bytes": 1929935584, "sha256": "c" * 64},
}


def test_notes_trace_the_model_back_to_its_data():
    text = release_notes.notes(RUN, name="me/my-model:3b")
    assert text.startswith("# me/my-model:3b\n")
    assert "Fine-tuned from `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`" in text
    assert "- Examples: 6321 train, 263 val" in text
    assert "- Settings: epochs 2, max seq length 1024, lr 0.0002, rank 32" in text  # max_steps -1 left out
    assert "- Steps: 792" in text and "- Training loss: 0.2134" in text and "- Validation loss: 0.24" in text
    assert f"| `train.jsonl` | `{'a' * 64}` |" in text
    assert "- `model.Q4_K_M.gguf`, Q4_K_M" in text and "1.80 GiB (1929935584 bytes)" in text
    assert f"SHA-256: `{'c' * 64}`" in text
    assert "## Using it" not in text and "## Evaluation" not in text


def test_usage_notes_for_the_application():
    run = {**RUN, "chat": {"system": None, "system_in_data": "mixed", "empty_think": True}}
    text = release_notes.notes(run)
    assert "Send a system message with every request" in text and "empty `<think></think>` block" in text


def test_evaluation_report_is_included_one_level_down():
    report = "| Model | Right |\n| --- | --- |\n| `a` | 3/5 |\n\n## Where the models disagree (1 examples, first 1)\n"
    text = release_notes.notes(RUN, report)
    assert "## Evaluation\n\n| Model | Right |" in text
    assert "\n### Where the models disagree" in text


def test_minimal_run_json():
    text = release_notes.notes({"base_model": "m"})
    assert "Fine-tuned from `m`" in text and "## Data" not in text and "## Model file" not in text


def test_command_line(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps(RUN), encoding="utf-8")
    (tmp_path / "eval.md").write_text("| Model | Right |\n", encoding="utf-8")
    script = Path(release_notes.__file__)
    out = tmp_path / "RELEASE.md"
    subprocess.run(
        [sys.executable, str(script), "--run", str(tmp_path), "--eval", str(tmp_path / "eval.md"), "-o", str(out)],
        check=True,
    )
    assert b"\r\n" not in out.read_bytes() and "## Evaluation" in out.read_text(encoding="utf-8")
