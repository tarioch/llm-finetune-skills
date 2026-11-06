import json

import check_data
import pytest


def write(path, conversations):
    path.write_text("".join(json.dumps({"messages": c}) + "\n" for c in conversations), encoding="utf-8")


def qa(q, a="A"):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["check_data.py", *map(str, argv)])
    return check_data.main()


def test_ok(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one"), qa("two")])
    write(tmp_path / "val.jsonl", [qa("three")])
    assert run(monkeypatch, tmp_path) == 0
    assert "train: 2 conversations, no system messages" in capsys.readouterr().out


def test_val_example_also_in_training(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one"), qa("two")])
    write(tmp_path / "val.jsonl", [qa(" one ", "A ")])
    assert run(monkeypatch, tmp_path) == 1
    assert "1 val examples also are in training" in capsys.readouterr().err


def test_same_question_other_answer_is_a_warning(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one"), qa("two")])
    write(tmp_path / "val.jsonl", [qa("one", "B")])
    assert run(monkeypatch, tmp_path) == 0
    assert "WARNING 1 val questions are in training with a different answer" in capsys.readouterr().err


def test_overlap_with_eval_set(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one")])
    write(tmp_path / "val.jsonl", [qa("two")])
    write(tmp_path / "eval.jsonl", [qa("one")])
    assert run(monkeypatch, "--eval", tmp_path / "eval.jsonl", tmp_path) == 1
    assert "1 eval examples" in capsys.readouterr().err


def test_system_usage_must_match(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [[{"role": "system", "content": "S"}, *qa("one")]])
    write(tmp_path / "val.jsonl", [qa("two")])
    assert run(monkeypatch, tmp_path) == 1
    assert "use system messages differently" in capsys.readouterr().err


def test_missing_file(tmp_path, monkeypatch):
    write(tmp_path / "train.jsonl", [qa("one")])
    assert run(monkeypatch, tmp_path) == 1


@pytest.mark.parametrize(
    "messages, problem",
    [
        ([{"role": "user", "content": "q"}], "does not end with an assistant message"),
        (
            [{"role": "user", "content": "q"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "do not alternate",
        ),
        ([{"role": "assistant", "content": "a"}], "first non-system message"),
        ([{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}], "is empty"),
        (
            [
                {"role": "user", "content": "q"},
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "a"},
            ],
            "only the first message",
        ),
        ([{"role": "tool", "content": "q"}], "needs a role"),
    ],
)
def test_check_messages(messages, problem):
    assert any(problem in p for p in check_data.check_messages(messages))


def test_broken_lines(tmp_path):
    (tmp_path / "train.jsonl").write_text('not json\n{"x": 1}\n\n', encoding="utf-8")
    conversations, errors = check_data.load(tmp_path / "train.jsonl")
    assert conversations == []
    assert [e.split(":")[1] for e in errors] == ["1", "2"]
