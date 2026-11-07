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
    assert (
        "1 val prompts are also in training (1 with the same answer, 0 with a different one)" in capsys.readouterr().err
    )


def test_same_prompt_other_answer_is_an_error(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one"), qa("two")])
    write(tmp_path / "val.jsonl", [qa("one", "B"), qa("three")])
    assert run(monkeypatch, tmp_path) == 1
    err = capsys.readouterr().err
    assert "ERROR 1 val prompts are also in training (0 with the same answer, 1 with a different one)" in err
    assert "take them out of val" in err


def test_repeats_within_training_are_fine(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one", "A"), qa("one", "B")])
    write(tmp_path / "val.jsonl", [qa("two")])
    assert run(monkeypatch, tmp_path) == 0


def test_overlap_with_eval_set(tmp_path, monkeypatch, capsys):
    write(tmp_path / "train.jsonl", [qa("one")])
    write(tmp_path / "val.jsonl", [qa("two")])
    write(tmp_path / "eval.jsonl", [qa("one")])
    assert run(monkeypatch, "--eval", tmp_path / "eval.jsonl", tmp_path) == 1
    assert "1 eval prompts are also in training" in capsys.readouterr().err


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
