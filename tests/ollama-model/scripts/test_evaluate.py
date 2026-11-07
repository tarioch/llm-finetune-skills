import sys

import evaluate
import pytest


def qa(q, a):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


EXAMPLES = [qa("one", "SELECT 1"), qa("two", "SELECT 2")]


def test_compares_models():
    answers = {("good", "one"): "SELECT  1\n", ("good", "two"): "SELECT 2", ("bad", "one"): "SELECT 1"}

    def ask(model, messages):
        assert messages[-1]["role"] == "user"  # the expected answer is never sent
        return answers.get((model, messages[-1]["content"]), "nonsense")

    results = evaluate.evaluate(EXAMPLES, ["good", "bad"], ask, evaluate.normalized_equal)
    assert [(r["model"], r["ok"]) for r in results] == [("good", True), ("bad", True), ("good", True), ("bad", False)]
    text = evaluate.summary(results, ["good", "bad"])
    assert "good: 2/2 right" in text and "bad: 1/2 right" in text and "disagree: [1]" in text


def test_one_model_after_the_other():
    asked = []

    def ask(model, messages):
        asked.append((model, messages[-1]["content"]))
        return "SELECT 1"

    results = evaluate.evaluate(EXAMPLES, ["a", "b"], ask, evaluate.normalized_equal)
    assert asked == [("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")]
    assert [(r["example"], r["model"]) for r in results] == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]


def test_empty_think_block_is_removed():
    [result, _] = evaluate.evaluate(
        EXAMPLES, ["m"], lambda model, messages: "<think>\n\n</think>\n\nSELECT 1", evaluate.normalized_equal
    )
    assert result["ok"] and result["answer"] == "SELECT 1"


def test_errors_count_as_wrong():
    def ask(model, messages):
        raise TimeoutError("slow")

    [result, _] = evaluate.evaluate(EXAMPLES, ["m"], ask, evaluate.normalized_equal)
    assert not result["ok"] and result["error"] == "TimeoutError: slow"


def test_custom_check(tmp_path):
    (tmp_path / "checks.py").write_text(
        "def same_digit(prompt, expected, answer):\n    return {'digit': expected[-1] == answer[-1]}\n",
        encoding="utf-8",
    )
    check = evaluate.load_check(f"{tmp_path / 'checks.py'}:same_digit")
    assert check([], "SELECT 1", "select 1") == {"digit": True}
    with pytest.raises(SystemExit):
        evaluate.load_check("checks.py")


def test_ollama_chat_request(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"message": {"role": "assistant", "content": "SELECT 1"}, "eval_count": 40, "eval_duration": 500000000}'

    def urlopen(request, timeout):
        sent["url"], sent["body"], sent["timeout"] = request.full_url, request.data, timeout
        return Response()

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", urlopen)
    reply = evaluate.ollama_chat("http://h:1", "m", [{"role": "user", "content": "q"}], 5)
    assert reply == {"content": "SELECT 1", "tokens": 40, "token_seconds": 0.5}
    assert sent["url"] == "http://h:1/api/chat" and sent["timeout"] == 5
    assert b'"temperature": 0' in sent["body"] and b'"stream": false' in sent["body"]


def test_statistics_and_paired_comparison():
    answers = {"new": ["SELECT 1", "SELECT 2", "x"], "old": ["SELECT 1", "x", "SELECT 3"]}
    examples = [*EXAMPLES, qa("three", "SELECT 3")]

    def ask(model, messages):
        i = ["one", "two", "three"].index(messages[-1]["content"])
        return {"content": answers[model][i], "tokens": 10, "token_seconds": 0.25}

    results = evaluate.evaluate(examples, ["new", "old"], ask, evaluate.normalized_equal)
    [new, old] = evaluate.model_stats(results, ["new", "old"], {"new": 2**31})
    assert (new["right"], new["total"], new["tokens_per_second"], new["bytes"]) == (2, 3, 40.0, 2**31)
    assert old["bytes"] is None
    assert evaluate.model_stats(results, ["new"], {"new:latest": 5})[0]["bytes"] == 5
    [p] = evaluate.paired(results, ["new", "old"])
    assert (p["only_first"], p["only_other"]) == ([1], [2])
    text = evaluate.summary(results, ["new", "old"], {"new": 2**31})
    assert "new: 2/3 right, median" in text and "40 tokens/s, 2.0 GiB" in text
    assert "old vs new: right only with old on 1, only with new on 1" in text


def test_report_is_markdown_with_disagreements():
    answers = {("a", "one"): "SELECT 1", ("a", "two"): "SELECT | 2", ("b", "one"): "SELECT 1", ("b", "two"): "SELECT 2"}
    results = evaluate.evaluate(
        EXAMPLES, ["a", "b"], lambda m, msgs: answers.get((m, msgs[-1]["content"]), "nope"), evaluate.normalized_equal
    )
    text = evaluate.report(results, ["a", "b"], {})
    assert "| `a` | 1/2 | 50% |" in text and "| `b` | 2/2 | 100% |" in text
    assert "right only with `b` on 1 examples, only with `a` on 0" in text
    assert "## Where the models disagree (1 examples, first 1)" in text
    assert "**two**" in text and "- `a` (wrong): `SELECT \\| 2`" in text and "- `b` (right): `SELECT 2`" in text


def test_errors_are_counted_in_the_statistics():
    def ask(model, messages):
        raise TimeoutError("slow")

    results = evaluate.evaluate(EXAMPLES, ["m"], ask, evaluate.normalized_equal)
    [stats] = evaluate.model_stats(results, ["m"])
    assert stats["errors"] == 2 and stats["tokens_per_second"] is None
    assert "| `m` | 0/2 (2 errors) | 0% |" in evaluate.report(results, ["m"])


def test_sizes_are_optional(monkeypatch):
    def urlopen(url, timeout):
        raise OSError("no ollama")

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", urlopen)
    assert evaluate.ollama_sizes("http://h:1", 1) == {}


def test_asks_one_model_everything_before_the_next():
    asked = []

    def ask(model, messages):
        asked.append((model, messages[-1]["content"]))
        return "SELECT 1"

    results = evaluate.evaluate(EXAMPLES, ["a", "b"], ask, evaluate.normalized_equal)
    # Ollama keeps the model loaded instead of swapping it for every answer
    assert asked == [("a", "one"), ("a", "two"), ("b", "one"), ("b", "two")]
    assert [(r["example"], r["model"]) for r in results] == [(0, "a"), (0, "b"), (1, "a"), (1, "b")]


class Wrapper:
    """What colorama.init() does to the streams on Windows, which TatSu calls on every parse."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        return self.stream.write(text)


def test_a_check_that_wraps_the_streams_cannot_break_printing(capsys):
    def wrapping_check(prompt, expected, answer):
        sys.stdout, sys.stderr = Wrapper(sys.stdout), Wrapper(sys.stderr)
        return {"exact": True}

    stdout, stderr = sys.stdout, sys.stderr
    examples = EXAMPLES * (sys.getrecursionlimit() // 2)  # with the wrappers piled up, print() would recurse too deep
    results = evaluate.evaluate(examples, ["m"], lambda model, messages: "SELECT 1", wrapping_check)
    assert sys.stdout is stdout and sys.stderr is stderr
    sys.stdout.write(evaluate.summary(results, ["m"]) + "\n")
    assert f"m: {len(examples)}/{len(examples)} right" in capsys.readouterr().out


def test_the_answer_is_kept_when_only_the_check_fails():
    def broken_check(prompt, expected, answer):
        raise AttributeError("'str' object has no attribute 'units'")

    [result] = evaluate.evaluate(EXAMPLES[:1], ["m"], lambda model, messages: "SELECT 9", broken_check)
    assert result["ok"] is False and result["error"].startswith("AttributeError")
    assert result["answer"] == "SELECT 9"


def test_there_is_no_answer_when_asking_fails():
    def ask(model, messages):
        raise TimeoutError("slow")

    [result] = evaluate.evaluate(EXAMPLES[:1], ["m"], ask, evaluate.normalized_equal)
    assert result["answer"] == "" and result["error"].startswith("TimeoutError")


def _scored(check, answers):
    return evaluate.evaluate(EXAMPLES, ["m"], lambda m, msgs: answers[msgs[-1]["content"]], check)


def test_a_check_can_return_named_scores_and_the_first_one_is_the_headline():
    def check(prompt, expected, answer):
        return {"strict": answer == expected, "loose": answer.startswith("SELECT")}

    results = _scored(check, {"one": "SELECT 1", "two": "SELECT 9"})
    assert [r["scores"] for r in results] == [{"strict": 1.0, "loose": 1.0}, {"strict": 0.0, "loose": 1.0}]
    assert [r["ok"] for r in results] == [True, False]  # right means the first score is 1
    [stats] = evaluate.model_stats(results, ["m"])
    assert (stats["right"], stats["scores"]) == (1, {"strict": 0.5, "loose": 1.0})
    text = evaluate.report(results, ["m"])
    assert "| Model | Right | Share | strict | loose |" in text and "| `m` | 1/2 | 50% | 50.0% | 100.0% |" in text
    assert "m: 1/2 right (strict 50.0%, loose 100.0%)" in evaluate.summary(results, ["m"])


def test_a_check_can_return_a_fraction():
    results = _scored(
        lambda prompt, expected, answer: {"quality": 0.5 if answer == "half" else 1.0},
        {"one": "half", "two": "SELECT 2"},
    )
    [stats] = evaluate.model_stats(results, ["m"])
    assert (stats["right"], stats["scores"]) == (1, {"quality": 0.75})  # a half score is not a right answer
    assert "| Model | Right | Share | quality |" in evaluate.report(results, ["m"])


def test_one_true_false_score_has_no_extra_column():
    results = _scored(evaluate.normalized_equal, {"one": "SELECT 1", "two": "x"})
    assert "| Model | Right | Share | Median s per answer |" in evaluate.report(results, ["m"])


def test_a_score_outside_zero_to_one_is_an_error_not_a_score():
    results = _scored(lambda prompt, expected, answer: {"quality": 2}, {"one": "SELECT 1", "two": "SELECT 2"})
    assert [r["ok"] for r in results] == [False, False]
    assert "between 0 and 1" in results[0]["error"]


@pytest.mark.parametrize("result", [True, 0.5, {}, "yes"])  # a bare value is not a dict of named scores
def test_a_check_must_return_named_scores(result):
    results = _scored(lambda prompt, expected, answer: result, {"one": "SELECT 1", "two": "SELECT 2"})
    assert [r["ok"] for r in results] == [False, False]
    assert "named scores" in results[0]["error"]


def test_a_bool_counts_as_one_or_zero_and_the_built_in_check_gives_numbers():
    assert evaluate.to_scores({"strict": True, "loose": 0.5}) == {"strict": 1.0, "loose": 0.5}
    assert evaluate.normalized_equal([], "SELECT  1", "SELECT 1") == {"exact": 1.0}
    assert evaluate.normalized_equal([], "SELECT 1", "SELECT 2") == {"exact": 0.0}
