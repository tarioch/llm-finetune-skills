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
        "def same_digit(prompt, expected, answer):\n    return expected[-1] == answer[-1]\n", encoding="utf-8"
    )
    check = evaluate.load_check(f"{tmp_path / 'checks.py'}:same_digit")
    assert check([], "SELECT 1", "select 1")
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
            return b'{"message": {"role": "assistant", "content": "SELECT 1"}}'

    def urlopen(request, timeout):
        sent["url"], sent["body"], sent["timeout"] = request.full_url, request.data, timeout
        return Response()

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", urlopen)
    assert evaluate.ollama_chat("http://h:1", "m", [{"role": "user", "content": "q"}], 5) == "SELECT 1"
    assert sent["url"] == "http://h:1/api/chat" and sent["timeout"] == 5
    assert b'"temperature": 0' in sent["body"] and b'"stream": false' in sent["body"]


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
        return True

    stdout, stderr = sys.stdout, sys.stderr
    examples = EXAMPLES * (sys.getrecursionlimit() // 2)  # with the wrappers piled up, print() would recurse too deep
    results = evaluate.evaluate(examples, ["m"], lambda model, messages: "SELECT 1", wrapping_check)
    assert sys.stdout is stdout and sys.stderr is stderr
    print(evaluate.summary(results, ["m"]))
    assert f"m: {len(examples)}/{len(examples)} right" in capsys.readouterr().out
