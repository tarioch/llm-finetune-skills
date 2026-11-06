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
