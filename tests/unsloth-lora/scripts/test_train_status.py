import json
import os

import train_status


def checkpoint(run, step, mtime, history, max_steps=300):
    folder = run / "checkpoints" / f"checkpoint-{step}"
    folder.mkdir(parents=True)
    state = folder / "trainer_state.json"
    state.write_text(json.dumps({"global_step": step, "max_steps": max_steps, "log_history": history}), "utf-8")
    os.utime(state, (mtime, mtime))


HISTORY = [
    {"loss": 1.2, "step": 50},
    {"eval_loss": 0.9, "step": 100},
    {"loss": 0.6, "step": 150},
    {"eval_loss": 0.7, "step": 200},
]


def test_running(tmp_path):
    checkpoint(tmp_path, 100, 1000, HISTORY[:2])
    checkpoint(tmp_path, 200, 1200, HISTORY)
    s = train_status.status(tmp_path, now=1200)
    assert (s["step"], s["max_steps"], s["train_loss"], s["eval_loss"]) == (200, 300, 0.6, 0.7)
    assert s["best_eval"] == (200, 0.7) and s["seconds_per_step"] == 2.0 and s["seconds_left"] == 200.0
    text = train_status.describe(s)
    assert text.startswith("step 200/300, train loss 0.6000, validation loss 0.7000")
    assert "about 3 min left (2.0 s per step)" in text and "last checkpoint 0 s ago" in text


def test_time_since_the_last_checkpoint_counts(tmp_path):
    checkpoint(tmp_path, 100, 1000, HISTORY[:2])
    checkpoint(tmp_path, 200, 1200, HISTORY)
    s = train_status.status(tmp_path, now=1260)  # 60 s at 2 s per step: about 30 steps further
    assert s["estimated_step"] == 230 and s["seconds_left"] == 140.0
    text = train_status.describe(s)
    assert text.startswith("step about 230/300 (last checkpoint at 200)") and "about 2 min left" in text
    assert train_status.status(tmp_path, now=99999)["seconds_left"] == 0


def test_best_step_is_not_the_last(tmp_path):
    checkpoint(tmp_path, 200, 1000, [*HISTORY, {"eval_loss": 0.75, "step": 300}])
    s = train_status.status(tmp_path, now=1000)
    assert s["best_eval"] == (200, 0.7) and s["eval_loss"] == 0.75 and "seconds_left" not in s


def test_checkpoints_sorted_by_step_not_name(tmp_path):
    checkpoint(tmp_path, 900, 1000, HISTORY, max_steps=1000)
    checkpoint(tmp_path, 1000, 1100, HISTORY, max_steps=1000)
    assert [step for step, _ in train_status.checkpoints(tmp_path)] == [900, 1000]
    assert train_status.status(tmp_path, now=1100)["step"] == 1000


def test_finished(tmp_path):
    checkpoint(tmp_path, 100, 1000, HISTORY)
    (tmp_path / "run.json").write_text(json.dumps({"steps": 792, "train_loss": 0.21, "eval_loss": 0.24}), "utf-8")
    text = train_status.describe(train_status.status(tmp_path))
    assert text == "finished after 792 steps, train loss 0.2100, validation loss 0.2400, not exported yet"


def test_nothing_yet(tmp_path):
    assert train_status.describe(train_status.status(tmp_path)).startswith("no checkpoint yet")
