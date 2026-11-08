"""The launcher skills/ollama-model/scripts/run."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[3] / "skills" / "ollama-model" / "scripts" / "run"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run(*args, path=None):
    env = dict(os.environ)
    if path is not None:
        env["PATH"] = f"{path}{os.pathsep}{env['PATH']}"
    return subprocess.run([shutil.which("bash"), RUN.as_posix(), *args], capture_output=True, text=True, env=env)


def test_runs_a_script_with_python():
    result = run("modelfile.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "run.json" in result.stdout


def test_uv_runs_in_the_given_project(tmp_path):
    fake_uv = tmp_path / "bin" / "uv"
    fake_uv.parent.mkdir()
    fake_uv.write_text('#!/usr/bin/env bash\necho "uv $*"\n')
    fake_uv.chmod(0o755)
    result = run("--uv", "/some/project", "evaluate.py", "--eval", "e.jsonl", path=fake_uv.parent)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("uv run --locked --project /some/project python ")
    assert result.stdout.rstrip().endswith("evaluate.py --eval e.jsonl")


@pytest.mark.parametrize("args", [(), ("--uv",), ("../modelfile.py",), ("missing.py",)])
def test_refuses_anything_else(args):
    assert run(*args).returncode == 2
