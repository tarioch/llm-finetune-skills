"""The launcher ``scripts/run``: picks the Unsloth Python, runs only the skill's own scripts, ``-C`` sets the folder."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[3] / "skills" / "unsloth-lora" / "scripts" / "run"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def launch(*args, env=None, cwd=None):
    return subprocess.run([BASH, str(RUN), *args], capture_output=True, text=True, env=env, cwd=cwd, check=False)


@pytest.fixture
def env(tmp_path):
    """No UNSLOTH_PYTHON and an empty config folder, so nothing from this machine leaks in."""
    environment = {k: v for k, v in os.environ.items() if k != "UNSLOTH_PYTHON"}
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    return environment


def test_runs_a_script_with_unsloth_python(env):
    result = launch("check_data", "--help", env={**env, "UNSLOTH_PYTHON": sys.executable})
    assert result.returncode == 0, result.stderr
    assert "usage: check_data.py" in result.stdout


def test_reads_the_python_from_the_config_file(env, tmp_path):
    conf = tmp_path / "config" / "llm-finetune" / "unsloth-python"
    conf.parent.mkdir(parents=True)
    conf.write_bytes(sys.executable.encode() + b"\r\n")  # written on Windows, with a CRLF
    result = launch("check_data", "--help", env=env)
    assert result.returncode == 0, result.stderr


def test_says_how_to_configure_the_python(env):
    result = launch("check_data", "--help", env=env)
    assert result.returncode == 2
    assert "UNSLOTH_PYTHON" in result.stderr and "unsloth-python" in result.stderr


@pytest.mark.parametrize("script", ["missing", "../../evil", "check_data.py", ""])
def test_only_runs_the_skills_own_scripts(env, script):
    result = launch(script, env={**env, "UNSLOTH_PYTHON": sys.executable})
    assert result.returncode == 2
    assert "check_data" in result.stderr  # lists what it can run


def test_dash_c_sets_the_working_folder(env, tmp_path):
    data = tmp_path / "runs" / "data"
    data.mkdir(parents=True)
    for name, question in (("train.jsonl", "a"), ("val.jsonl", "b")):
        line = json.dumps({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": "x"}]})
        (data / name).write_text(line + "\n", encoding="utf-8")
    env = {**env, "UNSLOTH_PYTHON": sys.executable}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert launch("check_data", "data", env=env, cwd=elsewhere).returncode != 0
    result = launch("-C", str(tmp_path / "runs"), "check_data", "data", env=env, cwd=elsewhere)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
