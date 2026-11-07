"""The launcher skills/unsloth-lora/scripts/run: which Python runs a script, and in which folder."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUN = Path(__file__).resolve().parents[3] / "skills" / "unsloth-lora" / "scripts" / "run"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.fixture
def fake_python(tmp_path):
    """A stand-in for the Unsloth Python that prints its working directory and arguments."""
    script = tmp_path / "fake-python"
    # pwd -W prints a Windows path in Git Bash, plain pwd elsewhere
    script.write_text('#!/usr/bin/env bash\necho "cwd=$(pwd -W 2>/dev/null || pwd)"\necho "args=$*"\n')
    script.chmod(0o755)
    return script


def cwd(result):
    line = next(line for line in result.stdout.splitlines() if line.startswith("cwd="))
    return Path(line.removeprefix("cwd=")).resolve()


def run(*args, env=None, home=None):
    environment = {k: v for k, v in os.environ.items() if k not in ("UNSLOTH_PYTHON", "XDG_CONFIG_HOME")}
    if home is not None:
        environment["HOME"] = str(home)
    environment.update(env or {})
    return subprocess.run(
        [shutil.which("bash"), RUN.as_posix(), *args], capture_output=True, text=True, env=environment
    )


def test_unsloth_scripts_use_unsloth_python_in_the_runs_folder(tmp_path, fake_python):
    out = tmp_path / "runs" / "my-run"
    result = run("train.py", "--model", "m", "--out", str(out), env={"UNSLOTH_PYTHON": str(fake_python)})
    assert result.returncode == 0, result.stderr
    assert cwd(result) == (tmp_path / "runs").resolve()  # unsloth_compiled_cache/ lands next to the runs
    assert (tmp_path / "runs").is_dir()  # created if missing
    assert f"train.py --model m --out {out}" in result.stdout


def test_run_option_with_equals_sign(tmp_path, fake_python):
    result = run("export_gguf.py", f"--run={tmp_path / 'runs' / 'r'}", env={"UNSLOTH_PYTHON": str(fake_python)})
    assert result.returncode == 0, result.stderr
    assert cwd(result) == (tmp_path / "runs").resolve()


def test_unsloth_python_from_the_config_file(tmp_path, fake_python):
    config = tmp_path / "home" / ".config" / "llm-finetune" / "unsloth-python"
    config.parent.mkdir(parents=True)
    config.write_text(f"{fake_python}\r\n")  # written on Windows
    result = run("preflight.py", "--out", str(tmp_path / "runs" / "r"), home=tmp_path / "home")
    assert result.returncode == 0, result.stderr
    assert "args=" in result.stdout


def test_no_unsloth_python_says_how_to_set_it(tmp_path):
    result = run("train.py", "--out", str(tmp_path / "r"), home=tmp_path / "empty-home")
    assert result.returncode == 2
    assert "UNSLOTH_PYTHON" in result.stderr and "unsloth-python" in result.stderr


def test_relative_run_folder_is_refused(tmp_path, fake_python):
    result = run("train.py", "--out", "runs/r", env={"UNSLOTH_PYTHON": str(fake_python)})
    assert result.returncode == 2
    assert "absolute" in result.stderr


@pytest.mark.parametrize("script", ["check_data.py", "train_status.py"])
def test_standard_library_scripts_run_with_plain_python(tmp_path, script):
    result = run(script, "--help", home=tmp_path)  # no Unsloth Python configured
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout


@pytest.mark.parametrize("script", ["../train.py", "/bin/sh", "missing.py", ""])
def test_only_scripts_of_the_skill(script):
    result = run(script) if script else run()
    assert result.returncode == 2


def test_launcher_is_executable():
    assert os.access(RUN, os.X_OK) or sys.platform == "win32"
