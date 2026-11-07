"""The CI check that a change to skills/ comes with a plugin.json version above the latest release tag."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_version.py"
spec = importlib.util.spec_from_file_location("check_version", SCRIPT)
check_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_version)

TAGS = ["v0.1.0", "v0.2.0", "v0.10.0", "nightly", "v1.0"]
SKILL_CHANGE = ["skills/ollama-model/SKILL.md", "README.md"]


def test_latest_release_compares_numbers_not_text():
    assert check_version.latest_release(TAGS) == (0, 10, 0)
    assert check_version.latest_release(["nightly"]) is None


@pytest.mark.parametrize("version", ["0.10.0", "0.9.0", "0.1.0"])
def test_skill_change_without_a_bump_fails(version):
    assert "not above the latest release v0.10.0" in check_version.problem(version, TAGS, SKILL_CHANGE)


@pytest.mark.parametrize("version", ["0.10.1", "0.11.0", "1.0.0"])
def test_skill_change_with_a_bump_passes(version):
    assert check_version.problem(version, TAGS, SKILL_CHANGE) is None


def test_changes_outside_skills_need_no_bump():
    assert check_version.problem("0.1.0", TAGS, ["README.md", ".github/workflows/ci.yml", "tests/test_x.py"]) is None


def test_no_release_yet_passes():
    assert check_version.problem("0.1.0", [], SKILL_CHANGE) is None


def test_version_must_be_x_y_z():
    with pytest.raises(ValueError, match="not X.Y.Z"):
        check_version.problem("0.2", TAGS, SKILL_CHANGE)
