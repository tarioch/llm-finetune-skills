"""Every command a SKILL.md tells Claude to run is pre-approved by its ``allowed-tools``, except publishing.

Claude Code matches a ``Bash(<prefix> *)`` rule against a command that starts with ``<prefix> `` and a rule without
``*`` against the exact command. A step whose command no rule covers needs a permission prompt, and in auto mode
Claude Code refuses to run the skill's scripts at all.
"""

import re
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "skills"
NOT_PRE_APPROVED = ("ollama cp ", "ollama push ", "ollama rm ")


def allowed_rules(skill: str) -> list[str]:
    head = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8").split("---")[1]
    line = next(line for line in head.splitlines() if line.startswith("allowed-tools: "))
    rules = re.findall(r"Bash\(([^)]*)\)", line.removeprefix("allowed-tools: "))
    assert " ".join(f"Bash({r})" for r in rules) == line.removeprefix("allowed-tools: "), "only Bash rules expected"
    return rules


def commands(skill: str) -> list[str]:
    text = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    return [line.strip() for block in blocks for line in block.splitlines() if line.strip()]


def approved(command: str, rules: list[str]) -> bool:
    for rule in rules:
        if rule.endswith(" *"):
            if command.startswith(rule[:-1]):
                return True
        elif command == rule:
            return True
    return False


@pytest.mark.parametrize("skill", sorted(p.name for p in SKILLS.iterdir()))
def test_every_step_is_pre_approved(skill):
    rules = allowed_rules(skill)
    for command in commands(skill):
        if command.startswith(NOT_PRE_APPROVED):
            assert not approved(command, rules), f"{skill}: {command!r} must keep asking"
        else:
            assert approved(command, rules), f"{skill}: no allowed-tools rule covers {command!r}"


@pytest.mark.parametrize("skill", sorted(p.name for p in SKILLS.iterdir()))
def test_scripts_are_called_through_the_skill_dir(skill):
    # ${CLAUDE_SKILL_DIR} is substituted in the rules and in SKILL.md alike, so they match wherever it is installed
    for command in commands(skill):
        assert "scripts/" not in command or "${CLAUDE_SKILL_DIR}/scripts/" in command, command


def test_rules_are_narrow():
    for skill in sorted(p.name for p in SKILLS.iterdir()):
        for rule in allowed_rules(skill):
            assert rule.split()[0] in {
                "${CLAUDE_SKILL_DIR}/scripts/run",
                "python",
                "python3",
                "uv",
                "ollama",
                "nvidia-smi",
            }
            assert not rule.startswith(("python *", "python3 *", "uv run python *")), (
                f"{skill}: {rule} allows any script"
            )
