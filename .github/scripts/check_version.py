#!/usr/bin/env python
"""Fail a pull request that changes ``skills/`` without a ``plugin.json`` version above the latest release tag.

Claude Code decides whether to update an installed plugin by its ``version``: without a bump, ``claude plugin update``
keeps users on the old copy. The first change after a release has to raise the version, the next ones pass until
the release is published and the tag catches up.

    python .github/scripts/check_version.py origin/main     # the base branch the pull request targets

Standard library only. Needs the tags and the base branch in the checkout (``fetch-depth: 0``).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
WATCHED = ("skills/",)


def parse(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"version {version!r} is not X.Y.Z")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def latest_release(tags: list[str]) -> tuple[int, int, int] | None:
    versions = [tuple(int(part) for part in m.groups()) for tag in tags if (m := TAG.match(tag))]
    return max(versions, default=None)  # type: ignore[type-var]


def problem(version: str, tags: list[str], changed: list[str]) -> str | None:
    """Why the pull request fails the check, or None when it passes."""
    if not any(path.startswith(WATCHED) for path in changed):
        return None
    latest = latest_release(tags)
    if latest is not None and parse(version) <= latest:
        newest = ".".join(map(str, latest))
        return (
            f"skills/ changed, but plugin.json's version {version} is not above the latest release v{newest}: "
            "bump it, or `claude plugin update` keeps users on the old copy"
        )
    return None


def git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)  # noqa: S603, S607
    return result.stdout.split()


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    version = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))["version"]
    message = problem(version, git("tag"), git("diff", "--name-only", f"{base}...HEAD"))
    if message:
        print(f"ERROR {message}", file=sys.stderr)
        return 1
    print(f"ok, plugin.json version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
