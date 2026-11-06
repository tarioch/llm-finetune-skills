# Contributing

This repository holds Claude skills (`skills/<name>/SKILL.md` with `scripts/` and `references/`) and is a Claude Code
plugin marketplace (`.claude-plugin/`). The Python in it is not a package: the scripts are standalone files run by
path, the project environment exists for the tests and the checks.

## Layout

| Path | Content |
|---|---|
| `skills/unsloth-ollama/SKILL.md` | the workflow Claude follows, frontmatter `name` must match the folder |
| `skills/unsloth-ollama/scripts/` | the scripts the skill runs, see the docstring of each |
| `skills/unsloth-ollama/references/` | details SKILL.md points to, read only when needed |
| `.claude-plugin/` | `plugin.json` (the plugin) and `marketplace.json` (so the repo can be added as a marketplace) |
| `tests/<skill>/` | tests for a skill, mirroring `skills/<skill>/` (`tests/unsloth-ollama/scripts/test_check_data.py` tests `skills/unsloth-ollama/scripts/check_data.py`); its `conftest.py` puts the skill's `scripts/` on the import path |

## Setup and checks

```bash
uv sync --locked --dev
uv run pre-commit run --all-files   # ruff, ruff format, uv-lock, zizmor, whitespace/yaml/toml/json
uv run pytest
```

CI runs the same, plus a check of the plugin manifests and the SKILL.md frontmatter.

- `pre-commit run --all-files` only looks at files tracked by git, `git add` new files first.
- Regenerate `uv.lock` with the uv version of the `uv-lock` pre-commit hook (`rev` in `.pre-commit-config.yaml`):
  `uvx --from uv==<rev> uv lock`.

## Scripts

- `check_data.py`, `chat_formats.py`, `make_modelfile.py` and `evaluate.py` use the standard library only, so they
  run anywhere. `preflight.py`, `train.py` and `export_gguf.py` run in the user's Unsloth environment: import
  `unsloth` before `transformers`/`trl`, and don't add dependencies it doesn't already have.
- `train.py` and `export_gguf.py` can't run in CI (no GPU). A change to them is checked with a smoke run
  (`train.py --max-steps 2`, then `export_gguf.py`) on a GPU machine, say in the pull request what was run.
- Tests for the scripts that don't need a GPU are required. Fixtures live next to the tests that use them
  (`tests/unsloth-ollama/scripts/fixtures/` holds real chat templates).
- A new chat format needs a test against the family's real chat template, see
  `skills/unsloth-ollama/references/chat-formats.md`.

## Skills

- SKILL.md is instructions for Claude: steps, the commands, what to check in the output, and when to ask the user.
  Keep it short and move details into `references/`.
- The `description` decides when the skill is used: say what it does and name the tools.
- No personal setup in the skills: machine names, paths, accounts, model names of a specific project.

## Git and pull requests

- Branch off `main`, named `feature/…`, `bugfix/…` or `chore/…` (snake_case after the prefix). The prefix labels
  the PR, the `Label check` fails without one of these labels, and the label decides the release notes category.
- Commit subjects are imperative and start with a capital letter, the body explains why.
- Changes go through pull requests into `main` and are merged with a merge commit. `CI ok` is the check to require.
- Releases are drafted by release-drafter. Bump `version` in `.claude-plugin/plugin.json` with a release, Claude
  Code uses it to notice updates.
