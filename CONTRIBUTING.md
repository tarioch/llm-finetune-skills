# Contributing

This repository holds Claude skills (`skills/<name>/SKILL.md` with `scripts/` and `references/`) and is a Claude Code
plugin marketplace (`.claude-plugin/`). The Python in it is not a package: the scripts are standalone files run by
path, the project environment exists for the tests and the checks.

## Layout

| Path | Content |
|---|---|
| `skills/<skill>/SKILL.md` | the workflow Claude follows, frontmatter `name` must match the folder |
| `skills/<skill>/scripts/` | the scripts the skill runs, see the docstring of each |
| `skills/<skill>/references/` | details SKILL.md points to, read only when needed |
| `.claude-plugin/` | `plugin.json` (the plugin, all skills) and `marketplace.json` (so the repo can be added as a marketplace) |
| `tests/<skill>/` | tests for one skill, mirroring `skills/<skill>/` (`tests/unsloth-lora/scripts/test_check_data.py` tests `skills/unsloth-lora/scripts/check_data.py`); its `conftest.py` puts the skill's `scripts/` on the import path |
| `tests/test_handoff.py` | tests across skills: what `unsloth-lora` records in `run.json` makes `ollama-model` build the prompt training saw |
| `tests/fixtures/` | fixtures shared by several skills' tests, `chat-templates/` holds real chat templates |

The skills are: `unsloth-lora` (training, GGUF export) and `ollama-model` (Modelfile, evaluation, publishing).

## Setup and checks

```bash
uv sync --locked --dev
uv run pre-commit run --all-files   # ruff, ruff format, uv-lock, zizmor, whitespace/yaml/toml/json
uv run pytest
```

CI runs the same, plus `claude plugin validate .` (Claude Code's own check of the manifests and skills) and a check of
the SKILL.md frontmatter.

- `pre-commit run --all-files` only looks at files tracked by git, `git add` new files first.
- Regenerate `uv.lock` with the uv version of the `uv-lock` pre-commit hook (`rev` in `.pre-commit-config.yaml`):
  `uvx --from uv==<rev> uv lock`.

## Scripts

- Each skill is self-contained: it is installed on its own, so its scripts import only each other, never another
  skill's. Script names are unique across skills, as each skill's tests put its `scripts/` on the import path.
- Skills hand work over through files, documented in both skills (`references/run-json.md` in `unsloth-lora` and
  `ollama-model`, kept identical by a test). A change to such a file format needs both sides and the handoff test.
- `unsloth-lora`: `check_data.py` and `chat_format.py` use the standard library only. `preflight.py`, `train.py` and
  `export_gguf.py` run in the user's Unsloth environment: import `unsloth` before `transformers`/`trl`, and don't
  add dependencies it doesn't already have. `train.py` and `export_gguf.py` can't run in CI (no GPU): check a change
  to them with a smoke run (`train.py --max-steps 2`, then `export_gguf.py`) on a GPU machine and say in the pull
  request what was run.
- `ollama-model`: standard library only.
- Tests for the scripts that don't need a GPU are required. A new chat format needs a test against the family's real
  chat template, see `skills/unsloth-lora/references/chat-formats.md`.

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
