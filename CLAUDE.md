# CLAUDE.md

@CONTRIBUTING.md

## Notes for Claude

- Open pull requests and stop there, the maintainer reviews and merges them. Do not merge unless explicitly asked.
- Run the checks from CONTRIBUTING.md before pushing, and `git add` new files first.
- The skills are public: never put real data, credentials, machine names, local paths or account details into them,
  their examples or the tests.
- Show that a test can fail: when fixing a bug, check that the new test fails without the fix.
- Keep pull request descriptions factual: the problem, the change, how it was verified (including what could not be
  verified, e.g. no GPU run).
