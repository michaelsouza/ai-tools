---
name: gh-issue-dag
description: List open GitHub issues that are currently executable because every formal GitHub `blocked_by` dependency is closed. Use when asked for unblocked issues, the issue work frontier, or ready work in a repository that uses GitHub's native issue dependencies; do not infer dependencies from issue-body text.
---

# GitHub Issue DAG

Determine the open issue frontier from GitHub's formal dependency graph by
running the bundled Python script. Treat an issue as unblocked when it is open
and none of the issues returned by GitHub's native `blocked_by` endpoint are
open.

## Run

From any directory inside the target repository, run:

```bash
python3 <skill-directory>/scripts/list_unblocked_issues.py
```

Pass `--repo OWNER/REPO` when the current directory is not in the target
repository. Pass `--json` when machine-readable output is useful.

The script requires an authenticated `gh` CLI and read access to the repository.
If GitHub's dependency endpoint cannot be queried, report the error instead of
assuming that the affected issue has no blockers.

## Report

Report the unblocked issues with their numbers, titles, and URLs. State that the
result uses formal GitHub `blocked_by` relationships and includes parent/spec
issues when they themselves have no open blockers. Do not silently exclude an
issue based on labels, issue-body references, parent/sub-issue relationships, or
whether it appears executable; apply such filtering only when the user asks.
