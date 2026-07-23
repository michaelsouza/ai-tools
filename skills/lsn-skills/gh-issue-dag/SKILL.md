---
name: gh-issue-dag
description: List open GitHub issues that are currently executable because every formal GitHub `blocked_by` dependency is closed. Use when asked for unblocked issues, the issue work frontier, or ready work in a repository that uses GitHub's native issue dependencies; do not infer dependencies from issue-body text.
---

# GitHub Issue DAG

Determine the open issue frontier from GitHub's formal dependency graph by running the bundled Python script. Treat an issue as unblocked when it is open and none of the issues returned by GitHub's native `blocked_by` endpoint are open.

## Requirements

- Authenticated GitHub CLI (`gh auth status`).
- Read access to the target repository.

## Run

Execute the script by replacing `<skill-directory>` with the absolute path to this skill's root folder:

```bash
python3 <skill-directory>/scripts/list_unblocked_issues.py
```

### Options

- `--repo OWNER/REPO`: Specify target repository if current working directory is outside the target repo.
- `--json`: Output raw JSON data for machine consumption.

> **Note:** If GitHub's dependency endpoint cannot be queried (e.g. permission or API errors), report the explicit failure instead of assuming the issue has no blockers.

## Report

Report all unblocked issues containing their issue numbers, titles, and direct URLs.

State explicitly that the result relies solely on formal GitHub `blocked_by` relationships and includes parent/spec issues if they have no open blockers.

**Strict rules:**
- Do **not** infer dependencies from issue text, comments, or informal mentions.
- Do **not** silently exclude issues based on labels, sub-issue status, or perceived executability unless explicitly requested by the user.

### Example Output Format

```markdown
### 🚀 Unblocked Issue Frontier
- [#101 - Refactor auth service](https://github.com/owner/repo/issues/101)
- [#104 - Update API specs](https://github.com/owner/repo/issues/104)

*Note: Filtered based strictly on GitHub formal `blocked_by` status.*
```
