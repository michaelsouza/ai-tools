#!/usr/bin/env python3
"""List open GitHub issues with no open formal blockers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import quote


def run_gh(arguments: list[str]) -> str:
    command = ["gh", *arguments]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("the GitHub CLI (`gh`) is not installed") from None

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"`{' '.join(command)}` failed: {detail}")
    return result.stdout


def gh_json(arguments: list[str], *, paginate: bool = False) -> Any:
    if paginate:
        output = run_gh([*arguments, "--paginate", "--jq", ".[]"])
        try:
            return [
                json.loads(line)
                for line in output.splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as error:
            raise RuntimeError(f"GitHub returned invalid JSON: {error}") from error

    output = run_gh(arguments)
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GitHub returned invalid JSON: {error}") from error

    return value


def discover_repo() -> str:
    return run_gh(
        ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    ).strip()


def open_issues(repo: str) -> list[dict[str, Any]]:
    path = f"repos/{repo}/issues?state=open&per_page=100"
    records = gh_json(["api", path], paginate=True)
    # GitHub's issues endpoint also returns pull requests.
    return [record for record in records if "pull_request" not in record]


def formal_blockers(repo: str, issue_number: int) -> list[dict[str, Any]]:
    path = (
        f"repos/{repo}/issues/{issue_number}/dependencies/blocked_by"
        "?per_page=100"
    )
    return gh_json(["api", path], paginate=True)


def calculate_frontier(
    repo: str, issues: list[dict[str, Any]], jobs: int
) -> list[dict[str, Any]]:
    blockers_by_issue: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(formal_blockers, repo, issue["number"]): issue["number"]
            for issue in issues
        }
        for future in as_completed(futures):
            number = futures[future]
            try:
                blockers_by_issue[number] = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"could not query formal blockers for issue #{number}: {error}"
                ) from error

    frontier = []
    for issue in issues:
        blockers = blockers_by_issue[issue["number"]]
        open_blockers = [
            {
                "number": blocker["number"],
                "title": blocker["title"],
                "url": blocker["html_url"],
            }
            for blocker in blockers
            if blocker.get("state") == "open"
        ]
        if not open_blockers:
            frontier.append(
                {
                    "number": issue["number"],
                    "title": issue["title"],
                    "url": issue["html_url"],
                    "labels": [label["name"] for label in issue.get("labels", [])],
                }
            )
    return sorted(frontier, key=lambda issue: issue["number"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List open issues with no open formal GitHub blockers."
    )
    parser.add_argument("--repo", help="GitHub repository as OWNER/REPO")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="concurrent dependency queries (default: 8)",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        repo = args.repo or discover_repo()
        # Reject malformed values before interpolating an API path.
        owner, separator, name = repo.partition("/")
        if not separator or not owner or not name or "/" in name:
            raise RuntimeError("--repo must have the form OWNER/REPO")
        repo = f"{quote(owner, safe='')}/{quote(name, safe='')}"
        frontier = calculate_frontier(repo, open_issues(repo), args.jobs)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"repository": repo, "issues": frontier}, indent=2))
    elif not frontier:
        print(f"No open, formally unblocked issues in {repo}.")
    else:
        print(f"Open, formally unblocked issues in {repo}:")
        for issue in frontier:
            print(f"#{issue['number']}  {issue['title']}")
            print(f"     {issue['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
