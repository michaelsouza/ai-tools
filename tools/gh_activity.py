#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
GitHub Activity - Collect a user's GitHub activity in a date window as Markdown

Gathers, for every repository with activity since a date, the commits on all
branches (each commit listed once, default branch first), the issues and pull
requests updated in the window, and the last comment of each. The output is a
compact Markdown (or JSON) file meant as raw material for an activity summary
written by a person or an LLM.

Usage:
    # Last 15 days, written to outputs/activity/<today>.md
    python tools/gh_activity.py

    # Explicit window and output
    python tools/gh_activity.py --since 2026-10-22 -o report.md

    # Also list local commits not yet pushed, from the repos under ~/gitrepos
    python tools/gh_activity.py --local ~/gitrepos

    # Structured output
    python tools/gh_activity.py --json -o activity.json

Repositories covered:
    - repos owned by the user that were pushed or updated in the window;
    - repos of other owners where the user authored, commented, was assigned
      or mentioned in an issue or pull request updated in the window.

Requirements:
    - GitHub CLI (`gh`) installed and authenticated (`gh auth login`).
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_QUERY = """
query($owner: String!, $name: String!, $since: GitTimestamp!, $issuesSince: DateTime!) {
  repository(owner: $owner, name: $name) {
    defaultBranchRef { name }
    refs(refPrefix: "refs/heads/", first: 100) {
      nodes {
        name
        target {
          ... on Commit {
            history(since: $since, first: 100) {
              pageInfo { hasNextPage }
              nodes { oid message committedDate author { name user { login } } }
            }
          }
        }
      }
    }
    issues(first: 100, filterBy: {since: $issuesSince}, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title state createdAt closedAt updatedAt author { login }
        comments(last: 1) { totalCount nodes { body createdAt author { login } } }
      }
    }
    pullRequests(first: 50, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title state createdAt closedAt mergedAt updatedAt author { login }
        comments(last: 1) { totalCount nodes { body createdAt author { login } } }
      }
    }
  }
}
"""


@dataclass
class Commit:
    sha: str
    headline: str
    date: str
    author: str
    branch: str


@dataclass
class Item:
    kind: str  # "issue" or "pr"
    number: int
    title: str
    state: str
    created: str
    closed: str
    updated: str
    author: str
    comments: int
    last_comment: str = ""
    last_comment_author: str = ""
    last_comment_date: str = ""


@dataclass
class RepoActivity:
    name: str
    default_branch: str = ""
    commits: list = field(default_factory=list)
    truncated_branches: list = field(default_factory=list)
    items: list = field(default_factory=list)
    unpushed: list = field(default_factory=list)
    dirty: int = 0


def gh(args):
    """Run a gh command and return its stdout."""
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()}")
    return result.stdout


def gh_json(args):
    return json.loads(gh(args) or "null")


def active_repos(login, since):
    """Owned repos touched since `since`, plus repos with issues/PRs involving the user."""
    repos = gh_json(["repo", "list", login, "--limit", "1000", "--json", "nameWithOwner,pushedAt,updatedAt"])
    names = {r["nameWithOwner"] for r in repos if max(r["pushedAt"] or "", r["updatedAt"] or "") >= since}
    for kind in ("issues", "prs"):
        found = gh_json(
            ["search", kind, "--involves=@me", f"--updated=>={since[:10]}", "--limit", "1000", "--json", "repository"]
        )
        names.update(f["repository"]["nameWithOwner"] for f in found)
    return sorted(names)


def parse_repo(name, data, since):
    """Turn the GraphQL repository payload into a RepoActivity."""
    repo = data["data"]["repository"]
    default = (repo.get("defaultBranchRef") or {}).get("name", "")
    activity = RepoActivity(name=name, default_branch=default)

    # Default branch first, so commits merged into it are attributed there
    refs = sorted(repo["refs"]["nodes"], key=lambda r: (r["name"] != default, r["name"]))
    seen = set()
    for ref in refs:
        history = (ref.get("target") or {}).get("history")
        if not history:
            continue
        if history["pageInfo"]["hasNextPage"]:
            activity.truncated_branches.append(ref["name"])
        for c in history["nodes"]:
            if c["oid"] in seen:
                continue
            seen.add(c["oid"])
            author = c["author"] or {}
            login = (author.get("user") or {}).get("login") or author.get("name") or ""
            activity.commits.append(
                Commit(c["oid"][:7], c["message"].split("\n", 1)[0], c["committedDate"], login, ref["name"])
            )

    for kind, key in (("issue", "issues"), ("pr", "pullRequests")):
        for n in repo[key]["nodes"]:
            if n["updatedAt"] < since:
                continue
            last = (n["comments"]["nodes"] or [{}])[0]
            activity.items.append(
                Item(
                    kind=kind,
                    number=n["number"],
                    title=n["title"],
                    state="MERGED" if n.get("mergedAt") else n["state"],
                    created=n["createdAt"],
                    closed=n.get("closedAt") or "",
                    updated=n["updatedAt"],
                    author=(n.get("author") or {}).get("login", ""),
                    comments=n["comments"]["totalCount"],
                    last_comment=(last.get("body") or "").strip(),
                    last_comment_author=(last.get("author") or {}).get("login", ""),
                    last_comment_date=last.get("createdAt", ""),
                )
            )
    return activity


def fetch_repo(name, since):
    owner, repo = name.split("/", 1)
    data = gh_json(
        [
            "api", "graphql",
            "-f", f"query={REPO_QUERY}",
            "-F", f"owner={owner}",
            "-F", f"name={repo}",
            "-F", f"since={since}",
            "-F", f"issuesSince={since}",
        ]
    )  # fmt: skip
    return parse_repo(name, data, since)


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def remote_name(path):
    """owner/name of the GitHub `origin` remote, or '' if there is none."""
    url = git(path, "remote", "get-url", "origin").strip()
    if "github.com" not in url:
        return ""
    tail = url.split("github.com", 1)[1].lstrip(":/")
    return tail.removesuffix(".git")


def recently_changed(path, since):
    """Number of uncommitted paths whose file was modified since `since`."""
    cutoff = datetime.fromisoformat(since.replace("Z", "+00:00")).timestamp()
    count = 0
    for line in git(path, "status", "--porcelain").splitlines():
        target = path / line[3:].split(" -> ")[-1].strip('"')
        try:
            if target.stat().st_mtime >= cutoff:
                count += 1
        except OSError:  # deleted files have no date, so they are not counted
            pass
    return count


def scan_local(root, since, activities):
    """Add unpushed commits and uncommitted changes of local clones under `root`."""
    by_name = {a.name.lower(): a for a in activities}
    for path in sorted(Path(root).expanduser().iterdir()):
        if not (path / ".git").exists():
            continue
        unpushed = git(path, "log", "--branches", "--not", "--remotes", f"--since={since}", "--format=%h %s")
        dirty = recently_changed(path, since)
        if not unpushed.strip() and not dirty:
            continue
        name = remote_name(path) or f"local:{path.name}"
        activity = by_name.get(name.lower())
        if activity is None:
            activity = RepoActivity(name=name)
            activities.append(activity)
            by_name[name.lower()] = activity
        activity.unpushed = unpushed.splitlines()
        activity.dirty = dirty


def in_window(stamp, since):
    return bool(stamp) and stamp >= since


def shorten(text, limit):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(activities, since, until, comment_chars):
    lines = [f"# GitHub activity {since[:10]} → {until[:10]}", ""]
    for a in activities:
        if not (a.commits or a.items or a.unpushed or a.dirty):
            continue
        lines += [f"## {a.name}", ""]
        if a.commits:
            lines.append(f"**Commits ({len(a.commits)})**")
            for c in a.commits:
                branch = "" if c.branch == a.default_branch else f" [{c.branch}]"
                lines.append(f"- {c.date[:10]} {c.sha}{branch} {c.headline} ({c.author})")
            if a.truncated_branches:
                lines.append(f"- … more commits not listed on: {', '.join(a.truncated_branches)}")
            lines.append("")
        if a.unpushed or a.dirty:
            lines.append(f"**Local** ({len(a.unpushed)} unpushed commits, {a.dirty} uncommitted changes)")
            lines += [f"- {u}" for u in a.unpushed]
            lines.append("")
        if a.items:
            lines.append(f"**Issues and PRs ({len(a.items)})**")
            for i in a.items:
                tags = []
                if in_window(i.created, since):
                    tags.append(f"opened {i.created[:10]}")
                if in_window(i.closed, since):
                    tags.append(f"{i.state.lower()} {i.closed[:10]}")
                elif i.state != "OPEN" or not tags:
                    tags.append(i.state.lower())
                prefix = "PR " if i.kind == "pr" else ""
                lines.append(f"- {prefix}#{i.number} [{', '.join(tags)}] {i.title} ({i.comments} comments)")
                if i.last_comment and in_window(i.last_comment_date, since):
                    body = shorten(i.last_comment, comment_chars)
                    lines.append(f"  - last comment {i.last_comment_date[:10]} by {i.last_comment_author}: {body}")
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect GitHub activity in a date window as Markdown or JSON.")
    parser.add_argument("--since", help="start date YYYY-MM-DD (default: 15 days ago)")
    parser.add_argument("-o", "--output", help="output file (default: outputs/activity/<today>.md or .json)")
    parser.add_argument("--json", action="store_true", help="write JSON instead of Markdown")
    parser.add_argument("--local", metavar="DIR", help="also scan local clones under DIR for unpushed work")
    parser.add_argument("--comment-chars", type=int, default=600, help="max characters per last comment (600)")
    args = parser.parse_args(argv)

    today = date.today()
    start = date.fromisoformat(args.since) if args.since else today - timedelta(days=15)
    since = datetime(start.year, start.month, start.day, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    until = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        login = gh(["api", "user", "-q", ".login"]).strip()
        names = active_repos(login, since)
        activities = []
        for k, name in enumerate(names, 1):
            print(f"[{k}/{len(names)}] {name}", file=sys.stderr)
            activities.append(fetch_repo(name, since))
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.local:
        scan_local(args.local, since, activities)

    if args.json:
        text = json.dumps({"since": since, "until": until, "repos": [asdict(a) for a in activities]}, indent=2)
    else:
        text = render_markdown(activities, since, until, args.comment_chars)

    suffix = ".json" if args.json else ".md"
    output = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent.parent / "outputs" / "activity" / f"{today.isoformat()}{suffix}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
