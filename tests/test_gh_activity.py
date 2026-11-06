import unittest

from tools.gh_activity import parse_repo, render_markdown

SINCE = "2026-10-22T00:00:00Z"


def commit(oid, message, date="2026-10-25T10:00:00Z", login="alice"):
    return {"oid": oid, "message": message, "committedDate": date, "author": {"name": "A", "user": {"login": login}}}


def ref(name, commits, more=False):
    return {"name": name, "target": {"history": {"pageInfo": {"hasNextPage": more}, "nodes": commits}}}


def item(
    number, state="OPEN", created="2026-10-01T00:00:00Z", closed=None, merged=None, updated="2026-10-30T00:00:00Z"
):
    return {
        "number": number,
        "title": f"Item {number}",
        "state": state,
        "createdAt": created,
        "closedAt": closed,
        "mergedAt": merged,
        "updatedAt": updated,
        "author": {"login": "alice"},
        "comments": {
            "totalCount": 1,
            "nodes": [{"body": "Done.\n\nDetails", "createdAt": "2026-10-30T00:00:00Z", "author": {"login": "bob"}}],
        },
    }


def payload(refs, issues=(), prs=()):
    return {
        "data": {
            "repository": {
                "defaultBranchRef": {"name": "main"},
                "refs": {"nodes": refs},
                "issues": {"nodes": list(issues)},
                "pullRequests": {"nodes": list(prs)},
            }
        }
    }


class ParseRepoTests(unittest.TestCase):
    def test_commits_listed_once_under_default_branch(self):
        shared = commit("a" * 40, "Shared commit\n\nBody")
        data = payload([ref("feature", [shared, commit("b" * 40, "Feature only")]), ref("main", [shared])])
        activity = parse_repo("me/repo", data, SINCE)
        self.assertEqual([(c.sha, c.branch) for c in activity.commits], [("aaaaaaa", "main"), ("bbbbbbb", "feature")])
        self.assertEqual(activity.commits[0].headline, "Shared commit")

    def test_truncated_history_is_reported(self):
        activity = parse_repo("me/repo", payload([ref("main", [commit("a" * 40, "x")], more=True)]), SINCE)
        self.assertEqual(activity.truncated_branches, ["main"])

    def test_items_outside_window_are_dropped_and_merged_prs_marked(self):
        old_pr = item(1, state="CLOSED", updated="2026-09-01T00:00:00Z")
        merged = item(2, state="CLOSED", closed="2026-10-30T00:00:00Z", merged="2026-10-30T00:00:00Z")
        activity = parse_repo("me/repo", payload([], prs=[merged, old_pr]), SINCE)
        self.assertEqual([(i.number, i.state) for i in activity.items], [(2, "MERGED")])


class RenderMarkdownTests(unittest.TestCase):
    def render(self, data):
        return render_markdown([parse_repo("me/repo", data, SINCE)], SINCE, "2026-11-06T00:00:00Z", 100)

    def test_tags_show_open_and_close_dates_once(self):
        issue = item(5, state="CLOSED", created="2026-10-23T00:00:00Z", closed="2026-10-30T00:00:00Z")
        text = self.render(payload([], issues=[issue]))
        self.assertIn("- #5 [opened 2026-10-23, closed 2026-10-30] Item 5 (1 comments)", text)
        self.assertIn("last comment 2026-10-30 by bob: Done. Details", text)

    def test_branch_shown_only_off_default(self):
        data = payload([ref("main", [commit("a" * 40, "On main")]), ref("dev", [commit("b" * 40, "On dev")])])
        text = self.render(data)
        self.assertIn("aaaaaaa On main", text)
        self.assertIn("bbbbbbb [dev] On dev", text)

    def test_repo_without_activity_is_omitted(self):
        self.assertNotIn("## me/repo", self.render(payload([])))


if __name__ == "__main__":
    unittest.main()
