import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

try:
    import pymupdf
except ImportError:  # run with ``uv run --with pymupdf``
    pymupdf = None

if pymupdf is not None:
    from tools.pdf_review import (
        CommentJournal,
        CommentStore,
        PassageIndex,
        ReviewHistory,
        StaleVersion,
        make_handler,
        merge_lines,
        review_copy,
    )


def fractions_of(page: "pymupdf.Page", word: str) -> list[float]:
    """Where ``word`` is shown on the page, as fractions, the way the browser sends it."""
    (rect,) = [pymupdf.Rect(w[:4]) for w in page.get_text("words") if w[4] == word]
    shown = rect * page.rotation_matrix
    width, height = page.rect.width, page.rect.height
    return [shown.x0 / width, shown.y0 / height, shown.x1 / width, shown.y1 / height]


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class CommentStoreTests(unittest.TestCase):
    def make_pdf(self, rotation: int = 0) -> Path:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "doc.pdf"
        document = pymupdf.open()
        for text in ("first page alpha", "second page beta gamma"):
            page = document.new_page()
            page.insert_text((72, 100), text)
            page.set_rotation(rotation)
        document.save(path)
        return path

    def rects_for(self, path: Path, page_number: int, word: str) -> list[list[float]]:
        with pymupdf.open(path) as document:
            return [fractions_of(document[page_number - 1], word)]

    def test_added_comment_is_listed_with_its_marked_text(self):
        path = self.make_pdf()
        store = CommentStore(path, "ana")
        store.add(2, self.rects_for(path, 2, "beta"), "check this")
        (comment,) = store.comments()
        self.assertEqual(
            (comment["page"], comment["marked"], comment["text"], comment["author"]),
            (2, "beta", "check this", "ana"),
        )

    def test_comment_is_a_highlight_other_readers_see(self):
        path = self.make_pdf()
        CommentStore(path, "ana").add(1, self.rects_for(path, 1, "alpha"), "note")
        with pymupdf.open(path) as document:
            page = document[0]
            (annot,) = page.annots()
            self.assertEqual((annot.type[1], annot.info["content"]), ("Highlight", "note"))

    def test_edit_and_remove(self):
        path = self.make_pdf()
        store = CommentStore(path, "ana")
        added = store.add(1, self.rects_for(path, 1, "alpha"), "old")
        store.edit(1, added["xref"], "new")
        self.assertEqual([c["text"] for c in store.comments()], ["new"])
        store.remove(1, added["xref"])
        self.assertEqual(store.comments(), [])

    def test_comments_follow_page_then_position(self):
        path = self.make_pdf()
        store = CommentStore(path, "ana")
        store.add(2, self.rects_for(path, 2, "gamma"), "c")
        store.add(1, self.rects_for(path, 1, "alpha"), "a")
        store.add(2, self.rects_for(path, 2, "beta"), "b")
        self.assertEqual([c["text"] for c in store.comments()], ["a", "b", "c"])

    def test_rotated_page_marks_the_selected_word(self):
        path = self.make_pdf(rotation=90)
        store = CommentStore(path, "ana")
        rects = self.rects_for(path, 1, "alpha")
        added = store.add(1, rects, "turned")
        self.assertEqual(added["marked"], "alpha")
        for got, sent in zip(added["rects"][0], rects[0]):
            self.assertAlmostEqual(got, sent, delta=0.01)

    def test_missing_comment_is_reported(self):
        store = CommentStore(self.make_pdf(), "ana")
        with self.assertRaises(KeyError):
            store.edit(1, 9999, "nothing here")

    def test_comment_needs_a_mark(self):
        store = CommentStore(self.make_pdf(), "ana")
        with self.assertRaises(ValueError):
            store.add(1, [], "no place")

    def test_change_against_an_older_version_is_refused(self):
        path = self.make_pdf()
        store = CommentStore(path, "ana")
        seen = store.version()
        added = store.add(1, self.rects_for(path, 1, "alpha"), "first", seen)
        for change in (
            lambda: store.add(2, self.rects_for(path, 2, "beta"), "late", seen),
            lambda: store.edit(1, added["xref"], "late", seen),
            lambda: store.remove(1, added["xref"], seen),
        ):
            with self.assertRaises(StaleVersion):
                change()
        self.assertEqual([c["text"] for c in store.comments()], ["first"])


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class MergeLinesTests(unittest.TestCase):
    def test_fragments_of_one_line_merge(self):
        self.assertEqual(merge_lines([[0.30, 0.1, 0.40, 0.12], [0.1, 0.1, 0.301, 0.12]]), [[0.1, 0.1, 0.40, 0.12]])

    def test_same_line_of_two_columns_stays_apart(self):
        left, right = [0.1, 0.1, 0.45, 0.12], [0.52, 0.1, 0.9, 0.12]
        self.assertEqual(merge_lines([right, left]), [left, right])

    def test_selection_across_columns_marks_only_selected_words(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "two-columns.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((50, 100), "left column alpha")
        page.insert_text((320, 100), "right column beta")
        document.save(path)
        with pymupdf.open(path) as document:
            rects = [fractions_of(document[0], word) for word in ("alpha", "right")]
        added = CommentStore(path, "ana").add(1, rects, "two columns")
        self.assertEqual((len(added["rects"]), added["marked"]), (2, "alpha right"))


def write_pdf(path: Path, lines: list[str]) -> Path:
    """A one-page PDF with one text line per entry, 14 points apart."""
    document = pymupdf.open()
    page = document.new_page()
    for number, line in enumerate(lines):
        page.insert_text((72, 100 + 14 * number), line)
    document.save(path)
    return path


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class PassageIndexTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def index(self, lines: list[str]) -> "PassageIndex":
        with pymupdf.open(write_pdf(self.folder / "doc.pdf", lines)) as document:
            return PassageIndex(document)

    def test_unchanged_passage_is_found_with_one_rectangle_per_line(self):
        found = self.index(["the order of the", "edges matters here"]).find("of the edges matters")
        self.assertEqual((found["status"], found["page"], len(found["rects"])), ("found", 1, 2))

    def test_punctuation_and_case_do_not_matter(self):
        self.assertEqual(self.index(["Edges, as (before)."]).find("edges as before")["status"], "found")

    def test_word_hyphenated_across_lines_is_found_whole(self):
        self.assertEqual(self.index(["a compu-", "tational cost"]).find("computational cost")["status"], "found")

    def test_rewritten_passage_is_changed(self):
        self.assertEqual(self.index(["the new wording"]).find("the old wording")["status"], "changed")

    def test_passage_in_several_places_is_repeated(self):
        found = self.index(["the cost", "and the cost"]).find("the cost")
        self.assertEqual((found["status"], found["count"]), ("repeated", 2))

    def test_note_without_marked_words(self):
        self.assertEqual(self.index(["anything"]).find("")["status"], "no-text")


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class HistoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.pdf = write_pdf(self.folder / "paper.pdf", ["alpha beta gamma", "delta epsilon"])
        self.store = CommentStore(self.pdf, "ana")
        self.journal = CommentJournal(self.folder / "paper.pdf.comments.json")

    def comment(self, word: str, text: str) -> dict:
        with pymupdf.open(self.pdf) as document:
            rects = [fractions_of(document[0], word)]
        return self.store.add(1, rects, text)

    def rebuild(self, lines: list[str]) -> None:
        """What a LaTeX build does: a new file without the annotations."""
        write_pdf(self.pdf, lines)

    def test_comments_erased_by_a_rebuild_become_history(self):
        self.comment("beta", "kept wording")
        self.comment("epsilon", "rewrite this")
        self.journal.record(self.store.comments(), self.store.version())
        self.rebuild(["alpha beta gamma", "delta zeta"])
        (round_,) = ReviewHistory(self.journal, []).rounds(self.store.comments(), self.pdf)
        states = {item["text"]: item["status"] for item in round_["items"]}
        self.assertEqual(states, {"kept wording": "found", "rewrite this": "changed"})

    def test_current_comments_are_not_history(self):
        self.comment("beta", "still here")
        current = self.store.comments()
        self.journal.record(current, self.store.version())
        self.assertEqual(ReviewHistory(self.journal, []).rounds(current, self.pdf), [])

    def test_edit_keeps_one_journal_entry(self):
        added = self.comment("beta", "first")
        self.journal.record(self.store.comments(), self.store.version())
        self.store.edit(1, added["xref"], "second")
        self.journal.record(self.store.comments(), self.store.version())
        self.assertEqual([entry["text"] for entry in self.journal.entries().values()], ["second"])

    def test_deleted_comment_is_forgotten(self):
        added = self.comment("beta", "mistake")
        self.journal.record(self.store.comments(), self.store.version())
        self.journal.forget(self.store.remove(1, added["xref"]))
        self.assertEqual(self.journal.entries(), {})

    def test_archived_pdf_adds_a_round_without_repeating_the_journal(self):
        self.comment("beta", "same note")
        archive = self.folder / "paper-01.pdf"
        archive.write_bytes(self.pdf.read_bytes())
        self.journal.record(self.store.comments(), self.store.version())
        self.rebuild(["alpha beta gamma"])
        rounds = ReviewHistory(self.journal, [archive]).rounds(self.store.comments(), self.pdf)
        self.assertEqual(sum(len(round_["items"]) for round_ in rounds), 1)


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class ServerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.pdf = write_pdf(Path(folder.name) / "paper.pdf", ["alpha beta gamma"])
        self.store = CommentStore(self.pdf, "ana")
        history = ReviewHistory(CommentJournal(Path(folder.name) / "journal.json"), [])
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store, history, b"<html></html>"))
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.port = server.server_address[1]

    def request(self, method, path, body=None, kind="application/json", host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port)
        self.addCleanup(connection.close)
        headers = {"Host": host or f"127.0.0.1:{self.port}", "Content-Type": kind}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")

    def new_comment(self, version: str) -> dict:
        with pymupdf.open(self.pdf) as document:
            rects = [fractions_of(document[0], "beta")]
        return {"page": 1, "rects": rects, "text": "note", "version": version}

    def current_version(self) -> str:
        return self.request("GET", "/api/version")[1]["version"]

    def test_version_is_a_string_and_a_change_with_it_is_saved(self):
        version = self.current_version()
        self.assertIsInstance(version, str)
        status, data = self.request("POST", "/api/comments", self.new_comment(version))
        self.assertEqual((status, data["comment"]["marked"]), (200, "beta"))

    def test_other_host_is_refused(self):
        status, _ = self.request("GET", "/pdf", host=f"rebind.example:{self.port}")
        self.assertEqual(status, 403)

    def test_change_that_is_not_json_is_refused(self):
        status, _ = self.request("POST", "/api/comments", self.new_comment(self.current_version()), kind="text/plain")
        self.assertEqual(status, 415)
        self.assertEqual(self.store.comments(), [])

    def test_change_on_an_older_version_is_a_conflict(self):
        old = self.current_version()
        self.request("POST", "/api/comments", self.new_comment(old))
        status, _ = self.request("POST", "/api/comments", self.new_comment(old))
        self.assertEqual((status, len(self.store.comments())), (409, 1))

    def test_change_without_a_version_is_a_bad_request(self):
        body = self.new_comment("")
        del body["version"]
        self.assertEqual(self.request("POST", "/api/comments", body)[0], 400)

    def test_malformed_comment_path_is_not_found(self):
        status, data = self.request("PUT", "/api/comments/x/y", {"text": "", "version": "1"})
        self.assertEqual((status, data["error"]), (404, "not found"))


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class ReviewCopyTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.source = Path(folder.name) / "paper.pdf"
        self.copy = Path(folder.name) / "paper-comments.pdf"
        self.source.write_bytes(b"first build")

    def test_first_run_copies_the_source(self):
        self.assertTrue(review_copy(self.source, self.copy))
        self.assertEqual(self.copy.read_bytes(), b"first build")

    def test_existing_copy_survives_a_rebuild(self):
        review_copy(self.source, self.copy)
        self.copy.write_bytes(b"commented")
        self.source.write_bytes(b"second build")
        self.assertFalse(review_copy(self.source, self.copy))
        self.assertEqual(self.copy.read_bytes(), b"commented")

    def test_copy_cannot_be_the_source(self):
        with self.assertRaises(ValueError):
            review_copy(self.source, self.source)


if __name__ == "__main__":
    unittest.main()
