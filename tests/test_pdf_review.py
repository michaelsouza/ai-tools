import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf
except ImportError:  # run with ``uv run --with pymupdf``
    pymupdf = None

if pymupdf is not None:
    from tools.pdf_review import CommentStore, review_copy


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
