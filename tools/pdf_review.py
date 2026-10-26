#!/usr/bin/env python3
"""Comment on a PDF in the browser: select a passage, type a note, done.

The page shows the PDF on the left and, on the right, the comment box and the
list of comments.  Releasing a text selection opens the box.  Each comment is
saved into the PDF itself as a highlight annotation whose note holds the
comment, so Okular, Acrobat and scripts that read annotations see it too.

    uv run --with pymupdf python tools/pdf_review.py paper.pdf

The PDF is written in place, so a program that regenerates it (a LaTeX or
Quarto build) erases the comments.  ``--copy`` avoids that by commenting on a
copy, which a later run reuses:

    uv run --with pymupdf python tools/pdf_review.py _output/paper.pdf --copy

Past comments stay visible below the current ones.  A journal next to the PDF
records every comment the page has shown, and ``--history`` adds the comments
of older annotated PDFs.  Each past comment is looked up in the current PDF by
its marked words: found once, it is highlighted in another colour; not found,
the passage was rewritten.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import webbrowser
from contextlib import contextmanager
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pymupdf

PAGE_HTML = Path(__file__).with_name("pdf_review.html")
UID_PREFIX = "pdf-review:"  # kept in the annotation's subject, so a comment keeps its identity when edited
NOTE_MARGIN = (-20, -5, 20, 5)  # words this close to a note's icon are what the note refers to
WORD_GAP = 0.02  # fragments of one line lie closer than this (page widths); a column gutter is wider


# ---------- Words and annotations ----------


def _words_inside(words: list[tuple], rect: pymupdf.Rect) -> list[tuple]:
    """Words whose centre lies inside ``rect``, in reading order."""
    return [word for word in words if rect.contains(pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2))]


def _same_line(word: tuple, other: tuple) -> bool:
    return word[5:7] == other[5:7]  # block and line numbers


def _joined(words: list[tuple]) -> str:
    """Join the words, undoing the hyphenation at the end of a PDF line."""
    text = ""
    for word, following in zip(words, words[1:] + [None]):
        text += word[4]
        if following is None:
            break
        if word[4].endswith("-") and not _same_line(word, following):
            text = text[:-1]
        else:
            text += " "
    return text


def _tokens(text: str) -> list[str]:
    """Lower-case words, without punctuation, for comparing passages across builds."""
    return re.findall(r"\w+", text.lower())


def _to_fractions(page: pymupdf.Page, rect: pymupdf.Rect) -> list[float]:
    """Unrotated page coordinates to fractions of the page as displayed."""
    shown = rect * page.rotation_matrix
    width, height = page.rect.width, page.rect.height
    return [shown.x0 / width, shown.y0 / height, shown.x1 / width, shown.y1 / height]


def _to_page(page: pymupdf.Page, fractions: list[float]) -> pymupdf.Rect:
    """Fractions of the page as displayed to unrotated page coordinates."""
    width, height = page.rect.width, page.rect.height
    x0, y0, x1, y1 = fractions
    return pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height) * page.derotation_matrix


def describe(page: pymupdf.Page, annot: pymupdf.Annot, words: list[tuple]) -> dict:
    """An annotation as the page shows it: where it is, the words it marks and its note."""
    vertices = annot.vertices or []
    quads = [pymupdf.Quad(vertices[i : i + 4]).rect for i in range(0, len(vertices), 4)]
    if quads:
        rects, marked = quads, _joined([word for quad in quads for word in _words_inside(words, quad)])
    else:
        rects, marked = [annot.rect], _joined(_words_inside(words, annot.rect + NOTE_MARGIN))
    info = annot.info
    text = info.get("content", "")
    subject = info.get("subject", "")
    if subject.startswith(UID_PREFIX):
        uid = subject
    else:  # made by another program: its content is the best identity there is
        uid = "h:" + hashlib.sha1(f"{page.number}|{marked}|{text}".encode()).hexdigest()[:16]
    return {
        "uid": uid,
        "page": page.number + 1,
        "xref": annot.xref,
        "rects": [_to_fractions(page, rect) for rect in rects],
        "marked": marked,
        "text": text,
        "author": info.get("title", ""),
    }


def _reading_order(comment: dict) -> tuple:
    return comment["page"], comment["rects"][0][1], comment["rects"][0][0]


def merge_lines(rects: list[list[float]]) -> list[list[float]]:
    """One rectangle per line of a column: the browser returns one per text fragment.

    Fragments merge when they share most of their height and almost touch, so
    the same line of two columns stays two rectangles.
    """
    lines: list[list[float]] = []
    for rect in sorted(rects, key=lambda r: (r[1], r[0])):
        for line in lines:
            overlap = min(line[3], rect[3]) - max(line[1], rect[1])
            gap = max(line[0], rect[0]) - min(line[2], rect[2])
            if overlap > 0.5 * min(line[3] - line[1], rect[3] - rect[1]) and gap < WORD_GAP:
                line[:] = [min(line[0], rect[0]), min(line[1], rect[1]), max(line[2], rect[2]), max(line[3], rect[3])]
                break
        else:
            lines.append(list(rect))
    return lines


# ---------- The PDF under review ----------


class StaleVersion(Exception):
    """The PDF changed since the page read it, so its pages and annotation xrefs may be different."""


class CommentStore:
    """Highlight annotations of one PDF file, read and written on each call.

    The file is reopened every time because a build may replace it while the
    review is open.  Rectangles cross the API as fractions of the page as
    displayed, which is what the browser measures.  A change can name the
    version it was made against (``expected``); it is refused when the file
    has changed since.
    """

    def __init__(self, pdf: Path, author: str) -> None:
        self.pdf = pdf
        self.author = author
        self._lock = threading.Lock()

    def version(self) -> int:
        return self.pdf.stat().st_mtime_ns

    def comments(self) -> list[dict]:
        with self._lock, pymupdf.open(self.pdf) as document:
            found = []
            for page in document:
                words = page.get_text("words")
                for annot in page.annots(types=[pymupdf.PDF_ANNOT_HIGHLIGHT]):
                    found.append(describe(page, annot, words))
        return sorted(found, key=_reading_order)

    @contextmanager
    def _changing(self, expected: int | None):
        """The open document, for a change made against version ``expected``."""
        with self._lock:
            if expected is not None and expected != self.version():
                raise StaleVersion("the PDF changed on disk since the page loaded it")
            with pymupdf.open(self.pdf) as document:
                yield document

    def add(self, page_number: int, rects: list[list[float]], text: str, expected: int | None = None) -> dict:
        if not rects:
            raise ValueError("a comment needs at least one marked rectangle")
        with self._changing(expected) as document:
            page = document[page_number - 1]
            displayed = [_to_page(page, rect) for rect in merge_lines(rects)]
            annot = page.add_highlight_annot(quads=[rect.quad for rect in displayed])
            annot.set_info(content=text, title=self.author, subject=UID_PREFIX + uuid.uuid4().hex)
            annot.update()
            comment = describe(page, annot, page.get_text("words"))
            self._save(document)
        return comment

    def edit(self, page_number: int, xref: int, text: str, expected: int | None = None) -> None:
        with self._changing(expected) as document:
            page = document[page_number - 1]
            annot = self._find(page, xref)
            annot.set_info(content=text)
            annot.update()
            self._save(document)

    def remove(self, page_number: int, xref: int, expected: int | None = None) -> str:
        """Delete the comment and return its uid."""
        with self._changing(expected) as document:
            page = document[page_number - 1]
            annot = self._find(page, xref)
            uid = describe(page, annot, page.get_text("words"))["uid"]
            page.delete_annot(annot)
            self._save(document)
        return uid

    @staticmethod
    def _find(page: pymupdf.Page, xref: int) -> pymupdf.Annot:
        """The annotation stays usable only while ``page`` is referenced."""
        if xref not in {found[0] for found in page.annot_xrefs()}:
            raise KeyError(f"no annotation {xref} on page {page.number + 1}")
        return page.load_annot(xref)

    def _save(self, document: pymupdf.Document) -> None:
        """Append the change to the file, or rewrite it when appending fails."""
        if document.can_save_incrementally():
            document.saveIncr()
            return
        partial = self.pdf.with_name(self.pdf.name + ".partial")
        document.save(partial, garbage=1)
        os.replace(partial, self.pdf)


# ---------- Past comments ----------


class CommentJournal:
    """Every comment the review has shown, kept after a rebuild erases it from the PDF.

    Entries are keyed by uid and remember the last version of the PDF that
    held them, which groups the past comments into review rounds.
    """

    FIELDS = ("page", "marked", "text", "author")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def entries(self) -> dict[str, dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def record(self, comments: list[dict], version: int) -> None:
        with self._lock:
            entries = self.entries()
            changed = False
            for comment in comments:
                entry = {field: comment[field] for field in self.FIELDS}
                old = entries.get(comment["uid"], {})
                if {field: old.get(field) for field in self.FIELDS} != entry or old.get("version") != version:
                    entries[comment["uid"]] = entry | {
                        "version": version,
                        "seen": datetime.now().isoformat("T", "minutes"),
                    }
                    changed = True
            if changed:
                self._write(entries)

    def forget(self, uid: str) -> None:
        """A deleted comment was not meant to be kept."""
        with self._lock:
            entries = self.entries()
            if entries.pop(uid, None) is not None:
                self._write(entries)

    def _write(self, entries: dict) -> None:
        partial = self.path.with_name(self.path.name + ".partial")
        partial.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(partial, self.path)


def archived_comments(pdf: Path) -> list[dict]:
    """The annotations of an older PDF that say something, read without changing it."""
    with pymupdf.open(pdf) as document:
        found = []
        for page in document:
            words = page.get_text("words")
            for annot in page.annots():
                comment = describe(page, annot, words)
                if comment["text"] or comment["marked"]:
                    found.append(comment)
    return sorted(found, key=_reading_order)


class PassageIndex:
    """The words of a PDF as one stream of tokens, to find an old passage in a new build.

    Each token remembers its page and the rectangles of its word, as
    fractions, with the line they belong to.  A word hyphenated across two
    lines becomes one token with both rectangles.
    """

    def __init__(self, document: pymupdf.Document) -> None:
        self.tokens: list[str] = []
        self.places: list[tuple[int, list[tuple[tuple, list[float]]]]] = []
        for page in document:
            words = page.get_text("words")
            i = 0
            while i < len(words):
                text, parts = words[i][4], [words[i]]
                while text.endswith("-") and i + 1 < len(words) and not _same_line(words[i], words[i + 1]):
                    i += 1
                    text, parts = text[:-1] + words[i][4], parts + [words[i]]
                place = (page.number + 1, [(w[5:7], _to_fractions(page, pymupdf.Rect(w[:4]))) for w in parts])
                for token in _tokens(text):
                    self.tokens.append(token)
                    self.places.append(place)
                i += 1

    def find(self, marked: str) -> dict:
        """``found`` with its page and line rectangles, ``repeated``, ``changed`` or ``no-text``."""
        wanted = _tokens(marked)
        if not wanted:
            return {"status": "no-text"}
        size = len(wanted)
        starts = [
            i
            for i, token in enumerate(self.tokens[: len(self.tokens) - size + 1])
            if token == wanted[0] and self.tokens[i : i + size] == wanted
        ]
        if not starts:
            return {"status": "changed"}
        if len(starts) > 1:
            return {"status": "repeated", "count": len(starts)}
        page = self.places[starts[0]][0]
        lines: dict[tuple, list[float]] = {}
        for place_page, parts in self.places[starts[0] : starts[0] + size]:
            if place_page != page:  # a passage running into the next page is marked on its first page
                continue
            for line, (x0, y0, x1, y1) in parts:
                box = lines.setdefault(line, [x0, y0, x1, y1])
                box[:] = [min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
        return {"status": "found", "page": page, "rects": list(lines.values())}


def _passage_key(comment: dict) -> tuple[str, str]:
    return " ".join(_tokens(comment["marked"])), comment["text"].strip()


class ReviewHistory:
    """Past comments in rounds, newest first, each looked up in the current PDF."""

    def __init__(self, journal: CommentJournal, archives: list[Path]) -> None:
        self.journal = journal
        self.archives = archives

    def rounds(self, current: list[dict], pdf: Path) -> list[dict]:
        uids = {comment["uid"] for comment in current}
        seen = {_passage_key(comment) for comment in current}
        rounds = []
        past = [(uid, entry) for uid, entry in self.journal.entries().items() if uid not in uids]
        for version in sorted({entry["version"] for _, entry in past}, reverse=True):
            items = [dict(entry, uid=uid) for uid, entry in past if entry["version"] == version]
            rounds.append({"label": f"until {items[0]['seen'].replace('T', ' ')}", "items": items})
        for archive in sorted(self.archives, key=lambda path: path.name, reverse=True):
            rounds.append({"label": archive.name, "items": archived_comments(archive)})

        with pymupdf.open(pdf) as document:
            index = PassageIndex(document)
        shown = []
        for number, round_ in enumerate(rounds):
            items = []
            for item in sorted(round_["items"], key=lambda c: (c["page"], c.get("rects", [[0, 0]])[0][1])):
                key = _passage_key(item)
                if key in seen:  # a comment appears once, in its latest round
                    continue
                seen.add(key)
                items.append(
                    {
                        "key": f"h{number}-{len(items)}",
                        "page_then": item["page"],
                        "marked": item["marked"],
                        "text": item["text"],
                        "author": item["author"],
                    }
                    | index.find(item["marked"])
                )
            if items:
                shown.append({"label": round_["label"], "items": items})
        return shown


def review_copy(source: Path, target: Path) -> bool:
    """Copy ``source`` to ``target`` unless the copy exists; tell whether it was made.

    An existing copy holds the comments of an earlier session, so it is kept
    even when the source was rebuilt since.
    """
    if target.resolve() == source.resolve():
        raise ValueError("the copy must not be the PDF itself")
    if target.exists():
        return False
    shutil.copy2(source, target)
    return True


# ---------- HTTP ----------


COMMENT_PATH = re.compile(r"/api/comments/(\d+)/(\d+)/?")


def make_handler(store: CommentStore, history: ReviewHistory, page_html: bytes) -> type[BaseHTTPRequestHandler]:
    """Requests are answered only for the page itself.

    Another site open in the browser may send requests here.  A Host other
    than this server's (DNS rebinding) is refused, and changes must be JSON,
    which a browser sends cross-origin only after a CORS check this server
    never passes.  Versions travel as strings: a nanosecond mtime does not
    fit in a JavaScript number.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._own_host():
                return
            if self.path == "/":
                self._send(HTTPStatus.OK, page_html, "text/html; charset=utf-8")
            elif self.path.startswith("/pdf"):
                self._send(HTTPStatus.OK, store.pdf.read_bytes(), "application/pdf")
            elif self.path == "/api/comments":
                version = store.version()
                current = store.comments()
                history.journal.record(current, version)
                self._json(
                    {
                        "name": store.pdf.name,
                        "version": str(version),
                        "comments": current,
                        "history": history.rounds(current, store.pdf),
                    }
                )
            elif self.path == "/api/version":
                self._json({"version": str(store.version())})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:
            if self.path != "/api/comments":
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            self._apply(
                lambda body, expected: store.add(int(body["page"]), body["rects"], str(body["text"]), expected)
            )

        def do_PUT(self) -> None:
            if target := self._target():
                self._apply(lambda body, expected: store.edit(*target, str(body["text"]), expected))

        def do_DELETE(self) -> None:
            if target := self._target():
                self._apply(lambda body, expected: history.journal.forget(store.remove(*target, expected)))

        def _own_host(self) -> bool:
            port = self.server.server_address[1]
            if self.headers.get("Host") in {f"localhost:{port}", f"127.0.0.1:{port}"}:
                return True
            self._error(HTTPStatus.FORBIDDEN, "unknown Host")
            return False

        def _target(self) -> tuple[int, int] | None:
            """``/api/comments/<page>/<xref>`` as numbers, or None after answering 404."""
            if match := COMMENT_PATH.fullmatch(self.path):
                return int(match[1]), int(match[2])
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return None

        def _apply(self, change) -> None:
            """Run ``change(body, expected_version)`` and answer with the new version."""
            if not self._own_host():
                return
            if self.headers.get_content_type() != "application/json":
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "changes must be sent as application/json")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                result = change(body, int(body["version"]))
            except StaleVersion as problem:
                return self._error(HTTPStatus.CONFLICT, str(problem))
            except (KeyError, ValueError, IndexError, TypeError) as problem:  # JSONDecodeError is a ValueError
                return self._error(HTTPStatus.BAD_REQUEST, f"{type(problem).__name__}: {problem}")
            except Exception as problem:  # the page shows it instead of a dropped connection
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(problem).__name__}: {problem}")
            self._json({"version": str(store.version()), "comment": result})

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json({"error": message}, status)

        def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path, help="PDF to comment on; it is written in place unless --copy")
    parser.add_argument(
        "--copy",
        nargs="?",
        const=True,
        metavar="PATH",
        help="comment on a copy, reused if it exists (default: ./<name>-comments.pdf)",
    )
    parser.add_argument(
        "--history", type=Path, nargs="+", default=[], metavar="PDF", help="older annotated PDFs whose comments to show"
    )
    parser.add_argument("--journal", type=Path, help="file recording past comments (default: <pdf>.comments.json)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--author", default=getpass.getuser(), help="name stored with each comment")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    missing = [str(path) for path in args.history if not path.is_file()]
    if missing:
        parser.error(f"history PDF not found: {', '.join(missing)}")

    pdf = args.pdf
    if args.copy:
        pdf = Path(f"{args.pdf.stem}-comments.pdf") if args.copy is True else Path(args.copy)
        try:
            made = review_copy(args.pdf, pdf)
        except ValueError as problem:
            parser.error(str(problem))
        print(f"{'Copied' if made else 'Reusing the existing copy'}: {pdf}")

    store = CommentStore(pdf.resolve(), args.author)
    journal = CommentJournal(args.journal or pdf.with_name(pdf.name + ".comments.json"))
    archives = [path.resolve() for path in args.history if path.resolve() != store.pdf]
    handler = make_handler(store, ReviewHistory(journal, archives), PAGE_HTML.read_bytes())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/"
    print(f"Reviewing {pdf} at {url} (Ctrl+C to stop)")
    print(f"Past comments are recorded in {journal.path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
