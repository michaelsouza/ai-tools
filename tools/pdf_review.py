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
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pymupdf

PAGE_HTML = Path(__file__).with_name("pdf_review.html")


def _words_inside(words: list[tuple], rect: pymupdf.Rect) -> list[str]:
    """Words whose centre lies inside ``rect``, in reading order."""
    return [word[4] for word in words if rect.contains(pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2))]


class CommentStore:
    """Highlight annotations of one PDF file, read and written on each call.

    The file is reopened every time because a build may replace it while the
    review is open.  Rectangles cross the API as fractions of the page as
    displayed, which is what the browser measures.
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
                    found.append(self._describe(page, annot, words))
        return sorted(found, key=lambda c: (c["page"], c["rects"][0][1], c["rects"][0][0]))

    def add(self, page_number: int, rects: list[list[float]], text: str) -> dict:
        if not rects:
            raise ValueError("a comment needs at least one marked rectangle")
        with self._lock, pymupdf.open(self.pdf) as document:
            page = document[page_number - 1]
            displayed = [self._to_page(page, rect) for rect in rects]
            annot = page.add_highlight_annot(quads=[rect.quad for rect in displayed])
            annot.set_info(content=text, title=self.author)
            annot.update()
            comment = self._describe(page, annot, page.get_text("words"))
            self._save(document)
        return comment

    def edit(self, page_number: int, xref: int, text: str) -> None:
        with self._lock, pymupdf.open(self.pdf) as document:
            page = document[page_number - 1]
            annot = self._find(page, xref)
            annot.set_info(content=text)
            annot.update()
            self._save(document)

    def remove(self, page_number: int, xref: int) -> None:
        with self._lock, pymupdf.open(self.pdf) as document:
            page = document[page_number - 1]
            page.delete_annot(self._find(page, xref))
            self._save(document)

    @staticmethod
    def _find(page: pymupdf.Page, xref: int) -> pymupdf.Annot:
        """The annotation stays usable only while ``page`` is referenced."""
        if xref not in {found[0] for found in page.annot_xrefs()}:
            raise KeyError(f"no annotation {xref} on page {page.number + 1}")
        return page.load_annot(xref)

    @staticmethod
    def _to_page(page: pymupdf.Page, fractions: list[float]) -> pymupdf.Rect:
        """Fractions of the displayed page to unrotated page coordinates."""
        width, height = page.rect.width, page.rect.height
        x0, y0, x1, y1 = fractions
        shown = pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
        return shown * page.derotation_matrix

    @staticmethod
    def _to_fractions(page: pymupdf.Page, rect: pymupdf.Rect) -> list[float]:
        shown = rect * page.rotation_matrix
        width, height = page.rect.width, page.rect.height
        return [shown.x0 / width, shown.y0 / height, shown.x1 / width, shown.y1 / height]

    def _describe(self, page: pymupdf.Page, annot: pymupdf.Annot, words: list[tuple]) -> dict:
        vertices = annot.vertices or []
        quads = [pymupdf.Quad(vertices[i : i + 4]).rect for i in range(0, len(vertices), 4)]
        rects = quads or [annot.rect]
        return {
            "page": page.number + 1,
            "xref": annot.xref,
            "rects": [self._to_fractions(page, rect) for rect in rects],
            "marked": " ".join(word for rect in rects for word in _words_inside(words, rect)),
            "text": annot.info.get("content", ""),
            "author": annot.info.get("title", ""),
        }

    def _save(self, document: pymupdf.Document) -> None:
        """Append the change to the file, or rewrite it when appending fails."""
        if document.can_save_incrementally():
            document.saveIncr()
            return
        partial = self.pdf.with_name(self.pdf.name + ".partial")
        document.save(partial, garbage=1)
        os.replace(partial, self.pdf)


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


def make_handler(store: CommentStore, page_html: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, page_html, "text/html; charset=utf-8")
            elif self.path.startswith("/pdf"):
                self._send(HTTPStatus.OK, store.pdf.read_bytes(), "application/pdf")
            elif self.path == "/api/comments":
                self._json({"name": store.pdf.name, "version": store.version(), "comments": store.comments()})
            elif self.path == "/api/version":
                self._json({"version": store.version()})
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/comments":
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            self._apply(lambda body: store.add(int(body["page"]), body["rects"], str(body["text"])))

        def do_PUT(self) -> None:
            page, xref = self._target()
            self._apply(lambda body: store.edit(page, xref, str(body["text"])))

        def do_DELETE(self) -> None:
            page, xref = self._target()
            self._apply(lambda body: store.remove(page, xref))

        def _target(self) -> tuple[int, int]:
            """``/api/comments/<page>/<xref>`` as numbers."""
            *_, page, xref = self.path.rstrip("/").split("/")
            return int(page), int(xref)

        def _apply(self, change) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            try:
                result = change(body)
            except (KeyError, ValueError, IndexError) as problem:
                return self._json({"error": str(problem)}, HTTPStatus.BAD_REQUEST)
            except Exception as problem:  # the page shows it instead of a dropped connection
                return self._json({"error": f"{type(problem).__name__}: {problem}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            self._json({"version": store.version(), "comment": result})

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
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--author", default=getpass.getuser(), help="name stored with each comment")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    pdf = args.pdf
    if args.copy:
        pdf = Path(f"{args.pdf.stem}-comments.pdf") if args.copy is True else Path(args.copy)
        try:
            made = review_copy(args.pdf, pdf)
        except ValueError as problem:
            parser.error(str(problem))
        print(f"{'Copied' if made else 'Reusing the existing copy'}: {pdf}")

    store = CommentStore(pdf.resolve(), args.author)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, PAGE_HTML.read_bytes()))
    url = f"http://localhost:{args.port}/"
    print(f"Reviewing {pdf} at {url} (Ctrl+C to stop)")
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
