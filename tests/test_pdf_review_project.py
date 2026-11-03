import argparse
import contextlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import pymupdf
except ImportError:  # run with ``uv run --with pymupdf``
    pymupdf = None

if pymupdf is not None:
    from tools import pdf_review, pdf_review_project
    from tools.pdf_review_project import (
        Builder,
        BuildError,
        Location,
        ProjectError,
        SourceLocator,
        by_words,
        changed_files,
        comments,
        load_project,
        pending,
        snapshot,
        source_lines,
    )


def write(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def touch_later(path: Path) -> None:
    """Change the mtime as an edit would, without waiting for the clock."""
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))


class Folder(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()


MULTI_TEXT = """\
pdf = "{target}/source/main.pdf"
sources = ["{target}/source/**/*.tex"]
review_dir = "{target}/review"
read = "{target}/review/lidos.json"

[build]
command = "make -s pdf T={target}/source"
log = "{target}/source/build.log"
"""


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class ConfigTests(Folder):
    def test_config_is_found_from_a_subfolder_and_paths_are_relative_to_it(self):
        write(self.root / "pdf-review.toml", 'pdf = "out/paper.pdf"\n[build]\ncommand = ["make"]\n')
        project = load_project(write(self.root / "deep/er/x.txt").parent)
        self.assertEqual((project.pdf, project.cwd), (self.root / "out/paper.pdf", self.root))

    def test_defaults_go_to_the_review_dir(self):
        write(self.root / "pdf-review.toml", 'pdf = "p.pdf"\nreview_dir = "rev"\n[build]\ncommand = ["make"]\n')
        project = load_project(self.root)
        self.assertEqual(
            (project.journal, project.read, project.log, project.mode, project.watch),
            (
                self.root / "rev/comments.json",
                self.root / "rev/read.json",
                self.root / "rev/build.log",
                "once",
                project.sources,
            ),
        )

    def test_target_fills_the_placeholders(self):
        write(self.root / "pdf-review.toml", MULTI_TEXT)
        text = self.root / "students/ana/thesis"
        (text / "source").mkdir(parents=True)
        project = load_project(self.root, "students/ana/thesis/")
        self.assertEqual(project.pdf, text / "source/main.pdf")
        self.assertEqual(project.journal, text / "review/comments.json")
        self.assertEqual(project.read, text / "review/lidos.json")
        self.assertEqual(project.command, ["make", "-s", "pdf", "T=students/ana/thesis/source"])
        self.assertEqual(project.watch, ["students/ana/thesis/source/**/*.tex"])

    def test_target_is_relative_to_the_current_folder(self):
        write(self.root / "pdf-review.toml", MULTI_TEXT)
        text = self.root / "students/ana/thesis"
        (text / "source").mkdir(parents=True)
        self.assertEqual(load_project(text / "source", "..").pdf, text / "source/main.pdf")

    def test_target_with_a_space_stays_one_argument(self):
        write(self.root / "pdf-review.toml", MULTI_TEXT)
        (self.root / "my text").mkdir()
        self.assertEqual(load_project(self.root, "my text").command[-1], "T=my text/source")

    def test_target_is_required_by_a_template(self):
        write(self.root / "pdf-review.toml", MULTI_TEXT)
        with self.assertRaisesRegex(ProjectError, "name the document"):
            load_project(self.root)

    def test_target_is_refused_without_a_template(self):
        write(self.root / "pdf-review.toml", 'pdf = "p.pdf"\n[build]\ncommand = ["make"]\n')
        (self.root / "x").mkdir()
        with self.assertRaisesRegex(ProjectError, "takes no TARGET"):
            load_project(self.root, "x")

    def test_missing_target_and_target_outside_are_refused(self):
        write(self.root / "repo/pdf-review.toml", MULTI_TEXT)
        (self.root / "elsewhere").mkdir()
        with self.assertRaisesRegex(ProjectError, "not found"):
            load_project(self.root / "repo", "nothing")
        with self.assertRaisesRegex(ProjectError, "outside"):
            load_project(self.root / "repo", "../elsewhere")

    def test_unknown_key_and_wrong_type_are_named(self):
        write(self.root / "pdf-review.toml", 'pdf = "p.pdf"\n[build]\ncommand = ["make"]\nwatcher = []\n')
        with self.assertRaisesRegex(ProjectError, "unknown key build.watcher"):
            load_project(self.root)
        write(self.root / "pdf-review.toml", 'pdf = "p.pdf"\n[build]\ncommand = ["make"]\nmode = "always"\n')
        with self.assertRaisesRegex(ProjectError, "build.mode must be"):
            load_project(self.root)

    def test_command_needs_the_pdf(self):
        write(self.root / "pdf-review.toml", '[build]\ncommand = ["make"]\n')
        with self.assertRaisesRegex(ProjectError, "which PDF"):
            load_project(self.root)

    def test_config_without_command_keeps_the_convention_build(self):
        write(self.root / "paper.qmd")
        write(self.root / "pdf-review.toml", 'journal = "_review/paper.comments.json"\n')
        project = load_project(self.root)
        self.assertEqual((project.mode, project.journal), ("watch", self.root / "_review/paper.comments.json"))

    def test_target_of_a_template_must_be_a_folder(self):
        write(self.root / "pdf-review.toml", MULTI_TEXT)
        write(self.root / "texts/a/source/main.tex")
        with self.assertRaisesRegex(ProjectError, "TARGET must be a folder"):
            load_project(self.root / "texts/a", "source/main.tex")


TEMPLATE_WITHOUT_COMMAND = 'pdf = "{target}/source/main.pdf"\nreview_dir = "{target}/review"\n'


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class ConfigWithoutCommandTests(Folder):
    """A config that only moves files: the convention builds the document of TARGET or of the pdf."""

    def test_template_builds_the_document_of_its_target(self):
        write(self.root / "pdf-review.toml", TEMPLATE_WITHOUT_COMMAND)
        write(self.root / "texts/a/source/main.tex", "\\documentclass{article}\n")
        write(self.root / "texts/a/source/Makefile")
        source = self.root / "texts/a/source"
        for start, target in ((self.root, "texts/a"), (source, "../")):
            project = load_project(start, target)
            self.assertEqual((project.command, project.mode, project.cwd), (["make"], "once", source))
            self.assertEqual(project.pdf, source / "main.pdf")
            self.assertEqual(project.journal, self.root / "texts/a/review/comments.json")
            self.assertEqual(project.sources, ["texts/a/source/**/*.tex"])
            self.assertIn("texts/a/source/**/*.bib", project.watch)

    def test_document_is_found_above_the_pdf_folder(self):
        write(self.root / "paper/paper.qmd")
        write(self.root / "pdf-review.toml", 'pdf = "paper/_output/paper.pdf"\nreview_dir = "paper/_review"\n')
        project = load_project(self.root)
        self.assertEqual((project.command[2], project.cwd, project.mode), ("paper.qmd", self.root / "paper", "watch"))
        self.assertEqual((project.pdf, project.sources), (self.root / "paper/_output/paper.pdf", ["paper/paper.qmd"]))

    def test_without_a_document_serve_fails_but_comments_work(self):
        write(self.root / "pdf-review.toml", TEMPLATE_WITHOUT_COMMAND)
        (self.root / "texts/a/source").mkdir(parents=True)
        with self.assertRaisesRegex(ProjectError, "set build.command in"):
            load_project(self.root, "texts/a")
        project = load_project(self.root, "texts/a", build=False)
        self.assertEqual((project.command, project.read), ([], self.root / "texts/a/review/read.json"))


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class ConventionTests(Folder):
    def test_single_qmd_is_a_quarto_preview(self):
        write(self.root / "paper.qmd")
        write(self.root / "_quarto.yml", "project:\n  type: default\n  output-dir: _output\n")
        project = load_project(self.root)
        self.assertEqual(project.pdf, self.root / "_output/paper.pdf")
        self.assertEqual(
            (project.command[:3], project.mode, project.ready),
            (["quarto", "preview", "paper.qmd"], "watch", "Watching files for changes"),
        )
        self.assertEqual(project.source_files(), [self.root / "paper.qmd"])

    def test_main_tex_with_a_makefile_is_made_again_on_changes(self):
        write(self.root / "thesis/main.tex", "\\documentclass{book}\n")
        write(self.root / "thesis/chapter.tex", "text\n")
        write(self.root / "thesis/Makefile")
        project = load_project(self.root, "thesis")
        self.assertEqual(
            (project.pdf, project.command, project.mode), (self.root / "thesis/main.pdf", ["make"], "once")
        )
        self.assertEqual(project.cwd, self.root / "thesis")
        self.assertIn("**/*.bib", project.watch)

    def test_tex_without_makefile_asks_for_a_config(self):
        write(self.root / "main.tex", "\\documentclass{article}\n")
        with self.assertRaisesRegex(ProjectError, "no Makefile"):
            load_project(self.root)

    def test_two_documents_are_ambiguous(self):
        write(self.root / "a.qmd")
        write(self.root / "b.qmd")
        with self.assertRaisesRegex(ProjectError, "a.qmd, b.qmd"):
            load_project(self.root)

    def test_named_file_settles_it(self):
        write(self.root / "a.qmd")
        write(self.root / "b.qmd")
        self.assertEqual(load_project(self.root, "b.qmd").pdf, self.root / "b.pdf")


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class WordLocatorTests(Folder):
    def lines(self) -> list:
        write(
            self.root / "a.tex",
            "% the order of the edges matters here\nintro line\nthe order of the edges matters here\n",
        )
        write(self.root / "b.tex", "Distances between atoms\nare known only for close pairs of atoms.\n")
        return source_lines([self.root / "a.tex", self.root / "b.tex"])

    def test_exact_phrase_skips_tex_comments(self):
        self.assertEqual(by_words("order of the edges", self.lines()), Location(self.root / "a.tex", 3, "exact text"))

    def test_overlap_finds_a_passage_across_lines(self):
        found = by_words("known only for close pairs", self.lines())
        self.assertEqual((found.path.name, found.line), ("b.tex", 2))

    def test_overlap_with_line_break_is_approximate(self):
        found = by_words("between atoms are known", self.lines())
        self.assertEqual((found.path.name, found.how), ("b.tex", "approximate"))

    def test_unknown_passage_is_not_found(self):
        self.assertIsNone(by_words("nothing like this anywhere", self.lines()))


HAS_LATEX = shutil.which("pdflatex") is not None and shutil.which("synctex") is not None


@unittest.skipIf(pymupdf is None or not HAS_LATEX, "requires pymupdf, pdflatex and synctex")
class SynctexLocatorTests(Folder):
    MAIN = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "First paragraph about the same repeated phrase\n"
        "that continues on a second source line\n"
        "and a third one here.\n"
        "\n"
        "\\input{chapter}\n"
        "\\end{document}\n"
    )
    CHAPTER = (
        "% a comment line\n"
        "The chapter starts with a sentence about\n"
        "graphs and distances, then another\n"
        "sentence about the same repeated phrase.\n"
    )

    def setUp(self):
        super().setUp()
        write(self.root / "main.tex", self.MAIN)
        write(self.root / "chapter.tex", self.CHAPTER)
        write(self.root / "pdf-review.toml", 'pdf = "main.pdf"\n[build]\ncommand = ["make"]\n')
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-synctex=1", "main.tex"],
            cwd=self.root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
        )
        self.project = load_project(self.root)

    def locate(self, entry: dict) -> Location:
        with SourceLocator(self.project) as locator:
            self.assertTrue(locator.uses_synctex)
            return locator.locate(entry)

    def rects_of(self, word: str, occurrence: int) -> list[list[float]]:
        with pymupdf.open(self.project.pdf) as document:
            page = document[0]
            rect = pymupdf.Rect([w for w in page.get_text("words") if w[4] == word][occurrence][:4])
            return [
                [
                    rect.x0 / page.rect.width,
                    rect.y0 / page.rect.height,
                    rect.x1 / page.rect.width,
                    rect.y1 / page.rect.height,
                ]
            ]

    def test_passage_in_an_input_file_is_found_by_synctex(self):
        found = self.locate({"page": 1, "marked": "graphs and distances"})
        self.assertEqual((found.path, found.line, found.how), (self.root / "chapter.tex", 3, "synctex"))

    def test_line_is_refined_by_the_words(self):
        found = self.locate({"page": 1, "marked": "and a third one here"})
        self.assertEqual((found.path.name, found.line), ("main.tex", 5))

    def test_repeated_passage_uses_the_recorded_place(self):
        entry = {"page": 1, "marked": "same repeated phrase", "rects": self.rects_of("same", 1)}
        found = self.locate(entry)
        self.assertEqual((found.path.name, found.line, found.how), ("chapter.tex", 4, "synctex, recorded position"))

    def test_repeated_passage_of_an_old_entry_falls_back_to_the_words(self):
        found = self.locate({"page": 1, "marked": "same repeated phrase"})
        self.assertEqual(found.how, "exact text")

    def test_synctex_naming_a_file_without_the_words_gives_way_to_them(self):
        # SyncTeX tags the last paragraph of an \\input file with the next file: here it names main.tex:2
        with mock.patch.object(pdf_review_project, "synctex_edit", return_value=(self.root / "main.tex", 2)):
            found = self.locate({"page": 1, "marked": "graphs and distances"})
        self.assertEqual(found, Location(self.root / "chapter.tex", 3, "exact text"))

    def test_synctex_line_without_the_words_anywhere_is_unconfirmed(self):
        entry = {"page": 1, "marked": "a passage rewritten since", "rects": self.rects_of("graphs", 0)}
        found = self.locate(entry)
        self.assertEqual(
            (found.path.name, found.line, found.how), ("chapter.tex", 2, "synctex, recorded position, unconfirmed")
        )


def project_for(root: Path, command: str, mode: str = "once", ready: str | None = None) -> "Project":
    lines = [
        'pdf = "out.pdf"',
        'sources = ["*.tex"]',
        "[build]",
        f"command = {json.dumps(['sh', '-c', command])}",
        f'mode = "{mode}"',
    ]
    if ready:
        lines.append(f'ready = "{ready}"')
    write(root / "pdf-review.toml", "\n".join(lines) + "\n")
    return load_project(root)


def wait_for(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class BuilderTests(Folder):
    def setUp(self):
        super().setUp()
        self.source = write(self.root / "a.tex", "text")
        self.output = io.StringIO()
        for redirect in (contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output)):
            self.enterContext(redirect)

    def runs(self) -> int:
        return len(_read_lines(self.root / "runs.txt"))

    def start(self, builder: "Builder") -> "Builder":
        self.addCleanup(builder.stop)
        builder.start()
        return builder

    def test_snapshot_sees_edits_but_not_the_files_the_review_writes(self):
        pdf = write(self.root / "out.pdf")
        write(self.root / "out.pdf.partial")
        before = snapshot(self.root, ["*"], exclude={pdf})
        self.assertEqual(list(before), [self.source])
        touch_later(self.source)
        touch_later(pdf)
        new = write(self.root / "b.tex")
        self.assertEqual(changed_files(before, snapshot(self.root, ["*"], exclude={pdf})), [self.source, new])

    def test_edit_rebuilds(self):
        project = project_for(self.root, "echo run >> runs.txt; touch out.pdf")
        self.start(Builder(project, interval=0.01))
        self.assertEqual(self.runs(), 1)
        touch_later(self.source)
        self.assertTrue(wait_for(lambda: self.runs() == 2))

    def test_comments_written_into_the_pdf_do_not_rebuild(self):
        project = project_for(self.root, "echo run >> runs.txt; touch out.pdf")
        self.start(Builder(project, interval=0.01))
        touch_later(project.pdf)
        time.sleep(0.1)
        self.assertEqual(self.runs(), 1)

    def test_failed_build_keeps_watching(self):
        write(self.root / "out.pdf")
        project = project_for(self.root, "echo run >> runs.txt; exit 1")
        self.start(Builder(project, interval=0.01))
        touch_later(self.source)
        self.assertTrue(wait_for(lambda: self.runs() == 2))
        self.assertTrue(wait_for(lambda: self.output.getvalue().count("build failed") == 2))

    def test_first_build_without_a_pdf_is_an_error(self):
        project = project_for(self.root, "true")
        with self.assertRaisesRegex(BuildError, "made no .*out.pdf"):
            Builder(project).start()

    def test_watcher_is_ready_on_its_pattern_and_stops_with_its_children(self):
        project = project_for(
            self.root, "touch out.pdf; echo ready; sleep 60 & echo $! > child.pid; wait", "watch", "^ready"
        )
        builder = self.start(Builder(project))
        self.assertIsNone(builder.process.poll())
        self.assertTrue(wait_for(lambda: (self.root / "child.pid").read_text().strip() != ""))
        child = int((self.root / "child.pid").read_text())
        builder.stop()
        self.assertIsNotNone(builder.process.poll())
        self.assertTrue(wait_for(lambda: not _alive(child)))

    def test_missing_command_is_an_error(self):
        write(self.root / "pdf-review.toml", 'pdf = "out.pdf"\n[build]\ncommand = ["no-such-command-here"]\n')
        with self.assertRaisesRegex(BuildError, "cannot run no-such-command-here"):
            Builder(load_project(self.root)).start()

    def test_watcher_that_dies_is_an_error(self):
        project = project_for(self.root, "echo oops; exit 3", "watch", "never")
        with self.assertRaisesRegex(BuildError, r"exit 3"):
            Builder(project).start()


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text().splitlines()
    except FileNotFoundError:
        return []


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class CommentsTests(Folder):
    def setUp(self):
        super().setUp()
        write(self.root / "paper.qmd", "Intro.\n\nThe order of the edges matters here.\n")
        write(self.root / "pdf-review.toml", 'journal = "_review/j.json"\nread = "_review/lidos.json"\n')
        self.project = load_project(self.root)
        entries = {
            "u2": {"page": 2, "marked": "order of the edges", "text": "why?", "author": "ana", "seen": "2026-10-02"},
            "u1": {"page": 1, "marked": "Intro", "text": "line one\nline two", "author": "ana", "seen": "2026-10-01"},
        }
        write(self.project.journal, json.dumps(entries))

    def run_comments(self, **options) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            comments(self.project, argparse.Namespace(**({"all": False, "mark_read": False} | options)))
        return output.getvalue()

    def test_listing_is_oldest_first_with_the_source_line(self):
        text = self.run_comments()
        self.assertLess(text.index("p. 1"), text.index("p. 2"))
        self.assertIn(f"{self.root / 'paper.qmd'}:3 (exact text)", text)
        self.assertIn("line one\n     line two", text)

    def test_marked_comments_are_not_listed_again(self):
        self.run_comments(mark_read=True)
        self.assertEqual(json.loads(self.project.read.read_text()), ["u1", "u2"])
        self.assertEqual(self.run_comments(), "No new comments.\n")
        self.assertEqual(len(pending(self.project, show_all=True)), 2)

    def test_existing_read_list_is_kept(self):
        write(self.project.read, json.dumps(["older", "u1"]))
        self.assertEqual([uid for uid, _ in pending(self.project)], ["u2"])
        self.run_comments(mark_read=True)
        self.assertEqual(json.loads(self.project.read.read_text()), ["older", "u1", "u2"])


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@unittest.skipIf(pymupdf is None, "requires pymupdf")
class CommandLineTests(Folder):
    def setUp(self):
        super().setUp()
        self.enterContext(contextlib.chdir(self.root))
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))

    def test_plain_pdf_keeps_its_old_form(self):
        pdf = self.root / "t.pdf"
        with pymupdf.open() as document:
            document.new_page().insert_text((72, 100), "alpha")
            document.save(pdf)
        with mock.patch.object(pdf_review, "review") as review:
            pdf_review.main(["t.pdf", "--no-browser", "--port", "8799"])
            pdf_review.main(["t.pdf", "--copy", "--author", "ana"])
        (plain, plain_journal, plain_args), (copied, copied_journal, copied_args) = [
            call.args for call in review.call_args_list
        ]
        self.assertEqual((plain, plain_journal), (Path("t.pdf"), Path("t.pdf.comments.json")))
        self.assertEqual((plain_args.port, plain_args.no_browser), (8799, True))
        self.assertEqual((copied, copied_journal), (Path("t-comments.pdf"), Path("t-comments.pdf.comments.json")))
        self.assertEqual((copied_args.author, copied_args.port), ("ana", 8765))
        self.assertTrue(copied.is_file())

    def test_subcommand_goes_to_the_project(self):
        write(self.root / "pdf-review.toml", TEMPLATE_WITHOUT_COMMAND)
        (self.root / "texts/a/source").mkdir(parents=True)
        entry = {"page": 1, "marked": "x", "text": "why?", "author": "ana", "seen": "2026-10-01"}
        write(self.root / "texts/a/review/comments.json", json.dumps({"u1": entry}))
        pdf_review.main(["comments", "texts/a", "--mark-read"])  # needs no build command
        self.assertIn("Comment: why?", self.output.getvalue())
        self.assertEqual(json.loads((self.root / "texts/a/review/read.json").read_text()), ["u1"])

    def serve(self, port: int) -> None:
        """Run ``serve`` with the build and the page replaced; the page gets a SIGTERM."""
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # serve sets its own handlers
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        write(self.root / "pdf-review.toml", 'pdf = "out.pdf"\n[build]\ncommand = ["make"]\n')
        self.builder = self.enterContext(mock.patch.object(pdf_review_project, "Builder"))
        self.review = self.enterContext(mock.patch.object(pdf_review, "review"))
        self.review.side_effect = lambda *_: (os.kill(os.getpid(), signal.SIGTERM), time.sleep(5))
        pdf_review_project.serve(load_project(self.root), argparse.Namespace(port=port))

    def test_busy_port_stops_before_building(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with self.assertRaisesRegex(SystemExit, "is in use"):
                self.serve(busy.getsockname()[1])
        self.builder.assert_not_called()
        self.review.assert_not_called()

    def test_sigterm_ends_the_build(self):
        started = time.monotonic()
        self.serve(free_port())
        self.assertLess(time.monotonic() - started, 4)
        self.review.assert_called_once()
        self.assertEqual([call[0] for call in self.builder.return_value.method_calls], ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
