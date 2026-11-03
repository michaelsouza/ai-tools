"""Review a document built from sources: keep its PDF current and find each comment in the sources.

    tools/pdf-review serve [TARGET] [--port N] [--no-browser] [--author NAME]
    tools/pdf-review comments [TARGET] [--all] [--mark-read]

``serve`` builds the PDF, rebuilds it when a source changes and opens the
review page of ``pdf_review.py`` on it, with a journal that keeps the
comments a rebuild erases from the PDF.  ``comments`` reads that journal and
lists the comments not yet handled, each with the file and line of the
passage it marks; ``--mark-read`` remembers them as handled.

The project is described by ``pdf-review.toml``, looked up from the current
directory upwards.  Without one the document is recognised by its files: a
single ``.qmd`` is built by ``quarto preview``, a main ``.tex`` next to a
``Makefile`` by ``make``.  A config may hold ``{target}``, filled from the
TARGET argument, so one file serves every text of a repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if __package__:  # imported as tools.pdf_review_project
    from . import pdf_review
else:  # run from tools/, through pdf_review.py
    import pdf_review

CONFIG_NAME = "pdf-review.toml"
TARGET = "{target}"
MODES = ("watch", "once")
POLL_SECONDS = 1.0  # how often a "once" build checks its watched files (no inotify on every system)
READY_SECONDS = 600  # how long a "watch" build may take to make the first PDF
STOP_SECONDS = 5  # how long a build may take to end after SIGTERM, before SIGKILL
QUARTO_READY = "Watching files for changes"
LATEX_WATCH = ["**/*.tex", "**/*.bib", "**/*.sty", "**/*.cls", "**/*.png", "**/*.jpg", "**/*.pdf"]
DEFAULT_SOURCES = ["**/*.tex", "**/*.qmd"]
MAIN_TEX = re.compile(r"^\s*\\documentclass", re.MULTILINE)
TEX_SUFFIXES = {".tex", ".sty", ".cls"}  # whole-line % comments are skipped when locating a passage

MIN_WORD_LENGTH = 4
MIN_OVERLAP = 0.6
MIN_PHRASE_WORDS = 3
SYNCTEX_WINDOW = 10  # lines around SyncTeX's answer where the marked words may refine it


# ---------- The project ----------


class ProjectError(Exception):
    """The project cannot be worked out; the message says what to fix."""


def _is_text(value) -> bool:
    return isinstance(value, str)


def _is_texts(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


SETTINGS = {
    "pdf": ("a path", _is_text),
    "sources": ("a list of globs", _is_texts),
    "review_dir": ("a path", _is_text),
    "journal": ("a path", _is_text),
    "read": ("a path", _is_text),
    "build": ("a table", lambda value: isinstance(value, dict)),
}
BUILD_SETTINGS = {
    "command": ("a string or a list of strings", lambda value: _is_text(value) or (_is_texts(value) and value)),
    "mode": ('"watch" or "once"', lambda value: value in MODES),
    "ready": ("a regular expression", _is_text),
    "cwd": ("a path", _is_text),
    "watch": ("a list of globs", _is_texts),
    "log": ("a path", _is_text),
}


@dataclass
class Project:
    """One document: its PDF, how to build it, its sources and its review files, as absolute paths."""

    origin: str  # the config file, or the file recognised by convention
    root: Path  # globs are relative to it
    pdf: Path
    command: list[str]
    mode: str
    cwd: Path
    ready: str | None
    watch: list[str]
    sources: list[str]
    journal: Path
    read: Path
    log: Path

    def source_files(self) -> list[Path]:
        return matching(self.root, self.sources)

    def watched(self) -> dict[Path, tuple[int, int]]:
        """The state of the watched files, without the files the build and the review page write."""
        return snapshot(self.root, self.watch, exclude={self.pdf, self.log, self.journal, self.read})


def matching(root: Path, patterns: list[str]) -> list[Path]:
    """Files matching any of the globs (``**`` included), relative to ``root``."""
    found = set()
    for pattern in patterns:
        found.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(found)


def snapshot(root: Path, patterns: list[str], exclude: set[Path] = frozenset()) -> dict[Path, tuple[int, int]]:
    """Modification time and size of each matching file, leaving out ``exclude`` and ``*.partial`` files."""
    state = {}
    for path in matching(root, patterns):
        if path in exclude or path.name.endswith(".partial"):  # pdf_review.py writes through .partial files
            continue
        try:
            info = path.stat()
        except FileNotFoundError:  # removed while scanning
            continue
        state[path] = (info.st_mtime_ns, info.st_size)
    return state


def changed_files(before: dict[Path, tuple], after: dict[Path, tuple]) -> list[Path]:
    """Files added, removed or modified between two snapshots."""
    return sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))


def find_config(start: Path) -> Path | None:
    """``pdf-review.toml`` in ``start`` or the nearest folder above it."""
    for folder in [start, *start.parents]:
        if (folder / CONFIG_NAME).is_file():
            return folder / CONFIG_NAME
    return None


def load_project(start: Path, target: str | None = None, build: bool = True) -> Project:
    """The project seen from folder ``start``; ``target`` is the TARGET argument, relative to ``start``.

    With ``build`` false (listing comments) the project needs no build
    command, only its PDF, sources and review files.
    """
    start = start.resolve()
    config = find_config(start)
    if config is None:
        document = (start / target).resolve() if target else start
        if not document.exists():
            raise ProjectError(f"not found: {target}")
        root = document if document.is_dir() else document.parent
        settings = convention(document)
        return make_project(root, settings, f"{settings['main']} (by convention, no {CONFIG_NAME})", build)

    try:
        settings = tomllib.loads(config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as problem:
        raise ProjectError(f"{config}: {problem}") from None
    check_settings(settings, config)
    root = config.parent
    target = relative_target(root, start, target)
    settings = fill_target(settings, target, config)
    if "command" not in settings.get("build", {}):  # the config only moves files: the convention builds
        try:
            settings = with_convention(root, settings, target, config)
        except ProjectError:
            if build or "pdf" not in settings:
                raise
    return make_project(root, settings, str(config), build)


def check_settings(settings: dict, config: Path) -> None:
    """Refuse unknown keys and values of the wrong type, naming them; split a command given as a string."""
    for table, known, prefix in ((settings, SETTINGS, ""), (settings.get("build", {}), BUILD_SETTINGS, "build.")):
        for key, value in table.items():
            if key not in known:
                raise ProjectError(f"{config}: unknown key {prefix}{key} (known: {', '.join(known)})")
            kind, valid = known[key]
            if not valid(value):
                raise ProjectError(f"{config}: {prefix}{key} must be {kind}")
    build = settings.get("build", {})
    if isinstance(build.get("command"), str):  # split before {target} is filled, so a path with spaces stays whole
        build["command"] = shlex.split(build["command"])


def relative_target(root: Path, start: Path, target: str | None) -> str | None:
    """TARGET as a path relative to the config's folder, whatever folder it was given from."""
    if target is None:
        return None
    path = (start / target).resolve()
    if not path.exists():
        raise ProjectError(f"TARGET not found: {target}")
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        raise ProjectError(f"TARGET {target} is outside {root}, the folder of {CONFIG_NAME}") from None


def fill_target(settings: dict, target: str | None, config: Path) -> dict:
    """Replace ``{target}`` in every string; a TARGET is required exactly when the config uses it."""
    uses = TARGET in json.dumps(settings)
    if uses and target is None:
        raise ProjectError(f"{config} uses {TARGET}: name the document, e.g. pdf-review serve <folder>")
    if target is not None and not uses:
        raise ProjectError(f"{config} has no {TARGET}, so it takes no TARGET (got {target})")
    if target is not None and not (config.parent / target).is_dir():
        raise ProjectError(f"TARGET must be a folder, the one {TARGET} stands for in {config} (got {target})")

    def filled(value):
        if isinstance(value, str):
            return value.replace(TARGET, target)
        if isinstance(value, list):
            return [filled(item) for item in value]
        if isinstance(value, dict):
            return {key: filled(item) for key, item in value.items()}
        return value

    return filled(settings) if uses else settings


def with_convention(root: Path, settings: dict, target: str | None, config: Path) -> dict:
    """``settings`` completed by the convention, for a config without a build command.

    The document is the one in the nearest folder holding one, from the
    folder of ``pdf`` (else TARGET, else the config's) up to the config's
    folder; the build runs there, and the convention's paths are rebased on
    it.  The config's own settings win.
    """
    folder = Path(os.path.normpath(root / (Path(settings["pdf"]).parent if "pdf" in settings else target or ".")))
    if folder != root and root not in folder.parents:
        folder = root
    while folder != root and not documents(folder):
        folder = folder.parent
    detected = convention(folder, fix=f"set build.command in {config}")
    place = folder.relative_to(root).as_posix()

    def rebased(path: str) -> str:
        return path if place == "." else f"{place}/{path}"

    build = detected["build"] | {"cwd": place}
    if "watch" in build:
        build["watch"] = [rebased(glob) for glob in build["watch"]]
    detected |= {"pdf": rebased(detected["pdf"]), "sources": [rebased(glob) for glob in detected["sources"]]}
    return detected | settings | {"build": build | settings.get("build", {})}


def documents(folder: Path) -> list[Path]:
    """The documents of a folder: its ``.qmd`` files or, when there is none, its main ``.tex`` files."""
    return sorted(folder.glob("*.qmd")) or [
        tex for tex in sorted(folder.glob("*.tex")) if MAIN_TEX.search(tex.read_text(errors="replace"))
    ]


def convention(path: Path, fix: str = f"write a {CONFIG_NAME}") -> dict:
    """Settings for a document recognised by its files, relative to its folder.

    ``path`` is a ``.qmd`` or ``.tex`` file, or a folder holding exactly one
    ``.qmd`` or, when there is none, exactly one main ``.tex`` (one with
    ``\\documentclass``).  A ``.qmd`` is built by ``quarto preview``, which
    keeps rebuilding by itself; a ``.tex`` needs a ``Makefile`` beside it and
    is built by ``make`` whenever a TeX source or figure changes.  ``fix``
    ends the error messages.
    """
    if path.is_dir():
        found = documents(path)
        if len(found) != 1:
            names = ", ".join(file.name for file in found) or "no .qmd and no .tex with \\documentclass"
            raise ProjectError(f"cannot tell which document to build in {path} ({names}): {fix}")
        path = found[0]
    folder = path.parent
    if path.suffix == ".qmd":
        output = re.search(r"^\s*output-dir:\s*[\"']?([^\"'\s#]+)", _read(folder / "_quarto.yml"), re.MULTILINE)
        return {
            "main": path.name,
            "pdf": f"{output[1]}/{path.stem}.pdf" if output else f"{path.stem}.pdf",
            "sources": [path.name],
            "build": {
                "command": ["quarto", "preview", path.name, "--to", "pdf", "--no-serve", "--no-browser"],
                "mode": "watch",
                "ready": QUARTO_READY,
            },
        }
    if path.suffix == ".tex":
        if not (folder / "Makefile").is_file():
            raise ProjectError(f"no Makefile next to {path}: {fix}")
        return {
            "main": path.name,
            "pdf": f"{path.stem}.pdf",
            "sources": ["**/*.tex"],
            "build": {"command": ["make"], "mode": "once", "watch": LATEX_WATCH},
        }
    raise ProjectError(f"{path} is neither a .qmd nor a .tex file: {fix}")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def make_project(root: Path, settings: dict, origin: str, build_needed: bool = True) -> Project:
    """The project from checked settings whose paths are relative to ``root``; a build command when needed."""
    build = settings.get("build", {})
    if "pdf" not in settings:
        raise ProjectError(f"{origin}: say which PDF the build makes (pdf = ...)")
    if build_needed and not build.get("command"):
        raise ProjectError(f"{origin}: no build command ([build] command = ...)")
    mode = build.get("mode", "once")
    ready = build.get("ready")
    if ready is not None:
        try:
            re.compile(ready)
        except re.error as problem:
            raise ProjectError(f"{origin}: build.ready is not a regular expression: {problem}") from None

    def under(path: str) -> Path:
        return Path(os.path.normpath(root / path))

    review_dir = settings.get("review_dir", "_review")
    sources = settings.get("sources", DEFAULT_SOURCES)
    return Project(
        origin=origin,
        root=root,
        pdf=under(settings["pdf"]),
        command=build.get("command", []),
        mode=mode,
        cwd=under(build.get("cwd", ".")),
        ready=ready,
        watch=build.get("watch", sources) if mode == "once" else [],
        sources=sources,
        journal=under(settings.get("journal", f"{review_dir}/comments.json")),
        read=under(settings.get("read", f"{review_dir}/read.json")),
        log=under(build.get("log", f"{review_dir}/build.log")),
    )


# ---------- Building ----------


class BuildError(Exception):
    """There is no PDF to show; the message says where to look."""


def shown(path: Path) -> str:
    """``path`` relative to the current directory when it lies below it."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def say(message: str, problem: bool = False) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr if problem else sys.stdout, flush=True)


def end_group(process: subprocess.Popen) -> None:
    """Terminate the process group led by ``process``, killing it if it lingers.

    The group is signalled only while its leader runs, so the number cannot
    belong to another group yet.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(STOP_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


class Builder:
    """Keeps the PDF of a project current, from a background thread.

    A ``watch`` build is one long-running command that rebuilds by itself
    (``quarto preview``).  A ``once`` build is run again whenever a watched
    file changes, which is checked every ``interval`` seconds.  Each command
    runs in its own process group, without input, with its output in the
    log; ``stop`` ends the whole group.
    """

    def __init__(self, project: Project, interval: float = POLL_SECONDS, ready_seconds: float = READY_SECONDS):
        self.project = project
        self.interval = interval
        self.ready_seconds = ready_seconds
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Make the first PDF, then keep it current; BuildError when there is no PDF to show."""
        project = self.project
        project.log.parent.mkdir(parents=True, exist_ok=True)
        say(f"Building {shown(project.pdf)} with: {shlex.join(project.command)} (log: {shown(project.log)})")
        if project.mode == "watch":
            self._launch()
            self._wait_ready()
            follow = self._follow
        else:
            seen = project.watched()  # before building, so an edit made during the build is seen
            built = self.build()
            if not project.pdf.is_file():
                raise BuildError(f"the build made no {shown(project.pdf)}; see {shown(project.log)}")
            if not built:
                say(f"build failed, showing the previous PDF; see {shown(project.log)}", problem=True)

            def follow():
                self._poll(seen)

        self._thread = threading.Thread(target=follow, daemon=True)
        self._thread.start()

    def build(self) -> bool:
        """Run the build command to its end; True when it succeeded."""
        process = self._launch()
        return process is not None and process.wait() == 0

    def stop(self) -> None:
        """End the running build, if any, and stop rebuilding."""
        with self._lock:
            self._stopping.set()
            process = self.process
        if process is not None:
            end_group(process)
        if self._thread is not None:
            self._thread.join(STOP_SECONDS)

    def _launch(self) -> subprocess.Popen | None:
        """Start the build command, unless the builder is stopping."""
        with self._lock:
            if self._stopping.is_set():
                return None
            with open(self.project.log, "wb") as log:
                try:
                    self.process = subprocess.Popen(
                        self.project.command,
                        cwd=self.project.cwd,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as problem:  # no such command or folder
                    raise BuildError(f"cannot run {shlex.join(self.project.command)}: {problem}") from None
            return self.process

    def _wait_ready(self) -> None:
        """Wait until the watcher has made the PDF: its log shows ``ready``, or, without one, the PDF exists."""
        project = self.project
        deadline = time.monotonic() + self.ready_seconds
        while not self._ready():
            if (code := self.process.poll()) is not None:
                raise BuildError(
                    f"{project.command[0]} ended (exit {code}) before the PDF was made; see {shown(project.log)}"
                )
            if time.monotonic() > deadline:
                raise BuildError(f"no PDF after {self.ready_seconds:.0f} s; see {shown(project.log)}")
            time.sleep(0.2)
        if not project.pdf.is_file():
            raise BuildError(f"the build is ready but made no {shown(project.pdf)}: set pdf in {CONFIG_NAME}")

    def _ready(self) -> bool:
        if self.project.ready is None:
            return self.project.pdf.is_file()
        return re.search(self.project.ready, _read(self.project.log)) is not None

    def _follow(self) -> None:
        """Tell when the watcher ends by itself: the PDF is no longer rebuilt."""
        code = self.process.wait()
        if not self._stopping.is_set():
            log = shown(self.project.log)
            say(f"{self.project.command[0]} ended (exit {code}): the PDF is no longer rebuilt; see {log}", problem=True)

    def _poll(self, seen: dict[Path, tuple]) -> None:
        """Rebuild whenever the watched files differ from ``seen``."""
        while not self._stopping.wait(self.interval):
            now = self.project.watched()
            changed = changed_files(seen, now)
            if not changed:
                continue
            seen = now
            cause = shown(changed[0]) + (f" and {len(changed) - 1} more" if len(changed) > 1 else "")
            try:
                built = self.build()
            except BuildError as problem:
                say(str(problem), problem=True)
                continue
            if self._stopping.is_set():
                return
            if built:
                say(f"PDF rebuilt ({cause} changed)")
            else:
                say(f"build failed ({cause} changed); see {shown(self.project.log)}", problem=True)


def port_in_use(port: int) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the server does
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def serve(project: Project, args: argparse.Namespace) -> None:
    """Build the PDF, keep it current and serve the review page on it until Ctrl+C or SIGTERM."""
    if port_in_use(args.port):
        sys.exit(f"pdf-review: port {args.port} is in use (another review?): pass --port N")
    project.journal.parent.mkdir(parents=True, exist_ok=True)
    print(f"Project: {project.origin}")
    builder = Builder(project)
    for signum in (signal.SIGTERM, signal.SIGHUP):  # end the build too, as Ctrl+C does
        signal.signal(signum, _interrupt)
    try:
        builder.start()
        pdf_review.review(project.pdf, project.journal, args)
    except BuildError as problem:
        sys.exit(f"pdf-review: {problem}")
    except KeyboardInterrupt:
        pass
    finally:
        # A second signal (uv forwards the group's SIGTERM) must not cut the cleanup short.
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, signal.SIG_IGN)
        builder.stop()


# ---------- Finding comments in the sources ----------


@dataclass
class Location:
    path: Path
    line: int
    how: str  # "synctex[, recorded position][, unconfirmed]", "exact text" or "approximate"


Line = tuple[Path, int, str]  # file, 1-based line number, text


def long_words(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) >= MIN_WORD_LENGTH}


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def source_lines(files: list[Path]) -> list[Line]:
    """Every line of ``files``, without the whole-line ``%`` comments of TeX files."""
    lines = []
    for path in files:
        tex = path.suffix in TEX_SUFFIXES
        for number, text in enumerate(_read(path).splitlines(), start=1):
            if not (tex and text.lstrip().startswith("%")):
                lines.append((path, number, text))
    return lines


def by_words(marked: str, lines: list[Line]) -> Location | None:
    """The line of ``lines`` that holds ``marked``, by its words.

    The exact search compares whole words only, so math, markup and LaTeX
    commands do not get in the way.  A short phrase counts only when a
    single line holds it, since a word or two repeat across a text.
    Otherwise the line with the largest fraction of the marked words wins,
    when that fraction reaches ``MIN_OVERLAP`` and the mark has at least
    ``MIN_PHRASE_WORDS`` long words.
    """
    phrase = normalized(marked)
    if phrase:
        holders = [line for line in lines if f" {phrase} " in f" {normalized(line[2])} "]
        if len(holders) == 1 or (holders and len(phrase.split()) >= MIN_PHRASE_WORDS):
            return Location(holders[0][0], holders[0][1], "exact text")
    wanted = long_words(marked)
    if len(wanted) < MIN_PHRASE_WORDS or not lines:
        return None
    scores = [len(wanted & long_words(line[2])) / len(wanted) for line in lines]
    best = max(range(len(lines)), key=scores.__getitem__)
    return Location(lines[best][0], lines[best][1], "approximate") if scores[best] >= MIN_OVERLAP else None


def synctex_edit(pdf: Path, page: int, x: float, y: float) -> tuple[Path, int] | None:
    """Source file and line that SyncTeX gives for a point of the PDF, in points from the top left."""
    try:
        result = subprocess.run(
            ["synctex", "edit", "-o", f"{page}:{x:.2f}:{y:.2f}:{pdf}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    source = line = None
    for text in result.stdout.splitlines():  # the first record: Input:<path>, then Line:<number>
        key, _, value = text.partition(":")
        if key == "Input" and source is None:
            source = value
        elif key == "Line" and source is not None:
            line = int(value) if value.strip().isdigit() else None
            break
    if not source or not line:
        return None
    return Path(os.path.normpath(pdf.parent / source)), line


class SourceLocator:
    """Finds the source line of a comment's passage.

    With a ``<pdf>.synctex.gz`` beside the PDF (``pdflatex -synctex=1``), the
    passage is looked up in the current PDF and SyncTeX names the line under
    its first word; the marked words then refine it within a few lines,
    since SyncTeX may point at the end of a paragraph.  A passage no longer
    in the PDF is looked up where the comment was, when the journal kept that
    place.  Without SyncTeX, when it points outside the sources, or when the
    marked words are not near its line, the words of the passage decide;
    SyncTeX's line is then kept, as unconfirmed, only if they are found nowhere.
    """

    def __init__(self, project: Project) -> None:
        self.pdf = project.pdf
        self.lines = source_lines(project.source_files())
        self.paths = {line[0] for line in self.lines}
        synctex = self.pdf.with_name(self.pdf.stem + ".synctex.gz")
        self.uses_synctex = bool(shutil.which("synctex")) and self.pdf.is_file() and synctex.is_file()
        self._document = None
        self._index = None

    def __enter__(self) -> SourceLocator:
        return self

    def __exit__(self, *exc) -> None:
        if self._document is not None:
            self._document.close()

    def locate(self, entry: dict) -> Location | None:
        found = self._by_synctex(entry) if self.uses_synctex else None
        return found or by_words(entry["marked"], self.lines)

    def _by_synctex(self, entry: dict) -> Location | None:
        place = self._place(entry)
        if place is None:
            return None
        page, fractions, moved = place
        rect = pdf_review._to_page(self._document[page - 1], fractions)
        hit = synctex_edit(self.pdf, page, rect.x0 + 1, (rect.y0 + rect.y1) / 2)
        if hit is None or hit[0] not in self.paths:  # a class or style file: the words do better
            return None
        path, line = hit
        how = "synctex, recorded position" if moved else "synctex"
        nearby = [item for item in self.lines if item[0] == path and abs(item[1] - line) <= SYNCTEX_WINDOW]
        refined = by_words(entry["marked"], nearby)
        if refined:
            return Location(path, refined.line, how)
        # The words are not near SyncTeX's line.  It may name the wrong file: the last paragraph of an \input
        # file is broken into lines once TeX reads the next file, and carries its name.  The words, found
        # anywhere, win; when they are found nowhere, SyncTeX's line is all there is.
        return by_words(entry["marked"], self.lines) or Location(path, line, how + ", unconfirmed")

    def _place(self, entry: dict) -> tuple[int, list[float], bool] | None:
        """Page and first line rectangle of the passage in the current PDF, or where the comment was."""
        if self._index is None:
            self._document = pdf_review.pymupdf.open(self.pdf)
            self._index = pdf_review.PassageIndex(self._document)
        found = self._index.find(entry["marked"], near_page=entry.get("page"))
        if found["status"] == "found":
            return found["page"], found["rects"][0], False
        if entry.get("rects") and 1 <= entry["page"] <= len(self._document):
            return entry["page"], entry["rects"][0], True
        return None


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def pending(project: Project, show_all: bool = False) -> list[tuple[str, dict]]:
    """Journal entries not marked read (all with ``show_all``), oldest first."""
    read = set(load_json(project.read, []))
    entries = load_json(project.journal, {})
    chosen = [(uid, entry) for uid, entry in entries.items() if show_all or uid not in read]
    return sorted(chosen, key=lambda item: (item[1].get("seen", ""), item[1].get("page", 0)))


def describe_comments(entries: list[tuple[str, dict]], locator: SourceLocator) -> str:
    out = []
    for number, (_, entry) in enumerate(entries, start=1):
        where = f"p. {entry.get('page', '?')}"
        found = locator.locate(entry)
        where += ", line not found" if found is None else f", {shown(found.path)}:{found.line} ({found.how})"
        out.append(f"{number}. **{where}** ({entry.get('seen', '?')}, {entry.get('author', '?')})")
        out.append(f"   - Passage: {entry.get('marked') or '—'}")
        out.append(f"   - Comment: {(entry.get('text') or '—').replace(chr(10), chr(10) + '     ')}")
    return "\n".join(out)


def mark_read(project: Project, uids: list[str]) -> None:
    """Add ``uids`` to the read file, a JSON list of comment uids."""
    read = sorted(set(load_json(project.read, [])) | set(uids))
    project.read.parent.mkdir(parents=True, exist_ok=True)
    project.read.write_text(json.dumps(read, indent=1) + "\n", encoding="utf-8")


def comments(project: Project, args: argparse.Namespace) -> None:
    entries = pending(project, args.all)
    if not entries:
        print(f"No comments in {shown(project.journal)}." if args.all else "No new comments.")
        return
    with SourceLocator(project) as locator:
        print(describe_comments(entries, locator))
    print(f"\nTotal: {len(entries)} comment{'' if len(entries) == 1 else 's'}.")
    if args.mark_read:
        mark_read(project, [uid for uid, _ in entries])
        print(f"Marked as read in {shown(project.read)}.")


# ---------- Command line ----------


TARGET_HELP = f"the document, when {CONFIG_NAME} uses {TARGET}; without a config, its folder or main file"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="pdf-review", description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser(
        "serve", help="build the PDF, rebuild it on changes and open the review page", description=serve.__doc__
    )
    serve_parser.add_argument("target", nargs="?", metavar="TARGET", help=TARGET_HELP)
    pdf_review.add_server_options(serve_parser)
    comments_parser = commands.add_parser(
        "comments",
        help="list the comments not yet handled, with their line in the sources",
        description="List the comments not yet handled, with their file and line in the sources.",
    )
    comments_parser.add_argument("target", nargs="?", metavar="TARGET", help=TARGET_HELP)
    comments_parser.add_argument("--all", action="store_true", help="list the comments already handled too")
    comments_parser.add_argument("--mark-read", action="store_true", help="remember the listed comments as handled")
    args = parser.parse_args(argv)
    try:
        project = load_project(Path.cwd(), args.target, build=args.command == "serve")
    except ProjectError as problem:
        sys.exit(f"pdf-review: {problem}")
    (serve if args.command == "serve" else comments)(project, args)


if __name__ == "__main__":
    main(sys.argv[1:])
