#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
Merge two BibTeX files using bibtexparser.

Default behavior:
    - Keep existing entries from the second file first.
    - Add non-duplicate entries from the first file.
    - Treat matching citation keys, DOIs, or normalized titles as duplicates.
    - If the same key has different content, rename the first-file entry before adding it.

Usage:
    tools/merge_bib.py base.bib extra.bib
    tools/merge_bib.py base.bib extra.bib -o merged.bib
    tools/merge_bib.py base.bib extra.bib --dry-run
    tools/merge_bib.py base.bib extra.bib --on-key-conflict keep-first
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from rich.console import Console
from rich.table import Table

error_console = Console(stderr=True)


@dataclass
class MergeResult:
    entries: List[Dict[str, str]]
    added_entries: List[Dict[str, str]]
    added: int
    skipped_duplicates: int
    renamed_conflicts: int
    replaced_conflicts: int
    errors: List[str]
    warnings: List[str]


def load_bib(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = False
    parser.interpolate_strings = False

    try:
        database = bibtexparser.loads(path.read_text(encoding="utf-8"), parser=parser)
    except Exception as exc:
        return [], [f"{path}: {exc}"]

    return database.entries, []


def normalize_doi(value: str) -> str:
    value = value.strip().casefold()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.strip()


def normalize_title(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+\s*", " ", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"[^0-9A-Za-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def canonical_entry(entry: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key.casefold(), str(value).strip()) for key, value in entry.items()))


def entry_key(entry: Dict[str, str]) -> str:
    return entry.get("ID", "")


def dedupe_tokens(entry: Dict[str, str], modes: Sequence[str]) -> Iterable[Tuple[str, str]]:
    key = entry_key(entry)
    if "key" in modes and key:
        yield ("key", key.casefold())
    if "doi" in modes:
        doi = normalize_doi(entry.get("doi", ""))
        if doi:
            yield ("doi", doi)
    if "title" in modes:
        title = normalize_title(entry.get("title", ""))
        if title:
            yield ("title", title)


def unique_key(base_key: str, used_keys: Dict[str, Dict[str, str]], suffix: str) -> str:
    candidate = f"{base_key}{suffix}"
    number = 2
    while candidate.casefold() in used_keys:
        candidate = f"{base_key}{suffix}{number}"
        number += 1
    return candidate


def merge_entries(
    first_entries: List[Dict[str, str]],
    second_entries: List[Dict[str, str]],
    dedupe_modes: Sequence[str],
    on_key_conflict: str,
    rename_suffix: str,
) -> MergeResult:
    merged: List[Dict[str, str]] = []
    added_entries: List[Dict[str, str]] = []
    token_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    key_index: Dict[str, Dict[str, str]] = {}
    added = 0
    skipped_duplicates = 0
    renamed_conflicts = 0
    replaced_conflicts = 0
    errors: List[str] = []
    warnings: List[str] = []

    def index_entry(entry: Dict[str, str]) -> None:
        merged.append(entry)
        key = entry_key(entry)
        if key:
            key_index[key.casefold()] = entry
        for token in dedupe_tokens(entry, dedupe_modes):
            token_index[token] = entry

    for entry in first_entries:
        key = entry_key(entry)
        if not key:
            warnings.append("Skipped an entry from the first file without citation key.")
            continue
        if key.casefold() in key_index:
            skipped_duplicates += 1
            warnings.append(f"Skipped duplicate key in first file: {key}")
            continue
        index_entry(entry)

    for entry in second_entries:
        key = entry_key(entry)
        if not key:
            warnings.append("Skipped an entry from the second file without citation key.")
            continue

        duplicate = next(
            (token_index[token] for token in dedupe_tokens(entry, dedupe_modes) if token in token_index),
            None,
        )
        key_conflict = key_index.get(key.casefold())

        if duplicate and not key_conflict:
            skipped_duplicates += 1
            continue

        if key_conflict and canonical_entry(key_conflict) == canonical_entry(entry):
            skipped_duplicates += 1
            continue

        if key_conflict:
            if on_key_conflict == "keep-first":
                skipped_duplicates += 1
                continue
            if on_key_conflict == "keep-second":
                position = merged.index(key_conflict)
                merged[position] = entry
                key_index[key.casefold()] = entry
                for token in dedupe_tokens(entry, dedupe_modes):
                    token_index[token] = entry
                replaced_conflicts += 1
                continue
            if on_key_conflict == "error":
                errors.append(f"key conflict: {key} appears in both files with different content")
                continue

            new_key = unique_key(key, key_index, rename_suffix)
            entry = dict(entry)
            entry["ID"] = new_key
            renamed_conflicts += 1

        elif duplicate:
            skipped_duplicates += 1
            continue

        index_entry(entry)
        added_entries.append(entry)
        added += 1

    return MergeResult(
        merged, added_entries, added, skipped_duplicates, renamed_conflicts, replaced_conflicts, errors, warnings
    )


def bibtex_text(entries: Sequence[Dict[str, str]]) -> str:
    database = BibDatabase()
    database.entries = list(entries)

    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = None
    writer.display_order = ["title", "author", "journal", "booktitle", "year", "doi", "url"]
    return writer.write(database)


def write_bib(entries: Sequence[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bibtex_text(entries), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two BibTeX files.")
    parser.add_argument("first", type=Path, help="Source .bib file. Missing entries are copied from here.")
    parser.add_argument(
        "second",
        type=Path,
        help="Target .bib file. Its existing entries are kept first and it is overwritten unless -o is set.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output .bib path. Defaults to the second .bib file.")
    parser.add_argument(
        "--dedupe-by",
        default="key,doi,title",
        help="Comma-separated duplicate checks: key,doi,title. Default: key,doi,title.",
    )
    parser.add_argument(
        "--on-key-conflict",
        choices=["rename", "keep-first", "keep-second", "error"],
        default="rename",
        help="What to do when both files use the same key for different entries. Default: rename.",
    )
    parser.add_argument("--rename-suffix", default="_2", help="Suffix used with --on-key-conflict rename.")
    parser.add_argument("--dry-run", action="store_true", help="Print a summary without writing output.")
    parser.add_argument("--quiet", action="store_true", help="Only print errors.")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> List[str]:
    errors = []
    for path in (args.first, args.second):
        if not path.exists():
            errors.append(f"{path} does not exist")
        elif not path.is_file():
            errors.append(f"{path} is not a file")

    modes = [mode.strip().casefold() for mode in args.dedupe_by.split(",") if mode.strip()]
    invalid_modes = sorted(set(modes) - {"key", "doi", "title"})
    if invalid_modes:
        errors.append(f"invalid --dedupe-by values: {', '.join(invalid_modes)}")
    if not modes:
        errors.append("--dedupe-by must include at least one mode")
    args.dedupe_modes = modes
    return errors


def print_summary(result: MergeResult, first_count: int, second_count: int, dry_run: bool) -> None:
    table = Table(title="BibTeX Merge Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Source file entries", str(first_count))
    table.add_row("Target file entries", str(second_count))
    table.add_row("Merged entries", str(len(result.entries)))
    table.add_row("Added to target", str(result.added))
    table.add_row("Skipped duplicates", str(result.skipped_duplicates))
    table.add_row("Renamed key conflicts", str(result.renamed_conflicts))
    table.add_row("Replaced key conflicts", str(result.replaced_conflicts))
    if dry_run:
        table.add_row("Dry run", "yes")
    Console(file=sys.stdout).print(table)


def print_added_entries(entries: Sequence[Dict[str, str]]) -> None:
    if not entries:
        return
    sys.stdout.write(bibtex_text(entries).rstrip() + "\n\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validation_errors = validate_args(args)
    if validation_errors:
        for error in validation_errors:
            error_console.print(f"[red]Error:[/] {error}")
        return 2

    first_entries, first_errors = load_bib(args.first)
    second_entries, second_errors = load_bib(args.second)
    output_path = args.output or args.second

    result = merge_entries(
        second_entries,
        first_entries,
        args.dedupe_modes,
        args.on_key_conflict,
        args.rename_suffix,
    )
    result.errors.extend(first_errors)
    result.errors.extend(second_errors)

    if not args.quiet:
        print_added_entries(result.added_entries)
        print_summary(result, len(first_entries), len(second_entries), args.dry_run)
        for warning in result.warnings:
            error_console.print(f"[yellow]Warning:[/] {warning}")

    for error in result.errors:
        error_console.print(f"[red]Error:[/] {error}")

    if result.errors:
        return 1

    if args.dry_run:
        return 0

    write_bib(result.entries, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
