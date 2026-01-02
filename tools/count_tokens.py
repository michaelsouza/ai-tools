#!/usr/bin/env python3
"""
Token Counter - Count tokens in text files using tiktoken

This script analyzes text files and directories to count the number of tokens
they contain using OpenAI's tiktoken library. It's useful for estimating API
costs, understanding context window usage, and analyzing document sizes for
LLM applications.

The script processes multiple file formats commonly used with LLMs and provides
detailed statistics including token counts, character counts, and token-to-character
ratios. Results are displayed in a formatted table sorted by token count.

Usage:
    # Count tokens in a single file
    python count_tokens.py document.txt

    # Count tokens in all text files in a directory (recursive)
    python count_tokens.py /path/to/project

    # Count tokens in current directory
    python count_tokens.py .

    # Use a different encoding (e.g., for GPT-3/Codex)
    python count_tokens.py document.txt -e p50k_base

    # Use GPT-2 encoding
    python count_tokens.py document.txt -e r50k_base

Supported Encodings:
    - o200k_base (default)
    - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    - p50k_base: GPT-3 models (davinci, curie, babbage, ada)
    - r50k_base: GPT-2, older GPT-3 models
    - p50k_edit: Older edit models

Supported File Types:
    Text files: .txt, .md, .log, .csv, .tex, .bib
    Code files: .py, .js, .ts, .java, .c, .cpp, .go, .rs, .rb, .php, .sh, .f90
    Config files: .json, .yaml, .yml, .toml, .ini, .cfg
    Markup files: .xml, .html, .css, .tex, .rst, .org, .adoc, .sql
    AMPL files: .mod

Features:
    - Recursive directory scanning
    - Multiple encoding support for different LLM models
    - Rich formatted output with color-coded tables
    - Progress indicator for large file sets
    - Token/character ratio analysis
    - Summary statistics with total counts
    - Files sorted by token count (largest first)

Output:
    - File path (relative to scan directory)
    - Token count per file
    - Character count per file
    - Token-to-character ratio
    - Total summary with aggregate statistics

Requirements:
    - tiktoken: OpenAI's token counting library
    - rich: Terminal formatting and progress display
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import tiktoken
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()

# Supported text file extensions
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".tex",
    ".bib",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".f90",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".log",
    ".tex",
    ".rst",
    ".org",
    ".adoc",
    ".sql",
    ".mod",
}


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        console.print(f"[red]Error encoding text: {e}[/red]")
        return 0


def is_text_file(file_path: Path) -> bool:
    """Check if file is a text file based on extension."""
    return file_path.suffix.lower() in TEXT_EXTENSIONS


def get_files_to_process(path: Path, all_files: bool = False) -> List[Path]:
    """Get list of file to process."""
    files = []

    if path.is_file():
        # Always process a single file, supported or not
        files.append(path)
    elif path.is_dir():
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = Path(root) / filename
                if all_files or is_text_file(file_path):
                    files.append(file_path)
    else:
        console.print(f"[red]Error: {path} does not exist[/red]")

    return files


def process_files(files: List[Path], encoding_name: str, base_path: Path) -> List[Tuple[str, int, int, str]]:
    """Process files and return results."""
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(files))

        for file_path in files:
            # Get relative path for display
            try:
                display_path = str(file_path.relative_to(base_path))
            except ValueError:
                display_path = str(file_path)

            if is_text_file(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        token_count = count_tokens(content, encoding_name)
                        char_count = len(content)
                        results.append((display_path, token_count, char_count, None))
                except Exception as e:
                    console.print(f"[red]Error reading {file_path}: {e}[/red]")
            else:
                try:
                    # Store raw size in char_count field (index 2) for binary files
                    size_bytes = file_path.stat().st_size
                    results.append((display_path, 0, size_bytes, "BINARY"))
                except Exception as e:
                    console.print(f"[red]Error checking size {file_path}: {e}[/red]")

            progress.advance(task)

    return results


def display_results(results: List[Tuple[str, int, int, str]], encoding_name: str):
    """Display results in a rich table."""
    if not results:
        console.print("[yellow]No files processed[/yellow]")
        return

    # Split results
    text_results = [r for r in results if r[3] is None]
    binary_results = [r for r in results if r[3] == "BINARY"]

    total_tokens = 0
    total_chars = 0
    total_files = len(results)
    total_text_files = len(text_results)

    # --- Display Text Files ---
    if text_results:
        # Sort by token count (descending)
        text_results.sort(key=lambda x: x[1], reverse=True)

        table = Table(
            title=f"Text Files (Encoding: {encoding_name})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("File", style="cyan", no_wrap=False)
        table.add_column("Tokens", justify="right", style="green")
        table.add_column("Characters", justify="right", style="blue")
        table.add_column("Ratio", justify="right", style="yellow")

        for file_path, tokens, chars, _ in text_results:
            ratio = f"{tokens/chars:.2f}" if chars > 0 else "N/A"
            table.add_row(file_path, f"{tokens:,}", f"{chars:,}", ratio)
            total_tokens += tokens
            total_chars += chars

        # Add total row for text files
        table.add_section()
        total_ratio = f"{total_tokens/total_chars:.2f}" if total_chars > 0 else "N/A"
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold]{total_chars:,}[/bold]",
            f"[bold]{total_ratio}[/bold]",
        )
        console.print(table)
        console.print()

    # --- Display Binary/Other Files ---
    if binary_results:
        # Sort by size (stored in index 2) descending
        binary_results.sort(key=lambda x: x[2], reverse=True)

        table_bin = Table(
            title="Other Files",
            show_header=True,
            header_style="bold magenta",
        )
        table_bin.add_column("File", style="cyan", no_wrap=False)
        table_bin.add_column("Size", justify="right", style="blue")

        for file_path, _, size_bytes, _ in binary_results:
            size_mb = size_bytes / (1024 * 1024)
            table_bin.add_row(file_path, f"{size_mb:.2f} MB")

        console.print(table_bin)
        console.print()

    # --- Summary ---
    summary = Text()
    summary.append(f"Total Files: ", style="bold")
    summary.append(f"{total_files}\n", style="cyan")

    if text_results:
        summary.append(f"Text Files: ", style="bold")
        summary.append(f"{total_text_files}\n", style="cyan")
        summary.append(f"Total Tokens: ", style="bold")
        summary.append(f"{total_tokens:,}\n", style="green")
        summary.append(f"Total Characters: ", style="bold")
        summary.append(f"{total_chars:,}\n", style="blue")
        if total_chars > 0:
            summary.append("Average Tokens/Char: ", style="bold")
            summary.append(f"{total_tokens/total_chars:.2f}", style="yellow")

    console.print(Panel(summary, title="Summary", border_style="green"))


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in text files using tiktoken",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file.txt                    # Count tokens in a single file
  %(prog)s /path/to/folder             # Count tokens in all text files in folder
  %(prog)s /path/to/folder -a          # Count tokens in text files, show size for other files
  %(prog)s . -e p50k_base              # Use different encoding
        """,
    )
    parser.add_argument("path", type=str, help="File or folder path to process")
    parser.add_argument(
        "-e",
        "--encoding",
        type=str,
        default="o200k_base",
        choices=["o200k_base", "cl100k_base", "p50k_base", "r50k_base", "p50k_edit"],
        help="Tiktoken encoding to use (default: o200k_base for gpt-oss/GPT-5)",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Process all files in directory (show size for non-text files)",
    )

    args = parser.parse_args()

    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Token Counter[/bold cyan]\n[dim]Using tiktoken encoding[/dim]",
            border_style="cyan",
        )
    )

    # Get path
    path = Path(args.path).resolve()

    # Get files to process
    files = get_files_to_process(path, args.all)

    if not files:
        console.print("[yellow]No files found to process[/yellow]")
        return

    console.print(f"[cyan]Found {len(files)} file(s) to process[/cyan]\n")

    # Process files
    results = process_files(files, args.encoding, path.parent if path.is_file() else path)

    # Display results
    console.print()
    display_results(results, args.encoding)


if __name__ == "__main__":
    main()
