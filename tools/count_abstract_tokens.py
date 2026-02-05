#!/usr/bin/env python3
"""
Count tokens in the 'abstract' field of BibTeX entries.
Uses bibtexparser to parse the file and tiktoken for counting.
"""

import sys
from pathlib import Path
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import tiktoken
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        console.print(f"[red]Error encoding text: {e}[/red]")
        return 0

def process_bib_file(bib_path: Path, encoding_name: str = "o200k_base"):
    """Parse bib file and count tokens in abstracts."""
    if not bib_path.exists():
        console.print(f"[red]Error: {bib_path} not found[/red]")
        return

    with open(bib_path, 'r', encoding='utf-8') as bibfile:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        bib_database = bibtexparser.load(bibfile, parser=parser)

    results = []
    total_tokens = 0
    entries_with_abstract = 0

    for entry in bib_database.entries:
        key = entry.get('ID', 'Unknown')
        abstract = entry.get('abstract', '')
        
        if abstract:
            tokens = count_tokens(abstract, encoding_name)
            char_count = len(abstract)
            results.append({
                'key': key,
                'tokens': tokens,
                'chars': char_count,
                'abstract': abstract[:50] + "..." if len(abstract) > 50 else abstract
            })
            total_tokens += tokens
            entries_with_abstract += 1
        else:
            # Optionally track entries without abstracts
            pass

    # Sort by token count descending
    results.sort(key=lambda x: x['tokens'], reverse=True)

    # Display results
    table = Table(
        title=f"Abstract Token Counts (File: {bib_path.name})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Citation Key", style="cyan")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Chars", justify="right", style="blue")
    table.add_column("Ratio", justify="right", style="yellow")
    table.add_column("Preview", style="dim")

    for res in results:
        ratio = f"{res['tokens']/res['chars']:.2f}" if res['chars'] > 0 else "N/A"
        table.add_row(
            res['key'],
            f"{res['tokens']:,}",
            f"{res['chars']:,}",
            ratio,
            res['abstract']
        )

    console.print(table)

    # Summary
    summary = Text()
    summary.append(f"Total Entries: ", style="bold")
    summary.append(f"{len(bib_database.entries)}\n", style="cyan")
    summary.append(f"Entries with Abstract: ", style="bold")
    summary.append(f"{entries_with_abstract}\n", style="cyan")
    summary.append(f"Total Abstract Tokens: ", style="bold")
    summary.append(f"{total_tokens:,}\n", style="green")
    if entries_with_abstract > 0:
        summary.append(f"Average Tokens/Abstract: ", style="bold")
        summary.append(f"{total_tokens/entries_with_abstract:.2f}", style="yellow")

    console.print(Panel(summary, title="Summary", border_style="green"))

if __name__ == "__main__":
    bib_file = Path("article/references.bib")
    if len(sys.argv) > 1:
        bib_file = Path(sys.argv[1])
    
    process_bib_file(bib_file)
