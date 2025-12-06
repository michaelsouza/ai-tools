#!/home/michael/gitrepos/ai-tools/.venv/bin/ python
"""
URL to Markdown Converter

This script downloads web pages and extracts their main content as clean Markdown.
It uses a three-tier extraction pipeline to ensure high-quality output:
1. Trafilatura for direct markdown extraction
2. Readability-lxml + Markdownify for article content
3. Heuristic container selection for documentation sites

Usage:
    # Basic usage - output to stdout
    python url2md.py https://example.com/article

    # Save to file
    python url2md.py https://example.com/article -o output.md

    # Save original HTML for debugging
    python url2md.py https://example.com/article -o output.md --save-html raw.html

    # Save cleaned HTML (post-extraction)
    python url2md.py https://example.com/article --save-clean-html clean.html

Features:
    - Robust character encoding detection
    - Multiple extraction strategies for different site types
    - Preserves document structure (headers, lists, tables, links)
    - Cleans up excessive whitespace and formatting
    - Works well with documentation sites, blogs, and articles

Requirements:
    - requests: HTTP client for fetching pages
    - chardet: Character encoding detection
    - beautifulsoup4: HTML parsing
    - trafilatura: Content extraction
    - readability-lxml: Article isolation
    - markdownify: HTML to Markdown conversion
"""
import argparse
import sys
import re
from pathlib import Path

import requests
import chardet
from bs4 import BeautifulSoup
import trafilatura
from readability.readability import Document
from markdownify import markdownify as html_to_md


def fetch_html(url: str, timeout: int = 20) -> str:
    """Download page HTML with basic robustness."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Robust decode if server headers are wrong
    content = resp.content
    encoding = resp.encoding
    if not encoding:
        guess = chardet.detect(content)
        encoding = guess.get("encoding") or "utf-8"
    return content.decode(encoding, errors="replace")


def clean_markdown(md: str) -> str:
    """Light cleanup for nicer Markdown."""
    # Collapse excessive blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Trim trailing spaces
    md = "\n".join(line.rstrip() for line in md.splitlines())
    return md.strip() + "\n"


def extract_with_trafilatura(html: str) -> str | None:
    """Try extracting main content via trafilatura, preferring markdown output if supported."""
    if trafilatura is None:
        return None
    try:
        # Some versions support output_format="markdown"
        md = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=True,
            favor_recall=True,
            output_format="markdown",  # will be ignored on older versions -> TypeError
        )
        if md:
            return md
    except TypeError:
        # Older trafilatura: no markdown output—get cleaned text/HTML and convert
        pass

    # Older versions return plain text; try to get cleaned HTML and convert
    # (trafilatura.extract returns str, often text. We'll fall back to readability + markdownify next.)
    txt = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=True,
        favor_recall=True,
    )
    if txt and len(txt.strip()) > 0:
        # Convert plain text to rudimentary Markdown paragraphs
        return "\n\n".join(p.strip() for p in txt.splitlines() if p.strip())

    return None


def pick_main_container(html: str) -> str | None:
    """Heuristic: pick likely main content container and return its inner HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Prefer <article>
    article = soup.find("article")
    if article:
        return str(article)

    # Docs sites often have these containers:
    for cls in [
        "markdown", "md-content", "prose", "content", "article-content",
        "main-content", "docs-content", "md-main__inner", "docMainContainer"
    ]:
        node = soup.find(attrs={"class": re.compile(rf"\b{re.escape(cls)}\b")})
        if node:
            return str(node)

    # As a last resort, the <main> tag
    main = soup.find("main")
    if main:
        return str(main)

    # Fallback: body (risky, but better than nothing)
    body = soup.find("body")
    if body:
        return str(body)
    return None


def readability_to_html(html: str) -> str | None:
    """Use readability-lxml to isolate article HTML."""
    if Document is None:
        return None
    try:
        doc = Document(html)
        return doc.summary(html_partial=True)
    except Exception:
        return None


def html_to_markdown(html_fragment: str) -> str:
    """Convert HTML to Markdown via markdownify if available; else naive text fallback."""
    if html_to_md is None:
        # Bare text fallback
        soup = BeautifulSoup(html_fragment, "html.parser")
        return soup.get_text("\n")
    return html_to_md(
        html_fragment,
        heading_style="ATX",
        strip=["script", "style", "noscript"],
        # Optional tweaks:
        # bullets="*",
        # code_language_detection=False,
    )


def extract_markdown_from_html(html: str) -> str:
    """Best-effort pipeline to produce Markdown from page HTML."""
    # 1) Try Trafilatura direct to Markdown
    md = extract_with_trafilatura(html)
    if md and len(md.strip()) > 100:
        return clean_markdown(md)

    # 2) If not good enough, try readability + markdownify
    readable = readability_to_html(html)
    if readable and len(BeautifulSoup(readable, "html.parser").get_text(strip=True)) > 80:
        md2 = html_to_markdown(readable)
        if md2 and len(md2.strip()) > 80:
            return clean_markdown(md2)

    # 3) Heuristic container pick + markdownify
    container = pick_main_container(html) or html
    md3 = html_to_markdown(container)
    return clean_markdown(md3)


def main():
    ap = argparse.ArgumentParser(
        description="Download a web page and extract the main content as Markdown."
    )
    ap.add_argument("url", help="Page URL to download")
    ap.add_argument("-o", "--output", help="Path to write .md file (default: stdout)")
    ap.add_argument("--save-html", help="Also save the original HTML to this path")
    ap.add_argument("--save-clean-html", help="Save cleaned/isolated HTML (post-extraction)")
    args = ap.parse_args()

    html = fetch_html(args.url)

    if args.save_html:
        Path(args.save_html).write_text(html, encoding="utf-8")

    md = extract_markdown_from_html(html)

    # Optionally save cleaned HTML (using readability/container heuristic)
    if args.save_clean_html:
        readable = readability_to_html(html)
        if not readable:
            readable = pick_main_container(html) or html
        Path(args.save_clean_html).write_text(readable, encoding="utf-8")

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
