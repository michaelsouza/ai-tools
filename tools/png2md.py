#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
PNG Screenshots to Markdown Converter (local OCR via EasyOCR)

Converts a directory of PNG (or other image) screenshots into a single Markdown
file. Each image becomes a section with its extracted text in a fenced code block.
Images are processed locally — no API key or internet connection required.

Designed for Linux terminal screenshots: monospaced text, high contrast.

Usage:
    # Convert all images in a directory
    python png2md.py ./screenshots

    # Convert a single image
    python png2md.py screenshot.png

    # Non-interactive mode
    python png2md.py ./screenshots -y

    # Custom output path
    python png2md.py ./screenshots -o notes.md

    # Skip preview
    python png2md.py ./screenshots --no-preview

    # Set minimum OCR confidence threshold (default: 0.3)
    python png2md.py ./screenshots --min-confidence 0.5

Options:
    input_path              Path to an image file or directory of images (required)
    -y, --yes               Non-interactive mode — assume Yes to all prompts
    -o, --output PATH       Output Markdown file path
    --no-preview            Skip the Markdown preview after processing
    --min-confidence FLOAT  Minimum confidence score to keep a text detection (default: 0.3)

Supported formats: PNG, JPG/JPEG, WEBP, BMP, TIFF

Notes:
    - EasyOCR downloads its model (~50 MB) on first run to ~/.EasyOCR/
    - GPU is used automatically if available (CUDA), otherwise falls back to CPU
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm
from rich.syntax import Syntax

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def parse_and_validate_arguments(console: Console) -> Optional[argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Convert PNG/image screenshots to Markdown using local EasyOCR."
    )
    parser.add_argument("input_path", help="Path to an image file or a directory of images.")
    parser.add_argument("-y", "--yes", action="store_true", help="Non-interactive mode.")
    parser.add_argument("-o", "--output", help="Output Markdown file path.")
    parser.add_argument("--no-preview", action="store_true", help="Skip Markdown preview.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Minimum OCR confidence threshold (default: 0.3).",
    )
    args = parser.parse_args()

    p = Path(args.input_path)
    if not p.exists():
        console.print(f"[bold red]Error:[/] Path not found: {args.input_path}")
        return None
    if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        console.print(f"[bold red]Error:[/] Unsupported file type: {p.suffix}")
        return None

    return args


def collect_images(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)


def prepare_image(image_path: Path) -> np.ndarray:
    """Load image and invert dark-background terminal screenshots for better OCR."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    # Invert if the image is predominantly dark (terminal default: dark bg, light text)
    if arr.mean() < 128:
        arr = 255 - arr
    return arr


def detections_to_text(detections: list, min_confidence: float) -> str:
    """
    Convert EasyOCR detections to ordered text.
    Detections: list of (bbox, text, confidence)
    bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    Groups by line using vertical midpoint clustering, preserves reading order.
    """
    filtered = [(bbox, text, conf) for bbox, text, conf in detections if conf >= min_confidence]
    if not filtered:
        return ""

    # Compute vertical midpoint for each detection
    items = []
    for bbox, text, conf in filtered:
        ys = [pt[1] for pt in bbox]
        y_mid = (min(ys) + max(ys)) / 2
        x_left = min(pt[0] for pt in bbox)
        items.append((y_mid, x_left, text))

    # Sort by y, then x
    items.sort(key=lambda t: (t[0], t[1]))

    # Cluster into lines: detections within ~half a typical line height are the same line
    if not items:
        return ""

    avg_height = np.mean(
        [max(pt[1] for pt in bbox) - min(pt[1] for pt in bbox) for bbox, _, _ in filtered]
    )
    threshold = max(avg_height * 0.6, 8)

    lines: list[list[tuple]] = []
    current_line: list[tuple] = [items[0]]
    for item in items[1:]:
        if abs(item[0] - current_line[-1][0]) <= threshold:
            current_line.append(item)
        else:
            lines.append(sorted(current_line, key=lambda t: t[1]))
            current_line = [item]
    lines.append(sorted(current_line, key=lambda t: t[1]))

    return "\n".join(" ".join(t for _, _, t in line) for line in lines)


def ocr_image(reader, image_path: Path, min_confidence: float, console: Console) -> Optional[str]:
    try:
        arr = prepare_image(image_path)
        detections = reader.readtext(arr)
        return detections_to_text(detections, min_confidence)
    except Exception as e:
        console.print(f"[bold red]Error processing {image_path.name}:[/] {e}")
        return None


def filename_to_title(name: str) -> str:
    stem = Path(name).stem
    return stem.replace("_", " ").replace("-", " ").title()


def build_markdown(images: list[Path], results: list[Optional[str]]) -> str:
    sections = []
    for image, content in zip(images, results):
        title = filename_to_title(image.name)
        body = f"```\n{content}\n```" if content and content.strip() else "_No content extracted._"
        sections.append(f"## {title}\n\n{body}")
    return "\n\n---\n\n".join(sections)


def determine_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    p = Path(args.input_path)
    if p.is_file():
        return p.with_suffix(".md")
    return p / "output.md"


def main() -> int:
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]PNG → Markdown[/]  [dim]via EasyOCR (local)[/]",
            border_style="cyan",
        )
    )

    args = parse_and_validate_arguments(console)
    if args is None:
        return 2

    images = collect_images(args.input_path)
    if not images:
        console.print("[bold yellow]No supported images found.[/]")
        return 1

    output_path = determine_output_path(args)

    console.print(f"[bold]Images found:[/] {len(images)}")
    for img in images:
        console.print(f"  [dim]•[/] {img.name}")
    console.print(f"[bold]Output:[/] {output_path}")
    console.print(f"[bold]Min confidence:[/] {args.min_confidence}")
    console.print()

    if not args.yes:
        if not Confirm.ask("Proceed?", default=True):
            console.print("[yellow]Aborted.[/]")
            return 0

    with console.status("[bold blue]Loading EasyOCR model...", spinner="dots"):
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    results: list[Optional[str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images...", total=len(images))
        for image in images:
            progress.update(task, description=f"[cyan]{image.name}[/]")
            content = ocr_image(reader, image, args.min_confidence, console)
            results.append(content)
            progress.advance(task)

    failed = sum(1 for r in results if r is None)
    if failed:
        console.print(f"[yellow]Warning:[/] {failed} image(s) failed to process.")

    markdown = build_markdown(images, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    console.print(f"\n[bold green]Saved:[/] {output_path}")
    console.print(f"[dim]{len(markdown):,} characters, {len(images)} section(s)[/]")

    if not args.no_preview:
        preview = markdown[:1200] + ("\n\n…" if len(markdown) > 1200 else "")
        console.print()
        console.print(
            Panel(
                Syntax(preview, "markdown", theme="monokai", word_wrap=True),
                title="Preview",
                border_style="dim",
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
