#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
PDF to Markdown Converter using Mistral OCR

This script converts PDF documents to Markdown format using Mistral's OCR API.
It uploads PDFs to Mistral's cloud service, processes them with their latest OCR
model (mistral-ocr-latest), and extracts text content while preserving document
structure including headers, lists, tables, and formatting.

The script provides an interactive terminal experience with progress tracking,
confirmation prompts, and formatted output. It can also extract and save images
from the PDF with automatic link rewriting in the generated markdown.

Usage:
    # Basic conversion (interactive mode)
    python pdf2md.py document.pdf

    # Non-interactive mode (auto-confirm all prompts)
    python pdf2md.py document.pdf -y

    # Include images from PDF (saved to disk with relative links)
    python pdf2md.py document.pdf --include-images

    # Specify custom output path
    python pdf2md.py document.pdf -o output/result.md

    # Non-interactive with images and custom output
    python pdf2md.py document.pdf -y --include-images -o result.md

    # Skip markdown preview after processing
    python pdf2md.py document.pdf --no-preview

Options:
    pdf_path             Path to the PDF file to convert (required)
    -y, --yes            Non-interactive mode - assume Yes to all prompts
    --include-images     Extract images from PDF, save to disk, and rewrite markdown links
    -o, --output PATH    Custom output file path (default: same location as PDF with .md extension)
    --no-preview         Skip the markdown preview display after processing

Environment Setup:
    This script requires a Mistral API key set in the environment:

    1. Create a .env file in the project directory:
       echo "MISTRAL_API_KEY=your_api_key_here" > .env

    2. Or export the environment variable:
       export MISTRAL_API_KEY="your_api_key_here"

    Get your API key from: https://console.mistral.ai/

Features:
    - PDF upload to Mistral's secure cloud service
    - High-quality OCR using mistral-ocr-latest model
    - Preserves document structure (headers, paragraphs, lists, tables)
    - Optional image extraction with base64 decoding
    - Automatic relative path rewriting for image links
    - Rich terminal UI with progress bars and spinners
    - Interactive confirmation prompts (can be disabled)
    - Markdown syntax highlighting in preview
    - Error handling with descriptive messages

Output:
    - Markdown file saved alongside PDF (or custom location)
    - Images saved to {pdf_name}_images/ directory (if --include-images)
    - Console preview showing first 500 characters
    - Processing summary with file locations

Image Handling:
    When --include-images is enabled:
    - Images are extracted from Mistral OCR response (base64 encoded)
    - Saved to disk as {pdf_name}_images/{image_id}.{ext}
    - Markdown image links automatically rewritten to relative paths
    - Supports: PNG, JPEG, GIF, WebP, BMP, TIFF formats

Requirements:
    - mistralai: Mistral AI Python SDK
    - python-dotenv: Environment variable management
    - PyPDF2: PDF metadata reading
    - rich: Terminal formatting and UI components

Exit Codes:
    0: Success
    1: Processing error (file not found, upload failed, OCR failed)
    2: Configuration error (invalid arguments, missing API key)

Notes:
    - PDF files are temporarily uploaded to Mistral's servers for processing
    - Signed URLs expire after 1 minute for security
    - Large PDFs may take longer to process
    - Requires active internet connection
    - API usage may incur costs based on Mistral's pricing
"""

import os
import argparse
import base64
import sys
import re
import io
from typing import Optional, Tuple, List, Any  # Added Any

from dotenv import load_dotenv
from mistralai import Mistral
import PyPDF2
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.syntax import Syntax

def parse_and_validate_arguments(console: Console) -> Optional[argparse.Namespace]:
    """Parses command-line arguments and validates the PDF path."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using Mistral OCR (no local fallback)."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to be processed.")
    parser.add_argument("-y", "--yes", action="store_true", help="Run non-interactively (assume Yes to prompts).")
    parser.add_argument("--include-images", action="store_true", help="Include images from OCR, save to disk, and rewrite links in Markdown.")
    parser.add_argument("-o", "--output", help="Output Markdown file path (default: alongside PDF with .md).")
    parser.add_argument("--no-preview", action="store_true", help="Do not print Markdown preview after processing.")
    parser.add_argument("--pages", help="Page range to process (e.g., '1-5', '1,3,5'). 1-based indexing.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        console.print(f"[bold red]Error:[/] The file {args.pdf_path} does not exist.")
        return None
    if not args.pdf_path.lower().endswith(".pdf"):
        console.print(f"[bold red]Error:[/] The file {args.pdf_path} is not a PDF.")
        return None
    return args


def parse_page_range(page_range_str: str) -> List[int]:
    """Parses a string like '1-5,8,11-13' into a list of 0-based page indices."""
    pages = []
    if not page_range_str:
        return pages
    
    parts = page_range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                # Convert 1-based to 0-based, inclusive end
                pages.extend(range(start - 1, end))
            except ValueError:
                continue
        else:
            try:
                # Convert 1-based to 0-based
                pages.append(int(part) - 1)
            except ValueError:
                continue
    return sorted(list(set(pages)))


def initialize_mistral_client(console: Console) -> Optional[Mistral]:
    """Initializes and returns the Mistral client if API key is set."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print(
            "[bold yellow]Warning:[/] MISTRAL_API_KEY environment variable not set. Mistral processing will be skipped."
        )
        return None
    try:
        return Mistral(api_key=api_key)
    except Exception as e:
        console.print(f"[bold red]Error initializing Mistral client:[/] {e}")
        return None


def get_pdf_details(
    pdf_path: str, console: Console, pages_arg: Optional[str] = None
) -> Tuple[Optional[bytes], Optional[int]]:
    """Reads PDF content (optionally filtered by page range) and gets the number of pages."""
    with console.status("[bold green]Reading PDF file...", spinner="dots"):
        try:
            # If no pages specified, read the file directly
            if not pages_arg:
                with open(pdf_path, "rb") as pdf_file_obj:
                    pdf_content = pdf_file_obj.read()
                # Re-open for PyPDF2 as it needs a seekable stream after full read
                with open(pdf_path, "rb") as pdf_file_for_pypdf2:
                    pdf_reader = PyPDF2.PdfReader(pdf_file_for_pypdf2)
                    num_pages = len(pdf_reader.pages)
                return pdf_content, num_pages
            
            # If pages ARE specified, extract them
            target_indices = parse_page_range(pages_arg)
            if not target_indices:
                 console.print(f"[bold red]Error:[/] Invalid page range '{pages_arg}'.")
                 return None, None

            output_pdf = PyPDF2.PdfWriter()
            with open(pdf_path, "rb") as pdf_file_for_pypdf2:
                pdf_reader = PyPDF2.PdfReader(pdf_file_for_pypdf2)
                total_pages = len(pdf_reader.pages)
                
                added_count = 0
                for idx in target_indices:
                    if 0 <= idx < total_pages:
                        output_pdf.add_page(pdf_reader.pages[idx])
                        added_count += 1
                    else:
                        console.print(f"[yellow]Warning:[/] Page {idx + 1} is out of bounds (1-{total_pages}). Skipped.")

                if added_count == 0:
                     console.print("[bold red]Error:[/] No valid pages selected.")
                     return None, None

                # Write the new PDF to bytes
                pdf_bytes_io = io.BytesIO()
                output_pdf.write(pdf_bytes_io)
                pdf_bytes_io.seek(0)
                return pdf_bytes_io.read(), added_count

        except Exception as e:
            console.print(
                f"[bold red]Error reading PDF {os.path.basename(pdf_path)}:[/] {e}"
            )
            return None, None


def display_pdf_info(pdf_filename: str, num_pages: int, console: Console):
    """Displays PDF information in a Rich panel."""
    console.print(
        Panel(
            f"[bold]PDF Details[/]\nFilename: [cyan]{pdf_filename}[/]\nPages: [yellow]{num_pages}[/]",
            border_style="green",
            title="File Information",
        )
    )


def confirm_and_configure_processing(
    num_pages: int, pdf_filename: str, console: Console, args: argparse.Namespace
) -> Tuple[bool, bool]:
    """Handles proceed/include-images decisions with support for non-interactive mode."""
    if args.yes:
        return True, bool(args.include_images)

    proceed = Confirm.ask(
        f"Do you want to proceed with processing {num_pages} pages of '{pdf_filename}'? (Y/n)",
        default=True,
        show_default=False,
    )
    if not proceed:
        return False, False

    include_mistral_images = (
        args.include_images
        if args.include_images
        else Confirm.ask(
            "Include images (base64 from Mistral), save to disk, and rewrite links? (y/N)",
            default=False,
            show_default=False,
        )
    )
    console.print(
        f"Images will {'[green]be included[/]' if include_mistral_images else '[yellow]not be included[/]'} in the output."
    )
    return True, bool(include_mistral_images)


def upload_pdf_to_mistral(
    client: Mistral, pdf_path: str, pdf_content: bytes, console: Console
) -> Optional[str]:
    """Uploads the PDF file to Mistral and returns the signed URL."""
    with console.status("[bold blue]Uploading PDF to Mistral...", spinner="dots"):
        try:
            uploaded_file = client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": pdf_content,
                },
                purpose="ocr",
            )
            signed_url_response = client.files.get_signed_url(
                file_id=uploaded_file.id, expiry=1
            )  # expiry in minutes
            return signed_url_response.url
        except Exception as e:
            console.print(f"[bold red]Error uploading PDF to Mistral:[/] {e}")
            return None


def process_ocr_with_mistral(
    client: Mistral, document_url: str, include_image_base64: bool, console: Console
) -> Optional[Any]:  # Using Any for Mistral's response type
    """Processes the document URL with Mistral OCR."""
    with console.status("[bold blue]Processing OCR with Mistral...", spinner="dots"):
        try:
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": document_url},
                include_image_base64=include_image_base64,
            )
            return response
        except Exception as e:
            console.print(f"[bold red]Error during Mistral OCR processing:[/] {e}")
            return None


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return sanitized[:255] if len(sanitized) > 255 else sanitized


def _ext_from_b64_header(b64_data: str) -> str:
    try:
        if b64_data.startswith("data:"):
            header = b64_data.split(",", 1)[0]
            # e.g., data:image/png;base64
            if ";" in header:
                mime = header.split(":", 1)[1].split(";", 1)[0]
            else:
                mime = header.split(":", 1)[1]
            if "/" in mime:
                subtype = mime.split("/", 1)[1].lower()
                # common normalizations
                if subtype in {"jpeg", "jpg"}:
                    return ".jpg"
                if subtype == "png":
                    return ".png"
                if subtype in {"gif", "webp", "bmp", "tiff"}:
                    return f".{subtype}"
        return ".png"
    except Exception:
        return ".png"


def _save_image_from_base64_data(images_dir: str, image_id: str, b64_data: str, console: Console) -> Optional[str]:
    """Decodes base64 image data and saves it to images_dir. Returns saved file path or None."""
    try:
        os.makedirs(images_dir, exist_ok=True)
        ext = _ext_from_b64_header(b64_data)
        comma_index = b64_data.find(",")
        b64_string = b64_data[comma_index + 1 :] if comma_index != -1 else b64_data
        image_bytes = base64.b64decode(b64_string)
        safe_id = _sanitize_filename(image_id)
        img_path = os.path.join(images_dir, safe_id + ext)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        return img_path
    except Exception as img_e:
        console.print(f"[bold red]Error processing and saving image {image_id}:[/] {img_e}")
        return None


def extract_pages_content_and_save_images_mistral(
    ocr_response: Any,
    include_image_base64: bool,
    console: Console,
    images_dir: Optional[str],
    output_dir: Optional[str],
) -> List[str]:
    """Extracts markdown from Mistral OCR pages and saves images if requested, with progress.

    When include_image_base64 is True, images are saved under images_dir and markdown links
    referencing image IDs are rewritten to relative file paths.
    """
    all_markdown_parts: List[str] = []
    total_tasks = len(ocr_response.pages)
    if include_image_base64:
        for page in ocr_response.pages:
            total_tasks += len(getattr(page, "images", []) or [])

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            "[bold green]Processing Mistral OCR content...", total=total_tasks
        )

        for page in ocr_response.pages:
            progress.update(
                task_id,
                advance=1,
                description=f"[bold green]Processing page {page.index + 1}/{len(ocr_response.pages)} (Mistral)...",
            )

            page_md = page.markdown or ""
            if include_image_base64 and getattr(page, "images", None):
                for image in page.images:
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[bold cyan]Saving image {image.id} (page {page.index + 1})...",
                    )
                    if images_dir and output_dir:
                        saved = _save_image_from_base64_data(
                            images_dir, image.id, image.image_base64, console
                        )
                        if saved:
                            rel_path = os.path.relpath(saved, start=output_dir)
                            # Replace only in link targets: ](ID)
                            page_md = re.sub(
                                rf"\]\({re.escape(image.id)}\)", f"]({rel_path})", page_md
                            )
            all_markdown_parts.append(page_md)
    return all_markdown_parts


# --- Common Utility Functions ---
def generate_output_filename(pdf_path: str) -> str:
    """Generates the output Markdown filename in the same directory as the PDF."""
    pdf_dir = os.path.dirname(pdf_path)
    base_name_without_ext, _ = os.path.splitext(os.path.basename(pdf_path))
    return os.path.join(pdf_dir, base_name_without_ext + ".md")


def save_markdown_to_file(final_markdown: str, output_filename: str):
    """Saves the final markdown content to a file."""
    with open(output_filename, "w", encoding="utf-8") as fd:
        fd.write(final_markdown)


def display_results_summary(
    output_filename: str, final_markdown: str, console: Console, show_preview: bool = True
):
    """Displays a success message and a preview of the generated markdown."""
    console.print(
        Panel(
            f"[bold green]Success![/] Output saved to [cyan]{output_filename}[/]",
            border_style="green",
            title="Processing Complete",
        )
    )
    if show_preview:
        console.print("\n[bold]Preview of generated markdown (first 500 chars):[/]")
        markdown_preview = (
            final_markdown[:500] + "..." if len(final_markdown) > 500 else final_markdown
        )
        console.print(
            Syntax(markdown_preview, "markdown", theme="monokai", line_numbers=True)
        )


#
# --- Main Orchestration Function ---
#
def main():
    """Main function that orchestrates the PDF processing workflow (Mistral only)."""
    load_dotenv('/home/michael/gitrepos/ai-tools/.env')
    console = Console()

    args = parse_and_validate_arguments(console)
    if not args:
        sys.exit(2)

    # Determine output path early for image link rewriting
    output_md_filename = (
        os.path.abspath(args.output)
        if args.output
        else generate_output_filename(args.pdf_path)
    )
    output_dir = os.path.dirname(output_md_filename)
    pdf_stem = os.path.splitext(os.path.basename(args.pdf_path))[0]
    images_dir = os.path.join(output_dir, f"{pdf_stem}_images") if args.include_images else None

    pdf_content, num_pages = get_pdf_details(args.pdf_path, console, args.pages)
    if pdf_content is None or num_pages is None:
        sys.exit(1)

    pdf_filename_basename = os.path.basename(args.pdf_path)
    display_pdf_info(pdf_filename_basename, num_pages, console)

    proceed, include_mistral_images_in_output = confirm_and_configure_processing(
        num_pages, pdf_filename_basename, console, args
    )
    if not proceed:
        console.print("[yellow]Processing cancelled by user.[/]")
        sys.exit(0)

    # --- Mistral OCR ---
    mistral_client = initialize_mistral_client(console)
    if not mistral_client:
        console.print("[bold red]Mistral client not available or configured.[/]")
        sys.exit(2)

    console.print("\n[cyan]Processing with Mistral OCR...[/]")
    try:
        signed_url_str = upload_pdf_to_mistral(
            mistral_client, args.pdf_path, pdf_content, console
        )
        if not signed_url_str:
            console.print("[bold red]Failed to upload PDF to Mistral.[/]")
            sys.exit(1)

        ocr_response = process_ocr_with_mistral(
            mistral_client,
            signed_url_str,
            include_mistral_images_in_output,
            console,
        )
        if not ocr_response or not hasattr(ocr_response, "pages"):
            console.print(
                "[bold red]Mistral OCR processing failed or returned no pages.[/]"
            )
            sys.exit(1)

        all_markdown_parts = extract_pages_content_and_save_images_mistral(
            ocr_response,
            include_mistral_images_in_output,
            console,
            images_dir=images_dir,
            output_dir=output_dir,
        )
        final_markdown = "\n\n---\n\n".join(all_markdown_parts)
        # Save output
        try:
            os.makedirs(os.path.dirname(output_md_filename), exist_ok=True)
            save_markdown_to_file(final_markdown, output_md_filename)
        except Exception as e:
            console.print(
                f"[bold red]Failed to write output file '{output_md_filename}':[/] {e}"
            )
            sys.exit(1)

        # Display summary/preview
        display_results_summary(
            output_md_filename, final_markdown, console, show_preview=(not args.no_preview)
        )
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error during Mistral processing:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
