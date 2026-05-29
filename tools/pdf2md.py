#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
PDF to Markdown Converter using Nougat (local, default) or Mistral OCR (cloud, opt-in).

Default mode uses Facebook's Nougat model which runs entirely locally -- no data leaves
your machine. Pass --model mistral to use Mistral's cloud OCR API instead (requires a
MISTRAL_API_KEY in .env). Images are only available in Mistral mode (--include-images).

Usage:
    # Basic conversion with default Nougat model
    python pdf2md.py document.pdf

    # Use Mistral OCR instead
    python pdf2md.py document.pdf --model mistral

    # Convert all PDFs in a directory
    python pdf2md.py ./pdfs

    # Non-interactive mode (auto-confirm all prompts)
    python pdf2md.py document.pdf -y

    # Specify custom output path
    python pdf2md.py document.pdf -o output/result.md

    # Nougat-specific: batch size, no-skipping, full precision
    python pdf2md.py document.pdf -b 8 --no-skipping --full-precision

    # Mistral-specific: include images
    python pdf2md.py document.pdf --model mistral --include-images

Options:
    pdf_path             Path to the PDF file or directory to process (required)
    -y, --yes            Non-interactive mode - assume Yes to all prompts
    -o, --output PATH    Custom output file path (default: same location as PDF with .md extension)
    --no-preview         Skip the markdown preview display after processing
    --pages PAGES        Page range to process (e.g., '1-5', '1,3,5'). 1-based indexing.

Model selection:
    --model {nougat,mistral}
                         OCR engine to use (default: nougat). 'nougat' runs locally,
                         'mistral' requires MISTRAL_API_KEY in .env.

Nougat options (default model):
    -b, --batch-size N   Batch size for inference (default: 4; increase if you have VRAM)
    --no-skipping        Disable failure-detection heuristic (use if you get [MISSING_PAGE])
    --full-precision     Use float32 instead of bfloat16 (can help on CPU)

Mistral options:
    --include-images     Extract images from PDF, save to disk, and rewrite markdown links

Output:
    - Markdown file saved alongside PDF (or custom location)
    - Images saved to {pdf_name}_images/ directory (Mistral mode with --include-images)

Requirements:
    - nougat-ocr (default): Facebook's Nougat model + torch
    - mistralai (optional): Mistral AI Python SDK + python-dotenv
    - PyPDF2: PDF page extraction
    - rich: Terminal formatting and UI components

Exit Codes:
    0: Success
    1: Processing error
    2: Configuration error

Notes:
    - Nougat: runs entirely locally, GPU strongly recommended, model weights (~2GB)
      downloaded on first use. Model weights are CC BY-NC licensed (non-commercial).
    - Mistral: requires internet, API key, and may incur costs.
"""

import os
import argparse
import base64
import sys
import re
import io
import tempfile
from typing import Optional, Tuple, List, Any

from dotenv import load_dotenv
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
    """Parses command-line arguments and validates the input path."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using Nougat (local, default) or Mistral OCR (cloud, opt-in)."
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file or a directory containing PDFs to be processed.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Run non-interactively (assume Yes to prompts).")
    parser.add_argument("-o", "--output", help="Output Markdown file path (default: alongside PDF with .md).")
    parser.add_argument("--no-preview", action="store_true", help="Do not print Markdown preview after processing.")
    parser.add_argument("--pages", help="Page range to process (e.g., '1-5', '1,3,5'). 1-based indexing.")

    parser.add_argument("--model", choices=["nougat", "mistral"], default="mistral",
                        help="OCR engine: 'nougat' (local, default) or 'mistral' (cloud API).")

    parser.add_argument("-b", "--batch-size", type=int, default=4,
                        help="Batch size for Nougat inference (default: 4).")
    parser.add_argument("--no-skipping", action="store_true",
                        help="Disable Nougat failure-detection (use if you get [MISSING_PAGE]).")
    parser.add_argument("--full-precision", action="store_true",
                        help="Use float32 instead of bfloat16 (Nougat, can help on CPU).")

    parser.add_argument("--include-images", action="store_true",
                        help="Include images from Mistral OCR, save to disk, and rewrite links in Markdown.")

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        console.print(f"[bold red]Error:[/] The path {args.pdf_path} does not exist.")
        return None
    if os.path.isfile(args.pdf_path) and not args.pdf_path.lower().endswith(".pdf"):
        console.print(f"[bold red]Error:[/] The file {args.pdf_path} is not a PDF.")
        return None
    if os.path.isdir(args.pdf_path) and args.pages:
        console.print(
            "[bold red]Error:[/] The --pages option is only supported for a single PDF file."
        )
        return None
    if (
        os.path.isdir(args.pdf_path)
        and args.output
        and os.path.exists(args.output)
        and not os.path.isdir(args.output)
    ):
        console.print(
            "[bold red]Error:[/] When processing a directory, --output must be a directory path."
        )
        return None
    if args.model == "nougat" and args.include_images:
        console.print(
            "[bold yellow]Warning:[/] --include-images is only supported with --model mistral. Ignored."
        )
        args.include_images = False
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
                pages.extend(range(start - 1, end))
            except ValueError:
                continue
        else:
            try:
                pages.append(int(part) - 1)
            except ValueError:
                continue
    return sorted(list(set(pages)))


def get_pdf_path_for_processing(
    pdf_path: str, console: Console, pages_arg: Optional[str] = None
) -> Tuple[Optional[str], Optional[int]]:
    """Returns the PDF path (possibly a temp file if page range specified) and page count.

    When pages_arg is provided, a temporary PDF containing only the requested pages
    is created. The caller is responsible for cleaning up the temp file (returned
    filename indicates if cleanup is needed via a leading '!' marker).
    """
    with console.status("[bold green]Reading PDF file...", spinner="dots"):
        try:
            if not pages_arg:
                with open(pdf_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    num_pages = len(pdf_reader.pages)
                return pdf_path, num_pages

            target_indices = parse_page_range(pages_arg)
            if not target_indices:
                console.print(f"[bold red]Error:[/] Invalid page range '{pages_arg}'.")
                return None, None

            output_pdf = PyPDF2.PdfWriter()
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                total_pages = len(pdf_reader.pages)

                added_count = 0
                for idx in target_indices:
                    if 0 <= idx < total_pages:
                        output_pdf.add_page(pdf_reader.pages[idx])
                        added_count += 1
                    else:
                        console.print(
                            f"[yellow]Warning:[/] Page {idx + 1} is out of bounds (1-{total_pages}). Skipped."
                        )

                if added_count == 0:
                    console.print("[bold red]Error:[/] No valid pages selected.")
                    return None, None

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                output_pdf.write(tmp)
                tmp.close()
                return tmp.name, added_count

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
) -> bool:
    """Asks user to proceed. Returns True to continue, False to skip."""
    if args.yes:
        return True

    proceed = Confirm.ask(
        f"Process {num_pages} pages of '{pdf_filename}' with [cyan]{args.model}[/]? (Y/n)",
        default=True,
        show_default=False,
    )
    return bool(proceed)


# --- Nougat (local) functions ---

def initialize_nougat_model(console: Console, args: argparse.Namespace) -> Optional[dict]:
    """Loads the Nougat model and processor from HuggingFace hub."""
    import torch
    from transformers import NougatProcessor, VisionEncoderDecoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.full_precision else torch.bfloat16
    model_id = "facebook/nougat-base"

    with console.status(f"[bold green]Loading Nougat ({model_id}) on {device}...", spinner="dots"):
        try:
            processor = NougatProcessor.from_pretrained(model_id)
            model = VisionEncoderDecoderModel.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(device)
            model.eval()
            console.print(f"[green]Nougat model and processor loaded on {device}.[/]")
            return {"model": model, "processor": processor}
        except Exception as e:
            console.print(f"[bold red]Error loading Nougat model:[/] {e}")
            return None


def _render_pdf_to_images(pdf_path: str, console: Console) -> Optional[List["Image.Image"]]:
    """Renders each page of a PDF to a PIL Image using pypdfium2."""
    import pypdfium2
    from PIL import Image  # noqa: F811

    with console.status("[bold blue]Rendering PDF pages to images...", spinner="dots"):
        try:
            pdf = pypdfium2.PdfDocument(pdf_path)
            images = []
            for i in range(len(pdf)):
                page = pdf.get_page(i)
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            pdf.close()
        except Exception as e:
            console.print(f"[bold red]Error rendering PDF:[/] {e}")
            return None
    return images


def process_nougat_ocr(
    pdf_path: str, nougat: dict, console: Console, args: argparse.Namespace
) -> Optional[List[str]]:
    """Runs Nougat inference on a PDF, returning a list of page markdown strings."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nougat["model"]
    processor = nougat["processor"]

    images = _render_pdf_to_images(pdf_path, console)
    if not images:
        return None

    all_predictions: List[str] = []
    num_batches = (len(images) + args.batch_size - 1) // args.batch_size

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
            "[bold green]Running Nougat inference...", total=num_batches
        )

        bad_words_ids = [[processor.tokenizer.unk_token_id]] if not args.no_skipping else None

        for batch_idx in range(0, len(images), args.batch_size):
            batch_images = images[batch_idx:batch_idx + args.batch_size]
            progress.update(
                task_id,
                advance=1,
                description=f"[bold green]Nougat batch {batch_idx // args.batch_size + 1}/{num_batches}...",
            )

            try:
                pixel_values = processor(
                    images=batch_images, return_tensors="pt"
                ).pixel_values.to(device=device, dtype=model.dtype)

                with torch.no_grad():
                    outputs = model.generate(
                        pixel_values,
                        min_length=1,
                        max_new_tokens=model.config.decoder.max_position_embeddings,
                        bad_words_ids=bad_words_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                batch_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                for text in batch_texts:
                    text = processor.post_process_generation(text, fix_markdown=False)
                    all_predictions.append(text)

            except Exception as e:
                console.print(
                    f"[bold red]Nougat inference error at batch {batch_idx // args.batch_size + 1}:[/] {e}"
                )
                return None

    return all_predictions


# --- Mistral (cloud) functions ---

def initialize_mistral_client(console: Console) -> Optional[Any]:
    """Initializes and returns the Mistral client if API key is set."""
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] MISTRAL_API_KEY not set. Cannot use --model mistral.")
        return None
    try:
        return Mistral(api_key=api_key)
    except Exception as e:
        console.print(f"[bold red]Error initializing Mistral client:[/] {e}")
        return None


def upload_pdf_to_mistral(
    client: Any, pdf_path: str, console: Console
) -> Optional[str]:
    """Uploads the PDF file to Mistral and returns the signed URL."""
    with console.status("[bold blue]Uploading PDF to Mistral...", spinner="dots"):
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            uploaded_file = client.files.upload(
                file={
                    "file_name": os.path.basename(pdf_path),
                    "content": pdf_content,
                },
                purpose="ocr",
            )
            signed_url_response = client.files.get_signed_url(
                file_id=uploaded_file.id, expiry=1
            )
            return signed_url_response.url
        except Exception as e:
            console.print(f"[bold red]Error uploading PDF to Mistral:[/] {e}")
            return None


def process_ocr_with_mistral(
    client: Any, document_url: str, include_image_base64: bool, console: Console
) -> Optional[Any]:
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
            if ";" in header:
                mime = header.split(":", 1)[1].split(";", 1)[0]
            else:
                mime = header.split(":", 1)[1]
            if "/" in mime:
                subtype = mime.split("/", 1)[1].lower()
                if subtype in {"jpeg", "jpg"}:
                    return ".jpg"
                if subtype == "png":
                    return ".png"
                if subtype in {"gif", "webp", "bmp", "tiff"}:
                    return f".{subtype}"
        return ".png"
    except Exception:
        return ".png"


def _build_mistral_image_filename(
    pdf_stem: str, page_number: int, image_number: int, b64_data: str
) -> str:
    """Builds a stable, readable image filename for a Mistral OCR image."""
    ext = _ext_from_b64_header(b64_data)
    safe_stem = _sanitize_filename(pdf_stem)
    return f"{safe_stem}_p{page_number:03d}_img{image_number:03d}{ext}"


def _strip_markdown_image_links(markdown: str) -> str:
    """Removes standalone Markdown image links when image files are not saved."""
    return re.sub(r"(?m)^[ \t]*!\[[^\]]*\]\([^)]+\)[ \t]*\n?", "", markdown)


def _save_image_from_base64_data(
    images_dir: str,
    image_filename: str,
    b64_data: str,
    console: Console,
) -> Optional[str]:
    """Decodes base64 image data and saves it to images_dir. Returns saved file path or None."""
    try:
        os.makedirs(images_dir, exist_ok=True)
        comma_index = b64_data.find(",")
        b64_string = b64_data[comma_index + 1:] if comma_index != -1 else b64_data
        image_bytes = base64.b64decode(b64_string)
        safe_filename = _sanitize_filename(image_filename)
        img_path = os.path.join(images_dir, safe_filename)
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        return img_path
    except Exception as img_e:
        console.print(f"[bold red]Error processing and saving image {image_filename}:[/] {img_e}")
        return None


def extract_pages_content_and_save_images_mistral(
    ocr_response: Any,
    include_image_base64: bool,
    console: Console,
    images_dir: Optional[str],
    output_dir: Optional[str],
    pdf_stem: str,
) -> List[str]:
    """Extracts markdown from Mistral OCR pages and saves images if requested, with progress."""
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
            if not include_image_base64:
                page_md = _strip_markdown_image_links(page_md)

            if include_image_base64 and getattr(page, "images", None):
                page_number = getattr(page, "index", 0) + 1
                for image_number, image in enumerate(page.images, start=1):
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"[bold cyan]Saving image {image.id} (page {page.index + 1})...",
                    )
                    if images_dir and output_dir:
                        image_filename = _build_mistral_image_filename(
                            pdf_stem, page_number, image_number, image.image_base64
                        )
                        saved = _save_image_from_base64_data(
                            images_dir, image_filename, image.image_base64, console
                        )
                        if saved:
                            rel_path = os.path.relpath(saved, start=output_dir)
                            page_md = re.sub(
                                rf"\]\({re.escape(image.id)}\)", f"]({rel_path})", page_md
                            )
            all_markdown_parts.append(page_md)
    return all_markdown_parts


# --- Common Utility Functions ---

def generate_output_filename(pdf_path: str) -> str:
    """Generates the output Markdown filename in the same directory as the PDF."""
    pdf_dir = os.path.dirname(os.path.abspath(pdf_path))
    base_name_without_ext, _ = os.path.splitext(os.path.basename(pdf_path))
    return os.path.join(pdf_dir, base_name_without_ext + ".md")


def generate_output_filename_for_directory(pdf_path: str, output_dir: str) -> str:
    """Generates the output Markdown filename inside the provided output directory."""
    base_name_without_ext, _ = os.path.splitext(os.path.basename(pdf_path))
    return os.path.join(os.path.abspath(output_dir), base_name_without_ext + ".md")


def collect_pdf_files(input_path: str) -> List[str]:
    """Collects PDF files from a single PDF path or a directory."""
    if os.path.isfile(input_path):
        return [os.path.abspath(input_path)]

    pdf_files = []
    for entry in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(".pdf"):
            pdf_files.append(os.path.abspath(full_path))
    return pdf_files


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


def process_single_pdf(
    pdf_path: str,
    output_md_filename: str,
    console: Console,
    args: argparse.Namespace,
    nougat_model: Optional[Any],
    mistral_client: Optional[Any],
    show_preview: bool,
) -> bool:
    """Processes a single PDF with the selected model. Returns True on success."""
    output_dir = os.path.dirname(output_md_filename)
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    if os.path.exists(output_md_filename):
        console.print(
            f"[yellow]Skipping[/] [cyan]{os.path.basename(pdf_path)}[/]: output already exists at [cyan]{output_md_filename}[/]."
        )
        return True

    pdf_path_to_use, num_pages = get_pdf_path_for_processing(pdf_path, console, args.pages)
    if pdf_path_to_use is None or num_pages is None:
        return False

    pdf_filename_basename = os.path.basename(pdf_path)
    display_pdf_info(pdf_filename_basename, num_pages, console)

    if not confirm_and_configure_processing(num_pages, pdf_filename_basename, console, args):
        console.print("[yellow]Processing cancelled by user.[/]")
        return True

    is_temp_file = pdf_path_to_use != pdf_path
    temp_files_to_clean = [pdf_path_to_use] if is_temp_file else []

    try:
        if args.model == "nougat":
            console.print(f"\n[cyan]Processing with Nougat (local)...[/]")
            all_markdown_parts = process_nougat_ocr(pdf_path_to_use, nougat_model, console, args)
            if all_markdown_parts is None:
                return False
        else:
            console.print(f"\n[cyan]Processing with Mistral OCR...[/]")
            signed_url_str = upload_pdf_to_mistral(mistral_client, pdf_path_to_use, console)
            if not signed_url_str:
                console.print("[bold red]Failed to upload PDF to Mistral.[/]")
                return False

            ocr_response = process_ocr_with_mistral(
                mistral_client,
                signed_url_str,
                args.include_images,
                console,
            )
            if not ocr_response or not hasattr(ocr_response, "pages"):
                console.print("[bold red]Mistral OCR processing failed or returned no pages.[/]")
                return False

            images_dir = os.path.join(output_dir, f"{pdf_stem}_images") if args.include_images else None
            all_markdown_parts = extract_pages_content_and_save_images_mistral(
                ocr_response,
                args.include_images,
                console,
                images_dir=images_dir,
                output_dir=output_dir,
                pdf_stem=pdf_stem,
            )

        final_markdown = "\n\n---\n\n".join(all_markdown_parts)

        os.makedirs(output_dir, exist_ok=True)
        save_markdown_to_file(final_markdown, output_md_filename)

        display_results_summary(
            output_md_filename, final_markdown, console, show_preview=show_preview
        )
        return True

    finally:
        for tmp in temp_files_to_clean:
            try:
                os.unlink(tmp)
            except OSError:
                pass


#
# --- Main Orchestration Function ---
#
def main():
    """Main function that orchestrates the PDF processing workflow."""
    load_dotenv('/home/michael/gitrepos/ai-tools/.env')
    console = Console()

    args = parse_and_validate_arguments(console)
    if not args:
        sys.exit(2)

    nougat_model = None
    mistral_client = None

    if args.model == "nougat":
        nougat_model = initialize_nougat_model(console, args)
        if not nougat_model:
            console.print("[bold red]Failed to load Nougat model.[/]")
            sys.exit(1)
    else:
        mistral_client = initialize_mistral_client(console)
        if not mistral_client:
            console.print("[bold red]Mistral client not available or configured.[/]")
            sys.exit(2)

    try:
        pdf_files = collect_pdf_files(args.pdf_path)
        if not pdf_files:
            console.print("[bold red]Error:[/] No PDF files were found to process.")
            sys.exit(1)

        is_directory_mode = os.path.isdir(args.pdf_path)
        failures = 0

        if is_directory_mode:
            console.print(
                Panel(
                    f"[bold]Directory mode[/]\nFound [yellow]{len(pdf_files)}[/] PDF file(s) in [cyan]{os.path.abspath(args.pdf_path)}[/].",
                    border_style="blue",
                    title="Batch Processing",
                )
            )

        for index, pdf_path in enumerate(pdf_files, start=1):
            if is_directory_mode:
                console.print(
                    f"\n[bold blue]File {index}/{len(pdf_files)}:[/] [cyan]{os.path.basename(pdf_path)}[/]"
                )

            if args.output:
                output_md_filename = (
                    os.path.abspath(args.output)
                    if not is_directory_mode
                    else generate_output_filename_for_directory(pdf_path, args.output)
                )
            else:
                output_md_filename = generate_output_filename(pdf_path)

            success = process_single_pdf(
                pdf_path=pdf_path,
                output_md_filename=output_md_filename,
                console=console,
                args=args,
                nougat_model=nougat_model,
                mistral_client=mistral_client,
                show_preview=(not args.no_preview) and not is_directory_mode,
            )
            if not success:
                failures += 1

        if failures:
            console.print(f"[bold red]Completed with {failures} failure(s).[/]")
            sys.exit(1)

        console.print("[bold green]All processing completed successfully.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error during processing:[/] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
