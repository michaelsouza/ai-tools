# AI Tools

CLI helpers for PDFs, web pages, and token counts.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

## Configure
Create `.env` in repo root for Mistral OCR:
```bash
MISTRAL_API_KEY=your_mistral_api_key
```

## Tools

- `pdf2md.py` — PDF → Markdown via Mistral OCR (no local fallback)
  - Accepts either a single PDF file or a directory containing PDFs
  - Flags: `-y/--yes`, `--include-images`, `-o/--output`, `--no-preview`
  - Example:
    ```bash
    python pdf2md.py file.pdf --yes --include-images -o file.md
    ```
  - Batch example:
    ```bash
    python pdf2md.py ./pdfs --yes
    ```
  - If `--include-images`, images save to `<output_dir>/<pdf_stem>_images/` and links are rewritten.
  - If the target `.md` already exists, that PDF is skipped and processing continues.

- `url2md.py` — Web page → Markdown
  - Flags: `-o/--output`, `--save-html`, `--save-clean-html`
  - Example:
    ```bash
    python url2md.py https://example.com/article -o article.md
    ```

- `slides2pdf.py` — HTML Slides → PDF
  - Captures each slide from an HTML file using Playwright and saves as a PDF.
  - Example:
    ```bash
    python tools/slides2pdf.py
    ```

- `count_tokens.py` — Token counts for files/dirs
  - Flag: `-e/--encoding` (default `o200k_base`; also `cl100k_base`, `p50k_base`, `r50k_base`, `p50k_edit`)
  - Example:
    ```bash
    python count_tokens.py . -e cl100k_base
    ```
