# Mistral OCR

## OCR 4 overview

Mistral OCR 4 is available through the same OCR endpoint used by this repository:

```text
POST https://api.mistral.ai/v1/ocr
```

Use `mistral-ocr-latest` to track the latest hosted OCR model, or pass an explicit
OCR 4 model ID such as `mistral-ocr-4-0` when you want more reproducible output.

OCR 4 adds structured document signals on top of Markdown text extraction:

- Paragraph-level bounding boxes and structural block labels via `include_blocks`.
- Page-level or word-level confidence scores via `confidence_scores_granularity`.
- Table extraction as separate Markdown or HTML artifacts via `table_format`.
- Header and footer extraction via `extract_header` and `extract_footer`.
- Image extraction with base64 payloads via `include_image_base64`.

The plain Markdown path remains the default output in `tools/pdf2md.py`. Use
`--save-ocr-json` when you need the full structured OCR response for blocks,
coordinates, confidence scores, tables, hyperlinks, or headers/footers.

## CLI examples

Basic Markdown conversion through the latest Mistral OCR model:

```bash
python tools/pdf2md.py file.pdf --yes --model mistral
```

Pin OCR 4 and save its structured response:

```bash
python tools/pdf2md.py file.pdf --yes --model mistral \
  --mistral-model mistral-ocr-4-0 \
  --include-blocks \
  --confidence-scores page \
  --save-ocr-json
```

Extract images and rewrite Markdown image links:

```bash
python tools/pdf2md.py file.pdf --yes --model mistral --include-images
```

Extract tables separately:

```bash
python tools/pdf2md.py file.pdf --yes --model mistral \
  --table-format markdown \
  --save-ocr-json
```

## API request shape

`tools/pdf2md.py` uploads local PDFs with the Mistral SDK to get a signed URL, then
calls the OCR endpoint directly with the Python standard library so OCR 4 parameters
are available even if the installed SDK has not exposed them yet.

```python
import os
import json
from urllib import request as urlrequest

payload = {
    "model": "mistral-ocr-latest",
    "document": {
        "type": "document_url",
        "document_url": signed_url,
    },
    "include_blocks": True,
    "confidence_scores_granularity": "page",
    "table_format": "markdown",
}

request = urlrequest.Request(
    "https://api.mistral.ai/v1/ocr",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    method="POST",
)

with urlrequest.urlopen(request, timeout=600) as response:
    ocr_response = json.loads(response.read().decode("utf-8"))
```

## Output files

- Markdown: written to `<pdf_stem>.md` or the path provided by `--output`.
- Images: written to `<output_dir>/<pdf_stem>_images/` when `--include-images` is set.
- OCR JSON: written to `<pdf_stem>.ocr.json` when `--save-ocr-json` is set.

## Notes

- OCR mode requires `MISTRAL_API_KEY` in `.env`.
- The Mistral API may incur per-page costs.
- `--pages` is supported for a single PDF by creating a temporary PDF containing the selected pages before upload.
- Directory mode processes direct child `.pdf` files and skips outputs that already exist.
