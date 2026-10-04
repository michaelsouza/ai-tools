# AI Tools

CLI helpers for PDFs, web pages, audio, bibliography, and token counts.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

## Configure
Create `.env` in repo root for Mistral OCR and OpenRouter (`scopus_rank.py`):
```bash
MISTRAL_API_KEY=your_mistral_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Tools

**Documents & web:** [`pdf2md.py`](#pdf2mdpy) · [`png2md.py`](#png2mdpy) · [`url2md.py`](#url2mdpy) · [`slides2pdf.py`](#slides2pdfpy) · [`html2png.py`](#html2pngpy) · [`pdf_review.py`](#pdf_reviewpy)

**Code & tokens:** [`flowchart.py`](#flowchartpy) · [`count_tokens.py`](#count_tokenspy)

**Bibliography:** [`merge_bib.py`](#merge_bibpy) · [`scopus_rank.py`](#scopus_rankpy)

**Audio & video:** [`audio_transcribe.py`](#audio_transcribepy) · [`audio_transcribe_gui.py`](#audio_transcribe_guipy) · [`md2audio.py`](#md2audiopy) · [`youtube2audio.py`](#youtube2audiopy) · [`demucs_separate.py`](#demucs_separatepy) · [`mix_audio.py`](#mix_audiopy)

---

### `pdf2md.py`

PDF → Markdown via Mistral OCR (cloud) or LightOnOCR-2 (local).

Which engine to use, the local OCR benchmark and lessons learned (including the Mistral rate-limit incident):
**[docs/pdf2md.md](docs/pdf2md.md)**.

- Accepts either a single PDF file or a directory containing PDFs.
- Engines (`--model`): `mistral` (default, cloud) or `lighton` (local LightOnOCR-2 via llama.cpp, ~6 s/page with 4 pages in parallel on an RTX 4070 Laptop; `--workers N`).
  The local engine needs its server: `tools/ocr_server.sh start lighton`. Setup: [docs/local_ocr.md](docs/local_ocr.md).
- If `--include-images`, images save to `<output_dir>/<pdf_stem>_images/` and links are rewritten.
- If `--save-ocr-json`, the full OCR response saves to `<pdf_stem>.ocr.json`.
- If the target `.md` already exists, that PDF is skipped and processing continues.
- Mistral mode uploads the PDF to Mistral's file storage and deletes it right after the OCR call, whether
  the call succeeds or fails (a failed delete only prints a warning).
- Mistral HTTP 429/5xx are retried up to 5 times with backoff (honours `Retry-After`, max 60 s). A 429 whose
  `x-ratelimit-limit-req-minute` is `0` means inference is disabled for the workspace (billing, plan or
  limits at [console.mistral.ai](https://console.mistral.ai/)): the run stops at once with exit code 2,
  without uploading the remaining PDFs of a batch.
- **Flags:** `-y/--yes`, `--include-images`, `-o/--output`, `--no-preview`, `--pages`, `--server-url`
  (`--model lighton`).
- **Mistral OCR 4 flags:** `--mistral-model`, `--include-blocks`, `--confidence-scores`, `--table-format`,
  `--extract-header`, `--extract-footer`, `--save-ocr-json`.

```bash
python tools/pdf2md.py file.pdf --yes --include-images -o file.md
python tools/pdf2md.py ./pdfs --yes
python tools/pdf2md.py file.pdf --pages 1-5 -o excerpt.md
python tools/pdf2md.py file.pdf --yes --mistral-model mistral-ocr-4-0 --include-blocks --save-ocr-json
tools/ocr_server.sh start lighton && python tools/pdf2md.py paper.pdf --model lighton -y
```

### `png2md.py`

Image screenshots → Markdown via local OCR (EasyOCR), no API key or internet needed.

- Accepts a single image or a directory; each image becomes a section with its text in a fenced code block.
- Designed for Linux terminal screenshots: monospaced text, high contrast.
- Supported formats: PNG, JPG/JPEG, WEBP, BMP, TIFF. EasyOCR downloads its model (~50 MB) on first run.
- **Flags:** `-y/--yes`, `-o/--output`, `--no-preview`, `--min-confidence` (default `0.3`).

```bash
python tools/png2md.py ./screenshots -y
python tools/png2md.py screenshot.png -o notes.md
python tools/png2md.py ./screenshots --min-confidence 0.5
```

### `url2md.py`

Web page → Markdown.

- **Flags:** `-o/--output`, `--save-html`, `--save-clean-html`.

```bash
python tools/url2md.py https://example.com/article -o article.md
```

### `slides2pdf.py`

HTML slides → PDF.

- Captures each slide from an HTML file using Playwright and saves as a PDF.

```bash
python tools/slides2pdf.py
```

### `html2png.py`

HTML → PNG with automatic whitespace cropping.

- **Flags:** `-s/--scale`, `--no-crop`, `-p/--padding`, `--selector`, `-b/--background`, `-q/--quiet`.

```bash
python tools/html2png.py diagram.html output.png -s 3
```

### `pdf_review.py`

Comment on a PDF in the browser: select a passage and the comment box opens.

- Left panel: the PDF (pdf.js). Right panel: the comment box and the list of comments, sorted by page and
  position. Click a comment to jump to it; edit or delete it from the list. In a window taller than wide (a
  portrait monitor) the comments go below the PDF instead. Drag the bar between them to resize (remembered by the
  browser, separately for each layout); double-click it to reset.
- **Enter** saves, **Shift+Enter** breaks the line, **Esc** cancels. Outside the comment box, **←** and **→** go to
  the previous and next page.
- Comments are saved into the PDF itself as highlight annotations with a note, so Okular, Acrobat and any
  script that reads PDF annotations see them.
- The PDF is written in place. When another program rewrites it (a LaTeX or Quarto build), the page reloads
  and the comments in the old file are gone. `--copy` comments on a copy instead (default
  `./<name>-comments.pdf`); a later run reuses that copy, so a review can span several sessions and builds.
- **Earlier comments** stay below the current ones, in another colour and grouped by round. A journal next to the
  PDF (`<pdf>.comments.json`, or `--journal PATH`) records every comment the page has shown, so comments erased by
  a rebuild move there. `--history` adds the comments of older annotated PDFs. Each earlier comment is looked up in
  the current PDF by its marked words: *unchanged* (highlighted, click to jump), *passage changed* (rewritten or
  removed), or *appears N times*. An unchanged passage hints that the comment is still open, but a comment can also
  be answered elsewhere in the text.
- Needs internet access to load pdf.js from jsDelivr.
- **Flags:** `--copy [PATH]`, `--history PDF...`, `--journal PATH`, `--port` (default 8765), `--author` (default: login
  name), `--no-browser`.

```bash
python tools/pdf_review.py paper.pdf
uv run --with pymupdf python tools/pdf_review.py paper.pdf --author "Ana"
python tools/pdf_review.py _output/paper.pdf --copy   # comments go to ./paper-comments.pdf
python tools/pdf_review.py _output/paper.pdf --history paper-commented-*.pdf
```

### `flowchart.py`

Source code → flowchart PNG (Python, C, C++).

- Analyzes function calls and generates a call graph using Graphviz.
- **Flags:** `--no-images`, `--json`, `--svg`, `--print-dot`.

```bash
python tools/flowchart.py script.py --svg
```

### `count_tokens.py`

Token counts for files/dirs.

- **Flags:** `-e/--encoding` (default `o200k_base`; also `cl100k_base`, `p50k_base`, `r50k_base`,
  `p50k_edit`), `-a/--all`.

```bash
python tools/count_tokens.py . -e cl100k_base
```

### `merge_bib.py`

Merge two BibTeX files.

- Keeps entries from the first file first, adds non-duplicates from the second file, and deduplicates by
  key, DOI, and normalized title by default.
- Same-key conflicts with different content are renamed by default using `_2`.
- **Flags:** `-o/--output`, `--dedupe-by`, `--on-key-conflict`, `--rename-suffix`, `--dry-run`.

```bash
tools/merge_bib.py project_a.bib project_b.bib -o merged.bib
tools/merge_bib.py project_a.bib project_b.bib --dry-run
tools/merge_bib.py project_a.bib project_b.bib --on-key-conflict keep-first -o merged.bib
```

### `scopus_rank.py`

Rank a Scopus CSV export against a research topic with Jev (OpenRouter).

Jev (`~typesafe/jev-latest`) is a decision model, not a text LLM: it cannot write a justification. For each
reference it receives a state (topic, title, abstract, keywords, year, source) plus the typed questions of a
screening profile and returns typed answers with probabilities. One request per reference answers all
questions; ~650 input tokens each at US$ 0.042 / 1M tokens (5,000 references ≈ US$ 0.15). Needs
`OPENROUTER_API_KEY` with credits (no free tier).

#### Workflow

1. Run the query in Scopus and export as **CSV**, ticking abstract and keywords besides the citation fields.
2. Copy [docs/scopus_rank_profile.example.json](docs/scopus_rank_profile.example.json) and edit it (see below).
3. `--dry-run` to inspect one payload and the token/cost estimate (no network).
4. `--limit 20` to check that the top of the ranking makes sense; adjust the profile and repeat.
5. Run on the full CSV.

```bash
tools/scopus_rank.py scopus.csv --profile profile.json --dry-run
tools/scopus_rank.py scopus.csv --profile profile.json --limit 20
tools/scopus_rank.py scopus.csv --profile profile.json -o ranked.csv --workers 16
```

#### Screening profile

```json
{
  "topic": "free-text description of the research: what is in scope and what is not",
  "rank_by": "relevance",
  "questions": {
    "relevance": {"type": "score", "instructions": "How relevant is this reference to research_topic?",
                  "criteria": ["unrelated: ...", "marginal: ...", "related: ...", "close: ...", "core: ..."]},
    "proposes_method": {"type": "noul", "instructions": "Does the reference propose an algorithm for research_topic?"},
    "kind": {"type": "choice", "instructions": "Main contribution?",
             "criteria": {"theory": "...", "algorithm": "...", "application": "...", "other": "..."}}
  }
}
```

- `topic` is sent in the state as `research_topic`; refer to it by that name in the instructions.
- `score`: `criteria` is an **ordered list** of levels, worst to best. The answer is the expected level
  index, a continuous value in `[0, levels - 1]` (0–4 with five levels). The level descriptions are the only
  rubric the model sees. Three levels tend to saturate and produce ties; five gave a usable spread.
- `noul`: yes/no question, no `criteria`. The answer is P(yes) in `[0, 1]`.
- `choice`: `criteria` is an object `label: description`. The answer is the chosen label.
- `rank_by` names the `score` or `noul` question that orders the output.

#### Output and cache

- `<stem>.ranked.csv` (or `-o`) = original CSV preceded by `jev_rank` and one `jev_<question>` column per
  question, plus `jev_<question>_conf` (confidence) and, for `choice`, `jev_<question>_p` (probability of the
  chosen label). Sorted by the `rank_by` column, descending; ties broken by `Cited by`; references that
  failed go last. Citations never enter the score. A summary table shows the top 15, the score distribution,
  and tokens/cost actually billed.
- Answers are appended to `<stem>.jev.jsonl` next to the input CSV, keyed by EID → DOI → normalized title.
  Re-running only requests what is missing, so an interrupted or partially failed run (exit code 1) is
  resumed by running the same command again. Changing the topic, the questions or `--model` invalidates the
  cache. Duplicate references in the CSV are requested once.
- References without abstract are scored from title and keywords only (`no_abstract: true` in the state);
  expect lower confidence for them.

#### Reference

- **Flags:** `--profile` (required), `-o/--output`, `--model` (default `~typesafe/jev-latest`),
  `--workers` (default 8), `--limit N`, `--dry-run`, `--no-cache`.
- **Exit codes:** `0` ok, `1` some references could not be scored, `2` configuration error.
- The OpenRouter decisions endpoint (`/api/alpha/decisions`) is alpha and may change; response parsing is
  isolated in `parse_answers`.

### `audio_transcribe.py`

Real-time or batch audio transcription via Faster-Whisper.

- **Flags:** `-i/--input`, `-o/--output`, `-m/--model`, `-d/--device`, `-l/--language`, `--timestamps`,
  `-v/--verbose`.

```bash
python tools/audio_transcribe.py                            # Real-time mic
python tools/audio_transcribe.py -i audio.mp3 -o out.txt    # Batch file
```

### `audio_transcribe_gui.py`

GUI for real-time audio transcription.

- Provides a customtkinter interface with model/language selection, recording controls, and clipboard copy.

```bash
python tools/audio_transcribe_gui.py
```

### `md2audio.py`

Markdown → narration WAV via Gemini TTS on OpenRouter (default) or local Qwen3-TTS CustomVoice.

Engine choice comes from the pt-BR benchmark ([benchmarks/tts/README.md](benchmarks/tts/README.md)):
in blind listening Gemini 3.8 Flash-Lite TTS scored 4.95/5 and the best local model 3.25–3.40.

- Requires explicit `--language en` or `--language pt-br` (used for the "omitted" cues; Gemini detects the
  spoken language itself).
- `--engine gemini` (default): `google/gemini-3.8-flash-lite-tts`, voice `Kore`, `OPENROUTER_API_KEY` from the
  repo `.env`. Chunks go out as parallel requests (`--workers`, default 4) with retries on 429/5xx. Roughly
  US$ 0.60 per hour of audio; `--dry-run` prints the character count. Other voices: `Leda`, `Aoede`, `Puck`,
  `Charon`… (30 in all, listed in the model's `supported_voices` on OpenRouter). No `--instruct`.
- `--engine qwen`: local GPU, defaults to `models/Qwen3-TTS-12Hz-0.6B-CustomVoice`, speaker `ryan`.
  If `qwen-tts` warns that SoX is missing, install the system binary with `sudo apt install sox`.
- Output defaults to `<input>.wav`. Reports elapsed time, audio duration and realtime factor.
- **Flags:** `-o/--output`, `--language`, `--engine`, `--model`, `--speaker`, `--instruct` (qwen),
  `--device` (qwen), `--chunk-chars`, `--batch-size` (qwen), `--workers` (gemini), `--dry-run`, `--keep-temp`,
  `--flash-attention` (qwen).

Setup for `--engine qwen`:
```bash
pip install -U qwen-tts soundfile torch
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

```bash
python tools/md2audio.py article.md --language pt-br --dry-run
python tools/md2audio.py article.md --language pt-br -o article.wav
python tools/md2audio.py article.md --language pt-br --speaker Leda
python tools/md2audio.py article.md --language en --engine qwen --speaker Aiden --instruct "Calm lecture style."
```

### `youtube2audio.py`

Download audio from YouTube videos/playlists.

- **Flags:** `-o/--output`, `-f/--format` (mp3, m4a, opus, flac, wav), `-q/--quality`, `-t/--template`,
  `-v/--verbose`.

```bash
python tools/youtube2audio.py "https://www.youtube.com/watch?v=ID" -o ~/Music -f m4a
```

### `demucs_separate.py`

Separate music into stems (vocals, drums, bass, other).

- **Flags:** `-o/--output`, `-m/--model`, `--two-stems`, `--mp3`, `--mp3-bitrate`, `-d/--device`, `-j/--jobs`.

```bash
python tools/demucs_separate.py song.mp3 --two-stems vocals --mp3
```

### `mix_audio.py`

Mix multiple WAV tracks with volume adjustment (dB).

- **Flags:** `-o/--output`.

```bash
python tools/mix_audio.py -o mixed.wav track1.wav:0 track2.wav:-3 track3.wav:+2
```

## Documentation

- [pdf2md — engines, decisions and lessons learned](docs/pdf2md.md) — Escolha do motor de OCR (benchmark local de 2026-09-22), falhas observadas e o incidente de rate limit do Mistral.
- [Local OCR setup](docs/local_ocr.md) — Instalação e operação do servidor local do LightOnOCR-2 (llama.cpp).
- [OCR formula benchmark](benchmarks/ocr/README.md) — Método, GT e scripts para reavaliar motores de OCR.
- [Guia de Boas Práticas e Avaliação de Skills (Evals)](docs/skills_best_practices_and_evals.md) — Boas práticas de autoria e framework de evals para skills de IA (baseado em Philipp Schmid / Google DeepMind).
- [Claude Skills Reference](docs/claude_skills.md) — Referência estendida para criação e uso de skills no Claude Code.
