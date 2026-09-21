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

**Documents & web:** [`pdf2md.py`](#pdf2mdpy) · [`png2md.py`](#png2mdpy) · [`url2md.py`](#url2mdpy) · [`slides2pdf.py`](#slides2pdfpy) · [`html2png.py`](#html2pngpy)

**Code & tokens:** [`flowchart.py`](#flowchartpy) · [`count_tokens.py`](#count_tokenspy)

**Bibliography:** [`merge_bib.py`](#merge_bibpy) · [`scopus_rank.py`](#scopus_rankpy)

**Audio & video:** [`audio_transcribe.py`](#audio_transcribepy) · [`audio_transcribe_gui.py`](#audio_transcribe_guipy) · [`md2audio.py`](#md2audiopy) · [`youtube2audio.py`](#youtube2audiopy) · [`demucs_separate.py`](#demucs_separatepy) · [`mix_audio.py`](#mix_audiopy)

---

### `pdf2md.py`

PDF → Markdown via Mistral OCR (cloud) or local engines.

- Accepts either a single PDF file or a directory containing PDFs.
- Engines (`--model`): `mistral` (default, cloud), `dots` (local, best quality — on par with Mistral for
  math), `paddle` (local, fastest), `nougat` (legacy). Local engines need a running server:
  `tools/ocr_server.sh start {dots|paddle}`. Setup and benchmarks: [docs/local_ocr.md](docs/local_ocr.md).
- If `--include-images`, images save to `<output_dir>/<pdf_stem>_images/` and links are rewritten.
- If `--save-ocr-json`, the full OCR response saves to `<pdf_stem>.ocr.json`.
- If the target `.md` already exists, that PDF is skipped and processing continues.
- **Flags:** `-y/--yes`, `--include-images`, `-o/--output`, `--no-preview`, `--pages`, `--server-url`
  (local engines).
- **Mistral OCR 4 flags:** `--mistral-model`, `--include-blocks`, `--confidence-scores`, `--table-format`,
  `--extract-header`, `--extract-footer`, `--save-ocr-json`.

```bash
python tools/pdf2md.py file.pdf --yes --include-images -o file.md
python tools/pdf2md.py ./pdfs --yes
python tools/pdf2md.py file.pdf --pages 1-5 -o excerpt.md
python tools/pdf2md.py file.pdf --yes --mistral-model mistral-ocr-4-0 --include-blocks --save-ocr-json
tools/ocr_server.sh start dots && python tools/pdf2md.py paper.pdf --model dots -y
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

Markdown → narration WAV via local Qwen3-TTS CustomVoice.

- Requires explicit `--language en` or `--language pt-br`.
- Defaults to `models/Qwen3-TTS-12Hz-0.6B-CustomVoice`, speaker `ryan`, and output path `<input>.wav`.
- Reports elapsed time from model load through final WAV write, plus audio duration and realtime factor.
- If `qwen-tts` warns that SoX is missing, install the system binary with `sudo apt install sox`.
- **Flags:** `-o/--output`, `--language`, `--model`, `--speaker`, `--instruct`, `--device`, `--chunk-chars`,
  `--batch-size`, `--dry-run`, `--keep-temp`, `--flash-attention`.

Manual setup:
```bash
pip install -U qwen-tts soundfile torch
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

```bash
python tools/md2audio.py article.md --language en --dry-run
python tools/md2audio.py article.md --language pt-br -o article.wav --batch-size 2
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

- [Guia de Boas Práticas e Avaliação de Skills (Evals)](docs/skills_best_practices_and_evals.md) — Boas práticas de autoria e framework de evals para skills de IA (baseado em Philipp Schmid / Google DeepMind).
- [Claude Skills Reference](docs/claude_skills.md) — Referência estendida para criação e uso de skills no Claude Code.
