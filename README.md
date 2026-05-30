# AI Tools

CLI helpers for PDFs, web pages, audio, and token counts.

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

- `pdf2md.py` — PDF → Markdown via Mistral OCR
  - Accepts either a single PDF file or a directory containing PDFs
  - Flags: `-y/--yes`, `--include-images`, `-o/--output`, `--no-preview`, `--pages`
  - Example:
    ```bash
    python tools/pdf2md.py file.pdf --yes --include-images -o file.md
    python tools/pdf2md.py ./pdfs --yes
    python tools/pdf2md.py file.pdf --pages 1-5 -o excerpt.md
    ```
  - If `--include-images`, images save to `<output_dir>/<pdf_stem>_images/` and links are rewritten.
  - If the target `.md` already exists, that PDF is skipped and processing continues.

- `url2md.py` — Web page → Markdown
  - Flags: `-o/--output`, `--save-html`, `--save-clean-html`
  - Example:
    ```bash
    python tools/url2md.py https://example.com/article -o article.md
    ```

- `slides2pdf.py` — HTML Slides → PDF
  - Captures each slide from an HTML file using Playwright and saves as a PDF.
  - Example:
    ```bash
    python tools/slides2pdf.py
    ```

- `html2png.py` — HTML → PNG with automatic whitespace cropping
  - Flags: `-s/--scale`, `--no-crop`, `-p/--padding`, `--selector`, `-b/--background`, `-q/--quiet`
  - Example:
    ```bash
    python tools/html2png.py diagram.html output.png -s 3
    ```

- `generate_flowchart.py` — Source code → flowchart PNG (Python, C, C++)
  - Analyzes function calls and generates a call graph using Graphviz.
  - Flags: `--no-images`, `--json`, `--svg`, `--print-dot`
  - Example:
    ```bash
    python tools/generate_flowchart.py script.py --svg
    ```

- `count_tokens.py` — Token counts for files/dirs
  - Flags: `-e/--encoding` (default `o200k_base`; also `cl100k_base`, `p50k_base`, `r50k_base`, `p50k_edit`), `-a/--all`
  - Example:
    ```bash
    python tools/count_tokens.py . -e cl100k_base
    ```

- `count_abstract_tokens.py` — Token counts for BibTeX abstract fields
  - Example:
    ```bash
    python tools/count_abstract_tokens.py references.bib
    ```

- `merge_bib.py` — Merge two BibTeX files
  - Keeps entries from the first file first, adds non-duplicates from the second file, and deduplicates by key, DOI, and normalized title by default.
  - Same-key conflicts with different content are renamed by default using `_2`.
  - Flags: `-o/--output`, `--dedupe-by`, `--on-key-conflict`, `--rename-suffix`, `--dry-run`
  - Example:
    ```bash
    tools/merge_bib.py project_a.bib project_b.bib -o merged.bib
    tools/merge_bib.py project_a.bib project_b.bib --dry-run
    tools/merge_bib.py project_a.bib project_b.bib --on-key-conflict keep-first -o merged.bib
    ```

- `audio_transcribe.py` — Real-time or batch audio transcription via Faster-Whisper
  - Flags: `-i/--input`, `-o/--output`, `-m/--model`, `-d/--device`, `-l/--language`, `--timestamps`, `-v/--verbose`
  - Example:
    ```bash
    python tools/audio_transcribe.py                            # Real-time mic
    python tools/audio_transcribe.py -i audio.mp3 -o out.txt    # Batch file
    ```

- `md2audio.py` — Markdown → narration WAV via local Qwen3-TTS CustomVoice
  - Requires explicit `--language en` or `--language pt-br`
  - Defaults to `models/Qwen3-TTS-12Hz-0.6B-CustomVoice`, speaker `ryan`, and output path `<input>.wav`
  - Flags: `-o/--output`, `--language`, `--model`, `--speaker`, `--instruct`, `--device`,
    `--chunk-chars`, `--batch-size`, `--dry-run`, `--keep-temp`, `--flash-attention`
  - Reports elapsed time from model load through final WAV write, plus audio duration and realtime factor
  - Manual setup:
    ```bash
    pip install -U qwen-tts soundfile torch
    huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
    ```
  - If `qwen-tts` warns that SoX is missing, install the system binary with `sudo apt install sox`
  - Example:
    ```bash
    python tools/md2audio.py article.md --language en --dry-run
    python tools/md2audio.py article.md --language pt-br -o article.wav --batch-size 2
    ```

- `audio_transcribe_gui.py` — GUI for real-time audio transcription
  - Provides a customtkinter interface with model/language selection, recording controls, and clipboard copy.
  - Example:
    ```bash
    python tools/audio_transcribe_gui.py
    ```

- `youtube2audio.py` — Download audio from YouTube videos/playlists
  - Flags: `-o/--output`, `-f/--format` (mp3, m4a, opus, flac, wav), `-q/--quality`, `-t/--template`, `-v/--verbose`
  - Example:
    ```bash
    python tools/youtube2audio.py "https://www.youtube.com/watch?v=ID" -o ~/Music -f m4a
    ```

- `demucs_separate.py` — Separate music into stems (vocals, drums, bass, other)
  - Flags: `-o/--output`, `-m/--model`, `--two-stems`, `--mp3`, `--mp3-bitrate`, `-d/--device`, `-j/--jobs`
  - Example:
    ```bash
    python tools/demucs_separate.py song.mp3 --two-stems vocals --mp3
    ```

- `mix_audio.py` — Mix multiple WAV tracks with volume adjustment (dB)
  - Flags: `-o/--output`
  - Example:
    ```bash
    python tools/mix_audio.py -o mixed.wav track1.wav:0 track2.wav:-3 track3.wav:+2
    ```
