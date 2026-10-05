#!/usr/bin/env python3
"""
Markdown to narration audio using Gemini TTS (OpenRouter, default) or local Qwen3-TTS CustomVoice.

Usage:
    python tools/md2audio.py article.md --language pt-br
    python tools/md2audio.py article.md --language pt-br -o article.wav --speaker Leda
    python tools/md2audio.py article.md --language en --dry-run
    python tools/md2audio.py article.md --language en --engine qwen --speaker Aiden --instruct "Calm lecture style."

Setup:
    gemini (default): OPENROUTER_API_KEY in the repo .env; pip install requests python-dotenv
    qwen (local GPU):
        pip install -U qwen-tts soundfile torch
        huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
        huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice

Engine choice (pt-BR benchmark, blind listening 1-5): Gemini 3.8 Flash-Lite TTS 4.95, Qwen3-TTS 1.7B 3.25;
see benchmarks/tts/README.md.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ENGINES = ("gemini", "qwen")
DEFAULT_ENGINE = "gemini"
GEMINI_MODEL = "google/gemini-3.8-flash-lite-tts"
GEMINI_SPEAKER = "Kore"
GEMINI_SAMPLE_RATE = 24000  # OpenRouter returns Gemini audio as raw PCM: 24 kHz, mono, 16-bit little-endian
OPENROUTER_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
GEMINI_RETRIES = 4
DEFAULT_WORKERS = 4
RECOMMENDED_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_MODEL = "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_SPEAKER = "ryan"
DEFAULT_CHUNK_CHARS = 900
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "audio"  # git-ignored; see AGENTS.md "Generated files"
DEFAULT_BATCH_SIZE = 2
LANGUAGE_NAMES = {
    "en": "English",
    "pt-br": "Portuguese",
}
OMISSION_CUES = {
    "en": {
        "code": "Code block omitted.",
        "table": "Table omitted.",
        "html": "HTML block omitted.",
        "image": "Image omitted.",
        "footnote": "Footnote omitted.",
    },
    "pt-br": {
        "code": "Bloco de codigo omitido.",
        "table": "Tabela omitida.",
        "html": "Bloco HTML omitido.",
        "image": "Imagem omitida.",
        "footnote": "Nota de rodape omitida.",
    },
}


class MarkdownNarrationError(Exception):
    """Raised when narration input cannot be prepared."""


@dataclass
class SynthesisMetrics:
    chunks: int
    batches: int
    load_seconds: float
    synthesis_seconds: float
    concat_seconds: float
    elapsed_seconds: float
    audio_seconds: float

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def realtime_factor(self) -> float:
        return self.audio_seconds / self.elapsed_seconds if self.elapsed_seconds else 0.0


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def omission(kind: str, language: str) -> str:
    return OMISSION_CUES[language][kind]


def preprocess_markdown(markdown_text: str, language: str) -> str:
    """Replace footnote definitions with cues before Markdown parsing."""
    lines = []
    in_footnote = False
    footnote_re = re.compile(r"^\[\^[^\]]+\]:")

    for line in markdown_text.splitlines():
        if footnote_re.match(line):
            lines.append(omission("footnote", language))
            in_footnote = True
            continue
        if in_footnote and (line.startswith("    ") or line.startswith("\t") or not line.strip()):
            continue
        in_footnote = False
        lines.append(line)

    return "\n".join(lines)


def inline_text(token: Token, language: str) -> Tuple[str, List[str]]:
    """Return speakable inline text and omission cues for unsupported inline content."""
    if not token.children:
        return normalize_space(token.content), []

    parts = []
    cues = []
    skip_link_destination = 0
    for child in token.children:
        if child.type == "image":
            cues.append(omission("image", language))
        elif child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append(" ")
        elif child.type == "hardbreak":
            parts.append(". ")
        elif child.type == "link_open":
            skip_link_destination += 1
        elif child.type == "link_close":
            skip_link_destination = max(0, skip_link_destination - 1)
        elif child.children:
            text, nested_cues = inline_text(child, language)
            parts.append(text)
            cues.extend(nested_cues)
        elif child.content and not skip_link_destination:
            parts.append(child.content)

    text = normalize_space("".join(parts))
    return text, cues


def looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    divider = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
    return "|" in lines[0] and bool(divider.match(lines[1]))


def markdown_to_blocks(markdown_text: str, language: str) -> List[str]:
    """Convert Markdown into speakable blocks and omission cues."""
    markdown_text = preprocess_markdown(markdown_text, language)
    if looks_like_table(markdown_text):
        # markdown-it table support catches normal tables; this covers all-table documents.
        markdown_text = re.sub(
            r"(?ms)^\s*\|.+?\|\s*\n\s*\|?\s*:?-{3,}:?.*?(?=\n\n|\Z)",
            omission("table", language),
            markdown_text,
        )

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = md.parse(markdown_text)
    blocks: List[str] = []
    in_table = False
    table_emitted = False

    for token in tokens:
        if token.type == "table_open":
            in_table = True
            table_emitted = False
            continue
        if token.type == "table_close":
            if not table_emitted:
                blocks.append(omission("table", language))
            in_table = False
            continue
        if in_table:
            if not table_emitted:
                blocks.append(omission("table", language))
                table_emitted = True
            continue

        if token.type in {"fence", "code_block"}:
            blocks.append(omission("code", language))
        elif token.type == "html_block":
            blocks.append(omission("html", language))
        elif token.type == "inline":
            text, cues = inline_text(token, language)
            if text:
                text = re.sub(r"\[\^[^\]]+\]", "", text)
                text = normalize_space(text)
                if text:
                    blocks.append(text)
            blocks.extend(cues)

    return dedupe_adjacent_blocks(blocks)


def dedupe_adjacent_blocks(blocks: Iterable[str]) -> List[str]:
    result = []
    for block in blocks:
        block = normalize_space(block)
        if block and (not result or result[-1] != block):
            result.append(block)
    return result


def split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(sentence[i : i + max_chars].strip() for i in range(0, len(sentence), max_chars))
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_blocks(blocks: Sequence[str], max_chars: int) -> List[str]:
    chunks = []
    current = ""
    for block in blocks:
        block_parts = split_long_text(block, max_chars)
        for part in block_parts:
            if len(current) + len(part) + 2 <= max_chars:
                current = f"{current}\n\n{part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
    if current:
        chunks.append(current)
    return chunks


def default_output_path(input_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}.wav"


def keep_source_next_to_audio(input_path: Path, output_path: Path) -> Optional[Path]:
    """Copy the narrated Markdown beside the audio (same base name) unless it is already there."""
    source_copy = output_path.with_suffix(input_path.suffix)
    if source_copy.resolve() == input_path.resolve():
        return None
    shutil.copy2(input_path, source_copy)
    return source_copy


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --device auto. Install torch or pass --device cpu.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu to run without GPU.")
    return "cuda"


def load_qwen_model(model_name_or_path: str, device: str, use_flash_attention: bool):
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-TTS dependencies are missing. Install them with: pip install -U qwen-tts soundfile torch"
        ) from exc

    kwargs = {
        "device_map": "cuda:0" if device == "cuda" else "cpu",
        "dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    if use_flash_attention and device == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"

    return Qwen3TTSModel.from_pretrained(model_name_or_path, **kwargs)


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    model_path = Path(model_name_or_path).expanduser()
    if model_name_or_path == DEFAULT_MODEL and not model_path.exists():
        raise RuntimeError(
            "Default Qwen3-TTS model is not downloaded. Run:\n"
            f"  huggingface-cli download {RECOMMENDED_MODEL_ID} --local-dir {DEFAULT_MODEL}\n"
            "Or pass --model with an existing local path or explicit Hugging Face model id."
        )
    if model_path.exists():
        return str(model_path.resolve())
    return model_name_or_path


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def gemini_speech(text: str, model: str, voice: str, api_key: str) -> bytes:
    """One OpenRouter TTS request; retries rate limits, server errors and dropped connections."""
    import requests

    body = {"model": model, "input": text, "voice": voice, "response_format": "pcm"}
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            response = requests.post(OPENROUTER_SPEECH_URL, headers=headers, json=body, timeout=300)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == GEMINI_RETRIES:
                raise
        else:
            if response.ok and response.content:
                return response.content
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == GEMINI_RETRIES:
                raise RuntimeError(f"OpenRouter TTS failed (HTTP {response.status_code}): {response.text[:300]}")
        time.sleep(2**attempt)
    raise RuntimeError("OpenRouter TTS failed after retries.")


def synthesize_gemini(
    chunks: Sequence[str],
    temp_dir: Path,
    model: str,
    speaker: str,
    workers: int,
    advance,
) -> Tuple[List[Path], float, int]:
    """Synthesize chunks with parallel OpenRouter requests; returns (chunk paths, load seconds, requests)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (repo .env). Or pass --engine qwen to run locally.")

    chunk_paths = [temp_dir / f"chunk_{index:04d}.wav" for index in range(1, len(chunks) + 1)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(gemini_speech, chunk, model, speaker, api_key): path for chunk, path in zip(chunks, chunk_paths)
        }
        for future in as_completed(futures):
            write_pcm_wav(futures[future], future.result(), GEMINI_SAMPLE_RATE)
            advance(1)
    return chunk_paths, 0.0, len(chunks)


def synthesize_qwen(
    chunks: Sequence[str],
    temp_dir: Path,
    language: str,
    model_name_or_path: str,
    speaker: str,
    instruct: Optional[str],
    device: str,
    batch_size: int,
    use_flash_attention: bool,
    advance,
) -> Tuple[List[Path], float, int]:
    """Synthesize chunks with local Qwen3-TTS CustomVoice; returns (chunk paths, load seconds, batches)."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required to write Qwen3-TTS audio. Install it with: pip install soundfile"
        ) from exc

    resolved_device = resolve_device(device)
    load_started_at = time.perf_counter()
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    model = load_qwen_model(resolved_model, resolved_device, use_flash_attention)
    load_seconds = time.perf_counter() - load_started_at
    supported_speakers = getattr(model, "get_supported_speakers", lambda: [])()
    if supported_speakers and speaker not in supported_speakers:
        raise RuntimeError(f"Unsupported speaker '{speaker}'. Supported speakers: {', '.join(supported_speakers)}")

    chunk_paths: List[Path] = []
    batches = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        wavs, sample_rate = model.generate_custom_voice(
            text=list(batch),
            language=[LANGUAGE_NAMES[language]] * len(batch),
            speaker=[speaker] * len(batch),
            instruct=[instruct or ""] * len(batch),
        )
        batches += 1
        for offset, wav in enumerate(wavs):
            index = batch_start + offset + 1
            chunk_path = temp_dir / f"chunk_{index:04d}.wav"
            sf.write(str(chunk_path), wav, sample_rate)
            chunk_paths.append(chunk_path)
        advance(len(batch))
    return chunk_paths, load_seconds, batches


def synthesize_chunks(
    chunks: Sequence[str],
    output_path: Path,
    language: str,
    engine: str,
    model_name_or_path: str,
    speaker: str,
    instruct: Optional[str],
    device: str,
    batch_size: int,
    workers: int,
    keep_temp: bool,
    use_flash_attention: bool,
    console: Console,
) -> SynthesisMetrics:
    started_at = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_chunks_", dir=str(output_path.parent)))
    concat_seconds = 0.0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Synthesizing chunks...", total=len(chunks))

            def advance(count: int) -> None:
                progress.advance(task, advance=count)

            if engine == "gemini":
                chunk_paths, load_seconds, batches = synthesize_gemini(
                    chunks, temp_dir, model_name_or_path, speaker, workers, advance
                )
            else:
                chunk_paths, load_seconds, batches = synthesize_qwen(
                    chunks,
                    temp_dir,
                    language,
                    model_name_or_path,
                    speaker,
                    instruct,
                    device,
                    batch_size,
                    use_flash_attention,
                    advance,
                )
        synthesis_seconds = time.perf_counter() - started_at - load_seconds

        concat_started_at = time.perf_counter()
        concatenate_wavs(chunk_paths, output_path)
        concat_seconds = time.perf_counter() - concat_started_at
        audio_seconds = wav_duration_seconds(output_path)
        elapsed_seconds = time.perf_counter() - started_at
        return SynthesisMetrics(
            chunks=len(chunks),
            batches=batches,
            load_seconds=load_seconds,
            synthesis_seconds=synthesis_seconds,
            concat_seconds=concat_seconds,
            elapsed_seconds=elapsed_seconds,
            audio_seconds=audio_seconds,
        )
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        if keep_temp:
            console.print(f"[yellow]Temporary chunk files kept at:[/] {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def concatenate_wavs(chunk_paths: Sequence[Path], output_path: Path) -> None:
    if not chunk_paths:
        raise RuntimeError("No chunk audio files were generated.")

    with wave.open(str(chunk_paths[0]), "rb") as first:
        params = first.getparams()

    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        for path in chunk_paths:
            with wave.open(str(path), "rb") as chunk:
                if chunk.getparams()[:3] != params[:3]:
                    raise RuntimeError(f"Chunk format mismatch while concatenating: {path}")
                output.writeframes(chunk.readframes(chunk.getnframes()))


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        frame_rate = wav.getframerate()
    return frames / frame_rate if frame_rate else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Markdown document into narration audio using Gemini TTS (OpenRouter) or local Qwen3-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s article.md --language pt-br
  %(prog)s article.md --language pt-br -o article.wav --speaker Leda
  %(prog)s article.md --language en --dry-run
  %(prog)s article.md --language en --engine qwen --speaker Aiden --instruct "Calm lecture style."

Setup:
  gemini (default): OPENROUTER_API_KEY in the repo .env
  qwen (local GPU):
  pip install -U qwen-tts soundfile torch
  huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
        """,
    )
    parser.add_argument("input", type=Path, help="Markdown document to narrate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output WAV file (default: outputs/audio/<input name>.wav, with a copy of the Markdown beside it)",
    )
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_NAMES), help="Narration language")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=DEFAULT_ENGINE,
        help=f"TTS engine: gemini (OpenRouter, paid) or qwen (local GPU) (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--model",
        help=f"OpenRouter model id (gemini, default: {GEMINI_MODEL}) or Qwen3-TTS id/local path (qwen, default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--speaker",
        help=f"Voice: Gemini voice name (default: {GEMINI_SPEAKER}) or Qwen CustomVoice speaker (default: {DEFAULT_SPEAKER})",
    )
    parser.add_argument("--instruct", help="Optional narration style instruction (qwen only)")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Runtime device for qwen (default: auto)",
    )
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="Maximum characters per TTS chunk")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"TTS chunks to synthesize per Qwen model call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel OpenRouter requests for gemini (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print rendered narration chunks without synthesizing")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary chunk WAV files after failures or completion",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Request FlashAttention 2 when loading the Qwen model on CUDA",
    )
    return parser


def main() -> int:
    console = Console()
    load_dotenv(REPO_ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args()
    gemini = args.engine == "gemini"
    model = args.model or (GEMINI_MODEL if gemini else DEFAULT_MODEL)
    speaker = args.speaker or (GEMINI_SPEAKER if gemini else DEFAULT_SPEAKER)

    if args.chunk_chars < 200:
        console.print("[bold red]Error:[/] --chunk-chars must be at least 200")
        return 2
    if args.batch_size < 1:
        console.print("[bold red]Error:[/] --batch-size must be at least 1")
        return 2
    if args.workers < 1:
        console.print("[bold red]Error:[/] --workers must be at least 1")
        return 2
    if gemini and args.instruct:
        console.print(
            "[bold red]Error:[/] --instruct only works with --engine qwen (OpenRouter TTS has no style field)"
        )
        return 2

    input_path = args.input.expanduser().resolve()
    output_path = (args.output or default_output_path(input_path)).expanduser().resolve()

    if not input_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {input_path}")
        return 1
    if input_path.suffix.lower() not in {".md", ".markdown"}:
        console.print(f"[bold red]Error:[/] Expected a Markdown file, got: {input_path.suffix}")
        return 2

    try:
        markdown_text = input_path.read_text(encoding="utf-8")
        blocks = markdown_to_blocks(markdown_text, args.language)
        chunks = chunk_blocks(blocks, args.chunk_chars)
    except MarkdownNarrationError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        return 1

    if not chunks:
        console.print("[bold red]Error:[/] No narratable content found.")
        return 1

    if args.dry_run:
        characters = sum(len(chunk) for chunk in chunks)
        console.print(
            Panel(
                f"Engine: [cyan]{args.engine}[/] ({model}, voice {speaker}) | "
                f"Chunks: [cyan]{len(chunks)}[/] | Characters: [cyan]{characters:,}[/]",
                title="Narration Preview",
                border_style="blue",
            )
        )
        for index, chunk in enumerate(chunks, start=1):
            console.print(f"\n[bold]Chunk {index}[/]")
            console.print(chunk)
        return 0

    try:
        metrics = synthesize_chunks(
            chunks=chunks,
            output_path=output_path,
            language=args.language,
            engine=args.engine,
            model_name_or_path=model,
            speaker=speaker,
            instruct=args.instruct,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            keep_temp=args.keep_temp,
            use_flash_attention=args.flash_attention,
            console=console,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/]")
        return 130
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        return 1

    source_copy = None if args.output else keep_source_next_to_audio(input_path, output_path)
    console.print(
        Panel(
            f"Output saved to: [cyan]{output_path}[/]\n"
            + (f"Source copied to: [cyan]{source_copy}[/]\n" if source_copy else "")
            + f"Elapsed: [cyan]{metrics.elapsed_seconds:.1f}s[/] | "
            f"Audio: [cyan]{metrics.audio_seconds:.1f}s[/] | "
            f"Realtime factor: [cyan]{metrics.realtime_factor:.2f}x[/]\n"
            f"Chunks: [cyan]{metrics.chunks}[/] | "
            f"{'Requests' if gemini else 'Batches'}: [cyan]{metrics.batches}[/] | "
            f"Load: [cyan]{metrics.load_seconds:.1f}s[/] | "
            f"Synthesis: [cyan]{metrics.synthesis_seconds:.1f}s[/] | "
            f"Concat: [cyan]{metrics.concat_seconds:.1f}s[/]",
            title="Done",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
