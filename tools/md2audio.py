#!/usr/bin/env python3
"""
Markdown to narration audio using Qwen3-TTS CustomVoice.

Usage:
    python tools/md2audio.py article.md --language en
    python tools/md2audio.py article.md --language pt-br -o article.wav
    python tools/md2audio.py article.md --language en --dry-run
    python tools/md2audio.py article.md --language en --speaker Aiden --instruct "Calm lecture style."

Setup:
    pip install -U qwen-tts soundfile torch
    huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
"""

import argparse
import re
import shutil
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

RECOMMENDED_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_MODEL = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "ryan"
DEFAULT_CHUNK_CHARS = 900
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
    return input_path.with_suffix(".wav")


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


def synthesize_chunks(
    chunks: Sequence[str],
    output_path: Path,
    language: str,
    model_name_or_path: str,
    speaker: str,
    instruct: Optional[str],
    device: str,
    batch_size: int,
    keep_temp: bool,
    use_flash_attention: bool,
    console: Console,
) -> SynthesisMetrics:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required to write Qwen3-TTS audio. Install it with: pip install soundfile"
        ) from exc

    started_at = time.perf_counter()
    resolved_device = resolve_device(device)
    load_started_at = time.perf_counter()
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    model = load_qwen_model(resolved_model, resolved_device, use_flash_attention)
    load_seconds = time.perf_counter() - load_started_at
    supported_speakers = getattr(model, "get_supported_speakers", lambda: [])()
    if supported_speakers and speaker not in supported_speakers:
        raise RuntimeError(f"Unsupported speaker '{speaker}'. Supported speakers: {', '.join(supported_speakers)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_chunks_", dir=str(output_path.parent)))
    chunk_paths: List[Path] = []
    batches = 0
    synthesis_seconds = 0.0
    concat_seconds = 0.0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Synthesizing chunks...", total=len(chunks))
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                batch_started_at = time.perf_counter()
                wavs, sample_rate = model.generate_custom_voice(
                    text=list(batch),
                    language=[LANGUAGE_NAMES[language]] * len(batch),
                    speaker=[speaker] * len(batch),
                    instruct=[instruct or ""] * len(batch),
                )
                synthesis_seconds += time.perf_counter() - batch_started_at
                batches += 1
                for offset, wav in enumerate(wavs):
                    index = batch_start + offset + 1
                    chunk_path = temp_dir / f"chunk_{index:04d}.wav"
                    sf.write(str(chunk_path), wav, sample_rate)
                    chunk_paths.append(chunk_path)
                progress.advance(task, advance=len(batch))

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
        description="Convert a Markdown document into narration audio using Qwen3-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s article.md --language en
  %(prog)s article.md --language pt-br -o article.wav
  %(prog)s article.md --language en --dry-run
  %(prog)s article.md --language en --speaker Aiden --instruct "Calm lecture style."

Manual setup:
  pip install -U qwen-tts soundfile torch
  huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
  huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
        """,
    )
    parser.add_argument("input", type=Path, help="Markdown document to narrate")
    parser.add_argument("-o", "--output", type=Path, help="Output WAV file (default: input name with .wav)")
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_NAMES), help="Narration language")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Qwen3-TTS model id or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        help=f"Qwen CustomVoice speaker (default: {DEFAULT_SPEAKER})",
    )
    parser.add_argument("--instruct", help="Optional narration style instruction")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Runtime device (default: auto)",
    )
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS, help="Maximum characters per TTS chunk")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"TTS chunks to synthesize per model call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print rendered narration chunks without loading Qwen")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary chunk WAV files after failures or completion",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Request FlashAttention 2 when loading the model on CUDA",
    )
    return parser


def main() -> int:
    console = Console()
    parser = build_parser()
    args = parser.parse_args()

    if args.chunk_chars < 200:
        console.print("[bold red]Error:[/] --chunk-chars must be at least 200")
        return 2
    if args.batch_size < 1:
        console.print("[bold red]Error:[/] --batch-size must be at least 1")
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
        console.print(Panel(f"Chunks: [cyan]{len(chunks)}[/]", title="Narration Preview", border_style="blue"))
        for index, chunk in enumerate(chunks, start=1):
            console.print(f"\n[bold]Chunk {index}[/]")
            console.print(chunk)
        return 0

    try:
        metrics = synthesize_chunks(
            chunks=chunks,
            output_path=output_path,
            language=args.language,
            model_name_or_path=args.model,
            speaker=args.speaker,
            instruct=args.instruct,
            device=args.device,
            batch_size=args.batch_size,
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

    console.print(
        Panel(
            f"Output saved to: [cyan]{output_path}[/]\n"
            f"Elapsed: [cyan]{metrics.elapsed_seconds:.1f}s[/] | "
            f"Audio: [cyan]{metrics.audio_seconds:.1f}s[/] | "
            f"Realtime factor: [cyan]{metrics.realtime_factor:.2f}x[/]\n"
            f"Chunks: [cyan]{metrics.chunks}[/] | "
            f"Batches: [cyan]{metrics.batches}[/] | "
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
