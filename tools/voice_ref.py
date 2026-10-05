#!/usr/bin/env python3
"""
Prepare a voice-cloning reference for `md2audio.py --voice` from your own recordings.

Usage:
    python tools/voice_ref.py michael /mnt/c/Users/Michael/Downloads/voz_*.m4a
    python tools/voice_ref.py michael /mnt/c/Users/Michael/Downloads/voz_*.m4a --take voz_b2.m4a

Each take is converted to 24 kHz mono WAV, trimmed, peak-normalized, checked for background noise and
transcribed with Whisper to measure how closely it matches one of the reading texts in
docs/voice_reference.md. The best take becomes models/voices/<name>.wav plus its transcript <name>.txt.

Requirements: ffmpeg and sox binaries, faster-whisper, numpy, soundfile, rapidfuzz, num2words.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXTS_DOC = REPO_ROOT / "docs" / "voice_reference.md"
VOICES_DIR = REPO_ROOT / "models" / "voices"  # git-ignored with the rest of models/
SAMPLE_RATE = 24000
MAX_WER = 0.05  # above this the reading does not match its text closely enough to trust the canonical transcript
MIN_SECONDS, MAX_SECONDS = 6.0, 25.0
MIN_SNR_DB = 30.0


@dataclass
class Take:
    source: Path
    wav: Path
    seconds: float
    snr_db: float
    heard: str
    text_id: str
    wer: float

    @property
    def usable(self) -> bool:
        return self.wer <= MAX_WER and MIN_SECONDS <= self.seconds <= MAX_SECONDS


def problem(take: Take) -> str:
    """Why a take is not usable, for the summary table."""
    if take.wer > MAX_WER:
        return "reading differs from text"
    if not MIN_SECONDS <= take.seconds <= MAX_SECONDS:
        return "too short" if take.seconds < MIN_SECONDS else "too long"
    return ""


def load_texts(doc: Path = TEXTS_DOC) -> Dict[str, str]:
    """Reading texts from the doc: each `### X: title` heading followed by a `> ` quote line."""
    texts, current = {}, None
    for line in doc.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^###\s+([A-Z]):", line)
        if heading:
            current = heading.group(1)
        elif current and line.startswith("> "):
            texts[current] = line[2:].strip()
            current = None
    return texts


def normalize(text: str) -> List[str]:
    """Lowercase words without punctuation; digits spelled out, since Whisper writes "40%" for "quarenta por cento"."""
    from num2words import num2words

    text = unicodedata.normalize("NFC", text.lower()).replace("%", " por cento ")
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='pt_BR')} ", text)
    return re.sub(r"[^\w\s]", " ", text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    from rapidfuzz.distance import Levenshtein

    ref, hyp = normalize(reference), normalize(hypothesis)
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def snr_db(samples, sample_rate: int) -> float:
    """Rough signal-to-noise ratio: loud speech frames (90th percentile RMS) over pauses (10th percentile)."""
    import numpy as np

    frame = int(0.02 * sample_rate)
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    rms_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-10)
    return float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))


def condition(source: Path, wav: Path) -> None:
    """Mono 24 kHz, edge silence trimmed (-45 dB), 0.2 s padding, peak-normalized to -1 dBFS."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ac", "1", "-ar", str(SAMPLE_RATE), str(raw)],
            check=True,
        )
        subprocess.run(
            [
                "sox",
                str(raw),
                str(wav),
                "silence",
                "1",
                "0.15",
                "-45d",
                "reverse",
                "silence",
                "1",
                "0.15",
                "-45d",
                "reverse",
                "pad",
                "0.2",
                "0.2",
                "norm",
                "-1",
            ],
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a voice-cloning reference for md2audio.py --voice.")
    parser.add_argument("name", help="Voice name, used as models/voices/<name>.wav/.txt (letters, digits, - and _)")
    parser.add_argument("takes", nargs="+", type=Path, help="Recordings of the texts in docs/voice_reference.md")
    parser.add_argument("--take", help="Use this recording (file name) instead of the best-scoring one")
    parser.add_argument("--whisper", default="large-v3", help="faster-whisper model for the check (default: large-v3)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    console = Console()
    args = build_parser().parse_args(argv)
    if not re.fullmatch(r"[\w-]+", args.name):
        console.print("[bold red]Error:[/] voice name may only contain letters, digits, - and _")
        return 2
    missing = [str(p) for p in args.takes if not p.exists()]
    if missing:
        console.print(f"[bold red]Error:[/] file not found: {', '.join(missing)}")
        return 1

    from faster_whisper import WhisperModel

    texts = load_texts()
    whisper = WhisperModel(args.whisper, device="cuda", compute_type="float16")
    work = Path(tempfile.mkdtemp(prefix="voice_ref_"))
    try:
        return prepare(console, args, texts, whisper, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def prepare(console: Console, args: argparse.Namespace, texts: Dict[str, str], whisper, work: Path) -> int:
    import soundfile as sf

    takes: List[Take] = []
    for source in args.takes:
        wav = work / f"{source.stem}.wav"
        condition(source, wav)
        samples, sample_rate = sf.read(str(wav))
        segments, _ = whisper.transcribe(str(wav), language="pt", beam_size=5)
        heard = " ".join(s.text.strip() for s in segments)
        text_id, wer = min(((tid, word_error_rate(t, heard)) for tid, t in texts.items()), key=lambda x: x[1])
        takes.append(Take(source, wav, len(samples) / sample_rate, snr_db(samples, sample_rate), heard, text_id, wer))

    if args.take:
        chosen = next((t for t in takes if t.source.name == args.take), None)
        if chosen is None:
            console.print(f"[bold red]Error:[/] --take {args.take} is not among the recordings given")
            return 2
    else:
        usable = [t for t in takes if t.usable]
        chosen = max(usable, key=lambda t: (t.snr_db, -t.wer)) if usable else min(takes, key=lambda t: t.wer)

    table = Table(title="Recordings")
    for column in ("file", "text", "WER", "seconds", "SNR dB", ""):
        table.add_column(column)
    for t in takes:
        table.add_row(
            t.source.name,
            t.text_id,
            f"{t.wer:.0%}",
            f"{t.seconds:.1f}",
            f"{t.snr_db:.0f}",
            "← chosen" if t is chosen else problem(t),
        )
    console.print(table)

    # The canonical text is exact when the reading matched it; otherwise keep what Whisper heard.
    transcript = texts[chosen.text_id] if chosen.wer <= MAX_WER else chosen.heard
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out_wav, out_txt = VOICES_DIR / f"{args.name}.wav", VOICES_DIR / f"{args.name}.txt"
    out_wav.write_bytes(chosen.wav.read_bytes())
    out_txt.write_text(transcript + "\n", encoding="utf-8")
    console.print(f"Saved [cyan]{out_wav}[/] and [cyan]{out_txt}[/]")

    if chosen.wer > MAX_WER:
        console.print(
            f"[yellow]Warning:[/] the reading differs from text {chosen.text_id} (WER {chosen.wer:.0%}); "
            f"saved Whisper's transcript instead. Check {out_txt.name} word by word, or record again."
        )
    if chosen.snr_db < MIN_SNR_DB:
        console.print(
            f"[yellow]Warning:[/] background noise is high (SNR {chosen.snr_db:.0f} dB); the clone copies it. "
            "A quieter room helps."
        )
    if not MIN_SECONDS <= chosen.seconds <= MAX_SECONDS:
        console.print(f"[yellow]Warning:[/] {chosen.seconds:.1f} s after trimming; aim for 10–20 s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
