"""Shared helpers for the pt-BR TTS benchmark runners (stdlib + soundfile only)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENTENCES = HERE / "sentences.json"
REF_WAV = HERE / "data" / "ref.wav"
REF_TXT = HERE / "data" / "ref.txt"
RESULTS = HERE / "results"


def load_sentences(only: str | None = None) -> list[dict]:
    items = json.loads(SENTENCES.read_text(encoding="utf-8"))
    if only:
        wanted = set(only.split(","))
        items = [s for s in items if s["id"] in wanted]
    return items


def ref_text() -> str:
    return REF_TXT.read_text(encoding="utf-8").strip()


def out_dir(engine: str) -> Path:
    path = RESULTS / "audio" / engine
    path.mkdir(parents=True, exist_ok=True)
    return path


def wav_seconds(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


class GpuPeak:
    """Peak `nvidia-smi` memory.used (MiB) while the block runs; counts every process on the GPU."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    @staticmethod
    def used_mib() -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0])

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.used_mib())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def save_run(engine: str, config: dict, rows: list[dict], peak_vram_mib: int, load_seconds: float) -> Path:
    """Write results/runs/<engine>.json with per-sentence timing and the run configuration."""
    path = RESULTS / "runs" / f"{engine}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "engine": engine,
        "config": config,
        "load_seconds": round(load_seconds, 2),
        "peak_vram_mib": peak_vram_mib,
        "sentences": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def timing_row(sid: str, wav: Path, gen_seconds: float) -> dict:
    dur = wav_seconds(wav)
    return {"id": sid, "gen_seconds": round(gen_seconds, 3), "audio_seconds": round(dur, 3),
            "rtf": round(gen_seconds / dur, 3) if dur else None}
