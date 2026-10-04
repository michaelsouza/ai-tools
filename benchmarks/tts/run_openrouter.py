#!/usr/bin/env python3
"""Paid TTS through OpenRouter's /api/v1/audio/speech (needs OPENROUTER_API_KEY in .env).

    python benchmarks/tts/run_openrouter.py gemini-flash-lite [--only s01]
    python benchmarks/tts/run_openrouter.py fish-s2.1-api

gen_seconds is the HTTP round trip (network included), so its RTF is not comparable with the local engines.
"""

import argparse
import base64
import subprocess
import time

import requests
from dotenv import dotenv_values

from common import HERE, REF_WAV, load_sentences, out_dir, ref_text, save_run, timing_row

API = "https://openrouter.ai/api/v1/audio/speech"
ENGINES = {
    "gemini-flash-lite": {"model": "google/gemini-3.8-flash-lite-tts", "voice": "Kore", "format": "pcm"},
    "fish-s2.1-api": {"model": "fish-audio/s2.1-pro", "clone": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=ENGINES)
    ap.add_argument("--only")
    args = ap.parse_args()

    cfg = ENGINES[args.engine]
    key = dotenv_values(HERE.parents[1] / ".env")["OPENROUTER_API_KEY"]
    fmt = cfg.get("format", "mp3")
    body = {"model": cfg["model"], "response_format": fmt}
    if "voice" in cfg:
        body["voice"] = cfg["voice"]
    if cfg.get("clone"):
        audio_b64 = base64.b64encode(REF_WAV.read_bytes()).decode()
        body["input_references"] = [
            {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio_b64}"}},
            {"type": "text", "text": ref_text()},
        ]

    dest = out_dir(args.engine)
    rows = []
    for s in load_sentences(args.only):
        t0 = time.perf_counter()
        resp = requests.post(API, headers={"Authorization": f"Bearer {key}"}, json={**body, "input": s["text"]}, timeout=300)
        elapsed = time.perf_counter() - t0
        if not resp.ok:
            raise SystemExit(f"{s['id']}: HTTP {resp.status_code} {resp.text[:500]}")
        raw = dest / f"{s['id']}.{fmt}"
        raw.write_bytes(resp.content)
        wav = raw.with_suffix(".wav")
        pcm_in = ["-f", "s16le", "-ar", "24000", "-ac", "1"] if fmt == "pcm" else []  # Gemini PCM: 24 kHz mono s16le
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *pcm_in, "-i", str(raw), str(wav)], check=True)
        raw.unlink()
        rows.append(timing_row(s["id"], wav, elapsed))
        print(f"{s['id']}: {rows[-1]}", flush=True)

    config = {k: v for k, v in cfg.items() if k not in ("clone", "format")} | {"mode": "voice clone" if cfg.get("clone") else "preset voice",
                                                                "timing": "HTTP round trip"}
    path = save_run(args.engine, config, rows, peak_vram_mib=None, load_seconds=0.0)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
