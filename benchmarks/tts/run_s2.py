#!/usr/bin/env python3
"""Fish Audio S2 Pro (GGUF) through the s2.cpp HTTP server: zero-shot clone of data/ref.wav, CUDA.

    python benchmarks/tts/run_s2.py [--quant q6_k] [--gpu-layers -1] [--only s01,s02]

The binary comes from S2_BIN (default ~/gitrepos/s2.cpp/build/s2); the GGUF files from
models/tts-bench/s2-pro-gguf/. The reference is encoded once into a saved voice profile.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests

from common import GpuPeak, HERE, REF_WAV, load_sentences, out_dir, ref_text, save_run, timing_row

S2_BIN = Path(os.environ.get("S2_BIN", Path.home() / "gitrepos/s2.cpp/build/s2"))
GGUF_DIR = HERE.parents[1] / "models/tts-bench/s2-pro-gguf"
PORT = 3031


def form(text: str, params: dict | None = None) -> dict:
    """multipart/form-data fields (the server rejects urlencoded bodies)."""
    fields = {"text": (None, text), "voice": (None, "ref")}
    if params:
        fields["params"] = (None, json.dumps(params))
    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="q6_k")
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--engine")
    ap.add_argument("--only")
    args = ap.parse_args()
    engine = args.engine or f"fish-s2-{args.quant}"

    model = GGUF_DIR / f"s2-pro-{args.quant}.gguf"
    common = ["--model", str(model), "--tokenizer", str(GGUF_DIR / "tokenizer.json"),
              "--cuda", "0", "--gpu-layers", str(args.gpu_layers), "--voice-dir", str(GGUF_DIR / "voices")]
    dest = out_dir(engine)
    log = open(HERE / "results" / f"s2-server-{engine}.log", "w")

    with GpuPeak() as gpu:
        t0 = time.perf_counter()
        subprocess.run([str(S2_BIN), *common, "--prompt-audio", str(REF_WAV), "--prompt-text", ref_text(),
                        "--voice", "ref", "--save-voice", "--text", "Olá.", "--output", str(dest / "_warmup.wav")],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        server = subprocess.Popen([str(S2_BIN), *common, "--server", "--port", str(PORT)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f"http://127.0.0.1:{PORT}/generate"
            for _ in range(600):
                try:
                    requests.post(url, files=form("Olá."), timeout=120).raise_for_status()
                    break
                except requests.ConnectionError:
                    time.sleep(1)
            load_seconds = time.perf_counter() - t0

            rows = []
            for s in load_sentences(args.only):
                t0 = time.perf_counter()
                resp = requests.post(url, files=form(s["text"], {"max_new_tokens": 2048}), timeout=900)
                resp.raise_for_status()
                elapsed = time.perf_counter() - t0
                wav = dest / f"{s['id']}.wav"
                wav.write_bytes(resp.content)
                rows.append(timing_row(s["id"], wav, elapsed))
                print(f"{s['id']}: {rows[-1]}", flush=True)
        finally:
            server.terminate()
            server.wait()
    (dest / "_warmup.wav").unlink(missing_ok=True)

    path = save_run(engine, {"model": model.name, "gpu_layers": args.gpu_layers, "mode": "voice clone (saved profile)",
                             "sampling": "s2.cpp defaults (temp 0.8, top-p 0.8, top-k 30)"},
                    rows, gpu.peak, load_seconds)
    print(f"peak VRAM {gpu.peak} MiB -> {path}")


if __name__ == "__main__":
    main()
