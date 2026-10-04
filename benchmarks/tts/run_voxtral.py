#!/usr/bin/env python3
"""Voxtral-4B-TTS int4 (torchao/HQQ backbone) via voxtral-int4, preset voice pt_female.

The open weights ship only preset voices (no cloning), so this engine cannot use data/ref.wav.
Run with the voxtral-int4 venv:

    ~/gitrepos/voxtral-int4/venv/bin/python benchmarks/tts/run_voxtral.py [--voice pt_female] [--only s01]
"""

import argparse
import os
import sys
import time
from pathlib import Path

VOXTRAL_SRC = Path(os.environ.get("VOXTRAL_INT4_SRC", Path.home() / "gitrepos/voxtral-int4/src"))
sys.path.insert(0, str(VOXTRAL_SRC))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from common import GpuPeak, HERE, load_sentences, out_dir, save_run, timing_row  # noqa: E402

MODEL_DIR = HERE.parents[1] / "models/tts-bench/voxtral-4b-tts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="pt_female")
    ap.add_argument("--flow-steps", type=int, default=8)
    ap.add_argument("--cfg-alpha", type=float, default=1.2)
    ap.add_argument("--engine", default="voxtral-int4")
    ap.add_argument("--only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from generate_fast import TekkenTokenizer, generate_speech_fast
    from torchao_inference import load_model_int4

    dest = out_dir(args.engine)
    voice_dir = str(MODEL_DIR / "voice_embedding")
    gen = dict(voice_name=args.voice, voice_dir=voice_dir, max_frames=1000, device="cuda",
               flow_steps=args.flow_steps, cfg_alpha=args.cfg_alpha)

    with GpuPeak() as gpu, torch.inference_mode():
        t0 = time.perf_counter()
        model = load_model_int4(str(MODEL_DIR), device="cuda")
        tok = TekkenTokenizer(str(MODEL_DIR / "tekken.json"))
        load_seconds = time.perf_counter() - t0

        generate_speech_fast(model, tok, "Olá.", **{**gen, "max_frames": 5})  # warmup
        rows = []
        for s in load_sentences(args.only):
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio, _ = generate_speech_fast(model, tok, s["text"], **gen)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            wav = dest / f"{s['id']}.wav"
            sf.write(str(wav), audio, 48000)  # generate_speech_fast returns 48 kHz (LPF + upsample postprocess)
            rows.append(timing_row(s["id"], wav, elapsed))
            print(f"{s['id']}: {rows[-1]}", flush=True)

    path = save_run(args.engine, {"model": "Voxtral-4B-TTS-2603 (int4 HQQ backbone, bf16 acoustic)", "voice": args.voice,
                                  "flow_steps": args.flow_steps, "cfg_alpha": args.cfg_alpha, "seed": args.seed},
                    rows, gpu.peak, load_seconds)
    print(f"peak VRAM {gpu.peak} MiB -> {path}")


if __name__ == "__main__":
    main()
