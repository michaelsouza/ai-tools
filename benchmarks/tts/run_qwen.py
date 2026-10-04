#!/usr/bin/env python3
"""Qwen3-TTS 12Hz Base: zero-shot clone of data/ref.wav (ICL mode), bf16, main repo venv.

    python benchmarks/tts/run_qwen.py [--model models/tts-bench/qwen3-tts-1.7b-base] [--only s01,s02]
"""

import argparse
import time

import soundfile as sf
import torch

from common import HERE, GpuPeak, REF_WAV, load_sentences, out_dir, ref_text, save_run, timing_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parents[1] / "models/tts-bench/qwen3-tts-1.7b-base"))
    ap.add_argument("--engine", default="qwen3-1.7b")
    ap.add_argument("--only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    dest = out_dir(args.engine)
    with GpuPeak() as gpu:
        t0 = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16)
        prompt = model.create_voice_clone_prompt(ref_audio=str(REF_WAV), ref_text=ref_text())
        load_seconds = time.perf_counter() - t0

        model.generate_voice_clone(text="Olá.", language="Portuguese", voice_clone_prompt=prompt)  # warmup
        rows = []
        for s in load_sentences(args.only):
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wavs, sr = model.generate_voice_clone(text=s["text"], language="Portuguese", voice_clone_prompt=prompt)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            wav = dest / f"{s['id']}.wav"
            sf.write(str(wav), wavs[0], sr)
            rows.append(timing_row(s["id"], wav, elapsed))
            print(f"{s['id']}: {rows[-1]}", flush=True)

    path = save_run(args.engine, {"model": args.model, "dtype": "bf16", "mode": "voice clone (ICL)", "seed": args.seed},
                    rows, gpu.peak, load_seconds)
    print(f"peak VRAM {gpu.peak} MiB -> {path}")


if __name__ == "__main__":
    main()
