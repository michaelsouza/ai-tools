#!/usr/bin/env python3
"""Build results/listening/: a blind listening test (index.html + MP3 clips) for every engine in results/audio/.

    python benchmarks/tts/make_listening.py [--engines a,b,c]

Each sentence shows its clips under shuffled letters; ratings (1-5) go to the artifact's `db`
(ratings/<viewer id>) when published as an Artifact, or to localStorage when opened as a local file.
"Revelar" shows the engine behind each letter, the mean rating per engine and the automatic metrics
from results/summary.json.
"""

import argparse
import base64
import json
import random
import subprocess

from common import HERE, REF_WAV, RESULTS, load_sentences

ENGINES = {
    "qwen3-1.7b": {"name": "Qwen3-TTS 1.7B", "where": "local", "voice": "clona a referência", "price": "grátis"},
    "fish-s2-q6_k": {"name": "Fish S2 Pro Q6_K", "where": "local", "voice": "clona a referência", "price": "grátis"},
    "voxtral-int4": {"name": "Voxtral 4B int4", "where": "local", "voice": "voz pt_female", "price": "grátis"},
    "fish-s2.1-api": {"name": "Fish S2.1 Pro", "where": "API", "voice": "clona a referência", "price": "US$ 15 / 1M car."},
    "gemini-flash-lite": {"name": "Gemini 3.8 Flash-Lite TTS", "where": "API", "voice": "voz Kore", "price": "~US$ 10–13 / 1M car."},
}


def to_mp3(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-codec:a", "libmp3lame",
                    "-q:a", "3", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = RESULTS / "listening"
    engines = args.engines.split(",")
    rng = random.Random(args.seed)
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

    to_mp3(REF_WAV, out / "audio" / "ref.mp3")
    items = []
    for s in load_sentences():
        clips = []
        for e in engines:
            wav = RESULTS / "audio" / e / f"{s['id']}.wav"
            if wav.exists():
                to_mp3(wav, out / "audio" / e / f"{s['id']}.mp3")
                clips.append({"engine": e, "src": f"audio/{e}/{s['id']}.mp3"})
        rng.shuffle(clips)
        items.append({"id": s["id"], "cat": s["cat"], "text": s["text"], "clips": clips})

    metrics = {}
    for e in engines:
        m = summary["engines"].get(e, {})
        metrics[e] = {**ENGINES.get(e, {"name": e}), "wer": m.get("wer_core"), "rtf": m.get("rtf"),
                      "vram": m.get("peak_vram_mib")}
    data = {"items": items, "engines": metrics}
    blob = base64.b64encode(json.dumps(data, ensure_ascii=False).encode()).decode()
    page = (HERE / "listening_template.html").read_text(encoding="utf-8").replace("__DATA__", blob)
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main()
