#!/usr/bin/env python3
"""Intelligibility (WER/CER via Whisper large-v3) plus speed and VRAM for every engine in results/audio/.

    python benchmarks/tts/evaluate.py [--engines qwen3-1.7b,voxtral-int4] [--whisper large-v3]

Writes results/transcripts/<engine>.json and results/summary.json, and prints a Markdown table.
"""

import argparse
import json
import re
import unicodedata

from num2words import num2words
from rapidfuzz.distance import Levenshtein

from common import REF_WAV, RESULTS, load_sentences, ref_text

# Sentences whose valid spoken forms vary (numbers, abbreviations): reported, but left out of the core WER.
VARIANT_READINGS = {"s04", "s05", "s09"}


def _number_words(match: re.Match) -> str:
    raw = match.group(0)
    value = float(raw.replace(",", ".")) if "," in raw else int(raw)
    return f" {num2words(value, lang='pt_BR')} "


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("r$", " ").replace("%", " por cento ")
    text = re.sub(r"(\d)\.(?=\d{3}\b)", r"\1", text)  # 1.250.000 -> 1250000
    text = re.sub(r"(\d+),00\b", r"\1", text)
    text = re.sub(r"(\d+)h(\d+)", r"\1 \2", text)  # 14h30 -> 14 30
    text = re.sub(r"(\d+)h\b", r"\1", text)
    text = re.sub(r"\d+(?:,\d+)?", _number_words, text)
    text = text.replace("quatorze", "catorze")
    text = re.sub(r"\b(reais|horas|hora)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def error_rates(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word edits, ref words, char edits, ref chars) after normalization."""
    r, h = normalize(ref), normalize(hyp)
    return Levenshtein.distance(r.split(), h.split()), len(r.split()), Levenshtein.distance(r, h), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", help="comma-separated; default: every folder in results/audio")
    ap.add_argument("--whisper", default="large-v3")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    audio_root = RESULTS / "audio"
    engines = args.engines.split(",") if args.engines else sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    sentences = load_sentences()
    model = WhisperModel(args.whisper, device="cuda", compute_type="float16")

    def transcribe(path) -> str:
        segments, _ = model.transcribe(str(path), language="pt", beam_size=5, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments)

    ref_hyp = transcribe(REF_WAV)
    w, nw, c, nc = error_rates(ref_text(), ref_hyp)
    summary = {"whisper": args.whisper, "control_ref_wav": {"wer": round(w / nw, 4), "cer": round(c / nc, 4)}, "engines": {}}

    for engine in engines:
        run_file = RESULTS / "runs" / f"{engine}.json"
        run = json.loads(run_file.read_text(encoding="utf-8")) if run_file.exists() else {}
        timing = {r["id"]: r for r in run.get("sentences", [])}
        rows, tot = [], {"all": [0, 0, 0, 0], "core": [0, 0, 0, 0]}
        for s in sentences:
            wav = audio_root / engine / f"{s['id']}.wav"
            if not wav.exists():
                continue
            hyp = transcribe(wav)
            rates = error_rates(s["text"], hyp)
            for key in ("all", "core") if s["id"] not in VARIANT_READINGS else ("all",):
                tot[key] = [a + b for a, b in zip(tot[key], rates)]
            rows.append({"id": s["id"], "cat": s["cat"], "wer": round(rates[0] / rates[1], 4),
                         "cer": round(rates[2] / rates[3], 4), "hyp": hyp, **timing.get(s["id"], {})})
        (RESULTS / "transcripts").mkdir(parents=True, exist_ok=True)
        (RESULTS / "transcripts" / f"{engine}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        gen = sum(r.get("gen_seconds", 0) for r in rows)
        dur = sum(r.get("audio_seconds", 0) for r in rows)
        summary["engines"][engine] = {
            "n": len(rows),
            "wer_all": round(tot["all"][0] / tot["all"][1], 4),
            "cer_all": round(tot["all"][2] / tot["all"][3], 4),
            "wer_core": round(tot["core"][0] / tot["core"][1], 4),
            "cer_core": round(tot["core"][2] / tot["core"][3], 4),
            "rtf": round(gen / dur, 3) if dur else None,
            "audio_seconds": round(dur, 1),
            "peak_vram_mib": run.get("peak_vram_mib"),
            "load_seconds": run.get("load_seconds"),
            "worst": sorted(rows, key=lambda r: -r["wer"])[:3],
        }

    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"control (Whisper on ref.wav): WER {summary['control_ref_wav']['wer']:.1%}\n")
    print("| engine | WER core | CER core | WER all | RTF | peak VRAM (MiB) | load (s) |")
    print("|---|---|---|---|---|---|---|")
    for engine, e in summary["engines"].items():
        print(f"| {engine} | {e['wer_core']:.1%} | {e['cer_core']:.1%} | {e['wer_all']:.1%} | {e['rtf']} | "
              f"{e['peak_vram_mib']} | {e['load_seconds']} |")


if __name__ == "__main__":
    main()
