#!/usr/bin/env python3
"""Download the pt-BR reference voice and cut the 15.2 s clip used for zero-shot cloning.

Source: alissonpadua/tts-dataset-pt-br (MIT), audio/2026-01-19.wav, one female Brazilian narrator.
The cut (20.40-35.60 s) and its transcript were chosen from a Whisper word-timestamp pass.
"""

import subprocess
import urllib.request

from common import REF_TXT, REF_WAV

URL = "https://huggingface.co/datasets/alissonpadua/tts-dataset-pt-br/resolve/main/audio/2026-01-19.wav"
START, END = 20.40, 35.60
TRANSCRIPT = ("Os convidados de um casamento poderiam, por acaso, fazer jejum enquanto o noivo está com eles? "
              "Enquanto o noivo está com eles, os convidados não podem jejuar, mas vai chegar o tempo em que "
              "o noivo será tirado do meio deles.")


def main():
    REF_WAV.parent.mkdir(parents=True, exist_ok=True)
    full = REF_WAV.with_name("ref_full.wav")
    if not full.exists():
        urllib.request.urlretrieve(URL, full)
    subprocess.run(["sox", str(full), "-r", "24000", str(REF_WAV), "trim", str(START), f"={END}",
                    "fade", "0.02", "0", "0.05"], check=True)
    REF_TXT.write_text(TRANSCRIPT + "\n", encoding="utf-8")
    print(REF_WAV, REF_TXT)


if __name__ == "__main__":
    main()
