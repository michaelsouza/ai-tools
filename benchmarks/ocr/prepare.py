"""Download the benchmark papers from arXiv and render the benchmark pages.

Creates data/pdf/, data/pages/ (200 dpi PNG + single-page PDF) and
data/pages_scan/ (150 dpi, JPEG q40, rotated 0.7 deg, as JPG/PNG/PDF).

Usage: python benchmarks/ocr/prepare.py
Requires pdftoppm and pdfseparate (poppler-utils) and Pillow.
"""

import subprocess
import time
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
DATA = HERE / "data"

PAPERS = {"2006.11239": "2006.11239", "1706.03762": "1706.03762", "math_0211159": "math/0211159"}
PAGES = {
    "ddpm_p2": ("2006.11239", 2), "ddpm_p3": ("2006.11239", 3),
    "att_p4": ("1706.03762", 4), "att_p5": ("1706.03762", 5),
    "per_p15": ("math_0211159", 15), "per_p16": ("math_0211159", 16),
}
SCAN_PAGES = ("ddpm_p3", "per_p16")


def main() -> None:
    for sub in ("pdf", "pages", "pages_scan"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    for name, arxiv_id in PAPERS.items():
        pdf = DATA / "pdf" / f"{name}.pdf"
        if not pdf.exists():
            req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "ai-tools ocr benchmark"})
            with urllib.request.urlopen(req, timeout=120) as r:
                pdf.write_bytes(r.read())
            time.sleep(3)

    quiet = dict(check=True, stderr=subprocess.DEVNULL)
    for name, (doc, page) in PAGES.items():
        src = DATA / "pdf" / f"{doc}.pdf"
        subprocess.run(["pdftoppm", "-r", "200", "-f", str(page), "-l", str(page), "-png", "-singlefile", str(src),
                        str(DATA / "pages" / name)], **quiet)
        subprocess.run(["pdfseparate", "-f", str(page), "-l", str(page), str(src), str(DATA / "pages" / f"{name}.pdf")], **quiet)

    for name in SCAN_PAGES:
        doc, page = PAGES[name]
        raw = DATA / "pages_scan" / f"{name}_raw"
        subprocess.run(["pdftoppm", "-r", "150", "-f", str(page), "-l", str(page), "-png", "-singlefile",
                        str(DATA / "pdf" / f"{doc}.pdf"), str(raw)], **quiet)
        im = Image.open(f"{raw}.png").convert("L").rotate(0.7, resample=Image.BICUBIC, expand=True, fillcolor=255)
        jpg = DATA / "pages_scan" / f"{name}.jpg"
        im.convert("RGB").save(jpg, quality=40)
        Path(f"{raw}.png").unlink()
        degraded = Image.open(jpg)
        degraded.save(DATA / "pages_scan" / f"{name}.png")
        degraded.save(DATA / "pages_scan" / f"{name}.pdf", resolution=150)

    print(f"pages ready in {DATA}")


if __name__ == "__main__":
    main()
