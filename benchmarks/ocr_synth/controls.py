"""Write control outputs that check the evaluator before any OCR engine is trusted.

  gt          the expected Markdown itself: every metric must be 1.0 and splits 0
  pdftotext   the PDF text layer (no LaTeX): text high, formulas low
  perturbed   expected Markdown with known damage: two paragraphs swapped, footnote 1 dropped,
              one formula with a wrong symbol, a running header inserted inside a paragraph

Usage: python benchmarks/ocr_synth/controls.py [OUT_ROOT]   (default results/controls)
then:  python benchmarks/ocr_synth/evaluate.py results/controls
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def perturb(md: str, gt: dict) -> str:
    paras = md.split("\n\n")
    blocks = [b["text"] for b in gt["blocks"] if b["id"].startswith("s")]
    # swap the first two body paragraphs that appear verbatim as Markdown blocks
    idx = [i for i, p in enumerate(paras) if any(p.startswith(b[:40]) for b in blocks)][:2]
    if len(idx) == 2:
        paras[idx[0]], paras[idx[1]] = paras[idx[1]], paras[idx[0]]
    md = "\n\n".join(p for p in paras if not p.startswith("[^1]:"))
    md = md.replace(r"\sqrt{2\eta_k}", r"\sqrt{2\beta_k}", 1)
    if not gt["headers"]:  # layouts without running headers (revtex) get no header perturbation
        return md
    header = gt["headers"][0]
    target = next(b for b in blocks if len(b) > 400)
    # cut after a sentence near the middle, outside inline math, like a page break would
    cuts = [i + 1 for i in range(len(target)) if target.startswith(". ", i) and target[:i].count("$") % 2 == 0]
    cut = min(cuts, key=lambda i: abs(i - len(target) // 2))
    return md.replace(target[:cut], target[:cut] + f"\n\n{header}\n\n", 1)


def main() -> None:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "controls"
    for name in ("gt", "pdftotext", "perturbed"):
        (out_root / name).mkdir(parents=True, exist_ok=True)
    for gt_path in sorted((DATA / "gt").glob("*.json")):
        doc = gt_path.stem
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        md = (DATA / "gt" / f"{doc}.md").read_text(encoding="utf-8")
        (out_root / "gt" / f"{doc}.md").write_text(md, encoding="utf-8")
        text = subprocess.run(["pdftotext", str(DATA / "pdf" / f"{doc}.pdf"), "-"], capture_output=True, text=True,
                              check=True).stdout
        (out_root / "pdftotext" / f"{doc}.md").write_text(text, encoding="utf-8")
        (out_root / "perturbed" / f"{doc}.md").write_text(perturb(md, gt), encoding="utf-8")
    print(f"controls written to {out_root}")


if __name__ == "__main__":
    main()
