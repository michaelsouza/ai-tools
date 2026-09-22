"""Formula Quality Score (FQS) for OCR outputs.

For each ground-truth formula: best partial alignment (rapidfuzz partial_ratio)
of the normalized LaTeX against the normalized predicted page. Normalization
removes meaning-free variation (fonts, sizing, spacing, braces, unicode vs
LaTeX, equivalent relation macros, sub/superscript order).

Usage: python benchmarks/ocr/evaluate.py OUTPUT_ROOT [--pages ddpm_p3,per_p16] [--json out.json]
OUTPUT_ROOT/<engine>/<page>.md, e.g. benchmarks/ocr/results/outputs
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from rapidfuzz import fuzz

HERE = Path(__file__).parent

UNICODE = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon", "ϵ": r"\epsilon",
    "θ": r"\theta", "μ": r"\mu", "σ": r"\sigma", "τ": r"\tau", "Σ": r"\Sigma", "∇": r"\nabla",
    "∫": r"\int", "∑": r"\sum", "∏": r"\prod", "√": r"\sqrt", "≤": r"\leq", "≥": r"\geq", "∈": r"\in",
    "×": r"\times", "·": r"\cdot", "⋅": r"\cdot", "⟨": "<", "⟩": ">", "〈": "<", "〉": ">", "−": "-",
    "‖": r"\|", "…": "...", "∼": r"\sim", "ℒ": "L", "𝔼": "E", "ℝ": "R", "∂": r"\partial", "≔": ":=",
}
FONT_CMDS = r"mathbf|boldsymbol|bm|pmb|mathrm|mathit|mathcal|mathbb|mathsf|mathtt|mathscr|mathfrak|operatorname\*?|text|textrm|textbf|textit|mbox|rm|bf|it|cal|mathnormal|mathrel|mathop|mathbin|mathord|mathpunct|mathopen|mathclose"
SIZE_CMDS = r"left|right|middle|bigl|bigr|biggl|biggr|Bigl|Bigr|Biggl|Biggr|big|Big|bigg|Bigg"
SPACE_CMDS = r"quad|qquad|displaystyle|textstyle|limits|nolimits|nonumber|notag"


def _swap_sup_sub(s: str) -> str:
    atom = r"(\{[^{}]*\}|\\[A-Za-z]+|[^\s{}\\])"
    pattern = re.compile(r"\^" + atom + r"_" + atom)
    prev = None
    while prev != s:
        prev = s
        s = pattern.sub(r"_\2^\1", s)
    return s


def normalize(s: str) -> str:
    s = s.replace("τ\u0304", r"\bar{\tau}").replace("\u0304", "")
    for u, l in UNICODE.items():
        s = s.replace(u, l + " " if l.startswith("\\") else l)
    s = s.replace("$", "")
    s = re.sub(r"\\(label|tag)\*?\{[^{}]*\}", "", s)
    s = re.sub(r"\\hspace\*?\{[^{}]*\}", "", s)
    s = re.sub(r"\\(" + SIZE_CMDS + r")(?![A-Za-z])", "", s)
    s = re.sub(r"\\(" + SPACE_CMDS + r")(?![A-Za-z])", "", s)
    s = re.sub(r"\\[,;:!> ]", "", s)
    s = s.replace("~", "")
    s = re.sub(r"\\(" + FONT_CMDS + r")(?![A-Za-z])", "", s)
    repl = [
        (r"\\langle(?![A-Za-z])", "<"), (r"\\rangle(?![A-Za-z])", ">"),
        (r"\\leqslant(?![A-Za-z])", r"\\leq"), (r"\\le(?![A-Za-z])", r"\\leq"),
        (r"\\geqslant(?![A-Za-z])", r"\\geq"), (r"\\ge(?![A-Za-z])", r"\\geq"),
        (r"\\coloneqq(?![A-Za-z])", ":="), (r"\\defeq(?![A-Za-z])", ":="), (r"\\eqqcolon(?![A-Za-z])", "=:"),
        (r"\\(t|d)frac(?![A-Za-z])", r"\\frac"),
        (r"\\(l|c)?dots[cbmoi]?(?![A-Za-z])", "..."), (r"\\cdots(?![A-Za-z])", "..."),
        (r"\\(lVert|rVert|Vert|parallel)(?![A-Za-z])", r"\\|"), (r"\\(lvert|rvert|vert|mid)(?![A-Za-z])", "|"),
        (r"\\varepsilon(?![A-Za-z])", r"\\epsilon"),
        (r"\\overline(?![A-Za-z])", r"\\bar"), (r"\\widetilde(?![A-Za-z])", r"\\tilde"), (r"\\widehat(?![A-Za-z])", r"\\hat"),
        (r"\\(top|intercal)(?![A-Za-z])", "T"),
        (r"\\ast(?![A-Za-z])", "*"),
    ]
    for a, b in repl:
        s = re.sub(a, b, s)
    s = s.replace("\\\\", "").replace("&", "")
    s = re.sub(r"\s+", " ", s)
    s = _swap_sup_sub(s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return s


def fqs(gt_latex: str, predicted_doc: str) -> float:
    g, d = normalize(gt_latex), normalize(predicted_doc)
    if not g or not d:
        return 0.0
    return fuzz.partial_ratio(g, d) / 100.0


def evaluate(root: Path, gt: list, pages: set | None = None) -> dict:
    results = {}
    for engine_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per = {}
        for item in gt:
            if pages and item["page"] not in pages:
                continue
            md = engine_dir / f"{item['page']}.md"
            doc = md.read_text(encoding="utf-8") if md.exists() else ""
            per[item["id"]] = round(fqs(item["latex"], doc), 4)
        if per:
            vals = list(per.values())
            results[engine_dir.name] = {
                "fqs": round(statistics.mean(vals), 4),
                "ge95": round(sum(v >= 0.95 for v in vals) / len(vals), 3),
                "lt75": round(sum(v < 0.75 for v in vals) / len(vals), 3),
                "n": len(vals),
                "per_formula": per,
            }
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--pages")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    gt = json.loads((HERE / "gt.json").read_text(encoding="utf-8"))
    res = evaluate(a.root, gt, set(a.pages.split(",")) if a.pages else None)
    for name, r in sorted(res.items(), key=lambda kv: -kv[1]["fqs"]):
        print(f"{name:16} FQS={r['fqs']:.3f}  >=0.95: {r['ge95']:.0%}  <0.75: {r['lt75']:.0%}  (n={r['n']})")
    if a.json:
        a.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.exit(0)
