"""Score converter outputs against the synthetic ground truth written by build.py.

OUTPUT_ROOT/<engine>/<doc>.md (whole document) or OUTPUT_ROOT/<engine>/<doc>/pNN.md (one file per page,
concatenated in page order). Scores are per document, so floats moving across pages do not matter.

Metrics (0..1, higher is better unless noted):
  text      mean best partial match of each text block (title, abstract, paragraphs), math and markup removed
  order     Kendall tau of the matched block positions (1 = reading order kept; 2-column layouts stress it)
  fqs       Formula Quality Score of benchmarks/ocr/evaluate.py over display and non-trivial inline formulas
  f_err     count of formulas scoring below 1.0 (the acceptance check: one wrong symbol barely moves the mean)
  notes     mean match of footnote texts
  captions  mean match of figure captions
  headers   informative only: running headers found in the output / headers printed on the pages
  splits    count of text blocks whose matched span contains a running header (must be 0)

Usage: python benchmarks/ocr_synth/evaluate.py OUTPUT_ROOT [--docs a01__ieee-2col,...] [--json out.json]
"""

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

from rapidfuzz import fuzz

HERE = Path(__file__).parent
GT_DIR = HERE / "data" / "gt"

_spec = importlib.util.spec_from_file_location("ocr_fqs", HERE.parent / "ocr" / "evaluate.py")
_fqs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fqs_mod)
fqs, normalize_latex = _fqs_mod.fqs, _fqs_mod.normalize

MIN_INLINE = 8  # normalized length below which inline formulas are trivial to match and are skipped
MATH = re.compile(
    r"\$\$.+?\$\$|\$.+?\$|\\\[.+?\\\]|\\\(.+?\\\)|\\begin\{(\w+\*?)\}.+?\\end\{\1\}", re.S
)


def norm_text(s: str) -> str:
    """Prose only: dehyphenate, drop math, markup and punctuation, lowercase, single spaces."""
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    s = MATH.sub(" ", s)
    s = re.sub(r"<[^>]+>", " ", s)                      # HTML tags (tables, <sup>)
    s = re.sub(r"\[\^?\d+\]|\^\{?\d+\}?|[¹²³⁴⁵⁶⁷⁸⁹⁰]", " ", s)  # footnote markers
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def best_match(needle: str, hay: str) -> tuple[float, int, int]:
    if not needle or not hay:
        return 0.0, -1, -1
    r = fuzz.partial_ratio_alignment(needle, hay)
    return r.score / 100.0, r.dest_start, r.dest_end


def kendall_tau(pos: list[int]) -> float:
    n = len(pos)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[i] < pos[j]:
                conc += 1
            elif pos[i] > pos[j]:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def read_output(engine_dir: Path, doc: str) -> str:
    single = engine_dir / f"{doc}.md"
    if single.exists():
        return single.read_text(encoding="utf-8")
    pages = sorted((engine_dir / doc).glob("p*.md"))
    return "\n\n".join(p.read_text(encoding="utf-8") for p in pages)


def score_doc(gt: dict, out: str) -> dict:
    hay = norm_text(out)
    blocks = []
    for b in gt["blocks"]:
        s, start, end = best_match(norm_text(b["text"]), hay)
        blocks.append({"id": b["id"], "score": round(s, 4), "start": start, "end": end})
    matched = [b for b in blocks if b["score"] >= 0.8]
    order = kendall_tau([b["start"] for b in matched])

    formulas = {}
    for f in gt["formulas"]:
        if f["kind"] == "inline" and len(normalize_latex(f["latex"])) < MIN_INLINE:
            continue
        formulas[f["id"]] = round(fqs(f["latex"], out), 4)

    notes = {n["id"]: round(best_match(norm_text(n["text"]), hay)[0], 4) for n in gt["footnotes"]}
    caps = {c["id"]: round(best_match(norm_text(c["text"]), hay)[0], 4) for c in gt["captions"]}

    header_norms = [norm_text(h) for h in gt["headers"]]
    found = sum(len(re.findall(re.escape(h), hay)) for h in header_norms if h)
    printed = gt.get("header_occurrences", 0)
    spans = [(b["start"], b["end"]) for b in matched]
    splits = sum(
        1 for st, en in spans for h in header_norms if h and h in hay[st:en]
    )

    mean = lambda d: round(statistics.mean(d), 4) if d else None  # noqa: E731
    return {
        "text": mean([b["score"] for b in blocks]),
        "order": round(order, 4),
        "fqs": mean(list(formulas.values())),
        "f_err": sum(v < 1.0 for v in formulas.values()),
        "notes": mean(list(notes.values())),
        "captions": mean(list(caps.values())),
        "headers": f"{found}/{printed}",
        "splits": splits,
        "n_formulas": len(formulas),
        "per_block": {b["id"]: b["score"] for b in blocks},
        "per_formula": formulas,
        "per_note": notes,
        "per_caption": caps,
    }


def evaluate(root: Path, docs: set | None = None) -> dict:
    gts = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(GT_DIR.glob("*.json"))}
    results = {}
    for engine_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        per_doc = {}
        for doc, gt in gts.items():
            if docs and doc not in docs:
                continue
            out = read_output(engine_dir, doc)
            if out.strip():
                per_doc[doc] = score_doc(gt, out)
        if per_doc:
            results[engine_dir.name] = per_doc
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--docs")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = evaluate(a.root, set(a.docs.split(",")) if a.docs else None)
    cols = ("text", "order", "fqs", "f_err", "notes", "captions", "headers", "splits")
    print(f"{'engine':12} {'doc':24} " + " ".join(f"{c:>8}" for c in cols))
    for engine, per_doc in res.items():
        for doc, r in per_doc.items():
            vals = [f"{r[c]:>8.3f}" if isinstance(r[c], float) else f"{str(r[c]):>8}" for c in cols]
            print(f"{engine:12} {doc:24} " + " ".join(vals))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
    sys.exit(0)
