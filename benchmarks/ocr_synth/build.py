"""Typeset the synthetic articles with journal LaTeX classes and write their ground truth.

Each article in articles/*.yaml is rendered once per layout (journal class and column count).
For every article x layout pair ("doc", e.g. a01__ieee-2col) this writes, under data/:

  tex/<doc>.tex, pdf/<doc>.pdf        the typeset article
  pages/<doc>/pNN.png, pNN.pdf        200 dpi page images and single-page PDFs (OCR input)
  gt/<doc>.json                       ground truth used by evaluate.py
  gt/<doc>.md                         expected Markdown (reading copy, also the "gt" control)

Usage: python benchmarks/ocr_synth/build.py [ARTICLE_ID ...] [--layouts elsarticle-1col,ieee-2col]
Requires latexmk with TeX Live (elsarticle, IEEEtran, pgfplots) and poppler-utils.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DATA = HERE / "data"

ELS_JOURNAL = "Journal of Computational Statistics and Learning"
ELS_RUNNING = f"{ELS_JOURNAL} 12 (2026) 101104"
ELS_FOOTER = "https://doi.org/10.5555/j.jcsl.2026.101104"  # 10.5555 is the DOI test prefix
IEEE_JOURNAL = "IEEE Transactions on Statistical Learning"
IEEE_VOLUME = "Vol.~12, No.~4, November 2026"

LAYOUTS = {
    "elsarticle-1col": {"columns": 1, "headers": [ELS_RUNNING, ELS_FOOTER]},
    "elsarticle-2col": {"columns": 2, "headers": [ELS_RUNNING, ELS_FOOTER]},
    "ieee-2col": {"columns": 2, "headers": [f"{IEEE_JOURNAL}, {IEEE_VOLUME}".replace("~", " ")]},
}


# ---------------------------------------------------------------- source parsing

def split_footnotes(text: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split '...^[note]...' into [(kind, chunk)] with kind 'text' or 'fn'; brackets may nest."""
    parts, i, start = [], 0, 0
    while i < len(text):
        if text.startswith("^[", i):
            parts.append(("text", text[start:i]))
            depth, j = 1, i + 2
            while depth:
                depth += {"[": 1, "]": -1}.get(text[j], 0)
                j += 1
            parts.append(("fn", text[i + 2 : j - 1].strip()))
            i = start = j
        else:
            i += 1
    parts.append(("text", text[start:]))
    return parts, [c for k, c in parts if k == "fn"]


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def inline_math(s: str) -> list[str]:
    return re.findall(r"(?<!\\)\$(.+?)(?<!\\)\$", s)


def tex_to_plain(s: str) -> str:
    """Prose as it appears on the page: LaTeX spacing/quote markup resolved, math kept as $...$."""
    s = s.replace("--", "–").replace("{,}", ",").replace("~", " ")
    return clean(s)


# ---------------------------------------------------------------- LaTeX rendering

def tex_paragraph(text: str) -> str:
    parts, _ = split_footnotes(text)
    return "".join(c if k == "text" else f"\\footnote{{{c}}}" for k, c in parts).strip()


def tex_figure(fig: dict, columns: int) -> str:
    width = r"0.95\columnwidth" if columns == 2 else r"0.62\textwidth"
    plots = "\n".join(
        f"    \\addplot[{style}, thick, domain={fig['xmin']}:{fig['xmax']}, samples=120] {{{c['expr']}}};\n"
        f"    \\addlegendentry{{{c['legend']}}}"
        for c, style in zip(fig["curves"], ["blue", "red, dashed", "black, dotted"])
    )
    return (
        "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
        f"  \\begin{{axis}}[width={width}, height=0.62*{width}, xlabel={{{fig['xlabel']}}}, ylabel={{{fig['ylabel']}}},\n"
        "    legend style={font=\\footnotesize, at={(0.98,0.98)}, anchor=north east}, tick label style={font=\\footnotesize},\n"
        "    scaled x ticks=false, xticklabel style={/pgf/number format/fixed}]\n"
        f"{plots}\n  \\end{{axis}}\n\\end{{tikzpicture}}\n\\caption{{{clean(fig['caption'])}}}\n\\end{{figure}}"
    )


def tex_body(art: dict, columns: int) -> str:
    out = []
    for sec in art["sections"]:
        out.append(f"\\section{{{sec['title']}}}")
        for b in sec["blocks"]:
            if "p" in b:
                out.append(tex_paragraph(clean(b["p"])))
            elif "eq" in b:
                out.append(f"\\begin{{equation}}\n{clean(b['eq'])}\n\\end{{equation}}")
            elif "align" in b:
                out.append("\\begin{align}\n" + b["align"].strip().replace("\\\\ ", "\\\\\n") + "\n\\end{align}")
            elif "figure" in b:
                out.append(tex_figure(b["figure"], columns))
    return "\n\n".join(out)


PREAMBLE = r"""\usepackage{amsmath,amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
"""


def render_tex(art: dict, layout: str) -> str:
    columns = LAYOUTS[layout]["columns"]
    body = tex_body(art, columns)
    abstract = clean(art["abstract"])
    if layout.startswith("elsarticle"):
        opts = "preprint,12pt,a4paper" if columns == 1 else "5p,twocolumn,a4paper"
        authors = "\n".join(f"\\author[inst]{{{a}}}" for a in art["authors"])
        keywords = " \\sep ".join(art["keywords"])
        first = art["authors"][0].split()
        running = (
            "\\usepackage{fancyhdr}\n\\pagestyle{fancy}\n\\fancyhf{}\n\\renewcommand{\\headrulewidth}{0.4pt}\n"
            f"\\fancyhead[L]{{\\footnotesize {first[0][0]}. {first[-1]} et al.}}\n"
            f"\\fancyhead[R]{{\\footnotesize {ELS_RUNNING}}}\n\\fancyfoot[C]{{\\footnotesize\\thepage}}\n"
            f"\\myfooter[L]{{{ELS_FOOTER}}}\n"
        )
        return (
            f"\\documentclass[{opts}]{{elsarticle}}\n{PREAMBLE}{running}\\journal{{{ELS_JOURNAL}}}\n\\begin{{document}}\n"
            f"\\begin{{frontmatter}}\n\\title{{{art['title']}}}\n{authors}\n\\affiliation[inst]{{organization={{{art['affiliation']}}}}}\n"
            f"\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n"
            f"\\begin{{keyword}}\n{keywords}\n\\end{{keyword}}\n\\end{{frontmatter}}\n\n"
            f"{body}\n\n\\end{{document}}\n"
        )
    if layout == "ieee-2col":
        names = art["authors"]
        author_line = ", ".join(names[:-1]) + ", and " + names[-1]
        last = names[0].split()[-1]
        return (
            f"\\documentclass[journal]{{IEEEtran}}\n{PREAMBLE}\\begin{{document}}\n"
            f"\\title{{{art['title']}}}\n\\author{{{author_line}\\thanks{{{art['affiliation']}.}}}}\n"
            f"\\markboth{{{IEEE_JOURNAL}, {IEEE_VOLUME}}}{{{last} \\MakeLowercase{{\\textit{{et al.}}}}: {art['short_title']}}}\n"
            f"\\maketitle\n\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n"
            f"\\begin{{IEEEkeywords}}\n{', '.join(art['keywords'])}\n\\end{{IEEEkeywords}}\n\n"
            f"{body}\n\n\\end{{document}}\n"
        )
    raise ValueError(layout)


# ---------------------------------------------------------------- ground truth

def ground_truth(art: dict) -> tuple[dict, str]:
    """Layout-independent ground truth: ordered text blocks, formulas, footnotes, captions; and expected Markdown."""
    blocks, formulas, footnotes, captions = [], [], [], []
    md = [f"# {art['title']}", "", ", ".join(art["authors"]), "", art["affiliation"], ""]
    abstract = tex_to_plain(art["abstract"])
    blocks.append({"id": "title", "text": art["title"]})
    blocks.append({"id": "abstract", "text": abstract})
    md += [f"**Abstract.** {abstract}", "", f"**Keywords:** {', '.join(art['keywords'])}", ""]
    for m in inline_math(art["abstract"]):
        formulas.append({"id": f"abstract_m{len(formulas)}", "latex": m, "kind": "inline"})
    fn_md = []
    for si, sec in enumerate(art["sections"], 1):
        md += [f"## {si}. {sec['title']}", ""]
        for bi, b in enumerate(sec["blocks"], 1):
            bid = f"s{si}b{bi}"
            if "p" in b:
                parts, notes = split_footnotes(clean(b["p"]))
                text = tex_to_plain("".join(c for k, c in parts if k == "text"))
                blocks.append({"id": bid, "text": text})
                para_md = ""
                for k, c in parts:
                    if k == "text":
                        para_md += tex_to_plain(c) if c.strip() else c
                    else:
                        n = len(footnotes) + 1
                        footnotes.append({"id": f"fn{n}", "text": tex_to_plain(c)})
                        fn_md.append(f"[^{n}]: {tex_to_plain(c)}")
                        para_md += f"[^{n}]"
                md += [para_md.strip(), ""]
                for k, c in parts:
                    for m in inline_math(c):
                        formulas.append({"id": f"{bid}_m{len(formulas)}", "latex": m, "kind": "inline"})
            elif "eq" in b:
                formulas.append({"id": bid, "latex": clean(b["eq"]), "kind": "display"})
                md += ["$$", clean(b["eq"]), "$$", ""]
            elif "align" in b:
                rows = [clean(r) for r in b["align"].split("\\\\") if r.strip()]
                for ri, r in enumerate(rows):
                    formulas.append({"id": f"{bid}_r{ri}", "latex": r, "kind": "display"})
                md += ["$$", "\\begin{aligned}", " \\\\\n".join(rows), "\\end{aligned}", "$$", ""]
            elif "figure" in b:
                cap = tex_to_plain(b["figure"]["caption"])
                captions.append({"id": bid, "text": cap})
                md += [f"Figure {len(captions)}: {cap}", ""]
    md += [line for fn in fn_md for line in (fn, "")]
    gt = {"article": art["id"], "blocks": blocks, "formulas": formulas, "footnotes": footnotes, "captions": captions}
    return gt, "\n".join(md).strip() + "\n"


# ---------------------------------------------------------------- build

def compile_pdf(tex_path: Path, out_pdf: Path) -> None:
    build = DATA / "build"
    build.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={build}", str(tex_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"LaTeX failed for {tex_path.name}; see {build / tex_path.with_suffix('.log').name}")
    shutil.copy(build / tex_path.with_suffix(".pdf").name, out_pdf)


def render_pages(pdf: Path, out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    quiet = dict(check=True, stderr=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-r", "200", "-png", str(pdf), str(out_dir / "p")], **quiet)
    subprocess.run(["pdfseparate", str(pdf), str(out_dir / "p%02d.pdf")], **quiet)
    pngs = sorted(out_dir.glob("p-*.png"))
    for png in pngs:  # pdftoppm pads to the page count's width; normalize to pNN.png
        png.rename(out_dir / f"p{int(png.stem.split('-')[1]):02d}.png")
    return len(pngs)


def header_occurrences(pdf: Path, headers: list[str]) -> int:
    """How many times the running headers are printed, read from the PDF text layer."""
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()  # noqa: E731
    return sum(norm(text).count(norm(h)) for h in headers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("articles", nargs="*", help="article ids (default: all in articles/)")
    ap.add_argument("--layouts", default=",".join(LAYOUTS))
    a = ap.parse_args()
    files = sorted((HERE / "articles").glob("*.yaml"))
    if a.articles:
        files = [f for f in files if f.stem in a.articles]
    for sub in ("tex", "pdf", "pages", "gt"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    for f in files:
        art = yaml.safe_load(f.read_text(encoding="utf-8"))
        gt, md = ground_truth(art)
        for layout in a.layouts.split(","):
            doc = f"{art['id']}__{layout}"
            tex = DATA / "tex" / f"{doc}.tex"
            tex.write_text(render_tex(art, layout), encoding="utf-8")
            pdf = DATA / "pdf" / f"{doc}.pdf"
            compile_pdf(tex, pdf)
            n_pages = render_pages(pdf, DATA / "pages" / doc)
            doc_gt = {**gt, "doc": doc, "layout": layout, "columns": LAYOUTS[layout]["columns"],
                      "headers": LAYOUTS[layout]["headers"], "n_pages": n_pages,
                      "header_occurrences": header_occurrences(pdf, LAYOUTS[layout]["headers"])}
            (DATA / "gt" / f"{doc}.json").write_text(json.dumps(doc_gt, indent=1, ensure_ascii=False), encoding="utf-8")
            (DATA / "gt" / f"{doc}.md").write_text(md, encoding="utf-8")
            print(f"{doc}: {n_pages} pages, {len(gt['blocks'])} text blocks, {len(gt['formulas'])} formulas, "
                  f"{len(gt['footnotes'])} footnotes, {len(gt['captions'])} captions")


if __name__ == "__main__":
    main()
