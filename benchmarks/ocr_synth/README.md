# Synthetic OCR benchmark (prototype)

Academic articles written once as structured YAML, typeset with journal LaTeX classes in one and two
columns, and scored against the converter's Markdown with exact full-page ground truth. Complements
the real-page formula benchmark in [`benchmarks/ocr`](../ocr/README.md). Design and plan: issue #2;
acceptance criteria for replacing the OCR engine: issue #1.

## Layout

| Path | Content |
|---|---|
| `articles/*.yaml` | article sources (fresh en-US prose, inline math, display equations, footnotes, figures) |
| `build.py` | renders each article per layout, compiles, writes page images and ground truth into `data/` |
| `evaluate.py` | scores `OUTPUT_ROOT/<engine>/<doc>.md` (or `<doc>/pNN.md`) against `data/gt/` |
| `controls.py` | writes the `gt`, `pdftotext` and `perturbed` controls that validate the evaluator |
| `data/` | generated, git-ignored: `.tex`, PDFs, 200 dpi pages, ground truth |

Layouts: `elsarticle-1col` (preprint, 12 pt), `elsarticle-2col` (5p), `ieee-2col` (IEEEtran journal).
Elsevier dropped the "Preprint submitted to" footer from `elsarticle` in October 2024, so the
Elsevier layouts get a journal running header (`fancyhdr`) and a first-page DOI footer (`\myfooter`,
test prefix 10.5555), as a published article has.

## Running

```bash
uv run python benchmarks/ocr_synth/build.py                 # all articles x layouts (needs TeX Live, poppler-utils)
uv run python benchmarks/ocr_synth/controls.py
uv run python benchmarks/ocr_synth/evaluate.py benchmarks/ocr_synth/results/controls
```

## Metrics

| Metric | Meaning | Role |
|---|---|---|
| `text` | mean best partial match of each text block (title, abstract, paragraphs), math and markup removed | acceptance |
| `order` | Kendall tau of matched block positions | acceptance |
| `fqs` / `f_err` | FQS from `benchmarks/ocr/evaluate.py`; `f_err` counts formulas below 1.0 | acceptance (`f_err`) |
| `notes`, `captions` | mean match of footnote and caption texts | acceptance |
| `headers` | running headers found / printed on the pages | informative |
| `splits` | text blocks with a running header inside their matched span | must be 0 |

Inline formulas whose normalized LaTeX is shorter than 8 characters are skipped (they match anywhere).

## Control results (a01, 2026-11-05)

| Control | text | order | f_err (of 39) | notes | headers | splits |
|---|---:|---:|---:|---:|---:|---:|
| `gt` | 1.000 | 1.000 | 0 | 1.000 | 0/n | 0 |
| `pdftotext` (2-col) | 0.91–0.93 | 0.90–0.92 | 35 | 0.90 | n/n | 0 |
| `perturbed` | 0.99 | 0.98 | 1 | 0.75 | 1/n | 1 |

Each perturbation (two paragraphs swapped, one footnote dropped, one wrong symbol, a header inside
a paragraph) moves the metric meant to catch it. The mean FQS of `perturbed` still rounds to 1.000,
which is why acceptance uses `f_err`.

## Not yet done

No OCR engine has been run (this machine has no GPU). Pending: more articles (~20), `revtex4-2` and
`acmart` layouts, tables, scan-degraded variants, a runner for `pdf2md.py`, and the check that the
benchmark separates LightOnOCR-2, GLM-OCR and DeepSeek-OCR-2.
