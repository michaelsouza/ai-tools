# OCR formula benchmark

Measures how faithfully an OCR engine transcribes math formulas from
academic papers, to pick the local engine for `tools/pdf2md.py`. Results and
decisions: [docs/pdf2md.md](../../docs/pdf2md.md).

## Method

- **Corpus:** 6 math-dense pages from 3 arXiv papers — DDPM (2006.11239,
  pages 2–3), Attention Is All You Need (1706.03762, pages 4–5, including the
  `q·k` footnote) and Perelman (math/0211159, pages 15–16, section 7).
- **Ground truth:** [gt.json](gt.json), 36 formulas copied from the official
  arXiv LaTeX sources with the papers' macros expanded.
- **Metric (FQS):** for each formula, the best partial alignment
  (`rapidfuzz.fuzz.partial_ratio`) of the normalized LaTeX against the
  normalized predicted page; the score is the mean over formulas. The
  normalization in [evaluate.py](evaluate.py) removes meaning-free variation:
  font and class wrappers (`\mathbf`, `\bm`, `\text`, `\mathrel`…), sizing
  (`\left`, `\bigg`…), spacing, braces, unicode vs LaTeX, `\le`/`\leq`,
  `⟨⟩`/`<>`, `:=`/`\coloneqq`, `\|`/`\Vert`/`\parallel`, and sub/superscript
  order (`W^Q_i` = `W_i^Q`). Controls: equivalent LaTeX scores 1.00, the PDF
  text layer 0.82, a single wrong symbol 0.93, an absent formula ~0.24.
- **Robustness:** pages DDPM 3 and Perelman 16 degraded as a scan (150 dpi,
  JPEG q40, 0.7° rotation), 13 formulas, same ground truth.
- **Speed and memory:** wall time per page and peak VRAM (llama-server runs).

## Running

```bash
python benchmarks/ocr/prepare.py                         # download papers, render pages into data/
python benchmarks/ocr/run_pdf2md.py dots                  # engines already wired into pdf2md
python benchmarks/ocr/run_pdf2md.py paddle
python benchmarks/ocr/run_llama.py glm-ocr --set clean    # candidates served by llama-server
python benchmarks/ocr/run_llama.py glm-ocr --set scan
python benchmarks/ocr/evaluate.py benchmarks/ocr/results/outputs --json /tmp/clean.json
python benchmarks/ocr/evaluate.py benchmarks/ocr/results/outputs_scan --pages ddpm_p3,per_p16
```

`run_llama.py` reads GGUF files from `OCR_BENCH_MODELS` (default
`models/ocr-bench/<engine>/`) and the server binary from `LLAMA_SERVER_BIN`;
the engine table at the top of the script lists the exact files, prompts and
sampling parameters. Run one engine at a time: the GPU has 8 GB.

`results/` keeps the raw Markdown produced by every engine, so the scores can
be recomputed after changing the normalization.

## Limitations

- Small corpus (36 formulas, 6 pages), all arXiv; the scan test only
  simulates degradation.
- FQS measures formula transcription, not document structure (reading order,
  tables, figures).
- The 2026-08-29 study used a different set of 31 formulas; its numbers are
  not directly comparable with these.
