# pdf2md — engines, decisions and lessons learned

`tools/pdf2md.py` converts PDFs to Markdown with LaTeX math. This document
records which OCR engine to use and why, the measurements behind that choice,
and what went wrong along the way. Server setup for the local engines lives in
[local_ocr.md](local_ocr.md); the benchmark itself in
[benchmarks/ocr/](../benchmarks/ocr/README.md).

## Current recommendation

| Use | Engine | Status in `pdf2md.py` |
|---|---|---|
| Local, default | **LightOnOCR-2-1B** via llama.cpp | not wired yet — next step |
| Local, available today | PaddleOCR-VL 1.6 (`--model paddle`) | wired |
| Local, heavier | dots.ocr (`--model dots`) | wired; superseded |
| Cloud | Mistral OCR (`--model mistral`, default) | wired; workspace blocked since 2026-09-22 |

**LightOnOCR-2** transcribed every benchmark formula correctly, on clean and on
scan-degraded pages, at ~8 s/page with 4.6 GB of VRAM (Apache-2.0). Adopting it
requires a llama.cpp build newer than the one at `~/gitrepos/llama.cpp`
(2026-03-27), which cannot load the model. **GLM-OCR** is the fallback: almost
the same quality and speed, MIT, and it already runs on the current llama.cpp
build.

## Local engine benchmark (2026-09-22)

36 formulas from the arXiv LaTeX sources of DDPM, Attention Is All You Need and
Perelman, 6 pages; scan robustness on 2 pages degraded to 150 dpi / JPEG q40 /
0.7° rotation. FQS is the mean normalized partial match of each formula against
the page output (1.0 = every formula transcribed exactly). Method, ground truth,
scripts and raw outputs: [benchmarks/ocr/](../benchmarks/ocr/README.md).
Hardware: RTX 4070 Laptop 8 GB, WSL2.

| Engine | FQS clean | FQS scan | s/page (median / max) | VRAM peak | License | Runtime |
|---|---:|---:|---:|---:|---|---|
| **LightOnOCR-2-1B** | **1.000** | **1.000** | 8.5 / 16.0 | 4.6 GB | Apache-2.0 | llama.cpp ≥ recent¹ |
| **GLM-OCR** (1.3B) | 0.990 | 0.990 | 8.3 / 14.2 | 4.2 GB | MIT | llama.cpp (current build) |
| dots.ocr (3.0B) | 0.981 | 0.947 | 17.0 / 26.6² | ~6.9 GB | MIT | vLLM 0.11 |
| HunyuanOCR 1.5 (1.1B) | 0.978 | 0.987 | 12.2 / 15.6 | 5.5 GB | Tencent (custom) | llama.cpp ≥ 2026-07-21³ |
| PaddleOCR-VL 1.6 (0.9B) | 0.975 | 1.000 | 11.3 / 12.8² | ~4.5 GB | Apache-2.0 | llama.cpp + paddleocr |
| DeepSeek-OCR-2 (3.4B MoE) | 0.966 | 0.993 | 6.6 / 11.7 | **7.8 GB** | Apache-2.0 | llama.cpp ≥ 2026-05-29 |

¹ Tested with llama.cpp `709fe75` (2026-09-22) and the ggml-org GGUF with the
**Q8_0 projector** (see failure modes). ² Measured through the full `pdf2md.py`
run (process start, PDF rendering, and for Paddle the CPU layout model); the
other rows time only the llama-server request, so these two carry a few
seconds of extra overhead. ³ GGUF converted locally from the official
`tencent/HunyuanOCR` weights (v1.5) with llama.cpp's `convert_hf_to_gguf.py`.

Prompts and sampling follow each model card: LightOnOCR-2 gets the page image
alone (longest side 1540 px, temperature 0.2, top-p 0.9); GLM-OCR the prompt
`Text Recognition:` on the full page (temperature 0); HunyuanOCR its official
Chinese document-parsing prompt (temperature 0, repeat penalty 1.08);
DeepSeek-OCR-2 `<|grounding|>Convert the document to markdown.` with the
`deepseek-ocr` chat template and llama.cpp's DRY settings from its test suite.

### Failure modes observed

- **Footnote formulas dropped by layout-driven models.** The `q·k = Σ q_i k_i`
  footnote on Attention p.4 is missing from PaddleOCR-VL (same as in the August
  study), HunyuanOCR and DeepSeek-OCR-2. LightOnOCR-2 and GLM-OCR keep it.
- **dots.ocr misreads accents and symbols:** `ᾱ` read as `α̃` with `β_t` moved
  inside the square root (DDPM eq. 7), and `d_model/h` read as `d_model/ℏ`. It
  is also the only engine that degrades on scans (0.964 → 0.947 on the same
  pages).
- **GLM-OCR dropped the middle term** of DDPM eq. 3 and wrote `:=` for `=:`.
  It is trained on crops (its official pipeline runs a layout detector first);
  on full pages it works well but can emit a figure caption out of reading
  order.
- **DeepSeek-OCR-2** writes `\bigtriangledown` for `∇` in places and peaks at
  7.8 GB of the 8 GB card — no headroom.
- **LightOnOCR-2 with the f16 vision projector on CUDA emits garbage**
  (`気に気に…` until `max_tokens`, ~96 s/page). The model is fine: the same
  f16 projector on CPU (`--no-mmproj-offload`) and the Q8_0 projector on GPU
  both score 1.000. Use the Q8_0 projector.

### Comparison with the 2026-08-29 study

The August study (31 formulas, report:
<https://claude.ai/code/artifact/db9590a3-8bfa-4465-b311-dd13d768f83f>) chose
dots.ocr for fidelity and PaddleOCR-VL for speed. Its ground truth and scripts
lived only in a session scratchpad and were lost, so the benchmark was rebuilt
from the report's method. The reference engines are consistent with it on the
new formula set (dots 0.995 → 0.981, PaddleOCR-VL 0.966 → 0.975), and
PaddleOCR-VL loses the same footnote formula as before.
The rebuilt benchmark is now versioned in `benchmarks/ocr/`.

## Lessons learned

- **Keep benchmarks in the repo.** Results that live only in a scratchpad
  disappear; `benchmarks/ocr/results/` keeps raw outputs so scores can be
  recomputed when the normalization changes.
- **llama.cpp is the one runtime that covers every candidate** on WSL2. vLLM
  is stuck at 0.11 here (0.28's runner needs UVA, unavailable on WSL2), and
  none of the 2026 OCR models run on 0.11. The newest models need a recent
  llama.cpp: build it in a separate directory so the build used by
  `--model paddle` is not disturbed.
- **`finish_reason=length` on every page is a red flag**, not a slow model:
  check the text before trusting a score of zero.
- **Inspect low scores before concluding.** Two apparent model errors were
  normalization gaps (`\mathrel{\text{:=}}`, `\parallel` for `\|`); the
  evaluator now treats both as equivalent. The remaining low scores were real
  transcription errors.
- `pgrep -f` inside a shell script matches the script's own command line; wait
  on files or task output instead.

## Mistral OCR (cloud)

### 2026-09-22: workspace blocked

Every Mistral inference call (OCR and chat) started returning HTTP 429
`rate_limited` with `x-ratelimit-limit-req-minute: 0` — the *limit* is zero,
not the remaining quota. Retrying cannot help. Ruled out, all normal: API key
(a new key behaves the same), organization limits (625 OCR pages/min),
monthly included usage (US$ 1.13 of US$ 10 on the Free plan), workspace
spending limit (off) and quota rules (none), regional endpoints
(`api.eu.mistral.ai`, `api.us.mistral.ai` also return 0). Authentication,
model listing and file upload keep working. A support ticket is the remaining
path.

To check the state with one request:

```bash
curl -s -o /dev/null -D - https://api.mistral.ai/v1/chat/completions \
  -H "Authorization: Bearer $MISTRAL_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"mistral-small-latest","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}' \
  | grep -i ratelimit
```

### Behaviour of `pdf2md.py` since then

- A 429 whose `x-ratelimit-limit-req-minute` is `0` stops the run at once with
  exit code 2 and a panel pointing to the Mistral console and to the local
  engines; in batch mode the remaining PDFs are not uploaded.
- Transient 429/5xx are retried up to 5 times with backoff, honouring
  `Retry-After` (max 60 s).
- The uploaded PDF is deleted from Mistral storage right after the OCR call,
  whether it succeeds or fails. Before this change every run left a copy
  behind: 597 files (1.1 GB, April 2025 onwards) were found and deleted.
