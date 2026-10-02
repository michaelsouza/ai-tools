# pdf2md — engines, decisions and lessons learned

`tools/pdf2md.py` converts PDFs to Markdown with LaTeX math. This document
records which OCR engine to use and why, the measurements behind that choice,
and what went wrong along the way. Setup of the local engine lives in
[local_ocr.md](local_ocr.md); the benchmark itself in
[benchmarks/ocr/](../benchmarks/ocr/README.md).

## Engines

| `--model` | Engine | Where |
|---|---|---|
| `lighton` | **LightOnOCR-2-1B** via llama.cpp (`tools/ocr_server.sh start lighton`) | local — the only local engine |
| `mistral` (default) | Mistral OCR | cloud — workspace blocked since 2026-09-22 |

**LightOnOCR-2** transcribed every benchmark formula correctly, on clean and on
scan-degraded pages, at ~8 s/page with 4.6 GB of VRAM (Apache-2.0). It needs a
recent llama.cpp (the 2026-03-27 build could not load it) and the Q8_0 vision
projector. **GLM-OCR** is the documented fallback if LightOn ever breaks:
almost the same quality and speed, MIT, and it runs on older llama.cpp builds.

On 2026-09-22 the previous local engines were removed from `pdf2md.py`:
`dots` (dots.ocr on vLLM), `paddle` (PaddleOCR-VL 1.6) and `nougat`. Their
benchmark numbers stay below for reference.

Since 2026-10-02 the server runs a **Q8_0 language model with 4 slots** and
`pdf2md.py` sends 4 pages at a time (`--workers`): ~6 s/page end to end
instead of ~12 s, same benchmark score. Those defaults were tuned for one
machine; see [Throughput tuning](#throughput-tuning-2026-10-02).

## Throughput tuning (2026-10-02)

**Hardware and software of every number in this section** — the settings are
not universal; re-measure on other hardware:

- GPU: NVIDIA GeForce RTX 4070 **Laptop** GPU, 8 GB, driver 32.0.16.1714
  (Windows), CUDA through **WSL2** (Ubuntu, kernel 6.6.87.2-microsoft-standard-WSL2).
- CPU: Intel Core i9-14900HX (32 threads); RAM: 64 GB on the host, 31 GB
  visible to WSL2. llama-server's decode loop runs on one core, so
  single-thread CPU speed matters as much as the GPU (see below).
- llama.cpp `f46bc30` (2026-09-22), CUDA build for compute capability 8.9.
- Laptop GPU on battery/thermal limits: repeated runs of the same
  configuration varied by ~20%, so configurations were compared back to back.

**Method.** A throwaway script sent the same 8 pages (manea2016, pp. 3–10,
rendered once) to llama-server at 1, 2, 4 and 8 concurrent requests, with the
model-card sampling plus a fixed seed: every run produced the same 10,321
tokens, so tokens/s compare like for like.

| Server configuration | 1 page at a time | 4 concurrent | s/page at 4 |
|---|---:|---:|---:|
| f16 LM, unified KV (`-c 32768`) | 105 tok/s | 159 tok/s | 8.1 |
| f16 LM, per-slot KV (`-np 4`) | 107 tok/s | 170–200 tok/s | 6.4–7.6 |
| f16 LM + `--backend-sampling` | 101 tok/s | 192 tok/s | 6.7 |
| **Q8_0 LM, per-slot KV** | 137 tok/s | 220–233 tok/s | 5.5–5.9 |
| Q8_0 LM + `--backend-sampling` | 132 tok/s | 225 tok/s | 5.7 |
| Q8_0 LM, 8 slots (`-np 8`) | — | 61–153 tok/s | 11.5–29 |

End to end with `pdf2md.py` (rendering included), manea2016 (17 pages):
221 s with the old setup (f16, one page at a time) → **107 s** with the
current defaults; peak VRAM 5.9 GB.

**Quality.** The Q8_0 language model scores the same as f16 on the formula
benchmark (FQS 1.000 clean, 0.997 scan; same pages, both runs through
`pdf2md.py`). The Q8_0 *vision projector* was already in use; the f16
projector remains broken on CUDA.

**Why 4 concurrent pages give ~1.7×, not 4×.** During decoding the
llama-server main thread sits at 100% of one CPU core while the GPU is at
~72%: the loop is CPU-bound (kernel launches and per-slot sampling over a
~151k-token vocabulary), and that cost grows with every slot. Ruled out:

- CUDA graphs — `GGML_CUDA_DISABLE_GRAPHS=1` leaves single-stream speed
  unchanged (104 → 103 tok/s), so losing them in batched decoding is not it.
- Unified KV cache — per-slot KV (`-np 4`) helps a little, within noise.
- GPU sampling (`--backend-sampling`, experimental) — no gain.
- Memory bandwidth — Q8_0 halves the weights read per token but speeds a
  single stream up only ~25%.

**8 slots do not fit**: 8 GB fill up (7.9 GB) and throughput collapses; the
output also changed (14,120 tokens instead of 10,321).

**What to re-check on other hardware:** a faster CPU core raises the ceiling;
a GPU with more VRAM may take more slots (`LIGHTON_SLOTS` in
`tools/ocr_server.sh`, `--workers` in `pdf2md.py`, keep them equal and give
each slot ≥ 8.5k tokens via `LIGHTON_CTX`); on native Linux the CPU overhead
of kernel launches may be lower than under WSL2.

## Problems found converting a real bibliography (2026-10-02)

19 papers, 357 pages (numerical linear algebra, reservoir simulation).

- **Repetition loops on dotted equation leaders.** SPE Journal papers put a
  dotted line between each equation and its number; LightOnOCR-2 sometimes
  transcribes it as `\cdots \quad \cdots …` until the 6144-token limit, and
  the rest of the page is lost (3 of 20 pages in two papers, 13–14k-character
  lines). It happens with one page at a time too, so it is not caused by
  concurrency, and sampling makes it intermittent. `pdf2md.py` now retries a
  page that stops on the token limit once with `repeat_penalty 1.1,
  repeat_last_n 256` and keeps the retry unless it also hits the limit. On
  the failing page the retry transcribed the full page and its equations
  matched the PDF. **DRY sampling was rejected**: it also broke the loop but
  dropped a repeated matrix from an equation (`\begin{pmatrix} A & -B \\ …`
  appears twice in the source). The penalty is applied only on the retry, so
  normal pages keep the model-card sampling.
- **AES-encrypted PDFs** (some SPE/Elsevier downloads) fail in PyPDF2 with
  `PyCryptodome is required for AES`; `pycryptodome` is now in
  `requirements.txt`.

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
| **GLM-OCR** (1.3B) | 0.990 | 0.990 | 8.3 / 14.2 | 4.2 GB | MIT | llama.cpp ≥ 2026-02-18 |
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
  llama.cpp; to evaluate a candidate without disturbing the build in use,
  compile a second copy in a separate directory first.
- **`finish_reason=length` on every page is a red flag**, not a slow model:
  check the text before trusting a score of zero.
- **Inspect low scores before concluding.** Two apparent model errors were
  normalization gaps (`\mathrel{\text{:=}}`, `\parallel` for `\|`); the
  evaluator now treats both as equivalent. The remaining low scores were real
  transcription errors.
- `pgrep -f` / `pkill -f` match the calling shell's own command line: a
  `pgrep -f pattern` wait loop never ends and `pkill -f pattern` kills the
  shell that ran it. Wait on files, PIDs (`$!`) or task output instead.
- **Laptop GPU timings drift ~20% between runs** (thermal and power state);
  compare configurations back to back, and use a fixed seed so every run
  generates the same tokens.

## Parallel OCR (`--workers`, `--server-url`) — Colab A100, 2026-09-30

`--model lighton` can keep several pages in flight: `--workers N` sends N page
requests at once, and `--server-url URL1,URL2,...` spreads the pages round-robin
over several `llama-server` processes. Rendering runs on the main thread and
feeds the workers page by page, so it overlaps with the OCR. Output order is
preserved; the first failed page cancels the rest and exits with code 1.

Measured on a Colab A100 80 GB with the 542-page `crippen1988distance.pdf`
(LightOnOCR-2-1B f16 + Q8_0 projector):

| Setup | Result |
|---|---|
| 1 server, 1 request at a time | ~2.4 s/page (~22 min, extrapolated) |
| 1 server, `-np 8`, `--workers 8` | ~2.9 s/page at the start, no visible gain |
| 4 servers x `-np 4`, `--workers 16` | 14 min 55 s of OCR (+ 1 min 30 s rendering, before the overlap change) |

Text-only decode benchmark (`/completion`, 256 tokens, 8 concurrent requests):
350 tokens/s with one server, 870 tokens/s with three. Sampler settings
(`top_p`, greedy, `samplers: [temperature]`) made no difference.

**Lesson: one `llama-server` process is CPU-bound, not GPU-bound.** With 8 slots
the step time rose from ~5 ms to ~22 ms, GPU utilisation stayed near 30% and the
process used a single CPU thread (Colab CPUs are slow). More slots in the same
process barely help; more processes do. Each server needs `-c N x 16384` for N
slots (2371 image tokens + up to 6144 output tokens per page) and used ~9 GB of
VRAM with 4 slots. Start servers with `start_new_session=True` so that
interrupting a notebook cell does not kill them. Colab setup and pitfalls:
[colab_ocr.md](colab_ocr.md).

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
  exit code 2 and a panel pointing to the Mistral console and to
  `--model lighton`; in batch mode the remaining PDFs are not uploaded.
- Transient 429/5xx are retried up to 5 times with backoff, honouring
  `Retry-After` (max 60 s).
- The uploaded PDF is deleted from Mistral storage right after the OCR call,
  whether it succeeds or fails. Before this change every run left a copy
  behind: 597 files (1.1 GB, April 2025 onwards) were found and deleted.
