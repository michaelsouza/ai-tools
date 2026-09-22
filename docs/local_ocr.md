# Local OCR engines for pdf2md.py

`tools/pdf2md.py` supports two local OCR engines as alternatives to the Mistral
cloud API, selected after an empirical study (2026-08-29) focused on academic
papers with complex math formulas. Full report:
<https://claude.ai/code/artifact/db9590a3-8bfa-4465-b311-dd13d768f83f>

> A newer benchmark (2026-09-22) and the current engine recommendation are in
> [pdf2md.md](pdf2md.md). This page covers setup and operation of the two
> engines already wired into `pdf2md.py`.

## Study summary

Formula Quality Score (FQS): best partial alignment of normalized LaTeX against
ground truth taken from arXiv LaTeX sources — 31 formulas across 6 math-dense
pages (DDPM, Attention Is All You Need, Perelman). Hardware: RTX 4070 Laptop
8 GB VRAM, WSL2.

| Engine | FQS | Scan robustness | s/page | License |
|---|---|---|---|---|
| **dots.ocr** (vLLM) | **0.995** | −0.013 | 21–39 | MIT |
| Mistral OCR (cloud reference) | 0.994 | +0.006 | 2–6 | paid API |
| Nougat (legacy) | 0.992¹ | −0.003 | 5–22 | CC BY-NC |
| marker 1.8.3 | 0.968 | **−0.307** | 9–59 | GPL + revenue clause |
| **PaddleOCR-VL 1.6** (llama.cpp) | 0.966 | +0.002 | **8–13** | Apache-2.0 |
| MinerU 2.5 (vlm-engine) | 0.916 | — | 37–58 | AGPL-3 |

¹ The corpus is inside Nougat's training distribution (arXiv), so its score
overstates generalization; it also carries a non-commercial license.

Chosen engines:

- `--model dots` — **dots.ocr**: best local quality, on par with Mistral for
  formulas. Served by vLLM.
- `--model paddle` — **PaddleOCR-VL 1.6**: best speed/robustness balance,
  served by the llama.cpp already on this machine (VL model on GPU via
  `llama-server`; the small layout model runs on CPU through the paddleocr
  pipeline).

## Setup

Weights live in `models/` (gitignored):

- `models/DotsOCR/` — HF `rednote-hilab/dots.ocr` snapshot. Two local fixes
  are applied and must be preserved:
  - the directory name has no dot (`DotsOCR`), because transformers' dynamic
    module import breaks on dots in the repo name;
  - `config.json` has `vision_config.attn_implementation` set to `"sdpa"` —
    upstream ships `"flash_attention_2"` and silently falls back to *eager*
    when flash-attn is absent (2.3× slower and OOM-prone; output is
    bit-identical under sdpa).
- `models/paddleocr-vl-gguf/` — official `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`
  (`model.gguf`, `mmproj.gguf`, `chat_template.jinja`).

Engine environments (repo root, gitignored):

```bash
# dots.ocr server (vLLM)
uv venv .venv-vllm -p 3.12
uv pip install -p .venv-vllm/bin/python "vllm==0.11.0" "transformers==4.56.2" ninja

# PaddleOCR-VL pipeline (layout on CPU; recognition goes to llama-server)
uv venv .venv-paddle -p 3.12
uv pip install -p .venv-paddle/bin/python paddlepaddle "paddleocr[doc-parser]>=3.6.0" python-docx
```

Version pins that matter (all failure modes observed on WSL2):

- **vLLM must be 0.11.x** — the 0.28 GPU runner requires UVA, which is not
  available on WSL2 (`RuntimeError: UVA is not available`).
- **transformers must be ≤4.56.x** in the vLLM env — newer versions removed
  `all_special_tokens_extended`, which vLLM 0.11 still uses.
- `ninja` must be on the PATH of the vLLM process (the server script handles
  this).
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on WSL2 — it
  causes `CUDA driver error: device not ready`.

## Running

```bash
tools/ocr_server.sh start dots     # vLLM, port 8092, ~50s startup, ~6.9 GB VRAM
tools/ocr_server.sh start paddle   # llama.cpp, port 8091, ~20s startup, ~4.5 GB VRAM
tools/ocr_server.sh status
tools/ocr_server.sh stop {dots|paddle}

python tools/pdf2md.py paper.pdf --model dots -y
python tools/pdf2md.py paper.pdf --model paddle -y
```

Both engines share the 8 GB GPU — run one server at a time.

The vLLM flags in `tools/ocr_server.sh` are tuned to the 8 GB budget (weights
alone are 5.7 GB bf16 — the model is 2.8B params total; "1.7B" refers only to
its language model): `--gpu-memory-utilization 0.84 --enforce-eager
--max-num-seqs 1 --max-model-len 6144 --mm-processor-kwargs
'{"max_pixels":1300000}'`. Raising any of these on this machine causes startup
OOM (only ~6.9 GB of the 8 GB are free under WSL2).

`llama-server` is expected at `~/gitrepos/llama.cpp/build-cuda/bin/llama-server`
(override with `LLAMA_SERVER_BIN`). The paddleocr CLI is resolved from
`.venv-paddle/bin/paddleocr` (override with `PADDLEOCR_BIN`).
