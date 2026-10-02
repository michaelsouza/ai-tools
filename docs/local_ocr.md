# Local OCR engine for pdf2md.py

`tools/pdf2md.py --model lighton` runs **LightOnOCR-2-1B** locally through
llama.cpp's `llama-server`. It was chosen in the 2026-09-22 benchmark (every
test formula transcribed correctly, ~8 s/page, 4.6 GB of VRAM, Apache-2.0);
see [pdf2md.md](pdf2md.md) for the comparison and the engines it replaced.

**Hardware.** Everything here was set up and tuned on one machine: NVIDIA
RTX 4070 Laptop GPU (8 GB) and Intel Core i9-14900HX, CUDA through WSL2
(Ubuntu). The server defaults (Q8_0 language model, 4 parallel slots) come
from measurements on it and are not universal; on other hardware re-measure
them as described in
[pdf2md.md, Throughput tuning](pdf2md.md#throughput-tuning-2026-10-02).

## Setup

**llama.cpp.** LightOnOCR-2 needs a recent build; the 2026-03-27 build could
not load it. Tested with commits `709fe75` and `f46bc30` (2026-09-22). Build
with CUDA for the RTX 4070 (compute capability 8.9):

```bash
git -C ~/gitrepos/llama.cpp pull --ff-only
cmake -S ~/gitrepos/llama.cpp -B ~/gitrepos/llama.cpp/build-cuda -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/gitrepos/llama.cpp/build-cuda -j 24
```

`tools/ocr_server.sh` expects `~/gitrepos/llama.cpp/build-cuda/bin/llama-server`
(override with `LLAMA_SERVER_BIN`).

**Weights** (gitignored, `models/lightonocr2/`), from
[ggml-org/LightOnOCR-2-1B-GGUF](https://huggingface.co/ggml-org/LightOnOCR-2-1B-GGUF):

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
for f in ['LightOnOCR-2-1B-f16.gguf', 'mmproj-LightOnOCR-2-1B-Q8_0.gguf']:
    hf_hub_download('ggml-org/LightOnOCR-2-1B-GGUF', f, local_dir='models/lightonocr2')"
```

The server uses a Q8_0 copy of the language model (faster, same benchmark
score as f16); create it from the f16 file:

```bash
~/gitrepos/llama.cpp/build-cuda/bin/llama-quantize \
  models/lightonocr2/LightOnOCR-2-1B-f16.gguf models/lightonocr2/LightOnOCR-2-1B-Q8_0.gguf Q8_0
```

`LIGHTON_MODEL=models/lightonocr2/LightOnOCR-2-1B-f16.gguf tools/ocr_server.sh start lighton`
runs the f16 model instead.

Use the **Q8_0 vision projector**. The f16 projector (`mmproj-…-f16.gguf`)
produces garbage when the vision encoder runs on CUDA (`気に気に…` until the
token limit); the f16 language model is fine, Q8_0 too.

## Running

```bash
tools/ocr_server.sh start lighton   # llama-server on port 8093, 4 slots, ~10 s startup, ~5.9 GB VRAM peak
tools/ocr_server.sh status
python tools/pdf2md.py paper.pdf --model lighton -y
tools/ocr_server.sh stop lighton
```

The server holds the GPU while it runs; stop it before other GPU work.
`pdf2md.py` checks `/health` before starting and points to the start command
if the server is down (`--server-url` selects another server).

What `pdf2md.py` sends, per the model card: each page rendered at 200 dpi and
downscaled to 1540 px on the longest side, the image alone (no text prompt),
temperature 0.2, top-p 0.9, up to 6144 output tokens. Pages go to the server
4 at a time (`--workers`, default 4, matching the server's slots), for
~6 s/page on the hardware above.

A page that stops on the token limit is almost always a repetition loop
(e.g. dotted equation leaders); it is retried once with `repeat_penalty` and
the retry is kept. If the retry also hits the limit, a warning is printed —
that is also how broken output (such as the f16 projector bug) shows up.

**Server settings** (env vars of `tools/ocr_server.sh`): `LIGHTON_SLOTS`
(default 4) and `LIGHTON_CTX` (default 36864, split evenly across slots; each
slot needs ~2.3k image tokens plus 6144 output tokens, so keep ≥ 8.5k per
slot). Keep `--workers` equal to `LIGHTON_SLOTS`. On the 8 GB card, 8 slots
run out of VRAM; with less VRAM use fewer slots.

To run it on a Colab GPU instead, see [colab_ocr.md](colab_ocr.md).

## History

Until 2026-09-22 `pdf2md.py` also had `--model dots` (dots.ocr on vLLM 0.11),
`--model paddle` (PaddleOCR-VL 1.6 on llama.cpp plus the paddleocr layout
pipeline) and `--model nougat`. They were removed after the benchmark in
[pdf2md.md](pdf2md.md); the 2026-08-29 study that introduced dots and paddle
is at <https://claude.ai/code/artifact/db9590a3-8bfa-4465-b311-dd13d768f83f>.
