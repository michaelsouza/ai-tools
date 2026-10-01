# Running pdf2md on Google Colab

Notebook: [notebooks/pdf2md_colab.ipynb](../notebooks/pdf2md_colab.ipynb). It runs
`tools/pdf2md.py --model lighton` against several `llama-server` processes on a
Colab GPU. Engine choice and benchmark: [pdf2md.md](pdf2md.md); local setup:
[local_ocr.md](local_ocr.md). Findings below are from the 2026-09-30/10-01
session on a Colab Pro **A100-SXM4-80GB** (compute capability 8.0, high-RAM
runtime, 12 vCPUs).

## Setup

- Drive folder `Colab Notebooks/PDF2MD/` holds `pdf2md.py` (copy of
  `tools/pdf2md.py`, refresh it after every change), `pdfs/`, `out/` and the
  `llama-server-bin/` cache. The notebook does not clone the repository (its
  remote is SSH).
- **Accelerators.** The runtime menu lists CPU, T4, L4, A100, G4, H100 (greyed
  out when unavailable) and TPUs. Any NVIDIA GPU works. TPUs do not: llama.cpp
  has no TPU backend. The model is 1B parameters, so L4 or T4 is enough and
  cheaper in compute units; the A100 is not needed for a single PDF.
- **llama.cpp is built from source** with `-DGGML_CUDA=ON` and
  `-DCMAKE_CUDA_ARCHITECTURES` taken from `nvidia-smi --query-gpu=compute_cap`
  (80 on the A100). A `--depth 1` clone of master loaded LightOnOCR-2 fine
  (no pinned commit needed). The build takes ~10 to 15 min.
- **Binary cache on Drive** (`llama-server-bin/`: `llama-server`, the `*.so`
  files and `arch.txt`). It is only valid for the same compute capability, so
  the cell rebuilds when `arch.txt` differs or is missing. Model weights are
  not cached: `/content` is wiped with the session and they are downloaded
  again (use the Q8_0 vision projector, see local_ocr.md).

## Pitfalls hit

- **The Drive path contains a space** (`Colab Notebooks`). Quote every
  `{DRIVE_DIR}` used in a `!` command; in Python code use `shutil`/`os`
  instead of `!cp`. An unquoted `!cp a "{x}/"` split the path and silently
  left the cache empty.
- **`!pkill -f llama-server` kills its own shell** (the `bash -c` command line
  matches the pattern; exit code 144). Use
  `subprocess.run(['pkill', '-f', 'llama-server'])`.
- **Interrupting a cell kills `Popen` children.** Start servers with
  `start_new_session=True`, otherwise stopping a conversion takes the servers
  down and the next run fails with "Connection refused".
- **A defunct (`<defunct>`) `llama-server` in `ps` means the process died**;
  `/health` then fails and `pdf2md.py` exits with "no local server".
- **Colab's "Gemini" cell suggestions** show up as a red/green diff over the
  cell. They are not from the repository or the agent; reject them, since they
  rename variables the other cells use.
- **The "Download .py" export of a notebook is not a script** (it keeps the
  `!` magics). Never save it as `pdf2md.py` in the Drive folder: it would
  overwrite the real script.

## Performance

Detail and numbers are in the "Parallel OCR" section of
[pdf2md.md](pdf2md.md). What matters for Colab:

- One `llama-server` is CPU-bound; running 4 processes with 4 slots each
  (`--server-url` with four URLs, `--workers 16`) did the 542-page PDF in
  14 min 55 s of OCR, against a progress-bar ETA of ~24 min for one process with
  8 slots (that run was interrupted, so the figure is an estimate).
- Each process used ~9 GB of VRAM (37 GB for four), and the GPU was never the
  limit, so there is room for more processes; `NS = 6` was not tested.
- Per-page latency inside the server is 20 to 40 s (~33 tokens/s per slot), so
  short PDFs are dominated by the waves of pages in flight: 27 pages took
  1 min 10 s. The parallelism pays off on long documents.
- The notebook terminal tab is useful for monitoring while a cell runs:
  `curl -s localhost:8093/slots` shows how many slots are busy, and
  `grep "eval time" /content/llama-server-8093.log` the time per request.
- Stop the servers (last notebook cell) and disconnect the runtime when done,
  to release the GPU and stop spending compute units.

## Hardware note

The A100 in Colab is the SXM4 module (NVLink baseboard, ~400 W), which cannot be
installed in a desktop. The A100 PCIe variant fits a PCIe x16 slot but is
passively cooled and uses an EPS power connector, so it needs server airflow.
For a desktop the practical choices are RTX-class cards; the local RTX 4070
(12 GB) already runs this model with room to spare (~4.6 GB).
