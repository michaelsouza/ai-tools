"""Run one OCR model through llama-server over every benchmark page.

Usage: python benchmarks/ocr/run_llama.py ENGINE [--set clean|scan]
Writes results/outputs[_scan]/<engine>/<page>.md plus run.json (timings, VRAM peak).

Environment:
    LLAMA_SERVER_BIN   llama-server binary (default ~/gitrepos/llama.cpp/build-cuda/bin/llama-server).
                       GLM-OCR and LightOnOCR-2 run on llama.cpp >= 2026-02-18; HunyuanOCR 1.5 and
                       DeepSeek-OCR-2 need a build from 2026-07-21 or later.
    OCR_BENCH_MODELS   directory with one sub-folder of GGUF files per engine (default models/ocr-bench).
"""

import argparse
import base64
import io
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
REPO = HERE.parents[1]
SERVER = Path(os.environ.get("LLAMA_SERVER_BIN", Path.home() / "gitrepos/llama.cpp/build-cuda/bin/llama-server"))
M = Path(os.environ.get("OCR_BENCH_MODELS", REPO / "models/ocr-bench"))
PORT = 8095
HUNYUAN_PROMPT = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"

ENGINES = {
    "glm-ocr": dict(
        args=["-m", M / "glm-ocr/GLM-OCR-f16.gguf", "--mmproj", M / "glm-ocr/{mmproj_glm}", "-c", "16384"],
        prompt="Text Recognition:", params=dict(temperature=0, max_tokens=8192), max_side=None,
    ),
    "lightonocr2": dict(
        # The f16 projector produces garbage on CUDA (llama.cpp 709fe75); the Q8_0 projector works on GPU.
        args=["-m", M / "lightonocr2/LightOnOCR-2-1B-f16.gguf", "--mmproj", M / "lightonocr2/mmproj-LightOnOCR-2-1B-Q8_0.gguf", "-c", "16384"],
        prompt=None, params=dict(temperature=0.2, top_p=0.9, max_tokens=6144), max_side=1540,
    ),
    "deepseek-ocr2": dict(
        args=["-m", M / "deepseek-ocr2/{ds_model}", "--mmproj", M / "deepseek-ocr2/{ds_mmproj}", "-c", "16384",
              "--chat-template", "deepseek-ocr", "--no-jinja", "--flash-attn", "off",
              "--dry-multiplier", "0.8", "--dry-base", "1.75", "--dry-allowed-length", "2",
              "--dry-penalty-last-n", "64", "--dry-sequence-breaker", "none"],
        prompt="<|grounding|>Convert the document to markdown.", params=dict(temperature=0, max_tokens=8192), max_side=None,
    ),
    "hunyuanocr15": dict(
        args=["-m", M / "hunyuan-gguf/HunyuanOCR-1.5-F16.gguf", "--mmproj", M / "hunyuan-gguf/mmproj-HunyuanOCR-1.5-F16.gguf", "-c", "24576"],
        prompt=HUNYUAN_PROMPT, params=dict(temperature=0, top_p=1.0, repeat_penalty=1.08, max_tokens=8000), max_side=None,
    ),
}


def resolve(engine: str) -> list:
    cfg = ENGINES[engine]
    subs = {}
    if engine == "deepseek-ocr2":
        ds = M / "deepseek-ocr2"
        subs["ds_model"] = next(p.relative_to(ds) for p in ds.rglob("*.gguf") if "mmproj" not in p.name.lower())
        subs["ds_mmproj"] = next(p.relative_to(ds) for p in ds.rglob("*.gguf") if "mmproj" in p.name.lower())
    if engine == "glm-ocr":
        subs["mmproj_glm"] = next(p.name for p in (M / "glm-ocr").rglob("mmproj*.gguf"))
    return [str(a).format(**subs) for a in cfg["args"]]


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def health() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def image_data_url(path: Path, max_side: int | None) -> str:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean(engine: str, text: str) -> str:
    if engine == "deepseek-ocr2":
        text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.S)
        text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=ENGINES)
    ap.add_argument("--set", default="clean", choices=["clean", "scan"])
    a = ap.parse_args()
    cfg = ENGINES[a.engine]
    pages_dir = HERE / "data" / ("pages" if a.set == "clean" else "pages_scan")
    out_dir = HERE / "results" / ("outputs" if a.set == "clean" else "outputs_scan") / a.engine
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(pages_dir.glob("*.png"))

    base = gpu_used_mib()
    log = open(out_dir / "server.log", "w")
    proc = subprocess.Popen([str(SERVER), *resolve(a.engine), "--port", str(PORT), "-ngl", "99", "--host", "127.0.0.1"],
                            stdout=log, stderr=subprocess.STDOUT)
    peak = [base]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], gpu_used_mib())
            except Exception:
                pass
            time.sleep(0.5)

    threading.Thread(target=sample, daemon=True).start()
    try:
        t0 = time.time()
        while not health():
            if proc.poll() is not None:
                raise SystemExit(f"server exited early, see {out_dir/'server.log'}")
            if time.time() - t0 > 300:
                raise SystemExit("server did not become healthy in 300 s")
            time.sleep(2)
        timings = {}
        for page in pages:
            content = [{"type": "image_url", "image_url": {"url": image_data_url(page, cfg["max_side"])}}]
            if cfg["prompt"]:
                content.append({"type": "text", "text": cfg["prompt"]})
            body = {"messages": [{"role": "user", "content": content}], **cfg["params"]}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            t = time.time()
            with urllib.request.urlopen(req, timeout=1800) as r:
                resp = json.loads(r.read())
            dt = time.time() - t
            msg = resp["choices"][0]["message"]["content"] or ""
            finish = resp["choices"][0].get("finish_reason")
            (out_dir / f"{page.stem}.md").write_text(clean(a.engine, msg), encoding="utf-8")
            timings[page.stem] = {"seconds": round(dt, 1), "finish": finish,
                                  "completion_tokens": resp.get("usage", {}).get("completion_tokens")}
            print(f"{a.engine} {a.set} {page.stem}: {dt:.1f}s finish={finish} tokens={timings[page.stem]['completion_tokens']}", flush=True)
        json.dump({"timings": timings, "vram_peak_mib": peak[0], "vram_base_mib": base},
                  open(out_dir / "run.json", "w"), indent=2)
        print(f"{a.engine}: VRAM peak {peak[0]} MiB (baseline {base} MiB)", flush=True)
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
