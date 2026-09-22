"""Run the production pdf2md engines (dots, paddle) over the benchmark pages.

Usage: python benchmarks/ocr/run_pdf2md.py ENGINE   (ENGINE in dots|paddle)
Starts the engine server with tools/ocr_server.sh, converts every clean and
scan page PDF with tools/pdf2md.py, records wall time, then stops the server.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PY = REPO / ".venv/bin/python"

engine = sys.argv[1]
subprocess.run([str(REPO / "tools/ocr_server.sh"), "start", engine], check=True)
try:
    for set_name, pages_dir, out_root in (("clean", "pages", "outputs"), ("scan", "pages_scan", "outputs_scan")):
        out_dir = HERE / "results" / out_root / engine
        out_dir.mkdir(parents=True, exist_ok=True)
        timings = {}
        for pdf in sorted((HERE / "data" / pages_dir).glob("*.pdf")):
            md = out_dir / f"{pdf.stem}.md"
            md.unlink(missing_ok=True)
            t = time.time()
            r = subprocess.run([str(PY), str(REPO / "tools/pdf2md.py"), str(pdf), "--model", engine, "-y", "--no-preview", "-o", str(md)],
                               capture_output=True, text=True, cwd=REPO)
            dt = time.time() - t
            timings[pdf.stem] = {"seconds": round(dt, 1), "exit": r.returncode}
            print(f"{engine} {set_name} {pdf.stem}: {dt:.1f}s exit={r.returncode}", flush=True)
            if r.returncode != 0:
                (out_dir / f"{pdf.stem}.err").write_text(r.stdout[-3000:] + r.stderr[-3000:], encoding="utf-8")
        json.dump({"timings": timings}, open(out_dir / "run.json", "w"), indent=2)
finally:
    subprocess.run([str(REPO / "tools/ocr_server.sh"), "stop", engine])
