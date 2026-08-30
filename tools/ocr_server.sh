#!/bin/bash
# Manage local OCR inference servers used by pdf2md.py.
#
# Usage:
#   tools/ocr_server.sh start dots     # vLLM serving dots.ocr (port 8092)
#   tools/ocr_server.sh start paddle   # llama.cpp serving PaddleOCR-VL 1.6 (port 8091)
#   tools/ocr_server.sh stop  {dots|paddle}
#   tools/ocr_server.sh status
#
# Both engines share the single 8 GB GPU; run one at a time.
# Overridable env vars:
#   LLAMA_SERVER_BIN  path to llama-server (default: ~/gitrepos/llama.cpp/build-cuda/bin/llama-server)
#   DOTS_PORT         default 8092
#   PADDLE_PORT       default 8091
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO/.ocr-servers"
mkdir -p "$RUN_DIR"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/gitrepos/llama.cpp/build-cuda/bin/llama-server}"
DOTS_PORT="${DOTS_PORT:-8092}"
PADDLE_PORT="${PADDLE_PORT:-8091}"

health() { curl -s --max-time 3 "http://127.0.0.1:$1/health" > /dev/null 2>&1; }

start_dots() {
    if health "$DOTS_PORT"; then echo "dots: already running on port $DOTS_PORT"; return 0; fi
    # WSL2/8GB tuning (see docs/local_ocr.md): vLLM 0.11 required (0.28's UVA
    # runner does not work on WSL2); weights are 5.7 GB bf16 so the memory
    # budget below is deliberately tight.
    setsid nohup env PATH="$REPO/.venv-vllm/bin:$PATH" "$REPO/.venv-vllm/bin/vllm" serve "$REPO/models/DotsOCR" \
        --served-model-name dots-ocr --port "$DOTS_PORT" \
        --gpu-memory-utilization 0.84 --enforce-eager \
        --max-model-len 6144 --max-num-seqs 1 \
        --limit-mm-per-prompt '{"image":1}' \
        --mm-processor-kwargs '{"max_pixels":1300000}' \
        --trust-remote-code > "$RUN_DIR/dots.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/dots.pid"
    printf "dots: starting (vLLM, ~90s)"
    for _ in $(seq 1 30); do
        sleep 10; printf "."
        if health "$DOTS_PORT"; then echo " up on port $DOTS_PORT"; return 0; fi
        if ! kill -0 "$(cat "$RUN_DIR/dots.pid")" 2>/dev/null; then
            echo " FAILED — see $RUN_DIR/dots.log"; return 1
        fi
    done
    echo " timeout — see $RUN_DIR/dots.log"; return 1
}

start_paddle() {
    if health "$PADDLE_PORT"; then echo "paddle: already running on port $PADDLE_PORT"; return 0; fi
    if [ ! -x "$LLAMA_SERVER_BIN" ]; then
        echo "paddle: llama-server not found at $LLAMA_SERVER_BIN (set LLAMA_SERVER_BIN)"; return 1
    fi
    setsid nohup "$LLAMA_SERVER_BIN" \
        -m "$REPO/models/paddleocr-vl-gguf/model.gguf" \
        --mmproj "$REPO/models/paddleocr-vl-gguf/mmproj.gguf" \
        --chat-template-file "$REPO/models/paddleocr-vl-gguf/chat_template.jinja" \
        --port "$PADDLE_PORT" --temp 0 -ngl 99 > "$RUN_DIR/paddle.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/paddle.pid"
    printf "paddle: starting (llama.cpp)"
    for _ in $(seq 1 12); do
        sleep 5; printf "."
        if health "$PADDLE_PORT"; then echo " up on port $PADDLE_PORT"; return 0; fi
    done
    echo " timeout — see $RUN_DIR/paddle.log"; return 1
}

stop_one() {
    local name="$1"
    local pid_file="$RUN_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
        local pid; pid=$(cat "$pid_file")
        kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
        rm -f "$pid_file"
        echo "$name: stopped"
    else
        echo "$name: no pid file (not started by this script?)"
    fi
}

status() {
    for entry in "dots:$DOTS_PORT" "paddle:$PADDLE_PORT"; do
        name="${entry%%:*}"; port="${entry##*:}"
        if health "$port"; then echo "$name: running on port $port"; else echo "$name: down"; fi
    done
}

case "${1:-}" in
    start)
        case "${2:-}" in
            dots) start_dots ;;
            paddle) start_paddle ;;
            *) echo "usage: $0 start {dots|paddle}"; exit 2 ;;
        esac ;;
    stop)
        case "${2:-}" in
            dots|paddle) stop_one "$2" ;;
            *) echo "usage: $0 stop {dots|paddle}"; exit 2 ;;
        esac ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status} [dots|paddle]"; exit 2 ;;
esac
