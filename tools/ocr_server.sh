#!/bin/bash
# Manage the local OCR inference server used by pdf2md.py --model lighton.
#
# Usage:
#   tools/ocr_server.sh start lighton   # llama.cpp serving LightOnOCR-2-1B (port 8093)
#   tools/ocr_server.sh stop  lighton
#   tools/ocr_server.sh status
#
# Setup and model files: docs/local_ocr.md. The vision projector must be the
# Q8_0 one: the f16 projector produces garbage on CUDA (see docs/pdf2md.md).
# Overridable env vars:
#   LLAMA_SERVER_BIN  path to llama-server (default: ~/gitrepos/llama.cpp/build-cuda/bin/llama-server)
#   LIGHTON_PORT      default 8093
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO/.ocr-servers"
mkdir -p "$RUN_DIR"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/gitrepos/llama.cpp/build-cuda/bin/llama-server}"
LIGHTON_PORT="${LIGHTON_PORT:-8093}"
LIGHTON_DIR="$REPO/models/lightonocr2"
LIGHTON_MODEL="$LIGHTON_DIR/LightOnOCR-2-1B-f16.gguf"
LIGHTON_MMPROJ="$LIGHTON_DIR/mmproj-LightOnOCR-2-1B-Q8_0.gguf"

health() { curl -s --max-time 3 "http://127.0.0.1:$1/health" > /dev/null 2>&1; }

start_lighton() {
    if health "$LIGHTON_PORT"; then echo "lighton: already running on port $LIGHTON_PORT"; return 0; fi
    if [ ! -x "$LLAMA_SERVER_BIN" ]; then
        echo "lighton: llama-server not found at $LLAMA_SERVER_BIN (set LLAMA_SERVER_BIN)"; return 1
    fi
    for f in "$LIGHTON_MODEL" "$LIGHTON_MMPROJ"; do
        if [ ! -f "$f" ]; then echo "lighton: missing $f (see docs/local_ocr.md)"; return 1; fi
    done
    setsid nohup "$LLAMA_SERVER_BIN" \
        -m "$LIGHTON_MODEL" --mmproj "$LIGHTON_MMPROJ" \
        -c 16384 -ngl 99 --port "$LIGHTON_PORT" > "$RUN_DIR/lighton.log" 2>&1 < /dev/null &
    echo $! > "$RUN_DIR/lighton.pid"
    printf "lighton: starting (llama.cpp)"
    for _ in $(seq 1 24); do
        sleep 5; printf "."
        if health "$LIGHTON_PORT"; then echo " up on port $LIGHTON_PORT"; return 0; fi
        if ! kill -0 "$(cat "$RUN_DIR/lighton.pid")" 2>/dev/null; then
            echo " FAILED — see $RUN_DIR/lighton.log"; return 1
        fi
    done
    echo " timeout — see $RUN_DIR/lighton.log"; return 1
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
    if health "$LIGHTON_PORT"; then echo "lighton: running on port $LIGHTON_PORT"; else echo "lighton: down"; fi
}

case "${1:-}" in
    start)
        case "${2:-}" in
            lighton) start_lighton ;;
            *) echo "usage: $0 start lighton"; exit 2 ;;
        esac ;;
    stop)
        case "${2:-}" in
            lighton) stop_one "$2" ;;
            *) echo "usage: $0 stop lighton"; exit 2 ;;
        esac ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status} [lighton]"; exit 2 ;;
esac
