#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  UCEF Real Experiment Setup & Runner for Apple Silicon (MLX)
#
#  Usage:
#    ./experiments/start_mlx.sh setup          # Install dependencies
#    ./experiments/start_mlx.sh server 7b      # Start MLX server (7B model)
#    ./experiments/start_mlx.sh server 14b     # Start MLX server (14B model)
#    ./experiments/start_mlx.sh run longbench mlx-qwen-7b 20
#    ./experiments/start_mlx.sh run all mlx-qwen-7b 50
#    ./experiments/start_mlx.sh status         # Check server status
#    ./experiments/start_mlx.sh kill           # Stop MLX server
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=8080
PID_FILE="/tmp/ucef_mlx_server.pid"

# ── Model definitions ──
MODEL_7B="mlx-community/Qwen2.5-7B-Instruct-4bit"
MODEL_14B="mlx-community/Qwen2.5-14B-Instruct-4bit"

usage() {
    echo "Usage: $0 {setup|server|run|status|kill} [args...]"
    echo ""
    echo "Commands:"
    echo "  setup                    Install mlx-lm and dependencies"
    echo "  server [7b|14b] [port]   Start MLX server (default: 7b on port 8080)"
    echo "  run <bench> <model> [n]  Run experiment"
    echo "  status                   Check MLX server status"
    echo "  kill                     Stop MLX server"
    echo ""
    echo "Examples:"
    echo "  $0 server 7b              # Start Qwen2.5 7B on port 8080"
    echo "  $0 run longbench mlx-qwen-7b 20"
    echo "  $0 run all mlx-qwen-7b 50"
    exit 1
}

# ── Setup ──
cmd_setup() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  UCEF MLX Experiment Setup                                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Check Python version
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python version: $PY_VERSION"

    # Check Apple Silicon
    ARCH=$(uname -m)
    if [[ "$ARCH" != "arm64" ]]; then
        echo "WARNING: Not running on Apple Silicon (arch=$ARCH). MLX requires arm64."
    fi

    # Install mlx-lm
    echo ""
    echo "Installing mlx-lm..."
    pip install mlx-lm 2>&1 || { echo "ERROR: Failed to install mlx-lm"; exit 1; }

    # Install openai client (for OpenAI-compatible API)
    echo "Installing openai client..."
    pip install openai 2>&1 || { echo "WARNING: Failed to install openai"; }

    # Verify
    echo ""
    echo "Verifying installation..."
    python3 -c "import mlx_lm; print(f'  mlx_lm: {mlx_lm.__version__}')" 2>/dev/null || echo "  mlx_lm: FAILED"
    python3 -c "import openai; print(f'  openai: {openai.__version__}')" 2>/dev/null || echo "  openai: FAILED"
    python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || echo "  numpy: FAILED"

    echo ""
    echo "Setup complete! Next steps:"
    echo "  1. $0 server 7b       # Start the MLX server"
    echo "  2. $0 run all mlx-qwen-7b 10   # Run experiment"
}

# ── Start MLX Server ──
cmd_server() {
    local size="${1:-7b}"
    local port="${2:-$PORT}"

    local model
    case "$size" in
        7b)  model="$MODEL_7B" ;;
        14b) model="$MODEL_14B" ;;
        *)   echo "Unknown model size: $size. Use 7b or 14b."; exit 1 ;;
    esac

    # Check if already running
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "MLX server already running (PID $(cat "$PID_FILE"))"
        echo "Stop it first: $0 kill"
        exit 1
    fi

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Starting MLX Server                                       ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Model: $model"
    echo "║  Port:  $port"
    echo "║  API:   http://127.0.0.1:$port/v1"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Loading model (this may take a minute on first run)..."

    # Start server in background
    python3 -m mlx_lm server \
        --model "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        &>/tmp/ucef_mlx_server.log &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Wait for server to be ready
    echo -n "Waiting for server"
    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            echo ""
            echo "Server ready! (PID $pid)"
            echo ""
            echo "Now run experiments:"
            echo "  $0 run all mlx-qwen-$(echo "$size" | tr -d 'b') 20"
            return
        fi
        echo -n "."
        sleep 2
    done

    echo ""
    echo "ERROR: Server did not start within 120s. Check log:"
    echo "  cat /tmp/ucef_mlx_server.log"
    exit 1
}

# ── Run Experiment ──
cmd_run() {
    local benchmark="${1:-all}"
    local model="${2:-mlx-qwen-7b}"
    local n_samples="${3:-20}"

    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  UCEF Real Experiment                                      ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Benchmark: $benchmark"
    echo "║  Model:     $model"
    echo "║  Samples:   $n_samples"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    # Ensure server is running for MLX models
    if [[ "$model" == mlx-* ]]; then
        if ! curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "ERROR: MLX server not running. Start it first:"
            echo "  $0 server 7b"
            exit 1
        fi
        echo "MLX server: ONLINE"
    fi

    cd "$PROJECT_DIR"
    PYTHONPATH=src python3 experiments/real_experiment.py \
        -b "$benchmark" \
        -m "$model" \
        -n "$n_samples"
}

# ── Status ──
cmd_status() {
    echo "UCEF MLX Server Status:"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "  PID:    $(cat "$PID_FILE")"
        echo "  Server: RUNNING"
        if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            echo "  API:    http://127.0.0.1:$PORT/v1 (ONLINE)"
            echo ""
            echo "  Available models:"
            curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null | grep '"id"' || echo "    (parse error)"
        else
            echo "  API:    http://127.0.0.1:$PORT/v1 (STARTING...)"
        fi
    else
        echo "  Server: STOPPED"
        rm -f "$PID_FILE"
    fi
}

# ── Kill ──
cmd_kill() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping MLX server (PID $pid)..."
            kill "$pid"
            rm -f "$PID_FILE"
            echo "Stopped."
        else
            echo "Server not running (stale PID file removed)."
            rm -f "$PID_FILE"
        fi
    else
        echo "No server PID file found."
        # Try to find and kill any running mlx_lm server
        local pids=$(pgrep -f "mlx_lm" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "Found running mlx_lm.server processes: $pids"
            read -p "Kill them? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "$pids" | xargs kill 2>/dev/null
                echo "Killed."
            fi
        else
            echo "No MLX server processes found."
        fi
    fi
}

# ── Main ──
case "${1:-}" in
    setup)  cmd_setup ;;
    server) cmd_server "${2:-7b}" "${3:-$PORT}" ;;
    run)    cmd_run "${2:-all}" "${3:-mlx-qwen-7b}" "${4:-20}" ;;
    status) cmd_status ;;
    kill)   cmd_kill ;;
    *)      usage ;;
esac
