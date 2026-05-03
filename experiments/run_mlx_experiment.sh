#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  UCEF MLX Real Experiment — One-Command Runner
#
#  在 Mac 上执行:
#    cd ~/github/extend-Context-System
#    bash experiments/run_mlx_experiment.sh
#
#  该脚本会:
#    1. 检查 mlx-lm 是否安装
#    2. 下载模型（首次运行）
#    3. 启动 MLX 推理服务
#    4. 运行全部 3 个基准测试 × 1 个模型 × 20 样本
#    5. 输出结果 JSON
#    6. 关闭服务
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8080
MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"
UCEF_MODEL="mlx-qwen-7b"
N_SAMPLES=20
PID_FILE="/tmp/ucef_mlx_server.pid"
LOG_FILE="/tmp/ucef_mlx_server.log"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  UCEF × MLX Real Experiment                                    ║"
echo "║  Model: Qwen2.5-7B-Instruct-4bit (MLX)                        ║"
echo "║  Target: 3 benchmarks × ${N_SAMPLES} samples                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check dependencies ──
echo "[1/6] Checking dependencies..."

if ! python3 -c "import mlx_lm" 2>/dev/null; then
    echo "  mlx-lm not found. Installing..."
    pip install mlx-lm
fi

MLX_VER=$(python3 -c "import mlx_lm; print(mlx_lm.__version__)" 2>/dev/null || echo "unknown")
echo "  ✓ mlx_lm: ${MLX_VER}"

if ! python3 -c "import openai" 2>/dev/null; then
    echo "  openai not found. Installing..."
    pip install openai
fi

OPENAI_VER=$(python3 -c "import openai; print(openai.__version__)" 2>/dev/null || echo "unknown")
echo "  ✓ openai: ${OPENAI_VER}"

# ── Step 2: Kill any existing server ──
echo ""
echo "[2/6] Cleaning up any existing server..."
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "  Stopping existing server (PID $(cat "$PID_FILE"))..."
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    sleep 2
fi
rm -f "$PID_FILE"

# ── Step 3: Start MLX server ──
echo ""
echo "[3/6] Starting MLX server..."
echo "  Model: ${MODEL}"
echo "  Port:  ${PORT}"
echo "  Log:   ${LOG_FILE}"
echo ""

python3 -m mlx_lm server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    &>"$LOG_FILE" &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "  Server PID: ${SERVER_PID}"

# ── Step 4: Wait for server ready ──
echo ""
echo "[4/6] Waiting for model to load (may download on first run)..."
echo -n "  "
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo ""
        echo "  ✓ Server ready!"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "  ✗ Server crashed! Check log: ${LOG_FILE}"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    echo -n "."
    sleep 2
done

# Final check
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  ✗ Server did not start within 240s"
    tail -20 "$LOG_FILE"
    exit 1
fi

# Quick smoke test
echo "  Running smoke test..."
SMOKE=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":10}' 2>/dev/null || echo "")
if echo "$SMOKE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
    echo "  ✓ Model responds correctly"
else
    echo "  ⚠ Smoke test inconclusive (server may still be warming up)"
fi

# ── Step 5: Run experiments ──
echo ""
echo "[5/6] Running experiments..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR"

BENCHMARKS=("longbench" "narrativeqa" "govreport")
RESULTS=()

for bench in "${BENCHMARKS[@]}"; do
    echo ""
    echo "  ▶ ${bench} × ${UCEF_MODEL} (n=${N_SAMPLES})"
    echo "  ─────────────────────────────────────────"

    PYTHONPATH=src python3 experiments/real_experiment.py \
        -b "$bench" \
        -m "$UCEF_MODEL" \
        -n "$N_SAMPLES" \
        2>&1 || echo "  ⚠ ${bench} had errors (continuing)"

    RESULTS+=("${bench}")
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 6: Summarize & cleanup ──
echo ""
echo "[6/6] Results summary:"
echo ""

RESULTS_DIR="${PROJECT_DIR}/experiments/results/real"
LATEST=$(ls -t "${RESULTS_DIR}"/combined_*.json 2>/dev/null | head -1)

if [[ -n "$LATEST" ]]; then
    echo "  Combined results: ${LATEST}"
    echo ""
    python3 -c "
import json
with open('${LATEST}') as f:
    data = json.load(f)
for key, val in data.items():
    if 'error' in val:
        print(f'  {key}: ERROR - {val[\"error\"]}')
    elif 'metrics' in val:
        m = val['metrics']
        quality = m.get('mean_quality', 0)
        rouge = m.get('mean_rouge_l', 0)
        n_blocks = m.get('mean_n_blocks', 0)
        tokens = m.get('mean_total_tokens', 0)
        latency = m.get('mean_retrieval_time_ms', 0)
        print(f'  {key}:')
        print(f'    Quality={quality:.3f}  ROUGE-L={rouge:.4f}  Blocks={n_blocks:.0f}  Tokens={tokens:.0f}  Latency={latency:.1f}ms')
    else:
        print(f'  {key}: {val}')
" 2>/dev/null || echo "  (Could not parse results)"
fi

# Stop server
echo ""
echo "  Stopping MLX server..."
kill "$SERVER_PID" 2>/dev/null || true
rm -f "$PID_FILE"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Experiment complete!                                         ║"
echo "║  Results: experiments/results/real/                           ║"
echo "║  Server log: ${LOG_FILE}"
echo "╚══════════════════════════════════════════════════════════════════╝"
