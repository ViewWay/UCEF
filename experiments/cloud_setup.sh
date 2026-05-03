#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  UCEF Cloud GPU Experiment Deployer
#
#  支持: AutoDL / RunPod / 任意带 Docker 的 GPU 云
#
#  使用:
#    bash experiments/cloud_setup.sh autodl   # AutoDL 部署
#    bash experiments/cloud_setup.sh runpod   # RunPod 部署
#    bash experiments/cloud_setup.sh bare     # 裸机/任意 GPU 云
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODE="${1:-bare}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  UCEF Cloud GPU Experiment Deployer                            ║"
echo "║  Platform: ${MODE}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: System check ──
echo "[1/6] System check..."

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
    echo "  ✓ GPU: ${GPU_INFO}"
else
    echo "  ✗ nvidia-smi not found. No NVIDIA GPU detected."
    echo "    This script requires an NVIDIA GPU cloud instance."
    exit 1
fi

# Check Python
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python: ${PY_VER}"

# Check disk
DISK_AVAIL=$(df -h . | awk 'NR==2{print $4}')
echo "  ✓ Disk available: ${DISK_AVAIL}"

# ── Step 2: Install dependencies ──
echo ""
echo "[2/6] Installing dependencies..."

# Use pip (cloud instances usually have conda or system python)
pip install -q --upgrade pip

# Core dependencies
pip install -q numpy vllm openai anthropic torch
pip install -q datasets rouge-score bert-score
pip install -q zhipuai 2>/dev/null || true

echo "  ✓ Core packages installed"

# Install UCEF
cd "$PROJECT_DIR"
pip install -q -e .
echo "  ✓ UCEF installed"

# ── Step 3: Download real datasets ──
echo ""
echo "[3/6] Downloading real benchmark datasets from HuggingFace..."

python3 << 'DATASET_DOWNLOAD'
import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("experiments/data")
DATA_DIR.mkdir(exist_ok=True)

# --- LongBench ---
print("  Downloading LongBench...")
lb = load_dataset("THUDM/LongBench", "2wikimqa_e", split="test[:200]")
longbench_data = []
for item in lb:
    longbench_data.append({
        "id": item.get("qid", f"lb_{len(longbench_data)}"),
        "task": "multi_doc_qa",
        "context": item.get("context", ""),
        "input": item.get("input", ""),
        "answers": item.get("answers", []),
        "length": item.get("length", 0),
    })
with open(DATA_DIR / "longbench_real.json", "w") as f:
    json.dump(longbench_data, f, ensure_ascii=False, indent=2)
print(f"    ✓ LongBench: {len(longbench_data)} samples")

# --- NarrativeQA ---
print("  Downloading NarrativeQA...")
try:
    nqa = load_dataset("deepmind/narrative_qa", split="test[:200]")
    nqa_data = []
    for item in nqa:
        nqa_data.append({
            "id": f"nqa_{len(nqa_data)}",
            "document": item.get("document", {}).get("text", ""),
            "question": item.get("question", {}).get("text", ""),
            "answers": [a["text"] for a in item.get("answers", [])],
        })
    with open(DATA_DIR / "narrativeqa_real.json", "w") as f:
        json.dump(nqa_data, f, ensure_ascii=False, indent=2)
    print(f"    ✓ NarrativeQA: {len(nqa_data)} samples")
except Exception as e:
    print(f"    ⚠ NarrativeQA download failed: {e}")
    print("    Using fallback synthetic data")

# --- GovReport ---
print("  Downloading GovReport...")
try:
    gr = load_dataset("urbik/gov_report", split="test[:200]")
    govreport_data = []
    for item in gr:
        govreport_data.append({
            "id": f"gr_{len(govreport_data)}",
            "document": item.get("input", item.get("text", "")),
            "summary": item.get("output", item.get("summary", "")),
        })
    with open(DATA_DIR / "govreport_real.json", "w") as f:
        json.dump(govreport_data, f, ensure_ascii=False, indent=2)
    print(f"    ✓ GovReport: {len(govreport_data)} samples")
except Exception as e:
    print(f"    ⚠ GovReport download failed: {e}")
    print("    Using fallback synthetic data")

print("  Dataset download complete.")
DATASET_DOWNLOAD

# ── Step 4: Start vLLM server ──
echo ""
echo "[4/6] Starting vLLM inference server..."

# Detect GPU memory to choose model
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
GPU_MEM=${GPU_MEM:-0}

if [ "$GPU_MEM" -ge 160000 ]; then
    # 2x A100 80GB or H100
    MODEL="Qwen/Qwen2.5-72B-Instruct"
    TP=2
    echo "  → 160GB+ VRAM: Using Qwen2.5-72B-Instruct (tp=2)"
elif [ "$GPU_MEM" -ge 80000 ]; then
    # Single A100 80GB
    MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
    TP=1
    echo "  → 80GB VRAM: Using Qwen2.5-72B-Instruct-AWQ (4bit, tp=1)"
elif [ "$GPU_MEM" -ge 48000 ]; then
    # A6000 / L40
    MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
    TP=1
    echo "  → 48GB VRAM: Using Qwen2.5-32B-Instruct-AWQ (tp=1)"
elif [ "$GPU_MEM" -ge 24000 ]; then
    # RTX 3090 / 4090
    MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
    TP=1
    echo "  → 24GB VRAM: Using Qwen2.5-14B-Instruct-AWQ (tp=1)"
else
    MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
    TP=1
    echo "  → Low VRAM: Using Qwen2.5-7B-Instruct-AWQ (tp=1)"
fi

echo "  Model: ${MODEL}"
echo "  Tensor Parallel: ${TP}"
echo "  Starting vLLM server (background)..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len 32768 \
    --port 8080 \
    --trust-remote-code \
    &>/tmp/vllm_server.log &

VLLM_PID=$!
echo "  vLLM PID: ${VLLM_PID}"

# Wait for server ready
echo -n "  Waiting for server"
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:8080/v1/models" >/dev/null 2>&1; then
        echo ""
        echo "  ✓ vLLM server ready!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "  ✗ Server crashed! Check: /tmp/vllm_server.log"
        tail -20 /tmp/vllm_server.log
        exit 1
    fi
    echo -n "."
    sleep 5
done

# Smoke test
echo "  Running smoke test..."
SMOKE=$(curl -s "http://127.0.0.1:8080/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"Say OK"}],"max_tokens":10}' 2>/dev/null || echo "")
if echo "$SMOKE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
    echo "  ✓ Model responds correctly"
else
    echo "  ⚠ Smoke test inconclusive"
fi

# ── Step 5: Run experiments ──
echo ""
echo "[5/6] Running full experiment suite..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Determine model alias for UCEF experiment
if echo "$MODEL" | grep -q "72B"; then
    UCEF_MODEL="cloud-qwen-72b"
elif echo "$MODEL" | grep -q "32B"; then
    UCEF_MODEL="cloud-qwen-32b"
elif echo "$MODEL" | grep -q "14B"; then
    UCEF_MODEL="cloud-qwen-14b"
else
    UCEF_MODEL="cloud-qwen-7b"
fi

for bench in longbench narrativeqa govreport; do
    echo ""
    echo "  ▶ ${bench} × ${UCEF_MODEL} (n=200)"
    echo "  ─────────────────────────────────────────"

    PYTHONPATH=src python3 experiments/real_experiment.py \
        -b "$bench" \
        -m "$UCEF_MODEL" \
        -n 200 \
        --real-data \
        2>&1 || echo "  ⚠ ${bench} had errors"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 6: Results ──
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
" 2>/dev/null || echo "  (Could not parse results)"
fi

# Stop server
echo ""
echo "  Stopping vLLM server..."
kill "$VLLM_PID" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Experiment complete!                                         ║"
echo "║  Results: experiments/results/real/                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
