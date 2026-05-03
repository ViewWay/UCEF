#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  UCEF Real Experiment — One-Click Runner with Real Data
#
#  Usage:
#    cd ~/github/extend-Context-System
#    source .venv/bin/activate
#    bash experiments/run_real_all.sh          # 全部: 本地 MLX + 所有已有 API
#    bash experiments/run_real_all.sh local    # 只跑本地 MLX
#    bash experiments/run_real_all.sh deepseek # 只跑 DeepSeek API
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-all}"
N_SAMPLES="${2:-50}"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  UCEF Real Experiment — Real Data + Multi-Model               ║"
echo "║  Mode: ${MODE}  Samples: ${N_SAMPLES}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Download real data ──
echo "[1/3] Downloading real benchmark data..."
python3 -c "
import json, zipfile, urllib.request
from pathlib import Path
DATA = Path('experiments/data')
DATA.mkdir(exist_ok=True)

# LongBench data.zip
zip_path = DATA / 'longbench_data.zip'
if not zip_path.exists():
    print('  Downloading LongBench data.zip...')
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip',
        str(zip_path)
    )
    print(f'  Downloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB')

    # Preview what's inside
    with zipfile.ZipFile(str(zip_path)) as zf:
        names = [n for n in zf.namelist() if n.endswith('.jsonl')]
        print(f'  Contains {len(names)} JSONL files:')
        for n in sorted(names)[:8]:
            print(f'    {n}')
        if len(names) > 8:
            print(f'    ... and {len(names)-8} more')
else:
    print(f'  LongBench data.zip already cached ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)')
print('  ✓ Data ready')
" || echo "  ⚠ Download failed (experiments will use synthetic fallback)"
echo ""

# ── Step 2: Check MLX server ──
MLX_UP=false
if [[ "$MODE" == "all" || "$MODE" == "local" ]]; then
    echo "[2/3] Checking MLX server..."
    if curl -s "http://127.0.0.1:8080/v1/models" >/dev/null 2>&1; then
        echo "  ✓ MLX server online"
        MLX_UP=true
    else
        echo "  ✗ MLX server offline"
        echo "  Start it in another terminal:"
        echo "    python3 -m mlx_lm server --model mlx-community/Qwen2.5-7B-Instruct-4bit --host 127.0.0.1 --port 8080"
        if [[ "$MODE" == "local" ]]; then
            echo ""
            echo "  Waiting for MLX server... (press Ctrl+C to cancel)"
            for i in $(seq 1 60); do
                if curl -s "http://127.0.0.1:8080/v1/models" >/dev/null 2>&1; then
                    echo "  ✓ MLX server online!"
                    MLX_UP=true
                    break
                fi
                sleep 2
            done
            if [[ "$MLX_UP" == "false" ]]; then
                echo "  ✗ Timeout. Exiting."
                exit 1
            fi
        fi
    fi
fi
echo ""

# ── Step 3: Run experiments ──
echo "[3/3] Running experiments..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_exp() {
    local bench="$1"
    local model="$2"
    local n="$3"
    echo ""
    echo "  ▶ ${bench} × ${model} (n=${n})"
    echo "  ─────────────────────────────────────────"
    PYTHONPATH=src python3 experiments/real_experiment.py \
        -b "$bench" -m "$model" -n "$n" 2>&1 || echo "  ⚠ ${bench} × ${model} had errors"
}

# Local MLX experiments
if [[ "$MLX_UP" == "true" ]]; then
    for bench in longbench narrativeqa govreport; do
        run_exp "$bench" "mlx-qwen-7b" "$N_SAMPLES"
    done
fi

# DeepSeek API experiments
if [[ "$MODE" == "all" || "$MODE" == "deepseek" ]]; then
    if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
        echo "  ✓ DEEPSEEK_API_KEY set"
        for bench in longbench narrativeqa govreport; do
            run_exp "$bench" "deepseek-v3" "$N_SAMPLES"
        done
    else
        echo "  ⚠ DEEPSEEK_API_KEY not set — skipping DeepSeek"
        echo "    export DEEPSEEK_API_KEY=sk-..."
    fi
fi

# OpenAI experiments
if [[ "$MODE" == "all" || "$MODE" == "openai" ]]; then
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
        echo "  ✓ OPENAI_API_KEY set"
        for bench in longbench narrativeqa govreport; do
            run_exp "$bench" "gpt-4o-mini" "$N_SAMPLES"
        done
    else
        echo "  ⚠ OPENAI_API_KEY not set — skipping GPT-4o"
        echo "    export OPENAI_API_KEY=sk-..."
    fi
fi

# SiliconFlow experiments
if [[ "$MODE" == "all" || "$MODE" == "siliconflow" ]]; then
    if [[ -n "${SILICONFLOW_API_KEY:-}" ]]; then
        echo "  ✓ SILICONFLOW_API_KEY set"
        for bench in longbench narrativeqa govreport; do
            run_exp "$bench" "siliconflow-qwen-7b" "$N_SAMPLES"
        done
    else
        echo "  ⚠ SILICONFLOW_API_KEY not set — skipping SiliconFlow"
        echo "    export SILICONFLOW_API_KEY=sk-..."
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Summary ──
echo ""
echo "Results summary:"
echo ""
RESULTS_DIR="${PROJECT_DIR}/experiments/results/real"
LATEST=$(ls -t "${RESULTS_DIR}"/combined_*.json 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "  File: ${LATEST}"
    echo ""
    python3 -c "
import json
with open('${LATEST}') as f:
    data = json.load(f)
for key, val in data.items():
    if 'error' in val:
        print(f'  {key}: ERROR')
    elif 'metrics' in val:
        m = val['metrics']
        rouge = m.get('mean_rouge_l', 0)
        bert = m.get('mean_bertscore_f1', 0)
        quality = m.get('mean_quality', 0)
        latency = m.get('mean_retrieval_time_ms', 0)
        n = val.get('n_samples', 0)
        print(f'  {key} (n={n}):')
        print(f'    ROUGE-L={rouge:.4f}  BERTScore={bert:.4f}  Quality={quality:.3f}  Latency={latency:.1f}ms')
" 2>/dev/null || echo "  (Could not parse)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Experiment complete!                                         ║"
echo "║  Results: experiments/results/real/                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
