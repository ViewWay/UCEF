#!/bin/bash
# Run all experiments

echo "=== UCEF Experiment Suite ==="
echo ""

# Activate virtual environment
source venv/bin/activate

# Run benchmarks
echo "Running benchmarks..."
python scripts/eval/run_all.py

# Run quality comparison
echo "Running quality comparison..."
python scripts/eval/quality_comparison.py

# Run ablation studies
echo "Running ablation studies..."
python scripts/eval/run_ablation.py

echo "=== All experiments complete! ==="
