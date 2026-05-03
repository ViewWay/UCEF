# Experiments Overview

UCEF provides two experiment tiers: simulated validation (no API keys) and real benchmarks.

## Simulated Experiments

6 experiment groups validating all core components with synthetic data.

| # | Experiment | Key Result |
|---|-----------|------------|
| 1 | Hyperbolic vs Euclidean Retrieval | Below baseline with random embeddings (needs SGD training) |
| 2 | Quantum vs Classical Selection | +1.9% accuracy |
| 3 | Compression Strategy Comparison | Aggressive 10%, Moderate 28%, Light 50% |
| 4 | End-to-End Quality at Scale | UCEF 89.5% vs RAG 71.0% |
| 5 | Feedback Loop Convergence | 100% within ≤3 iterations |
| 6 | Pipeline Latency | ~10ms average |

## Real Experiments

Infrastructure supporting real LLM APIs and public benchmarks.

### Supported Benchmarks

| Benchmark | Task | Description |
|-----------|------|-------------|
| LongBench | Multi-task | QA, summarization, retrieval, classification |
| NarrativeQA | Document QA | Narrative document question answering |
| GovReport | Summarization | Government report summarization |

### Supported Models

| Model | Provider | Context Window |
|-------|----------|---------------|
| GPT-4o | OpenAI | 128K |
| Claude 3.5 Sonnet | Anthropic | 200K |
| GLM-4 | Zhipu AI | 128K |
| GLM-4-Long | Zhipu AI | 1M |

## Quick Start

```bash
# Simulated experiments (no API key needed)
python experiments/simulated_experiment.py

# Real experiments
export OPENAI_API_KEY=sk-...
python experiments/real_experiment.py -b all -m gpt-4o -n 100

# List available models and benchmarks
python experiments/real_experiment.py --list
```
