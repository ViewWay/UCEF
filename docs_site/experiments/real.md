# Real Experiments

Infrastructure for validating UCEF against real LLM APIs and public benchmarks.

## Setup

### Option A: Local MLX (Apple Silicon, no API key needed)

```bash
# 1. Install dependencies
./experiments/start_mlx.sh setup

# 2. Start MLX server (Qwen2.5 7B, ~4GB)
./experiments/start_mlx.sh server 7b

# 3. Run experiments
./experiments/start_mlx.sh run all mlx-qwen-7b 20
./experiments/start_mlx.sh run longbench mlx-qwen-7b 50

# 4. Check server status
./experiments/start_mlx.sh status

# 5. Stop server
./experiments/start_mlx.sh kill
```

### Option B: Cloud APIs

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export ZHIPU_API_KEY=...
```

### Run

```bash
# All benchmarks with one model
python experiments/real_experiment.py -b all -m gpt-4o -n 100

# One benchmark with all models
python experiments/real_experiment.py -b longbench -m all -n 50

# List available options
python experiments/real_experiment.py --list
```

## Available Models

### Cloud APIs

| Model | Provider | Context Window |
|-------|----------|---------------|
| GPT-4o | OpenAI | 128K |
| Claude 3.5 Sonnet | Anthropic | 200K |
| GLM-4 | Zhipu AI | 128K |
| GLM-4-Long | Zhipu AI | 1M |

### Local MLX (Apple Silicon)

| Model | Size | Context Window | Memory |
|-------|------|---------------|--------|
| mlx-qwen-7b | ~4GB | 32K | Fits in 24GB |
| mlx-qwen-14b | ~9GB | 16K | Fits in 24GB |

MLX exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, so it works with the existing OpenAI adapter.

## Benchmarks

### LongBench

Multi-task long context understanding benchmark covering:
- Single/multi-document QA
- Summarization
- Retrieval
- Classification

**Metric**: ROUGE-L + approximate BERTScore

### NarrativeQA

Document-based question answering on narrative texts:
- Adventure, mystery, romance, sci-fi, fantasy stories
- Questions about plot, characters, events

**Metric**: ROUGE-L

### GovReport

Government report summarization:
- GAO, CBO, CRS, OMB reports
- Federal policy documents

**Metric**: ROUGE-L + approximate BERTScore

## Metrics

### ROUGE-L

F1 score based on longest common subsequence between prediction and reference.

### BERTScore (Approximate)

Token overlap approximation for quick evaluation. For production, use the full BERTScore with transformer embeddings.

### Recall@K

Fraction of relevant documents found in top-K retrieved results.

## Mock Mode

When no API keys are available, the framework automatically degrades to mock mode:
- Generates synthetic benchmark data
- Uses mock model responses
- Validates pipeline correctness (not model quality)

Results are saved to `experiments/results/real/` as JSON files.
