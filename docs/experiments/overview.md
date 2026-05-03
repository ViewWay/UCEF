# Experiments

UCEF is evaluated through a combination of simulated experiments (for controlled ablation studies) and real benchmark experiments (for end-to-end validation).

## Experimental Setup

### Models

| Model | Context Window | API Provider |
|-------|---------------|--------------|
| GLM-4-flash | 128K | Zhipu AI |
| DeepSeek-v3 | 128K | DeepSeek |

### Benchmarks

We evaluate on **8 tasks** from the LongBench benchmark suite:

| Task | Type | Description |
|------|------|-------------|
| 2wikimqa_e | Multi-hop QA | Wikipedia-based multi-hop reasoning |
| hotpotqa_e | Multi-hop QA | HotPotQA English subset |
| musique | Multi-step QA | Multi-step reasoning questions |
| gov_report_e | Summarization | Government report summarization |
| narrativeqa | Document QA | Narrative comprehension questions |
| qasper_e | Academic QA | Research paper question answering |
| passage_retrieval_en_e | Retrieval | Passage retrieval evaluation |
| multifieldqa_en_e | Multi-field QA | Cross-domain question answering |

### Baselines

| Method | Description |
|--------|-------------|
| **Truncate** | Simple truncation to context budget (first N tokens) |
| **RAG top-k** | TF-IDF chunk scoring + top-k selection |
| **UCEF** | Full pipeline: hyperbolic scoring + quantum selection + adaptive compression |

### Metrics

- **ROUGE-L**: F1 score based on longest common subsequence
- **Token Overlap F1**: Approximate semantic similarity via token overlap
- **Latency**: End-to-end pipeline time per query

## Sample Size

- **30 samples per task** (240 per method per model)
- **1,440 total LLM API calls** across all experiments
