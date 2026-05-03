# LongBench Results

## Main Results

We evaluate UCEF on 8 LongBench tasks with 30 samples per task, using GLM-4-flash and DeepSeek-v3 as backbone models. The context budget is set to 4,000 tokens (simulating a small-context model processing long documents).

### Overall Comparison

| Model | Method | Avg ROUGE-L | Avg TokenF1 | vs RAG |
|-------|--------|------------|------------|--------|
| GLM-4-flash | Truncate | 0.1433 | 0.1633 | +6.9% |
| GLM-4-flash | RAG top-k | 0.1340 | 0.1498 | — |
| GLM-4-flash | **UCEF** | **0.1479** | **0.1671** | **+10.3%** |
| DeepSeek-v3 | Truncate | 0.1889 | 0.2040 | +5.0% |
| DeepSeek-v3 | RAG top-k | 0.1800 | 0.1919 | — |
| DeepSeek-v3 | **UCEF** | **0.2146** | **0.2306** | **+19.3%** |

### Per-Task Breakdown (DeepSeek-v3)

| Task | Truncate | RAG | UCEF | Best |
|------|----------|-----|------|------|
| 2wikimqa_e | 0.2273 | 0.2095 | **0.2697** | UCEF |
| hotpotqa_e | **0.4402** | 0.3660 | 0.4133 | Trunc |
| musique | 0.0960 | 0.0941 | **0.1418** | UCEF |
| gov_report_e | **0.0707** | 0.0545 | 0.0602 | Trunc |
| narrativeqa | 0.0939 | 0.1751 | **0.2066** | UCEF |
| qasper_e | 0.1521 | 0.0571 | **0.1615** | UCEF |
| passage_retrieval_en_e | 0.0086 | 0.0086 | 0.0083 | — |
| multifieldqa_en_e | 0.4225 | **0.4748** | 0.4558 | RAG |

### Per-Task Breakdown (GLM-4-flash)

| Task | Truncate | RAG | UCEF | Best |
|------|----------|-----|------|------|
| 2wikimqa_e | **0.2109** | 0.2062 | 0.2079 | Trunc |
| hotpotqa_e | **0.2195** | 0.1796 | 0.2131 | Trunc |
| musique | 0.0330 | 0.0336 | **0.0458** | UCEF |
| gov_report_e | **0.0950** | 0.0819 | 0.0840 | Trunc |
| narrativeqa | 0.0826 | 0.1369 | **0.1391** | UCEF |
| qasper_e | **0.1068** | 0.0461 | 0.0949 | Trunc |
| passage_retrieval_en_e | 0.0000 | 0.0000 | 0.0000 | — |
| multifieldqa_en_e | **0.3989** | 0.3879 | 0.3979 | Trunc |

## Statistical Significance

Paired statistical tests comparing UCEF vs RAG across all 240 samples:

| Model | Test | p-value | Significant? |
|-------|------|---------|-------------|
| GLM-4-flash | Wilcoxon | 0.288 | No |
| GLM-4-flash | Paired t-test | 0.225 | No |
| DeepSeek-v3 | Wilcoxon | **0.011** | ✅ Yes |
| DeepSeek-v3 | Paired t-test | **0.007** | ✅ Yes |

## Key Observations

1. **UCEF consistently outperforms RAG** on multi-hop QA tasks (2wikimqa, musique) where document inter-relationships matter
2. **Truncation is surprisingly strong** on tasks where answer-relevant content appears early in documents
3. **UCEF excels on DeepSeek-v3** — the stronger backbone model amplifies the benefit of better context selection
4. **Passage retrieval** scores near zero for all methods, suggesting this task requires exact-match retrieval not suited to any compression approach

## Latency

| Component | GLM-4-flash | DeepSeek-v3 |
|-----------|------------|------------|
| Context processing | ~2ms | ~1ms |
| LLM generation | ~1,500ms | ~900ms |
| **Total per query** | **~2,000ms** | **~1,100ms** |
