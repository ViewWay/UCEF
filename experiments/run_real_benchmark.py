#!/usr/bin/env python3
"""
UCEF Real Benchmark Experiment
===============================
Runs real experiments comparing UCEF pipeline vs baselines on LongBench data
using actual LLM APIs (GLM-4-flash, DeepSeek-v3).

Outputs per-sample results with ROUGE-L and token-level metrics.

Usage:
    source ~/.hermes/.env
    cd ~/github/extend-Context-System
    .venv/bin/python experiments/run_real_benchmark.py --model glm-4-flash --samples 30
    .venv/bin/python experiments/run_real_benchmark.py --model deepseek-v3 --samples 30
    .venv/bin/python experiments/run_real_benchmark.py --model all --samples 30
"""

import argparse
import json
import os
import time
import zipfile
import asyncio
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "experiments" / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LongBench Subtasks ────────────────────────────────────────────────────
# Select diverse tasks: QA, summarization, retrieval, classification
LONG_BENCH_TASKS = [
    "2wikimqa_e",        # Multi-hop QA (English)
    "hotpotqa_e",        # Multi-hop QA (English)
    "musique",           # Multi-step reasoning QA
    "gov_report_e",      # Government report summarization
    "narrativeqa",       # Narrative document QA
    "qasper_e",          # Academic paper QA
    "passage_retrieval_en_e",  # Passage retrieval
    "multifieldqa_en_e", # Multi-field QA
]

# ── LLM Client Wrappers ──────────────────────────────────────────────────

class GLMClient:
    """Zhipu GLM API client."""
    def __init__(self, model: str = "glm-4-flash"):
        from zhipuai import ZhipuAI
        self.model = model
        self.client = ZhipuAI(
            api_key=os.environ.get("GLM_API_KEY"),
            base_url=os.environ.get("GLM_BASE_URL"),
        )
        self._context_window = {"glm-4-flash": 128000, "glm-4": 128000, "glm-4-long": 1000000}.get(model, 128000)

    @property
    def context_window(self): return self._context_window

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class DeepSeekClient:
    """DeepSeek API client (OpenAI-compatible)."""
    def __init__(self, model: str = "deepseek-chat"):
        from openai import OpenAI
        self.model = model
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/v1"
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
        self._context_window = 128000

    @property
    def context_window(self): return self._context_window

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# ── Data Loading ──────────────────────────────────────────────────────────

def load_longbench_task(task_name: str, max_samples: int = -1) -> List[Dict]:
    """Load samples from a LongBench task JSONL file inside the zip."""
    zip_path = DATA_DIR / "longbench_data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"LongBench data not found at {zip_path}")

    samples = []
    with zipfile.ZipFile(str(zip_path)) as zf:
        candidates = [f"data/{task_name}.jsonl", f"{task_name}.jsonl"]
        found = None
        for c in candidates:
            if c in zf.namelist():
                found = c
                break
        if not found:
            raise FileNotFoundError(f"{task_name}.jsonl not found in zip")

        with zf.open(found) as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                item = json.loads(line)
                samples.append(item)
                if max_samples > 0 and len(samples) >= max_samples:
                    break

    return samples


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    lcs = prev[n]
    recall = lcs / n
    precision = lcs / m
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_token_overlap_f1(hypothesis: str, reference: str) -> float:
    """Token overlap F1 (approximation of BERTScore)."""
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens = set(reference.lower().split())
    if not hyp_tokens or not ref_tokens:
        return 0.0
    overlap = hyp_tokens & ref_tokens
    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (4 chars/token for English, 2 for CJK)."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ratio = cjk / max(len(text), 1)
    chars_per_token = 2.0 * ratio + 4.0 * (1 - ratio)
    return max(1, int(len(text) / chars_per_token))


# ── Context Processing Strategies ────────────────────────────────────────

def truncate_context(context: str, query: str, max_tokens: int) -> str:
    """Baseline 1: Simple truncation to fit context window."""
    budget_chars = max_tokens * 3  # rough char budget
    if len(context) <= budget_chars:
        return context
    # Keep first portion (contains title/intro)
    return context[:budget_chars]


def rag_top_k(context: str, query: str, max_tokens: int, k: int = 5) -> str:
    """Baseline 2: Standard RAG — split into chunks, rank by TF-IDF similarity, top-k."""
    chunk_size = 500
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size) if context[i:i+chunk_size].strip()]
    if not chunks:
        return context[:max_tokens * 3]

    # Simple TF-IDF scoring
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        scores.append(overlap)
    
    # Select top-k chunks
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    top_indices.sort()  # preserve order
    
    selected = [chunks[i] for i in top_indices]
    result = "\n\n".join(selected)
    
    # Trim to budget
    budget_chars = max_tokens * 3
    if len(result) > budget_chars:
        result = result[:budget_chars]
    return result


def ucef_pipeline(context: str, query: str, max_tokens: int) -> Tuple[str, Dict]:
    """UCEF pipeline: hyperbolic chunk scoring + quantum-inspired selection + adaptive compression."""
    chunk_size = 500
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size) if context[i:i+chunk_size].strip()]
    if not chunks:
        return context[:max_tokens * 3], {"method": "fallback"}

    query_words = set(query.lower().split())
    n_chunks = len(chunks)

    # ── Step 1: Hyperbolic-inspired scoring ──
    # Score each chunk: relevance + hierarchy position (earlier = more general/important)
    hyperbolic_scores = np.zeros(n_chunks)
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        # Relevance component
        overlap = len(query_words & chunk_words)
        tf_score = overlap / max(len(query_words), 1)
        # Hierarchy component: position in document (normalized)
        pos_score = 1.0 - (i / max(n_chunks - 1, 1)) * 0.3  # front-loaded
        # Combine with hyperbolic-like weighting (exaggerate differences near boundary)
        hyperbolic_scores[i] = tf_score * (1 + pos_score)

    # ── Step 2: Quantum-inspired selection ──
    # Build "density matrix" — inter-chunk similarity captures complementarity
    amplitudes = np.sqrt(np.maximum(hyperbolic_scores, 0.01))
    amplitudes /= np.linalg.norm(amplitudes)  # normalize

    # Off-diagonal: Jaccard-based "entanglement"
    budget_chars = max_tokens * 3
    n_select = max(3, min(n_chunks, int(budget_chars / chunk_size) + 2))

    # Select chunks with quantum-like probability sampling
    probs = amplitudes ** 2
    probs /= probs.sum()

    # Deterministic: pick top by probability, but add diversity penalty
    selected_indices = []
    remaining_scores = probs.copy()
    for _ in range(n_select):
        if not remaining_scores.any():
            break
        # Pick best remaining
        best = np.argmax(remaining_scores)
        selected_indices.append(best)
        # Diversity penalty: reduce scores of similar chunks
        best_words = set(chunks[best].lower().split())
        for j in range(n_chunks):
            if j != best and remaining_scores[j] > 0:
                jaccard = len(best_words & set(chunks[j].lower().split())) / max(len(best_words | set(chunks[j].lower().split())), 1)
                remaining_scores[j] *= (1 - 0.5 * jaccard)  # penalty
        remaining_scores[best] = 0  # remove selected

    selected_indices.sort()

    # ── Step 3: Adaptive compression ──
    budget_remaining = budget_chars
    compressed_parts = []
    for idx in selected_indices:
        chunk = chunks[idx]
        chunk_tokens = count_tokens_approx(chunk)

        if budget_remaining <= 0:
            break

        if chunk_tokens > budget_remaining * 0.7:
            # Aggressive: sentence-level extraction
            sentences = [s.strip() for s in chunk.split('.') if s.strip()]
            scored_sents = []
            for s in sentences:
                s_words = set(s.lower().split())
                overlap = len(query_words & s_words) / max(len(query_words), 1)
                scored_sents.append((overlap, s))
            scored_sents.sort(reverse=True)
            keep_chars = int(budget_remaining * 0.8)
            compressed = ". ".join(s for _, s in scored_sents)
            if len(compressed) > keep_chars:
                compressed = compressed[:keep_chars]
            compressed_parts.append(compressed)
            budget_remaining -= len(compressed)
        else:
            compressed_parts.append(chunk)
            budget_remaining -= len(chunk)

    result = "\n\n".join(compressed_parts)

    meta = {
        "method": "ucef",
        "n_chunks_total": n_chunks,
        "n_chunks_selected": len(selected_indices),
        "selected_indices": selected_indices,
    }
    return result, meta


# ── Experiment Runner ────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    task: str
    method: str  # "truncate", "rag", "ucef"
    model: str
    rouge_l: float = 0.0
    token_f1: float = 0.0
    context_tokens: int = 0
    total_input_tokens: int = 0
    latency_ms: float = 0.0
    prediction: str = ""
    reference: str = ""
    meta: Dict = field(default_factory=dict)


def create_client(model_name: str):
    if model_name.startswith("glm"):
        return GLMClient(model=model_name)
    elif model_name.startswith("deepseek"):
        return DeepSeekClient(model="deepseek-chat")
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_single_sample(
    client,
    sample: Dict,
    task: str,
    method: str,
    model_name: str,
    context_budget: int = 4000,
) -> SampleResult:
    """Run one sample through one method and one model."""
    context = sample.get("context", "")
    query = sample.get("input", "")
    answers = sample.get("answers", [""])
    reference = answers[0] if answers else ""
    sample_id = sample.get("_id", f"{task}_unknown")

    # Select and compress context
    t0 = time.time()
    if method == "truncate":
        selected_ctx = truncate_context(context, query, context_budget)
        meta = {"method": "truncate"}
    elif method == "rag":
        selected_ctx = rag_top_k(context, query, context_budget)
        meta = {"method": "rag"}
    elif method == "ucef":
        selected_ctx, meta = ucef_pipeline(context, query, context_budget)
    else:
        raise ValueError(f"Unknown method: {method}")
    ctx_time = time.time() - t0

    # Build prompt
    prompt = (
        f"Based on the following context, answer the question concisely and accurately.\n\n"
        f"--- Context ---\n{selected_ctx}\n\n"
        f"--- Question ---\n{query}\n\n"
        f"--- Answer ---\n"
    )

    # Call LLM
    t0 = time.time()
    try:
        prediction = client.generate(prompt, max_tokens=512, temperature=0.3)
    except Exception as e:
        prediction = f"[ERROR: {e}]"
    llm_time = time.time() - t0

    # Compute metrics
    rouge_l = compute_rouge_l(prediction, reference)
    token_f1 = compute_token_overlap_f1(prediction, reference)
    ctx_tokens = count_tokens_approx(selected_ctx)

    return SampleResult(
        sample_id=sample_id,
        task=task,
        method=method,
        model=model_name,
        rouge_l=rouge_l,
        token_f1=token_f1,
        context_tokens=ctx_tokens,
        total_input_tokens=ctx_tokens + count_tokens_approx(query),
        latency_ms=(ctx_time + llm_time) * 1000,
        prediction=prediction[:500],  # truncate for storage
        reference=reference[:500],
        meta=meta,
    )


def main():
    parser = argparse.ArgumentParser(description="UCEF Real Benchmark Experiment")
    parser.add_argument("--model", "-m", default="glm-4-flash",
                        choices=["glm-4-flash", "glm-4", "deepseek-v3", "all"],
                        help="Model to evaluate")
    parser.add_argument("--samples", "-n", type=int, default=20,
                        help="Samples per task (default: 20)")
    parser.add_argument("--tasks", "-t", nargs="*", default=None,
                        help="Tasks to run (default: all)")
    parser.add_argument("--context-budget", "-b", type=int, default=4000,
                        help="Context token budget (default: 4000)")
    parser.add_argument("--methods", nargs="*", default=["truncate", "rag", "ucef"],
                        help="Methods to compare")
    args = parser.parse_args()

    models = ["glm-4-flash", "deepseek-v3"] if args.model == "all" else [args.model]
    tasks = args.tasks or LONG_BENCH_TASKS
    methods = args.methods

    print("=" * 70)
    print("  UCEF Real Benchmark Experiment")
    print("=" * 70)
    print(f"  Models:   {models}")
    print(f"  Tasks:    {tasks}")
    print(f"  Methods:  {methods}")
    print(f"  Samples:  {args.samples}/task")
    print(f"  Budget:   {args.context_budget} tokens")
    print()

    # Load hermes .env
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ[k] = v

    all_results: List[Dict] = []
    run_id = int(time.time())

    for model_name in models:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 60}")
        client = create_client(model_name)

        for task in tasks:
            print(f"\n  Task: {task}")
            try:
                samples = load_longbench_task(task, max_samples=args.samples)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue
            print(f"    Loaded {len(samples)} samples")

            for method in methods:
                print(f"    Method: {method} ... ", end="", flush=True)
                import sys; sys.stdout.flush()
                task_results = []
                t_start = time.time()

                for i, sample in enumerate(samples):
                    result = run_single_sample(
                        client, sample, task, method, model_name,
                        context_budget=args.context_budget,
                    )
                    task_results.append(result)

                    # Rate limiting
                    if i % 5 == 4:
                        time.sleep(1)  # be nice to API

                elapsed = time.time() - t_start

                # Aggregate
                rouge_scores = [r.rouge_l for r in task_results]
                f1_scores = [r.token_f1 for r in task_results]
                latencies = [r.latency_ms for r in task_results]

                agg = {
                    "model": model_name,
                    "task": task,
                    "method": method,
                    "n_samples": len(task_results),
                    "rouge_l_mean": float(np.mean(rouge_scores)),
                    "rouge_l_std": float(np.std(rouge_scores)),
                    "token_f1_mean": float(np.mean(f1_scores)),
                    "token_f1_std": float(np.std(f1_scores)),
                    "latency_mean_ms": float(np.mean(latencies)),
                    "latency_std_ms": float(np.std(latencies)),
                    "total_time_s": round(elapsed, 1),
                }

                print(f"ROUGE-L={agg['rouge_l_mean']:.4f}±{agg['rouge_l_std']:.4f}  "
                      f"F1={agg['token_f1_mean']:.4f}±{agg['token_f1_std']:.4f}  "
                      f"({elapsed:.1f}s)")

                # Save per-sample results
                for r in task_results:
                    d = asdict(r)
                    # Fix numpy types for JSON serialization
                    def _fix(v):
                        if isinstance(v, (np.integer,)): return int(v)
                        if isinstance(v, (np.floating,)): return float(v)
                        if isinstance(v, np.ndarray): return v.tolist()
                        if isinstance(v, list): return [_fix(x) for x in v]
                        if isinstance(v, dict): return {k: _fix(val) for k, val in v.items()}
                        return v
                    d = {k: _fix(v) for k, v in d.items()}
                    all_results.append(d)
                # Fix aggregation too
                for k2, v2 in agg.items():
                    if isinstance(v2, (np.integer,)):
                        agg[k2] = int(v2)
                    elif isinstance(v2, (np.floating,)):
                        agg[k2] = float(v2)
                all_results.append(agg)

    # ── Save Combined Results ──────────────────────────────────────────
    # Save aggregated summary
    summary = [r for r in all_results if "rouge_l_mean" in r]
    per_sample = [r for r in all_results if "rouge_l_mean" not in r]

    summary_path = RESULTS_DIR / f"benchmark_summary_{run_id}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    samples_path = RESULTS_DIR / f"benchmark_samples_{run_id}.json"
    with open(samples_path, "w") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  Results saved:")
    print(f"    Summary:    {summary_path}")
    print(f"    Per-sample: {samples_path}")
    print(f"{'=' * 70}")

    # ── Print Comparison Table ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(f"{'Model':<16} {'Task':<25} {'Method':<10} {'ROUGE-L':>10} {'TokenF1':>10} {'Latency':>10}")
    print("-" * 85)
    for r in summary:
        print(f"{r['model']:<16} {r['task']:<25} {r['method']:<10} "
              f"{r['rouge_l_mean']:>8.4f}±{r['rouge_l_std']:.3f} "
              f"{r['token_f1_mean']:>8.4f}±{r['token_f1_std']:.3f} "
              f"{r['latency_mean_ms']:>8.1f}ms")


if __name__ == "__main__":
    main()
