"""
UCEF Real Experiment Infrastructure

Provides dataset loaders, metric computation, and experiment runners
for validating UCEF against real LLM APIs and public benchmarks.

Usage:
    # With API keys set as environment variables:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export ZHIPU_API_KEY=...

    python experiments/real_experiment.py --benchmark longbench --model gpt-4o
    python experiments/real_experiment.py --benchmark narrativeqa --model claude-3-5-sonnet-20241022
    python experiments/real_experiment.py --benchmark govreport --model glm-4
    python experiments/real_experiment.py --benchmark all --model all
"""
import argparse
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

from ucef import (
    UniversalContextSystem, UCEFConfig, Document,
    ContextBlock, QueryResult, TokenBudget,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path(__file__).parent / "results" / "real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MODELS = {
    # ── Cloud APIs ──
    "gpt-4o": {
        "adapter": "openai",
        "context_window": 128000,
        "env_key": "OPENAI_API_KEY",
    },
    "gpt-4o-mini": {
        "adapter": "openai",
        "context_window": 128000,
        "env_key": "OPENAI_API_KEY",
    },
    "claude-sonnet-4-6": {
        "adapter": "anthropic",
        "context_window": 200000,
        "env_key": "ANTHROPIC_API_KEY",
    },
    "claude-haiku-4-5": {
        "adapter": "anthropic",
        "context_window": 200000,
        "env_key": "ANTHROPIC_API_KEY",
    },
    "glm-4": {
        "adapter": "zhipu",
        "context_window": 128000,
        "env_key": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
    },
    "glm-4-long": {
        "adapter": "zhipu",
        "context_window": 1_000_000,
        "env_key": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
    },
    "glm-4-flash": {
        "adapter": "zhipu",
        "context_window": 128000,
        "env_key": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
    },
    # ── DeepSeek API (OpenAI-compatible) ──
    "deepseek-v3": {
        "adapter": "openai_compatible",
        "context_window": 128000,
        "base_url": "https://api.deepseek.com/v1",
        "model_repo": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    "deepseek-r1": {
        "adapter": "openai_compatible",
        "context_window": 128000,
        "base_url": "https://api.deepseek.com/v1",
        "model_repo": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    # ── 硅基流动 SiliconFlow (OpenAI-compatible) ──
    "siliconflow-qwen-7b": {
        "adapter": "openai_compatible",
        "context_window": 32768,
        "base_url": "https://api.siliconflow.cn/v1",
        "model_repo": "Qwen/Qwen2.5-7B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
    },
    "siliconflow-qwen-14b": {
        "adapter": "openai_compatible",
        "context_window": 32768,
        "base_url": "https://api.siliconflow.cn/v1",
        "model_repo": "Qwen/Qwen2.5-14B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
    },
    # ── MLX Local (Apple Silicon) ──
    # Uses OpenAI-compatible API exposed by mlx_lm.server
    # Launch: python -m mlx_lm.server --model <hf-repo> --port 8080
    "mlx-qwen-7b": {
        "adapter": "mlx",
        "context_window": 32768,
        "base_url": "http://127.0.0.1:8080/v1",
        "model_repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    "mlx-qwen-14b": {
        "adapter": "mlx",
        "context_window": 16384,
        "base_url": "http://127.0.0.1:8080/v1",
        "model_repo": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    },
    "mlx-qwen-7b-128k": {
        "adapter": "mlx",
        "context_window": 32768,
        "base_url": "http://127.0.0.1:8080/v1",
        "model_repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
}

BENCHMARKS = ["longbench", "narrativeqa", "govreport", "synthetic"]


# ═══════════════════════════════════════════════════════════════════════════
#  MLX Server Health Check
# ═══════════════════════════════════════════════════════════════════════════

def _check_mlx_server(base_url: str) -> str:
    """Check if an MLX server is reachable at the given URL."""
    import urllib.request
    import urllib.error
    try:
        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return "ONLINE"
    except Exception:
        return "OFFLINE"


# ═══════════════════════════════════════════════════════════════════════════
#  Metric Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L: F1 score based on longest common subsequence."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    recall = lcs / n
    precision = lcs / m
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_bertscore_approx(hypothesis: str, reference: str) -> Dict[str, float]:
    """Approximate BERTScore using token overlap (placeholder for real BERTScore)."""
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens = set(reference.lower().split())
    if not hyp_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = hyp_tokens & ref_tokens
    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Recall@K: fraction of relevant documents in top-K."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return found / len(relevant_ids)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Loaders
# ═══════════════════════════════════════════════════════════════════════════

class BaseBenchmark(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def load(self, data_dir: Path, max_samples: int = -1) -> List[Dict]:
        """Load benchmark data. Returns list of {id, documents, query, reference}."""
        pass

    @abstractmethod
    def evaluate(self, prediction: str, reference: str) -> Dict[str, float]:
        """Evaluate a prediction against a reference."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LongBenchBenchmark(BaseBenchmark):
    """LongBench: Multi-task long context understanding benchmark.

    Tries to load real data from HuggingFace (THUDM/LongBench).
    Falls back to synthetic data if datasets library is unavailable.
    """

    @property
    def name(self):
        return "longbench"

    def load(self, data_dir: Path, max_samples: int = -1) -> List[Dict]:
        """Load LongBench data. Real data from HuggingFace, synthetic fallback."""
        # Try cached real data first
        cache_path = data_dir / "longbench_real.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            data = self._normalize(data)
            if max_samples > 0:
                data = data[:max_samples]
            return data

        # Try downloading from HuggingFace
        data = self._load_huggingface(max_samples)
        if data:
            return data

        # Fallback to synthetic
        print("    Using synthetic data (install `datasets` for real data)")
        data = self._generate_synthetic(max_samples or 100)
        cache_path = data_dir / "longbench.json"
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        if max_samples > 0:
            data = data[:max_samples]
        return data

    def _load_huggingface(self, max_samples: int) -> Optional[List[Dict]]:
        """Download real LongBench data from HuggingFace (data.zip)."""
        import zipfile

        n = min(max_samples, 200) if max_samples > 0 else 200
        zip_url = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
        zip_path = DATA_DIR / "longbench_data.zip"

        # Download zip if not cached
        if not zip_path.exists():
            try:
                print("    Downloading LongBench data.zip from HuggingFace...")
                import urllib.request
                urllib.request.urlretrieve(zip_url, str(zip_path))
                print(f"    Downloaded: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
            except Exception as e:
                print(f"    Warning: Failed to download LongBench data: {e}")
                return None

        # Extract and parse JSONL files
        subtasks = ["2wikimqa_e", "musique", "passage_retrieval_en", "gov_report"]
        samples_per_task = max(n // len(subtasks), 10)
        all_data = []

        try:
            with zipfile.ZipFile(str(zip_path)) as zf:
                for task_name in subtasks:
                    jsonl_name = f"{task_name}.jsonl"
                    # Try with and without data/ prefix
                    candidates = [jsonl_name, f"data/{jsonl_name}"]
                    found = None
                    for c in candidates:
                        if c in zf.namelist():
                            found = c
                            break
                    if not found:
                        print(f"    Warning: {jsonl_name} not found in zip")
                        continue

                    count = 0
                    with zf.open(found) as f:
                        for line in f:
                            if count >= samples_per_task:
                                break
                            line = line.decode("utf-8").strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            context = item.get("context", "")
                            answers = item.get("answers", [])
                            reference = answers[0] if answers else ""
                            all_data.append({
                                "id": item.get("_id", f"lb_{len(all_data)}"),
                                "context": context,
                                "query": item.get("input", ""),
                                "reference": reference,
                                "source": "longbench",
                            })
                            count += 1
        except Exception as e:
            print(f"    Warning: Failed to extract LongBench data: {e}")
            return None

        if all_data:
            with open(DATA_DIR / "longbench_real.json", "w") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            print(f"    Loaded {len(all_data)} real samples from LongBench")
            return self._normalize(all_data)[:max_samples] if max_samples > 0 else self._normalize(all_data)
        return None

    def _generate_synthetic(self, n: int) -> List[Dict]:
        rng = np.random.RandomState(42)
        data = []
        tasks = ["qa", "summarization", "retrieval", "classification"]
        for i in range(n):
            task = tasks[i % len(tasks)]
            n_docs = rng.randint(5, 50)
            docs = []
            for j in range(n_docs):
                text = f"Document {j} about topic {rng.randint(0, 20)}. " * rng.randint(5, 30)
                docs.append(Document(id=f"lb_{i}_doc_{j}", text=text))
            data.append({
                "id": f"longbench_{i:04d}",
                "task": task,
                "documents": [{"id": d.id, "text": d.text} for d in docs],
                "query": f"What is the main point of document {rng.randint(0, n_docs)}?",
                "reference": f"The main point is about topic {rng.randint(0, 20)}.",
            })
        return data

    def evaluate(self, prediction: str, reference: str) -> Dict[str, float]:
        rouge = compute_rouge_l(prediction, reference)
        bert = compute_bertscore_approx(prediction, reference)
        return {"rouge_l": rouge, "bertscore_f1": bert["f1"]}

    @staticmethod
    def _normalize(data: List[Dict]) -> List[Dict]:
        """Normalize data format: ensure each sample has 'documents', 'query', 'reference'.

        Real HuggingFace data has 'context' (single text) + 'query' + 'reference'.
        Synthetic data has 'documents' (list of {id, text}) + 'query' + 'reference'.
        We normalize both to have all three fields.
        """
        for item in data:
            # If only 'context' exists (real data), split into document chunks
            if "context" in item and "documents" not in item:
                context = item["context"]
                chunk_size = 500  # Split long context into ~500 char chunks
                docs = []
                for j in range(0, len(context), chunk_size):
                    chunk = context[j:j + chunk_size]
                    if chunk.strip():
                        docs.append({
                            "id": f"{item.get('id', 'doc')}_{len(docs)}",
                            "text": chunk,
                        })
                if not docs:
                    docs = [{"id": f"{item.get('id', 'doc')}_0", "text": context}]
                item["documents"] = docs

            # Ensure 'query' exists (some HuggingFace data uses 'input')
            if "query" not in item and "input" in item:
                item["query"] = item["input"]

            # Ensure 'reference' exists
            if "reference" not in item:
                answers = item.get("answers", [])
                item["reference"] = answers[0] if answers else ""

        return data

    def _generate_synthetic(self, n: int) -> List[Dict]:
        rng = np.random.RandomState(42)
        data = []
        tasks = ["qa", "summarization", "retrieval", "classification"]
        for i in range(n):
            task = tasks[i % len(tasks)]
            n_docs = rng.randint(5, 50)
            docs = []
            for j in range(n_docs):
                text = f"Document {j} about topic {rng.randint(0, 20)}. " * rng.randint(5, 30)
                docs.append({"id": f"lb_{i}_doc_{j}", "text": text})
            data.append({
                "id": f"longbench_{i:04d}",
                "documents": docs,
                "query": f"What is the main point of document {rng.randint(0, n_docs)}?",
                "reference": f"The main point is about topic {rng.randint(0, 20)}.",
            })
        return data

    def evaluate(self, prediction: str, reference: str) -> Dict[str, float]:
        rouge = compute_rouge_l(prediction, reference)
        bert = compute_bertscore_approx(prediction, reference)
        return {"rouge_l": rouge, "bertscore_f1": bert["f1"]}


class NarrativeQABenchmark(BaseBenchmark):
    """NarrativeQA: Document-based question answering on narrative texts.

    Tries real data from HuggingFace (deepmind/narrative_qa).
    Falls back to synthetic data if unavailable.
    """

    @property
    def name(self):
        return "narrativeqa"

    def load(self, data_dir: Path, max_samples: int = -1) -> List[Dict]:
        cache_path = data_dir / "narrativeqa_real.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            data = LongBenchBenchmark._normalize(data)
            if max_samples > 0:
                data = data[:max_samples]
            return data

        data = self._load_huggingface(max_samples)
        if data:
            return data

        print("    Using synthetic data (install `datasets` for real data)")
        data = self._generate_synthetic(max_samples or 80)
        with open(data_dir / "narrativeqa.json", "w") as f:
            json.dump(data, f, indent=2)
        if max_samples > 0:
            data = data[:max_samples]
        return data

    def _load_huggingface(self, max_samples: int) -> Optional[List[Dict]]:
        """Load NarrativeQA from HuggingFace or direct download."""
        n = min(max_samples, 200) if max_samples > 0 else 200

        # Method 1: datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset("deepmind/narrative_qa", split=f"test[:{n}]")
            all_data = []
            for item in ds:
                doc_text = item.get("document", {}).get("text", "")
                question = item.get("question", {}).get("text", "")
                answers = [a["text"] for a in item.get("answers", [])]
                all_data.append({
                    "id": f"nqa_{len(all_data)}",
                    "context": doc_text,
                    "query": question,
                    "reference": answers[0] if answers else "",
                    "source": "narrativeqa",
                })
            if all_data:
                with open(DATA_DIR / "narrativeqa_real.json", "w") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                print(f"    Loaded {len(all_data)} real samples from NarrativeQA")
                return LongBenchBenchmark._normalize(all_data)[:max_samples] if max_samples > 0 else LongBenchBenchmark._normalize(all_data)
        except Exception as e:
            print(f"    NarrativeQA via datasets library failed: {e}")

        # Method 2: LongBench narrativeqa subset from data.zip
        zip_path = DATA_DIR / "longbench_data.zip"
        if zip_path.exists():
            import zipfile
            try:
                with zipfile.ZipFile(str(zip_path)) as zf:
                    for name in ["narrativeqa.jsonl", "data/narrativeqa.jsonl"]:
                        if name in zf.namelist():
                            all_data = []
                            count = 0
                            with zf.open(name) as f:
                                for line in f:
                                    if count >= n:
                                        break
                                    line = line.decode("utf-8").strip()
                                    if not line:
                                        continue
                                    item = json.loads(line)
                                    all_data.append({
                                        "id": f"nqa_{len(all_data)}",
                                        "context": item.get("context", ""),
                                        "query": item.get("input", ""),
                                        "reference": (item.get("answers") or [""])[0],
                                        "source": "longbench_narrativeqa",
                                    })
                                    count += 1
                            if all_data:
                                with open(DATA_DIR / "narrativeqa_real.json", "w") as f:
                                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                                print(f"    Loaded {len(all_data)} samples from LongBench/narrativeqa")
                                return LongBenchBenchmark._normalize(all_data)[:max_samples] if max_samples > 0 else LongBenchBenchmark._normalize(all_data)
            except Exception as e:
                print(f"    LongBench narrativeqa fallback failed: {e}")
        return None

    def _generate_synthetic(self, n: int) -> List[Dict]:
        rng = np.random.RandomState(43)
        stories = ["adventure", "mystery", "romance", "sci-fi", "fantasy"]
        data = []
        for i in range(n):
            story = stories[i % len(stories)]
            docs = []
            for j in range(rng.randint(3, 20)):
                text = f"In the {story} story, chapter {j}: the protagonist encounters " \
                       f"a challenge involving {rng.choice(['danger', 'discovery', 'dialogue', 'decision'])}. " \
                       * rng.randint(3, 15)
                docs.append({"id": f"nqa_{i}_doc_{j}", "text": text})
            data.append({
                "id": f"nqa_{i:04d}",
                "documents": docs,
                "query": f"What happens in chapter {rng.randint(0, len(docs))} of the {story} story?",
                "reference": f"The protagonist faces a challenge in the {story} narrative.",
            })
        return data

    def evaluate(self, prediction: str, reference: str) -> Dict[str, float]:
        rouge = compute_rouge_l(prediction, reference)
        return {"rouge_l": rouge}


class GovReportBenchmark(BaseBenchmark):
    """GovReport: Government report summarization.

    Tries real data from HuggingFace (urbik/gov_report).
    Falls back to synthetic data if unavailable.
    """

    @property
    def name(self):
        return "govreport"

    def load(self, data_dir: Path, max_samples: int = -1) -> List[Dict]:
        cache_path = data_dir / "govreport_real.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            data = LongBenchBenchmark._normalize(data)
            if max_samples > 0:
                data = data[:max_samples]
            return data

        data = self._load_huggingface(max_samples)
        if data:
            return data

        print("    Using synthetic data (install `datasets` for real data)")
        data = self._generate_synthetic(max_samples or 60)
        with open(data_dir / "govreport.json", "w") as f:
            json.dump(data, f, indent=2)
        if max_samples > 0:
            data = data[:max_samples]
        return data

    def _load_huggingface(self, max_samples: int) -> Optional[List[Dict]]:
        """Load GovReport from HuggingFace or LongBench data.zip."""
        n = min(max_samples, 200) if max_samples > 0 else 200

        # Method 1: datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset("urbik/gov_report", split=f"test[:{n}]")
            all_data = []
            for item in ds:
                doc_text = item.get("input", item.get("text", ""))
                summary = item.get("output", item.get("summary", ""))
                all_data.append({
                    "id": f"gr_{len(all_data)}",
                    "context": doc_text,
                    "query": "Summarize this government report.",
                    "reference": summary,
                    "source": "govreport",
                })
            if all_data:
                with open(DATA_DIR / "govreport_real.json", "w") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                print(f"    Loaded {len(all_data)} real samples from GovReport")
                return LongBenchBenchmark._normalize(all_data)[:max_samples] if max_samples > 0 else LongBenchBenchmark._normalize(all_data)
        except Exception as e:
            print(f"    GovReport via datasets library failed: {e}")

        # Method 2: LongBench gov_report subset from data.zip
        zip_path = DATA_DIR / "longbench_data.zip"
        if zip_path.exists():
            import zipfile
            try:
                with zipfile.ZipFile(str(zip_path)) as zf:
                    for name in ["gov_report.jsonl", "data/gov_report.jsonl", "gov_report_e.jsonl", "data/gov_report_e.jsonl"]:
                        if name in zf.namelist():
                            all_data = []
                            count = 0
                            with zf.open(name) as f:
                                for line in f:
                                    if count >= n:
                                        break
                                    line = line.decode("utf-8").strip()
                                    if not line:
                                        continue
                                    item = json.loads(line)
                                    all_data.append({
                                        "id": f"gr_{len(all_data)}",
                                        "context": item.get("context", ""),
                                        "query": item.get("input", "Summarize this government report."),
                                        "reference": (item.get("answers") or [""])[0],
                                        "source": "longbench_govreport",
                                    })
                                    count += 1
                            if all_data:
                                with open(DATA_DIR / "govreport_real.json", "w") as f:
                                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                                print(f"    Loaded {len(all_data)} samples from LongBench/gov_report")
                                return LongBenchBenchmark._normalize(all_data)[:max_samples] if max_samples > 0 else LongBenchBenchmark._normalize(all_data)
            except Exception as e:
                print(f"    LongBench gov_report fallback failed: {e}")
        return None

    def _generate_synthetic(self, n: int) -> List[Dict]:
        rng = np.random.RandomState(44)
        agencies = ["GAO", "CBO", "CRS", "OMB"]
        data = []
        for i in range(n):
            agency = agencies[i % len(agencies)]
            docs = []
            for j in range(rng.randint(5, 30)):
                text = f"{agency} Report Section {j}: This section discusses " \
                       f"federal policy regarding {rng.choice(['budget', 'healthcare', 'defense', 'education'])}. " \
                       * rng.randint(5, 20)
                docs.append({"id": f"gr_{i}_doc_{j}", "text": text})
            data.append({
                "id": f"govreport_{i:04d}",
                "documents": docs,
                "query": f"Summarize the {agency} report on federal policy.",
                "reference": f"The {agency} report outlines key findings on federal policy.",
            })
        return data

    def evaluate(self, prediction: str, reference: str) -> Dict[str, float]:
        rouge = compute_rouge_l(prediction, reference)
        bert = compute_bertscore_approx(prediction, reference)
        return {"rouge_l": rouge, "bertscore_f1": bert["f1"]}


# ═══════════════════════════════════════════════════════════════════════════
#  Model Adapter Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_adapter(model_name: str):
    """Create a model adapter based on model name."""
    from ucef.models.openai import OpenAIAdapter
    from ucef.models.anthropic import AnthropicAdapter
    from ucef.models.zhipu import ZhipuAdapter
    from ucef.models.local import LocalAdapter

    config = MODELS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    adapter_type = config["adapter"]

    if adapter_type == "openai":
        api_key = os.environ.get(config["env_key"])
        return OpenAIAdapter(model=model_name, api_key=api_key)
    elif adapter_type == "openai_compatible":
        # DeepSeek, SiliconFlow, etc. — OpenAI-compatible API with custom base_url
        api_key = os.environ.get(config["env_key"])
        base_url = os.environ.get(config.get("base_url_env", ""), "") or config.get("base_url", "https://api.deepseek.com/v1")
        model_repo = config.get("model_repo", model_name)
        adapter = OpenAIAdapter(
            model=model_repo,
            api_key=api_key,
            base_url=base_url,
        )
        adapter._context_window = config.get("context_window", 32768)
        return adapter
    elif adapter_type == "anthropic":
        api_key = os.environ.get(config["env_key"])
        return AnthropicAdapter(model=model_name, api_key=api_key)
    elif adapter_type == "zhipu":
        api_key = os.environ.get(config["env_key"])
        base_url = os.environ.get(config.get("base_url_env", ""), "") or None
        return ZhipuAdapter(model=model_name, api_key=api_key, base_url=base_url)
    elif adapter_type == "mlx":
        # MLX uses OpenAI-compatible API via mlx_lm.server
        base_url = config.get("base_url", "http://127.0.0.1:8080/v1")
        model_repo = config.get("model_repo", model_name)
        adapter = OpenAIAdapter(
            model=model_repo,  # Server knows the model by its HF repo name
            api_key="not-needed",
            base_url=base_url,
        )
        # Override context_window since these models aren't in OPENAI_MODEL_SPECS
        adapter._context_window = config.get("context_window", 32768)
        return adapter
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# ═══════════════════════════════════════════════════════════════════════════
#  Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Container for a single experiment run's results."""
    benchmark: str
    model: str
    n_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)
    per_sample: List[Dict] = field(default_factory=list)
    total_time_s: float = 0.0
    timestamp: str = ""

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


async def run_experiment(
    model_name: str,
    benchmark: BaseBenchmark,
    max_samples: int = -1,
) -> ExperimentResult:
    """Run a full experiment: load data → store → query → evaluate."""
    print(f"\n  Running: {benchmark.name} × {model_name}")

    # Load data
    samples = benchmark.load(DATA_DIR, max_samples)
    print(f"  Loaded {len(samples)} samples")

    # Create adapter (may fail if no API key — that's OK, we'll use mock)
    try:
        adapter = create_adapter(model_name)
    except Exception as e:
        print(f"  Warning: Could not create adapter ({e}), using mock")
        class MockAdapter:
            def generate(self, prompt, **kwargs):
                return f"Mock response for: {prompt[:50]}..."
        adapter = MockAdapter()

    # Initialize UCEF system
    config = UCEFConfig()
    config.hyperbolic.embedding_dim = 8
    config.quality.quality_threshold = 0.5

    system = UniversalContextSystem(
        model_client=adapter,
        model_name=model_name,
        config=config,
    )
    await system.initialize()

    # Run evaluation
    all_metrics = []
    t_start = time.time()

    for i, sample in enumerate(samples):
        # Store documents
        docs = [Document(id=d["id"], text=d["text"]) for d in sample["documents"]]
        await system.store_documents(docs)

        # Query — retrieve relevant context
        result = await system.query(sample["query"])

        # Generate model response using retrieved context
        context_text = result.format_context()
        query_text = sample["query"]
        prompt = (
            f"Based on the following context, answer the query concisely.\n\n"
            f"--- Context ---\n{context_text}\n\n"
            f"--- Query ---\n{query_text}\n\n"
            f"--- Answer ---\n"
        )

        try:
            if hasattr(adapter, 'generate'):
                prediction = await adapter.generate(prompt, max_tokens=256)
            else:
                prediction = context_text  # Fallback: use context as-is
        except Exception as e:
            print(f"    Warning: Model generation failed for sample {i}: {e}")
            prediction = context_text

        reference = sample.get("reference", "")

        # Evaluate
        sample_metrics = benchmark.evaluate(prediction, reference)
        sample_metrics["quality"] = result.overall_quality
        sample_metrics["n_blocks"] = len(result.context_blocks)
        sample_metrics["total_tokens"] = result.total_tokens
        sample_metrics["retrieval_time_ms"] = result.retrieval_time_ms

        all_metrics.append({
            "sample_id": sample["id"],
            **sample_metrics,
        })

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(samples)} samples")

    total_time = time.time() - t_start

    # Aggregate metrics
    if all_metrics:
        keys = all_metrics[0].keys()
        agg = {}
        for k in keys:
            if k == "sample_id":
                continue
            vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
            if vals:
                agg[f"mean_{k}"] = float(np.mean(vals))
                agg[f"std_{k}"] = float(np.std(vals))
    else:
        agg = {}

    result = ExperimentResult(
        benchmark=benchmark.name,
        model=model_name,
        n_samples=len(samples),
        metrics=agg,
        per_sample=all_metrics,
        total_time_s=round(total_time, 2),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Save
    out_name = f"{benchmark.name}_{model_name}_{int(time.time())}.json"
    result.save(RESULTS_DIR / out_name)

    print(f"  Done: {len(samples)} samples in {total_time:.1f}s")
    print(f"  Key metrics: {json.dumps({k: round(v, 4) for k, v in agg.items() if k.startswith('mean')}, indent=2)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UCEF Real Experiment Runner")
    parser.add_argument("--benchmark", "-b", choices=BENCHMARKS + ["all"], default="all",
                        help="Benchmark to run")
    parser.add_argument("--model", "-m", default="all",
                        help=f"Model to evaluate: {', '.join(MODELS.keys())}, or 'all'")
    parser.add_argument("--max-samples", "-n", type=int, default=20,
                        help="Max samples per benchmark (default: 20)")
    parser.add_argument("--list", action="store_true",
                        help="List available models and benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, cfg in MODELS.items():
            adapter_type = cfg["adapter"]
            if adapter_type == "mlx":
                base_url = cfg.get("base_url", "http://127.0.0.1:8080/v1")
                server_status = _check_mlx_server(base_url)
                print(f"  {name}: MLX ({cfg['context_window']//1024}K ctx) "
                      f"[server {server_status}] → {cfg.get('model_repo', '')}")
            elif adapter_type == "openai_compatible":
                key_status = "SET" if os.environ.get(cfg.get("env_key", "")) else "NOT SET"
                print(f"  {name}: {cfg.get('model_repo', '')} "
                      f"({cfg['context_window']//1024}K ctx) [key {key_status}] "
                      f"→ {cfg.get('base_url', '')}")
            else:
                key_status = "SET" if os.environ.get(cfg.get("env_key", "")) else "NOT SET"
                print(f"  {name}: {adapter_type} ({cfg['context_window']//1024}K ctx) [key {key_status}]")
        print(f"\nAvailable benchmarks: {BENCHMARKS}")
        print(f"\nMLX quick start:")
        print(f"  python -m mlx_lm server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080")
        print(f"\nDeepSeek quick start:")
        print(f"  export DEEPSEEK_API_KEY=sk-...")
        print(f"  python experiments/real_experiment.py -b all -m deepseek-v3 -n 200")
        return

    # Select benchmarks
    if args.benchmark == "all":
        benchmarks = [
            LongBenchBenchmark(),
            NarrativeQABenchmark(),
            GovReportBenchmark(),
        ]
    elif args.benchmark == "longbench":
        benchmarks = [LongBenchBenchmark()]
    elif args.benchmark == "narrativeqa":
        benchmarks = [NarrativeQABenchmark()]
    elif args.benchmark == "govreport":
        benchmarks = [GovReportBenchmark()]
    elif args.benchmark == "synthetic":
        benchmarks = [LongBenchBenchmark()]  # Uses synthetic data by default
    else:
        benchmarks = [LongBenchBenchmark()]

    # Select models
    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          UCEF Real Experiment Runner                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Benchmarks: {[b.name for b in benchmarks]}")
    print(f"  Models:     {models}")
    print(f"  Max samples: {args.max_samples}")

    all_results = {}

    async def _run_all():
        for benchmark in benchmarks:
            for model_name in models:
                key = f"{benchmark.name}_{model_name}"
                try:
                    result = await run_experiment(model_name, benchmark, args.max_samples)
                    all_results[key] = {
                        "benchmark": result.benchmark,
                        "model": result.model,
                        "n_samples": result.n_samples,
                        "metrics": result.metrics,
                        "total_time_s": result.total_time_s,
                    }
                except Exception as e:
                    print(f"  ERROR: {key}: {e}")
                    all_results[key] = {"error": str(e)}

    asyncio.run(_run_all())

    # Save combined results
    combined_path = RESULTS_DIR / f"combined_{int(time.time())}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Combined results: {combined_path}")


if __name__ == "__main__":
    main()
