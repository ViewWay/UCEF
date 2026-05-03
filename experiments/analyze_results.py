import json, numpy as np
from pathlib import Path

results_dir = Path("experiments/results/real")

for label, f in [
    ("GLM-4-flash", results_dir / "benchmark_summary_1777819684.json"),
    ("DeepSeek-v3", results_dir / "benchmark_summary_1777819685.json"),
]:
    data = json.load(open(f))
    model = data[0]["model"]
    print(f"\n{'='*75}")
    print(f"  {model} - 30 samples/task - Per-task breakdown")
    print(f"{'='*75}")

    by_task = {}
    for r in data:
        t = r["task"]
        if t not in by_task:
            by_task[t] = {}
        by_task[t][r["method"]] = r

    header = f"{'Task':<28} {'Trunc':>7} {'RAG':>7} {'UCEF':>7} {'Best':>5} {'Win?':>5} {'UCEF std/mean':>14}"
    print(header)
    print("-" * 75)

    ucef_wins = 0
    issues = []
    for task in sorted(by_task.keys()):
        m = by_task[task]
        t_v = m.get("truncate", {}).get("rouge_l_mean", 0)
        r_v = m.get("rag", {}).get("rouge_l_mean", 0)
        u_v = m.get("ucef", {}).get("rouge_l_mean", 0)
        u_std = m.get("ucef", {}).get("rouge_l_std", 0)
        best_v = max(t_v, r_v, u_v)
        best_n = "T" if t_v == best_v else ("R" if r_v == best_v else "U")
        win = "YES" if u_v >= max(t_v, r_v) else "no"
        if u_v >= max(t_v, r_v):
            ucef_wins += 1
        else:
            issues.append(task)
        ratio = u_std / max(u_v, 0.001)
        stable = "OK" if ratio < 1.0 else "UNSTABLE"
        print(f"{task:<28} {t_v:>7.4f} {r_v:>7.4f} {u_v:>7.4f} {best_n:>5} {win:>5} {ratio:>8.2f} ({stable})")

    print(f"\nUCEF wins: {ucef_wins}/8 tasks")
    if issues:
        print(f"UCEF loses on: {', '.join(issues)}")

    # Per-sample significance check (load sample data)
    sample_files = sorted(results_dir.glob("benchmark_samples_*.json"))
    for sf in sample_files:
        try:
            samples = json.load(open(sf))
        except:
            continue
        if not samples or "rouge_l" not in samples[0] or samples[0].get("model") != model:
            continue
            # Paired comparison: UCEF vs RAG per sample
            from collections import defaultdict
            by_sample = defaultdict(dict)
            for s in samples:
                if "rouge_l" in s and "method" in s and "sample_id" in s:
                    by_sample[(s["task"], s["sample_id"])][s["method"]] = s["rouge_l"]

            ucef_scores = []
            rag_scores = []
            trunc_scores = []
            for key, methods in by_sample.items():
                if "ucef" in methods and "rag" in methods:
                    ucef_scores.append(methods["ucef"])
                    rag_scores.append(methods["rag"])
                if "ucef" in methods and "truncate" in methods:
                    trunc_scores.append(methods["ucef"])

            if len(ucef_scores) >= 10:
                ucef_arr = np.array(ucef_scores)
                rag_arr = np.array(rag_scores)
                diff = ucef_arr - rag_arr
                n_pos = (diff > 0).sum()
                n_neg = (diff < 0).sum()
                n_tie = (diff == 0).sum()
                mean_diff = diff.mean()

                # Wilcoxon-like sign test
                from scipy.stats import wilcoxon, ttest_rel
                try:
                    stat, p_val = wilcoxon(ucef_arr, rag_arr)
                    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
                except:
                    p_val = 1.0
                    sig = "N/A (all ties)"

                try:
                    t_stat, t_pval = ttest_rel(ucef_arr, rag_arr)
                except:
                    t_pval = 1.0

                print(f"\n--- Statistical significance ({model}) ---")
                print(f"  Paired samples: {len(ucef_scores)}")
                print(f"  UCEF > RAG: {n_pos}, UCEF < RAG: {n_neg}, Tie: {n_tie}")
                print(f"  Mean diff (UCEF-RAG): {mean_diff:+.4f}")
                print(f"  Wilcoxon p-value: {p_val:.4f} -> {sig}")
                print(f"  Paired t-test p-value: {t_pval:.4f}")
            break

print("\n" + "="*75)
print("  DIAGNOSIS SUMMARY")
print("="*75)
