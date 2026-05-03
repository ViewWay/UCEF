# UCEF 论文工作交接文档

**日期**: 2026-05-03 23:45
**状态**: arXiv 论文未写，实验已完成，需继续

---

## 1. 项目位置

- **代码**: `~/github/extend-Context-System/`
- **论文目录**: `~/github/extend-Context-System/paper/arxiv/`
- **已有文件**:
  - `paper/arxiv/references.bib` — 39条引用（已验证，已补充 MemWalker/Longformer/StreamLLM/RAG等）
  - `paper/arxiv/ucef.tex` — **不存在，待写**
  - `paper/ieee/ucef-en.tex` — IEEE格式英文旧版草稿（可做骨架）
  - `paper/ieee/ucef-cn.tex` — IEEE格式中文旧版
  - `paper/chinese-journal/ucef-en.tex` — 计算机学报英文版
  - `paper/chinese-journal/ucef-cn.tex` — 计算机学报中文版

---

## 2. 实验数据

### 2.1 模拟实验（已完成）
- `experiments/results/simulated_results.json` — 6组实验
- `experiments/results/ablation_results.json` — 5组消融（5种子）
- 运行命令: `cd ~/github/extend-Context-System && .venv/bin/python experiments/simulated_experiment.py`

### 2.2 真实实验（已完成）⭐

**实验脚本**: `experiments/run_real_benchmark.py`（新写的，修复了API key适配）

**API Key**: 在 `~/.hermes/.env` 中
- `GLM_API_KEY` → 智谱 GLM-4-flash
- `DEEPSEEK_API_KEY` → DeepSeek-v3
- 加载方式: `source ~/.hermes/.env`

**venv**: `~/github/extend-Context-System/.venv/`（已装 numpy/scipy/openai/zhipuai）

**30样本实验结果（论文用）**:
- `experiments/results/real/benchmark_summary_1777819684.json` — GLM-4-flash
- `experiments/results/real/benchmark_summary_1777819685.json` — DeepSeek-v3
- `experiments/results/real/benchmark_samples_1777819684.json` — GLM 逐条数据
- `experiments/results/real/benchmark_samples_1777819685.json` — DeepSeek 逐条数据

**汇总数据** (8 tasks × 30 samples × 3 methods × 2 models = 1440次LLM调用):

| 模型 | Truncate ROUGE-L | RAG ROUGE-L | UCEF ROUGE-L | UCEF vs RAG |
|------|-----------------|-------------|-------------|-------------|
| GLM-4-flash | 0.1433 | 0.1340 | 0.1479 | +10.3% |
| DeepSeek-v3 | 0.1889 | 0.1800 | 0.2146 | +19.3% |

**统计显著性**:
- DeepSeek: Wilcoxon p=0.0108, t-test p=0.0072 — **显著**
- GLM: Wilcoxon p=0.288 — 不显著（win/loss接近打平）

**任务覆盖**: 2wikimqa_e, hotpotqa_e, musique, gov_report_e, narrativeqa, qasper_e, passage_retrieval_en_e, multifieldqa_en_e

**LongBench数据**: 已下载 `experiments/data/longbench_data.zip` (108.7MB, 34个JSONL)

---

## 3. 论文待写内容

### 3.1 定位
- **arXiv**: framework design paper + initial validation
- 不是"碾压baseline"的立场，而是"新框架设计与初步验证"
- 后续补充数据投计算机学报

### 3.2 arXiv 论文结构 (ucef.tex)

```
\documentclass{article}  % arXiv标准格式
使用 \bibliography{references} 编译

1. Abstract (~200 words)
2. Introduction (~1.5 pages) — 上下文瓶颈问题, 三类方法局限
3. Related Work (~1.5 pages) — 双曲NLP, 量子IR, 上下文压缩, 记忆架构
4. Method (~3 pages):
   4.1 Problem Formulation (Eq 1: quality maximization)
   4.2 Hyperbolic Retrieval Engine (Poincaré ball, Eq 2-5)
   4.3 Quantum-Inspired Selection (density matrix, Eq 6-9)
   4.4 Adaptive Compression with Quality Feedback (Eq 10-13)
   4.5 Three-Layer Memory Architecture
   4.6 System Architecture (pipeline overview)
5. Experiments (~2.5 pages):
   5.1 Setup: GLM-4-flash, DeepSeek-v3, LongBench (8 tasks, 30 samples/task)
   5.2 Baselines: Truncation, RAG top-k
   5.3 Main Results (Table: per-task ROUGE-L + TokenF1)
   5.4 Ablation Study (from simulated: feedback +62.8%, memory 10.9x)
   5.5 Statistical Significance (Wilcoxon test)
   5.6 Latency Analysis
6. Discussion (~0.5 page) — 局限: 未训练嵌入, O(n²)密度矩阵, 样本量
7. Conclusion (~0.3 page)
```

### 3.3 关键公式（从源码确认）
- Poincaré距离: `d(u,v) = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))`
- 指数映射: `exp_0(v) = tanh(||v||) * v/||v||`
- 密度矩阵: `ρ = Σ p_k |ψ_k><ψ_k|` + Jaccard纠缠修正
- 干涉滤波: `I(i) = Σ_j Re(α_i * α_j* * cos(θ_ij))`
- MDL目标: `MDL = w*L(context) + (1-w)*L(query|context)`
- 质量评分: `Q = 0.30*R + 0.30*C + 0.20*H + 0.20*A`

### 3.4 注意事项
- **不要**过度宣称"量子优势"，措辞用"quantum-inspired classical algorithm"
- **必须**注明双曲嵌入未训练（train_embeddings是stub）
- **引用修复**: tex中引用了 `sasaki2023quantumctx` 但bib中对应条目是 `alodjants2024quantum`
- **引用修复**: tex中引用了 `bronstein2024hyplora` 但bib中对应条目是 `yang2025hyplora`
- passage_retrieval_en_e 任务全零，论文中应排除或解释

---

## 4. 技术方向验证结论

| 方向 | 成立性 | 热度 | 支撑文献数 |
|------|--------|------|-----------|
| 双曲几何RAG | 强 | 上升 | HyperbolicRAG(2025), HypRAG(2026) |
| 量子启发选择 | 中强 | 稳定 | van Rijsbergen(2004), QISA(2026) |
| 自适应压缩+反馈 | 强 | 上升 | ATACompressor(2026), LongLLMLingua(2024) |
| 三层记忆 | 成立 | 稳定 | MemWalker, MemoryBank |

---

## 5. 下次工作清单

1. **写 `paper/arxiv/ucef.tex`** — 用 IEEE 版草稿做骨架，加入真实实验数据
2. **编译验证**: `cd paper/arxiv && pdflatex ucef && bibtex ucef && pdflatex ucef && pdflatex ucef`
3. **提交 arXiv**: 需注册 arXiv 账号，上传 .tex + .bib + 编译后的 PDF
4. **后续改进**（投计算机学报前）:
   - 实现 Riemannian SGD 训练双曲嵌入
   - 增大样本量到 100+/task
   - 增加 GPT-4o 或 Claude 模型
   - 换指标：加 EM (Exact Match) for QA tasks

---

## 6. 快速启动命令

```bash
# 加载环境
source ~/.hermes/.env
cd ~/github/extend-Context-System

# 激活 venv
source .venv/bin/activate

# 重跑实验（如需要）
python experiments/run_real_benchmark.py --model glm-4-flash --samples 30

# 编译论文
cd paper/arxiv
pdflatex ucef && bibtex ucef && pdflatex ucef && pdflatex ucef
```
