# UCEF 系统实施路线图

**版本**: 2.0
**更新日期**: 2026-05-03
**当前状态**: Phase 1-2 已完成，Phase 3 待开始

---

## 项目目标

实现通用上下文扩展框架（UCEF），结合双曲几何、量子概率论、信息论，为所有 LLM 提供 4K→1M+ tokens 无限上下文能力。

---

## 总体进度

```
Phase 1 ████████████████████ 100%  核心基础设施
Phase 2 ████████████████████ 100%  检索引擎 + 三层记忆
Phase 3 ░░░░░░░░░░░░░░░░░░░░   0%  压缩引擎 + 物理模型
Phase 4 ░░░░░░░░░░░░░░░░░░░░   0%  质量闭环 + 监控
Phase 5 ░░░░░░░░░░░░░░░░░░░░   0%  模型适配器
Phase 6 ░░░░░░░░░░░░░░░░░░░░   0%  测试 + 文档 + Benchmark
```

---

## 技术栈

### 核心依赖（已使用）
```python
numpy>=1.26.0            # 数值计算 — types, retrieval, memory 均依赖
pydantic>=2.8.0          # 配置验证 — config.py 全部使用
```

### 扩展依赖（Phase 3+ 按需引入）
```python
# 双曲几何训练（Phase 3）
geoopt>=0.1.0            # Riemannian SGD
torch>=2.2.0             # 张量计算（训练嵌入时需要）

# 可选（已内置 fallback）
chromadb>=0.5.0          # Warm 存储（已有 numpy fallback）
redis>=5.0.0             # Hot 存储（已有 OrderedDict fallback）
h5py>=3.11.0             # Cold 存储（已有 JSON fallback）
scipy>=1.13.0            # 统计检验（entropy 计算可选用）

# LLM 适配（Phase 5）
openai>=1.30.0           # GPT-4o
anthropic>=0.28.0        # Claude
zhipuai>=2.0.0           # 智谱 GLM
```

---

## 已完成阶段详情

### Phase 1: 核心基础设施 ✅ (Week 1-2)

**产出文件:**
- `src/ucef/core/types.py` — 17 类 + 32 函数（双曲几何、量子态、信息论、文档模型）
- `src/ucef/core/config.py` — 9 个 Pydantic v2 配置类（含 model_validator）
- `src/ucef/core/system.py` — UniversalContextSystem 管线（store → query → retrieve → select → compress → evaluate）
- `src/ucef/quality/profiler.py` — 12 个已知模型规格，自动策略推荐

**修复记录:** FIX-001 ~ FIX-013（13 项）

### Phase 2: 检索引擎 + 三层记忆 ✅ (Week 3-4)

**产出文件:**
- `src/ucef/retrieval/hyperbolic.py` — Poincaré 球嵌入 + 测地线最近邻
- `src/ucef/retrieval/quantum.py` — 4 步量子管线（叠加态 → 密度矩阵 → 测量 → 坍缩）
- `src/ucef/retrieval/fusion.py` — RRF + WeightedScore + Hybrid 融合
- `src/ucef/retrieval/adaptive.py` — Small/Medium/Large 三种策略
- `src/ucef/memory/hot.py` — Redis / OrderedDict fallback
- `src/ucef/memory/warm.py` — ChromaDB / numpy fallback
- `src/ucef/memory/cold.py` — HDF5 / JSON / Parquet
- `src/ucef/memory/three_layer.py` — 三层协调器（存储分布、跨层检索、自动升降级）
- `src/ucef/quality/preservation.py` — QualityPreservationEngine

**验证结果:** 19 文件语法检查通过，67 跨模块导入验证通过

**修复记录:** FIX-014 ~ FIX-020（7 项）

---

## 待完成阶段

### Phase 3: 压缩引擎 + 物理模型 (Week 5-8)

> **目标**: 将 system.py 中 `_compress_to_budget()` 从简单截断升级为 MDL + 熵 + 任务感知的自适应压缩

**前置依赖分析:**
- `system.py:_compress_to_budget()` 当前仅按相关性排序截断 → 需要接入 compression 引擎
- `config.py:CompressionConfig` 已定义 `use_mdl`, `use_entropy`, `entropy_threshold` → 配置就绪
- `types.py:CompressionResult` 已定义 `compression_ratio`, `reduction_percentage` → 类型就绪
- `adaptive.py` 的三种策略均调用 `_*_compression()` 占位方法 → 需要接入真实压缩

**文件清单:**

```
src/ucef/compression/
├── __init__.py            # 导出
├── adaptive.py            # AdaptiveCompressor 主入口
├── mdl.py                 # MDL 压缩器
├── entropy.py             # 熵最大化压缩器
└── task_aware.py          # 任务感知压缩（提取摘要/关键句）

src/ucef/physics/
├── __init__.py            # 导出
├── thermodynamic.py       # 热力学温度模型（free energy = E - TS）
└── quantum_field.py       # 量子场论启发（重整化群多尺度摘要）
```

**接口设计:**

```python
# compression/adaptive.py — 主入口
class AdaptiveCompressor:
    def __init__(self, config: CompressionConfig): ...
    async def compress(self, blocks: List[ContextBlock], budget: TokenBudget,
                       strategy: CompressionStrategy) -> Tuple[List[ContextBlock], CompressionResult]: ...

# compression/mdl.py — MDL 压缩
class MDLCompressor:
    def description_length(self, text: str) -> int: ...
    def compress(self, block: ContextBlock, target_ratio: float) -> ContextBlock: ...

# compression/entropy.py — 熵最大化
class EntropyCompressor:
    def information_entropy(self, blocks: List[ContextBlock]) -> float: ...
    def select_diverse_subset(self, blocks: List[ContextBlock], budget: int) -> List[ContextBlock]: ...

# compression/task_aware.py — 任务感知
class TaskAwareCompressor:
    async def compress_for_query(self, blocks: List[ContextBlock],
                                  query: str, model_client: ModelClient) -> List[ContextBlock]: ...
```

**集成点:**
1. `system.py:initialize()` — 取消注释 `self._compression_engine = AdaptiveCompressor(...)`
2. `system.py:_compress_to_budget()` — 调用 `self._compression_engine.compress()`
3. `adaptive.py:SmallContextStrategy._aggressive_compression()` → 调用真实压缩器
4. `hyperbolic.py:train_embeddings()` — 实现 Riemannian SGD

**详细计划:** 见 `docs/plans/phase3-detailed-plan.md`

---

### Phase 4: 质量闭环 + 监控 (Week 9-10)

**前置依赖:** Phase 3 的压缩引擎需要质量评估反馈

**文件清单:**
```
src/ucef/quality/
├── monitor.py             # 实时质量监控
└── feedback.py            # 自动反馈 + 重新选择循环
```

**核心功能:**
- 质量监控 dashboard 数据收集
- quality < threshold 时的自动重新检索 + 重新选择
- 自一致性采样 (self-consistency sampling)
- 信心校准 (confidence calibration)

**集成点:**
1. `system.py:_evaluate_quality()` — accuracy 从 0.85 硬编码改为真实评估
2. `system.py:query()` — 添加 quality refinement 循环

---

### Phase 5: 模型适配器 (Week 11-12)

**文件清单:**
```
src/ucef/models/
├── __init__.py            # 当前 placeholder，替换为真实导出
├── base.py                # BaseModelAdapter 抽象类
├── openai.py              # OpenAIAdapter (GPT-4o, GPT-4o-mini)
├── anthropic.py           # AnthropicAdapter (Claude 3.5 Sonnet)
├── zhipu.py               # ZhipuAdapter (GLM-4, GLM-5.1)
└── local.py               # LocalModelAdapter (transformers)
```

**接口:** 实现 `ucef.core.types.ModelClient` Protocol

---

### Phase 6: 测试 + 文档 + Benchmark (Week 13-16)

**测试结构:**
```
tests/
├── unit/
│   ├── test_types.py          # 数学类型正确性
│   ├── test_config.py         # 配置验证
│   ├── test_system.py         # 系统管线
│   ├── test_retrieval.py      # 检索引擎
│   ├── test_memory.py         # 三层记忆
│   ├── test_compression.py    # 压缩引擎
│   └── test_quality.py        # 质量评估
├── integration/
│   ├── test_e2e_pipeline.py   # 端到端管线
│   └── test_cross_module.py   # 跨模块一致性
└── benchmark/
    ├── bench_retrieval.py     # 检索延迟
    └── bench_compression.py   # 压缩质量
```

**评估协议（RESEARCH_SURVEY.md §5.5）:**
- Baselines: 标准 RAG, LongLLMLingua, 原生长上下文
- 数据集: LongBench, NarrativeQA, GovReport
- 指标: ROUGE-L, BERTScore, Recall@K, 延迟
- 成功标准: quality ≥ 85%, latency < 500ms, ≥3 模型族验证

---

## 更新后的时间线

```
2026-05-02 ─── Phase 1 开始
2026-05-03 ─── Phase 1 完成 ✅
2026-05-03 ─── Phase 2 开始
2026-05-03 ─── Phase 2 完成 ✅
2026-05-04 ─── Phase 3 开始（压缩引擎 + 物理模型）
2026-05-17 ─── Phase 3 目标完成（2 周）
2026-05-18 ─── Phase 4 开始（质量闭环）
2026-05-25 ─── Phase 4 目标完成（1 周）
2026-05-26 ─── Phase 5 开始（模型适配器）
2026-06-01 ─── Phase 5 目标完成（1 周）
2026-06-02 ─── Phase 6 开始（测试 + Benchmark）
2026-06-22 ─── Phase 6 目标完成（3 周）
2026-06-23 ─── 实验运行 + 论文撰写
2026-07-15 ─── 论文初稿完成
```

---

## 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Riemannian SGD 训练不收敛 | Phase 3 双曲嵌入质量 | 使用 geoopt 库成熟实现，fallback 到随机初始化 |
| 压缩质量不达标 | 端到端性能 | 多策略可选（MDL / 熵 / 任务感知），adaptive 自动切换 |
| 外部 API 限流 | Phase 5 适配器测试 | Mock 客户端优先，真实 API 可选 |
| Benchmark 数据集获取 | Phase 6 评估 | 使用公开数据集（LongBench 已开源） |

---

## 成功标准

- [ ] 至少 3 个不同模型族成功扩展（OpenAI / Anthropic / 智谱）
- [ ] 质量评分 ≥ 0.85（对比原生上下文）
- [ ] 检索时间 < 500ms（1M tokens corpus）
- [ ] 压缩后质量保持 ≥ 92%（参考 ATACompressor 基线）
- [ ] 通过 80%+ 单元测试
- [ ] 完成论文初稿

---

**创建时间**: 2026-05-02
**最后更新**: 2026-05-03
**状态**: Phase 2 已完成，Phase 3 待开始
