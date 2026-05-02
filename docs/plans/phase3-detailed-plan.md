# Phase 3 详细实施计划 — 压缩引擎 + 物理模型

**版本**: 1.0
**创建日期**: 2026-05-03
**目标周期**: Week 5-8 (2026-05-04 ~ 2026-05-17)

---

## 1. 目标

将 `system.py:_compress_to_budget()` 从简单截断升级为基于信息论（MDL）、熵最大化、和任务感知的自适应压缩管线。同时实现物理启发模型（热力学、量子场论）作为压缩策略的理论驱动层。

---

## 2. 当前状态分析

### 已有基础设施

| 组件 | 状态 | 说明 |
|------|------|------|
| `CompressionConfig` | ✅ 已存在 | config.py 中定义了 `use_mdl`, `use_entropy`, `entropy_threshold`, 三种 ratio |
| `CompressionResult` | ✅ 已存在 | types.py 中定义了 `compression_ratio`, `reduction_percentage` |
| `CompressionStrategy` | ✅ 已存在 | types.py 枚举: AGGRESSIVE/MODERATE/LIGHT/ADAPTIVE |
| `system.py:_compress_to_budget()` | ⚠️ 占位 | 当前仅按相关性截断，需要接入真实压缩 |
| `system.py:initialize()` | ⚠️ 注释 | `self._compression_engine = AdaptiveCompressor(...)` 已注释 |
| `adaptive.py:三策略` | ⚠️ 占位 | `_aggressive_compression()` 等方法返回 top-N 切片 |

### 关键接口约束

system.py 的压缩接口签名:
```python
async def _compress_to_budget(
    self, blocks: List[ContextBlock], budget: TokenBudget
) -> List[ContextBlock]
```

输入: 已选中的 ContextBlock 列表 + token 预算
输出: 压缩后适合预算的 ContextBlock 列表

---

## 3. 文件清单与职责

### 3.1 压缩模块 (`src/ucef/compression/`)

```
src/ucef/compression/
├── __init__.py            # 模块导出
├── adaptive.py            # AdaptiveCompressor — 主入口，策略路由
├── mdl.py                 # MDLCompressor — 最小描述长度压缩
├── entropy.py             # EntropyCompressor — 熵最大化选择
└── task_aware.py          # TaskAwareCompressor — 利用 LLM 的任务感知压缩
```

### 3.2 物理模块 (`src/ucef/physics/`)

```
src/ucef/physics/
├── __init__.py            # 模块导出
├── thermodynamic.py       # 热力学模型 — 自由能优化 + 温度调度
└── quantum_field.py       # 量子场论模型 — 重整化群多尺度摘要
```

### 3.3 需修改的现有文件

| 文件 | 修改内容 |
|------|----------|
| `src/ucef/core/system.py` | 取消注释 compression_engine，重写 `_compress_to_budget()` |
| `src/ucef/retrieval/adaptive.py` | 三个策略的压缩方法接入真实压缩器 |
| `src/ucef/retrieval/hyperbolic.py` | 实现 `train_embeddings()` 的 Riemannian SGD |
| `src/ucef/__init__.py` | 添加 compression 和 physics 模块导出 |

---

## 4. 详细设计

### 4.1 `compression/adaptive.py` — AdaptiveCompressor

```python
class AdaptiveCompressor:
    """
    自适应压缩器 — 根据 CompressionStrategy 路由到具体压缩算法。

    策略映射:
    - AGGRESSIVE: MDL 硬截断 (保留 10%)
    - MODERATE:   熵最大化 + MDL (保留 30%)
    - LIGHT:      任务感知摘要 (保留 50%)
    - ADAPTIVE:   根据 quality_retention 自动选择
    """

    def __init__(self, config: CompressionConfig, model_client: Optional[ModelClient] = None):
        self._config = config
        self._mdl = MDLCompressor(config) if config.use_mdl else None
        self._entropy = EntropyCompressor(config) if config.use_entropy else None
        self._task_aware = TaskAwareCompressor(model_client) if model_client else None

    async def compress(
        self,
        blocks: List[ContextBlock],
        budget: TokenBudget,
        strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE,
    ) -> Tuple[List[ContextBlock], CompressionResult]:
        """
        压缩入口。返回 (压缩后 blocks, 压缩统计)。
        """
        ...

    def _resolve_strategy(self, strategy: CompressionStrategy,
                          quality_retention: float) -> CompressionStrategy:
        """ADAPTIVE 策略根据质量保持率自动选择。"""
        ...
```

### 4.2 `compression/mdl.py` — MDLCompressor

**理论基础**: Grünwald, "The Minimum Description Length Principle", MIT Press 2007

```python
class MDLCompressor:
    """
    MDL 压缩器。

    原理:
        MDL = L(context) + L(query | context)
        目标: 在 L(context) ≤ budget 约束下最小化 MDL

    实现:
        1. 计算每个 block 的描述长度 L(block) = -log₂ P(block | query)
        2. 计算 L(query | block) = 交叉熵估计
        3. 按 MDL 分数排序，贪心选择直到预算用完
    """

    def description_length(self, text: str, query: str) -> float:
        """计算给定查询下文本的描述长度（bits）。"""
        # 基于词频的近似: L(text) = -Σ log₂ P(word)
        ...

    def compress_block(self, block: ContextBlock, target_ratio: float) -> ContextBlock:
        """压缩单个 block 到目标比例（句子级选择）。"""
        ...

    def compress_blocks(self, blocks: List[ContextBlock],
                        budget: int) -> List[ContextBlock]:
        """在预算内按 MDL 最优选择 block 子集。"""
        ...
```

### 4.3 `compression/entropy.py` — EntropyCompressor

**理论基础**: Shannon 信息熵, 最大熵原理 (Jaynes 1957)

```python
class EntropyCompressor:
    """
    熵最大化压缩器。

    原理:
        最大化 H(selected) = -Σ p_i log p_i
        约束: Σ tokens_i ≤ budget

    在保留与 query 最相关内容的同时，最大化所选内容的多样性，
    避免冗余信息占据预算。
    """

    def block_entropy(self, block: ContextBlock, all_blocks: List[ContextBlock]) -> float:
        """计算单个 block 的边际信息熵贡献。"""
        ...

    def select_diverse_subset(self, blocks: List[ContextBlock],
                              budget: int) -> List[ContextBlock]:
        """贪心选择最大化总熵的子集。"""
        ...

    def redundancy_score(self, block_a: ContextBlock,
                         block_b: ContextBlock) -> float:
        """计算两个 block 之间的冗余度（Jaccard / cosine）。"""
        ...
```

### 4.4 `compression/task_aware.py` — TaskAwareCompressor

```python
class TaskAwareCompressor:
    """
    任务感知压缩器 — 利用 LLM 生成摘要和提取关键句。

    策略:
        1. 将 block 拆分为句子
        2. 对每个句子计算 query 相关性分数
        3. 选择 top-k 句子重组为压缩 block
        4. (可选) 使用 LLM 生成摘要

    参考: ATACompressor (Jiang et al., 2023) — 60% 压缩 + 92% 性能保持
    """

    def __init__(self, model_client: Optional[ModelClient] = None):
        self._client = model_client

    def extract_key_sentences(self, text: str, query: str,
                               top_k: int = 5) -> List[str]:
        """提取与 query 最相关的 top-k 句子。"""
        ...

    async def summarize_block(self, block: ContextBlock,
                               query: str) -> ContextBlock:
        """使用 LLM 生成 block 的 query-aware 摘要。"""
        ...

    async def compress(self, blocks: List[ContextBlock],
                        budget: int, query: str) -> List[ContextBlock]:
        """任务感知压缩管线。"""
        ...
```

### 4.5 `physics/thermodynamic.py` — 热力学模型

**理论基础**: 统计力学 — 自由能 F = E - TS

```python
class ThermodynamicModel:
    """
    热力学上下文优化模型。

    将上下文选择映射为自由能最小化:
        F = E(context) - T · S(context)

    其中:
        E(context) = 上下文与查询的"能量"（不相关度）
        S(context) = 上下文的信息熵（多样性）
        T = 温度参数（控制探索 vs 利用）

    温度调度:
        - 高温 (T → ∞): 最大化多样性（探索阶段）
        - 低温 (T → 0): 最小化能量（利用阶段）
        - 退火: T 随压缩轮次递减
    """

    def __init__(self, temperature: float = 1.0, cooling_rate: float = 0.95):
        self._temperature = temperature
        self._cooling_rate = cooling_rate

    def energy(self, block: ContextBlock, query: str) -> float:
        """计算 block 相对于 query 的能量（不相关度）。"""
        ...

    def entropy_contribution(self, block: ContextBlock,
                              selected: List[ContextBlock]) -> float:
        """计算 block 对已选集合的边际熵贡献。"""
        ...

    def free_energy(self, block: ContextBlock, query: str,
                     selected: List[ContextBlock]) -> float:
        """F = E - T·S, 越低越好。"""
        ...

    def select_by_free_energy(self, blocks: List[ContextBlock],
                               query: str, budget: int) -> List[ContextBlock]:
        """按自由能排序选择 block。"""
        ...

    def anneal(self) -> None:
        """温度退火: T = T * cooling_rate。"""
        self._temperature *= self._cooling_rate
```

### 4.6 `physics/quantum_field.py` — 量子场论模型

**理论基础**: 重整化群 (Renormalization Group)

```python
class RenormalizationGroup:
    """
    重整化群多尺度摘要。

    受 Wilson 重整化群启发，将长文本通过多尺度粗粒化压缩:
        1. 最低尺度: 保留完整文本
        2. 中间尺度: 句子级摘要（保留关键句）
        3. 最高尺度: 段落级摘要（仅保留核心观点）

    类比:
        - 紫外截断 (UV cutoff) = token 预算
        - 有效作用量 (effective action) = 压缩后的上下文
        - 标度变换 (scale transformation) = 摘要/删减操作
    """

    def __init__(self, n_scales: int = 3):
        self._n_scales = n_scales

    def coarse_grain(self, text: str, target_ratio: float) -> str:
        """单次粗粒化操作：提取核心内容。"""
        ...

    def multiscale_compress(self, blocks: List[ContextBlock],
                             budget: int) -> List[ContextBlock]:
        """
        多尺度压缩:
            1. 评估每个 block 的重要性
            2. 重要性低的 block 进行更多尺度压缩
            3. 重要性高的 block 保留更多细节
        """
        ...

    def relevance_flow(self, blocks: List[ContextBlock],
                        query: str) -> List[float]:
        """
        相关性流 — 类比耦合常数随能标跑动。
        在不同压缩尺度下追踪 block 与 query 的相关度变化。
        """
        ...
```

### 4.7 `retrieval/hyperbolic.py` — Riemannian SGD 补充

```python
# 在现有 HyperbolicRetriever 类中实现
async def train_embeddings(
    self,
    documents: List[Document],
    n_epochs: int = 100,
    learning_rate: float = 0.01,
) -> None:
    """
    使用 Riemannian SGD 训练双曲嵌入。

    算法:
        1. 初始化: 在 Poincaré 球中随机采样
        2. 对每对 (doc_i, doc_j):
           a. 计算 d_hyp(emb_i, emb_j)
           b. 计算与语义相似度的差距
           c. 计算 Riemannian 梯度: ∇_hyp = (1-||x||²)²/4 · ∇_eucl
           d. 沿测地线更新: x' = exp_x(-η · ∇_hyp)
        3. 投影回 Poincaré 球: if ||x'|| ≥ 1, normalize to max_norm

    Reference: Nickel & Kiela (2017), Algorithm 1
    """
    ...
```

---

## 5. 集成修改

### 5.1 `system.py` 修改

```python
# initialize() 中取消注释:
from ucef.compression.adaptive import AdaptiveCompressor
self._compression_engine = AdaptiveCompressor(
    self._config.compression,
    model_client=self._model_client,
)

# _compress_to_budget() 重写:
async def _compress_to_budget(self, blocks, budget):
    if self._compression_engine is None:
        return self._simple_truncate(blocks, budget)  # fallback

    strategy = self._model_profile.recommended_strategy
    compressed, result = await self._compression_engine.compress(
        blocks, budget, strategy
    )
    return compressed
```

### 5.2 `adaptive.py` 修改

三个策略类的 `_*_compression()` 方法改为调用 `AdaptiveCompressor`:
```python
# SmallContextStrategy._aggressive_compression → 使用 MDLCompressor
# MediumContextStrategy._moderate_compression → 使用 EntropyCompressor
# LargeContextStrategy._light_compression → 使用 TaskAwareCompressor
```

### 5.3 `__init__.py` 更新

```python
# src/ucef/__init__.py 添加:
from ucef.compression.adaptive import AdaptiveCompressor
from ucef.compression.mdl import MDLCompressor
from ucef.compression.entropy import EntropyCompressor
from ucef.physics.thermodynamic import ThermodynamicModel
from ucef.physics.quantum_field import RenormalizationGroup
```

---

## 6. 实施顺序

```
Week 5 (Day 1-7):
├── Day 1-2: compression/__init__.py + compression/mdl.py
│   └── MDLCompressor: 描述长度计算 + 句子级选择
├── Day 3-4: compression/entropy.py
│   └── EntropyCompressor: 熵计算 + 多样性选择 + 冗余消除
└── Day 5-7: compression/adaptive.py
    └── AdaptiveCompressor: 策略路由 + 集成 MDL/Entropy

Week 6 (Day 8-14):
├── Day 8-10: compression/task_aware.py
│   └── TaskAwareCompressor: 关键句提取 + LLM 摘要
├── Day 11-12: physics/thermodynamic.py
│   └── ThermodynamicModel: 自由能 + 退火
└── Day 13-14: physics/quantum_field.py
    └── RenormalizationGroup: 多尺度粗粒化

Week 7 (Day 15-21):
├── Day 15-16: system.py 集成
│   └── _compress_to_budget() 接入 AdaptiveCompressor
├── Day 17-18: adaptive.py 集成
│   └── 三策略接入真实压缩器
└── Day 19-21: hyperbolic.py Riemannian SGD
    └── train_embeddings() 实现

Week 8 (Day 22-28):
├── Day 22-24: Review + Fix
│   └── 语法检查 + 交叉引用验证 + fix 记录
└── Day 25-28: 集成测试准备
    └── 手动端到端验证 + examples 更新
```

---

## 7. 验证标准

### 语法与引用
- [ ] 所有新文件通过 `ast.parse()` 语法检查
- [ ] 所有跨模块导入通过交叉引用验证
- [ ] 0 个 import 错误

### 功能验证
- [ ] `AdaptiveCompressor.compress()` 能正确路由策略
- [ ] `MDLCompressor` 输出 ≤ target_ratio tokens
- [ ] `EntropyCompressor` 选择结果多样性 > 纯相关性排序
- [ ] `ThermodynamicModel.free_energy()` 数值合理（负值 = 好选择）
- [ ] `RenormalizationGroup` 多尺度输出保持核心信息
- [ ] `train_embeddings()` 收敛（嵌入距离与语义相似度正相关）

### 集成验证
- [ ] `system.py:query()` 端到端通过，使用真实压缩
- [ ] `adaptive.py` 三策略不再返回简单切片
- [ ] `CompressionResult` 统计数据正确填充

---

## 8. 参考

- Grünwald, "The Minimum Description Length Principle", MIT Press 2007
- Jaynes, "Information Theory and Statistical Mechanics", Physical Review 1957
- Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations", NeurIPS 2017
- Wilson, "The Renormalization Group: Critical Phenomena and the Kondo Problem", Rev. Mod. Phys. 1975
- Jiang et al., "LLMLingua: Compressing Prompts for Accelerated Inference", ICLR 2023
