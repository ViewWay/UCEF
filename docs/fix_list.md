# UCEF Fix List

记录项目中所有修复的问题，包括 Phase 1 实现修复和 Review 修复。

---

## Phase 1 实现修复（2026-05-03）

来源：交接文档 `docs/HANDOFF.md` 中标记的 3 个 HIGH 优先级问题。

### FIX-001: setup.py — entry_points 引用不存在的 CLI 模块

- **严重度**: HIGH
- **文件**: `setup.py:44-48`
- **问题**: `entry_points` 中 `"ucef=ucef.cli:main"` 引用了不存在的 `ucef.cli` 模块，会导致 `pip install` 失败。
- **修复**: 注释掉 CLI entry point，待 Phase 1 收尾实现 CLI 后恢复。
- **修复内容**:
  ```python
  # 修复前
  entry_points={
      "console_scripts": [
          "ucef=ucef.cli:main",
      ],
  },
  
  # 修复后
  entry_points={
      "console_scripts": [
          # CLI entry point — will be implemented in Phase 1 completion
          # "ucef=ucef.cli:main",
      ],
  },
  ```

### FIX-002: examples/basic_usage.py — 导入不存在的类和 API

- **严重度**: HIGH
- **文件**: `examples/basic_usage.py`
- **问题**:
  1. 导入 `from ucef import UniversalContextSystem, ModelCapabilityProfiler, AdaptiveContextExtender` 时 `UniversalContextSystem` 尚未实现
  2. 构造函数签名与实际实现不匹配（传入了 `tokenizer` 参数）
  3. 使用 `transformers` 库加载模型，但未在 dependencies 中声明
  4. 引用不存在的 `load_large_corpus()` 和 `load_documents()` helper 函数
  5. Example 3-5 使用旧 API 签名（`from ucef.quality import ...`）
- **修复**: 完全重写 `examples/basic_usage.py`，包含 5 个可运行示例：
  - Example 1: 基本文档存储和查询（使用 MockModelClient）
  - Example 2: 模型能力分析（已知模型无需 client）
  - Example 3: 双曲几何操作演示
  - Example 4: 量子启发选择演示
  - Example 5: 完整管线（Mock 模型）
- **修复内容**:
  - 导入改为 `from ucef import (UniversalContextSystem, UCEFConfig, Document, HyperbolicPoint, QuantumState, ...)`
  - 所有示例使用 MockModelClient，不依赖外部模型库
  - 添加 `import numpy as np` 依赖
  - 每个示例包含完整的可运行代码

### FIX-003: profiler.py — 缺少 model_client None 验证

- **严重度**: HIGH
- **文件**: `src/ucef/quality/profiler.py:66-80`
- **问题**: `profile_model(model_client, model_name)` 接受 `model_client=None`，但不验证。传入 None 且模型不在已知列表中时，后续方法会在 `_measure_performance_curve` 中因 `model_client` 为 None 而产生 AttributeError。
- **修复**: 在方法入口添加验证逻辑——未知模型必须提供 client，否则抛出清晰的 ValueError。
- **修复内容**:
  ```python
  # 新增验证逻辑
  normalized = model_name.lower().strip()
  is_known = any(known_name in normalized for known_name in self.MODEL_SPECS)
  
  if model_client is None and not is_known:
      raise ValueError(
          f"Model '{model_name}' is not in known specs. "
          f"A model_client is required for profiling unknown models. "
          f"Known models: {list(self.MODEL_SPECS.keys())}"
      )
  ```
- **附带修改**: 类型标注从 `model_client: Any` 改为 `model_client: Optional[Any]`

### FIX-004: __init__.py — 导出不存在的模块

- **严重度**: HIGH
- **文件**: `src/ucef/__init__.py`
- **问题**: `from ucef.core.system import UniversalContextSystem` 引用了尚未创建的 `core/system.py`，导致整个包无法导入。
- **修复**: 随着 `core/system.py`、`core/config.py`、`core/types.py` 的创建，更新 `__init__.py` 导出所有已实现的符号。
- **修复内容**:
  ```python
  # 新增导出
  from ucef.core.config import UCEFConfig
  from ucef.core.types import (
      CompressionStrategy, ContextBlock, ContextCategory, Document,
      HyperbolicPoint, ModelProfile, QuantumState, QueryResult, TokenBudget,
  )
  ```

---

## Review 修复（2026-05-03）

来源：Phase 1 完成后的全面代码审查。

### FIX-005: RESEARCH_SURVEY.md — Riemannian 梯度公式缺少 /4 因子

- **严重度**: HIGH（数学错误）
- **文件**: `docs/RESEARCH_SURVEY.md:47`
- **问题**: Riemannian 梯度公式写作 `∇_hyp = (1 - ||x||²)² · ∇_eucl`，缺少分母 `4`。
- **正确公式**: `∇_hyp = (1 - ||x||²)² / 4 · ∇_eucl`
- **参考文献**: Nickel & Kiela (2017), Supplementary Material Eq. S7
- **修复**: 补正确公式和引用出处。

### FIX-006: RESEARCH_SURVEY.md — 量子文献引用不完整

- **严重度**: HIGH（学术规范）
- **文件**: `docs/RESEARCH_SURVEY.md:74-82`
- **问题**:
  - Zuccon et al. 缺少 arXiv ID（应为 arXiv:1305.6247）
  - QISA 论文缺少具体 arXiv ID
  - 缺少 van Rijsbergen 2004 年专著引用（量子 IR 的理论基础）
  - 部分论文缺少 venue 信息
- **修复**: 补全 6 篇文献的完整引用（含 arXiv ID、venue、年份）。
- **修复内容**:
  | 论文 | 补充信息 |
  |------|----------|
  | van Rijsbergen | CUP (book), 2004 |
  | Zuccon et al. | ICTIR / arXiv:1305.6247 |
  | QISA | arXiv:2603.03318 |
  | Quantum Contextual Search | Entropy 25(5):828, MDPI |

### FIX-007: RESEARCH_SURVEY.md — 缺少评估协议和计算复杂度分析

- **严重度**: HIGH（完整性）
- **文件**: `docs/RESEARCH_SURVEY.md`
- **问题**: 研究调研缺少两个关键部分：
  1. 各操作的计算复杂度分析（实际实现的可行性依据）
  2. 评估协议（如何验证框架性能声称）
- **修复**: 新增两个章节：
  - **5.4 计算复杂度分析**: 列出 Poincaré 距离 O(d)、双曲最近邻 O(n·d) / O(log n) with HNSW、量子态构造 O(n)、密度矩阵 O(n²)、完整管线 O(n·d + n log n)
  - **5.5 评估协议**: 定义 baselines（标准 RAG、LongLLMLingua、原生长上下文）、数据集（LongBench、NarrativeQA、GovReport）、指标（ROUGE-L、BERTScore、Recall@K、延迟）、成功标准（质量≥85%、延迟<500ms、≥3 模型族验证）

### FIX-008: types.py — 零振幅量子态归一化不应回退到等概率叠加

- **严重度**: HIGH（数学正确性）
- **文件**: `src/ucef/core/types.py` QuantumState.normalize()
- **问题**: 当所有振幅为零时，normalize() 回退到等概率叠加。这在数学上是错误的——零振幅不是有效量子态，静默回退会掩盖调用方的逻辑错误。
- **修复**: 改为抛出 `ValueError("Cannot normalize a zero-amplitude quantum state.")`

### FIX-009: types.py — 密度矩阵 PSD 检查容差过大

- **严重度**: MEDIUM（数值精度）
- **文件**: `src/ucef/core/types.py` DensityMatrix.is_valid
- **问题**: 正半定检查使用 `eigenvalues >= -1e-6`，容差偏大。对于纯态密度矩阵 ρ = |ψ⟩⟨ψ|，特征值应为精确非负，允许 -1e-6 可能放过数值不稳定的矩阵。
- **修复**: 收紧容差为 `eigenvalues >= -1e-10`

### FIX-010: config.py — budget 百分比验证器无实际逻辑

- **严重度**: MEDIUM（验证缺失）
- **文件**: `src/ucef/core/config.py` MemorySystemConfig
- **问题**: `validate_budget_pct` 验证器直接 `return v`，不做任何检查。`validate_total_budget()` 方法存在但不在 Pydantic 验证链中，需要手动调用。
- **修复**:
  - 删除无用的 `validate_budget_pct` field_validator
  - 删除手动调用的 `validate_total_budget()` 方法
  - 改用 Pydantic v2 的 `@model_validator(mode="after")` 在模型创建后自动验证三项之和
  - 不满足时抛出 `ValueError` 包含具体数值

### FIX-011: system.py — 量子振幅构造缺少 Born rule 说明和归一化检查

- **严重度**: MEDIUM（数学严谨性）
- **文件**: `src/ucef/core/system.py` _quantum_select()
- **问题**:
  1. 从 scores 构造概率后直接创建 QuantumState，缺少 Born rule `P(i) = |αᵢ|²` 的注释说明
  2. 未检查归一化状态
- **修复**:
  - 添加注释说明 Born rule 转换过程：`αᵢ = √P(i), P(i) = |αᵢ|²`
  - 构造后添加归一化检查：`if not state.is_normalized: state = state.normalize()`

### FIX-012: system.py — uuid import 位置

- **严重度**: LOW（代码风格）
- **文件**: `src/ucef/core/system.py` store_text()
- **问题**: `import uuid` 在方法体内部，每次调用都执行 import 语句。
- **修复**: 移到模块级 import 区域。

### FIX-013: system.py — get_stats 引用不存在的 context_category 属性

- **严重度**: MEDIUM（运行时错误）
- **文件**: `src/ucef/core/system.py` get_stats()
- **问题**: `self._model_profile.context_category.value` 访问的 `context_category` 是字符串而非 `ContextCategory` 枚举（profiler.py 返回的是旧格式字符串），导致 `.value` 属性不存在。
- **修复**: 改用 `profile.classify_context_category().value`，该方法根据 `native_context_window` 重新计算正确的 `ContextCategory` 枚举值。

---

## 未修复（非阻塞 / 后续 Phase 处理）

| ID | 严重度 | 描述 | 计划 |
|----|--------|------|------|
| TODO-001 | MEDIUM | PROJECT_STATUS.md 和 README.md 仍描述规划的完整状态 | 各 Phase 完成时同步更新 |
| TODO-002 | MEDIUM | compression/ 和 physics/ 目录尚未创建 | Phase 3-4 |
| TODO-003 | MEDIUM | 单元测试尚未编写 | Phase 2 Review 后 |
| TODO-004 | LOW | types.py 缺少 parallel_transport、坐标变换等操作 | Phase 3 按需添加 |
| TODO-005 | LOW | requirements.txt 与 setup.py 依赖版本不完全对齐（pydantic 2.0 vs 2.8） | 统一为 2.8 |
| ~~TODO-006~~ | ~~LOW~~ | ~~profiler.py 返回的 context_category 是字符串~~ | **已在 FIX-016 修复** |

---

## Phase 2 Review 修复（2026-05-03）

### FIX-014: three_layer.py — import asyncio 在文件底部

- **严重度**: CRITICAL（运行时错误）
- **文件**: `src/ucef/memory/three_layer.py`
- **问题**: `import asyncio` 在文件最后一行，但 `asyncio.gather()` 在 `delete()` 方法中被调用。Python 执行到方法时 asyncio 尚未导入，导致 NameError。
- **修复**: 将 `import asyncio` 移到模块级 import 区域。

### FIX-015: retrieval/__init__.py — 缺少 fusion 模块导出

- **严重度**: HIGH（功能不可用）
- **文件**: `src/ucef/retrieval/__init__.py`
- **问题**: `fusion.py` 定义了 `ReciprocalRankFusion`、`WeightedScoreFusion`、`HybridFusion` 三个公开类，但 `__init__.py` 没有导入和导出它们，导致用户无法通过 `from ucef.retrieval import ReciprocalRankFusion` 访问。
- **修复**: 添加 fusion 模块的三个类到 `__init__.py` 的导入和 `__all__`。

### FIX-016: profiler.py — ModelProfile 重复定义

- **严重度**: HIGH（类型冲突）
- **文件**: `src/ucef/quality/profiler.py`
- **问题**: profiler.py 定义了自己的 `ModelProfile` dataclass（`context_category: str`），而 types.py 也定义了 `ModelProfile`（`context_category: ContextCategory` 枚举）。两个类名字相同但类型不兼容，导致 system.py 导入时产生歧义。
- **修复**:
  - 删除 profiler.py 中的本地 `ModelProfile` 定义
  - 改为从 `ucef.core.types` 导入 `ModelProfile`, `ContextCategory`, `CompressionStrategy`
  - `MODEL_SPECS` 中的 category 字符串改为 `ContextCategory` 枚举
  - `_recommend_strategy` 返回 `CompressionStrategy` 枚举而非字符串
  - `_get_context_spec` 默认值改为 `ContextCategory.MEDIUM`

### FIX-017: types.py — 冗余配置类

- **严重度**: MEDIUM（设计冗余）
- **文件**: `src/ucef/core/types.py`
- **问题**: types.py 底部定义了 `RetrievalConfig` 和 `MemoryConfig` 两个 dataclass，而 config.py 已经有更完整的 Pydantic 版本（`HyperbolicConfig`、`HotMemoryConfig` 等）。两者功能重叠，容易造成混淆。
- **修复**: 删除 types.py 中的 `RetrievalConfig` 和 `MemoryConfig`。统一使用 config.py 中带验证的 Pydantic 版本。

---

## Phase 2 交叉引用验证修复（2026-05-03）

### FIX-018: three_layer.py — TokenBudget 从错误模块导入

- **严重度**: HIGH（导入错误）
- **文件**: `src/ucef/memory/three_layer.py:27`
- **问题**: `from ucef.core.config import MemorySystemConfig, TokenBudget` 但 `TokenBudget` 定义在 `ucef.core.types`，不在 `config` 中。
- **修复**: 将 `TokenBudget` 移到 `from ucef.core.types` 的导入行。

### FIX-019: quality/__init__.py — 导入不存在的 monitor 模块

- **严重度**: HIGH（功能不可用）
- **文件**: `src/ucef/quality/__init__.py:5`
- **问题**: `from ucef.quality.monitor import QualityMonitor` 引用不存在的 `monitor.py`，导致整个 quality 包无法导入。
- **修复**: 删除对 `QualityMonitor` 的导入和导出，待后续 Phase 实现。

### FIX-020: models/__init__.py — 导入不存在的 Phase 3 适配器

- **严重度**: HIGH（功能不可用）
- **文件**: `src/ucef/models/__init__.py`
- **问题**: 导入 5 个 Phase 3 才实现的适配器（base、openai、anthropic、zhipu、local），导致 models 包无法导入。
- **修复**: 替换为 Phase 3 placeholder，仅导出 `ModelClient` Protocol 供用户直接实现。

---

## 运行时验证修复（2026-05-03）

### FIX-021: config.py — pydantic 硬依赖导致整个包无法导入

- **严重度**: CRITICAL（整个包不可用）
- **文件**: `src/ucef/core/config.py`
- **问题**: `from pydantic import BaseModel, Field, ...` 是硬导入。当 pydantic 未安装时（新环境 / CI / 最小安装），`config.py` 无法加载，导致 9 个模块全部 ImportError。整个 UCEF 包不可导入。
- **修复**: 实现 pydantic / dataclass 双后端。`try: from pydantic import ... except ImportError:` 触发 dataclass fallback。所有 9 个配置类在两个分支中保持相同字段名和默认值。dataclass 版本保留了关键验证（curvature < 0、budget 之和 ≈ 1.0）。
- **验证**: 无 pydantic 环境下全包导入 + 端到端管线测试通过。

### 验证结果

运行时测试覆盖 6 个模块 + 端到端管线：

| 模块 | 测试 | 结果 |
|------|------|------|
| `ucef.core.types` | 12 个数学操作 | ✅ |
| `ucef.retrieval.hyperbolic` | 索引 + 检索 + 文本检索 | ✅ |
| `ucef.retrieval.quantum` | 量子选择 + 归一化 | ✅ |
| `ucef.retrieval.fusion` | RRF + WeightedScore + Hybrid | ✅ |
| `ucef.memory` (三层) | store + retrieve + delete + stats | ✅ |
| `ucef.quality.profiler` | 4 个模型 profile + 错误处理 | ✅ |
| `UniversalContextSystem` E2E | 完整 query 管线 | ✅ |

---

## Phase 3 实现修复（2026-05-03）

### FIX-022: config.py — pydantic 硬依赖导致整个包不可导入

- **严重度**: CRITICAL（整个包不可用）
- **文件**: `src/ucef/core/config.py`
- **问题**: `from pydantic import ...` 是硬导入。无 pydantic 时 9 个模块 ImportError。
- **修复**: 实现 pydantic/dataclass 双后端。所有 9 个配置类两个分支字段完全一致。dataclass 版保留关键验证（curvature < 0、budget ≈ 1.0）。

### FIX-023: system.py — 压缩引擎初始化从注释改为真实实现

- **严重度**: HIGH（功能缺失）
- **文件**: `src/ucef/core/system.py:initialize()` + `_compress_to_budget()`
- **问题**: `self._compression_engine = AdaptiveCompressor(...)` 被注释，`_compress_to_budget()` 使用简单截断。
- **修复**: 取消注释并接入真实 AdaptiveCompressor。`_compress_to_budget()` 优先使用压缩引擎，保留 truncation fallback。

---

## Phase 3 Review 修复（2026-05-03）

### FIX-024: system.py — query 未传递给压缩引擎

- **严重度**: HIGH（压缩质量受损）
- **文件**: `src/ucef/core/system.py:256, 450`
- **问题**: `_compress_to_budget(blocks, budget)` 不接受 query 参数，内部 `query=""` 硬编码。压缩器无法根据用户查询进行 query-aware 压缩，降低了相关性保持率。
- **修复**: `_compress_to_budget` 添加 `query: str = ""` 参数；`query()` 方法调用处改为 `_compress_to_budget(selected, budget, query=query)`。

### FIX-025: adaptive.py — MDL 压缩器无异常保护

- **严重度**: MEDIUM（健壮性）
- **文件**: `src/ucef/compression/adaptive.py:118-119`
- **问题**: `_compress_aggressive()` 直接调用 `self._mdl.compress_blocks()` 无 try-except。如果 MDL 计算遇到异常（空文本、数值下溢等），整个压缩管线崩溃。
- **修复**: 添加 try-except 包裹，异常时 fallback 到 `_truncate_by_relevance()`。

---

## Phase 4 实现修复（2026-05-03）

### FIX-026: system.py — 反馈循环无限递归风险

- **严重度**: CRITICAL（运行时无限递归）
- **文件**: `src/ucef/core/system.py:query()`
- **问题**: `query()` 方法在质量低于阈值时调用 `self._feedback_loop.refine(requery_fn=self.query)`，而 `refine()` 内部调用 `requery_fn(query)` 即 `self.query()`，后者又会检查质量并再次触发反馈循环，导致无限递归。
- **修复**: 添加 `self._in_feedback_loop` 布尔标志。`query()` 在触发反馈循环前设为 `True`，反馈循环内部递归调用 `query()` 时跳过反馈逻辑，`finally` 块确保标志重置。

### FIX-027: preservation.py — QualityIssue 重复定义与 types.py 冲突

- **严重度**: HIGH（类型冲突）
- **文件**: `src/ucef/quality/preservation.py`
- **问题**: `preservation.py` 本地定义了 `QualityIssue` dataclass（`type: str`），而 `types.py` 也定义了 `QualityIssue`（`issue_type: QualityIssueType` 枚举）。两个字段名不同、类型不兼容。
- **修复**: 删除 `preservation.py` 的本地 `QualityIssue` 定义，改为从 `ucef.core.types` 导入。所有引用从 `issue.type` 改为 `issue.issue_type`，字符串键改为枚举值。

### FIX-028: config.py — QualityConfig 缺少 monitor_window_size 和 max_refinement_iterations

- **严重度**: MEDIUM（配置缺失）
- **文件**: `src/ucef/core/config.py` QualityConfig
- **问题**: `system.py` 引用 `self._config.quality.monitor_window_size` 和 `self._config.quality.max_refinement_iterations`，但 `QualityConfig` 未定义这两个字段，导致 `AttributeError`。
- **修复**: 在 `QualityConfig` 的 pydantic 和 dataclass 两个分支中添加 `monitor_window_size: int = 100` 和 `max_refinement_iterations: int = 3`。

---

## Phase 5 Review 修复（2026-05-03）

### FIX-029: openai.py — token 使用统计被丢弃

- **严重度**: MEDIUM（统计不准确）
- **文件**: `src/ucef/models/openai.py:123-127`
- **问题**: `_generate_impl()` 中 `response.usage` 的 token 统计数据被 `self._stats[-1:] = []` 清空后未重新填充，导致 OpenAI 适配器的 token 计数始终为 0。
- **修复**: 改为将 `response.usage.prompt_tokens`、`completion_tokens`、`total_tokens` 正确写入 `self._stats[-1]`。

### FIX-030: zhipu.py — 使用已弃用的 asyncio.get_event_loop()

- **严重度**: MEDIUM（兼容性）
- **文件**: `src/ucef/models/zhipu.py:116-117`
- **问题**: 使用 `asyncio.get_event_loop()` 在 Python 3.10+ 中已弃用，且在 async 上下文中可能返回错误的事件循环。zhipuai SDK 是同步的，需要通过 `run_in_executor` 包装。
- **修复**: 改为 `asyncio.get_running_loop().run_in_executor(None, _sync_call)`，确保使用当前运行的事件循环。

---

**最后更新**: 2026-05-03
**维护者**: UCEF 项目自动记录
