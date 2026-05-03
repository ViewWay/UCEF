# UCEF 实验升级方案

本地 24GB MLX + 7B 量化模型的实验已经跑通，但要对标竞品需要更强的实验条件。以下列出 **5 种方案**，从成本最低到最强配置。

---

## 方案对比总览

| 方案 | 模型级别 | 预估成本 | 上手难度 | 数据说服力 | 推荐度 |
|------|---------|---------|---------|-----------|-------|
| A. 低成本 API | 7B-14B 开源模型 | ¥0-10 | ⭐ | ★★☆ | 验证阶段 |
| B. DeepSeek API | V3.2 / R1 | ¥5-20 | ⭐ | ★★★★ | **强烈推荐** |
| C. Cloud GPU (AutoDL) | 72B 开源 | ¥30-80 | ⭐⭐ | ★★★★ | 需要开源模型数据 |
| D. 顶级 API (GPT-4o/Claude) | SOTA 闭源 | $5-20 | ⭐ | ★★★★★ | 最终对标 |
| E. 本地 + 真实数据 | 7B 本地 | ¥0 | ⭐ | ★★★ | 基线对照 |

---

## 方案 A：低成本 API（硅基流动 / OpenRouter 免费）

**适合**：验证实验框架 + 真实数据集链路是否跑通

### 平台

- **硅基流动 SiliconFlow**：国内，延迟低，注册送免费额度
  - Qwen2.5-7B-Instruct: **免费**
  - DeepSeek-V3: ~¥4/百万 token
  - [siliconflow.com](https://siliconflow.com/zh/pricing)

- **OpenRouter**：海外，聚合多家模型
  - 多个模型完全免费
  - 统一 OpenAI 兼容 API
  - [openrouter.ai/collections/free-models](https://openrouter.ai/collections/free-models)

### 操作

```python
# 硅基流动 — 改一行就能用
from openai import OpenAI
client = OpenAI(
    api_key="sk-你的硅基流动key",
    base_url="https://api.siliconflow.cn/v1"
)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=256
)
```

### 成本估算

- 硅基流动 Qwen2.5-7B: **免费**
- 200 样本 × 3 基准 × ~3000 tokens/样本: ~1.8M tokens → **¥0**
- 验证框架 + 真实数据集下载后一次性跑通

---

## 方案 B：DeepSeek API（强烈推荐）

**适合**：低成本产出有竞争力的对标数据

DeepSeek V3.2 是当前性价比最高的推理 API，1M token 只需 $0.14-0.27。

### 价格

| 模型 | 输入 ($/MTok) | 输出 ($/MTok) | 上下文窗口 | 免费额度 |
|------|:-------------:|:-------------:|:----------:|:-------:|
| V3.2 | $0.14 | $0.28 | 128K | 500M/月 |
| R1 | $0.55 | $2.19 | 128K | 500M/月 |
| V4-Pro | ~$1.74 | — | **1M** | 促销中 |

### 成本估算

```
实验规模: 3 基准 × 200 样本
每样本: ~3000 input tokens + ~300 output tokens
总计: ~1.8M input + ~180K output

DeepSeek V3.2:
  Input:  1.8M × $0.14/MTok = $0.25
  Output: 0.18M × $0.28/MTok = $0.05
  总计: $0.30 (≈ ¥2)

加上 GPT-4o 对比跑一轮:
  Input:  1.8M × $2.50/MTok = $4.50
  Output: 0.18M × $10/MTok = $1.80
  总计: $6.30 (≈ ¥45)
```

**推荐组合**: DeepSeek V3.2（免费额度内）+ GPT-4o（$6.30）= 总计约 **¥47**

### 优势

- OpenAI 兼容 API，改 base_url 即可
- 免费额度 500M tokens/月，足够跑完全部实验
- 1M 上下文窗口，远超本地 7B 的 32K

---

## 方案 C：Cloud GPU (AutoDL)

**适合**：需要开源大模型（72B+）的完整实验数据

### 推荐配置

| 配置 | GPU | 模型 | 价格 | 适合场景 |
|------|-----|------|------|---------|
| 入门 | RTX 4090 (24GB) | Qwen2.5-14B-AWQ | ¥1-2/h | 消融实验 |
| 主力 | A100 80GB × 1 | Qwen2.5-72B-AWQ | ¥4-6/h | 主实验 |
| 高端 | A100 80GB × 2 | Qwen2.5-72B BF16 | ¥8-12/h | 最佳质量 |

### 操作流程

```bash
# 1. 在 AutoDL 创建实例 (PyTorch + Python 3.12 镜像)
# 2. SSH 登录后:
git clone <your-repo> && cd extend-Context-System
pip install -e ".[dev]"
pip install vllm datasets rouge-score

# 3. 下载真实数据
python3 -c "
from datasets import load_dataset
lb = load_dataset('THUDM/LongBench', '2wikimqa_e', split='test[:200]')
"

# 4. 启动 vLLM 服务
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct-AWQ \
    --max-model-len 32768 \
    --port 8080

# 5. 跑实验
PYTHONPATH=src python3 experiments/real_experiment.py -b all -m cloud-qwen-72b -n 200
```

### 成本估算

- 下载模型 + 数据: ~30 min
- 跑 3 基准 × 200 样本: ~2-3h
- A100 80GB 单卡: ¥4-6/h × 3h = **¥12-18**
- **总计约 ¥15-25**（含启动和关机时间）

### 平台推荐

| 平台 | 优点 | 缺点 |
|------|------|------|
| [AutoDL](https://www.autodl.com/) | 国内最便宜，镜像丰富 | 偶尔缺卡 |
| [RunPod](https://www.runpod.io/) | 海外，A100 $1.39/h 起 | 需要信用卡 |
| [阿里云 PAI](https://www.aliyun.com/product/pai) | 稳定，企业级 | 稍贵 |

---

## 方案 D：顶级 API (GPT-4o + Claude)

**适合**：最终对标，产出论文级数据

### 价格对比

| 模型 | 输入 ($/MTok) | 输出 ($/MTok) | 上下文 |
|------|:-------------:|:-------------:|:------:|
| GPT-4o | $2.50 | $10.00 | 128K |
| GPT-4o (Batch) | $1.25 | $5.00 | 128K |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 200K |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K |
| DeepSeek V3.2 | $0.14 | $0.28 | 128K |

### 推荐: 用 Batch API 省 50%

OpenAI Batch API 异步执行，24h 内返回结果，价格减半。适合非实时实验。

```python
# Batch API 示例
client.batches.create(
    input_file_id="file-xxx",
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### 成本估算（全套对标）

```
4 模型 × 3 基准 × 200 样本 = 2400 次推理

DeepSeek V3.2 (免费额度):  $0
GPT-4o (Batch):            $3.15
Claude Haiku 4.5:          $3.78
Claude Sonnet 4.6:         $9.00
───────────────────────────
总计: ~$16 (≈ ¥115)
```

---

## 方案 E：本地 + 真实数据（零成本升级）

**适合**：不花钱，先把数据说服力提上去

现有合成数据的 ROUGE-L 普遍偏低是因为 reference 质量差。换成 HuggingFace 真实数据集后，即使还是 7B 模型，ROUGE-L 也会有显著提升。

### 操作

```bash
# 本地 MLX 服务器已经在跑
pip install datasets

# 下载 LongBench 真实数据
python3 -c "
from datasets import load_dataset
import json
lb = load_dataset('THUDM/LongBench', '2wikimqa_e', split='test[:50]')
data = [{'id': i, 'context': x['context'], 'input': x['input'], 'answers': x['answers']} for i, x in enumerate(lb)]
json.dump(data, open('experiments/data/longbench.json', 'w'), ensure_ascii=False, indent=2)
print(f'Downloaded {len(data)} samples')
"

# 跑实验
PYTHONPATH=src python3 experiments/real_experiment.py -b longbench -m mlx-qwen-7b -n 50
```

---

## 推荐执行顺序

```
Step 1: 方案 E — 本地 + 真实数据            成本: ¥0
        ↓ 确认数据链路跑通
Step 2: 方案 B — DeepSeek API (免费额度)    成本: ¥0
        ↓ 产出有竞争力的对标基线
Step 3: 方案 D — GPT-4o Batch API           成本: ~$3
        ↓ 加上 SOTA 模型对比
Step 4: 方案 C — AutoDL A100 (可选)         成本: ~¥20
        ↓ 开源 72B 模型数据
Step 5: 汇总全部结果，撰写 competitive analysis
```

**总预算: ¥0 ~ ¥140**（取决于选几个方案叠加）

---

## 参考链接

- [硅基流动 SiliconFlow](https://siliconflow.com/zh/pricing)
- [OpenRouter 免费模型](https://openrouter.ai/collections/free-models)
- [DeepSeek API 定价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
- [OpenAI API 定价](https://developers.openai.com/api/docs/pricing)
- [Claude API 定价](https://platform.claude.com/docs/en/about-claude/pricing)
- [AutoDL](https://www.autodl.com/)
- [RunPod A100 $1.39/h](https://www.runpod.io/gpu-models/a100-pcie)
