---
license: apache-2.0
language:
  - zh
  - en
pipeline_tag: text-generation
library_name: transformers
tags:
  - minimind
  - math
  - pretrain
  - qwen3
  - from-scratch
---

# MiniMind-Math-v3

`minimind-math-v3` 是一个基于 [MiniMind](https://github.com/jingyaogong/minimind) 项目从零训练的小型中文数学语言模型，权重已转换为 Qwen3 兼容格式，可直接使用 HuggingFace `transformers` 加载推理。

## 模型描述

- **训练方式**：从零预训练（随机初始化，无任何基座权重）
- **架构**：Dense Transformer（Decoder-only），未使用 MoE
- **数据**：以数学/推理语料为主的预训练数据（pretrain_t2t_mini.jsonl），并经过数学 SFT 训练
- **Tokenizer**：基于 MiniMind 自训 tokenizer 修改的 patched tokenizer，详见下方说明

## 代码仓库

本模型的预训练与 SFT 均基于 [jiwubu/minimind](https://github.com/jiwubu/minimind) 代码仓库完成。相对上游 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的主要修改是**新增 `math/` 目录**，用于生成数学 SFT 用的加减乘除算术数据（`gen_math_data_addsub.py`、`gen_math_data_muldiv.py` 等），并提供配套评测脚本（`eval_math.py`）。

## 模型规格

| 参数 | 值 |
|---|---|
| 参数量 | ~64M |
| hidden_size | 768 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads (GQA) | 4 |
| head_dim | 96 |
| intermediate_size | 2432 |
| 激活函数 | SiLU (SwiGLU) |
| 位置编码 | RoPE (theta = 1e6) |
| 最大上下文长度 | 32768 |
| 词表大小 | 6400 |
| tie_word_embeddings | true |
| 精度 | float16 |

## 快速开始

最简单的方式是用仓库自带的交互式验证脚本（chat template + 贪心解码 + 流式输出）：

```bash
python verify_model.py
# 输入> 竖式计算 1245 + 111
# 🧠: 竖式计算 1245 + 111
# ...逐位演算过程...
# 答案(3位): 1356
```

或直接用 transformers：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "jiwubu/minimind-math-v3",
    torch_dtype="float16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("jiwubu/minimind-math-v3")

messages = [{"role": "user", "content": "竖式计算 1245 + 111"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,   # 贪心解码，结果可复现
)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> 注意：本模型经过预训练与数学 SFT 训练，推理时**必须套 chat template**——裸文本输入（不套模板）是分布外输入，会产生无意义的重复输出。数学能力范围见下方评测一节。

## 训练细节

- 预训练 2 个 epoch，batch size 32，最大序列长度 340
- 优化器 AdamW，初始学习率 5e-4（带 warmup + 余弦衰减）
- 混合精度：bfloat16

## Tokenizer 修改

本模型使用基于 MiniMind 原版 tokenizer 修改的 **patched tokenizer**（`Sequence[Digits, ByteLevel]`）：

- **数字全局一位一 token**：每个数字字符（0-9）单独切分为一个 token，使模型能更稳定地处理算术运算中的数字对齐
- **ID 兼容**：修改后的 tokenizer 与原版 token id 完全兼容，不影响已有 token 序列
- **词表冻结**：词表大小保持 6400 不变，embedding 层未重训

## 数学能力评测

评测使用 `math/eval_math.py`（`--max_digits 8 --n 25`，贪心解码，按整数值判定对错），**训练范围内总体准确率 98.6%（3975 题）**。

### 加减法（8 位以内，各分档准确率 %）

| 档位 | 准确率 | 档位 | 准确率 |
|---|---|---|---|
| 等长± | 100 | 负数 | 98.9 |
| 不等长± | 100 | 负数·等长接近 | 95.4 |
| 短在前 | 99.3 | 前导零 | 88.0 |
| 紧凑写法 | 100 | 长串答案 | 99.3 |
| 紧凑不等长 | 99.3 | 全同数字 | 99.7 |
| 负数共后缀 | 99.4 | 长零串 | 100 |

### 乘除法（≤4 位，各分档准确率 %）

| 档位 | 准确率 | 档位 | 准确率 |
|---|---|---|---|
| 乘法（通用） | 100 | 整除 | 97.0 |
| 乘一位乘数 | 100 | 带余 | 89.0 |
| 乘末尾零 | 99.0 | 商含 0 | 98.0 |
| 乘内嵌 0 | 100 | 商零串 | 98.0 |
| 乘整十幂 | 100 | 小除以大 | 96.0 |
| 乘全同数字 | 100 | 除数 1 位 | 100 |
| 乘 9 串 | 100 | 除数 2 位 | 88.9 |
| 乘相等 | 100 | — | |

### 已知限制

- 近等值比较判定（负数·等长接近 95.4%、前导零 88.0%）为 64M / 单 epoch SFT 的能力边界
- 带余除法 89.0% 为残余难点
- 乘除 ≥5 位**未训练**，属于外推场景（实测 0~12%）
- 固定输出格式：模型始终输出完整竖式演算过程

## 局限性

- 模型规模很小（~64M），数学推理能力有限，仅用于学习与研究
- 经过数学 SFT 但规模有限，可能生成不准确或无意义的内容
- 上下文以短序列训练（340 tokens），长文本能力未经充分训练

## 引用

如果你使用了本模型，欢迎引用 MiniMind 项目：

```bibtex
@misc{minimind,
  author = {jingyaogong},
  title  = {MiniMind: Training a small LLM from scratch},
  url    = {https://github.com/jingyaogong/minimind},
  year   = {2024}
}
```

## 许可证

Apache License 2.0
