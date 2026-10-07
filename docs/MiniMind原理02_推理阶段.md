# MiniMind 原理 02：推理阶段 —— 与预训练不同的那些事

> 姊妹篇：[MiniMind原理01_模型架构与预训练.md](./MiniMind原理01_模型架构与预训练.md) 已讲清前向计算本体（嵌入 → GQA → FFN → logits），本文**不重复**，只聚焦推理阶段与预训练**不一样**的地方：两阶段（Prefill/Decode）、左 Padding、KV Cache、只取最后一个位置、采样策略、停止条件。

**目录**

- [1. 总览：一次对话在推理时发生了什么](#1-总览一次对话在推理时发生了什么)
- [2. Prefill 与 Decode：推理的两个阶段](#2-prefill-与-decode推理的两个阶段)
- [3. KV Cache：推理提速的关键](#3-kv-cache推理提速的关键)
- [4. 只取最后一个位置 + 采样策略](#4-只取最后一个位置--采样策略)
- [5. 何时停止：EOS 与 finished 掩码](#5-何时停止eos-与-finished-掩码)
- [6. 训练 vs 推理 差异总对照](#6-训练-vs-推理-差异总对照)
- [7. 新手术语表（推理篇）](#7-新手术语表推理篇)

---

## 1. 总览：一次对话在推理时发生了什么

本文逐行引用的是 MiniMind 自己的 `generate`（`model/model_minimind.py:262-293`），入口是 `eval_llm.py`（默认 `--load_from model`，加载 `MiniMindForCausalLM`）：

```python
# eval_llm.py（节选）
inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
generated_ids = model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               max_new_tokens=args.max_new_tokens, do_sample=True,
                               top_p=args.top_p, temperature=args.temperature, ...)
```

> `scripts/minimind-math-v3/verify_model.py` 加载的是转换后的 Qwen3 格式模型，走 transformers 自带的 `generate`，实现不同但原理完全一样。

先记住一句话：**预训练是"一次前向算所有位置的 loss"，推理是"循环前向、每步只要最后一个位置"**。完整差异对照见 [§6](#6-训练-vs-推理-差异总对照)。

**用一个例子把全文串起来**——以句子 B `我爱蔚蓝的天空` 为输入。套上 chat template 后，prompt 一共 L 个 token（L 是实际长度，约二十个，推理时**不会**像预训练那样补齐到 340）：

| 步骤 | 阶段 | 这次前向的输入 | 输出 |
|---|---|---|---|
| 第 1 次前向 | Prefill | 完整 prompt，形状 `[1, L]` | 第一个新 token，如 `，` |
| 第 2 次前向 | Decode | 只送 `，`，形状 `[1, 1]` | `因为` |
| 第 3 次前向 | Decode | 只送 `因为`，形状 `[1, 1]` | `空气` |
| …… | Decode | 只送上一步生成的 token | 逐 token 续写 |
| 最后一步 | Decode | …… | `<|im_end|>` → 触发停止 |

后文各节分别解释这张表的细节：

- batch 时输入怎么对齐：§2.1 左 Padding
- 为什么 Decode 每步只送 1 个 token：§3 KV Cache
- 每步的 token 怎么选出来：§4 采样
- 什么时候停：§5 EOS

---

## 2. Prefill 与 Decode：推理的两个阶段

推理不是"一口气生成"，而是两个性质完全不同的阶段。

### 2.1 Prefill（预填充）：并行吃下整个 prompt

第一次前向，把完整的 prompt（L 个 token）**一次性并行**送入：

```
输入:   [B, L]
注意力: scores [B, 8, L, L]
输出:   logits [B, L, 6400]，只取 [:, -1]（最后一个位置）出第一个新 token
副产物: 每层缓存 K、V，各 [B, L, 4, 96]，共 8 层
```

- 走 flash 路径：`model_minimind.py:125-126` 的 `F.scaled_dot_product_attention`（条件之一是 `seq_len > 1`）
- 特点：**计算密集**——L 个位置的大矩阵乘并行做满，GPU 利用率高

**batch 推理的左 Padding**：Prefill 取的是最后一个位置的 logits `logits[:, -1]`。batch 内 prompt 长度不一，必须 pad 才能拼成矩形张量——如果 pad 加在右边，`[:, -1]` 取到的就是 pad 的打分：

```
batch 内两条 prompt，右 padding（错误）:
  句子A: [解释 下 雪花 是 怎么 形成 的 pad]   ← [:, -1] 取到 pad ✗
  句子B: [我 爱 蔚蓝 的 天空 pad pad pad]     ← 同样错位 ✗

左 padding（正确）:
  句子A: [pad 解释 下 雪花 是 怎么 形成 的]   ← 真实 token 顶到最后 ✅
  句子B: [pad pad pad 我 爱 蔚蓝 的 天空]     ← ✅
```

项目里的配套要求：

- batch rollout 时显式左 padding：`train_grpo.py:74` 的 `padding_side="left"`
- 同时传 `attention_mask`，把 pad 的分数压到 -inf（`model_minimind.py:130`）。左 padding 时 pad 排在真实 token **前面**，因果掩码挡不住它，必须靠 mask 屏蔽
- flash 快路径有个保护（`model_minimind.py:125`）：整个 batch 无 padding（mask 全 1）才走，否则退回手工路径——这就是带 padding 的 batch 推理稍慢的原因
- 对照预训练：右 padding、不传 mask。pad 都排在真实 token 后面，因果掩码天然就让真实 token 看不到它们，pad 位置的 loss 再标 -100 即可（姊妹篇 §6.7）

### 2.2 Decode（解码）：一次吐一个 token

之后每生成一个 token，前向的序列长度只有 **1**：

```
输入:   [B, 1]（只有最新的那个 token）
注意力: scores [B, 8, 1, 已有长度]   ← 只需新 token 的 Q，K/V 来自缓存
输出:   logits [B, 1, 6400] → 选出下一个 token
```

- 走手工路径：`model_minimind.py:128-131`（`seq_len == 1` 不满足 flash 条件）
- 特点：**访存密集**——每步计算量很小，但要读取全部已缓存的 K/V，瓶颈在显存带宽而不是算力

> 两个阶段瓶颈不同，大规模部署时有"PD 分离"（Prefill 和 Decode 拆到不同机器）这类工程手段，属于推理框架的范畴，与本项目无关。

---

## 3. KV Cache：推理提速的关键

### 3.1 为什么需要

没有缓存时，生成第 501 个 token 要把前 500 个 token 整个重新前向一遍——每生成一个字都要重算全部历史。而事实上，**旧 token 的 K/V 永远不变**（因果掩码下，它们只取决于自己之前的内容），完全没必要重算。

### 3.2 怎么存

`Attention.forward`（`model_minimind.py:120-123`）：

```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)   # 旧 K 拼上新 K（dim=1 是序列维）
    xv = torch.cat([past_key_value[1], xv], dim=1)   # 旧 V 拼上新 V
past_kv = (xk, xv) if use_cache else None
```

每层缓存一对 `(K, V)`，随生成逐步变长：

```
Prefill 后:    每层 K、V 各 [B, L, 4, 96]       （L 个位置 × 4 个 KV 头 × 96 维）
生成 t 步后:   每层 K、V 各 [B, L+t, 4, 96]     （每步序列维 +1）
全模型:        8 层 × (K, V)
```

缓存的是 **RoPE 旋转之后、`repeat_kv` 复制之前**的 K/V，所以只有 4 个头而不是 8 个——这正是 GQA 的价值兑现处：**cache 直接减半**。

### 3.3 两个必须配套的细节

**① 每步只喂新 token**（`generate` 循环，`model_minimind.py:269-270`）：

```python
past_len = past_key_values[0][0].shape[1] if past_key_values else 0   # 缓存的序列长度
outputs = self.forward(input_ids[:, past_len:], ...)                  # 只送缓存之后的新 token
```

**② RoPE 位置接续，而不是从 0 重来**（`MiniMindModel.forward`，`model_minimind.py:218/224`）：

```python
start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
position_embeddings = (self.freqs_cos[start_pos:start_pos+seq_length], ...)
```

位置从 0 开始计数，prompt 占了位置 0 ~ L-1，所以 Decode 第一步的新 token 位置是 **L**，必须用 `freqs_cos[L]` 那一行的角度——如果从 0 开始，位置编码就全乱了。预训练里 `start_pos` 永远是 0，这个变量正是为推理准备的。

---

## 4. 只取最后一个位置 + 采样策略

预训练保留全部位置的 logits 算 loss（`[..., :-1, :]` 去掉最后一个），推理只要下一个词（`[:, -1, :]` 只取最后一个）：

```python
logits = outputs.logits[:, -1, :] / temperature          # [B, 6400]
```

拿到这行 6400 维打分后，`generate`（`model_minimind.py:272-283`）依次做五步：

```python
# ① 温度：除以 T，T < 1 拉尖分布（更确定），T > 1 压平（更随机）
logits = outputs.logits[:, -1, :] / temperature

# ② 重复惩罚：已出现过的 token 打分打折，抑制复读（repetition_penalty != 1 时生效）
seen = torch.unique(input_ids[i]); logits[i, seen] = ...

# ③ Top-K：只保留打分前 K 名，其余置 -inf
logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

# ④ Top-P：按概率从高到低累加，累计超过 p 之后的尾部置 -inf
mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p

# ⑤ 二选一：贪心取最大，或按剩余概率随机抽
next_token = torch.multinomial(softmax(logits), 1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
```

| 策略 | 效果 | 典型场景 |
|---|---|---|
| `do_sample=False`（argmax） | 贪心，可复现 | 数学评测、代码生成 |
| `temperature` | 调节分布尖锐度 | 对话常用 0.7~0.9 |
| `top_k` | 只留前 K 名（`generate` 默认 50） | 防长尾乱词 |
| `top_p` | 只留累计概率前 p 的部分（默认 0.85） | 比 top_k 更自适应 |

**迷你数值例子**：假设词表只有 6 个 token，"我爱蔚蓝的天空，因为"之后的打分是 `空气:5.0, 干净:4.2, 冰晶:3.8, 冬天:3.0, 水汽:0.5, 云:0.2`：

```
① temperature=0.7  → 分数 ÷0.7，差距拉大：空气 7.14, 干净 6.00, 冰晶 5.43, ...
② 重复惩罚          → 本例设为 1，不生效
③ top_k=3          → 只留 [空气, 干净, 冰晶]，其余置 -inf
④ top_p            → softmax 后三者概率约 0.67 / 0.21 / 0.12，累计 0.67 / 0.88 / 1.00
                      top_p=0.9 ：三个全留
                      top_p=0.85：累计超过 0.85 之后的"冰晶"被删，只剩 [空气, 干净]
⑤ do_sample=False  → argmax，永远选"空气"（可复现）
   do_sample=True   → 按剩余候选的归一化概率随机抽，每次可能不同
```

注意 top_k 和 top_p 都**只会删候选、不会加候选**，两者叠加时取更严的那个。如果什么都不加，softmax 是对全部 6400 个 token 算的——长尾里的怪词也有小概率被抽中，这就是采样截断存在的意义。

---

## 5. 何时停止：EOS 与 finished 掩码

`model_minimind.py:284-290`：

```python
next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((..., 1), eos_token_id), next_token)
...
finished |= next_token.squeeze(-1).eq(eos_token_id)
if finished.all(): break
```

- 某条序列生成出 `<|im_end|>`（eos）后，它在 `finished` 里的标记置为 True
- **已结束的序列继续被强制填 eos**（保持 batch 内形状对齐），其余序列继续生成
- 全部结束才退出循环——张量必须是矩形的，batch 里的序列不能各自提前退出，只能"装作还在生成"

**双序列例子**：batch=2，句子 A 问雪花怎么形成（回答长），句子 B 是"我爱蔚蓝的天空"（回答短）：

```
step 20: B 生成出 <|im_end|>          → finished = [False, True ]
step 21: B 被强制填 <|im_end|>（凑形状），A 正常生成
                                      → finished = [False, True ]
step 35: A 也生成 <|im_end|>          → finished = [True,  True ] → break
```

解码时 `skip_special_tokens=True` 会把这些 eos 去掉：B 的回答实际在第 20 步就结束了，后面的 eos 只是占位填充。单条输入（batch=1）时第一个 eos 就会退出，感知不到这个机制。

---

## 6. 训练 vs 推理 差异总对照

| 维度 | 预训练 | 推理 | 预训练细节见姊妹篇 |
|---|---|---|---|
| 输入格式 | 纯文本 + bos/eos，右 padding 补齐到 340 | chat template，不补齐；batch 时左 padding | §2 |
| 前向形状 | `[B, 340]` 一次算完 | Prefill `[B, L]` 一次 + Decode `[B, 1]` × N | §1 |
| K/V | 算完即弃 | 缓存复用（KV Cache，GQA 减半） | §6.2、§6.5 |
| RoPE 位置 | 恒从 0 开始 | Decode 时从 `start_pos` 接续 | §4 |
| logits 用法 | 全部位置 → loss | 仅 `[:, -1]` → 采样 | §8.2 |
| labels / -100 | 有 | 无 | §8.2 |
| attention_mask | 不传（右 padding + 因果掩码已足够） | batch 时必须传（左 padding 的 pad 在前面） | §6.7 |
| 随机性 | 无（loss 是确定的） | 贪心，或 temperature/top-k/top-p 采样 | — |
| 模式 | `model.train()` | `model.eval()` | — |
| 循环结构 | 按 step 循环数据 | 按 token 循环生成，EOS 终止 | §8.3 |

## 7. 新手术语表（推理篇）

| 术语 | 一句话解释 |
|---|---|
| Prefill | 一次性并行处理整个 prompt 的前向，产出第一个新 token 和全部 KV cache |
| Decode | 之后逐 token 的前向，每步序列长度只有 1 |
| KV Cache | 缓存每层历史 K/V，避免重复计算旧 token |
| 左 Padding | batch 生成时把 pad 放左边，让真实 token 对齐到最后一个位置 |
| start_pos | 已缓存的长度，决定新 token 用哪个位置的 RoPE 角度 |
| 温度 | 打分除以 T 调节分布尖锐度：越小越确定，越大越随机 |
| Top-K / Top-P | 采样前截断候选集：固定留 K 名 / 留累计概率前 p |
| 贪心解码 | 每步取 argmax，可复现，评测用 |
| repetition_penalty | 对已出现 token 的打分打折，抑制复读 |
| finished 掩码 | 记录 batch 中哪些序列已生成 eos |
