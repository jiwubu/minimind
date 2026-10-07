# MiniMind 原理 01：模型架构与预训练 —— 从两句中文到一次预训练

> 配套架构图：*The Structure of MiniMind (Dense Model)*（左：宏观流水线；中：Layer k；右：GQA 与 FFN 展开）
>
> 读完本文，你应该能对着架构图上的**每一个框**说出：它在代码里对应哪一行、输入输出张量是什么形状、以及它为什么存在。

![The Structure of MiniMind (Dense Model)](../images/LLM-structure.jpg)

**目录**

- [0. 基本设定](#0-基本设定)
- [1. 左列总览：一句话的完整旅程](#1-左列总览一句话的完整旅程)
- [2. ① Tokenizer Encoder：文本 → token id](#2--tokenizer-encoder文本--token-id)
- [3. ② Input Embedding：id → 向量](#3--input-embeddingid--向量)
- [4. RoPE 旋转位置编码表](#4-rope-旋转位置编码表预计算无参数)
- [5. ③ Transformer Layer ×8：Layer k 总览](#5--transformer-layer-8中列-layer-k-总览)
- [6. 右图 (a)：GQA 注意力逐框解析](#6-右图-agqa-注意力逐框解析)
- [7. 右图 (b)：FFN 逐框解析](#7-右图-bffn-逐框解析)
- [8. ④⑤⑥ 出口：RMSNorm → Linear → SoftMax → loss](#8--出口rmsnorm--linear--softmax--loss)
- [9. 维度速查表](#9-维度速查表背诵版) · [10. 参数量核算](#10-参数量核算对照-config) · [11. 全图检查清单](#11-全图检查清单) · [12. 术语表](#12-新手术语表)

> 推理阶段（KV Cache、左 Padding、采样等）见姊妹篇 [MiniMind原理02_推理阶段.md](./MiniMind原理02_推理阶段.md)。

---

## 0. 基本设定

本文以 MiniMind **Dense（非 MoE）默认配置**为例，两个配置来源：`model/model_minimind.py` 的 `MiniMindConfig` 与训练脚本的默认参数。

| 配置项 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 6400 | 词表大小（自训 tokenizer） |
| `hidden_size` | 768 | 隐藏层维度 d |
| `num_hidden_layers` | 8 | Transformer 层数（图中的 ×K，K=8） |
| `num_attention_heads` | 8 | Q 头数 |
| `num_key_value_heads` | 4 | KV 头数（GQA，每头被 2 个 Q 头共享） |
| `head_dim` | 96 | 每头维度（768 ÷ 8） |
| `intermediate_size` | 2432 | FFN 中间维度 |
| `max_seq_len` | 340 | 预训练截断/补齐长度 |
| `rope_theta` | 1e6 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm 防除零 |
| `dropout` | 0.0 | 预训练关闭 dropout |
| `tie_word_embeddings` | true | 输入嵌入与输出 lm_head 共享权重 |

贯穿全文的两个例句：

```
句子A：解释下雪花是怎么形成的
句子B：我爱蔚蓝的天空
```

---

## 1. 左列总览：一句话的完整旅程

下图以句子 B 为例。句子 A、B 一起组成一个 batch（batch_size=2），所以每个张量的第一维都是 2。

```
"我爱蔚蓝的天空"                                    ← 原始文本
      ↓ ① Tokenizer Encoder（分词）
[bos] 我 爱 蔚蓝 的 天空 [eos] + pad×333             ← token id 序列 [2, 340]
      ↓ ② Input Embedding（查表）
[2, 340, 768]                                        ← 每个词变成 768 维向量
      ↓ ③ Transformer Layer ×8（中列：GQA + FFN）
[2, 340, 768]                                        ← 逐层提炼上下文语义
      ↓ ④ RMSNorm（最终归一化）
[2, 340, 768]
      ↓ ⑤ Linear（lm_head 768→6400）
[2, 340, 6400]                                       ← 每个位置对全词表的打分
      ↓ ⑥ SoftMax + CrossEntropy(labels)
标量 loss                                             ← 与 labels 的错一位交叉熵
      ↓ ⑦ Tokenizer Decoder（仅推理时）
"我爱蔚蓝的天空 是..."                                ← 逐 token 生成
```

下面逐段展开。

---

## 2. ① Tokenizer Encoder：文本 → token id

**代码**：`dataset/lm_dataset.py` 的 `PretrainDataset.__getitem__`

```python
tokens = self.tokenizer(text, add_special_tokens=False, ...).input_ids
tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
```

以句子 B 为例（token 数量以实际 tokenizer 为准，这里示意为 5 个）：

```
文本:      "我爱蔚蓝的天空"
分词:      我 | 爱 | 蔚蓝 | 的 | 天空
加框:      [bos] 我 爱 蔚蓝 的 天空 [eos]
补齐到340: [bos] 我 爱 蔚蓝 的 天空 [eos] pad pad ... pad
```

同 batch 的句子 A（假设 11 个 token）也补齐到 340，最终：

```
input_ids : [2, 340]    int64 的 token id
labels    : [2, 340]    现在与 input_ids 相同，稍后 pad 位置会被改成 -100
```

**新手要点**：

- **token** 是模型世界的"字"，模型只认识 id（整数），不认识汉字
- **bos / eos**：开头/结尾标记，让模型学会"一句话从哪开始、到哪结束"
- **pad**：同 batch 内短句补齐用的占位符（为了拼成矩形张量做并行计算）
- 本项目 tokenizer 做过 **数字逐位切分** 的 patch（`Sequence[Digits, ByteLevel]`）：任何数字 0-9 都是一个独立 token，这是后续数学能力的基础（见 `docs/tokenizer_digits.md`）

---

## 3. ② Input Embedding：id → 向量

**代码**：`model/model_minimind.py:206` 与 `:219`

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)   # [6400, 768]
hidden_states = self.dropout(self.embed_tokens(input_ids))
```

**维度变化**：

```
input_ids [2, 340]  →（查表）→  hidden_states [2, 340, 768]
```

嵌入矩阵可以想象成一本 6400 行的"字典"，每行是一个 token 的 768 维初始语义向量。"雪"这个 id 查出的那一行，就是"雪"在训练开始时的初始表示——**这个向量是可学习的，预训练的大部分学习就发生在调整这张表和后面的权重上**。

> **实现细节**：`tie_word_embeddings=True` 时，这张表与最后输出层的 `lm_head` **共享同一份权重**（`model_minimind.py:247`），即"读入词的字典"和"输出词的字典"是同一本，省了约 4.9M 参数。

---

## 4. RoPE 旋转位置编码表（预计算，无参数）

**代码**：`model/model_minimind.py:62-78`（预计算）与 `:224`（取本批位置）

```python
freqs_cos, freqs_sin = precompute_freqs_cis(dim=96, end=32768, rope_base=1e6)
position_embeddings = (self.freqs_cos[start_pos:start_pos+340], self.freqs_sin[start_pos:start_pos+340])
```

- 按 `head_dim=96` 生成 48 个频率 `f(i) = 1e6^(-2i/96)`，与位置 0~32767 做外积，再拼接成 **`[32768, 96]`** 的 cos / sin 两张表
- 每个 batch 只取前 340 行：**`cos, sin 各 [340, 96]`**

**新手要点**：

- 没有位置编码时，注意力打分只看内容、不看位置——"蔚蓝"出现在第 3 位还是第 30 位，它与"天空"的打分完全一样。RoPE 的作用是给每个**位置**一个旋转角度
- RoPE **不作用于嵌入向量**，而是稍后在注意力内部只旋转 Q 和 K（V 不旋转）
- 旋转后两个位置的内积自然带上**相对位置差**：位置 4 的"雪"对位置 10 的"形成"的分数会包含 |10-4| 的相位信息

---

## 5. ③ Transformer Layer ×8：中列 Layer k 总览

**代码**：`model/model_minimind.py` 的 `MiniMindBlock`（183-199 行）

```
        x [2,340,768]
        │
        ├─────────────────────┐   残差捷径（图中虚线）
        ↓                     │
   [ RMSNorm → GQA ]          │   右图 (a)
        ↓                     │
        ⊕ ←───────────────────┘   第一个 ⊕：相加
        │
        ├─────────────────────┐   残差捷径（图中虚线）
        ↓                     │
   [ RMSNorm → FFN ]          │   右图 (b)
        ↓                     │
        ⊕ ←───────────────────┘   第二个 ⊕：相加
        ↓
   下一层 [2,340,768]
```

对应代码：

```python
residual = hidden_states
hidden_states = self.self_attn(self.input_layernorm(hidden_states), ...)
hidden_states += residual                                                  # 第一个 ⊕
hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))  # 第二个 ⊕
```

**新手要点**：

- 图中两条**虚线**就是残差连接：子层的输入直接加到子层的输出上。这样即使某个子层学得不好（输出≈0），信息也能无损通过，几十层堆叠才训得动
- RMSNorm 放在子层**之前**（Pre-Norm），而不是之后，这是现代 LLM（LLaMA/Qwen 等）的标准做法，训练更稳定
- 8 层结构完全相同但**权重各自独立**，逐层从"字面语义"提炼到"上下文语义"

下面把两个子层拆到最细。

---

## 6. 右图 (a)：GQA 注意力逐框解析

以句子 B 的视角走一遍，`L=340`（含 pad），`bsz=2`。

### 6.1 RMSNorm

**公式**：`y = x / sqrt(mean(x²) + ε) × weight`（只按最后一维归一化，`model_minimind.py:56-60`）

```
[2, 340, 768] → [2, 340, 768]    逐 token 把 768 维向量拉回稳定长度
```

**为什么需要**：多层累加后向量长度会漂移，归一化让每层输入分布稳定。

### 6.2 Linear ×3：投出 Q、K'、V'

**代码**：`model_minimind.py:100-103`

```python
self.q_proj = nn.Linear(768, 8*96)   # 768 → 768
self.k_proj = nn.Linear(768, 4*96)   # 768 → 384
self.v_proj = nn.Linear(768, 4*96)   # 768 → 384
```

```
Q = q_proj(x): [2,340,768] → [2,340,768]  → 分头 → [2, 340, 8, 96]   （8 个 Query 头）
K'= k_proj(x): [2,340,768] → [2,340,384]  → 分头 → [2, 340, 4, 96]   （4 个 Key 头）
V'= v_proj(x): [2,340,768] → [2,340,384]  → 分头 → [2, 340, 4, 96]   （4 个 Value 头）
```

**三个角色**（检索类比）：

- **Q（Query，查询）**：当前位置"我想找什么"
- **K（Key，键）**：每个位置"我有什么可以被找到"
- **V（Value，值）**：每个位置"被关注后我实际给出的内容"

Q 头 8 个、KV 头只有 4 个 —— 这就是 **GQA（Grouped-Query Attention）**：4 组 KV 被 8 个 Q 头共享（每组 2 个 Q 头共用一份 K/V）。好处：KV 参数和 KV cache 直接减半，几乎不损质量。

### 6.3 N 圈：QK 归一化

**代码**：`model_minimind.py:117`

```python
xq, xk = self.q_norm(xq), self.k_norm(xk)     # 对每头 96 维做 RMSNorm
```

维度不变。Q/K 的数值尺度稳定后，后面的 softmax 不容易被个别大值主导。

### 6.4 RoPE：只旋转 Q 和 K

**代码**：`model_minimind.py:80-84` 与 `:119`

```python
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)   # cos/sin [340,96] 在头维广播
```

```
Q: [2,340,8,96] →（按位置旋转）→ [2,340,8,96]
K':[2,340,4,96] →（按位置旋转）→ [2,340,4,96]
V'：不旋转！
```

**新手要点**：图中 RoPE 只画在 Q 和 K' 下面，V' 没有——因为位置信息只需要参与"打分"（Q·K），不需要参与"内容传递"（V）。

### 6.5 (R) 圈：repeat_kv 复制

**代码**：`model_minimind.py:86-89` 与 `:124`

```python
xq, xk, xv = (xq.transpose(1,2), repeat_kv(xk, self.n_rep).transpose(1,2), repeat_kv(xv, self.n_rep).transpose(1,2))
```

GQA 的 4 份 KV 要复制给 8 个 Q 头（`n_rep = 8/4 = 2`）：

```
K, V: [2,340,4,96] →（每头复制2份）→ [2,340,8,96]
再转置成头优先:  Q [2,8,340,96]   K [2,8,340,96]   V [2,8,340,96]
```

### 6.6 ⊗：注意力打分

**代码**：`model_minimind.py:128`

```python
scores = (xq @ xk.transpose(-2,-1)).float() / math.sqrt(self.head_dim)
```

```
[2,8,340,96] @ [2,8,96,340] / √96  →  scores [2, 8, 340, 340]
```

- 含义：每个位置的 Q 与所有位置的 K 做点积 = "我在多大程度上该关注它"
- **÷√96**：向量维度越高点积越大，除以 √维度防止 softmax 饱和（梯度消失）

### 6.7 mask：0/-inf 三角

**代码**：`model_minimind.py:129`

```python
scores[:, :, :, -seq_len:] += torch.full((340, 340), float("-inf")).triu(1)
```

把分数矩阵的**上三角**加 -inf（softmax 后变成 0）：

```
        我  爱  蔚蓝  的  天空
我    [ ✓   ✗   ✗   ✗   ✗ ]
爱    [ ✓   ✓   ✗   ✗   ✗ ]      ✓ = 可以看（分数保留）
蔚蓝  [ ✓   ✓   ✓   ✗   ✗ ]      ✗ = 未来（-inf → 概率0）
的    [ ✓   ✓   ✓   ✓   ✗ ]
天空  [ ✓   ✓   ✓   ✓   ✓ ]
```

**为什么**：预训练是"预测下一个词"，如果当前位置能看到右边，就是抄答案。

**pad 需不需要额外屏蔽？** 预训练不传 `attention_mask`，但这没问题：右 padding 让 pad 全部排在真实 token 之后，因果掩码本来就让真实 token 看不到后面的 pad。pad 自己虽然能看到前面的真实 token，但 pad 位置的 loss 已经标成 -100，不会影响训练。

### 6.8 SoftMax → ⊗ 加权求和

**代码**：`model_minimind.py:131`

```python
output = self.attn_dropout(F.softmax(scores, dim=-1)...) @ xv
```

```
softmax → [2,8,340,340]（每行 340 个权重，和为 1）
@ V     → [2,8,340,340] @ [2,8,340,96] → [2,8,340,96]
```

含义："天空"位置的输出 = 所有位置内容 V 按注意力权重加权混合。走到这里，"天空"的向量里已经融进了"我爱蔚蓝的"的信息。

### 6.9 Linear（o_proj）+ ⊕ 残差

**代码**：`model_minimind.py:132-133` 与 `:197`

```python
output = output.transpose(1,2).reshape(bsz, seq_len, -1)   # [2,8,340,96] → [2,340,768]（8头拼回）
output = self.resid_dropout(self.o_proj(output))           # 768 → 768
hidden_states += residual                                   # ⊕ 残差
```

多头的"意见"拼接后由 `o_proj` 综合翻译回 768 维，再加回主干。

**GQA 子层维度全链**：

```
[2,340,768] → Q/K/V → Q、K/V 各 [2,8,340,96]（K/V 经 GQA 复制） → scores [2,8,340,340] → @V [2,8,340,96] → 拼头 [2,340,768] → ⊕
```

---

## 7. 右图 (b)：FFN 逐框解析

**代码**：`model/model_minimind.py:136-146` 与 `:198`

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(768, 2432, bias=False)
        self.down_proj = nn.Linear(2432, 768, bias=False)
        self.up_proj   = nn.Linear(768, 2432, bias=False)
        self.act_fn = ACT2FN['silu']
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

逐框对应：

| 图中框 | 运算 | 维度变化 |
|---|---|---|
| RMSNorm | 逐 token 归一化 | `[2,340,768]` 不变 |
| **两个并列 Linear** | `gate_proj` 与 `up_proj` 同时从 x 投影 | 两路各 `[2,340,2432]` |
| **SiLU** | 只作用在 gate 路：`x·sigmoid(x)`，平滑版 ReLU | `[2,340,2432]` |
| **⊙** 逐元素乘 | gate 路（门控：决定"放行多少"）× up 路（内容） | `[2,340,2432]` |
| Linear | `down_proj` 投回 | `[2,340,768]` |
| **Dropout** | 图中有、代码无（`dropout=0.0` 且 FFN 未实现） | —— |
| **⊕** 残差 | `hidden_states + self.mlp(...)` | `[2,340,768]` |

这套 `Linear → SiLU ⊙ Linear → Linear` 结构叫 **SwiGLU**。

**新手要点——FFN 和注意力分工不同**：

- 注意力是"**跨位置**"通信：让"天空"看到"我爱蔚蓝的"
- FFN 是"**单位置**"加工：对每个位置独立做一次 768→2432→768 的非线性变换（约 2/3 的参数在这里），承担主要的"知识存储"
- 中间维度 2432 的来历：`ceil(768 × π / 64) × 64`（`model_minimind.py:26`），先放大 3 倍再压回

---

## 8. ④⑤⑥ 出口：RMSNorm → Linear → SoftMax → loss

### 8.1 最终 RMSNorm + Linear

```python
hidden_states = self.norm(hidden_states)          # [2,340,768]
logits = self.lm_head(hidden_states)              # [768→6400] → [2,340,6400]
```

`logits [2, 340, 6400]`：每个位置对词表 6400 个 token 的原始打分。

`lm_head` 即 `nn.Linear(768, 6400, bias=False)`（`model_minimind.py:246`），且与输入嵌入共享同一份 `[6400,768]` 权重（第 247 行）——正着用是查表的 embedding，反着用就是打分的 lm_head。

### 8.2 与 labels 算 loss：错一位的"猜下一词"

**代码**：`model/model_minimind.py` 的 `MiniMindForCausalLM.forward`（255-257 行）

```python
x, y = logits[..., :-1, :], labels[..., 1:]
loss = F.cross_entropy(x.view(-1, 6400), y.view(-1), ignore_index=-100)
```

**错一位**是理解预训练的钥匙——位置 i 的打分要预测位置 i+1 的 token：

```
输入:   [bos]   我    爱   蔚蓝   的    天空   [eos]   pad ...
预测:    我     爱   蔚蓝   的    天空  [eos]    ？    ...
```

- `x = logits[:, :-1, :]` → `[2, 339, 6400]`（最后一个位置没有"下一个"，丢掉了最后一个）
- `y = labels[:, 1:]` → `[2, 339]`（丢掉了第一个）
- `cross_entropy` 内部四步：softmax 把 6400 个打分变概率 → 用答案 id 当索引取出"正确 token 的概率" → 取 −log → 平均（`ignore_index=-100` 的 pad 位置跳过）。本 batch 句子 B 共 7 个真实 token（bos + 5 个词 + eos），错一位后剩 6 个有效预测（我、爱、蔚蓝、的、天空、eos）；句子 A 共 13 个真实 token，剩 12 个。loss 是这 18 个 −log 概率的平均

**SoftMax / Tokenizer Decoder 框的说明**：图顶部的 SoftMax → Decoder → "hello + world" 是**推理视角**（取概率最大的 token，解码回文字，再拼回输入循环生成）。预训练阶段走到 loss 就结束了，不真的解码；但两者共享同一条前向路径。

### 8.3 loss 之后发生了什么（训练循环一瞥）

`trainer/train_pretrain.py` 的 `train_epoch`：

```python
loss = res.loss / args.accumulation_steps   # 梯度累积：小 batch 凑大有效 batch
loss.backward()                             # 反向传播：算出所有参数的梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪防爆炸
scaler.step(optimizer); optimizer.zero_grad()             # AdamW 更新参数
```

学习率由 `get_lr` 做 warmup + 余弦衰减。前向（本文 §2~8.2）→ 反向 → 更新，重复成千上万次 step，这就是预训练。

---

## 9. 维度速查表（背诵版）

| 阶段 | 张量 | 形状 |
|---|---|---|
| Tokenizer | input_ids / labels | `[2, 340]` |
| Input Embedding | hidden_states | `[2, 340, 768]` |
| RoPE 表 | cos / sin | `[340, 96]` |
| Q 投影+分头 | xq | `[2,340,8,96]` → 转置 `[2,8,340,96]` |
| K/V 投影+分头（GQA 复制×2） | xk, xv | `[2,340,4,96]` → `[2,340,8,96]` → `[2,8,340,96]` |
| 注意力分数 | scores | `[2, 8, 340, 340]` |
| 注意力输出 | output | `[2,8,340,96]` → 拼头 `[2,340,768]` |
| FFN 中间层 | gate/up | `[2, 340, 2432]` |
| 最终 logits | — | `[2, 340, 6400]` |
| loss | — | 标量（本 batch 共 18 个有效位置） |

## 10. 参数量核算（对照 config）

| 模块 | 每层参数 | 8 层合计 |
|---|---|---|
| QKV + O | 768×768 + 2×768×384 + 768×768 ≈ 1.77M | 14.2M |
| FFN（SwiGLU） | 3 × 768×2432 ≈ 5.60M | 44.8M |
| RMSNorm ×2 | 2×768（可忽略） | — |
| **嵌入表（与 lm_head 共享）** | 6400×768 ≈ 4.92M | 4.92M |
| **合计** | | **≈ 63.9M** ✅ |

---

## 11. 全图检查清单

| 架构图元素 | 对应代码 | 本文章节 |
|---|---|---|
| Tokenizer Encoder | `PretrainDataset.__getitem__` | §2 |
| Input Embedding | `embed_tokens` | §3 |
| Transformer Layer ×K（红圈） | `MiniMindBlock × 8` | §5 |
| Layer k 双残差 ⊕（虚线） | `MiniMindBlock.forward` 两次 `+=` | §5 |
| GQA: RMSNorm | `input_layernorm` | §6.1 |
| GQA: Linear（Q/K'/V' 三路） | `q_proj/k_proj/v_proj` | §6.2 |
| GQA: N 圈 | `q_norm/k_norm` | §6.3 |
| GQA: RoPE（仅 Q、K） | `apply_rotary_pos_emb` | §6.4 |
| GQA: (R) 圈 | `repeat_kv(n_rep=2)` | §6.5 |
| GQA: ⊗ 打分 | `xq @ xk^T / √96` | §6.6 |
| GQA: mask（0/-inf 三角） | 因果掩码 `triu(1)` | §6.7 |
| GQA: SoftMax → ⊗ | `softmax @ xv` | §6.8 |
| GQA: Linear + ⊕ | `o_proj` + 残差 | §6.9 |
| FFN: RMSNorm | `post_attention_layernorm` | §7 |
| FFN: 双 Linear + SiLU + ⊙ | `gate_proj/up_proj/act_fn` | §7 |
| FFN: Linear（down） | `down_proj` | §7 |
| FFN: Dropout | 图中预留，代码未实现（dropout=0.0） | §7 |
| 顶部 RMSNorm | `model.norm` | §8.1 |
| 顶部 Linear | `lm_head`（与嵌入共享） | §8.1 |
| SoftMax + Decoder | 推理路径；预训练中在 `cross_entropy` 内部 | §8.2 |
| hello + world | 下一 token 预测 / 推理时解码 | §8.2 |

## 12. 新手术语表

| 术语 | 一句话解释 |
|---|---|
| token | 模型处理文本的最小单位，对应一个整数 id |
| embedding | id 到语义向量的查表 |
| logits | softmax 之前的原始打分，形状 [..., vocab_size] |
| 头（head） | 把 768 维切成 8 份各自独立做注意力，让模型从 8 个不同角度关注信息 |
| GQA | KV 头比 Q 头少，多组 Q 共享一份 KV，省参数和缓存 |
| RoPE | 给 Q/K 按位置旋转，让点积带上相对位置信息 |
| 残差（⊕） | 子层输出加回子层输入，信息高速公路 |
| RMSNorm | 按向量长度归一化，稳定每层输入分布 |
| SwiGLU | FFN 的门控结构：SiLU(gate) ⊙ up 再压回 |
| 因果掩码 | 只许看过去、不许看未来的三角 mask |
| cross entropy | −log(正确 token 的概率)，预训练唯一的监督信号 |
| -100 | PyTorch 约定的"此位置不计 loss"标记 |
