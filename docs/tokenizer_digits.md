# 数字分词修正（方案 A）

## 问题

`model/tokenizer.json` 原本的 `pre_tokenizer` 只有一个裸的 `ByteLevel`：

```json
{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
```

GPT-2 那套预切分正则里数字是 `\p{N}+`，没有长度上限，所以 BPE 可以自由地把数字合并成任意长度的块。6400 的词表里因此产生了 49 个纯数字 token：10 个单数字、34 个两位、4 个三位、1 个四位（`2023`）。

带来三个问题：

1. **切分不一致**。`1234567` 切成 `123|45|67`，只差末位的 `1234568` 切成 `123|45|6|8`，前缀切法都不同，模型无法把"某一数位"和固定 token 位置对齐。
2. **非右对齐**。数字的语义权重由右向左确定（个位、十位……），但 BPE 从左贪心合并。`100+250=350` 切成 `100|+|2|50|=|3|50`，位值信息被打碎。
3. **语料偏置进了词表**。`2023` 成为单独 token 纯粹因为训练语料里年份多，而 `1999` 要占两个 token。

结果是算术、数值比较、单位换算这类任务明显偏弱。

## 修正方式

关键点：**问题不在 vocab 里，而在 pre_tokenizer 的切分规则**，这两者可以拆开处理。

把 `pre_tokenizer` 包一层 `Sequence`，让 `Digits` 先把数字串拆成单字符：

```json
{"type": "Sequence", "pretokenizers": [
  {"type": "Digits", "individual_digits": true},
  {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
]}
```

`vocab` 和 `merges` 一个字节都不改。Digits 先切开后，BPE 只能在单字符片段内合并，而单字符没有可合并对象——`123`、`2023` 这些 merge 规则仍在词表里，只是永远触发不到，成为死 token。

这是 Llama / Mistral 的做法。另一条路是 Qwen / GPT-4 的「最多三位一组、从右向左」，但 `tokenizers` 库没有现成组件，需要自己写 `Split` 正则；逐位更简单，且对小模型更稳。

## 用法

```bash
cd scripts

python patch_tokenizer_digits.py --check     # 查看当前状态，不修改
python patch_tokenizer_digits.py             # 应用
python patch_tokenizer_digits.py --revert    # 回滚（字节级还原）

python eval_tokenizer_digits.py --compare    # 与未 patch 版本逐项对比
python eval_tokenizer_digits.py              # 只看当前行为
```

patch 与 revert 都是幂等的，重复执行会提示无需改动；如果 `pre_tokenizer` 被手工改成了别的结构，脚本会拒绝覆盖而不是猜测意图。

## 实测结果

对 `model/tokenizer.json` 应用后：

| 检查项 | 结果 |
|---|---|
| vocab 是否一致 | ✅ 完全相同，6400，id 逐个对应 |
| 特殊 token id | ✅ 不变（`<|endoftext|>`=0, `<|im_start|>`=1, `<|im_end|>`=2, `<think>`=25, `</think>`=26） |
| 无数字文本切分 | ✅ 逐 token 一致（中文 / 英文 / 代码 / 混合全部 SAME） |
| roundtrip | ✅ 全部用例 encode→decode 还原 |
| chat_template | ✅ 正常，roundtrip 通过 |
| 数字位值对齐 | ✅ `7` 在任何位置都是 id 58 |
| `git diff` 范围 | ✅ 451KB 文件里仅 `pre_tokenizer` 一处改动 |

数字切分变化：

```
1234567      1 2 3 4 5 6 7            (旧: 123|45|67)
1234568      1 2 3 4 5 6 8            (旧: 123|45|6|8   ← 前缀不一致已消除)
1999         1 9 9 9                  (旧: 19|99)
2023年        2 0 2 3 年                (旧: 2023|年)
100+250=350  1 0 0 + 2 5 0 = 3 5 0    (旧: 100|+|2|50|=|3|50)
```

token 长度影响：

| 场景 | before | after | 变化 |
|---|---|---|---|
| 纯中文（无数字） | 17 | 17 | +0.0% |
| 纯英文（无数字） | 16 | 16 | +0.0% |
| 数字密集 | 28 | 37 | +32.1% |
| 算术 | 28 | 40 | +42.9% |

无数字文本完全不受影响，压缩率不变；数字部分变长是换取位值一致性的必然代价。README 里「中文 1.5~1.7 字符/token、英文 4~5」的结论仍然成立。

## 改动的文件

| 文件 | 改动 |
|---|---|
| `model/tokenizer.json` | `pre_tokenizer` 换成 `Sequence[Digits, ByteLevel]`，vocab / merges 不变 |
| `trainer/train_tokenizer.py` | 未改动。若将来重训 tokenizer，需先把 pre_tokenizer 换成 `Sequence[Digits, ByteLevel]`，否则多位数字 token 会回来 |
| `scripts/patch_tokenizer_digits.py` | 新增，幂等的 patch / revert / check |
| `scripts/eval_tokenizer_digits.py` | 新增，验证脚本 |

无需改动的地方（已验证）：全部 10 处 `AutoTokenizer.from_pretrained` 最终都指向 `model/` 目录（`--load_from` 默认 `../model`），改配置文件即自动传播到训练、评测、WebUI 与 OpenAI API 服务；`scripts/convert_model.py` 走 `save_pretrained`，已验证 `Digits` 字段会被完整保留并能正确复载；`model/tokenizer_config.json` 不含 pre_tokenizer 相关键；`dataset/lm_dataset.py` 只调用 tokenizer，无硬编码。

验证过训练数据链路：`SFTDataset` 对 assistant 回答 `3750` 的监督 token 为 `['3','7','5','0','<|im_end|>','Ċ']`，逐位拆分已生效。

## ⚠️ 重要：权重仍需继续训练

**tokenizer 改完不等于事情做完了。**

token id 没有错位，所以旧 checkpoint 加载不会乱码，无数字文本的行为完全不受影响。但是：

- 那 39 个多位数字 token（`123`、`2023`、`99`……）的 embedding 在原先训练中吸走了大量数值语义，现在永远不会再被激活；
- 单数字 token 的 embedding 此前只在少数场景出现过，还不足以承载位值信息。

直接换上去推理，**数字相关能力会明显退化**。需要以旧权重为初始化，在含数字的语料上继续预训练一段，让单数字 embedding 重新学起来：

```bash
cd trainer
python train_pretrain.py --from_weight pretrain --data_path ../dataset/pretrain_t2t.jsonl
```

这比从头训练便宜得多，非数字能力也基本保留。

另外注意 `max_seq_len`（`train_pretrain.py` 默认 340，`train_full_sft.py` 默认 768）：数字变长会挤占上下文，如果语料数字密度高，同样长度装的内容会变少。这取决于实际语料，建议先跑一轮看 token 统计再决定是否调整。

## 与「不建议重训 tokenizer」的关系

README 里那句警告指的是**重训 BPE**——vocab 和 merges 重新拟合，token id 全部错位，旧权重和社区 checkpoint 全部失效。

本方案只换 pre_tokenizer，vocab 与 merges 逐字节不变、id 完全兼容，不属于那个警告覆盖的情况。代价只在权重需要继续训练，而不是生态不兼容。
