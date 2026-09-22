# 对比实验：数字逐位切分（Digits + ByteLevel）对算术能力的影响。
#
# 两个模型除 tokenizer 外完全一致——同样的题目、同样的参数量、同样的训练步数、同样的随机种子，
# 唯一变量是 pre_tokenizer。测试集与训练集严格不重叠，评测按整数值判定是否答对。
#
# 用法：python eval_math_digits.py --task add --steps 3000
import os
import sys
import json
import time
import random
import shutil
import argparse
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
BYTE_LEVEL_ONLY = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
DIGITS = {"type": "Digits", "individual_digits": True}


# ----------------------------------------------------------------------------- tokenizer
def make_tokenizer_dir(base_dir, digits):
    """复制一份 model/ 并强制把 pre_tokenizer 设为指定形态，两组实验都从同一份词表出发。"""
    tmp = tempfile.mkdtemp(prefix='tk_math_')
    dst = os.path.join(tmp, 'model')
    shutil.copytree(base_dir, dst)
    path = os.path.join(dst, 'tokenizer.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['pre_tokenizer'] = (
        {"type": "Sequence", "pretokenizers": [dict(DIGITS), dict(BYTE_LEVEL_ONLY)]}
        if digits else dict(BYTE_LEVEL_ONLY)
    )
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return tmp, dst


# ----------------------------------------------------------------------------- 数据
def gen_problems(task, n, lo, hi, rng):
    """生成 (题面, 答案) 对。答案统一为十进制整数字符串。"""
    out = []
    while len(out) < n:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if task == 'add':
            q, ans = f'{a}+{b}=', a + b
        elif task == 'sub':
            a, b = max(a, b), min(a, b)  # 避免负号引入额外 token 变量
            q, ans = f'{a}-{b}=', a - b
        elif task == 'mul':
            q, ans = f'{a}*{b}=', a * b
        else:
            raise ValueError(task)
        out.append((q, str(ans)))
    return out


def split_problems(task, n_train, n_test, lo, hi, seed):
    """先抽测试集，再抽训练集并排除掉测试题面，保证零泄漏。"""
    rng = random.Random(seed)
    test = gen_problems(task, n_test, lo, hi, rng)
    test_q = {q for q, _ in test}
    train, seen = [], set()
    while len(train) < n_train:
        for q, a in gen_problems(task, n_train, lo, hi, rng):
            if q in test_q or q in seen:
                continue
            seen.add(q)
            train.append((q, a))
            if len(train) >= n_train:
                break
    return train, test


def encode_batch(tk, pairs, max_len, device):
    """拼成 '题面答案<eos>'，只对答案段计损失（题面部分置 -100）。"""
    eos = tk.convert_tokens_to_ids('<|im_end|>')
    pad = tk.convert_tokens_to_ids('<|endoftext|>')
    ids_list, lab_list = [], []
    for q, a in pairs:
        q_ids = tk.encode(q)
        a_ids = tk.encode(a) + [eos]
        ids = (q_ids + a_ids)[:max_len]
        labels = ([-100] * len(q_ids) + a_ids)[:max_len]
        n_pad = max_len - len(ids)
        ids_list.append(ids + [pad] * n_pad)
        lab_list.append(labels + [-100] * n_pad)
    return (torch.tensor(ids_list, device=device), torch.tensor(lab_list, device=device))


# ----------------------------------------------------------------------------- 训练 / 评测
def train(tk, train_pairs, cfg, steps, batch_size, max_len, lr, device, seed, log_every):
    torch.manual_seed(seed)
    model = MiniMindForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = random.Random(seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        batch = [train_pairs[rng.randrange(len(train_pairs))] for _ in range(batch_size)]
        ids, labels = encode_batch(tk, batch, max_len, device)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if log_every and (step + 1) % log_every == 0:
            print(f'      step {step + 1:5}/{steps}  loss {loss.item():.4f}  {time.time() - t0:5.0f}s', flush=True)
    return model, n_params


@torch.no_grad()
def evaluate(model, tk, test_pairs, max_new, device, batch_size=64):
    """贪心解码，逐 token 生成到 eos，按整数值判定对错。"""
    model.eval()
    eos = tk.convert_tokens_to_ids('<|im_end|>')
    correct, digit_hits, digit_total, records = 0, 0, 0, []
    for i in range(0, len(test_pairs), batch_size):
        chunk = test_pairs[i:i + batch_size]
        # 题面等长才能批量贪心解码，按长度分组
        by_len = {}
        for q, a in chunk:
            by_len.setdefault(len(tk.encode(q)), []).append((q, a))
        for group in by_len.values():
            ids = torch.tensor([tk.encode(q) for q, _ in group], device=device)
            out = ids.clone()
            done = torch.zeros(len(group), dtype=torch.bool, device=device)
            for _ in range(max_new):
                logits = model(input_ids=out).logits[:, -1, :]
                nxt = logits.argmax(-1)
                nxt[done] = eos
                out = torch.cat([out, nxt[:, None]], dim=1)
                done |= (nxt == eos)
                if done.all():
                    break
            for (q, gold), seq in zip(group, out):
                gen = tk.decode(seq[ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
                ok = gen == gold
                correct += ok
                # 逐位命中率：右对齐比较，衡量"接近程度"而非全对全错
                g, p = gold[::-1], gen[::-1]
                digit_total += len(g)
                digit_hits += sum(1 for k in range(len(g)) if k < len(p) and p[k] == g[k])
                records.append((q, gold, gen, ok))
    return correct / len(test_pairs), digit_hits / digit_total, records


def run_arm(name, digits, args, device):
    print(f'\n{"=" * 78}\n{name}\n{"=" * 78}')
    tmp, tk_dir = make_tokenizer_dir(MODEL_DIR, digits)
    try:
        tk = AutoTokenizer.from_pretrained(tk_dir)
        train_pairs, test_pairs = split_problems(
            args.task, args.n_train, args.n_test, args.lo, args.hi, args.data_seed)

        sample_q, sample_a = test_pairs[0]
        print(f'  切分示例: {sample_q}{sample_a}')
        print(f'      题面 {[tk.convert_ids_to_tokens(i) for i in tk.encode(sample_q)]}')
        print(f'      答案 {[tk.convert_ids_to_tokens(i) for i in tk.encode(sample_a)]}')
        seq_lens = [len(tk.encode(q + a)) + 1 for q, a in train_pairs[:500]]
        print(f'  平均序列长度: {sum(seq_lens) / len(seq_lens):.1f} tokens')

        cfg = MiniMindConfig(
            hidden_size=args.hidden_size, num_hidden_layers=args.num_layers,
            num_attention_heads=4, num_key_value_heads=2, flash_attn=False,
            vocab_size=len(tk.get_vocab()),
        )
        print(f'  训练中（{args.steps} steps, batch {args.batch_size}）...')
        model, n_params = train(tk, train_pairs, cfg, args.steps, args.batch_size,
                                args.max_len, args.lr, device, args.train_seed, args.log_every)
        print(f'  参数量: {n_params / 1e6:.2f}M')
        acc, digit_acc, records = evaluate(model, tk, test_pairs, args.max_new, device)
        print(f'  ▶ 完全正确率: {acc:.1%}   逐位命中率: {digit_acc:.1%}')
        print('  样例:')
        for q, gold, gen, ok in records[:8]:
            print(f'      {"✓" if ok else "✗"} {q}{gen}' + ('' if ok else f'   (正确答案 {gold})'))
        return acc, digit_acc, records
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='验证数字分词对算术能力的影响')
    p.add_argument('--task', default='add', choices=['add', 'sub', 'mul'])
    p.add_argument('--lo', type=int, default=0)
    p.add_argument('--hi', type=int, default=999)
    p.add_argument('--n_train', type=int, default=20000)
    p.add_argument('--n_test', type=int, default=500)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_len', type=int, default=24)
    p.add_argument('--max_new', type=int, default=10)
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--data_seed', type=int, default=1234)
    p.add_argument('--train_seed', type=int, default=42)
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available()
                   else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = p.parse_args()

    device = torch.device(args.device)
    print(f'任务: {args.task}  范围: [{args.lo}, {args.hi}]  设备: {device}')
    print(f'训练题 {args.n_train}  测试题 {args.n_test}（与训练集不重叠）')

    base_acc, base_digit, _ = run_arm('对照组：原始 ByteLevel（数字可合并为多位 token）', False, args, device)
    new_acc, new_digit, _ = run_arm('实验组：Digits + ByteLevel（数字逐位切分）', True, args, device)

    print(f'\n{"=" * 78}\n结论\n{"=" * 78}')
    print(f'{"":26}{"完全正确率":>12}{"逐位命中率":>12}')
    print(f'{"原始 ByteLevel":26}{base_acc:>11.1%}{base_digit:>13.1%}')
    print(f'{"Digits + ByteLevel":26}{new_acc:>11.1%}{new_digit:>13.1%}')
    delta = new_acc - base_acc
    print(f'{"差值":26}{delta:>+11.1%}{new_digit - base_digit:>+13.1%}')
