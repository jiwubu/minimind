"""对生成的数学数据做插桩统计（方法论 #6：设了 X% 必须实测命中 X%）。

从 jsonl 反解出 (a, op, b) 与结果，统计场景覆盖率与 token 长度分布。
覆盖率必须与 gen_math_data.py 的分支标称比例对照，被模板、交换、条件
概率稀释掉的档位在这里现形。

用法:
    python math/stats_math_data.py dataset/math_addsub_v6.jsonl
    python math/stats_math_data.py dataset/math_addsub_v6.jsonl --token_sample 20000
"""

import os
import re
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_ADD_HEAD = re.compile(r'^竖式计算 (\d+) \+ (\d+)$')
_SUB_HEAD = re.compile(r'^先比较 (\d+) 和 (\d+) 的大小')
_MUL_HEAD = re.compile(r'^竖式计算 (\d+) \* (\d+)$')
_DIV_HEAD = re.compile(r'^竖式计算 (\d+) / (\d+)$')
_REMAINDER_LINE = re.compile(r'余数\s*[:：]\s*(\d+)')
# 自然语境题的标记词：只出现在 CONTEXT_* 模板里，普通算式模板不含
_CTX_MARK = ('元', '页', '千米', '册', '人口', '件', '箱', '瓶', '袋', '颗', '支')


def max_run(s, ch=None):
    """最长连续相同字符段；ch 指定时只统计该字符"""
    best = cur = 0
    for c in s:
        if ch is None or c == ch:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def parse_row(conv):
    """从会话里反解 (a, op, b, result, question, answer)。非数学行返回 None。

    除法的 result 取商（余数由调用方另解 `余数:` 行）。
    """
    q = next((m['content'] for m in conv if m['role'] == 'user'), '')
    ans = next((m['content'] for m in conv if m['role'] == 'assistant'), '')
    head = ans.splitlines()[0] if ans else ''
    for regex, op, calc in ((_ADD_HEAD, '+', lambda x, y: x + y),
                            (_MUL_HEAD, '*', lambda x, y: x * y),
                            (_DIV_HEAD, '/', lambda x, y: x // y)):
        m = regex.match(head)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return a, op, b, calc(a, b), q, ans
    m = _SUB_HEAD.match(ans)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a, '-', b, a - b, q, ans
    return None


def main():
    ap = argparse.ArgumentParser(description='数学数据插桩统计')
    ap.add_argument('path', help='生成的 jsonl')
    ap.add_argument('--token_sample', type=int, default=20000,
                    help='token 长度统计的抽样间隔取模（0=跳过）')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    total = math_n = 0
    op_cnt = {'+': 0, '-': 0, '*': 0, '/': 0}
    neg = neg_equal_close = 0
    ctx = compact = spaced = word = 0
    div_rem = div_corr = div_q0 = div_d1 = 0
    leadzero_hist = {}
    run3 = run4 = zero3 = zero4 = 0
    token_rows = []

    with open(args.path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                conv = json.loads(line).get('conversations')
            except json.JSONDecodeError:
                continue
            if not conv:
                continue
            parsed = parse_row(conv)
            if parsed is None:
                continue
            math_n += 1
            a, op, b, r, q, ans = parsed
            op_cnt[op] += 1

            if any(w in q for w in _CTX_MARK):
                ctx += 1
            elif re.search(rf'{a}[+\-*/]{b}', q):
                compact += 1
            elif re.search(rf'{a} [+\-*/] {b}', q):
                spaced += 1
            elif '加' in q or '减' in q or '乘' in q or '除以' in q:
                word += 1

            if op == '-':
                if r < 0:
                    neg += 1
                    # 等长且接近：差值比操作数短 >= 2 位
                    if len(str(a)) == len(str(b)) and len(str(b)) - len(str(-r)) >= 2:
                        neg_equal_close += 1
                # 前导零：竖式按操作数帧 d 位演算，结果 r 的帧里前 k 位是 0
                d = max(len(str(a)), len(str(b)))
                k = d - len(str(r)) if r > 0 else d - 1
                if k >= 1:
                    leadzero_hist[k] = leadzero_hist.get(k, 0) + 1

            if op == '/':
                # 除法专档：试商修正（轨迹里出现 `太大`）、带余数、商含 0、
                # 除数位数 —— gen_math_data_muldiv 的标称配比在这里对账
                if '太大' in ans:
                    div_corr += 1
                m = _REMAINDER_LINE.search(ans)
                if m and int(m.group(1)) > 0:
                    div_rem += 1
                if '0' in str(r):
                    div_q0 += 1
                if len(str(b)) == 1:
                    div_d1 += 1

            rs = str(abs(r))
            if max_run(rs) >= 3:
                run3 += 1
            if max_run(rs) >= 4:
                run4 += 1
            if max_run(rs, '0') >= 3:
                zero3 += 1
            if max_run(rs, '0') >= 4:
                zero4 += 1

            if args.token_sample and math_n % 7 == 0:
                token_rows.append(conv)

    print(f'总行数 {total}  数学行 {math_n}（其余为通用数据）')
    if not math_n:
        return
    print(f'\n运算符: 加 {op_cnt["+"]/math_n:.1%}  减 {op_cnt["-"]/math_n:.1%}  '
          f'乘 {op_cnt["*"]/math_n:.1%}  除 {op_cnt["/"]/math_n:.1%}')
    print(f'问法: 紧凑 {compact/math_n:.1%}  带空格 {spaced/math_n:.1%}  '
          f'口语 {word/math_n:.1%}  自然语境 {ctx/math_n:.1%}')
    if op_cnt['/']:
        ndiv = op_cnt['/']
        print(f'\n除法专档（{ndiv} 条）: 试商修正 {div_corr/ndiv:.1%}'
              f'  [标称 ~{0.3 * 0.94:.0%}+]  带余数 {div_rem/ndiv:.1%}'
              f'  商含0 {div_q0/ndiv:.1%}  除数1位 {div_d1/ndiv:.1%}')
    nsub = op_cnt['-'] or 1
    if op_cnt['-']:
        print(f'\n负数占减法: {neg/nsub:.1%}（占数学行 {neg/math_n:.1%}）')
        print(f'负数·等长且接近（差值比操作数短≥2位）: {neg_equal_close/math_n:.2%}'
              f'  [达标线 ≥2%]')
    lz_total = sum(leadzero_hist.values())
    if op_cnt['-']:
        print(f'\n需去前导零（减法帧）: {lz_total/nsub:.1%}（占减法）')
        for k in sorted(leadzero_hist):
            print(f'  k={k}: {leadzero_hist[k]/math_n:.2%}（占数学行）')
        lz3 = sum(v for k, v in leadzero_hist.items() if k >= 3)
        lz5 = sum(v for k, v in leadzero_hist.items() if k >= 5)
        print(f'  k>=3 合计 {lz3/math_n:.2%}（v5 基线 2.26%）  '
              f'k>=5 合计 {lz5/math_n:.2%}（v5 基线 0.83%）')
    print(f'\n答案最长重复串 ≥3: {run3/math_n:.1%}  ≥4: {run4/math_n:.1%}')
    print(f'答案最长零串   ≥3: {zero3/math_n:.1%}  ≥4: {zero4/math_n:.1%}'
          f'  [≥4 达标线 6%]')

    if args.token_sample and token_rows:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained('model')
        random.seed(args.seed)
        sample = random.sample(token_rows, min(args.token_sample, len(token_rows)))
        lens = []
        for conv in sample:
            text = tk.apply_chat_template(conv, tokenize=False)
            lens.append(len(tk.encode(text)))
        lens.sort()
        n = len(lens)
        print(f'\ntoken 长度（chat template 全样本，n={n}）:')
        print(f'  median={lens[n // 2]}  p95={lens[int(n * 0.95)]}  '
              f'max={lens[-1]}  超512占比 {sum(x > 512 for x in lens) / n:.2%}')


if __name__ == '__main__':
    main()
