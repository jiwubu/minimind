"""math_format v6 的 fuzz：轨迹生成 → 解析 roundtrip + 格式不变量。

沿用 v5 的验证方法（加减法能力提升方案.md §9）：每个采样分支 × 1–12 位
× 加减各 2000 例，外加手工难例清单。任何格式改动后必须先过本脚本。

用法:
    python math/fuzz_math_format.py              # 全量
    python math/fuzz_math_format.py --per 200    # 快速冒烟
"""

import os
import re
import sys
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math_format as mf  # noqa: E402
from gen_math_data import (  # noqa: E402
    sample_pair, sample_close_pair, sample_repdigit_pair, sample_equal_pair,
    sample_leadzero_pair, sample_zerorun_pair, sample_longrun_answer_pair,
    sample_negative_pair, sample_context_pair, make_sample)

_ANS_RE = re.compile(r'答案\((\d+)位\): (-?\d+)')


def check_trace(a, op, b):
    gt = a + b if op == '+' else a - b
    text = mf.build_trace(a, op, b)

    # 1. roundtrip：解析结果必须等于真值
    pred = mf.parse_answer(text)
    assert pred == gt, f'roundtrip 失败: {a}{op}{b}={gt}, parse={pred}\n{text}'

    # 2. 不变量：数字一律紧凑，不允许出现"数字 空格 数字"（v5 格式残留）
    assert not re.search(r'\d \d', text), f'发现空格分隔数字:\n{text}'

    # 3. 位数锚点与实际位数一致
    for m in _ANS_RE.finditer(text):
        assert int(m.group(1)) == len(m.group(2).lstrip('-')), \
            f'锚点错位: {m.group(0)} in\n{text}'

    if op == '-':
        # 4. 交换状态行：交换与否都必须显式声明
        if a < b:
            assert f'{a} < {b}，交换后计算 {b} - {a}' in text, f'缺交换行:\n{text}'
            assert text.rstrip().endswith(f'结果为负: {gt}'), f'缺结果为负行:\n{text}'
        else:
            rel = '>' if a > b else '='
            assert f'{a} {rel} {b}，不交换' in text, f'缺不交换行:\n{text}'

    # 5. 逆序行与答案行互为反转（低位在前 ↔ 高位在前）
    rev = re.search(r'逆序: (\d+)', text).group(1)
    ans = _ANS_RE.search(text).group(2).lstrip('-')
    assert rev == ans[::-1], f'逆序/答案不互为反转:\n{text}'
    return text


def fuzz_branches(per, max_digits):
    branches = [
        ('sample_pair', lambda d: (lambda p: (p[0], p[1], random.choice('+-')))(sample_pair(d))),
        ('sample_close_pair', lambda d: (lambda p: (p[0], p[1], random.choice('+-')))(sample_close_pair(d))),
        ('sample_repdigit_pair', lambda d: (lambda p: (p[0], p[1], random.choice('+-')))(sample_repdigit_pair(d))),
        ('sample_equal_pair', lambda d: (lambda p: (p[0], p[1], '-'))(sample_equal_pair(d))),
        ('sample_leadzero_pair', lambda d: (lambda p: (p[0], p[1], '-'))(sample_leadzero_pair(d))),
        ('sample_zerorun_pair', lambda d: (lambda p: (p[0], p[1], '-'))(sample_zerorun_pair(d))),
        ('sample_negative_pair', lambda d: (lambda p: (p[0], p[1], '-'))(sample_negative_pair(d))),
        ('sample_longrun_answer_pair', lambda d: sample_longrun_answer_pair(d)),
        ('sample_context_pair', lambda d: sample_context_pair(d)),
    ]
    n = 0
    for name, fn in branches:
        for _ in range(per):
            a, b, op = fn(max_digits)
            check_trace(a, op, b)
            n += 1
        for d in range(1, max_digits + 1):          # 每个分支 × 每个位数都要过
            for _ in range(per // 10 + 1):
                a, b, op = fn(d)
                check_trace(a, op, b)
                n += 1
        print(f'  {name:<28} ok')
    return n


def fuzz_make_sample(per, max_digits):
    for _ in range(per):
        row = make_sample(max_digits, neg_ratio=0.12)
        answer = row['conversations'][-1]['content']
        assert mf.parse_answer(answer) is not None, f'端到端解析失败:\n{answer}'
    print(f'  make_sample 端到端               ok')


def fuzz_parse_compat():
    """解析器对 v5 旧格式与新格式都要兼容"""
    assert mf.parse_answer('答案: - 2') == -2
    assert mf.parse_answer('答案: 6 8 5') == 685
    assert mf.parse_answer('答案(3位): 685') == 685
    assert mf.parse_answer('答案(1位): 2\n结果为负: -2') == -2, '结果为负必须优先'
    assert mf.parse_answer('答案(3位): 685\n所以 431 + 254 = 685') == 685
    assert mf.parse_answer('没有答案行 99+1=100') == 100
    assert mf.parse_answer('什么都提取不到') is None
    print('  parse 兼容性（v5/v6）            ok')


def fuzz_edges():
    cases = [
        (999, '+', 1), (1000, '-', 1), (25, '-', 83), (100, '-', 100),
        (1000000, '+', 7), (0, '-', 0), (1, '-', 1000000), (7777, '-', 7777),
        (44444444, '-', 444), (99, '-', 199), (77, '-', 177), (90, '-', 100),
        (44444, '-', 4444), (10 ** 12 - 1, '-', 10 ** 12 - 1),
        (0, '+', 0), (5, '-', 5), (999999999999, '+', 1),
    ]
    for a, op, b in cases:
        check_trace(a, op, b)
    print(f'  手工难例 {len(cases)} 条            ok')


def main():
    ap = argparse.ArgumentParser(description='math_format v6 fuzz')
    ap.add_argument('--per', type=int, default=2000, help='每分支每位数的用例数')
    ap.add_argument('--max_digits', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    assert args.max_digits <= mf.MAX_DIGITS
    random.seed(args.seed)
    print(f'fuzz: 每分支 {args.per} 例 × 位数 1..{args.max_digits}')
    fuzz_parse_compat()
    fuzz_edges()
    n = fuzz_branches(args.per, args.max_digits)
    fuzz_make_sample(min(args.per, 2000), args.max_digits)
    print(f'全部通过，共 {n} 例 roundtrip')


if __name__ == '__main__':
    main()
