"""生成「一位一token」乘除法课程数据（v1），输出 SFT 会话格式 jsonl。

沿用加减法 v6 已验证的方法论（gen_math_data.py 与数学专用模型训练方案 §6）：
每个演算行只做一次一位数决策，格式单点维护在 math_format.py。

乘法 = 部分积竖式（方案 §6 的设计）：乘数取位数少的操作数、逐位乘
（行格式与加法进位行同构）→ 错位补零显式写出 → 逐行累加（复用加法核心）。

除法 = 长除法 + 显式试商修正：每个商位先写候选积（两位除数展开成乘法行
同构的逐位块），偏大则 `试 c-1` 下调，再逐位减、落下一位。真商的一次性
猜测是除法唯一的高熵决策，overshoot_ratio 的样本让首个候选故意偏大 1~2，
教模型走自纠路径 —— 推理时猜偏也能收敛。4位÷2位 overshoot=2 的轨迹
max 1232 token，与 4×4 乘法（max 1263）共同决定 SFT 的 max_seq_len=1344
（p99.9 ≈ 1290 + 余量）；若训练预算紧张，--max_digits 3 可降回 896。

位数默认上限 4（乘 4×4、除 4位÷2位），小学四则的常见范围。tokenizer 下
* 与 / 是单 token；× ÷ 是多字节碎片，题面一律用 ASCII 符号。

用法:
    python math/gen_math_data_muldiv.py --out dataset/math_muldiv_v1.jsonl --n 400000
    python math/gen_math_data_muldiv.py --out dataset/math_muldiv_tiny.jsonl --n 2000  # 本机冒烟
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_format import mul_trace, div_trace, MAX_DIGITS  # noqa: E402
from gen_math_data import (  # noqa: E402
    TEMPLATES, WORD_TEMPLATES, sample_operand, reservoir_sample)

# 与加减法共用问法模板（{op} 槽位对 * / 同样成立），口语动词按运算符区分。
# 注意不用裸「除」：口语里「3 除 18」= 18÷3，语义反向，只保留无歧义的
# 「乘/乘以/除以」。
OP_WORD = {'*': ['乘', '乘以'], '/': ['除以']}

CONTEXT_MUL = [
    '每箱有 {b} 瓶饮料，{a} 箱一共有多少瓶？',
    '一支笔 {b} 元，买 {a} 支一共多少元？',
    '每页写 {b} 个字，{a} 页一共能写多少个字？',
]

# 除法语境题只出整除样本：「装多少袋」「每人分得多少」语义上不该带余数，
# 带余情形留给裸算式模板。
CONTEXT_DIV = [
    '{a} 个苹果，每 {b} 个装一袋，可以装多少袋？',
    '{a} 颗糖，平均分给 {b} 个小朋友，每人分得多少颗？',
    '一本书共 {a} 页，每天读 {b} 页，多少天能读完？',
]

SYSTEM_PROMPTS = [
    '你是一个擅长计算的AI助手，计算时请列竖式逐位演算。',
    '你是minimind，请按位计算乘除法，给出完整步骤。',
]


def make_question(a, op, b):
    """把 (a, op, b) 套进一个随机问法模板，风格与 gen_math_data 一致"""
    if random.random() < 0.25:
        w = random.choice(OP_WORD[op])
        return random.choice(WORD_TEMPLATES).format(a=a, b=b, w=w)
    return random.choice(TEMPLATES).format(a=a, b=b, op=op)


def nonzero_operand(digits):
    """采指定位数的非零整数。gen_math_data.sample_operand 在 1 位时允许 0，
    那是给加法用的；除数为 0 无定义，除数一律走这里"""
    if digits == 1:
        return random.randint(1, 9)
    return random.randint(10 ** (digits - 1), 10 ** digits - 1)


# ---------------------------------------------------------------------------
# 乘法采样：通用 + 七个难例档。随机采样几乎生成不出全同数字/9串/整十幂，
# 这些形状必须显式构造（加减法 v4~v6 的同一教训）。
# ---------------------------------------------------------------------------

def sample_mul_general(max_digits):
    d1 = random.randint(1, max_digits)
    # 20% 位数差 >= 2（3428 * 7 类），其余位数接近
    if random.random() < 0.2 and max_digits >= 3:
        d2 = random.randint(1, max(1, d1 - 2))
    else:
        d2 = max(1, min(max_digits, d1 + random.randint(-1, 1)))
    return sample_operand(d1), sample_operand(d2)


def sample_mul_onedigit(max_digits):
    """乘数一位（口诀及延伸 7 * 3428），小学最高频的乘法形态"""
    a = sample_operand(random.randint(1, max_digits))
    b = random.randint(1, 9)
    return (b, a) if random.random() < 0.5 else (a, b)


def sample_mul_tens(max_digits):
    """带末尾零的操作数（240 * 30）：乘法行对 0 数位照常演算，
    但错位补零行叠上操作数自带的零，结果的长零串是最难拷贝的形状"""
    if max_digits < 2:
        return sample_operand(1), sample_operand(1)
    k = random.randint(1, min(3, max_digits - 1))
    base = sample_operand(random.randint(1, max_digits - k)) * 10 ** k
    other = sample_operand(random.randint(1, max_digits))
    return (other, base) if random.random() < 0.5 else (base, other)


def sample_mul_zeroinside(max_digits):
    """数位内嵌 0（305 * 24）：0 数位行写 0，累加时对齐易错"""
    if max_digits < 2:
        return sample_operand(1), sample_operand(1)
    d = random.randint(2, max_digits)
    chars = [str(random.randint(0, 9)) for _ in range(d)]
    chars[0] = str(random.randint(1, 9))
    chars[random.randint(1, d - 1)] = '0'
    return int(''.join(chars)), sample_operand(random.randint(1, max_digits))


def sample_mul_power10(max_digits):
    """整十幂（1000 * 234）：部分积行大量为 0，考验 0 行的照写与累加"""
    if max_digits < 2:
        return sample_operand(1), sample_operand(1)
    k = random.randint(1, max_digits - 1)
    return 10 ** k, sample_operand(random.randint(1, max_digits))


def sample_mul_repdigit(max_digits):
    """全同数字（777 * 77）：加减法 §6.3 证明长重复串数不清位数是计数问题，
    与 tokenizer 无关，乘法的部分积行会更长，必须显式构造"""
    def rep(d):
        return int(str(random.randint(1, 9)) * d)

    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)
    a = rep(d1)
    b = rep(d2) if random.random() < 0.8 else sample_operand(d2)
    return a, b


def sample_mul_nines(max_digits):
    """9 串（999 * 99）：每位都进位，进位链是乘法行的主要错源"""
    d1 = random.randint(1, max_digits)
    d2 = random.choice([1, 2, d1])
    return 10 ** d1 - 1, 10 ** min(d2, max_digits) - 1


def sample_mul_equal(max_digits):
    """两操作数相等（4444 * 4444）：部分积各行相同，逐行累加步进最易抄错"""
    d = random.randint(1, max_digits)
    if random.random() < 0.5:
        a = int(str(random.randint(1, 9)) * d)
    else:
        a = sample_operand(d)
    return a, a


# ---------------------------------------------------------------------------
# 除法采样：先定除数与商，再反推被除数（a = b*q 或 b*q + r）。
# ---------------------------------------------------------------------------

def sample_div_general(max_digits, max_dd, exact_ratio=0.55):
    db = random.randint(1, max_dd)
    # 商位数上限保证被除数不超过 max_digits 位
    dq = random.randint(1, max(1, max_digits - db))
    b = nonzero_operand(db)
    q = sample_operand(dq)
    if random.random() < exact_ratio or b == 1:
        return b * q, b
    return b * q + random.randint(1, b - 1), b


def sample_div_zeroq(max_digits, max_dd):
    """商内嵌 0（624 / 6 = 104）：中间商 0 要在「余 < 除数」处显式商 0，
    是长除法最经典的错型，随机采样几乎采不到"""
    db = random.randint(1, max_dd)
    dq = random.randint(2, max(2, max_digits - db))
    chars = [str(random.randint(0, 9)) for _ in range(dq)]
    chars[0] = str(random.randint(1, 9))
    chars[random.randint(1, dq - 1)] = '0'
    q = int(''.join(chars))
    b = nonzero_operand(db)
    return b * q, b


def sample_div_zerorun(max_digits, max_dd):
    """被除数/商带长零串（4000 / 25）：商末尾的 0 靠「不够商」连续触发，
    与加减法 zerorun 档同源的拷贝难点"""
    db = random.randint(1, max_dd)
    if max_digits - db < 2:
        # 除数占了全部位数，放不下 "q0 至少 1 位 + k>=1 位零串"，退化成通用档
        return sample_div_general(max_digits, max_dd)
    k = random.randint(1, min(3, max_digits - db - 1))
    q0 = max(1, sample_operand(random.randint(1, max(1, max_digits - db - k))))
    b = nonzero_operand(db)
    a = b * q0 * 10 ** k
    if random.random() < 0.4 and b > 1:
        a += random.randint(1, b - 1)      # 余数 r < b 不破坏商的零串
    return a, b


def sample_div_small(max_digits, max_dd):
    """被除数小于除数（3 / 16）：商 0 余 a 的独立分支，主采样采不到"""
    b = nonzero_operand(random.randint(1, max_dd))
    return random.randint(0, b - 1), b


def make_sample(max_digits, max_dd, mul_ratio=0.5, overshoot_ratio=0.3,
                ctx_ratio=0.05):
    r = random.random()
    op = '*' if random.random() < mul_ratio else '/'
    ctx = r < ctx_ratio

    if ctx:
        # 语境题：除法只出整除样本（语境语义不带余数）
        if op == '/':
            db = random.randint(1, max_dd)
            b = nonzero_operand(db)
            q = sample_operand(random.randint(1, max(1, max_digits - db)))
            a = b * q
        else:
            a, b = sample_mul_general(max_digits)
    elif op == '*':
        r2 = random.random()
        if r2 < 0.45:
            a, b = sample_mul_general(max_digits)
        elif r2 < 0.60:
            a, b = sample_mul_onedigit(max_digits)
        elif r2 < 0.70:
            a, b = sample_mul_tens(max_digits)
        elif r2 < 0.78:
            a, b = sample_mul_zeroinside(max_digits)
        elif r2 < 0.82:
            a, b = sample_mul_power10(max_digits)
        elif r2 < 0.90:
            a, b = sample_mul_repdigit(max_digits)
        elif r2 < 0.96:
            a, b = sample_mul_nines(max_digits)
        else:
            a, b = sample_mul_equal(max_digits)
    else:
        r2 = random.random()
        if r2 < 0.78:
            a, b = sample_div_general(max_digits, max_dd)
        elif r2 < 0.90:
            a, b = sample_div_zeroq(max_digits, max_dd)
        elif r2 < 0.98:
            a, b = sample_div_zerorun(max_digits, max_dd)
        else:
            a, b = sample_div_small(max_digits, max_dd)

    if ctx:
        pool = CONTEXT_MUL if op == '*' else CONTEXT_DIV
        question = random.choice(pool).format(a=a, b=b)
    else:
        question = make_question(a, op, b)

    if op == '/':
        # 修正样本：首个试商候选故意偏大，教模型走 `太大 → 试 c-1` 自纠路径。
        # 全程 0 会导致推理时一猜偏就无路可走；偏大上限 2，再多则轨迹过长。
        if random.random() < overshoot_ratio:
            ov = 1 if random.random() < 0.85 else 2
        else:
            ov = 0
        answer = div_trace(a, b, overshoot=ov)
    else:
        answer = mul_trace(a, b)

    conv = []
    if random.random() < 0.15:
        conv.append({'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)})
    conv.append({'role': 'user', 'content': question})
    conv.append({'role': 'assistant', 'content': answer})
    return {'conversations': conv}


def main():
    ap = argparse.ArgumentParser(description='生成乘除法课程数据')
    ap.add_argument('--out', required=True, help='输出 jsonl 路径')
    ap.add_argument('--n', type=int, default=400000, help='生成条数')
    ap.add_argument('--max_digits', type=int, default=4,
                    help=f'操作数最大位数（乘法需 2*max_digits <= {MAX_DIGITS}）')
    ap.add_argument('--max_divisor_digits', type=int, default=2,
                    help='除法除数的最大位数（默认 2，小学范围）')
    ap.add_argument('--mul_ratio', type=float, default=0.5,
                    help='乘法样本占总数比例')
    ap.add_argument('--overshoot_ratio', type=float, default=0.3,
                    help='除法试商修正样本比例')
    ap.add_argument('--mix_general', type=str, default='',
                    help='混入的通用SFT数据路径（防灾难性遗忘）')
    ap.add_argument('--mix_ratio', type=float, default=0.3,
                    help='通用数据占总量的比例')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if 2 * args.max_digits > MAX_DIGITS:
        sys.exit(f'--max_digits 不能超过 {MAX_DIGITS // 2}（乘积位数帧上限）')
    if args.max_divisor_digits > args.max_digits:
        sys.exit('--max_divisor_digits 不能大于 --max_digits')

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = [make_sample(args.max_digits, args.max_divisor_digits,
                        args.mul_ratio, args.overshoot_ratio)
            for _ in range(args.n)]

    if args.mix_general:
        k = int(args.n * args.mix_ratio / max(1e-9, 1 - args.mix_ratio))
        picked = reservoir_sample(args.mix_general, k)
        print(f'混入通用数据 {len(picked)} 条（目标 {k}）')
        rows.extend(picked)

    random.shuffle(rows)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'已写入 {len(rows)} 条 -> {args.out}')


if __name__ == '__main__':
    main()
