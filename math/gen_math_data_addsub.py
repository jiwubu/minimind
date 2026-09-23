"""生成「一位一token」加减法课程数据，输出 SFT 会话格式 jsonl。

与自然语料的分布不同，这里位数是**均匀采样**的：自然数据里 83% 的算式
集中在1~2位，模型因此只在记忆范围内正确。均匀采样确保高位数样本充足。

相对上一版的变化（配合 patched tokenizer，见 math_format.py 模块注释）：
- 数字一律紧凑十进制，模板不再区分带空格/紧凑两套（tokenizer 已保证
  位值对齐），只保留问法多样性；
- 新增 sample_leadzero_pair：结果需去掉 k 个前导零，k 均匀覆盖；
- sample_negative_pair 的等长档改为"共享高位前缀、末 k 位分出大小"，
  覆盖上一版负数档全部错例的形状；
- 新增自然语境数字题（约 5%）：数字嵌在真实语境里读题。

用法:
    python math/gen_math_data_addsub.py --out dataset/math_addsub_v1.jsonl --n 200000
    python math/gen_math_data_addsub.py --out dataset/math_addsub_tiny.jsonl --n 2000  # 本机冒烟
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_format import build_trace, MAX_DIGITS  # noqa: E402

# 同一道题的多种问法，避免模型只认识一种模板。
# 数字一律紧凑十进制；运算符两侧有无空格各留一部分，保持读题鲁棒性
# （评测器的 nospace/repdigit 档仍在测两种写法）。
TEMPLATES = [
    '{a} {op} {b} = ?',
    '{a} {op} {b} 等于多少？',
    '计算 {a} {op} {b}',
    '请计算 {a} {op} {b} 的结果',
    '{a} {op} {b}',
    '帮我算一下 {a} {op} {b}',
    '算式：{a} {op} {b}，结果是多少？',
    '{a}{op}{b}',
    '{a}{op}{b}=?',
    '计算{a}{op}{b}',
]

OP_WORD = {'+': ['加', '加上'], '-': ['减', '减去']}

WORD_TEMPLATES = [
    '{a} {w} {b} 等于多少？',
    '{a} {w} {b} 是多少？',
    '一个数是 {a}，{w} {b}，结果是多少？',
]

# 自然语境数字题（v6 新增）：数字嵌在真实语境里，答案仍是同一条竖式轨迹。
# 这是 patched tokenizer 才给得了的能力 —— 旧格式下自然文本里的数字
# （2023、1250）被 BPE 合成与数位无关的块，模型读题时位值就是乱的。
# 减法模板语义固定为"还剩多少"，操作数保证 a >= b，不引入负数语义。
CONTEXT_ADD = [
    '一辆车先行驶了 {a} 千米，又行驶了 {b} 千米，一共行驶了多少千米？',
    '图书馆原有藏书 {a} 册，新购进 {b} 册，现在共有多少册？',
    '某地去年人口为 {a} 人，今年增加了 {b} 人，现在有多少人？',
]

CONTEXT_SUB = [
    '小明有 {a} 元，花掉 {b} 元，还剩多少元？',
    '一本书共 {a} 页，已经读了 {b} 页，还剩多少页？',
    '仓库里有 {a} 件货物，运走了 {b} 件，还剩多少件？',
]

SYSTEM_PROMPTS = [
    '你是一个擅长计算的AI助手，计算时请列竖式逐位演算。',
    '你是minimind，请按位计算加减法，给出完整步骤。',
]


def sample_operand(digits):
    """采样指定位数的非负整数（1位时允许0）"""
    if digits == 1:
        return random.randint(0, 9)
    return random.randint(10 ** (digits - 1), 10 ** digits - 1)


def sample_pair(max_digits):
    """按均匀位数采样一对操作数，并混入难例"""
    r = random.random()

    # 15% 难例：连续进位/借位（999+1、1000-1 这类）
    if r < 0.15:
        d = random.randint(min(2, max_digits), max_digits)
        if random.random() < 0.5:
            a = 10 ** d - random.randint(1, 9)          # 99..9 附近
            b = random.randint(1, 10 ** min(d, 3))
        else:
            # 10**d 有 d+1 位，取 min 才不会在 max_digits=MAX_DIGITS 时
            # 造出超过 UNITS 表长的操作数（--max_digits 12 会 IndexError）
            a = 10 ** min(d, max_digits - 1)             # 1000.. 整数
            b = random.randint(1, 10 ** min(d, 3))
        return (b, a) if random.random() < 0.5 else (a, b)

    # 10% 位数差异大（123456 + 7），强制模型处理不等长对齐
    if r < 0.25 and max_digits >= 3:
        # 位数差 >= 2 至少要 3 位；max_digits < 3 时本档无意义，落到后面的均匀采样
        d1 = random.randint(3, max_digits)
        d2 = random.randint(1, max(1, d1 - 2))
        a, b = sample_operand(d1), sample_operand(d2)
        # 短数在前（1 + 112）也要覆盖，否则模型只会给第二个操作数补零。
        # 减法在 make_sample 里按 neg_ratio 决定是否换回，故此处只影响加法。
        return (b, a) if random.random() < 0.5 else (a, b)

    # 8% 含0操作数
    if r < 0.33:
        d = random.randint(1, max_digits)
        a = sample_operand(d)
        return (0, a) if random.random() < 0.5 else (a, 0)

    # 其余：位数均匀，两数位数接近
    d1 = random.randint(1, max_digits)
    d2 = max(1, min(max_digits, d1 + random.randint(-1, 1)))
    return sample_operand(d1), sample_operand(d2)


def make_question(a, op, b):
    """把 (a, op, b) 套进一个随机问法模板。

    v5 曾把模板分成"带空格/紧凑"两套并按分支调配比例，那是针对旧
    tokenizer 数字合并缺陷的补丁；v6 下两种写法的数字切分完全一致，
    只保留问法多样性，不再需要 force_compact。
    """
    if random.random() < 0.25:
        w = random.choice(OP_WORD[op])
        return random.choice(WORD_TEMPLATES).format(a=a, b=b, w=w)
    return random.choice(TEMPLATES).format(a=a, b=b, op=op)


def sample_close_pair(max_digits):
    """采样两个非常接近的同位数操作数（b/a > 0.9）。

    符号判断的难度完全取决于两数接近程度：自然采样下 b/a>0.95 的等长减法
    只占 6%，模型因此学成"按首位猜大小"，实测该区间准确率仅 19%。
    """
    d = random.randint(min(2, max_digits), max_digits)
    lo, hi = 10 ** (d - 1), 10 ** d - 1
    a = random.randint(lo, hi)
    # 让 b 落在 a 附近，含 b>a、b==a、b<a 三种情况
    delta = random.randint(-max(1, a // 50), max(1, a // 50))
    b = min(hi, max(lo, a + delta))
    return a, b


def sample_repdigit_pair(max_digits):
    """采样至少一侧是全同数字（99、999、7777）的操作数对。

    旧 tokenizer 下全同数字在紧凑写法里会丢失位数信息（['99','9'] 与
    ['99','99'] 都是两个 token 且内容重复），v4 实测 '99+999' 被抄成
    竖式 '9 9 + 9 9 9 9'。patched tokenizer 下这道伤不存在了，但
    这一档**仍然保留**：上一版已证明长重复串数不清位数的公共根因是
    计数，不是 tokenizer —— 带空格写法、一位一 token 的 4~8 照样错。
    全同数字是最难的形状（444-222、333333+999999），必须显式构造。

    随机采样几乎不会生成全同数字，所以这类样本必须显式构造。
    两侧都全同最难，给 80% 权重。
    """
    def rep(d):
        return int(str(random.randint(1, 9)) * d)

    d1 = random.randint(1, max_digits)
    d2 = random.randint(1, max_digits)
    a = rep(d1)
    b = rep(d2) if random.random() < 0.8 else sample_operand(d2)
    return (b, a) if random.random() < 0.5 else (a, b)


def sample_equal_pair(max_digits):
    """采样两个完全相等的操作数，使 a - b 恰好为 0。

    结果为 0 是"去掉前导零"的极端情形（k = d，全部塌缩）—— 去 1~2 个
    前导零的普通情形模型做得很好（实测 98.5%/100%），但全塌缩这一档
    自然采样只占减法样本的 0.75%，v4 实测 7777-7777 输出
    "逆序: 0000 / 答案: 70"，把 0 的个数和首位都写错了。
    中间深度的 k 由 sample_leadzero_pair 覆盖。
    """
    d = random.randint(1, max_digits)
    # 一半用全同数字（7777-7777 这类，长重复串与位数塌缩叠加最难）
    if random.random() < 0.5:
        a = int(str(random.randint(1, 9)) * d)
    else:
        a = sample_operand(d)
    return a, a


def sample_leadzero_pair(max_digits):
    """采样结果需去掉 k 个前导零的减法对（本方案新增）。

    旧采样只覆盖两个极端：k=1~2 靠自然命中（98%+），k=d 全塌缩靠
    sample_equal_pair 专门构造，中间的 k=3~6 没有任何分支 —— v5 实测
    `14691927 - 14691816 = 111` 答成 0：8 位竖式跑完得 00000111，
    一次性去 5 个前导零时把 111 也一起吃掉了。

    构造：结果 r 恰好 d-k 位（首位非零），a = b + r 且 a 恰好 d 位，
    则竖式按 d 位演算、结果的帧里正好有 k 个前导零要去。
    """
    if max_digits < 2:
        a = random.randint(0, 9)
        return a, a
    d = random.randint(2, max_digits)
    k = random.randint(1, d - 1)                    # 要去掉的前导零个数
    r = random.randint(10 ** (d - k - 1), 10 ** (d - k) - 1)   # 恰好 d-k 位
    lo = max(1, 10 ** (d - 1) - r)                  # 保证 a = b + r 仍是 d 位
    hi = 10 ** d - 1 - r
    b = random.randint(lo, hi)
    return b + r, b


def sample_zerorun_pair(max_digits):
    """采样结果带一长串 0 的减法操作数对（`d…d000…0`）。

    v4 训完实测的失败模式，与 sample_equal_pair 的"全塌缩成 0"不同：

        2222 - 222   → 预测 0        （0 全丢了）
        66666 - 6666 → 预测 600060   （0 的个数数错）

    两例的竖式演算都是对的（逐位相减写 0），错在**逆序 → 答案**那一步
    要把长串 0 原样拷贝过去。模型倾向于"数不清写了几个"，这是长重复串
    拷贝的经典失败。

    构造方式：取 aa…a - aa…a（同一数字、位数不同），差是该数字重复
    `d1-d2` 次再跟 `d2` 个 0（`44444444-444 = 44444000`），两段都是长
    重复串。位数差偏向 1~2，因为实测失败的两例正是差 1 位（零串最长）。
    """
    # 本档需要 d1 > d2，至少 2 位；max_digits=1 时退化成 1-1（结果 0，仍合法）
    d1 = random.randint(min(2, max_digits), max_digits)
    # 偏向小位数差：差越小，结果里的 0 串越长，也就是实测失败的形状
    gap = min(d1 - 1, random.choice([1, 1, 2, random.randint(1, max(1, d1 - 1))]))
    d2 = d1 - gap
    digit = random.randint(1, 9)
    a = int(str(digit) * d1)
    b = int(str(digit) * d2)
    return a, b


def _longrun_number(max_digits, min_run=3):
    """构造一个含 >= min_run 位同数字连续段的整数（长串位置随机）。"""
    # max_digits < min_run 时放不下长串，退化成「整个数都是同一个数字」
    min_run = min(min_run, max_digits)
    k = random.randint(min_run, min(max_digits, 6))
    d = random.randint(k, max_digits)
    # 0 串占 40%（而非 1/10）：实测失败的 2222-222→0、66666-6666→600060
    # 都是零串，且 0 在答案里天然就比其它数字更容易连成长段
    digit = '0' if random.random() < 0.4 else str(random.randint(1, 9))

    if digit == '0':
        # 0 串不能顶在首位（会被当成前导零），给首位留一个非零位
        if d == k:
            d = min(max_digits, d + 1)
        if d == k:
            digit = str(random.randint(1, 9))   # max_digits 太小，改用非零串
            pos = 0
        else:
            pos = random.randint(1, d - k)
    else:
        pos = random.randint(0, d - k)

    chars = [str(random.randint(0, 9)) for _ in range(d)]
    chars[pos:pos + k] = [digit] * k
    if chars[0] == '0':
        chars[0] = str(random.randint(1, 9))    # pos >= 1，不会破坏长串
    return int(''.join(chars))


def sample_longrun_answer_pair(max_digits):
    """反向采样：先构造带长重复串的**答案**，再倒推操作数。返回 (a, b, op)。

    sample_zerorun_pair 只造 `aa…a - aa…a` 这一种形状 —— 操作数必须全同、
    只有减法、长串必然贴在答案末尾。但模型数不清的是**自己正在写的那一串**，
    跟操作数长什么样、串在第几位都无关。

    这里不限定操作数形状，直接指定答案 = 随机数字串里嵌一段 >= 3 位的
    重复串（位置随机、可以是 0 串也可以是非 0 串），再随机拆成 a + b
    或 a - b。于是 `1234000 + 5678`、`9999999 - 1` 这类操作数完全普通、
    答案却带长串的题也能覆盖到 —— 这正是 zerorun 档够不着的那部分。

    加法一并覆盖：`55555555+8888888` 答成 14444443（进位处抄错）说明
    写长串出错不分加减，而 zerorun 是 force_op='-' 的。
    """
    c = _longrun_number(max_digits)
    limit = 10 ** max_digits - 1

    if random.random() < 0.5:
        a = random.randint(0, c)
        return a, c - a, '+'          # a + b = c
    b = random.randint(0, max(0, limit - c))
    return c + b, b, '-'              # a - b = c


def _close_pair(d, k):
    """构造两个 d 位数，共享高位前缀、只在末 k 位分出大小，返回 a < b。"""
    head = str(random.randint(1, 9)) + ''.join(
        str(random.randint(0, 9)) for _ in range(d - k - 1))
    tail_a = random.randint(0, 10 ** k - 2)
    tail_b = random.randint(tail_a + 1, 10 ** k - 1)
    return int(head + str(tail_a).zfill(k)), int(head + str(tail_b).zfill(k))


def sample_negative_pair(max_digits):
    """采样必定得负数、且大小判断最容易出错的减法对（a < b）。

    两个实测来源决定了本档的形状配比：

    1. 共享后缀（b = 前缀 + a，199-99）：两个数字串共享后缀、只差最前面
       一位时最难区分。v4 实测模型在比较行把 99 的位数报成 3 位，走进
       "位数相同→逐位比较"分支，结论反了，整个丢掉负号。
    2. 等长且接近（本方案主修）：上一版负数档的全部错例（14-15、
       382-393、77583099-78524249）都是等长且接近的形状 —— 旧实现
       `a` 均匀取自 [lo, b-1]，绝大多数首位就分出大小，真正需要比到
       末几位的样本被稀释到 0.23%。

    形状配比：共享后缀 30% / 等长接近 55% / 一般不等长 15% ——
    等长接近按 neg_shape_ratio=5% 折算约 2.75% 标称；"差值比操作数短
    ≥2 位"的严格口径下实测 ≈2.1%（小位数样本稀释），达到本方案的
    ≥2% 目标。配比是按严格口径反推的，不要凭标称值估算。
    """
    # 负数至少要两位数参与（1 位时 b 也只能是 1 位，构不出共享后缀）
    if max_digits < 2:
        a = random.randint(0, 8)
        return a, random.randint(a + 1, 9)

    r = random.random()

    if r < 0.30:
        # b 在 a 前面接 1~2 位，构成共享后缀（99 -> 199 / 5099）
        da = random.randint(1, max_digits - 1)
        a = sample_operand(da)
        extra = 1 if random.random() < 0.8 else min(2, max_digits - da)
        lead = random.randint(1, 9)
        mid = ''.join(str(random.randint(0, 9)) for _ in range(extra - 1))
        b = int(f'{lead}{mid}{str(a).zfill(da)}')
        return a, b

    if r < 0.85:
        # 等长且接近：共享高位前缀，只在末 k 位分出大小
        d = random.randint(2, max_digits)
        k = random.randint(1, min(3, d - 1))
        return _close_pair(d, k)

    # 一般不等长
    db = random.randint(2, max_digits)
    da = random.randint(1, db - 1)
    a, b = sample_operand(da), sample_operand(db)
    if a >= b:
        a, b = b, a
    return a, b


def sample_context_pair(max_digits):
    """自然语境题的操作数（v6 新增）。减法保证 a >= b（"还剩"不应为负）。

    位数分别均匀采样、允许不等长：语境题的价值在"自然文本里的数字
    读题"，不在难度，位数差交给竖式的不等长路径处理即可。
    """
    def nonzero(d):
        return random.randint(1, 10 ** d - 1)

    op = random.choice('+-')
    if op == '+':
        return nonzero(random.randint(1, max_digits)), \
               nonzero(random.randint(1, max_digits)), op
    a, b = nonzero(random.randint(1, max_digits)), nonzero(random.randint(1, max_digits))
    return (a, b, op) if a >= b else (b, a, op)


def make_sample(max_digits, neg_ratio, close_ratio=0.12, rep_ratio=0.08,
                zero_ratio=0.03, zerorun_ratio=0.03, neg_shape_ratio=0.05,
                longrun_ratio=0.08, ctx_ratio=0.05):
    r = random.random()
    force_op = None
    force_neg = False
    ctx = False
    # 12% 的样本专门覆盖"两数接近"的难判区
    if r < close_ratio:
        a, b = sample_close_pair(max_digits)
    elif r < close_ratio + rep_ratio:
        a, b = sample_repdigit_pair(max_digits)
    elif r < close_ratio + rep_ratio + zero_ratio:
        # 前导零档对半分：k=d 全塌缩（sample_equal_pair）/ k=1..d-1 均匀
        # （sample_leadzero_pair，补中间深度的空档）
        if random.random() < 0.5:
            a, b = sample_equal_pair(max_digits)
        else:
            a, b = sample_leadzero_pair(max_digits)
        force_op = '-'          # 这一档只对减法有意义
    elif r < close_ratio + rep_ratio + zero_ratio + zerorun_ratio:
        a, b = sample_zerorun_pair(max_digits)
        force_op = '-'          # 差才是 d000…0，和则是 d…d，没有长零串
    elif r < close_ratio + rep_ratio + zero_ratio + zerorun_ratio + neg_shape_ratio:
        a, b = sample_negative_pair(max_digits)
        force_op = '-'
        force_neg = True        # 本档存在的意义就是负数，不能被 neg_ratio 换回去
    elif r < (close_ratio + rep_ratio + zero_ratio + zerorun_ratio
              + neg_shape_ratio + longrun_ratio):
        a, b, force_op = sample_longrun_answer_pair(max_digits)
    elif r < (close_ratio + rep_ratio + zero_ratio + zerorun_ratio
              + neg_shape_ratio + longrun_ratio + ctx_ratio):
        a, b, force_op = sample_context_pair(max_digits)
        ctx = True
    else:
        a, b = sample_pair(max_digits)
    op = force_op or random.choice('+-')

    if op == '-' and not force_neg:
        # 默认保证 a >= b，按比例放行负数结果。
        # 注意 neg_ratio 是**条件概率**（a<b 时才生效），且采样出 a<b 本身
        # 只占减法的 43.7% —— 名义 12% 实际稀释到 2.8%。真正需要保量的负数
        # 形状走 sample_negative_pair 专档，不依赖这里。
        if a < b and random.random() > neg_ratio:
            a, b = b, a

    if ctx:
        pool = CONTEXT_ADD if op == '+' else CONTEXT_SUB
        question = random.choice(pool).format(a=a, b=b)
    else:
        question = make_question(a, op, b)
    answer = build_trace(a, op, b)

    conv = []
    if random.random() < 0.15:
        conv.append({'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)})
    conv.append({'role': 'user', 'content': question})
    conv.append({'role': 'assistant', 'content': answer})
    return {'conversations': conv}


def main():
    ap = argparse.ArgumentParser(description='生成加减法课程数据')
    ap.add_argument('--out', required=True, help='输出 jsonl 路径')
    ap.add_argument('--n', type=int, default=400000, help='生成条数')
    ap.add_argument('--max_digits', type=int, default=8,
                    help=f'最大操作数位数（上限 {MAX_DIGITS}）')
    ap.add_argument('--neg_ratio', type=float, default=0.12,
                    help='减法中允许结果为负的比例')
    ap.add_argument('--mix_general', type=str, default='',
                    help='混入的通用SFT数据路径（防灾难性遗忘）')
    ap.add_argument('--mix_ratio', type=float, default=0.3,
                    help='通用数据占总量的比例')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.max_digits > MAX_DIGITS:
        sys.exit(f'--max_digits 不能超过 {MAX_DIGITS}')

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = [make_sample(args.max_digits, args.neg_ratio) for _ in range(args.n)]

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


def reservoir_sample(path, k):
    """单遍水库抽样，避免把上GB的通用数据整个读进内存"""
    if k <= 0:
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'conversations' not in obj:
                continue
            if len(out) < k:
                out.append(obj)
            else:
                j = random.randint(0, i)
                if j < k:
                    out[j] = obj
    return out


if __name__ == '__main__':
    main()
