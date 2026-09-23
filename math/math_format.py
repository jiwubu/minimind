"""加减法「一位一token」轨迹的生成与解析（v6）。

数据生成(gen_math_data_addsub.py / gen_math_data_muldiv.py)与评测(eval_math.py)共用本模块，保证二者格式完全一致
——上一版评测把正确输出判成错误，就是因为生成和解析各写了一套。

v6 相对 v5 的变化（配合 patched tokenizer，见 docs/tokenizer_digits.md）：
1. 数字一律紧凑十进制（431 就写 "431"）：Digits 预切分已保证一位一 token，
   v5 的空格分隔（"4 3 1"）退役 —— 它是"不能动 tokenizer"时代的 workaround；
2. 减法在比较之后增加显式交换状态行，交换与否都写明：v5 负数错例集中在
   模型跳过交换、直接对 小-大 跑借位得到 10 的补码（15-17 → 98）；
3. 答案行带位数锚点 `答案(3位): 685`，给反转拷贝一个显式终止条件：
   v5 实测"已写 k 个 0 要不要继续"在 k=2~4 几乎是掷硬币；
4. 负数符号由末行 `结果为负: -2` 单独声明，答案行保持纯 magnitude 拷贝
   —— 符号判断发生在交换状态行之后（位置早、有参照），不再压在
   高熵的答案行末尾。

乘除扩展（配合 gen_math_data_muldiv.py，设计详见 数学专用模型训练方案_v1.md §6）：
5. 乘法 = 部分积竖式：乘数取位数少的操作数，逐位乘（行格式与加法进位
   行同构：`十位: 3 * 4 + 1 = 13，写 3 进位 1`）→ 错位补零显式写出
   → 逐行累加（复用 _add_core）。中间累加行尾写 `合计:` 而非 `答案:`
   —— 一条轨迹只允许一个答案行，否则 parse_answer 命中的是中间结果。
6. 除法 = 长除法 + 显式试商：每个商位写候选积（一位除数是口诀一行，
   两位除数展开成与乘法行同构的逐位块），太大则 `试 c-1` 自纠，再
   逐位减、落下一位。真商的一次性猜测是除法唯一的高熵决策，靠数据侧
   的 overshoot 修正样本兜底（见 div_trace）。末行 `余数: X` 与
   `结果为负:` 同构 —— 余数不压进答案行，解析各走各的正则。
"""

import re

UNITS = ['个位', '十位', '百位', '千位', '万位', '十万位',
         '百万位', '千万位', '亿位', '十亿位', '百亿位', '千亿位']

MAX_DIGITS = len(UNITS)


def _low_digits(n):
    """返回低位在前的数字列表：431 -> [1, 3, 4]"""
    return [int(c) for c in reversed(str(n))]


def _strip_leading_zeros(low_first):
    """去掉高位多余的0（低位在前的列表里即末尾的0），至少保留一位"""
    out = list(low_first)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _low_to_int(low_first):
    """低位在前的数字列表 -> int"""
    return int(''.join(map(str, reversed(_strip_leading_zeros(low_first)))))


def _finish(lines, low_first):
    """统一收尾：逆序行 + 带位数锚点的答案行。

    曾经还有一行自然语言结论 `所以 a op b = result`，v4 实测把它删掉了：
    它对正确性零贡献（parse_answer 优先读"答案:"行），却要求模型把整个
    结果再正序抄一遍 —— 1850 题里 46 例出现"答案行≠结论行"，其中 39 例
    是答案行正确而结论行抄错。多一次拷贝就多一次出错机会，且位数越多越糟。

    v6 的答案行写作 `答案(3位): 685`：锚点是反转拷贝的显式终止条件。
    逆序行与答案行同为紧凑十进制 —— patched tokenizer 下 "685" 本来就是
    三个一位 token，空格只徒增 token 数。
    """
    stripped = _strip_leading_zeros(low_first)
    if len(stripped) != len(low_first):
        lines.append('去掉前导零')
    lines.append('逆序: ' + ''.join(map(str, stripped)))
    digits = ''.join(map(str, reversed(stripped)))
    lines.append(f'答案({len(stripped)}位): {digits}')
    return '\n'.join(lines)


def _add_core(a, b):
    """a+b 的逐位进位演算行（不含表头与收尾）。

    独立成核是为了乘法部分积的逐行累加能复用同一套行格式，同时不产出
    `答案:` 行 —— 中间步骤的合计行写 `合计:`，一条轨迹只有一个答案行。
    """
    da, db = _low_digits(a), _low_digits(b)
    lines, carry, out = [], 0, []
    for i in range(max(len(da), len(db))):
        x = da[i] if i < len(da) else 0
        y = db[i] if i < len(db) else 0
        s = x + y + carry
        expr = f'{x} + {y} + {carry} = {s}' if carry else f'{x} + {y} = {s}'
        lines.append(f'{UNITS[i]}: {expr}，写 {s % 10} 进位 {s // 10}')
        out.append(s % 10)
        carry = s // 10
    if carry:
        lines.append(f'最高位进位 {carry}')
        out.append(carry)
    return lines, out


def add_trace(a, b):
    """非负整数加法的分步轨迹"""
    lines = [f'竖式计算 {a} + {b}']
    core, out = _add_core(a, b)
    lines.extend(core)
    return _finish(lines, out)


def compare_trace(a, b):
    """逐位比较 a 和 b 的大小，返回演算行列表。

    符号判断必须有显式演算过程：若只写"a 小于 b"这一句结论，模型就得在
    单次前向里猜出谁大，实测 b/a>0.95 时准确率仅 19%（首位相同就掷硬币）。
    拆成"先比位数、再从高位逐位比"之后，每一步都只需比较两个一位数。
    """
    sa, sb = str(a), str(b)
    lines = [f'先比较 {a} 和 {b} 的大小']

    if len(sa) != len(sb):
        rel = '多' if len(sa) > len(sb) else '少'
        big = a if len(sa) > len(sb) else b
        lines.append(f'{a} 是 {len(sa)} 位，{b} 是 {len(sb)} 位，'
                     f'位数{rel}的更大，所以 {big} 更大')
        return lines

    lines.append(f'位数相同，都是 {len(sa)} 位，从最高位逐位比较')
    for i, (x, y) in enumerate(zip(sa, sb)):
        unit = UNITS[len(sa) - 1 - i]
        if x == y:
            lines.append(f'{unit}: {x} = {y}，相同，继续比下一位')
        else:
            big = a if x > y else b
            sym = '>' if x > y else '<'
            lines.append(f'{unit}: {x} {sym} {y}，分出大小，所以 {big} 更大')
            return lines

    lines.append(f'所有位都相同，{a} 等于 {b}')
    return lines


def _sub_core(a, b):
    """a-b（要求 a>=b）的逐位借位演算行（不含比较/交换/表头/收尾）。

    除法每步的 `余 - 商*除数` 复用本核心，使减法借位行全库只有一种格式。
    """
    da, db = _low_digits(a), _low_digits(b)
    lines, borrow, out = [], 0, []
    for i in range(len(da)):
        x = da[i]
        y = db[i] if i < len(db) else 0
        t = x - y - borrow
        if t < 0:
            lines.append(f'{UNITS[i]}: {x} - {y} - {borrow} 不够减，借1 → '
                         f'{x + 10} - {y} - {borrow} = {t + 10}，写 {t + 10} 借位 1')
            out.append(t + 10)
            borrow = 1
        else:
            lines.append(f'{UNITS[i]}: {x} - {y} - {borrow} = {t}，写 {t} 借位 0')
            out.append(t)
            borrow = 0
    return lines, out


def sub_trace(a, b, compare=True):
    """整数减法的分步轨迹；a < b 时先交换再取负。

    compare=True 时在开头插入逐位大小比较（见 compare_trace），随后是
    一行显式交换状态（v6 新增）：交换与否都写明，让"是否已交换"成为
    可寻址的 token 而非隐式状态。内层递归传 compare=False，避免比较
    过程重复出现两次。

    交换路径的符号处理（v6）：内层算 b - a，答案行保持正数 magnitude，
    符号由最末行 `结果为负: -X` 单独声明 —— v5 是用正则把答案行改写成
    `答案: - X`，符号决策被压在高熵的行尾；v6 把它提前到交换状态行
    之后，答案行变成纯拷贝。
    """
    head = compare_trace(a, b) if compare else []

    if a < b:
        if compare:
            head.append(f'{a} < {b}，交换后计算 {b} - {a}')
        lines = head + sub_trace(b, a, compare=False).split('\n')
        lines.append(f'结果为负: -{b - a}')
        return '\n'.join(lines)

    if compare:
        rel = '>' if a > b else '='
        head.append(f'{a} {rel} {b}，不交换')

    lines = head + [f'竖式计算 {a} - {b}']
    core, out = _sub_core(a, b)
    lines.extend(core)
    return _finish(lines, out)


def _mul_block_lines(n, d):
    """n * d（d 为一位数）的逐位演算行，返回 (行列表, 低位在前结果)。

    乘法部分积的每一行与除法的候选积都走这里，保证 `x * d` 的行格式
    全库只有一种。进位链上限 9*9+8=89，一行只含一次一位乘法。
    """
    dn = _low_digits(n)
    lines, carry, out = [], 0, []
    for i, x in enumerate(dn):
        t = x * d + carry
        expr = f'{x} * {d} + {carry} = {t}' if carry else f'{x} * {d} = {t}'
        lines.append(f'{UNITS[i]}: {expr}，写 {t % 10} 进位 {t // 10}')
        out.append(t % 10)
        carry = t // 10
    if carry:
        lines.append(f'最高位进位 {carry}')
        out.append(carry)
    return lines, out


def mul_trace(a, b):
    """非负整数乘法的部分积竖式轨迹。

    结构：乘数取位数少的操作数（显式声明选择）→ 乘数每一位单独成行
    （逐位乘，与加法进位行同构）→ 错位补零显式写出（不靠心算对齐）
    → 逐行累加（两两相加，复用 _add_core）。错位不写隐式对齐而写补零
    后的完整数，与「竖式演算行的数字都是显式字符串」的全库原则一致。
    """
    sa, sb = str(a), str(b)
    if len(sa) + len(sb) > MAX_DIGITS:
        raise ValueError(f'{a} * {b} 的积可能超过 {MAX_DIGITS} 位帧，需缩短操作数')
    # 乘数 m 逐位去乘被乘数 n；位数少的作乘数（打平时取 b），既缩短轨迹
    # 又让"选哪边逐位乘"成为可寻址的显式决策（与减法交换状态行同理）
    m, n = (b, a) if len(sb) <= len(sa) else (a, b)
    lines = [f'竖式计算 {a} * {b}']
    if len(str(m)) < len(str(n)):
        lines.append(f'乘数用位数少的 {m}，逐位乘 {n}')
    else:
        lines.append(f'乘数用 {m}，逐位乘 {n}')

    rows = []
    for k, d in enumerate(_low_digits(m)):
        lines.append(f'第{k + 1}行: {n} * {d}（{UNITS[k]}）')
        block, out = _mul_block_lines(n, d)
        lines.extend(block)
        val = _low_to_int(out)
        row = f'第{k + 1}行结果: {val}'
        if k and val:
            row += f'，错 {k} 位: {val * 10 ** k}'
        lines.append(row)
        rows.append(val * 10 ** k)

    acc = rows[0]
    if len(rows) > 1:
        lines.append('逐行累加')
        for r in rows[1:]:
            lines.append(f'累加: {acc} + {r}')
            core, out = _add_core(acc, r)
            lines.extend(core)
            acc = _low_to_int(out)
            lines.append(f'合计: {acc}')
    return _finish(lines, _low_digits(acc))


def div_trace(a, b, overshoot=0):
    """整数长除法轨迹：试商 → 逐位乘验证 → 逐位减 → 落下一位。

    真商 `rem // b` 是除法唯一的一次性"猜"决策（训练方案 §6 标记的难点），
    对策是把验证完全拆成一位数步骤：候选积 c*b 逐位展开（一位除数是口诀
    一行，多位除数走 _mul_block_lines），偏大则 `试 c-1` 逐个下调。数据侧
    用 overshoot>0 的样本教模型自纠，推理时猜偏 1~2 位仍能收敛。

    overshoot: 首个试商候选比真商大多少（数据侧 0~2），0 表示直接命中。
    """
    if b <= 0:
        raise ValueError('除数必须为正整数')
    sa = str(a)
    if len(sa) > MAX_DIGITS:
        raise ValueError(f'被除数 {a} 超过 {MAX_DIGITS} 位帧')
    lines = [f'竖式计算 {a} / {b}']

    # 取被除数最短前缀使得 >= 除数：商的最高位位置由此确定
    L = next((l for l in range(1, len(sa) + 1) if int(sa[:l]) >= b), None)
    if L is None:
        # a < b：商 0 余 a。保留为独立分支，避免主循环对空商特判
        lines.append(f'{a} < {b}，首位就不够商，商 0')
        lines.append('答案(1位): 0')
        lines.append(f'余数: {a}')
        return '\n'.join(lines)

    lines.append(f'前 {L} 位 {sa[:L]} >= {b}，'
                 f'商从{UNITS[len(sa) - L]}起，共 {len(sa) - L + 1} 位')
    rem = int(sa[:L])
    q_digits = []
    pos = len(sa) - L                     # 当前商位的位名（UNITS 下标）
    for j in range(L, len(sa) + 1):
        q = rem // b                      # 真商 ≤ 9：rem < 10*b 由前缀构造保证
        c = min(9, q + max(0, overshoot))
        lines.append(f'{UNITS[pos]}: {rem} 试商')
        while True:
            bc = b * c
            if len(_low_digits(b)) > 1:
                lines.append(f'{b} * {c}')
                block, _ = _mul_block_lines(b, c)
                lines.extend(block)
            lines.append(f'{b} * {c} = {bc}')
            if bc > rem:
                lines.append(f'{bc} > {rem}，太大，试 {c - 1}')
                c -= 1
                continue
            break
        if bc == 0:
            # 商 0 档：x - 0 的逐位借位行全是平凡抄写，一行直写更省
            lines.append(f'{bc} <= {rem}，{rem} - 0 = {rem}')
            rem2 = rem
        else:
            lines.append(f'{bc} <= {rem}，够减，逐位减')
            core, out = _sub_core(rem, bc)
            lines.extend(core)
            rem2 = _low_to_int(out)
            lines.append(f'差: {rem2}')
        lines.append(f'{rem2} < {b}，商 {c}')
        q_digits.append(c)
        rem = rem2
        if j < len(sa):
            d = int(sa[j])
            lines.append(f'落下 {d}，{rem} * 10 + {d} = {rem * 10 + d}')
            rem = rem * 10 + d
            pos -= 1

    q = int(''.join(map(str, q_digits)))
    lines.append(f'答案({len(q_digits)}位): {q}')
    lines.append(f'余数: {rem}')
    return '\n'.join(lines)


def build_trace(a, op, b):
    if op == '+':
        return add_trace(a, b)
    if op == '-':
        return sub_trace(a, b)
    if op == '*':
        return mul_trace(a, b)
    if op == '/':
        return div_trace(a, b)
    raise ValueError(f'未知运算符 {op}')


# ----------------------------------------------------------------------------
# 解析：从模型输出里抽取答案
# ----------------------------------------------------------------------------

_NEG_LINE = re.compile(r'结果为负\s*[:：]\s*(-\s*[\d\s]+)')
_ANS_LINE = re.compile(r'答案(?:\(\d+位\))?\s*[:：]\s*(-?\s*[\d\s]+)')
_CONCLUSION = re.compile(r'=\s*(-?\d+)\s*$')


def parse_answer(text):
    """抽取模型给出的最终答案。

    优先级：`结果为负:` 行（v6 交换路径的末行）> `答案:` 行（同时兼容
    v5 的 `答案: - 2` 与 v6 的 `答案(3位): 685`）> 末尾 `= N` 结论 >
    文本中最后一个整数。返回 int，无法解析时返回 None。

    结果为负行必须优先于答案行：v6 交换路径里答案行是 magnitude（正数），
    符号在末行声明，读反了会把 -2 判成 2。
    """
    m = _NEG_LINE.search(text)
    if m:
        digits = re.sub(r'\D', '', m.group(1))
        if digits:
            return -int(digits)

    m = _ANS_LINE.search(text)
    if m:
        raw = m.group(1)
        neg = raw.lstrip().startswith('-')
        digits = re.sub(r'\D', '', raw)
        if digits:
            return -int(digits) if neg else int(digits)

    for line in reversed(text.strip().splitlines()):
        m = _CONCLUSION.search(line.strip())
        if m:
            return int(m.group(1))

    nums = re.findall(r'-?\d+', text.replace(',', '').replace(' ', ''))
    return int(nums[-1]) if nums else None


_REMAINDER_LINE = re.compile(r'余数\s*[:：]\s*(\d+)')


def parse_remainder(text):
    """从除法轨迹里抽 `余数: X` 行。无该行返回 None（加减乘轨迹恒 None）。"""
    m = _REMAINDER_LINE.search(text)
    return int(m.group(1)) if m else None


def parse_conclusion(text):
    """单独抽取结论行 `所以 a op b = N` 里的 N。

    v4 之后训练数据不再生成结论行（见 _finish），本函数只用于评测
    v2~v4 这些旧 checkpoint —— 它们仍会输出结论行，评测器据此统计
    "答案行≠结论行"。对新模型恒返回 None，该项统计自然消失。
    """
    for line in reversed(text.strip().splitlines()):
        m = _CONCLUSION.search(line.strip())
        if m:
            return int(m.group(1))
    return None
