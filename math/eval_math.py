"""四则运算评测：按「位数 × 运算符 × 写法」分档给出准确率。

务必用本脚本（而非肉眼或通用提取逻辑）判断改动是否有效：曾发生过
正确输出 `答案(3位): 685` 被"取最后一个数字"的提取逻辑判成 5，
导致 55% 的真实准确率被误报为 0% —— 答案解析必须与轨迹格式单点维护。

用法:
    python math/eval_math.py --weight full_sft_math_v1 --max_digits 8 --n 25
"""

import os
import sys
import json
import random
import argparse
import warnings

import torch
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from math_format import parse_answer, parse_remainder  # noqa: E402
from gen_math_data_muldiv import (  # noqa: E402
    sample_mul_general, sample_mul_onedigit, sample_mul_tens,
    sample_mul_zeroinside, sample_mul_power10, sample_mul_repdigit,
    sample_mul_nines, sample_mul_equal,
    sample_div_zeroq, sample_div_zerorun, sample_div_small)

warnings.filterwarnings('ignore')

# 分档 → 展示名的唯一出处：汇总表、错例打印与 eval_math_fast 共用，防止漂移
GROUP_LABELS = (
    ('equal', '等长'), ('uneq', '不等长'), ('short1st', '短在前'),
    ('nospace', '紧凑写法'), ('nospace_uneq', '紧凑不等长'),
    ('repdigit', '全同数字'), ('zerorun', '长零串'), ('neg_suffix', '负数共后缀'),
    ('neg', '负数'), ('neg_close', '负数·等长接近'), ('leadzero', '前导零'),
    ('longrun', '长串答案'), ('mul', '乘法（通用）'), ('mul_1d', '乘一位乘数'),
    ('mul_tens', '乘末尾零'), ('mul_zeroinside', '乘内嵌0'),
    ('mul_power10', '乘整十幂'), ('mul_repdigit', '乘全同数字'),
    ('mul_nines', '乘9串'), ('mul_equal', '乘相等'), ('div_exact', '整除'),
    ('div_remain', '带余'), ('div_zeroq', '商含0'), ('div_zerorun', '商零串'),
    ('div_small', '小除以大'))


def generate_and_score(model, tokenizer, cases, args, device):
    """批量生成 + 判分。eval_math 与 eval_math_fast 共用，保证快速评测与
    全量评测的判分口径完全一致（单点维护）。"""
    for i in range(0, len(cases), args.batch_size):
        chunk = cases[i:i + args.batch_size]
        outs = batch_ask(model, tokenizer, [c['q'] for c in chunk],
                         device, args.max_new_tokens)
        for c, o in zip(chunk, outs):
            c['raw'] = o
            c['pred'] = parse_answer(o)
            if c['op'] == '/':
                # 除法商余双判分（§7.4）；轨迹含`太大` = 首候选未命中、走了自纠
                c['pred_r'] = parse_remainder(o)
                c['ok'] = c['pred'] == c['gt'] and c['pred_r'] == c['gt_r']
                c['self_correct'] = '太大' in o
            else:
                c['ok'] = c['pred'] == c['gt']
        done = min(i + args.batch_size, len(cases))
        print(f'\r进度 {done}/{len(cases)}', end='', flush=True)
    print('\n')


def pick_device(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = MiniMindConfig(hidden_size=args.hidden_size,
                         num_hidden_layers=args.num_hidden_layers,
                         use_moe=bool(args.use_moe))
    model = MiniMindForCausalLM(cfg)
    suffix = '_moe' if args.use_moe else ''
    ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{suffix}.pth')
    model.load_state_dict(torch.load(ckp, map_location='cpu'), strict=True)
    dtype = torch.float32 if device in ('cpu', 'mps') else torch.float16
    return model.to(dtype).eval().to(device), tokenizer


@torch.no_grad()
def batch_ask(model, tokenizer, questions, device, max_new_tokens):
    """左padding批量生成，单卡下比逐条快很多"""
    prompts = [
        tokenizer.apply_chat_template([{'role': 'user', 'content': q}],
                                      tokenize=False, add_generation_prompt=True,
                                      open_thinking=False)
        for q in questions
    ]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    enc = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    tokenizer.padding_side = old_side

    out = model.generate(
        inputs=enc['input_ids'], attention_mask=enc['attention_mask'],
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[:, enc['input_ids'].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def _rand_operand(d):
    lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
    return random.randint(lo, hi)


def build_cases(args):
    """固定seed生成评测集，保证不同checkpoint之间可比。

    分组原则：每一档都对应一个曾经在评测里不可见的维度：
      equal        两操作数位数相同（12 + 34）
      uneq         位数差 >= 2，长数在前（123 + 12）
      short1st     位数差 >= 2，短数在前（12 + 123）—— 未覆盖时实测仅 0.7%
      nospace      运算符两侧无空格、且等长（99+99）—— 未覆盖时实测仅 46%
      nospace_uneq 无空格且不等长（99+999）—— nospace 只测等长时完全不可见
      repdigit     全同数字 + 无空格（99+999）—— 随机采样生成不出，实测 v4 必错
      zerorun      结果带长串 0（2222-222=2000）—— 竖式对、正序拷贝时数不清 0
      neg_suffix   负数且 b = 前缀+a（99-199）—— 共享后缀，比较行报错位数
      neg          负数一般情形（123-456）—— 前六档全部 b>a 时交换，0 覆盖
      neg_close    负数·等长且接近（共享前缀，末几位分出大小）—— v5 负数档
                   全部错例的形状，旧 neg 档均匀采样下只占 3%，看不见
      leadzero     结果带 k 个前导零（2222-2222+111 → 00111）—— 中间深度
                   的 k=3~6 是上一版的采样空档，按 k 分桶统计
      longrun      答案带长重复串但操作数普通（1234000+5678）—— zerorun 够不着
      mul*         乘法八档（通用/一位/末尾零/内嵌0/整十幂/全同/9串/相等）——
                   直接复用训练采样器，评测形状 = 训练形状
      div*         除法：通用拆整除/带余（商余双判分）+ 商含0/商零串/小除以大；
                   除法专项统计除数位数 × 首候选是否命中（§7.4）

    nospace 档保留的原因：v6 patched tokenizer 下两种写法的数字切分已完全
    一致，但它测的"读紧凑题面"能力本身仍要单独可见。
    """
    random.seed(args.seed)
    cases = []
    for d in range(1, args.max_digits + 1):
        for op in '+-':
            for _ in range(args.n):
                a, b = _rand_operand(d), _rand_operand(d)
                if op == '-' and b > a:
                    a, b = b, a
                gt = a + b if op == '+' else a - b
                cases.append({'group': 'equal', 'digits': d, 'op': op,
                              'a': a, 'b': b, 'gt': gt, 'q': f'{a} {op} {b} = ?'})

    # 不等长：长操作数为 d 位，短的为 1..d-2 位
    for d in range(3, args.max_digits + 1):
        for op in '+-':
            for _ in range(args.n):
                a = _rand_operand(d)
                b = _rand_operand(random.randint(1, d - 2))
                if op == '-' and b > a:
                    a, b = b, a
                gt = a + b if op == '+' else a - b
                cases.append({'group': 'uneq', 'digits': d, 'op': op,
                              'a': a, 'b': b, 'gt': gt, 'q': f'{a} {op} {b} = ?'})

    # 短数在前（1 + 112）：训练数据里长期是0覆盖，不单独成档就测不出来。
    # 仅加法——减法短在前必然是负数结果，属于另一条代码路径。
    for d in range(3, args.max_digits + 1):
        for _ in range(args.n):
            b = _rand_operand(d)
            a = _rand_operand(random.randint(1, d - 2))
            cases.append({'group': 'short1st', 'digits': d, 'op': '+',
                          'a': a, 'b': b, 'gt': a + b, 'q': f'{a} + {b} = ?'})

    # 紧凑写法（99+999）：tokenizer 切分不同，转写容易数错位数
    for d in range(2, args.max_digits + 1):
        for op in '+-':
            for _ in range(args.n):
                a, b = _rand_operand(d), _rand_operand(d)
                if op == '-' and b > a:
                    a, b = b, a
                gt = a + b if op == '+' else a - b
                cases.append({'group': 'nospace', 'digits': d, 'op': op,
                              'a': a, 'b': b, 'gt': gt, 'q': f'{a}{op}{b}'})

    # 紧凑 + 不等长（99+999）：必须与紧凑等长分开统计。
    # 紧凑写法下数字按两位一组贪心合并，奇数位会在末尾留一个单字符 token：
    #   '99+999'  -> ['99']['+']['99']['9']
    #   '99+9999' -> ['99']['+']['99']['99']
    # 两者 token 数相同（都是 4 个），仅末位是 ['9'] 还是 ['99'] 之差，
    # 而这两个 ID 在 embedding 里毫无“长度”关系 —— 位数只能靠死记。
    # 实测 v4 会把 '99+999' 抄成竖式 '9 9 + 9 9 9 9'：演算全对，读题错。
    # 带空格写法不踩这个坑（' 9' 起头后边界整齐），等长写法两侧对称也不踩，
    # 所以只测等长的旧 nospace 档对此完全不可见。
    for d in range(3, args.max_digits + 1):
        for op in '+-':
            for _ in range(args.n):
                a = _rand_operand(d)
                b = _rand_operand(random.randint(1, d - 1))
                if op == '-' and b > a:
                    a, b = b, a
                if random.random() < 0.5 and op == '+':
                    a, b = b, a          # 短数在前也要覆盖
                gt = a + b if op == '+' else a - b
                cases.append({'group': 'nospace_uneq', 'digits': d, 'op': op,
                              'a': a, 'b': b, 'gt': gt, 'q': f'{a}{op}{b}'})

    # 全同数字（99+999 / 2222 - 222）：随机采样几乎生成不出这种数，必须显式构造。
    # 见 gen_math_data_addsub.sample_repdigit_pair 的说明：紧凑写法下全同数字时
    # ['99','9'] 与 ['99','99'] token 数相同且内容重复，位数信息丢失。
    #
    # **写法必须两种都测**。本档原先 362 题紧凑 / 1 题带空格，于是
    # v4 训完手工试出的 `2222 - 222`（带空格、两侧全同，答成 200）在本档
    # 里几乎不可见。而带空格并不安全：真正的公共难点是长重复串数不清位数，
    # tokenizer 只是在 1/2/3/9 上额外添一刀（两位重复 token 只有
    # 00 11 22 33 99 五个，4~8 本就一位一 token 却照样错）。
    for d in range(2, args.max_digits + 1):
        for op in '+-':
            for _ in range(args.n):
                a = int(str(random.randint(1, 9)) * d)
                b = int(str(random.randint(1, 9)) * random.randint(1, d))
                if op == '-' and b > a:
                    a, b = b, a
                if random.random() < 0.5 and op == '+':
                    a, b = b, a
                gt = a + b if op == '+' else a - b
                # 一半紧凑、一半带空格，两类诱因分别可见
                q = f'{a}{op}{b}' if random.random() < 0.5 else f'{a} {op} {b}'
                cases.append({'group': 'repdigit', 'digits': d, 'op': op,
                              'a': a, 'b': b, 'gt': gt, 'q': q})

    # 结果带长串 0（2222-222=2000）：竖式逐位写 0 是对的，错在"逆序→答案"
    # 那一步要把这串 0 一位不差地拷过去。v4 训完实测 2222-222 答成 0、
    # 66666-6666 答成 600060 —— 0 全丢、或多写一位并重复了首位数字。
    # 这与 repdigit 档的位数读错是两种机制（那是读题错，这是拷贝错），
    # 且结果里 ≥4 位同数字连续段自然采样只占 7.5%，必须单独成档。
    # 写法同样两种都测：4~8 这些数字本就一位一 token，紧凑与否都会错。
    for d in range(3, args.max_digits + 1):
        for _ in range(args.n):
            digit = random.randint(1, 9)
            d2 = random.randint(1, d - 1)
            a = int(str(digit) * d)
            b = int(str(digit) * d2)
            q = f'{a}-{b}' if random.random() < 0.5 else f'{a} - {b}'
            cases.append({'group': 'zerorun', 'digits': d, 'op': '-',
                          'a': a, 'b': b, 'gt': a - b, 'q': q})

    # 负数结果（99-199 = -100）：**前六档一个都没有**。每档都写了
    # `if op == '-' and b > a: a, b = b, a`，2000 题里负数结果 0 题，
    # 而训练数据里有 2.8%。v4 训完手工实测 99-199 答成 800、77-177 答成
    # 600、90-100 答成 -10，评测却报 99.0% —— 整个维度不可见。
    #
    # 最难的是 b = 前缀 + a（199 的后两位就是 99）：两个数字串共享后缀，
    # 模型在比较行把 99 的位数报成 3 位，走进"位数相同→逐位比较"分支，
    # 结论反了（"所以 99 更大"），符号和竖式跟着全错。而 77-777 这种不
    # 共享后缀的，v4 答对 —— 所以本档必须按形状分开采，不能只随机取 a<b。
    # d 是 b 的位数，a 固定为 d-1 位 —— 这样各档难度随位数真实递增
    # （否则 d=8 那档也只在测 9-29 这种一位数）
    for d in range(2, args.max_digits + 1):
        for _ in range(args.n):
            a = _rand_operand(d - 1)
            lead = random.randint(1, 9)
            b = int(f'{lead}{str(a).zfill(d - 1)}')   # 前缀 + a，共享后缀
            q = f'{a}-{b}' if random.random() < 0.5 else f'{a} - {b}'
            cases.append({'group': 'neg_suffix', 'digits': d, 'op': '-',
                          'a': a, 'b': b, 'gt': a - b, 'q': q})

    # 负数结果的一般情形：等长（靠逐位比较判负）+ 不等长（靠位数判负）
    for d in range(2, args.max_digits + 1):
        for _ in range(args.n):
            if random.random() < 0.5:
                lo, hi = 10 ** (d - 1), 10 ** d - 1
                b = random.randint(lo + 1, hi)
                a = random.randint(lo, b - 1)
            else:
                b = _rand_operand(d)
                a = _rand_operand(random.randint(1, d - 1))
                if a >= b:
                    a, b = b, a
            q = f'{a}-{b}' if random.random() < 0.5 else f'{a} - {b}'
            cases.append({'group': 'neg', 'digits': d, 'op': '-',
                          'a': a, 'b': b, 'gt': a - b, 'q': q})

    # 答案带长重复串，但操作数完全普通（1234000 + 5678）。
    # zerorun 档只测 `aa…a - aa…a`：操作数必须全同、只有减法、长串必然
    # 贴在答案末尾。而模型数不清的是**自己正在写的那一串**，与操作数形状
    # 无关 —— 55555555+8888888 答成 14444443 就是加法、操作数不全同。
    # 本档反向构造：先定答案（随机串里嵌一段 >= 3 位重复串，位置随机、
    # 0 串占 40%），再倒推操作数，加减各半。
    for d in range(3, args.max_digits + 1):
        for _ in range(args.n):
            k = random.randint(3, min(d, 6))
            pos = random.randint(0, d - k)
            digit = '0' if random.random() < 0.4 else str(random.randint(1, 9))
            if digit == '0' and pos == 0:
                pos = 1 if d > k else 0
                if pos == 0:
                    digit = str(random.randint(1, 9))
            chars = [str(random.randint(0, 9)) for _ in range(d)]
            chars[pos:pos + k] = [digit] * k
            if chars[0] == '0':
                chars[0] = str(random.randint(1, 9))
            c = int(''.join(chars))
            if random.random() < 0.5:
                a = random.randint(0, c)
                b, op, gt = c - a, '+', c
            else:
                b = random.randint(0, 10 ** args.max_digits - 1 - c)
                a, op, gt = c + b, '-', c
            q = f'{a}{op}{b}' if random.random() < 0.5 else f'{a} {op} {b}'
            cases.append({'group': 'longrun', 'digits': d, 'op': op,
                          'a': a, 'b': b, 'gt': gt, 'q': q})

    # 负数·等长且接近（共享高位前缀，末 k 位分出大小）：v5 负数档的全部
    # 错例（14-15、382-393、77583099-78524249）都是这个形状 —— 模型没执行
    # 交换，直接对小-大跑借位得到 10 的补码。旧 neg 档的等长子区间是均匀
    # 采样，接近形状只占 3%；训练覆盖提到 2% 后评测必须单独成档才看得见
    # （档位总分达标而档内分布塌陷，是最容易复发的评测盲区）。
    for d in range(2, args.max_digits + 1):
        for _ in range(args.n):
            # k 分布与训练采样同步偏置（向 2~3），否则深链改进在评测里不可见
            k = min(random.choice([1, 2, 2, 3, 3]), d - 1)
            head = str(random.randint(1, 9)) + ''.join(
                str(random.randint(0, 9)) for _ in range(d - k - 1))
            tail_a = random.randint(0, 10 ** k - 2)
            tail_b = random.randint(tail_a + 1, 10 ** k - 1)
            a = int(head + str(tail_a).zfill(k))
            b = int(head + str(tail_b).zfill(k))
            q = f'{a}-{b}' if random.random() < 0.5 else f'{a} - {b}'
            cases.append({'group': 'neg_close', 'digits': d, 'op': '-',
                          'a': a, 'b': b, 'gt': a - b, 'q': q, 'k': k})

    # 结果带 k 个前导零（14691927-14691816=111 曾答成 0：8 位竖式跑完得
    # 00000111，一次去掉 5 个前导零时把 111 也吃了）。k 用 d 的帧度量、
    # 按 k 分桶在汇总里单独打印 —— 只看档位总分看不见中间深度的塌陷。
    for d in range(2, args.max_digits + 1):
        for _ in range(args.n):
            # k 分布与训练采样同步偏置：k>=5 加量（深度剥离是实测短板）
            if d >= 6 and random.random() < 0.4:
                k = random.randint(5, d - 1)
            else:
                k = random.randint(1, d - 1)
            r = random.randint(10 ** (d - k - 1), 10 ** (d - k) - 1)
            lo = max(1, 10 ** (d - 1) - r)
            hi = 10 ** d - 1 - r
            b = random.randint(lo, hi)
            a = b + r
            q = f'{a}-{b}' if random.random() < 0.5 else f'{a} - {b}'
            cases.append({'group': 'leadzero', 'digits': d, 'op': '-',
                          'a': a, 'b': b, 'gt': a - b, 'q': q, 'k': k})

    # ---------------- 乘除（§7.4）：分档与训练课程一一对应 ----------------
    # 乘法直接复用训练采样器保证形状一致；档位 d = 较宽操作数的位数。
    # 乘积帧上限 MAX_DIGITS=12 → 操作数位数封顶 6（6*6 恰好 12 位帧，
    # mul_trace 超帧会 raise）；超出训练范围（默认 4）的行正好观察外推。
    mul_cap = min(args.mul_digits or args.max_digits, 6)
    mul_samplers = [('mul', sample_mul_general), ('mul_1d', sample_mul_onedigit),
                    ('mul_tens', sample_mul_tens),
                    ('mul_zeroinside', sample_mul_zeroinside),
                    ('mul_power10', sample_mul_power10),
                    ('mul_repdigit', sample_mul_repdigit),
                    ('mul_nines', sample_mul_nines), ('mul_equal', sample_mul_equal)]
    for group, fn in mul_samplers:
        for _ in range(args.n * mul_cap):
            a, b = fn(mul_cap)
            q = f'{a}*{b}' if random.random() < 0.5 else f'{a} * {b}'
            cases.append({'group': group, 'digits': max(len(str(a)), len(str(b))),
                          'op': '*', 'a': a, 'b': b, 'gt': a * b, 'q': q})

    # 除法：整除 / 带余单独成档（商余双判分），商含 0 / 商零串 / 小除以大
    # 复用训练采样器。档位 d = 被除数位数，除数 1~2 位与训练默认一致。
    div_cap = min(args.div_digits or args.max_digits, 8)

    def _div_pair(dd, exact):
        lo, hi = (1, 9) if dd == 1 else (10, 99)
        b = random.randint(lo if exact else max(2, lo), hi)   # 带余需 b >= 2
        q = _rand_operand(random.randint(1, max(1, div_cap - dd)))
        return (b * q, b) if exact else (b * q + random.randint(1, b - 1), b)

    for group, exact in (('div_exact', True), ('div_remain', False)):
        for _ in range(args.n * div_cap):
            a, b = _div_pair(random.randint(1, 2), exact)
            gq, gr = divmod(a, b)
            q = f'{a}/{b}' if random.random() < 0.5 else f'{a} / {b}'
            cases.append({'group': group, 'digits': len(str(a)), 'op': '/',
                          'a': a, 'b': b, 'gt': gq, 'gt_r': gr, 'q': q,
                          'dd': len(str(b))})

    for group, fn in (('div_zeroq', sample_div_zeroq),
                      ('div_zerorun', sample_div_zerorun)):
        for _ in range(args.n * div_cap // 2):
            a, b = fn(div_cap, 2)
            gq, gr = divmod(a, b)
            q = f'{a}/{b}' if random.random() < 0.5 else f'{a} / {b}'
            cases.append({'group': group, 'digits': len(str(a)), 'op': '/',
                          'a': a, 'b': b, 'gt': gq, 'gt_r': gr, 'q': q,
                          'dd': len(str(b))})

    for _ in range(args.n):
        a, b = sample_div_small(div_cap, 2)
        gq, gr = divmod(a, b)
        q = f'{a}/{b}' if random.random() < 0.5 else f'{a} / {b}'
        cases.append({'group': 'div_small', 'digits': len(str(a)), 'op': '/',
                      'a': a, 'b': b, 'gt': gq, 'gt_r': gr, 'q': q,
                      'dd': len(str(b))})
    return cases


def main():
    ap = argparse.ArgumentParser(description='加减法分档评测')
    ap.add_argument('--weight', default='full_sft_math_v1', help='out/ 下的权重前缀')
    ap.add_argument('--save_dir', default='out')
    ap.add_argument('--tokenizer', default='model')
    ap.add_argument('--hidden_size', type=int, default=768)
    ap.add_argument('--num_hidden_layers', type=int, default=8)
    ap.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    ap.add_argument('--max_digits', type=int, default=8, help='评测到几位数')
    ap.add_argument('--mul_digits', type=int, default=4,
                    help='乘法操作数最大位数（默认=训练范围 4；测外推需显式加大）')
    ap.add_argument('--div_digits', type=int, default=4,
                    help='除法被除数最大位数（默认=训练范围 4；测外推需显式加大）')
    ap.add_argument('--n', type=int, default=25, help='每档题数')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_new_tokens', type=int, default=1500,
                    help='乘除轨迹最长 ≈1300 token（加减仅 ≈440），按乘除给足')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--dump', default='', help='把逐题结果写入jsonl')
    ap.add_argument('--show_errors', type=int, default=2, help='每档打印几条错例')
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load_model(args, device)
    cases = build_cases(args)
    print(f'权重={args.weight} 设备={device} 题数={len(cases)}\n')

    generate_and_score(model, tokenizer, cases, args, device)

    cols = [('equal', '+', '等长+', 7), ('equal', '-', '等长-', 7),
            ('uneq', '+', '不等长+', 8), ('uneq', '-', '不等长-', 8),
            ('short1st', '+', '短在前+', 8),
            ('nospace', '+', '紧凑+', 7), ('nospace', '-', '紧凑-', 7),
            ('nospace_uneq', '+', '紧凑不等长+', 11),
            ('nospace_uneq', '-', '紧凑不等长-', 11),
            ('repdigit', '+', '全同数字+', 9),
            ('repdigit', '-', '全同数字-', 9),
            ('zerorun', '-', '长零串-', 7),
            ('neg_suffix', '-', '负数共后缀-', 11),
            ('neg', '-', '负数-', 7),
            ('neg_close', '-', '负数接近-', 10),
            ('leadzero', '-', '前导零-', 9),
            ('longrun', '+', '长串答案+', 9),
            ('longrun', '-', '长串答案-', 9),
            ('mul', '*', '乘法*', 7), ('mul_1d', '*', '乘一位*', 8),
            ('mul_tens', '*', '末尾零*', 8), ('mul_zeroinside', '*', '内嵌0*', 8),
            ('mul_power10', '*', '整十幂*', 8), ('mul_repdigit', '*', '全同乘*', 8),
            ('mul_nines', '*', '9串乘*', 8), ('mul_equal', '*', '相等乘*', 8),
            ('div_exact', '/', '整除/', 7),
            ('div_remain', '/', '带余/', 7), ('div_zeroq', '/', '商含0/', 8),
            ('div_zerorun', '/', '商零串/', 8), ('div_small', '/', '小除以大/', 9)]
    print(f'{"位数":>4} ' + ' '.join(f'{label:>{width + 1}}'
                                     for _, _, label, width in cols))
    for d in range(1, args.max_digits + 1):
        row = [f'{d:>4}']
        for group, op, _, width in cols:
            sub = [c for c in cases if c['group'] == group
                   and c['digits'] == d and c['op'] == op]
            if sub:
                acc = sum(c['ok'] for c in sub) / len(sub) * 100
                row.append(f'{acc:{width}.0f}%')
            else:
                row.append(' ' * width + '-')
        print(' '.join(row))

    overall = sum(c['ok'] for c in cases) / len(cases) * 100
    print(f'\n总体准确率: {overall:.1f}%')
    for group, label in GROUP_LABELS:
        sub = [c for c in cases if c['group'] == group]
        if sub:
            acc = sum(c['ok'] for c in sub) / len(sub) * 100
            print(f'  {label}: {acc:.1f}%  ({len(sub)} 题)')

    # 档内按 k 分桶：档位总分达标而档内塌陷是第 12 轮的老坑
    for group in ('neg_close', 'leadzero'):
        sub = [c for c in cases if c['group'] == group and 'k' in c]
        if sub:
            parts = []
            for k in sorted({c['k'] for c in sub}):
                bucket = [c for c in sub if c['k'] == k]
                acc = sum(c['ok'] for c in bucket) / len(bucket) * 100
                parts.append(f'k={k}: {acc:.0f}%({len(bucket)})')
            print(f'  {group} 按 k: ' + '  '.join(parts))

    # 除法专项（§7.4）：除数位数分档 + 首候选是否命中（含`太大`即走了自纠）
    divs = [c for c in cases if c['op'] == '/']
    if divs:
        for dd, lbl in ((1, '除数1位'), (2, '除数2位')):
            sub = [c for c in divs if c['dd'] == dd]
            if sub:
                acc = sum(c['ok'] for c in sub) / len(sub) * 100
                print(f'  {lbl}: {acc:.1f}%  ({len(sub)} 题)')
        first = [c for c in divs if not c['self_correct']]
        fixed = [c for c in divs if c['self_correct']]
        if first:
            acc = sum(c['ok'] for c in first) / len(first) * 100
            print(f'  首候选命中（无自纠）: {acc:.1f}%  ({len(first)} 题)')
        if fixed:
            acc = sum(c['ok'] for c in fixed) / len(fixed) * 100
            print(f'  触发自纠（含`太大`）: {acc:.1f}%  ({len(fixed)} 题)'
                  '  ← 验证 overshoot 自纠是否真救回猜偏的样本')

    if args.show_errors:
        print('\n--- 错例 ---')
        for group, label in GROUP_LABELS:
            for d in range(1, args.max_digits + 1):
                errs = [c for c in cases if c['group'] == group
                        and c['digits'] == d and not c['ok']]
                for c in errs[:args.show_errors]:
                    gt_s = (f"{c['gt']} 余 {c['gt_r']}" if c['op'] == '/'
                            else str(c['gt']))
                    pred_s = (f"{c['pred']} 余 {c['pred_r']}"
                              if c['op'] == '/' and c['pred_r'] is not None
                              else str(c['pred']))
                    print(f"[{label}] {c['a']} {c['op']} {c['b']} = {gt_s}，"
                          f"预测 {pred_s}")

    if args.dump:
        with open(args.dump, 'w', encoding='utf-8') as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f'\n逐题结果 -> {args.dump}')


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
