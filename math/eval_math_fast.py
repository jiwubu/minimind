"""快速回归评测：eval_math.py 的精简版，用于训练/调参迭代中的快速检查。

与全量版共用 build_cases 的分档构造和 generate_and_score 的判分口径
（单点维护，防止快速与全量两套标准），差别只有三点：
1. 每档题数少（默认每位数 3 题，全量 25）；
2. 乘除默认只测训练范围（4 位）——训练范围外（5 位以上）在全量版里
   本来就接近 0%，却要为每题生成 ~1300 token，是评测耗时的大头；
3. 输出紧凑汇总与逐档对错标记，不打印逐位数大表。

验收、错例归档与逐位数分析仍用 eval_math.py 全量版。

用法:
    python math/eval_math_fast.py --weight full_sft_math_all_v1
    python math/eval_math_fast.py --weight full_sft_math_all_v1 --n 6   # 稍多题
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_math as em  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='快速回归评测（验收与归档请用 eval_math.py 全量版）')
    ap.add_argument('--weight', default='full_sft_math_all_v1', help='out/ 下的权重前缀')
    ap.add_argument('--save_dir', default='out')
    ap.add_argument('--tokenizer', default='model')
    ap.add_argument('--hidden_size', type=int, default=768)
    ap.add_argument('--num_hidden_layers', type=int, default=8)
    ap.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    ap.add_argument('--max_digits', type=int, default=8, help='加减评测位数')
    ap.add_argument('--mul_digits', type=int, default=4, help='乘法操作数最大位数（默认=训练范围）')
    ap.add_argument('--div_digits', type=int, default=4, help='除法被除数最大位数（默认=训练范围）')
    ap.add_argument('--n', type=int, default=3, help='每档每位数题数（全量版 25）')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_new_tokens', type=int, default=1500)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--dump', default='', help='把逐题结果写入jsonl')
    ap.add_argument('--show_errors', type=int, default=2, help='每档打印几条错例')
    args = ap.parse_args()

    if args.mul_digits > 4 or args.div_digits > 4:
        print('⚠ 乘除位数超过训练范围（4）：包含预期外的长生成，耗时上升且分数接近 0')

    device = em.pick_device(args.device)
    model, tokenizer = em.load_model(args, device)
    cases = em.build_cases(args)
    print(f'权重={args.weight} 设备={device} 题数={len(cases)}（快速模式）')
    em.generate_and_score(model, tokenizer, cases, args, device)

    def acc(sub):
        return sum(c['ok'] for c in sub) / len(sub) * 100 if sub else float('nan')

    addsub = [c for c in cases if c['op'] in '+-']
    muls = [c for c in cases if c['op'] == '*']
    divs = [c for c in cases if c['op'] == '/']
    print(f'\n总体 {acc(cases):.1f}%   加减 {acc(addsub):.1f}%   '
          f'乘 {acc(muls):.1f}%   除 {acc(divs):.1f}%\n')

    # 逐档对错标记：比分数更容易暴露"集中在某一档/某一段位数"的塌陷
    for group, label in em.GROUP_LABELS:
        sub = [c for c in cases if c['group'] == group]
        if sub:
            marks = ''.join('✓' if c['ok'] else '✗' for c in sub)
            print(f'  {label:10} {acc(sub):6.1f}%  {len(sub):3d} 题  {marks}')

    if divs:
        for dd, lbl in ((1, '除数1位'), (2, '除数2位')):
            sub = [c for c in divs if c['dd'] == dd]
            if sub:
                print(f'  {lbl}: {acc(sub):.1f}%  ({len(sub)} 题)')
        first = [c for c in divs if not c['self_correct']]
        fixed = [c for c in divs if c['self_correct']]
        if first:
            print(f'  首候选命中（无自纠）: {acc(first):.1f}%  ({len(first)} 题)')
        if fixed:
            print(f'  触发自纠（含`太大`）: {acc(fixed):.1f}%  ({len(fixed)} 题)')
    for group in ('neg_close', 'leadzero'):
        sub = [c for c in cases if c['group'] == group and 'k' in c]
        if sub:
            parts = [f'k={k}: {acc([c for c in sub if c["k"] == k]):.0f}%'
                     for k in sorted({c['k'] for c in sub})]
            print(f'  {group} 按 k: ' + '  '.join(parts))

    if args.show_errors:
        print('\n--- 错例（每档最多 2 条）---')
        for group, label in em.GROUP_LABELS:
            errs = [c for c in cases if c['group'] == group and not c['ok']]
            for c in errs[:args.show_errors]:
                gt_s = f"{c['gt']} 余 {c['gt_r']}" if c['op'] == '/' else str(c['gt'])
                pred_s = (f"{c['pred']} 余 {c['pred_r']}"
                          if c['op'] == '/' and c['pred_r'] is not None
                          else str(c['pred']))
                print(f"[{label}] {c['a']} {c['op']} {c['b']} = {gt_s}，预测 {pred_s}")

    if args.dump:
        with open(args.dump, 'w', encoding='utf-8') as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f'\n逐题结果 -> {args.dump}')


if __name__ == '__main__':
    main()
