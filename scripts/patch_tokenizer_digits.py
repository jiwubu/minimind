# 把 model/tokenizer.json 的 pre_tokenizer 换成 Digits + ByteLevel，让数字逐位切分。
# vocab 和 merges 完全不动，所以 token id 不变、旧权重仍可加载；只是那些多位数字的 merge
# 规则从此触发不到（例如 "2023" 不再是一个 token）。详见 docs/tokenizer_digits.md。
import os
import json
import argparse

TOKENIZER_JSON = os.path.join(os.path.dirname(__file__), '..', 'model', 'tokenizer.json')

BYTE_LEVEL = {
    "type": "ByteLevel",
    "add_prefix_space": False,
    "trim_offsets": True,
    "use_regex": True
}
DIGITS = {"type": "Digits", "individual_digits": True}


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save(path, data):
    # 原文件就是 indent=2 + ensure_ascii=False 且结尾无换行，保持一致好让 diff 只剩 pre_tokenizer
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def state_of(pre_tokenizer):
    """判断当前 pre_tokenizer 处于哪种状态，用于保证幂等。"""
    if pre_tokenizer is None:
        return 'unknown'
    if pre_tokenizer.get('type') == 'ByteLevel':
        return 'original'
    if pre_tokenizer.get('type') == 'Sequence':
        subs = pre_tokenizer.get('pretokenizers', [])
        types = [s.get('type') for s in subs]
        if types == ['Digits', 'ByteLevel']:
            return 'patched'
    return 'unknown'


def patch(path):
    data = load(path)
    state = state_of(data.get('pre_tokenizer'))
    if state == 'patched':
        print('已经是 Digits + ByteLevel，无需重复修改')
        return False
    if state == 'unknown':
        raise SystemExit(
            f'pre_tokenizer 不是预期的 ByteLevel，已被手工改过，拒绝覆盖：\n'
            f'{json.dumps(data.get("pre_tokenizer"), ensure_ascii=False)}'
        )
    byte_level = data['pre_tokenizer']  # 保留原字段值，不硬编码覆盖
    data['pre_tokenizer'] = {"type": "Sequence", "pretokenizers": [dict(DIGITS), byte_level]}
    save(path, data)
    print('已切换为 Digits(individual_digits=True) + ByteLevel')
    return True


def revert(path):
    data = load(path)
    state = state_of(data.get('pre_tokenizer'))
    if state == 'original':
        print('已经是原始的 ByteLevel，无需回滚')
        return False
    if state == 'unknown':
        raise SystemExit(
            f'pre_tokenizer 不是本脚本写入的结构，拒绝回滚：\n'
            f'{json.dumps(data.get("pre_tokenizer"), ensure_ascii=False)}'
        )
    for sub in data['pre_tokenizer']['pretokenizers']:
        if sub.get('type') == 'ByteLevel':
            data['pre_tokenizer'] = sub
            break
    save(path, data)
    print('已回滚为原始的 ByteLevel')
    return True


def check(path):
    data = load(path)
    state = state_of(data.get('pre_tokenizer'))
    print(f'pre_tokenizer 状态: {state}')
    print(json.dumps(data.get('pre_tokenizer'), ensure_ascii=False, indent=2))
    print(f'vocab 大小: {len(data["model"]["vocab"])}（本脚本不会改动 vocab / merges）')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='切换 minimind tokenizer 的数字切分策略')
    parser.add_argument('--revert', action='store_true', help='回滚为原始的单个 ByteLevel')
    parser.add_argument('--check', action='store_true', help='只打印当前状态，不修改文件')
    parser.add_argument('--path', default=TOKENIZER_JSON, help='tokenizer.json 路径')
    args = parser.parse_args()

    if args.check:
        check(args.path)
    elif args.revert:
        revert(args.path)
    else:
        patch(args.path)
