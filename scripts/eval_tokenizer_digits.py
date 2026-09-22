# 验证 patch_tokenizer_digits.py 的效果：数字是否逐位切分、其余文本是否完全没受影响。
# 用法：python eval_tokenizer_digits.py            # 只看当前 model/ 的行为
#       python eval_tokenizer_digits.py --compare  # 与未 patch 的版本逐项对比
import os
import json
import shutil
import tempfile
import argparse

from transformers import AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

NUMERIC_CASES = [
    '1234567', '1234568', '12345', '1999', '2023年', '100000',
    '3.14159', '0.5', '9999999999',
    '价格是1250元', '1+1=2', '100+250=350', '共 100 元', '(2023)', 'x1y2',
]

# 通用语料：必须完全不含数字，这样切分才应当与 patch 前逐 token 一致
GENERAL_CASES = [
    '人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。',
    'Large language models (LLMs) are a type of artificial intelligence trained on vast amounts of text data.',
    'Python 是一种高级编程语言。It is widely used in data science, machine learning, and web development.',
    'def square(x): return x * x',
    '你好，世界！',
]
assert not any(c.isdigit() for t in GENERAL_CASES for c in t), 'GENERAL_CASES 不应含数字'


def unpatched_copy(model_dir):
    """复制一份 model/ 并把 pre_tokenizer 还原成单个 ByteLevel，作为对照组。"""
    tmp = tempfile.mkdtemp(prefix='tk_baseline_')
    dst = os.path.join(tmp, 'model')
    shutil.copytree(model_dir, dst)
    path = os.path.join(dst, 'tokenizer.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pre = data.get('pre_tokenizer', {})
    if pre.get('type') == 'Sequence':
        for sub in pre.get('pretokenizers', []):
            if sub.get('type') == 'ByteLevel':
                data['pre_tokenizer'] = sub
                break
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return tmp, dst


def toks(tk, text):
    return [tk.convert_ids_to_tokens(i) for i in tk.encode(text)]


def report_single(tk):
    print('=' * 88)
    print('数字切分')
    print('=' * 88)
    for text in NUMERIC_CASES:
        print(f'  {text!r:18} -> {toks(tk, text)}')

    print()
    print('=' * 88)
    print('一致性：同一数字在不同位置应当是同一个 id')
    print('=' * 88)
    for text in ['7', '17', '777', '1000007', 'x=7']:
        ids = tk.encode(text)
        print(f'  {text!r:12} -> {toks(tk, text)}  ids={ids}')

    print()
    print('=' * 88)
    print('roundtrip 与压缩率')
    print('=' * 88)
    for text in GENERAL_CASES:
        ids = tk.encode(text)
        ok = tk.decode(ids) == text
        print(f'  roundtrip={ok}  {len(text) / len(ids):.2f} 字符/token  {text[:40]}')


def report_compare(new_dir, old_dir):
    new = AutoTokenizer.from_pretrained(new_dir)
    old = AutoTokenizer.from_pretrained(old_dir)

    print('=' * 88)
    print('词表兼容性（方案 A 的前提：id 必须完全不变）')
    print('=' * 88)
    same_vocab = new.get_vocab() == old.get_vocab()
    print(f'  vocab 完全一致: {same_vocab}  (size={len(new.get_vocab())})')
    specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<think>', '</think>']
    same_special = new.convert_tokens_to_ids(specials) == old.convert_tokens_to_ids(specials)
    print(f'  特殊 token id 一致: {same_special}  {new.convert_tokens_to_ids(specials)}')

    print()
    print('=' * 88)
    print('数字切分：patch 前 -> patch 后')
    print('=' * 88)
    for text in NUMERIC_CASES:
        print(f'  {text!r}')
        print(f'      before: {toks(old, text)}')
        print(f'      after : {toks(new, text)}  roundtrip={new.decode(new.encode(text)) == text}')

    print()
    print('=' * 88)
    print('通用语料：应当全部 SAME（数字改动不应波及其他文本）')
    print('=' * 88)
    all_same = True
    for text in GENERAL_CASES:
        eo, en = old.encode(text), new.encode(text)
        same = eo == en
        all_same &= same
        print(f'  {"SAME" if same else "DIFF"}  {len(eo)} -> {len(en)} tokens  {text[:40]}')
    print(f'  全部一致: {all_same}')

    print()
    print('=' * 88)
    print('chat_template')
    print('=' * 88)
    messages = [
        {'role': 'user', 'content': '2023年有365天，算一下1250*3'},
        {'role': 'assistant', 'content': '结果是3750'},
    ]
    prompt = new.apply_chat_template(messages, tokenize=False)
    ids = new.encode(prompt)
    print(f'  roundtrip={new.decode(ids, skip_special_tokens=False) == prompt}  tokens={len(ids)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证数字分词 patch 的效果')
    parser.add_argument('--compare', action='store_true', help='与未 patch 的 tokenizer 对比')
    parser.add_argument('--path', default=MODEL_DIR, help='tokenizer 目录')
    args = parser.parse_args()

    if args.compare:
        tmp, baseline = unpatched_copy(args.path)
        try:
            report_compare(args.path, baseline)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        report_single(AutoTokenizer.from_pretrained(args.path))
