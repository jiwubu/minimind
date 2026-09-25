#!/usr/bin/env python3
"""交互式验证 jiwubu/minimind-math-v3（Qwen3 架构 64M 模型）。

支持任意自然语言输入（模型除数学外也有通用语言能力），若输入恰好是
算式（如 `1245 + 111`）则额外附上 Python 计算的正确答案作参照。

用法:
    python verify_model.py          # chat template + 贪心解码
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "jiwubu/minimind-math-v3"

ap = argparse.ArgumentParser()
ap.add_argument("--max-new-tokens", type=int, default=512)
args = ap.parse_args()

print(f"加载 {MODEL_ID} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="float16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("就绪。输入任意文本或算式验证（如 1245 + 111 / 875 / 12 / 你好），回车退出\n")

while True:
    try:
        line = input("输入> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        break
    q = line

    # 模型按 chat 格式训练，套 chat template + 贪心解码才稳定出结果
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False,
        add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(do_sample=False)
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print('🧠: ', end='')
    model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        **gen_kwargs,
    )
    print('\n')
