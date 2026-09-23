#!/usr/bin/env bash
set -eou pipefail

# 数学专用模型训练一键脚本：数据生成 → 预训练 → 合并 SFT（方案 §2/§3/§7）
# 前提：dataset/sft_t2t_mini.jsonl 与 pretrain_t2t_mini.jsonl 已同步到本机
#（两个通用语料大文件不入 git，需 rsync/下载；算术数据在下面现生成）

# =========================
# 环境准备
# =========================
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# =========================
# 数据生成（§3；dataset/*.jsonl 不入 git，服务器上现生成，约几分钟）
# =========================
python math/gen_math_data_addsub.py \
  --out dataset/math_addsub_v1.jsonl \
  --n 200000 --max_digits 8 \
  --mix_general dataset/sft_t2t_mini.jsonl --mix_ratio 0.3

python math/gen_math_data_muldiv.py \
  --out dataset/math_muldiv_v1.jsonl \
  --n 200000 \
  --mix_general dataset/sft_t2t_mini.jsonl --mix_ratio 0.3

cat dataset/math_addsub_v1.jsonl dataset/math_muldiv_v1.jsonl > dataset/math_all_v1.jsonl
wc -l dataset/math_all_v1.jsonl    # 应为 571,428

# =========================
# 预训练（§2.2，从零，64M 默认配置）
# =========================
cd trainer

python train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight none \
  --save_weight pretrain_math \
  --max_seq_len 340 \
  --epochs 2 \
  --batch_size 32 \
  --dtype bfloat16 \
  --from_resume 1

# =========================
# 合并 SFT（§7.3：四则一起训；max_seq_len=1344 由乘除轨迹长度决定，
# 576 会把乘除样本全部截断）
# =========================
python train_full_sft.py \
  --data_path ../dataset/math_all_v1.jsonl \
  --from_weight pretrain_math \
  --save_weight full_sft_math_all_v1 \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --max_seq_len 1344 \
  --dtype bfloat16 \
  --num_workers 8 \
  --from_resume 1

echo "✅ Training finished successfully."
