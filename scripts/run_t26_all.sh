#!/usr/bin/env bash
# T26 DeepSpeed 训练：单卡 ZeRO-2 最小训练 + checkpoint + 指标。
# 从仓库根目录运行：bash scripts/run_t26_all.sh
set -euo pipefail

echo "== T26 DeepSpeed training（gpp-deepspeed-0.19.5）=="
conda run --no-capture-output -n gpp-deepspeed-0.19.5 deepspeed \
  --num_gpus 1 \
  src/t26_deepspeed_train/train_deepspeed.py \
  --num-samples 8 \
  --max-steps 2 \
  --batch-size 2 \
  --max-length 128 \
  --zero-stage 2 \
  --output-dir caches/t26_output
