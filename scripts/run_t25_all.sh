#!/usr/bin/env bash
# T25 PyTorch 训练基线：最小训练循环 + checkpoint + 单卡指标。
# 从仓库根目录运行：bash scripts/run_t25_all.sh
set -euo pipefail

echo "== T25 PyTorch training baseline（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I \
  src/t25_pytorch_train/train_baseline.py \
  --num-samples 8 \
  --max-steps 2 \
  --batch-size 2 \
  --max-length 128 \
  --output-dir caches/t25_output
