#!/usr/bin/env bash
# T27 ms-swift SFT/LoRA：通过 src wrapper 调用官方 swift sft 最小训练。
# 从仓库根目录运行：bash scripts/run_t27_all.sh
set -euo pipefail

MODEL_DIR="$PWD/assets/modelscope/qwen2.5-0.5b-instruct"

echo "== T27 ms-swift SFT/LoRA（gpp-swift-4.4.3）=="
echo "== model: ${MODEL_DIR}"

conda run --no-capture-output -n gpp-swift-4.4.3 python -I \
  src/t27_ms_swift_train/run_sft.py \
  --model "${MODEL_DIR}" \
  --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#8' \
  --tuner-type lora \
  --torch-dtype bfloat16 \
  --max-steps 2 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --lora-rank 8 \
  --lora-alpha 32 \
  --target-modules all-linear \
  --max-length 512 \
  --output-dir caches/t27_output \
  --seed 42
