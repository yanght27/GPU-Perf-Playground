#!/usr/bin/env bash
# T19 Qwen/Transformers 基线：固定快照加载、chat template、确定性、prefill/decode 指标。
# 从仓库根目录运行：bash scripts/run_t19_all.sh
set -euo pipefail

echo "== T19 Qwen2.5-0.5B-Instruct baseline（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I \
  src/t19_qwen_baseline/pytorch_qwen_baseline.py
