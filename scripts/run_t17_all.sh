#!/usr/bin/env bash
# T17 FlashAttention（Triton 官方 tutorial 版）：五路径（CUDA/CT/CUTE 为 N/A 检查）。
set -euo pipefail
echo "== 1/5 PyTorch SDPA 黄金参考（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t17_flash_attention_triton/pytorch_attention.py
echo "== 2/5 Triton FA forward + 与 T15 朴素版对比（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t17_flash_attention_triton/triton_flash_attention.py
echo "== 3/5 CUDA FA 层级说明（N/A 检查，T18 专用 -> N/A）=="
python3 src/t17_flash_attention_triton/cuda_attention.py
echo "== 4/5 cuTile 官方 flash 层检查（gpp-cutile，框架层 -> N/A）=="
conda run --no-capture-output -n gpp-cutile python -I src/t17_flash_attention_triton/cutile_attention.py
echo "== 5/5 CuTe 官方 flash 层检查（gpp-cute，框架层 -> N/A）=="
conda run --no-capture-output -n gpp-cute python -I src/t17_flash_attention_triton/cute_attention.py
