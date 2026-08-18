#!/usr/bin/env bash
# T15 Attention 朴素前向：五路径复现（cuTile/CuTe 为官方示例层检查 + N/A）。
set -euo pipefail
echo "== 1/5 PyTorch 双参考（gpp-core，eager 公式 + SDPA）=="
conda run --no-capture-output -n gpp-core python -I src/t15_attention_naive/pytorch_attention.py
echo "== 2/5 CUDA attentionNaive（系统 nvcc，本 Ticket 核心路径）=="
nvcc -O3 -arch=sm_89 -o src/t15_attention_naive/cuda/attention_naive src/t15_attention_naive/cuda/attention_naive.cu
./src/t15_attention_naive/cuda/attention_naive
echo "== 3/5 Triton 朴素 qk/softmax/pv（gpp-core，官方 tutorial 06 步骤）=="
conda run --no-capture-output -n gpp-core python -I src/t15_attention_naive/triton_attention.py
echo "== 4/5 cuTile 官方示例层检查（gpp-cutile，flash/online 层 -> N/A）=="
conda run --no-capture-output -n gpp-cutile python -I src/t15_attention_naive/cutile_attention.py
echo "== 5/5 CuTe 官方示例层检查（gpp-cute，flash 层 -> N/A）=="
conda run --no-capture-output -n gpp-cute python -I src/t15_attention_naive/cute_attention.py
