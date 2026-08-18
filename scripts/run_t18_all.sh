#!/usr/bin/env bash
# T18 FlashAttention CUDA 手工映射：五路径（TR/CT/CUTE 为层级说明或官方能力实测）。
set -euo pipefail
echo "== 1/5 PyTorch SDPA 黄金参考（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t18_flash_attention_cuda/pytorch_attention.py
echo "== 2/5 CUDA FA forward（系统 nvcc，本 Ticket 核心路径）=="
nvcc -O3 -arch=sm_89 -o src/t18_flash_attention_cuda/cuda/flash_attention src/t18_flash_attention_cuda/cuda/flash_attention.cu
./src/t18_flash_attention_cuda/cuda/flash_attention
echo "== 3/5 Triton 层级说明（gpp-core，T17 已完成 -> N/A）=="
conda run --no-capture-output -n gpp-core python -I src/t18_flash_attention_cuda/triton_attention.py
echo "== 4/5 cuTile 官方 FMHA 能力实测（gpp-cutile）=="
conda run --no-capture-output -n gpp-cutile python -I src/t18_flash_attention_cuda/cutile_attention.py
echo "== 5/5 CuTe 官方 flash 能力实测（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t18_flash_attention_cuda/cute_attention.py
echo "== 附：CUDA vs SDPA 同数据直接对比 =="
conda run --no-capture-output -n gpp-core python -I src/t18_flash_attention_cuda/compare_cuda_sdpa.py
