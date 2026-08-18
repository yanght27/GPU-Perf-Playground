#!/usr/bin/env bash
# T13 Softmax 朴素 3-pass：一键复现五路径。
set -euo pipefail
echo "== 1/5 PyTorch naive_softmax（gpp-core，官方 tutorial 02 形态）=="
conda run --no-capture-output -n gpp-core python -I src/t13_softmax_naive/pytorch_softmax.py
echo "== 2/5 CUDA softmaxNaive（系统 nvcc，本 Ticket 核心路径）=="
nvcc -O3 -arch=sm_89 -o src/t13_softmax_naive/cuda/softmax_naive src/t13_softmax_naive/cuda/softmax_naive.cu
./src/t13_softmax_naive/cuda/softmax_naive
echo "== 3/5 Triton 3-pass（gpp-core，官方 tutorial 02 mask/BLOCK 技巧）=="
conda run --no-capture-output -n gpp-core python -I src/t13_softmax_naive/triton_softmax.py
echo "== 4/5 cuTile softmax_per_row（gpp-cutile，官方 test_softmax 形态）=="
conda run --no-capture-output -n gpp-cutile python -I src/t13_softmax_naive/cutile_softmax.py
echo "== 5/5 CuTe tutorial 06 kernel2（gpp-cute，官方 block smem 3-pass）=="
conda run --no-capture-output -n gpp-cute python -I src/t13_softmax_naive/cute_softmax.py
