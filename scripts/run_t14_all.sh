#!/usr/bin/env bash
# T14 Softmax Online/融合版：五路径复现 + T13 基线对照。
set -euo pipefail
echo "== 1/5 PyTorch torch.softmax（gpp-core，fp64 参考 + naive 对照）=="
conda run --no-capture-output -n gpp-core python -I src/t14_softmax_online/pytorch_softmax.py
echo "== 2/5 CUDA softmaxOnline（系统 nvcc，本 Ticket 核心路径）=="
nvcc -O3 -arch=sm_89 -o src/t14_softmax_online/cuda/softmax_online src/t14_softmax_online/cuda/softmax_online.cu
./src/t14_softmax_online/cuda/softmax_online
echo "== 3/5 Triton 官方 fused softmax（gpp-core，tutorial 02 persistent-program 形态）=="
conda run --no-capture-output -n gpp-core python -I src/t14_softmax_online/triton_softmax.py
echo "== 4/5 cuTile fused tile softmax（gpp-cutile，官方 test_softmax per_row 形态）=="
conda run --no-capture-output -n gpp-cutile python -I src/t14_softmax_online/cutile_softmax.py
echo "== 5/5 CuTe official Kernel 5 Online Naive（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t14_softmax_online/cute_softmax.py
echo "== 6/6 T13 CUDA 3-pass 基线对照（系统 nvcc，用于同 shape 对比）=="
nvcc -O3 -arch=sm_89 -o src/t13_softmax_naive/cuda/softmax_naive src/t13_softmax_naive/cuda/softmax_naive.cu
./src/t13_softmax_naive/cuda/softmax_naive
