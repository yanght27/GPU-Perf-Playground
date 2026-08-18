#!/usr/bin/env bash
# T09 Transpose 朴素版：一键复现五路径。
set -euo pipefail
echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t09_transpose_naive/pytorch_transpose.py
echo "== 2/5 CUDA 两方向（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t09_transpose_naive/cuda/transpose src/t09_transpose_naive/cuda/transpose.cu
./src/t09_transpose_naive/cuda/transpose
echo "== 3/5 Triton（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t09_transpose_naive/triton_transpose.py
echo "== 4/5 cuTile tile=1（gpp-cutile，官方 Transpose.py）=="
conda run --no-capture-output -n gpp-cutile python -I src/t09_transpose_naive/cutile_transpose.py
echo "== 5/5 CuTe DSL（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t09_transpose_naive/cute_transpose.py
