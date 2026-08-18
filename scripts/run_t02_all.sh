#!/usr/bin/env bash
# T02 ReLU 标量版：一键复现五路径的正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t02_all.sh
set -euo pipefail

echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t02_relu/pytorch_relu.py

echo "== 2/5 CUDA C++（系统 nvcc，sm_89）=="
nvcc -O3 -arch=sm_89 -o src/t02_relu/cuda/relu src/t02_relu/cuda/relu.cu
./src/t02_relu/cuda/relu

echo "== 3/5 Triton（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t02_relu/triton_relu.py

echo "== 4/5 cuTile Python（gpp-cutile）=="
conda run --no-capture-output -n gpp-cutile python -I src/t02_relu/cutile_relu.py

echo "== 5/5 CuTe DSL（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t02_relu/cute_relu.py
