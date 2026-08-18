#!/usr/bin/env bash
# T03 ReLU 向量化版：一键复现五路径正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t03_all.sh
set -euo pipefail

echo "== 1/5 PyTorch（gpp-core，框架已自动向量化）=="
conda run --no-capture-output -n gpp-core python -I src/t03_relu/pytorch_relu.py

echo "== 2/5 CUDA C++ float4（系统 nvcc，sm_89）=="
nvcc -O3 -arch=sm_89 -o src/t03_relu/cuda/relu_vec src/t03_relu/cuda/relu_vec.cu
./src/t03_relu/cuda/relu_vec

echo "== 3/5 Triton（gpp-core，PTX 显示 ld/st.global.v4）=="
conda run --no-capture-output -n gpp-core python -I src/t03_relu/triton_relu.py

echo "== 4/5 cuTile Python（gpp-cutile，tile load/store 自动向量化）=="
conda run --no-capture-output -n gpp-cutile python -I src/t03_relu/cutile_relu.py

echo "== 5/5 CuTe DSL（gpp-cute，官方切片语法向量化 load）=="
conda run --no-capture-output -n gpp-cute python -I src/t03_relu/cute_relu.py
