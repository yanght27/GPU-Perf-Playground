#!/usr/bin/env bash
# T01 Vector Add：一键复现五条路径的正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t01_all.sh
set -euo pipefail

echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t01_vector_add/pytorch_vadd.py

echo "== 2/5 CUDA C++（系统 nvcc，sm_89）=="
nvcc -O3 -arch=sm_89 -o src/t01_vector_add/cuda/vector_add src/t01_vector_add/cuda/vector_add.cu
./src/t01_vector_add/cuda/vector_add

echo "== 3/5 Triton（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t01_vector_add/triton_vadd.py

echo "== 4/5 cuTile Python（gpp-cutile）=="
conda run --no-capture-output -n gpp-cutile python -I src/t01_vector_add/cutile_vadd.py

echo "== 5/5 CuTe DSL（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t01_vector_add/cute_vadd.py
