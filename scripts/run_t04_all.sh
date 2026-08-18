#!/usr/bin/env bash
# T04 朴素 GEMM + cuBLAS：一键复现五路径正确性与 benchmark。
# 从仓库根目录运行：bash scripts/run_t04_all.sh
set -euo pipefail

echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t04_gemm_naive/pytorch_gemm.py

echo "== 2/5 CUDA 朴素 + cuBLAS（系统 nvcc/cuBLAS）=="
nvcc -O3 -arch=sm_89 -o src/t04_gemm_naive/cuda/gemm_naive \
  src/t04_gemm_naive/cuda/gemm_naive.cu -lcublas
./src/t04_gemm_naive/cuda/gemm_naive 17 31 33
./src/t04_gemm_naive/cuda/gemm_naive 1 128 1
./src/t04_gemm_naive/cuda/gemm_naive 512 512 512
./src/t04_gemm_naive/cuda/gemm_naive 1024 1024 1024

echo "== 3/5 Triton 朴素（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t04_gemm_naive/triton_gemm.py

echo "== 4/5 cuTile 朴素（gpp-cutile，tile=1×1×1）=="
conda run --no-capture-output -n gpp-cutile python -I src/t04_gemm_naive/cutile_gemm.py

echo "== 5/5 CuTe DSL 朴素（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t04_gemm_naive/cute_gemm.py
