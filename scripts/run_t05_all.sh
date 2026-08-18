#!/usr/bin/env bash
# T05 GEMM Tiling：一键复现五路径正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t05_all.sh
set -euo pipefail

echo "== 1/5 PyTorch 参考（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t05_gemm_tiled/pytorch_gemm.py

echo "== 2/5 CUDA shared-memory tiled（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t05_gemm_tiled/cuda/gemm_tiled \
  src/t05_gemm_tiled/cuda/gemm_tiled.cu
./src/t05_gemm_tiled/cuda/gemm_tiled

echo "== 3/5 Triton tiled（gpp-core，官方 tutorial 03 写法）=="
conda run --no-capture-output -n gpp-core python -I src/t05_gemm_tiled/triton_gemm.py

echo "== 4/5 cuTile tiled（gpp-cutile，官方 MatMul tile=16）=="
conda run --no-capture-output -n gpp-cutile python -I src/t05_gemm_tiled/cutile_gemm.py

echo "== 5/5 CuTe DSL smem tiled（gpp-cute，官方 03_gemm_tiled_smem）=="
conda run --no-capture-output -n gpp-cute python -I src/t05_gemm_tiled/cute_gemm.py
