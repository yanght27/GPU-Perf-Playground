#!/usr/bin/env bash
# T06 GEMM 共享内存优化：一键复现五路径正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t06_all.sh
set -euo pipefail

echo "== 1/3 CUDA 四档对比（base/pad/vec4/vecpad）=="
nvcc -O3 -arch=sm_89 -o src/t06_gemm_smem/cuda/gemm_smem \
  src/t06_gemm_smem/cuda/gemm_smem.cu
./src/t06_gemm_smem/cuda/gemm_smem

echo "== 2/3 PyTorch 参考 + Triton block 配置对比（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t06_gemm_smem/pytorch_gemm.py
conda run --no-capture-output -n gpp-core python -I src/t06_gemm_smem/triton_gemm.py

echo "== 3/3 cuTile 编译器对照 + CuTe DSL padding 版 =="
conda run --no-capture-output -n gpp-cutile python -I src/t06_gemm_smem/cutile_gemm.py
conda run --no-capture-output -n gpp-cute python -I src/t06_gemm_smem/cute_gemm.py
