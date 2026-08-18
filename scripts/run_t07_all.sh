#!/usr/bin/env bash
# T07 GEMM 异步拷贝与流水线：一键复现五路径正确性 + benchmark。
# 从仓库根目录运行：bash scripts/run_t07_all.sh
set -euo pipefail

echo "== 1/2 CUDA cp.async double buffer（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t07_gemm_pipeline/cuda/gemm_pipe \
  src/t07_gemm_pipeline/cuda/gemm_pipe.cu
./src/t07_gemm_pipeline/cuda/gemm_pipe

echo "== 2/2 PyTorch 参考 + Triton num_stages 对比（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t07_gemm_pipeline/pytorch_gemm.py
conda run --no-capture-output -n gpp-core python -I src/t07_gemm_pipeline/triton_gemm.py

echo "== 补充：cuTile latency hint + CuTe cp.async 双缓冲 GEMM =="
conda run --no-capture-output -n gpp-cutile python -I src/t07_gemm_pipeline/cutile_gemm.py
conda run --no-capture-output -n gpp-cute python -I src/t07_gemm_pipeline/cute_gemm.py
