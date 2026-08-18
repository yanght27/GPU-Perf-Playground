#!/usr/bin/env bash
# T08 GEMM Tensor Core 与 CUTLASS：一键复现五路径 + 官方示例。
# 从仓库根目录运行：bash scripts/run_t08_all.sh
set -euo pipefail

echo "== 1/5 PyTorch fp16/bf16（gpp-core，cuBLAS 后端）=="
conda run --no-capture-output -n gpp-core python -I src/t08_gemm_tensorcore/pytorch_gemm.py

echo "== 2/5 CUDA WMMA bf16（系统 nvcc，官方 cuda-samples 写法）=="
nvcc -O3 -arch=sm_89 -o src/t08_gemm_tensorcore/cuda/wmma_bf16 \
  src/t08_gemm_tensorcore/cuda/wmma_bf16.cu
./src/t08_gemm_tensorcore/cuda/wmma_bf16

echo "== 3/5 Triton fp16 tl.dot（gpp-core，tutorial 03 写法）=="
conda run --no-capture-output -n gpp-core python -I src/t08_gemm_tensorcore/triton_gemm.py

echo "== 4/5 cuTile tf32（gpp-cutile，官方 MatMul 写法）=="
conda run --no-capture-output -n gpp-cutile python -I src/t08_gemm_tensorcore/cutile_gemm.py

echo "== 5/5 CuTe DSL 官方 ampere tensorop + CUTLASS 官方 tf32 示例 =="
conda run --no-capture-output -n gpp-cute python -I src/t08_gemm_tensorcore/cute_gemm.py
/tmp/cutlass-build/examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm \
  --m=1024 --n=1024 --k=1024 --iterations=20
