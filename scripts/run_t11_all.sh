#!/usr/bin/env bash
# T11 Reduction 共享内存规约：一键复现五路径。
set -euo pipefail
echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t11_reduction_smem/pytorch_reduction.py
echo "== 2/5 CUDA reduceSmem（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t11_reduction_smem/cuda/reduce_smem src/t11_reduction_smem/cuda/reduce_smem.cu
./src/t11_reduction_smem/cuda/reduce_smem
echo "== 3/5 Triton tl.sum（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t11_reduction_smem/triton_reduction.py
echo "== 4/5 cuTile ct.sum（gpp-cutile，官方 test_reduction.py 形态）=="
conda run --no-capture-output -n gpp-cutile python -I src/t11_reduction_smem/cutile_reduction.py
echo "== 5/5 CuTe smem tree（gpp-cute，官方 block_smem_reduce 纯 smem 形态）=="
conda run --no-capture-output -n gpp-cute python -I src/t11_reduction_smem/cute_reduction.py
