#!/usr/bin/env bash
# T12 Reduction Warp Shuffle：一键复现五路径（v1.7 五路径齐）。
set -euo pipefail
echo "== 1/5 PyTorch（gpp-core，语义与 fp64 黄金参考）=="
conda run --no-capture-output -n gpp-core python -I src/t12_reduction_shuffle/pytorch_reduction.py
echo "== 2/5 CUDA reduceShfl（系统 nvcc，本 Ticket 核心路径）=="
nvcc -O3 -arch=sm_89 -o src/t12_reduction_shuffle/cuda/reduce_shfl src/t12_reduction_shuffle/cuda/reduce_shfl.cu
./src/t12_reduction_shuffle/cuda/reduce_shfl
echo "== 3/5 Triton tl.sum + 生成 PTX/TTGIR 取证（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t12_reduction_shuffle/triton_reduction.py
echo "== 4/5 cuTile ct.sum（gpp-cutile，shuffle 级 API N/A 的最近官方能力对照）=="
conda run --no-capture-output -n gpp-cutile python -I src/t12_reduction_shuffle/cutile_reduction.py
echo "== 5/5 CuTe warp shuffle 树（gpp-cute，官方 warp_vector_reduce 形态）=="
conda run --no-capture-output -n gpp-cute python -I src/t12_reduction_shuffle/cute_reduction.py
