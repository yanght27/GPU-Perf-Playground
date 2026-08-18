#!/usr/bin/env bash
# T10 Transpose Tile：一键复现五路径。
set -euo pipefail
echo "== 1/5 PyTorch（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t10_transpose_tiled/pytorch_transpose.py
echo "== 2/5 CUDA tile nopad/pad（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t10_transpose_tiled/cuda/transpose_tiled src/t10_transpose_tiled/cuda/transpose_tiled.cu
./src/t10_transpose_tiled/cuda/transpose_tiled
echo "== 3/5 Triton tile+tl.trans（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t10_transpose_tiled/triton_transpose.py
echo "== 4/5 cuTile tile=32（gpp-cutile，官方 Transpose.py）=="
conda run --no-capture-output -n gpp-cutile python -I src/t10_transpose_tiled/cutile_transpose.py
echo "== 5/5 CuTe smem tile+padding（gpp-cute）=="
conda run --no-capture-output -n gpp-cute python -I src/t10_transpose_tiled/cute_transpose.py
