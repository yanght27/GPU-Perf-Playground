#!/usr/bin/env bash
# T16 KV Cache：五路径（cuTile/CuTe 为官方能力检查 + N/A）。
set -euo pipefail
echo "== 1/5 PyTorch DynamicCache 语义参考（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t16_kv_cache/pytorch_kv_cache.py
echo "== 2/5 CUDA 投影/append/decode（系统 nvcc）=="
nvcc -O3 -arch=sm_89 -o src/t16_kv_cache/cuda/kv_cache src/t16_kv_cache/cuda/kv_cache.cu
./src/t16_kv_cache/cuda/kv_cache
echo "== 3/5 Triton append + decode（gpp-core）=="
conda run --no-capture-output -n gpp-core python -I src/t16_kv_cache/triton_kv_cache.py
echo "== 4/5 cuTile KV-Cache 官方能力检查（gpp-cutile，无官方示例 -> N/A）=="
conda run --no-capture-output -n gpp-cutile python -I src/t16_kv_cache/cutile_kv_cache.py
echo "== 5/5 CuTe KV-Cache 官方能力检查（gpp-cute，仅 Blackwell MLA -> N/A）=="
conda run --no-capture-output -n gpp-cute python -I src/t16_kv_cache/cute_kv_cache.py
