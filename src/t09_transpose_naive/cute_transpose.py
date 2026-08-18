"""T09 Transpose 朴素版 —— 路径 5：CUTLASS CuTe DSL。

官方依据：CuTe DSL 03_gemm_tiled_smem / 07_vectorized_array 的 thread/block 索引写法
（台账 S02）。官方无 transpose 专属 tutorial，本实现是最小二维映射。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@cute.kernel
def transpose_kernel(a: cutlass.Array, b: cutlass.Array,
                     W: cutlass.Int32, H: cutlass.Int32):
    tx, _, _ = cute.arch.thread_idx()
    _, ty, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    _, by, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    _, bdy, _ = cute.arch.block_dim()
    col = bx * bdx + tx
    row = by * bdy + ty
    if row < H and col < W:
        b[col, row] = a[row, col]


@cute.jit
def cute_transpose_host(a, b, W: cutlass.Int32, H: cutlass.Int32):
    block = (16, 16, 1)
    grid = ((W + 15) // 16, (H + 15) // 16, 1)
    transpose_kernel(a, b, W, H).launch(grid=grid, block=block)


def cute_transpose(a: torch.Tensor):
    H, W = a.shape
    b = torch.zeros((W, H), device=a.device, dtype=a.dtype)
    cute_transpose_host(cute.runtime.from_dlpack(a), cute.runtime.from_dlpack(b), W, H)
    return b


def run(W, H):
    torch.manual_seed(0)
    a = torch.rand((H, W), device="cuda")
    ref = a.double().t().float()
    out = cute_transpose(a)
    summarize_error(out, ref, f"cute_transpose_{H}x{W}")


if __name__ == "__main__":
    run(512, 512)
    run(513, 257)
    run(1, 128)
