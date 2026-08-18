"""T06 共享内存优化 —— 路径 5：CUTLASS CuTe DSL。
官方依据：CuTe DSL 03_gemm_tiled_smem.py / swizzle 官方实现（台账 S02f）。

在官方 03_gemm_tiled_smem.py 基础上把 shared tile 从 (TS,TS) 改成 (TS,TS+1)，
用 padding 改变行 stride 以消除 bank conflict（与 CUDA 的 gemmVecPad 思路一致）。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@cute.kernel
def gemm_kernel(
    a: cutlass.Array, b: cutlass.Array, c: cutlass.Array,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
    TS: cutlass.Constexpr[int],
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    a_smem = cutlass.Array(cutlass.Float32, (TS, TS + 1), space=cutlass.AddressSpace.smem)
    b_smem = cutlass.Array(cutlass.Float32, (TS, TS + 1), space=cutlass.AddressSpace.smem)
    row = by * TS + ty
    col = bx * TS + tx
    acc = 0.0
    for bk in range(0, K, TS):
        a_smem[ty, tx] = a[row, bk + tx] if (row < M and bk + tx < K) else 0.0
        b_smem[ty, tx] = b[bk + ty, col] if (bk + ty < K and col < N) else 0.0
        prims.barrier_cta_sync(0)
        for j in range(TS):
            acc += a_smem[ty, j] * b_smem[j, tx]
        prims.barrier_cta_sync(0)
    if row < M and col < N:
        c[row, col] = acc


@cute.jit
def cute_gemm_tiled(a, b, c, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                    TS: cutlass.Constexpr[int]):
    block = (TS, TS, 1)
    grid = ((N + TS - 1) // TS, (M + TS - 1) // TS, 1)
    gemm_kernel(a, b, c, M, N, K, TS).launch(grid=grid, block=block)


def cute_gemm(a, b, TS=32):
    M, K = a.shape; _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    cute_gemm_tiled(cute.runtime.from_dlpack(a), cute.runtime.from_dlpack(b),
                    cute.runtime.from_dlpack(c), M, N, K, TS)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    TS = 16 if max(M, N, K) < 100 else 32
    c = cute_gemm(a, b, TS)
    summarize_error(c, ref, f"cute_t06_pad_{M}x{N}x{K}", tolerance=5e-3)


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
