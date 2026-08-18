"""T05 GEMM Tiling —— 路径 5：CUTLASS CuTe DSL。
官方依据：CuTe DSL 03_gemm_tiled_smem.py（台账 S02e）。

直接对齐官方教程 03_gemm_tiled_smem.py（commit 564d267e）：
- cutlass.Array(space=cutlass.AddressSpace.smem) 分配共享内存 tile；
- prims.barrier_cta_sync(0) 做 block 内同步（对应 CUDA __syncthreads）。
官方教程假设 M/N/K 可被 TS 整除；本文件加了 zero-fill 边界保护以通过非整除测试。
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
    a: cutlass.Array,
    b: cutlass.Array,
    c: cutlass.Array,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
    TS: cutlass.Constexpr[int],
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # 官方写法：显式分配 shared memory tile
    a_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)
    b_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)

    row = by * TS + ty
    col = bx * TS + tx
    acc = 0.0
    for bk in range(0, K, TS):
        # zero-fill：非整除边界时，越界位置给 0，不影响累加结果
        a_smem[ty, tx] = a[row, bk + tx] if (row < M and bk + tx < K) else 0.0
        b_smem[ty, tx] = b[bk + ty, col] if (bk + ty < K and col < N) else 0.0

        prims.barrier_cta_sync(0)     # 对应 __syncthreads()：搬完再算

        for j in range(TS):
            acc += a_smem[ty, j] * b_smem[j, tx]

        prims.barrier_cta_sync(0)     # 算完再覆盖下一 tile

    if row < M and col < N:
        c[row, col] = acc


@cute.jit
def cute_gemm_tiled(a, b, c, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                    TS: cutlass.Constexpr[int]):
    block = (TS, TS, 1)
    grid = ((N + TS - 1) // TS, (M + TS - 1) // TS, 1)
    gemm_kernel(a, b, c, M, N, K, TS).launch(grid=grid, block=block)


def cute_gemm(a: torch.Tensor, b: torch.Tensor, TS=32) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    cute_gemm_tiled(
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(b),
        cute.runtime.from_dlpack(c),
        M, N, K, TS,
    )
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    c = cute_gemm(a, b, TS=16 if max(M, N, K) < 100 else 32)
    summarize_error(c, ref, f"cute_tiled_{M}x{N}x{K}", tolerance=5e-3)


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
