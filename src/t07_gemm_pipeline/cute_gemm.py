"""T07 流水线 —— 路径 5：CUTLASS CuTe DSL 的完整 GEMM cp.async 双缓冲。
官方依据：CuTe DSL cp_async_shared_global.py（台账 S02g）。

官方 `cp_async_shared_global.py` 提供 cp.async 原语；这里把 T06 的 padded smem tiled
GEMM 升级为 STAGES=2 双缓冲：搬运下一块 tile 与计算当前块重叠。
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
def gemm_pipe_kernel(
    a: cutlass.Array, b: cutlass.Array, c: cutlass.Array,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
    TS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int],
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row = by * TS + ty
    col = bx * TS + tx

    As = cutlass.Array(cutlass.Float32, (STAGES, TS, TS + 1), space=cutlass.AddressSpace.smem)
    Bs = cutlass.Array(cutlass.Float32, (STAGES, TS, TS + 1), space=cutlass.AddressSpace.smem)

    acc = 0.0
    ntiles = (K + TS - 1) // TS
    for tile in range(ntiles):
        buf = tile % STAGES
        # 先异步搬下一块
        if tile + 1 < ntiles:
            nb = (tile + 1) % STAGES
            bk2 = (tile + 1) * TS
            a_smem = As.data_ptr() + nb * TS * (TS + 1) + ty * (TS + 1) + tx
            b_smem = Bs.data_ptr() + nb * TS * (TS + 1) + ty * (TS + 1) + tx
            if row < M and bk2 + tx < K:
                prims.cp_async_shared_global(
                    a_smem, a.data_ptr() + row * K + bk2 + tx, 4, "ca")
            else:
                a_smem.store(cutlass.Float32(0.0))
            if bk2 + ty < K and col < N:
                prims.cp_async_shared_global(
                    b_smem, b.data_ptr() + (bk2 + ty) * N + col, 4, "ca")
            else:
                b_smem.store(cutlass.Float32(0.0))
            prims.cp_async_commit_group()

        if tile > 0:
            prims.cp_async_wait_group(STAGES - 1)
        if tile == 0:
            a_smem = As.data_ptr() + ty * (TS + 1) + tx
            b_smem = Bs.data_ptr() + ty * (TS + 1) + tx
            if row < M and tx < K:
                prims.cp_async_shared_global(a_smem, a.data_ptr() + row * K + tx, 4, "ca")
            else:
                a_smem.store(cutlass.Float32(0.0))
            if ty < K and col < N:
                prims.cp_async_shared_global(b_smem, b.data_ptr() + ty * N + col, 4, "ca")
            else:
                b_smem.store(cutlass.Float32(0.0))
            prims.cp_async_commit_group()
            prims.cp_async_wait_group(0)

        prims.barrier_cta_sync(0)
        for j in range(TS):
            acc += As[buf, ty, j] * Bs[buf, j, tx]
        prims.barrier_cta_sync(0)

    if row < M and col < N:
        c[row, col] = acc


@cute.jit
def cute_gemm_pipe(a, b, c, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                   TS: cutlass.Constexpr[int], STAGES: cutlass.Constexpr[int]):
    block = (TS, TS, 1)
    grid = ((N + TS - 1) // TS, (M + TS - 1) // TS, 1)
    gemm_pipe_kernel(a, b, c, M, N, K, TS, STAGES).launch(grid=grid, block=block)


def cute_gemm(a: torch.Tensor, b: torch.Tensor, TS=16, STAGES=2):
    M, K = a.shape; _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    cute_gemm_pipe(cute.runtime.from_dlpack(a), cute.runtime.from_dlpack(b),
                   cute.runtime.from_dlpack(c), M, N, K, TS, STAGES)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    TS = 16 if max(M, N, K) < 100 else 16
    c = cute_gemm(a, b, TS=TS)
    summarize_error(c, ref, f"cute_t07_pipe_{M}x{N}x{K}", tolerance=5e-3)


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
