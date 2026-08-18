"""T04 朴素 GEMM —— 路径 5：CUTLASS CuTe DSL。
官方依据：CuTe DSL fp16_gemm_4_iket.py（台账 S02d）。

官方 CuTe DSL 的 GEMM tutorials 都从分块/tensor-core 起步（fp16_gemm_*.py），
T05 会回到那些教程；T04 先用 CuTe DSL 最基本的 Array 索引 + 动态 Python 循环写
“一个线程一个输出元素”的朴素版本，以对齐“只学索引映射，不引入 shared memory”的范围。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@cute.kernel
def gemm_naive_kernel(
    a_arr: cutlass.Array,
    b_arr: cutlass.Array,
    c_arr: cutlass.Array,
    M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
):
    tx, _, _ = cute.arch.thread_idx()
    _, ty, _ = cute.arch.thread_idx()  # 读 y 维线程号
    bx, _, _ = cute.arch.block_idx()   # x 维给“列”
    _, by, _ = cute.arch.block_idx()   # y 维给“行”
    bdx, _, _ = cute.arch.block_dim()
    _, bdy, _ = cute.arch.block_dim()
    col = bx * bdx + tx
    row = by * bdy + ty
    if row < M and col < N:
        acc = 0.0
        for k in range(K):                      # K 是运行时 Int32：朴素串行累加
            acc = acc + a_arr[row * K + k] * b_arr[k * N + col]
        c_arr[row * N + col] = acc


@cute.jit
def cute_gemm_host(a, b, c, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    block = (16, 16, 1)
    grid = ((N + 15) // 16, (M + 15) // 16, 1)
    gemm_naive_kernel(a, b, c, M, N, K).launch(grid=grid, block=block)


def cute_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    cute_gemm_host(
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(b),
        cute.runtime.from_dlpack(c),
        M, N, K,
    )
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    c = cute_gemm(a, b)
    summarize_error(c, ref, f"cute_naive_{M}x{N}x{K}", tolerance=5e-3)
    import time
    cute_gemm(a, b)  # warmup/JIT
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cute_gemm(a, b)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    print(f"[cute_naive] M={M} N={N} K={K} python_call_ms={ms:.2f}")


def run_boundary():
    torch.manual_seed(0)
    for M, N, K in [(17, 31, 33), (1, 128, 1)]:
        a = torch.rand((M, K), device="cuda") - 0.5
        b = torch.rand((K, N), device="cuda") - 0.5
        ref = (a.double() @ b.double()).float()
        c = cute_gemm(a, b)
        summarize_error(c, ref, f"cute_boundary_{M}x{N}x{K}", tolerance=5e-3)

if __name__ == "__main__":
    run_boundary()
    run(512, 512, 512)
    run(1024, 1024, 1024)
