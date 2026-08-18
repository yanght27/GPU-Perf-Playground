"""T01 Vector Add —— 路径 5：CUTLASS CuTe DSL（Python DSL）。
官方依据：CUTLASS CuTe DSL 官方 07_vectorized_array.py（台账 S02a）。

kernel 主体来自官方 CUTLASS commit 564d267e 的官方示例：
examples/python/CuTeDSL/experimental/primitives/tutorial/07_vectorized_array.py
中的 vector_add_kernel。官方示例是单 block、16 元素；这里按同样的官方 API
（cute.arch.block_idx()/block_dim() + 切片语法）扩成 grid>1 的大向量，以参加统一 benchmark。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps

N = 1 << 20
BLOCK_THREADS = 256
VEC = 4  # 每个线程一次处理 4 个 float32（官方示例的“向量化”概念，T03 再展开）


@cute.kernel
def vector_add_kernel(
    a_arr: cutlass.Array,
    b_arr: cutlass.Array,
    c_arr: cutlass.Array,
    vector_size: cutlass.Constexpr[int],
):
    """CuTe DSL 的向量加法。

    官方示例写法：切片语法 c[idx:idx+vector_size] 表示一段连续内存，
    a[...] + b[...] 是元素级加法。我们加了两行 grid/block 索引以支持大向量。
    """
    tx, _, _ = cute.arch.thread_idx()    # 我在 block 里的线程号
    bx, _, _ = cute.arch.block_idx()     # 我在 grid 里的 block 号
    bdx, _, _ = cute.arch.block_dim()    # 每个 block 有多少线程（运行时读取）
    idx = (bx * bdx + tx) * vector_size  # 全局起始下标（和 CUDA 的公式一模一样）
    c_arr[idx:vector_size] = a_arr[idx:vector_size] + b_arr[idx:vector_size]


@cute.jit
def cute_vadd_host(a, b, c, vector_size: cutlass.Constexpr[int]):
    """host 端：从 DLPack 包装的 torch tensor 启动 kernel。"""
    n = a.shape[0]
    block = (BLOCK_THREADS, 1, 1)
    grid = (n // (block[0] * vector_size), 1, 1)  # 总线程数 * 每线程元素数 = N
    vector_add_kernel(a, b, c, vector_size).launch(grid=grid, block=block)


def cute_vadd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.zeros_like(a)
    cute_vadd_host(
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(b),
        cute.runtime.from_dlpack(c),
        VEC,
    )
    return c


def main() -> None:
    torch.manual_seed(0)
    a32 = torch.rand(N, dtype=torch.float32, device="cuda")
    b32 = torch.rand(N, dtype=torch.float32, device="cuda")
    ref = (a32.double() + b32.double()).float()

    out = cute_vadd(a32, b32)
    summarize_error(out, ref, "cute_fp32_vs_fp64")

    ITERS = 100
    for _ in range(10):
        cute_vadd(a32, b32)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        cute_vadd(a32, b32)
    end.record()
    torch.cuda.synchronize()
    call_ms = start.elapsed_time(end) / ITERS
    # 注意：这里量的是“一次 Python 调用从 enqueue 到完成”的时间，包含 CuTe DSL
    # 的 Python/JIT launch 开销，不是纯 kernel 时间。纯 kernel 时间用 NCU 测
    # （证据 docs/evidence/T01/t01-cute-ncu-kernel.ncu-rep：Duration≈35.55us）。
    print(
        f"[cute] python_call_avg_ms={call_ms:.4f} "
        f"(kernel-only 见 NCU 证据；带宽比较以 NCU Duration 为准)"
    )


if __name__ == "__main__":
    main()
