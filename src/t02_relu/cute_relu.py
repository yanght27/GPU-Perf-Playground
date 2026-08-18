"""T02 ReLU 标量版 —— 路径 5：CUTLASS CuTe DSL。
官方依据：CUTLASS CuTe DSL 07_vectorized_array.py（台账 S02b）。

骨架来自官方示例 07_vectorized_array.py（CUTLASS 564d267e）的 thread/block 索引写法；
本轮刻意用“每线程循环 4 个元素 + if 边界保护”的标量形式，以对齐 T02 学习变量。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

N = 1 << 20
VEC = 4
BLOCK_THREADS = 256


def make_inputs(n: int):
    i = torch.arange(n, device="cuda", dtype=torch.float32)
    return torch.where(i % 7 == 0, torch.zeros_like(i), ((i % 97).float() - 48.0) * 0.5)


@cute.kernel
def relu_kernel(
    a_arr: cutlass.Array,
    c_arr: cutlass.Array,
    n_elements: cutlass.Int32,
    vector_size: cutlass.Constexpr[int],
):
    """标量 ReLU：每个元素独立判断一次；不合并访存（T03 再向量化）。"""
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = (bx * bdx + tx) * vector_size
    for i in range(vector_size):
        if idx + i < n_elements:                # 和 CUDA 的 if(i<n) 一样：边界保护
            val = a_arr[idx + i]
            c_arr[idx + i] = val if val != val else (val if val > 0.0 else 0.0)


@cute.jit
def cute_relu_host(a, c, n_elements: cutlass.Int32, vector_size: cutlass.Constexpr[int]):
    block = (BLOCK_THREADS, 1, 1)
    # ceil 除法：总线程容量 = block * vector_size
    grid = ((n_elements + block[0] * vector_size - 1) // (block[0] * vector_size), 1, 1)
    relu_kernel(a, c, n_elements, vector_size).launch(grid=grid, block=block)


def cute_relu(a: torch.Tensor) -> torch.Tensor:
    c = torch.zeros_like(a)
    cute_relu_host(
        cute.runtime.from_dlpack(a),
        cute.runtime.from_dlpack(c),
        a.numel(),
        VEC,
    )
    return c


def main() -> None:
    for tag, n in [("aligned", N), ("unaligned", 1_000_003)]:
        a = make_inputs(n)
        ref = torch.clamp(a.double(), min=0.0).float()
        out = cute_relu(a)
        summarize_error(out, ref, f"cute_relu_{tag}")

    a = make_inputs(N)
    ITERS = 100
    for _ in range(10):
        cute_relu(a)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        cute_relu(a)
    end.record()
    torch.cuda.synchronize()
    call_ms = start.elapsed_time(end) / ITERS
    # 同 T01：这是“Python 调用级”时间；纯 kernel 时间以 NCU Duration 为准
    print(f"[cute_relu] python_call_avg_ms={call_ms:.4f} (kernel-only 以 NCU 证据为准)")


if __name__ == "__main__":
    main()
