"""T03 ReLU 向量化版 —— 路径 5：CUTLASS CuTe DSL。
官方依据：CUTLASS CuTe DSL 07_vectorized_array.py（台账 S02c）。

官方 vectorized_array.py 支持切片语法做“向量化 load + 元素级算术 + 向量化 store”，
但当前官方 DSL 尚不支持对 Vector 做 max/where（本 Ticket 已实测记录），所以 CuTe 路径
采用官方能力范围内的“向量化 load（v4）+ lane 标量 ReLU + 标量 store”，并在讲义中
如实标注：这是官方 API 能力边界，不是偷懒。
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
def relu_vec_kernel(
    a_arr: cutlass.Array,
    c_arr: cutlass.Array,
    n_elements: cutlass.Int32,
    vector_size: cutlass.Constexpr[int],
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = (bx * bdx + tx) * vector_size
    # 官方切片语法：一次 16B 的向量化 load（v4）
    v = a_arr[idx:vector_size]
    for i in range(vector_size):
        if idx + i < n_elements:            # 尾部边界保护
            val = v[i]
            # NaN 传播：NaN != NaN 为真；否则按 ReLU 分支
            c_arr[idx + i] = val if val != val else (val if val > 0.0 else 0.0)


@cute.jit
def cute_relu_host(a, c, n_elements: cutlass.Int32, vector_size: cutlass.Constexpr[int]):
    block = (BLOCK_THREADS, 1, 1)
    grid = ((n_elements + block[0] * vector_size - 1) // (block[0] * vector_size), 1, 1)
    relu_vec_kernel(a, c, n_elements, vector_size).launch(grid=grid, block=block)


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
        summarize_error(out, ref, f"cute_relu_vec_{tag}")

    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1e38, -1e38],
                     device="cuda", dtype=torch.float32)
    ref_ext = torch.clamp(x.double(), min=0.0).float()
    out_ext = cute_relu(x)
    ok = torch.allclose(out_ext, ref_ext, rtol=0.0, atol=1e-5, equal_nan=True)
    print(f"[cute_relu_extreme] {'CORRECT_PASS' if ok else 'CORRECT_FAIL'} out={out_ext.tolist()}")

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
    print(f"[cute_relu_vec] python_call_avg_ms={call_ms:.4f} (kernel-only 以 NCU 证据为准)")


if __name__ == "__main__":
    main()
