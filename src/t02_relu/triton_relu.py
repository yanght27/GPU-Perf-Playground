"""T02 ReLU 标量版 —— 路径 3：Triton。
官方依据：Triton tl.maximum / tutorial 01（台账 S01b）。

kernel 骨架沿用官方 tutorial 01-vector-add（v3.7.1）的 program/mask/load/store 写法；
ReLU 语义用官方语言参考的 `tl.maximum(x, 0.0)`（element-wise maximum，Math Ops）。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps


def make_inputs(n: int, device="cuda"):
    i = torch.arange(n, device=device, dtype=torch.float32)
    return torch.where(i % 7 == 0, torch.zeros_like(i), ((i % 97).float() - 48.0) * 0.5)


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                     # 对应 blockIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                     # Triton 的边界保护
    x = tl.load(x_ptr + offsets, mask=mask)
    # tl.maximum 是官方 Math Op：逐元素取 max(x, 0)；等价于 tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def triton_relu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n, BLOCK_SIZE=block_size)
    return output


def main() -> None:
    for tag, N in [("aligned", 1 << 20), ("unaligned", 1_000_003)]:
        a = make_inputs(N)
        ref = torch.clamp(a.double(), min=0.0).float()
        out = triton_relu(a)
        summarize_error(out, ref, f"triton_relu_{tag}")

    N = 1 << 20
    a = make_inputs(N)
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: triton_relu(a), quantiles=[0.5, 0.2, 0.8]
    )
    print(
        f"[triton_relu] median_ms={ms:.4f} min_ms={min_ms:.4f} max_ms={max_ms:.4f} "
        f"effective_gbps={effective_gbps(N, ms / 1e3, accesses=2):.1f}"
    )


if __name__ == "__main__":
    main()
