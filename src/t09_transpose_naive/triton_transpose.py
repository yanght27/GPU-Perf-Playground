"""T09 Transpose 朴素版 —— 路径 3：Triton。

官方依据：Triton 语言 tl.load/tl.store/tl.program_id（tutorial 01 的指针/mask 写法，
台账 S01）；官方无 transpose 专属 tutorial，本实现是最小二维映射。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@triton.jit
def transpose_kernel(x_ptr, y_ptr, H, W):
    pid = tl.program_id(0)
    row = pid // W
    col = pid % W
    if row < H and col < W:
        v = tl.load(x_ptr + row * W + col)   # A[row,col]
        tl.store(y_ptr + col * H + row, v)   # B[col,row]


def triton_transpose(a: torch.Tensor):
    H, W = a.shape
    y = torch.empty((W, H), device=a.device, dtype=a.dtype)
    transpose_kernel[(H * W,)](a, y, H, W)
    return y


def run(W, H):
    torch.manual_seed(0)
    a = torch.rand((H, W), device="cuda")
    ref = a.double().t().float()
    out = triton_transpose(a)
    summarize_error(out, ref, f"triton_transpose_{H}x{W}")


if __name__ == "__main__":
    run(512, 512)
    run(513, 257)
    run(1, 128)
