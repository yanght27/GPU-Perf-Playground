"""T13 Softmax 朴素 3-pass —— 路径 3：Triton（block 级行归约）。

官方依据：Triton 官方 tutorial 02-fused-softmax（台账 S01m）：
  - 每 program 负责一行，BLOCK_SIZE = next_power_of_2(n_cols)；
  - mask + other 处理非 2 的幂行宽（other 对 max 用 -inf、对 sum 相当于 0）；
  - tl.max/tl.exp/tl.sum 的行归约语义。
本 Ticket 把官方 fused 单 load 教程改写为“3 次 tl.load 的 3-pass 基线”，用于对照
CUDA 的 3 次 global 读 + 1 次写；优化版（单 load）留给 T14 Online/融合。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5


@triton.jit
def softmax_naive_kernel(x_ptr, y_ptr, C, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # = blockIdx.x：一行一个 program
    offs = pid * C + tl.arange(0, BLOCK)         # 本行第 0..BLOCK-1 列
    mask = tl.arange(0, BLOCK) < C               # 处理 C 不是 2 的幂的边界

    # pass 1：读第 1 遍，求行最大值 m（数值稳定用）
    row = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    m = tl.max(row, axis=0)

    # pass 2：读第 2 遍，求 exp(x-m) 与行和
    row2 = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    e = tl.exp(row2 - m)
    denom = tl.sum(e, axis=0)

    # pass 3：读第 3 遍，写归一化结果
    row3 = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    out = tl.exp(row3 - m) / denom
    tl.store(y_ptr + offs, out, mask=mask)


def fp64_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.double(), dim=1)


def make_cases():
    torch.manual_seed(0)
    a = torch.rand(1024, 4096, device="cuda") * 10.0 - 5.0
    a[0].fill_(7.0)
    a[1, 0], a[1, 1] = -1000.0, 1000.0
    a[2, 0], a[2, 1] = 1000.0, -1000.0

    b = torch.rand(37, 999, device="cuda") * 10.0 - 5.0
    b[0].fill_(-7.0)
    b[1, 0], b[1, 1] = 1000.0, -1000.0
    b[2, 0], b[2, 1] = -1000.0, 1000.0

    c = torch.full((1, 1), 1000.0, device="cuda")
    return [(a, "R=1024_C=4096"), (b, "R=37_C=999_unaligned"), (c, "R=1_C=1")]


def triton_softmax(x):
    R, C = x.shape
    BLOCK = triton.next_power_of_2(C)            # 官方 tutorial 的 row-in-block 技巧
    y = torch.empty_like(x)
    softmax_naive_kernel[(R,)](x, y, C, BLOCK=BLOCK)
    return y


def run_case(x, name):
    if name == "R=1024_C=4096":                   # 只给主 shape 计时，其余只验证正确性
        for _ in range(10):
            triton_softmax(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            triton_softmax(x)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 50
        bytes_moved = x.numel() * 4 * 4          # 3 次读 + 1 次写
        print(f"[triton_t13_timing] R=1024 C=4096 avg_ms={ms:.4f} effective_GBps={bytes_moved/ms/1e6:.2f}")
    out = triton_softmax(x)
    ref = fp64_ref(x)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"triton_t13_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, name in make_cases():
        run_case(x, name)
