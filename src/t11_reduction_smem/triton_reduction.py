"""T11 Reduction 共享内存规约 —— 路径 3：Triton（tl.sum）。

官方依据：Triton 语言 tl.sum（台账 S01k）。
结构：每个 program（= 一个 block）归约 BLOCK=1024 个元素，写 1 个部分和；
host 用 fp64 汇总所有部分和。tl.sum 内部的树/寄存器细节由 Triton 编译器生成。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

BLOCK = 1024  # 一个 program 负责的元素数；必须是 2 的幂
TOL = 1e-4


@triton.jit
def sum_kernel(x_ptr, p_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                        # = blockIdx.x：第几个 block
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # 本 block 负责的 1024 个全局下标
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)  # 越界位置填加法单位元 0
    part = tl.sum(x, axis=0)                      # 核心：1024 个数折叠成 1 个标量
    tl.store(p_ptr + pid, part)                   # 每个 block 写一个部分和


def triton_reduce(a):
    N = a.numel()                                 # 元素个数
    grid = triton.cdiv(N, BLOCK)                  # ceil(N/1024)：需要多少个 block
    partial = torch.empty((grid,), device=a.device, dtype=a.dtype)
    sum_kernel[(grid,)](a, partial, N, BLOCK=BLOCK)  # [grid] 是 grid 形状
    return partial.double().sum()                 # host 用 fp64 汇总部分和


def run(N, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(N, device="cuda") * 0.001
    out = triton_reduce(a)
    ref = a.double().sum()
    summarize_error(out, ref, f"triton_t11_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)
    run(999983)
    run(1)
