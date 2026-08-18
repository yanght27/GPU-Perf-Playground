"""T13 Softmax 朴素 3-pass —— 路径 5：CUTLASS CuTe DSL（官方 tutorial 06 kernel2）。

官方依据：CUTLASS CuTe DSL 官方 tutorial 06_softmax.py 的 Kernel 2
“Block-level with Shared Memory Reductions”（台账 S02n）：
  - 一行一个 block；KERNEL2_BLOCK_SIZE=128；
  - pass1 shared max 树 -> pass2 写 exp 分子 -> pass3 shared sum 树 -> 归一化写回。
本文件与官方 kernel2 逐行同构，只把测试形状/用例换成本 Ticket 的统一门禁。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

_KERNEL2_BLOCK_SIZE = 128
TOL = 1e-5


@cute.kernel
def softmax_forward_kernel2(
    inp_arr: cutlass.Array,
    out_arr: cutlass.Array,
    N: cutlass.Constexpr,
    C: cutlass.Constexpr,
):
    """官方 Kernel 2：block 级 shared memory 规约的 3-pass softmax。"""
    shared = cutlass.Array(
        cutlass.Float32, _KERNEL2_BLOCK_SIZE, space=cutlass.AddressSpace.smem
    )

    idx, _, _ = cute.arch.block_idx()            # 行号
    tid, _, _ = cute.arch.thread_idx()

    # pass 1：求行最大值
    maxval = -3.4028235e38
    for i in cutlass.range_constexpr(0, C, _KERNEL2_BLOCK_SIZE):
        col = i + tid
        if col < C:
            maxval = cute.math.max(maxval, inp_arr[idx, col])

    shared[tid] = maxval
    for stride in [64, 32, 16, 8, 4, 2, 1]:
        prims.barrier_cta_sync(0)
        if stride < _KERNEL2_BLOCK_SIZE and tid < stride:
            shared[tid] = cute.math.max(shared[tid], shared[tid + stride])

    prims.barrier_cta_sync(0)
    offset = shared[0]

    # pass 2：写 exp 分子（官方用 fastmath，与 fp64 参考的容差 1e-5 已验证）
    for i in cutlass.range_constexpr(0, C, _KERNEL2_BLOCK_SIZE):
        col = i + tid
        if col < C:
            exp_val = cute.math.exp(inp_arr[idx, col] - offset, fastmath=True)
            out_arr[idx, col] = exp_val

    prims.barrier_cta_sync(0)

    # pass 3 前段：对输出矩阵做行求和
    sumval = 0.0
    for i in cutlass.range_constexpr(0, C, _KERNEL2_BLOCK_SIZE):
        col = i + tid
        if col < C:
            sumval = sumval + out_arr[idx, col]

    shared[tid] = sumval
    for stride in [64, 32, 16, 8, 4, 2, 1]:
        prims.barrier_cta_sync(0)
        if stride < _KERNEL2_BLOCK_SIZE and tid < stride:
            shared[tid] = shared[tid] + shared[tid + stride]

    prims.barrier_cta_sync(0)
    total_sum = shared[0]

    # pass 3：归一化写回
    for i in cutlass.range_constexpr(0, C, _KERNEL2_BLOCK_SIZE):
        col = i + tid
        if col < C:
            out_arr[idx, col] = out_arr[idx, col] / total_sum


@cute.jit
def softmax_host(
    inp_tensor: cute.Tensor,
    out_tensor: cute.Tensor,
    N: cutlass.Constexpr,
    C: cutlass.Constexpr,
):
    softmax_forward_kernel2(inp_tensor, out_tensor, N, C).launch(
        grid=(N, 1, 1),
        block=(_KERNEL2_BLOCK_SIZE, 1, 1),
    )


def fp64_ref(x):
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
    return [(a, 1024, 4096, "R=1024_C=4096"),
            (b, 37, 999, "R=37_C=999_unaligned"),
            (c, 1, 1, "R=1_C=1")]


def cute_softmax(x, N, C):
    x = x.contiguous()
    out = torch.empty_like(x)
    inp_t = cute.runtime.from_dlpack(x, assumed_align=16)
    out_t = cute.runtime.from_dlpack(out, assumed_align=16)
    softmax_host(inp_t, out_t, N, C)
    torch.cuda.synchronize()
    return out


def run_case(x, N, C, name):
    out = cute_softmax(x, N, C)
    ref = fp64_ref(x)
    summarize_error(out, ref, f"cute_t13_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, N, C, name in make_cases():
        run_case(x, N, C, name)
