"""T14 Softmax Online/融合版 —— 路径 5：CUTLASS CuTe DSL（官方 tutorial 06 Kernel 5）。

官方依据：CUTLASS CuTe DSL 官方 tutorial 06_softmax.py 的 Kernel 5
“Online Naive: one thread per row”（台账 S02n）。
官方 kernel 用 running max/sum 的 rescale 更新式单遍扫行，再第二遍写输出
（2 读 + 1 写）；T13 的 kernel2 是 3-pass（3 读 + 1 写）。
本文件与官方 Kernel 5 逐行同构，只换统一门禁矩阵。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5
_BLOCK_SIZE = 128


@cute.kernel
def softmax_forward_online_kernel1(
    inp_arr: cutlass.Array,
    out_arr: cutlass.Array,
    N: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()

    i = bx * bdx + tx
    C = inp_arr.shape[1]

    if i < N:
        maxval = -3.4028235e38
        sumval = 0.0
        # online 单遍：发现更大的值就 rescale 旧和，否则累加 exp(v-m)
        for j in range(C):
            current_val = inp_arr[i, j]
            maxval_prev = maxval
            if current_val > maxval:
                maxval = current_val
                sumval = sumval * cute.math.exp(maxval_prev - maxval, fastmath=True) + 1.0
            else:
                sumval = sumval + cute.math.exp(current_val - maxval, fastmath=True)

        for j in range(C):
            out_arr[i, j] = cute.math.exp(inp_arr[i, j] - maxval, fastmath=True) / sumval


@cute.jit
def softmax_host(
    inp_tensor: cute.Tensor,
    out_tensor: cute.Tensor,
    N: cutlass.Constexpr,
):
    grid_size = (N + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    softmax_forward_online_kernel1(inp_tensor, out_tensor, N).launch(
        grid=(grid_size, 1, 1),
        block=(_BLOCK_SIZE, 1, 1),
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
    return [(a, 1024, "R=1024_C=4096"), (b, 37, "R=37_C=999_unaligned"), (c, 1, "R=1_C=1")]


def cute_online_softmax(x, N):
    x = x.contiguous()
    out = torch.empty_like(x)
    softmax_host(cute.runtime.from_dlpack(x, assumed_align=16),
                 cute.runtime.from_dlpack(out, assumed_align=16), N)
    torch.cuda.synchronize()
    return out


def run_case(x, N, name):
    out = cute_online_softmax(x, N)
    ref = fp64_ref(x)
    summarize_error(out, ref, f"cute_t14_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, N, name in make_cases():
        run_case(x, N, name)
