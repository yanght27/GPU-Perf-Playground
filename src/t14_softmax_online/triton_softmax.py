"""T14 Softmax Online/融合版 —— 路径 3：Triton（官方 tutorial 02 fused softmax）。

官方依据：Triton 官方 tutorial 02-fused-softmax 的 softmax_kernel 与其 host 的
persistent-program 配置方法（台账 S01m）。
本文件保留官方 kernel 主体（一次 tl.load、块内 max/exp/sum/normalize）与官方
occupancy 计算；global 访问 = 1 读 + 1 写（T13 为 3 读 + 1 写）。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.runtime import driver

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5
_cache = {}


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))  # 唯一一次 global 读
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def _compiled(C):
    if C not in _cache:
        BLOCK = triton.next_power_of_2(C)
        num_warps = 8
        num_stages = 2
        dummy = torch.empty(1024, C, device="cuda")  # 用真实 stride 编译，避免错误的 stride 特化
        kernel = softmax_kernel.warmup(
            dummy, dummy, C, C, 1024, C, BLOCK_SIZE=BLOCK, num_stages=num_stages,
            num_warps=num_warps, grid=(1,))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        props = driver.active.utils.get_device_properties(0)
        NUM_SM = props["multiprocessor_count"]
        NUM_REGS = props["max_num_regs"]
        SIZE_SMEM = props["max_shared_mem"]
        occupancy = min(NUM_REGS // (n_regs * 32 * num_warps),
                        SIZE_SMEM // size_smem)
        num_programs = min(NUM_SM * occupancy, 1024)  # 官方教程同式；主 shape 行数即上限
        _cache[C] = (kernel, num_programs, BLOCK, num_warps, num_stages)
    return _cache[C]


def triton_softmax(x):
    R, C = x.shape
    y = torch.empty_like(x)
    kernel, num_programs, BLOCK, nw, ns = _compiled(C)
    num_programs = min(num_programs, R)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), R, C, BLOCK, ns)
    return y


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
    return [(a, "R=1024_C=4096"), (b, "R=37_C=999_unaligned"), (c, "R=1_C=1")]


def run_case(x, name):
    if name == "R=1024_C=4096":
        for _ in range(10):
            triton_softmax(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            triton_softmax(x)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 50
        bytes_moved = x.numel() * 4 * 2          # 1 读 + 1 写
        print(f"[triton_t14_timing] R=1024 C=4096 avg_ms={ms:.4f} effective_GBps={bytes_moved/ms/1e6:.2f}")
    out = triton_softmax(x)
    ref = fp64_ref(x)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"triton_t14_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, name in make_cases():
        run_case(x, name)
