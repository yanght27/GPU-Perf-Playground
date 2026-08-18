"""T12 Reduction Warp Shuffle —— 路径 3：Triton（tl.sum + 观察生成代码中的 shuffle）。

官方依据：
  - Triton 语言 tl.sum（台账 S01k）：高层面只写“把一行折叠成一个标量”；
  - Triton 编译器官方源码 `python/triton/compiler/compiler.py` 的
    CompiledKernel.asm（台账 S01l）：compile 返回对象里带 ttir/ttgir/llir/ptx/cubin，
    用 h.asm["ptx"] 观察编译器自动生成的 shfl.sync。
结论：Triton 不暴露 __shfl_down_sync 这种 lane 级指令，但 tl.sum 由编译器
自动 lower 成共享内存跨 warp 规约 + warp 内 shfl.sync.bfly 树（证据见
docs/evidence/T12/t12-triton-ptx.txt）。
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
ROOT = Path(__file__).resolve().parents[2]
EVID = ROOT / "docs" / "evidence" / "T12"


@triton.jit
def sum_kernel(x_ptr, p_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                        # = blockIdx.x：第几个 block
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # 本 block 负责的 1024 个全局下标
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)  # 越界位置填加法单位元 0
    part = tl.sum(x, axis=0)                      # 核心：1024 个数折叠成 1 个标量
    tl.store(p_ptr + pid, part)                   # 每个 block 写一个部分和


def dump_codegen_evidence(a):
    """取 1024 元素的编译产物并落盘：PTX 全文 + shuffle/barrier 相关行。"""
    EVID.mkdir(parents=True, exist_ok=True)
    grid = (triton.cdiv(a.numel(), BLOCK),)
    partial = torch.empty((grid[0],), device=a.device, dtype=a.dtype)
    h = sum_kernel[grid](a, partial, a.numel(), BLOCK=BLOCK)   # 返回 CompiledKernel

    ptx = h.asm["ptx"]
    ttgir = h.asm["ttgir"]
    ptx_lines = ptx.splitlines()
    ptx_hits = [ln for ln in ptx_lines if "shfl" in ln.lower() or "bar.sync" in ln.lower()]
    ttgir_lines = ttgir.splitlines()
    ttgir_hits = [ln for ln in ttgir_lines if "tt.reduce" in ln.lower()
                  or "barrier" in ln.lower()]

    with open(EVID / "t12-triton-ptx.txt", "w") as f:
        f.write("# T12 Triton tl.sum（BLOCK=1024, num_warps=4 默认）生成 PTX 全文\n")
        f.write("# 官方依据：S01k tl.sum、S01l CompiledKernel.asm\n")
        f.write("# 观察：warp 内树规约 = shfl.sync.bfly；跨 warp 交接 = bar.sync + shared。\n\n")
        f.write(ptx)
    with open(EVID / "t12-triton-ttgir.txt", "w") as f:
        f.write("# T12 Triton tl.sum 生成 TTGIR 全文（关键 op：tt.reduce，axis=0）\n")
        f.write("# 官方依据：S01k tl.sum、S01l CompiledKernel.asm\n\n")
        f.write(ttgir)
    with open(EVID / "t12-triton-ptx-hits.txt", "w") as f:
        f.write("# T12 Triton PTX 中与 shuffle/barrier 相关的行\n")
        f.write("# 实际：先 5 步 bfly（每 warp 自己的 4 元素/线程 + 32 lane 折叠），\n")
        f.write("# 再 bar.sync 跨 4 个 warp 交接，最后 2 步 bfly 折叠 4 个 warp 结果。\n")
        f.write("\n".join(ptx_hits) + "\n")
    with open(EVID / "t12-triton-ttgir-hits.txt", "w") as f:
        f.write("# T12 Triton TTGIR 中与 tt.reduce/barrier 相关的行\n")
        f.write("\n".join(ttgir_hits) + "\n")

    print("[triton_t12_codegen] ptx_shfl_or_bar_lines=%d ttgir_reduce_or_barrier_lines=%d"
          % (len(ptx_hits), len(ttgir_hits)))
    return h


def triton_reduce(a):
    N = a.numel()                                 # 元素个数
    grid = triton.cdiv(N, BLOCK)                  # ceil(N/1024)：需要多少个 block
    partial = torch.empty((grid,), device=a.device, dtype=a.dtype)
    sum_kernel[(grid,)](a, partial, N, BLOCK=BLOCK)  # [grid] 是 grid 形状
    return partial.double().sum()                 # host 用 fp64 汇总部分和


def run(N, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(N, device="cuda") * 0.001
    if N == 1048576:
        dump_codegen_evidence(a)                  # 只 dump 一次；三 shape 使用同一份编译产物
    out = triton_reduce(a)
    ref = a.double().sum()
    summarize_error(out, ref, f"triton_t12_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)
    run(999983)
    run(1)
