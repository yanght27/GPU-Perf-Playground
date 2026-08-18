"""T11 Reduction 共享内存规约 —— 路径 5：CUTLASS CuTe DSL（纯 smem 树规约）。

官方依据：CuTe DSL 官方 block_smem_reduce.py 的 smem/barrier 原语（台账 S02l）。
官方示例把 warp shuffle 与 smem 组合使用；T11 为对准“block 同步 + shared 协作”，
只取官方示例的 smem + barrier_cta_sync 部分，128 线程的 7 轮树规约逐轮手写展开（shuffle 归 T12）。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TILE = 128  # 一个 block = 128 线程 = 4 个 warp
TOL = 1e-4


@cute.kernel
def reduce_kernel(a: cutlass.Array, partial: cutlass.Array):
    tx, _, _ = cute.arch.thread_idx()            # 线程编号 (x,y,z)，只取 x
    bx, _, _ = cute.arch.block_idx()             # block 编号 (x,y,z)，只取 x
    sdata = cutlass.Array(cutlass.Float32, (TILE,),
                          space=cutlass.AddressSpace.smem)  # 128 个 float 的 shared 黑板
    sdata[tx] = a[bx, tx]                        # 第 bx 行第 tx 列：一个线程写一格
    prims.barrier_cta_sync(0)                    # = __syncthreads()：黑板写满才能读
    if tx < 64:
        sdata[tx] = sdata[tx] + sdata[tx + 64]   # 轮 1：128 个数并成 64 组
    prims.barrier_cta_sync(0)
    if tx < 32:
        sdata[tx] = sdata[tx] + sdata[tx + 32]   # 轮 2：64 → 32
    prims.barrier_cta_sync(0)
    if tx < 16:
        sdata[tx] = sdata[tx] + sdata[tx + 16]   # 轮 3：32 → 16
    prims.barrier_cta_sync(0)
    if tx < 8:
        sdata[tx] = sdata[tx] + sdata[tx + 8]    # 轮 4：16 → 8
    prims.barrier_cta_sync(0)
    if tx < 4:
        sdata[tx] = sdata[tx] + sdata[tx + 4]    # 轮 5：8 → 4
    prims.barrier_cta_sync(0)
    if tx < 2:
        sdata[tx] = sdata[tx] + sdata[tx + 2]    # 轮 6：4 → 2
    prims.barrier_cta_sync(0)
    if tx < 1:
        sdata[tx] = sdata[tx] + sdata[tx + 1]    # 轮 7：2 → 1
    prims.barrier_cta_sync(0)
    if tx == 0:
        partial[bx] = sdata[0]                   # 树顶：每个 block 写一个部分和


@cute.jit
def reduce_host(a, partial, rows: cutlass.Int32):
    reduce_kernel(a, partial).launch(grid=(rows, 1, 1),
                                      block=(TILE, 1, 1))  # <<<rows, 128>>> 的 CuTe 写法


def cute_reduce(a):
    N = a.numel()
    rows = (N + TILE - 1) // TILE                # ceil(N/128)：需要几个 block
    pad = rows * TILE - N                        # 尾行补 0 的个数
    x = torch.cat([a, torch.zeros(pad, device=a.device, dtype=a.dtype)]).reshape(rows, TILE)
    partial = torch.empty((rows,), device=a.device, dtype=a.dtype)
    reduce_host(cute.runtime.from_dlpack(x), cute.runtime.from_dlpack(partial), rows)
    torch.cuda.synchronize()                     # 等 kernel 完成再读 partial
    return partial.double().sum()                # host 用 fp64 汇总部分和


def run(N, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(N, device="cuda") * 0.001
    out = cute_reduce(a)
    ref = a.double().sum()
    summarize_error(out, ref, f"cute_t11_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)
    run(999983)
    run(1)
