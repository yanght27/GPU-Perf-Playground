"""T12 Reduction Warp Shuffle —— 路径 5：CUTLASS CuTe DSL（官方 warp shuffle 树）。

官方依据：CUTLASS CuTe DSL 官方示例
`experimental/primitives/reduction/warp_vector_reduce.py`（台账 S02m）。
官方示例 = 每个 CTA 一个 warp、每 lane 向量加载 ITEMS_PER_LANE 个数并用
`Vector.reduce` 折叠，随后用 `cute.arch.shuffle_sync_bfly` 做 5 轮 butterfly
shuffle 树（f32 的 add 路径，文档同文件 docstring 的表格）。
本文件把官方示例的“每 CTA 一行”改成“每 CTA 一段全局数组”，数组尾行补 0；
这是为了与 T11/T12 的 1-D 求和同一语义，学习变量仍是 warp shuffle。
"""

import sys
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

_WARP_SIZE = 32
ITEMS_PER_LANE = 4          # 官方示例 argparse 允许 2/4/8/16；取 4 与 T11 的 128 列对齐
TILE = _WARP_SIZE * ITEMS_PER_LANE  # 一个 CTA（一个 warp）负责 128 个数
TOL = 1e-4


@cute.kernel
def reduce_kernel(src: cute.Tensor, out: cute.Tensor):
    row, _, _ = cute.arch.block_idx()            # 第几段
    tx, _, _ = cute.arch.thread_idx()            # warp 里的 lane id

    src_ptr = src.iterator.raw_ptr() + row * TILE + tx * ITEMS_PER_LANE
    out_ptr = out.iterator.raw_ptr() + row

    # 官方写法：一次向量化读 4 个连续 float，再在寄存器里折叠成 1 个 lane 局部和
    values = src_ptr.load(count=ITEMS_PER_LANE, alignment=ITEMS_PER_LANE * 4)
    lane = values.reduce("add")

    # 官方 _warp_reduce_tree 的 5 轮 butterfly：lane 与 lane^16、lane^8、...、lane^1 配对
    lane = lane + cute.arch.shuffle_sync_bfly(lane, 16)
    lane = lane + cute.arch.shuffle_sync_bfly(lane, 8)
    lane = lane + cute.arch.shuffle_sync_bfly(lane, 4)
    lane = lane + cute.arch.shuffle_sync_bfly(lane, 2)
    lane = lane + cute.arch.shuffle_sync_bfly(lane, 1)

    if tx == 0:
        out_ptr.store(lane)                      # 每段写一个部分和


@cute.jit
def reduce_host(src: cute.Tensor, out: cute.Tensor, rows: cutlass.Int32):
    reduce_kernel(src, out).launch(grid=(rows, 1, 1),
                                   block=(_WARP_SIZE, 1, 1))  # <<<rows, 32>>> 的 CuTe 写法


def cute_reduce(a):
    N = a.numel()
    rows = (N + TILE - 1) // TILE               # ceil(N/128)：需要几个段
    pad = rows * TILE - N                       # 尾段补 0 的个数
    x = torch.cat([a, torch.zeros(pad, device=a.device, dtype=a.dtype)]).reshape(rows, TILE)
    partial = torch.empty((rows,), device=a.device, dtype=a.dtype)
    reduce_host(cute.runtime.from_dlpack(x), cute.runtime.from_dlpack(partial), rows)
    torch.cuda.synchronize()                    # 等 kernel 完成再读 partial
    return partial.double().sum()               # host 用 fp64 汇总部分和


def run(N, seed=0):
    torch.manual_seed(seed)
    a = torch.rand(N, device="cuda") * 0.001
    out = cute_reduce(a)
    ref = a.double().sum()
    summarize_error(out, ref, f"cute_t12_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)
    run(999983)
    run(1)
