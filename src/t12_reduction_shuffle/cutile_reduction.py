"""T12 Reduction Warp Shuffle —— 路径 4：cuTile Python（最接近官方能力对照 + shuffle 级 N/A）。

官方依据：cuTile 官方 reduction 测试 test_reduction.py 的 ct.sum/axis 写法（台账 S03k）。
T12 唯一学习变量是 warp shuffle 规约；cuTile 1.5.0 只暴露 tile 级 `ct.sum(axis=...)`，
不提供 `__shfl_*` 等 lane 级原语，也没有生成代码接口让用户观察其内部是否用 shuffle。
按 v1.7“五路径齐”规则：给出最接近官方能力（ct.sum）的同步实现对照，shuffle 机制记为
N/A（官方 test_reduction.py 中 grep 不到 shuffle/shfl，证据见 T12 讲义 §2）。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TILE = 128  # 一行一个 block；列数必须是编译期常量
TOL = 1e-4


@ct.kernel
def reduce_kernel(x, y, T: ct.Constant[int]):
    px = ct.bid(0)                               # = blockIdx.x：第几行 tile
    row = ct.load(x, index=(px, 0), shape=(1, T))  # 读 1 行 × T 列的一个 tile
    out = ct.sum(row, axis=1)                    # 官方轴规约：沿列把 1×T 压成 1×1
    ct.store(y, index=(px,), tile=out)           # 每个 block 写 y[px]


def cutile_reduce(a):
    N = a.shape[0]
    rows = ct.cdiv(N, TILE)                      # ceil(N/128)
    pad = rows * TILE - N                        # 尾行需要补的 0 的个数
    x = cp.zeros((rows, TILE), dtype=a.dtype)    # 先造一块全 0 的规则 2D 垫子
    if N:
        x.reshape(-1)[:N] = a                    # 把真实数据铺进前 N 个位置
    y = cp.empty((rows,), dtype=a.dtype)         # 每个 block 输出一个部分和
    ct.launch(cp.cuda.get_current_stream(), (rows, 1, 1),
              reduce_kernel, (x, y, TILE))       # grid=(rows,1,1)，block 由编译器定
    cp.cuda.get_current_stream().synchronize()   # 等 kernel 完成再读结果
    return y.astype(cp.float64).sum()            # host 用 fp64 汇总部分和


def run(N, seed=0):
    rng = cp.random.default_rng(seed)            # 可复现随机数
    a = rng.random((N,), dtype=cp.float32) * 0.001
    out = cutile_reduce(a)
    ref = a.astype(cp.float64).sum()             # fp64 黄金参考
    summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_t12_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)
    run(999983)
    run(1)
