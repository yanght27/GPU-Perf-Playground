"""T13 Softmax 朴素 3-pass —— 路径 4：cuTile Python（官方 test_softmax 行 kernel）。

官方依据：cuTile 官方 test/test_softmax.py 的 softmax_per_row（台账 S03l）：
  numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))
  denominator = ct.sum(numerator, axis=1, keepdims=True)
  ct.store(... numerator / denominator)
本 Ticket 取一行一个 block 的 3-pass 语义；列宽不是 2 的幂时，按 tile 规则把列补到
编译期常量 TILE，补齐值用 -1e30（exp(-1e30)=0，不改变 max/sum/输出）。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5
PAD = -1.0e30  # 有限负数：cuTile exp 后为 0，避免 -inf 的边界语义差异


@ct.kernel
def softmax_per_row(x, y, C: ct.Constant[int]):
    px = ct.bid(0)                               # = blockIdx.x：一行一个 block
    row = ct.load(x, index=(px, 0), shape=(1, C))  # 读 1 行 × C 列（C 为编译期 tile 宽）
    numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))   # pass 1+2 的 max/exp
    denominator = ct.sum(numerator, axis=1, keepdims=True)         # pass 2 行归约
    ct.store(y, index=(px, 0), tile=numerator / denominator)       # pass 3 归一化写回


def make_cases():
    rng = cp.random.default_rng(0)
    a = rng.random((1024, 4096), dtype=cp.float32) * 10.0 - 5.0
    a[0].fill(7.0)
    a[1, 0], a[1, 1] = -1000.0, 1000.0
    a[2, 0], a[2, 1] = 1000.0, -1000.0

    b = rng.random((37, 999), dtype=cp.float32) * 10.0 - 5.0
    b[0].fill(-7.0)
    b[1, 0], b[1, 1] = 1000.0, -1000.0
    b[2, 0], b[2, 1] = -1000.0, 1000.0

    c = cp.full((1, 1), 1000.0, dtype=cp.float32)
    return [(a, 4096, "R=1024_C=4096"), (b, 999, "R=37_C=999_unaligned"), (c, 1, "R=1_C=1")]


def fp64_ref(x):
    x64 = x.astype(cp.float64)
    m = x64.max(axis=1, keepdims=True)           # 稳定 softmax：先减行 max
    e = cp.exp(x64 - m)
    return e / e.sum(axis=1, keepdims=True)


def cutile_softmax(x, C, TILE):
    R = x.shape[0]
    pad = TILE - C
    yp = cp.empty((R, TILE), dtype=x.dtype)
    if C:
        xp = cp.concatenate([x, cp.full((R, TILE - C), PAD, dtype=x.dtype)], axis=1)
    else:
        xp = cp.full((R, TILE), PAD, dtype=x.dtype)
    ct.launch(cp.cuda.get_current_stream(), (R, 1, 1),
              softmax_per_row, (xp, yp, TILE))
    cp.cuda.get_current_stream().synchronize()
    return yp[:, :C] if C else yp[:, :0]


def run_case(x, C, TILE, name):
    out = cutile_softmax(x, C, TILE)
    ref = fp64_ref(x)
    summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_t13_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, C, name in make_cases():
        run_case(x, C, C if C in (4096, 1) else 1024, name)
