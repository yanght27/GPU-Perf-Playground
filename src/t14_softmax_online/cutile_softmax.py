"""T14 Softmax Online/融合版 —— 路径 4：cuTile Python（官方 fused tile softmax）。

官方依据：cuTile 官方 test/test_softmax.py 的 `softmax_per_row`（台账 S03l）：
  numerator/denominator 在 tile 内完成，global 读 1 次 + 写 1 次（tile 级融合）。
cuTile 不暴露 running max/sum 的 online 更新原语，因此“online 公式”对本路径记 N/A；
本文件取官方最接近的 fused tile 能力作为对照。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5
PAD = -1.0e30


@ct.kernel
def softmax_per_row(input, output, N: ct.Constant[int]):
    row = ct.load(input, index=(ct.bid(0), 0), shape=(1, N))     # 一次读 1 行
    numerator = ct.exp(row - ct.max(row, axis=1, keepdims=True))
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    ct.store(output, index=(ct.bid(0), 0), tile=numerator / denominator)


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
    return [(a, 4096, "R=1024_C=4096"), (b, 1024, "R=37_C=999_unaligned"), (c, 1, "R=1_C=1")]


def fp64_ref(x):
    x64 = x.astype(cp.float64)
    m = x64.max(axis=1, keepdims=True)
    e = cp.exp(x64 - m)
    return e / e.sum(axis=1, keepdims=True)


def cutile_softmax(x, TILE):
    R, C = x.shape
    if C < TILE:
        xp = cp.concatenate([x, cp.full((R, TILE - C), PAD, dtype=x.dtype)], axis=1)
    else:
        xp = x
    yp = cp.empty_like(xp)
    ct.launch(cp.cuda.get_current_stream(), (R, 1, 1),
              softmax_per_row, (xp, yp, TILE))
    cp.cuda.get_current_stream().synchronize()
    return yp[:, :C] if C else yp[:, :0]


def run_case(x, TILE, name):
    out = cutile_softmax(x, TILE)
    ref = fp64_ref(x)
    summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_t14_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, TILE, name in make_cases():
        run_case(x, TILE, name)
