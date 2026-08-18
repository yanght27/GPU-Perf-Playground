"""T09 Transpose 朴素版 —— 路径 4：cuTile Python。

官方依据：cutile-python samples/Transpose.py（台账 S03）。官方示例是 tile 转置；
T09 用 tile=(1,1) 退化为朴素版，T10 再放大 tile。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@ct.kernel
def transpose_kernel(x, y, tm: ct.Constant[int], tn: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)


def cutile_transpose(x, tm=1, tn=1):
    M, N = x.shape
    y = cp.zeros((N, M), dtype=x.dtype)
    grid = (ct.cdiv(M, tm), ct.cdiv(N, tn), 1)
    ct.launch(cp.cuda.get_current_stream(), grid, transpose_kernel, (x, y, tm, tn))
    return y


def run(W, H):
    rng = cp.random.default_rng(0)
    a = rng.random((H, W), dtype=cp.float32)
    ref = a.astype(cp.float64).T.astype(cp.float32)
    out = cutile_transpose(a)
    summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_transpose_{H}x{W}")


if __name__ == "__main__":
    run(512, 512)
    run(513, 257)
    run(1, 128)
