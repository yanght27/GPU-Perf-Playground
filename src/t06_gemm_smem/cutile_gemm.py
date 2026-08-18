"""T06 共享内存优化 —— 路径 4：cuTile Python。
官方依据：cuTile 官方 MatMul.py（台账 S03e）。

官方 MatMul.py 的 ct.load/ct.mma 由编译器负责 shared 布局/bank/向量化；
T06 用官方 tile=16/32 做“编译器替你优化 shared”的能力对照。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@ct.kernel
def mm_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0); bidy = ct.bid(1)
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=acc)


def cutile_gemm(A, B, tm=16, tn=16, tk=16):
    M, K = A.shape; _, N = B.shape
    C = cp.zeros((M, N), dtype=cp.float32)
    grid = (ct.cdiv(M, tm), ct.cdiv(N, tn), 1)
    ct.launch(cp.cuda.get_current_stream(), grid, mm_kernel, (A, B, C, tm, tn, tk))
    return C


def run(M, N, K):
    rng = cp.random.default_rng(0)
    A = rng.random((M, K), dtype=cp.float32) - 0.5
    B = rng.random((K, N), dtype=cp.float32) - 0.5
    ref = (A.astype(cp.float64) @ B.astype(cp.float64)).astype(cp.float32)
    for tile in [(16, 16, 16), (32, 32, 16)]:
        C = cutile_gemm(A, B, *tile)
        summarize_error(cp.asarray(C), cp.asarray(ref),
                        f"cutile_t06_{M}x{N}x{K}_t{tile[0]}", tolerance=5e-3)


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
