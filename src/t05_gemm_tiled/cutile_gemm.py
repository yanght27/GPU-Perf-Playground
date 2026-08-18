"""T05 GEMM Tiling —— 路径 4：cuTile Python（官方 MatMul.py 的 tile 写法）。
官方依据：cuTile MatMul.py tile=16（台账 S03e）。

与 T04 tile=1 的朴素版唯一区别：tile 放大到 16×16×16，让每个 processor 一次算
256 个输出，K 循环步长 16。官方 API：ct.load / ct.mma / ct.store。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@ct.kernel
def mm_kernel(A, B, C,
              tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    bidx = ct.bid(0)   # M 方向 tile 编号
    bidy = ct.bid(1)   # N 方向 tile 编号
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=acc)


def cutile_gemm_tiled(A, B, tm=16, tn=16, tk=16):
    M, K = A.shape
    _, N = B.shape
    C = cp.zeros((M, N), dtype=cp.float32)
    grid = (ct.cdiv(M, tm), ct.cdiv(N, tn), 1)
    ct.launch(cp.cuda.get_current_stream(), grid, mm_kernel, (A, B, C, tm, tn, tk))
    return C


def run(M, N, K):
    rng = cp.random.default_rng(0)
    A = rng.random((M, K), dtype=cp.float32) - 0.5
    B = rng.random((K, N), dtype=cp.float32) - 0.5
    ref = (A.astype(cp.float64) @ B.astype(cp.float64)).astype(cp.float32)
    C = cutile_gemm_tiled(A, B)
    summarize_error(cp.asarray(C), cp.asarray(ref), f"cutile_tiled_{M}x{N}x{K}", tolerance=5e-3)

    for _ in range(2):
        cutile_gemm_tiled(A, B)
    cp.cuda.Stream.null.synchronize()
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    for _ in range(5):
        cutile_gemm_tiled(A, B)
    e.record(); e.synchronize()
    ms = cp.cuda.get_elapsed_time(s, e) / 5
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[cutile_tiled] M={M} N={N} K={K} avg_ms={ms:.4f} gflops={gflops:.1f}")


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
