"""T04 朴素 GEMM —— 路径 4：cuTile Python。
官方依据：cuTile MatMul.py（台账 S03d）。

把官方 MatMul.py 的 tiled `ct.mma` 写法退化成 tile=1×1×1：
每个 processor 只算 C 的一个元素，沿 K 逐 tile（1 个元素）累加。
这保持了官方 API 形态，同时把“朴素”讲清楚；tile>1 的版本归 T05。
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
    M = A.shape[0]
    N = B.shape[1]
    bidx = ct.bid(0)   # 行 tile 编号
    bidy = ct.bid(1)   # 列 tile 编号
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)          # 官方矩阵乘累加原语
    ct.store(C, index=(bidx, bidy), tile=acc)


def cutile_gemm(A, B, tm=1, tn=1, tk=1):
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
    C = cutile_gemm(A, B)
    summarize_error(cp.asarray(C), cp.asarray(ref), f"cutile_naive_{M}x{N}x{K}", tolerance=5e-3)

    ITERS = 5
    for _ in range(2):
        cutile_gemm(A, B)
    cp.cuda.Stream.null.synchronize()
    s = cp.cuda.Event(); e = cp.cuda.Event()
    s.record()
    for _ in range(ITERS):
        cutile_gemm(A, B)
    e.record(); e.synchronize()
    ms = cp.cuda.get_elapsed_time(s, e) / ITERS
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[cutile_naive] M={M} N={N} K={K} avg_ms={ms:.4f} gflops={gflops:.1f}")


def run_boundary():
    rng = cp.random.default_rng(0)
    for M, N, K in [(17, 31, 33), (1, 128, 1)]:
        A = rng.random((M, K), dtype=cp.float32) - 0.5
        B = rng.random((K, N), dtype=cp.float32) - 0.5
        ref = (A.astype(cp.float64) @ B.astype(cp.float64)).astype(cp.float32)
        C = cutile_gemm(A, B)
        summarize_error(cp.asarray(C), cp.asarray(ref), f"cutile_boundary_{M}x{N}x{K}", tolerance=5e-3)

if __name__ == "__main__":
    run_boundary()
    run(512, 512, 512)
    run(1024, 1024, 1024)
