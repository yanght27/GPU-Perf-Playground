"""T07 流水线 —— 路径 4：cuTile Python（官方能力边界记录）。
官方依据：cuTile 官方 MatMul.py + latency hint（台账 S03g/S03e）。

官方 cuTile 仓库当前没有 async/cp.async/pipeline 示例（已在 source-ledger 记录 N/A）。
本路径用官方 MatMul.py 的同步 tile 实现做正确性对照，明确标注：
“异步流水线：官方示例缺失 → N/A；同步 tile 实现：可用并已通过正确性。”
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@ct.kernel
def mm_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int],
              latency_hint: ct.Constant[int]):
    bidx = ct.bid(0); bidy = ct.bid(1)
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        # 官方 latency hint（1-10）：提示编译器把后续 tile 的 load 提前（软件流水线）
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO, latency=latency_hint)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=ct.PaddingMode.ZERO, latency=latency_hint)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=acc)


def cutile_gemm(A, B, tm=16, tn=16, tk=16, latency_hint=1):
    M, K = A.shape; _, N = B.shape
    C = cp.zeros((M, N), dtype=cp.float32)
    grid = (ct.cdiv(M, tm), ct.cdiv(N, tn), 1)
    ct.launch(cp.cuda.get_current_stream(), grid, mm_kernel,
              (A, B, C, tm, tn, tk, latency_hint))
    return C


def run(M, N, K):
    rng = cp.random.default_rng(0)
    A = rng.random((M, K), dtype=cp.float32) - 0.5
    B = rng.random((K, N), dtype=cp.float32) - 0.5
    ref = (A.astype(cp.float64) @ B.astype(cp.float64)).astype(cp.float32)
    for lat in [1, 2, 4]:
        C = cutile_gemm(A, B, latency_hint=lat)
        summarize_error(cp.asarray(C), cp.asarray(ref),
                        f"cutile_t07_pipe_{M}x{N}x{K}_latency{lat}", tolerance=5e-3)
    # 官方 latency hint 的 benchmark（512/1024 上对比）
    if M == 512 and N == 512 and K == 512:
        for lat in [1, 2, 4]:
            for _ in range(2):
                cutile_gemm(A, B, latency_hint=lat)
            cp.cuda.Stream.null.synchronize()
            s = cp.cuda.Event(); e = cp.cuda.Event()
            s.record()
            for _ in range(5):
                cutile_gemm(A, B, latency_hint=lat)
            e.record(); e.synchronize()
            ms = cp.cuda.get_elapsed_time(s, e) / 5
            print(f"[cutile_t07_pipe] M={M} N={N} K={K} latency={lat} ms={ms:.4f}")


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
