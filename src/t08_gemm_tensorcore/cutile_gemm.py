"""T08 Tensor Core —— 路径 4：cuTile Python（官方 tf32 mma）。

官方依据：cutile-python samples/MatMul.py（台账 S03d）。官方示例对 fp32 输入执行
`.astype(ct.tfloat32)`，再由 `ct.mma` 映射到 Tensor Core。
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
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype   # 官方写法
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
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
    ref = A @ B
    C = cutile_gemm(A, B)
    summarize_error(cp.asarray(C), cp.asarray(ref), f"cutile_t8_tf32_{M}x{N}x{K}", tolerance=0.02)


if __name__ == "__main__":
    run(512, 512, 512)
    run(1024, 1024, 1024)
