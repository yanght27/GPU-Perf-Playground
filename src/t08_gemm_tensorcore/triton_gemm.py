"""T08 Tensor Core —— 路径 3：Triton fp16 tl.dot。

官方依据：Triton tutorial 03（v3.7.1，台账 S01g）。fp16 输入的 tl.dot 由编译器
映射到 Tensor Core mma。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

BM, BN, BK = 32, 32, 32


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * K + ok[None, :]
    bp = b_ptr + ok[:, None] * N + on[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=ok[None, :] < K - k0 * BK, other=0.0)
        b = tl.load(bp, mask=ok[:, None] < K - k0 * BK, other=0.0)
        acc += tl.dot(a, b)   # fp16×fp16→fp32：Tensor Core mma
        ap += BK; bp += BK * N
    mask = (om[:, None] < M) & (on[None, :] < N)
    tl.store(c_ptr + om[:, None] * N + on[None, :], acc, mask=mask)


def triton_gemm(a, b):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a32 = torch.rand((M, K), device="cuda") - 0.5
    b32 = torch.rand((K, N), device="cuda") - 0.5
    ref = a32 @ b32
    a16 = a32.half(); b16 = b32.half()
    out = triton_gemm(a16, b16)
    summarize_error(out, ref, f"triton_t8_fp16_{M}x{N}x{K}", tolerance=0.02)
    ms = triton.testing.do_bench(lambda: triton_gemm(a16, b16))
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[triton_t8_fp16] M={M} N={N} K={K} ms={ms:.4f} gflops={gflops:.0f}")


if __name__ == "__main__":
    run(512, 512, 512)
    run(1024, 1024, 1024)
