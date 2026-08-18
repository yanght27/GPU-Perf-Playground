"""T04 朴素 GEMM —— 路径 3：Triton。
官方依据：Triton tutorial 03-matrix-multiplication.py（台账 S01d）。

采用最直白的“一个 program 算一个输出元素，沿 K 循环”的朴素写法（与官方 tutorial 03
的分块写法相对照；分块留给 T05）。官方 tutorial 03（v3.7.1）是 T05 的权威来源。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@triton.jit
def gemm_naive_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    pid = tl.program_id(0)
    row = pid // N          # 一维 grid 手动映射到二维输出 (row, col)
    col = pid % N
    acc = 0.0
    for k in tl.range(0, K):                     # 沿 K 串行累加：朴素 GEMM 的核心
        a = tl.load(a_ptr + row * K + k)         # A 行主序
        b = tl.load(b_ptr + k * N + col)         # B 行主序
        acc += a * b
    tl.store(c_ptr + row * N + col, acc)


def triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    gemm_naive_kernel[(M * N,)](a, b, c, M, N, K)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    out = triton_gemm(a, b)
    summarize_error(out, ref, f"triton_naive_{M}x{N}x{K}", tolerance=5e-3)
    ms, mn, mx = triton.testing.do_bench(lambda: triton_gemm(a, b), quantiles=[0.5, 0.2, 0.8])
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[triton_naive] M={M} N={N} K={K} median_ms={ms:.4f} gflops={gflops:.1f}")


def run_boundary():
    for M, N, K in [(17, 31, 33), (1, 128, 1)]:
        torch.manual_seed(0)
        a = torch.rand((M, K), device="cuda") - 0.5
        b = torch.rand((K, N), device="cuda") - 0.5
        ref = (a.double() @ b.double()).float()
        out = triton_gemm(a, b)
        summarize_error(out, ref, f"triton_boundary_{M}x{N}x{K}", tolerance=5e-3)

if __name__ == "__main__":
    run_boundary()
    run(512, 512, 512)
    run(1024, 1024, 1024)
