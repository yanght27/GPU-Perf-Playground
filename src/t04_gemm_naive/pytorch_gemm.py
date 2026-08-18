# 官方依据：torch.einsum/torch.matmul（PyTorch 2.13 文档，台账 S15）。
"""T04 朴素 GEMM —— 路径 1：PyTorch（黄金参考 + 显式 einsum 语义 + 库基线）。

- fp64 参考：a.double() @ b.double()
- 显式朴素语义：torch.einsum('mk,kn->mn', a, b)（PyTorch 会调度优化 kernel，但表达式
  与朴素 GEMM 逐行对应，适合学习对照）
- 库基线：torch.matmul（cuBLAS 后端）
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


def gemm_ref(a, b):
    """fp64 黄金参考，再转回 fp32。"""
    return (a.double() @ b.double()).float()


def gemm_einsum(a, b):
    """显式朴素语义：m,k × k,n -> m,n。"""
    return torch.einsum("mk,kn->mn", a, b)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda", dtype=torch.float32) - 0.5
    b = torch.rand((K, N), device="cuda", dtype=torch.float32) - 0.5
    ref = gemm_ref(a, b)

    out = gemm_einsum(a, b)
    summarize_error(out, ref, f"pytorch_einsum_{M}x{N}x{K}", tolerance=5e-3)

    ms = bench(lambda: gemm_einsum(a, b))
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[pytorch_einsum] M={M} N={N} K={K} avg_ms={ms:.4f} gflops={gflops:.1f}")

    ms = bench(lambda: a @ b)
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[pytorch_matmul] M={M} N={N} K={K} avg_ms={ms:.4f} gflops={gflops:.1f}")


def run_boundary():
    for M, N, K in [(17, 31, 33), (1, 128, 1)]:
        torch.manual_seed(0)
        a = torch.rand((M, K), device="cuda") - 0.5
        b = torch.rand((K, N), device="cuda") - 0.5
        ref = gemm_ref(a, b)
        out = gemm_einsum(a, b)
        summarize_error(out, ref, f"pytorch_boundary_{M}x{N}x{K}", tolerance=5e-3)

if __name__ == "__main__":
    run_boundary()
    run(512, 512, 512)
    run(1024, 1024, 1024)
