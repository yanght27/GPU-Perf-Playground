"""T07 流水线 —— 路径 3：Triton（num_stages 对比）。
官方依据：Triton tutorial 03 num_stages 配置（台账 S01g）。

Triton 的 tl.load/tl.dot 由编译器插入异步搬运与软件流水线；num_stages 控制
“预取几块”。官方 tutorial 03 的 autotune 配置就是 num_stages=3..5。
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
    om_load = (pm * BM + tl.arange(0, BM)) % M
    on_load = (pn * BN + tl.arange(0, BN)) % N
    om_store = pm * BM + tl.arange(0, BM)
    on_store = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om_load[:, None] * K + ok[None, :]
    bp = b_ptr + ok[:, None] * N + on_load[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        a = tl.load(ap, mask=ok[None, :] < K - k0 * BK, other=0.0)
        b = tl.load(bp, mask=ok[:, None] < K - k0 * BK, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        ap += BK; bp += BK * N
    mask = (om_store[:, None] < M) & (on_store[None, :] < N)
    tl.store(c_ptr + om_store[:, None] * N + on_store[None, :], acc, mask=mask)


def triton_gemm(a, b, num_stages=1):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    gemm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, BM=BM, BN=BN, BK=BK, num_stages=num_stages)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    out = triton_gemm(a, b, num_stages=2)
    summarize_error(out[:M, :N], ref, f"triton_pipe_{M}x{N}x{K}", tolerance=5e-3)
    for ns in [1, 2, 3]:
        ms = triton.testing.do_bench(lambda: triton_gemm(a, b, num_stages=ns))
        gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
        print(f"[triton_pipe] M={M} N={N} K={K} num_stages={ns} ms={ms:.4f} gflops={gflops:.0f}")


if __name__ == "__main__":
    run(512, 512, 512)
    run(1024, 1024, 1024)
