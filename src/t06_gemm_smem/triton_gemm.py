"""T06 共享内存优化 —— 路径 3：Triton。
官方依据：Triton tutorial 03 autotune 配置表（台账 S01f）。

Triton 编译器自动处理 shared memory 布局/bank 与向量化；T06 只换 block 配置做对照，
学习变量是“编译器替你优化了什么”。官方 tutorial 03 的 autotune 配置表就是权威依据。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


@triton.jit
def gemm_tiled_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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


def triton_gemm(a, b, BM=32, BN=32, BK=32):
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    gemm_tiled_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    for BM, BN, BK in [(32, 32, 32), (64, 64, 32)]:
        out = triton_gemm(a, b, BM, BN, BK)
        summarize_error(out[:M, :N], ref, f"triton_t06_{M}x{N}x{K}_b{BM}x{BN}x{BK}", tolerance=5e-3)
        ms = triton.testing.do_bench(lambda: triton_gemm(a, b, BM, BN, BK))
        gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
        print(f"[triton_t06] M={M} N={N} K={K} block={BM}x{BN}x{BK} ms={ms:.4f} gflops={gflops:.0f}")


if __name__ == "__main__":
    run(512, 512, 512)
    run(1024, 1024, 1024)
