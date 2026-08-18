"""T05 GEMM Tiling —— 路径 3：Triton（官方 tutorial 03 的 tiled matmul 写法）。
官方依据：Triton tutorial 03（台账 S01e）。

与 T04 朴素版不同：每个 program 负责 BLOCK_M×BLOCK_N 个输出，内层按 BLOCK_K 分块，
Triton 编译器自动使用 shared memory 缓存这些 tile。未做 GROUP 重排（那是官方教程后半
的 L2 优化，T07 再讲）。
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
def gemm_tiled_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 官方 tutorial 03 的边界处理：load 地址用 %M/%N 折回，store 地址用真实坐标
    offs_m_load = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n_load = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_m_store = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_store = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m_load[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n_load[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k0 * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k0 * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")   # fp32 精确乘加，不用 TF32
        a_ptrs += BLOCK_K                # 指针步进一个 K tile
        b_ptrs += BLOCK_K * N

    c_mask = (offs_m_store[:, None] < M) & (offs_n_store[None, :] < N)
    tl.store(c_ptr + offs_m_store[:, None] * N + offs_n_store[None, :], acc, mask=c_mask)


def triton_gemm_tiled(a, b, BM=BM, BN=BN, BK=BK):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    gemm_tiled_kernel[grid](a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    out = triton_gemm_tiled(a, b)
    # 非整除边界时 kernel 只在 mask 内写结果；只比较有效区域 [0:M, 0:N]
    summarize_error(out[:M, :N], ref, f"triton_tiled_{M}x{N}x{K}", tolerance=5e-3)
    ms, mn, mx = triton.testing.do_bench(lambda: triton_gemm_tiled(a, b),
                                         quantiles=[0.5, 0.2, 0.8])
    gflops = 2.0 * M * N * K / 1e9 / (ms / 1e3)
    print(f"[triton_tiled] M={M} N={N} K={K} median_ms={ms:.4f} gflops={gflops:.1f}")


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
