"""T08 Tensor Core —— 路径 1：PyTorch fp16/bf16 参考与库基线。

官方依据：PyTorch matmul（2.13 文档，台账 S15）。GPU 上 fp16/bf16 matmul 由 cuBLAS
后端调度，通常走 Tensor Core。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


def run(M, N, K):
    torch.manual_seed(0)
    a32 = torch.rand((M, K), device="cuda") - 0.5
    b32 = torch.rand((K, N), device="cuda") - 0.5
    ref32 = a32 @ b32
    for dtype, tol in [(torch.float16, 0.02), (torch.bfloat16, 0.08)]:
        a = a32.to(dtype); b = b32.to(dtype)
        out = (a @ b).float()
        summarize_error(out, ref32, f"pytorch_t8_{dtype}_{M}x{N}x{K}", tolerance=tol)
        if dtype == torch.float16:
            for _ in range(5): a @ b
            torch.cuda.synchronize()
            st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(20): a @ b
            en.record(); torch.cuda.synchronize()
            ms=st.elapsed_time(en)/20
            print(f"[pytorch_t8_fp16] M={M} N={N} K={K} ms={ms:.4f} gflops={2*M*N*K/1e9/(ms/1e3):.0f}")


if __name__ == "__main__":
    run(512, 512, 512)
    run(1024, 1024, 1024)
