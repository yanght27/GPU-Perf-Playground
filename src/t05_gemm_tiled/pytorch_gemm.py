# 官方依据：torch.matmul 黄金参考（PyTorch 2.13 文档，台账 S15）。
"""T05 GEMM Tiling —— 路径 1：PyTorch（黄金参考 + 库基线，不参与手写 tiling）。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


def run(M, N, K):
    torch.manual_seed(0)
    a = torch.rand((M, K), device="cuda") - 0.5
    b = torch.rand((K, N), device="cuda") - 0.5
    ref = (a.double() @ b.double()).float()
    out = a @ b
    summarize_error(out, ref, f"pytorch_matmul_{M}x{N}x{K}", tolerance=5e-3)


if __name__ == "__main__":
    run(17, 31, 33)
    run(512, 512, 512)
    run(1024, 1024, 1024)
