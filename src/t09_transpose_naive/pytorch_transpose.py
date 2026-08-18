"""T09 Transpose 朴素版 —— 路径 1：PyTorch 参考。

官方依据：torch.Tensor.transpose / contiguous（PyTorch 2.13 文档，台账 S15）。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


def run(W, H):
    torch.manual_seed(0)
    a = torch.rand((H, W), device="cuda", dtype=torch.float32)
    ref = (a.double().t()).float()
    out = a.t().contiguous()
    summarize_error(out, ref, f"pytorch_transpose_{H}x{W}")
    for _ in range(10):
        a.t().contiguous()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        a.t().contiguous()
    e.record(); torch.cuda.synchronize()
    print(f"[pytorch_transpose] W={W} H={H} avg_ms={s.elapsed_time(e)/50:.4f}")


if __name__ == "__main__":
    run(512, 512)
    run(513, 257)
    run(1, 128)
