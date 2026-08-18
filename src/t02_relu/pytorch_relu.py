"""T02 ReLU 标量版 —— 路径 1：PyTorch（先建立正确语义）。
官方依据：PyTorch torch.nn.functional.relu（台账 S15b）。

ReLU(x) = max(x, 0)。本轮只做“标量”版本：每个元素独立判断一次，
不做 128-bit 向量化（那是 T03 的学习变量）。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps


def make_inputs(n: int, device="cuda"):
    """所有路径共用的确定性输入：i%7==0 处放精确 0，其余有正有负。"""
    i = torch.arange(n, device=device, dtype=torch.float32)
    a = torch.where(i % 7 == 0, torch.zeros_like(i), ((i % 97).float() - 48.0) * 0.5)
    return a


def pytorch_relu(a: torch.Tensor) -> torch.Tensor:
    """官方 API：torch.nn.functional.relu。"""
    return F.relu(a)


def main() -> None:
    N = 1 << 20
    a = make_inputs(N)
    # fp64 黄金参考：先 double 算 max(x,0)，再转回 fp32 比较
    ref = torch.clamp(a.double(), min=0.0).float()
    out = pytorch_relu(a)
    summarize_error(out, ref, "pytorch_relu_fp32_vs_fp64")

    # 边界用例：N 不被任何常用 block 大小整除，验证越界保护
    N_odd = 1_000_003
    a_odd = make_inputs(N_odd)
    ref_odd = torch.clamp(a_odd.double(), min=0.0).float()
    out_odd = pytorch_relu(a_odd)
    summarize_error(out_odd, ref_odd, "pytorch_relu_unaligned_N")

    ITERS = 100
    for _ in range(10):
        pytorch_relu(a)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        pytorch_relu(a)
    end.record()
    torch.cuda.synchronize()
    avg_s = start.elapsed_time(end) / 1000.0 / ITERS
    print(f"[pytorch_relu] avg_ms={avg_s * 1e3:.4f} effective_gbps={effective_gbps(N, avg_s, accesses=2):.1f}")


if __name__ == "__main__":
    main()
