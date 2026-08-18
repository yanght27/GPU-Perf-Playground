# 官方依据：torch.nn.functional.relu（PyTorch 2.13 文档，台账 S15b）。
"""T03 ReLU 向量化版 —— 路径 1：PyTorch（参考 + 观察它已自动向量化）。

PyTorch 的 F.relu 在 GPU 上会调度到 vectorized_elementwise_kernel<4>（NCU 证据里可见），
即每个线程一次处理 4 个 float32。T03 的 PyTorch 路径不写新代码，而是把
“框架替你做了向量化”这件事用工具证据讲清楚。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps


def make_inputs(n: int):
    i = torch.arange(n, device="cuda", dtype=torch.float32)
    return torch.where(i % 7 == 0, torch.zeros_like(i), ((i % 97).float() - 48.0) * 0.5)


def pytorch_relu(a: torch.Tensor) -> torch.Tensor:
    return F.relu(a)


def main() -> None:
    for tag, N in [("aligned", 1 << 20), ("unaligned", 1_000_003)]:
        a = make_inputs(N)
        ref = torch.clamp(a.double(), min=0.0).float()
        out = pytorch_relu(a)
        summarize_error(out, ref, f"pytorch_relu_{tag}")

    N = 1 << 20
    a = make_inputs(N)
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
    # 极值语义：±Inf、NaN、1e38（与 fp64 参考比较，equal_nan=True）
    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1e38, -1e38],
                     device="cuda", dtype=torch.float32)
    ref_ext = torch.clamp(x.double(), min=0.0).float()
    out_ext = pytorch_relu(x)
    ok = torch.allclose(out_ext, ref_ext, rtol=0.0, atol=1e-5, equal_nan=True)
    print(f"[pytorch_relu_extreme] {'CORRECT_PASS' if ok else 'CORRECT_FAIL'} out={out_ext.tolist()}")

    avg_s = start.elapsed_time(end) / 1000.0 / ITERS
    print(f"[pytorch_relu] avg_ms={avg_s * 1e3:.4f} effective_gbps={effective_gbps(N, avg_s, accesses=2):.1f}")


if __name__ == "__main__":
    main()
