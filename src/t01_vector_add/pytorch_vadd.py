"""T01 Vector Add —— 路径 1：PyTorch（最容易理解，也是后面所有实现的黄金参考）。
官方依据：PyTorch torch.Tensor.add（台账 S15a）。

学习主线只有一行：out = torch.add(a, b)
其余代码（计时、误差）都是从 common 引入或标准的 CUDA event 模板。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps

N = 1 << 20  # 2^20 = 1,048,576 个元素


def pytorch_vadd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 实现：直接调用官方张量加法。

    - `a + b` 在 CPU/GPU 上都是同一个写法，PyTorch 负责把计算调度到当前设备；
    - 这里显式写 torch.add 是为了让“算子名”出现在代码里，便于和后面的 kernel 对照。
    """
    return torch.add(a, b)


def main() -> None:
    torch.manual_seed(0)

    # fp32：被测实现（后面 CUDA/Triton/cuTile/CuTe 都用 float32）
    a32 = torch.rand(N, dtype=torch.float32, device="cuda")
    b32 = torch.rand(N, dtype=torch.float32, device="cuda")

    # fp64：黄金参考。先在 64 位精度下算，再转回 fp32 比较，
    # 这样能区分“我们的实现错了”和“fp32 本身的舍入误差”。
    ref = (a32.double() + b32.double()).float()
    out = pytorch_vadd(a32, b32)
    summarize_error(out, ref, "pytorch_fp32_vs_fp64")

    # 计时：CUDA event 记录 GPU 时间线，不包含 Python 调度开销。
    # 先 warmup 让 JIT/缓存稳定，再测 ITERS 次取平均。
    ITERS = 100
    for _ in range(10):
        out = pytorch_vadd(a32, b32)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        out = pytorch_vadd(a32, b32)
    end.record()
    torch.cuda.synchronize()
    avg_s = start.elapsed_time(end) / 1000.0 / ITERS
    print(f"[pytorch] avg_ms={avg_s * 1e3:.4f} effective_gbps={effective_gbps(N, avg_s):.1f}")


if __name__ == "__main__":
    main()
