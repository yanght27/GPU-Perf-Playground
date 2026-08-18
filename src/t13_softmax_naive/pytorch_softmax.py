"""T13 Softmax 朴素 3-pass —— 路径 1：PyTorch（naive_softmax 参考实现）。

官方依据：
  - Triton 官方 tutorial 02-fused-softmax 的 naive_softmax（台账 S01m）：
    max -> z=x-max -> exp -> sum -> divide 的 5 步 3 读 2 写朴素写法；
  - PyTorch torch.softmax / Tensor.max / torch.exp / Tensor.sum 语义（台账 S15d）；
  - fp64 黄金参考使用 torch.softmax(double)（台账 S15d）。
本路径角色：用 PyTorch 高层 op 显式写出 3-pass 语义，作为五路径的“数学定义放大镜”；
torch.softmax(x) 黑盒版只用于对照打印，不作为主输出。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5  # fp32 softmax 与 fp64 参考的最大绝对误差容差


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """官方 tutorial 的 naive_softmax：3 次全矩阵读（max/exp/divide 各读一遍）。"""
    x_max = x.max(dim=1, keepdim=True).values   # pass 1：每行最大值
    z = x - x_max                               # 数值稳定：先减 max，exp 不溢出
    numerator = torch.exp(z)                    # pass 2：分子 exp
    denominator = numerator.sum(dim=1, keepdim=True)  # pass 2 的行归约
    return numerator / denominator              # pass 3：归一化


def fp64_ref(x: torch.Tensor) -> torch.Tensor:
    """黄金参考：同一矩阵转 fp64 后走 PyTorch softmax。"""
    return torch.softmax(x.double(), dim=1)


def make_cases():
    torch.manual_seed(0)
    a = torch.rand(1024, 4096, device="cuda") * 10.0 - 5.0
    a[0].fill_(7.0)                             # 全相同行：softmax = 全 1/128
    a[1, 0], a[1, 1] = -1000.0, 1000.0          # 极值 ±1000：检验减 max 防溢出
    a[2, 0], a[2, 1] = 1000.0, -1000.0

    b = torch.rand(37, 999, device="cuda") * 10.0 - 5.0
    b[0].fill_(-7.0)                            # 全相同负值行 + 未对齐列数 999
    b[1, 0], b[1, 1] = 1000.0, -1000.0
    b[2, 0], b[2, 1] = -1000.0, 1000.0

    c = torch.full((1, 1), 1000.0, device="cuda")  # N=1 边界
    return [(a, "R=1024_C=4096"), (b, "R=37_C=999_unaligned"), (c, "R=1_C=1")]


def run_case(x, name):
    out = naive_softmax(x)
    ref = fp64_ref(x)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"pytorch_t13_{name}", tolerance=TOL)
    blackbox = torch.softmax(x, dim=1)         # 黑盒版只做同 shape 对照
    summarize_error(blackbox, ref, f"pytorch_blackbox_t13_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, name in make_cases():
        run_case(x, name)
