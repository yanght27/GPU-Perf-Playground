"""T14 Softmax Online/融合版 —— 路径 1：PyTorch（语义与 fp64 黄金参考）。

官方依据：PyTorch `torch.softmax`（台账 S15d）。
本 Ticket 关注 online/fused 机制；PyTorch 的 torch.softmax 是官方融合实现，对用户是
黑盒，因此本路径只做“同一 shape 的正确性参考”，并同时给出 T13 的 naive_softmax
（S01m）做访存账对照：naive 读 5MN+2M/写 3MN+2M，torch.softmax 内部为融合 kernel。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-5


def naive_softmax(x):
    x_max = x.max(dim=1, keepdim=True).values
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator


def fp64_ref(x):
    return torch.softmax(x.double(), dim=1)


def make_cases():
    torch.manual_seed(0)
    a = torch.rand(1024, 4096, device="cuda") * 10.0 - 5.0
    a[0].fill_(7.0)
    a[1, 0], a[1, 1] = -1000.0, 1000.0
    a[2, 0], a[2, 1] = 1000.0, -1000.0

    b = torch.rand(37, 999, device="cuda") * 10.0 - 5.0
    b[0].fill_(-7.0)
    b[1, 0], b[1, 1] = 1000.0, -1000.0
    b[2, 0], b[2, 1] = -1000.0, 1000.0

    c = torch.full((1, 1), 1000.0, device="cuda")
    return [(a, "R=1024_C=4096"), (b, "R=37_C=999_unaligned"), (c, "R=1_C=1")]


def run_case(x, name):
    out = torch.softmax(x, dim=1)            # 官方融合实现（黑盒）
    ref = fp64_ref(x)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"pytorch_t14_{name}", tolerance=TOL)
    if name == "R=1024_C=4096":
        out_naive = naive_softmax(x)
        summarize_error(out_naive, out, f"pytorch_t14_naive_vs_fused_{name}", tolerance=TOL)


if __name__ == "__main__":
    for x, name in make_cases():
        run_case(x, name)
