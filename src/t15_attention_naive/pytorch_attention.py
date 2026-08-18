"""T15 Attention 朴素前向 —— 路径 1：PyTorch 双参考。

官方依据：PyTorch `torch.matmul` / `torch.nn.functional.softmax` /
`torch.nn.functional.scaled_dot_product_attention`（台账 S15e）。
角色：显式 eager 公式是“计算图放大镜”，F.scaled_dot_product_attention 是官方融合
黑盒参考，fp64 eager 是黄金参考。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-4  # fp32 attention 误差容差（softmax + 加权和，小 shape 下通常 1e-6 量级）


def eager_attention(q, k, v, scale, causal):
    """显式计算图：S = QK^T*scale -> mask -> softmax -> O = PV。"""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        N = scores.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=scores.device, dtype=torch.bool),
                          diagonal=1)          # 上三角（k>q）为 True
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def fp64_ref(q, k, v, scale, causal):
    """黄金参考：同一输入转 fp64 走显式公式。"""
    return eager_attention(q.double(), k.double(), v.double(), scale, causal)


def make_cases():
    torch.manual_seed(0)
    # 主 shape：B=2,H=2,N=64,D=32，causal
    q1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    k1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    v1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    # 未对齐 shape：B=1,H=1,N=37,D=17，non-causal，显式不同 scale
    q2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    k2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    v2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    # N=1 边界
    q3 = torch.randn(1, 1, 1, 8, device="cuda")
    k3 = torch.randn(1, 1, 1, 8, device="cuda")
    v3 = torch.randn(1, 1, 1, 8, device="cuda")
    return [(q1, k1, v1, 32 ** -0.5, True, "B2_H2_N64_D32_causal"),
            (q2, k2, v2, 0.5, False, "B1_H1_N37_D17_noncausal"),
            (q3, k3, v3, 8 ** -0.5, True, "B1_H1_N1_D8")]


def run_case(q, k, v, scale, causal, name):
    out = eager_attention(q, k, v, scale, causal)
    ref = fp64_ref(q, k, v, scale, causal)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"pytorch_eager_t15_{name}", tolerance=TOL)
    sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                          dropout_p=0.0, is_causal=causal, scale=scale)
    summarize_error(sdpa, ref, f"pytorch_sdpa_t15_{name}", tolerance=TOL)
    summarize_error(out, sdpa, f"pytorch_eager_vs_sdpa_t15_{name}", tolerance=TOL)


if __name__ == "__main__":
    for case in make_cases():
        run_case(*case)
