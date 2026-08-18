"""T17 FlashAttention（Triton 官方 tutorial 版）—— 路径 1：PyTorch SDPA 黄金参考。

官方依据：PyTorch `torch.nn.functional.scaled_dot_product_attention`（台账 S15e）。
本路径角色：为 Triton FA 提供 causal/non-causal、多个 seq_len 的黄金参考；
另外给出 fp64 eager 参考用于锁定 SDPA 本身（S15e、T15 公式）。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error


def eager_attention(q, k, v, causal, scale):
    scores = torch.matmul(q.double(), k.double().transpose(-2, -1)) * scale
    N = scores.shape[-1]
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool,
                                                          device=scores.device), 1), -1e300)
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v.double())


def run_case(B, H, N, D, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.2
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                         is_causal=causal, scale=D ** -0.5)
    ref64 = eager_attention(q, k, v, causal, D ** -0.5)
    torch.cuda.synchronize()
    summarize_error(out, ref64, f"pytorch_sdpa_t17_B{B}_H{H}_N{N}_D{D}_causal{causal}", tolerance=1e-2)


if __name__ == "__main__":
    run_case(4, 4, 512, 64, True)
    run_case(4, 4, 1024, 64, True)
    run_case(4, 4, 2048, 64, True)
    run_case(2, 4, 512, 64, False)
