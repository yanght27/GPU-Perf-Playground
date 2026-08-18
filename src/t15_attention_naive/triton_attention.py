"""T15 Attention 朴素前向 —— 路径 3：Triton（官方 tutorial 06 的 qk/softmax/pv 步骤）。

官方依据：Triton 官方 tutorial 06-fused-attention（台账 S01o）的 qk = QK^T*scale、
causal mask、softmax、pv = PV 四步语义。
本 Ticket 明确不做 IO-Aware 分块：每个 program 只处理一个 query 行，K/V 整段重复加载；
外层 LOOP 分块与 flash 结构留给 T17。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-4


@triton.jit
def attention_naive_kernel(Q, K, V, O, scale, N, D,
                           BN: tl.constexpr, BD: tl.constexpr, CAUSAL: tl.constexpr):
    q = tl.program_id(0)                    # query 行号
    bh = tl.program_id(1)                   # batch*heads 平铺下标

    offs_k = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    mask_k = offs_k < N
    mask_d = offs_d < D

    q_ptrs = Q + (bh * N + q) * D + offs_d
    qv = tl.load(q_ptrs, mask=mask_d, other=0.0)

    k_ptrs = K + (bh * N + offs_k[:, None]) * D + offs_d[None, :]
    kv = tl.load(k_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)

    # 计算图第 1 步：S = QK^T * scale
    scores = tl.sum(qv[None, :] * kv, axis=1) * scale
    # 计算图第 2 步：padding mask（k>=N -> -inf）与 causal mask（k>q -> -inf）
    scores = tl.where(mask_k, scores, -float("inf"))
    if CAUSAL:
        scores = tl.where(offs_k <= q, scores, -float("inf"))
    # 计算图第 3 步：softmax（数值稳定）
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom
    # 计算图第 4 步：O = PV
    v_ptrs = V + (bh * N + offs_k[:, None]) * D + offs_d[None, :]
    vv = tl.load(v_ptrs, mask=mask_k[:, None] & mask_d[None, :], other=0.0)
    o = tl.sum(p[:, None] * vv, axis=0)

    o_ptrs = O + (bh * N + q) * D + offs_d
    tl.store(o_ptrs, o, mask=mask_d)


def triton_attention(q, k, v, scale, causal):
    B, H, N, D = q.shape
    BH = B * H
    O = torch.empty_like(q)
    BN = triton.next_power_of_2(N)
    BD = triton.next_power_of_2(D)
    attention_naive_kernel[(N, BH)](q, k, v, O, scale, N, D,
                                    BN=BN, BD=BD, CAUSAL=causal, num_warps=4)
    return O


def make_cases():
    torch.manual_seed(0)
    q1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    k1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    v1 = torch.randn(2, 2, 64, 32, device="cuda") * 0.5
    q2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    k2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    v2 = torch.randn(1, 1, 37, 17, device="cuda") * 0.3
    q3 = torch.randn(1, 1, 1, 8, device="cuda")
    k3 = torch.randn(1, 1, 1, 8, device="cuda")
    v3 = torch.randn(1, 1, 1, 8, device="cuda")
    return [(q1, k1, v1, 32 ** -0.5, True, "B2_H2_N64_D32_causal"),
            (q2, k2, v2, 0.5, False, "B1_H1_N37_D17_noncausal"),
            (q3, k3, v3, 8 ** -0.5, True, "B1_H1_N1_D8")]


def fp64_ref(q, k, v, scale, causal):
    scores = torch.matmul(q.double(), k.double().transpose(-2, -1)) * scale
    if causal:
        N = scores.shape[-1]
        scores = scores.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool,
                                                          device=scores.device), 1), -1e300)
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v.double())


def run_case(q, k, v, scale, causal, name):
    if name == "B2_H2_N64_D32_causal":
        for _ in range(10):
            triton_attention(q, k, v, scale, causal)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            triton_attention(q, k, v, scale, causal)
        e.record(); torch.cuda.synchronize()
        print(f"[triton_t15_timing] {name} avg_ms={s.elapsed_time(e)/50:.4f}")
    out = triton_attention(q, k, v, scale, causal)
    ref = fp64_ref(q, k, v, scale, causal)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"triton_t15_{name}", tolerance=TOL)


if __name__ == "__main__":
    for case in make_cases():
        run_case(*case)
