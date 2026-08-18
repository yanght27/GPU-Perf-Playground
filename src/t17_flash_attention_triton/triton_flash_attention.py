"""T17 FlashAttention（Triton 官方 tutorial 版）—— 路径 2：Triton forward。

官方依据：Triton 官方 tutorial 06-fused-attention 的 FA2 算法结构（台账 S01o）：
  - 外层 K/V tile 循环（start_n += BLOCK_N），内层 Q tile（BLOCK_M rows）；
  - running max/sum（m_i/l_i）online rescale，acc 同步 rescale；
  - tl.dot(q, k^T) 与 tl.dot(p, v)。
本文件采用官方 tutorial 的数值结构（float16 + fp32 accumulator），省略 TensorDescriptor
与 warp-specialization 等 Hopper/Blackwell 专属加速，适合 sm_89 教学复现。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-2
BLOCK_M = 64
BLOCK_N = 64
HEAD_DIM = 64


@triton.jit
def _fwd_kernel(Q, K, V, O, scale, BH, N, D: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)                # Q tile：第几组 BLOCK_M 行
    off_hz = tl.program_id(1)                 # batch*head 铺平下标

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + off_hz * N * D + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_m < N)[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 外层 K/V tile 循环：每个 tile 只读一次，online 修正 (m,l,acc)
    for start_n in range(0, N, BLOCK_N):
        n_mask = (start_n + offs_n) < N
        k_ptrs = K + off_hz * N * D + (start_n + offs_n)[:, None] * D + offs_d[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))           # [BLOCK_M, BLOCK_N]
        qk = qk * scale                        # scale 已在 host 乘 log2(e)，配合 exp2
        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))   # 新 running max
        p = tl.math.exp2(qk - m_ij[:, None])         # 官方 exp2：与 tutorial 06 同数值路径
        alpha = tl.math.exp2(m_i - m_ij)             # 旧基准 -> 新基准的 rescale 系数
        l_i = l_i * alpha + tl.sum(p, axis=1)        # online 分母

        v_ptrs = V + off_hz * N * D + (start_n + offs_n)[:, None] * D + offs_d[None, :]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)   # 旧 O 同步 rescale + 新 PV
        m_i = m_ij

    acc = acc / l_i[:, None]                   # 最终归一化
    o_ptrs = O + off_hz * N * D + offs_m[:, None] * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.float16), mask=(offs_m < N)[:, None])


def flash_attention(q, k, v, causal, sm_scale=None):
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D ** -0.5
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    qk_scale = sm_scale * 1.4426950408889634     # 1/log(2)：官方 tutorial 的 exp2 换算
    _fwd_kernel[grid](q, k, v, o, qk_scale, B * H, N, D,
                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
                      num_warps=4, num_stages=3)
    return o


def dump_codegen_evidence(N=1024):
    """保存 FA kernel 的 PTX 与 cubin，供 cuobjdump 生成 SASS 证据。"""
    from pathlib import Path as _P
    ev = _P(__file__).resolve().parents[2] / "docs" / "evidence" / "T17"
    ev.mkdir(parents=True, exist_ok=True)
    q, k, v = make_case(4, 4, N, True)
    grid = (triton.cdiv(N, BLOCK_M), 4 * 4)
    h = _fwd_kernel[grid](q, k, v, torch.empty_like(q), HEAD_DIM ** -0.5,
                          16, N, HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                          CAUSAL=True, num_warps=4, num_stages=3)
    (ev / "t17-triton-ptx.txt").write_text(h.asm["ptx"])
    (ev / "t17-triton.cubin").write_bytes(h.asm["cubin"])
    print("[triton_fa_t17_codegen] ptx_and_cubin_saved")


def make_case(B, H, N, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.2
    k = torch.randn(B, H, N, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.2
    v = torch.randn(B, H, N, HEAD_DIM, device="cuda", dtype=torch.float16) * 0.2
    return q, k, v


def run_case(B, H, N, causal):
    q, k, v = make_case(B, H, N, causal)
    out = flash_attention(q, k, v, causal)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                         is_causal=causal, scale=HEAD_DIM ** -0.5)
    torch.cuda.synchronize()
    summarize_error(out, ref, f"triton_fa_t17_B{B}_H{H}_N{N}_D{HEAD_DIM}_causal{causal}", tolerance=TOL)

    # 只对主 shape 计时（warmup + 20 次 event）
    if N == 1024 and causal:
        for _ in range(5):
            flash_attention(q, k, v, causal)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):
            flash_attention(q, k, v, causal)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 20
        print(f"[triton_fa_t17_timing] B{B}_H{H}_N{N}_D{HEAD_DIM} avg_ms={ms:.4f}")
        s.record()
        for _ in range(20):
            F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                           is_causal=causal, scale=HEAD_DIM ** -0.5)
        e.record(); torch.cuda.synchronize()
        ms_sdpa = s.elapsed_time(e) / 20
        print(f"[sdpa_t17_timing] B{B}_H{H}_N{N}_D{HEAD_DIM} avg_ms={ms_sdpa:.4f}")


def time_compare(N):
    """T15 朴素 Triton（每 query 重读整段 K/V） vs T17 FA（K/V tile 只读一次）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t15_triton", str(Path(__file__).resolve().parents[2] /
                          "src/t15_attention_naive/triton_attention.py"))
    t15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t15)
    q, k, v = make_case(4, 4, N, True)
    for name, fn in [("t15", t15.triton_attention), ("fa", flash_attention)]:
        for _ in range(3):
            if name == "t15":
                fn(q, k, v, HEAD_DIM ** -0.5, True)
            else:
                fn(q, k, v, True)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(5):
            if name == "t15":
                fn(q, k, v, HEAD_DIM ** -0.5, True)
            else:
                fn(q, k, v, True)
        e.record(); torch.cuda.synchronize()
        print(f"[{name}_t17_vs] N={N} avg_ms={s.elapsed_time(e)/5:.4f}")


if __name__ == "__main__":
    run_case(4, 4, 512, True)
    run_case(4, 4, 1024, True)
    run_case(4, 4, 2048, True)
    run_case(2, 4, 512, False)
    dump_codegen_evidence(1024)
    time_compare(512)
    time_compare(1024)
    time_compare(2048)
