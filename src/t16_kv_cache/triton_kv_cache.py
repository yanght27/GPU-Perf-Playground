"""T16 KV Cache —— 路径 3：Triton（append + 带缓存 decode 注意力）。

官方依据：Transformers DynamicCache 的 append 语义（S16a）；Triton 注意力步骤沿用
官方 tutorial 06 的 qk/softmax/pv（S01o）。
本路径用 torch 做最小线性投影（与 PyTorch 参考同语义），用 Triton 写 cache append 与
单 query decode 注意力，对比“每步重投全部 K/V”和“只投当前 token + append”。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-4
B, H, T, D, LPRE = 2, 2, 128, 64, 64
BH = B * H
DEC = T - LPRE


@triton.jit
def append_kv_kernel(src, dst, offset, DD, TT, BD: tl.constexpr):
    bh = tl.program_id(0)
    offs = tl.arange(0, BD)
    mask = offs < DD
    v = tl.load(src + bh * DD + offs, mask=mask)
    tl.store(dst + (bh * TT + offset) * DD + offs, v, mask=mask)


@triton.jit
def attention_decode_kernel(Q, K, V, O, t, N, DD, QT, KT, scale,
                            BN: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    offs_k = tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    mask_k = offs_k < N
    mask_d = offs_d < DD

    qv = tl.load(Q + (bh * QT + t) * DD + offs_d, mask=mask_d, other=0.0)
    kv = tl.load(K + (bh * KT + offs_k[:, None]) * DD + offs_d[None, :],
                 mask=mask_k[:, None] & mask_d[None, :], other=0.0)
    scores = tl.sum(qv[None, :] * kv, axis=1) * scale
    scores = tl.where(mask_k, scores, -float("inf"))
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom
    vv = tl.load(V + (bh * KT + offs_k[:, None]) * DD + offs_d[None, :],
                 mask=mask_k[:, None] & mask_d[None, :], other=0.0)
    o = tl.sum(p[:, None] * vv, axis=0)
    tl.store(O + (bh * QT + t) * DD + offs_d, o, mask=mask_d)


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(B, H, T, D, device="cuda") * 0.5
    wq = torch.randn(D, D, device="cuda") * (D ** -0.5)
    wk = torch.randn(D, D, device="cuda") * (D ** -0.5)
    wv = torch.randn(D, D, device="cuda") * (D ** -0.5)
    return x, wq, wk, wv


def project(x, w):
    return torch.einsum("bhtd,de->bhte", x, w)


def decode_no_cache(x, wq, wk, wv):
    q = project(x, wq).reshape(BH, T, D)
    k = project(x, wk).reshape(BH, T, D)
    v = project(x, wv).reshape(BH, T, D)
    o = torch.empty(BH, T, D, device=x.device)
    BN = triton.next_power_of_2(T)
    BD = triton.next_power_of_2(D)
    for t in range(LPRE, T):
        # 无缓存：每步重新投影 0..t 的全部历史 K/V
        kt = project(x[:, :, :t + 1], wk).reshape(BH, t + 1, D)
        vt = project(x[:, :, :t + 1], wv).reshape(BH, t + 1, D)
        attention_decode_kernel[(BH,)](q, kt, vt, o, t, t + 1, D, T, t + 1,
                                       D ** -0.5, BN=BN, BD=BD, num_warps=4)
    return o[:, LPRE:, :].reshape(B, H, DEC, D)


def decode_cache(x, wq, wk, wv):
    q = project(x, wq).reshape(BH, T, D)
    kcache = torch.zeros(BH, T, D, device=x.device)
    vcache = torch.zeros(BH, T, D, device=x.device)
    kpre = project(x[:, :, :LPRE], wk).reshape(BH, LPRE, D)
    vpre = project(x[:, :, :LPRE], wv).reshape(BH, LPRE, D)
    kcache[:, :LPRE] = kpre
    vcache[:, :LPRE] = vpre
    o = torch.empty(BH, T, D, device=x.device)
    BN = triton.next_power_of_2(T)
    BD = triton.next_power_of_2(D)
    for t in range(LPRE, T):
        ktok = project(x[:, :, t:t + 1], wk).reshape(BH, D)
        vtok = project(x[:, :, t:t + 1], wv).reshape(BH, D)
        append_kv_kernel[(BH,)](ktok, kcache, t, D, T, BD=BD)
        append_kv_kernel[(BH,)](vtok, vcache, t, D, T, BD=BD)
        attention_decode_kernel[(BH,)](q, kcache, vcache, o, t, t + 1, D, T, T,
                                       D ** -0.5, BN=BN, BD=BD, num_warps=4)
    return o[:, LPRE:, :].reshape(B, H, DEC, D)


if __name__ == "__main__":
    x, wq, wk, wv = make_inputs()
    out_no = decode_no_cache(x, wq, wk, wv)
    out_cache = decode_cache(x, wq, wk, wv)
    torch.cuda.synchronize()
    summarize_error(out_cache, out_no, "triton_t16_cache_vs_nocache", tolerance=TOL)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); decode_no_cache(x, wq, wk, wv); e.record(); torch.cuda.synchronize(); ms_no = s.elapsed_time(e)
    s.record(); decode_cache(x, wq, wk, wv); e.record(); torch.cuda.synchronize(); ms_cache = s.elapsed_time(e)
    print(f"[triton_t16_timing] no_cache_ms={ms_no:.3f} cache_ms={ms_cache:.3f} speedup={ms_no/ms_cache:.2f}x")
    print(f"[triton_t16_cache_bytes] final_kv_cache_bytes={2*BH*T*D*4}")
