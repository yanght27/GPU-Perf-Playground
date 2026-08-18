"""T16 KV Cache —— 路径 1：PyTorch 语义参考（对齐 Transformers DynamicCache）。

官方依据：HuggingFace Transformers `cache_utils.DynamicCache`（台账 S16a，本机
transformers 5.14.1；key/value 形状 [B,H,seq,D]，update 后返回增长后的 cache）。
本路径用最小 attention 层比较 use_cache=False/True，不引入生成循环：
prefill 16 个 token -> 16 步 decode；无 cache 每步重投整段 K/V，有 cache 只投当前
token 并 append。
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TOL = 1e-4
B, H, T, D, LPRE = 2, 4, 512, 128, 256
DEC = T - LPRE


def project(x, w):
    return torch.einsum("bhtd,de->bhte", x, w)   # 最小线性投影：X @ W


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn(B, H, T, D, device="cuda") * 0.5
    wq = torch.randn(D, D, device="cuda") * (D ** -0.5)
    wk = torch.randn(D, D, device="cuda") * (D ** -0.5)
    wv = torch.randn(D, D, device="cuda") * (D ** -0.5)
    return x, wq, wk, wv


def decode_no_cache(x, wq, wk, wv):
    q_all = project(x, wq)
    outs = []
    for t in range(LPRE, T):
        k = project(x[:, :, :t + 1, :], wk)       # 每步重投全部历史 K
        v = project(x[:, :, :t + 1, :], wv)       # 每步重投全部历史 V
        out = F.scaled_dot_product_attention(q_all[:, :, t:t + 1, :], k, v,
                                             attn_mask=None, dropout_p=0.0, scale=D ** -0.5)
        outs.append(out)
    return torch.cat(outs, dim=2)


def decode_cache(x, wq, wk, wv):
    q_all = project(x, wq)
    cache = DynamicCache()
    k_pre = project(x[:, :, :LPRE, :], wk)        # prefill：一次投影并写入 cache
    v_pre = project(x[:, :, :LPRE, :], wv)
    k, v = cache.update(k_pre, v_pre, layer_idx=0)
    outs = []
    for t in range(LPRE, T):
        k_new = project(x[:, :, t:t + 1, :], wk)  # decode：只投影当前 token
        v_new = project(x[:, :, t:t + 1, :], wv)
        k, v = cache.update(k_new, v_new, layer_idx=0)  # 官方 append 语义
        out = F.scaled_dot_product_attention(q_all[:, :, t:t + 1, :], k, v,
                                             attn_mask=None, dropout_p=0.0, scale=D ** -0.5)
        outs.append(out)
    return torch.cat(outs, dim=2)


def timeit(fn, *args):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); out = fn(*args); e.record(); torch.cuda.synchronize()
    return out, s.elapsed_time(e)


if __name__ == "__main__":
    x, wq, wk, wv = make_inputs()
    out_no, ms_no = timeit(decode_no_cache, x, wq, wk, wv)
    out_cache, ms_cache = timeit(decode_cache, x, wq, wk, wv)
    summarize_error(out_cache, out_no, "pytorch_t16_cache_vs_nocache", tolerance=TOL)
    print(f"[pytorch_t16_timing] no_cache_ms={ms_no:.3f} cache_ms={ms_cache:.3f} speedup={ms_no/ms_cache:.2f}x")
    cache_bytes = 2 * B * H * T * D * 4
    print(f"[pytorch_t16_cache_bytes] final_kv_cache_bytes={cache_bytes} "
          f"no_cache_reprojected_elems_per_step_avg={sum(t+1 for t in range(LPRE,T))*2*B*H*D}")
