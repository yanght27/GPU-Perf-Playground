"""T16 KV Cache —— 路径 4：cuTile Python（官方 KV-Cache 能力检查 + 最近官方能力实测）。

官方依据：cuTile 官方 samples/AttentionFMHA.py（台账 S03m）。
v1.7 五路径规则：T16 学习变量是 KV Cache（增量 append 历史 K/V）。cuTile 1.5.0
官方仓库没有 KV-Cache 专项 attention 示例，因此机制记 N/A；最接近的官方能力是
AttentionFMHA（flash/online-tiled 前向），本文件在同一路径槽内完成两件事：
1) 静态检查官方 sample 是否存在 KV-Cache 能力；
2) 实际运行官方 `cutile_fmha`（fp16，causal/非 causal）并与 PyTorch SDPA 对照，
   证明最接近官方 attention 能力在本机 gpp-cutile（含用户新装的 torch）可运行。
"""

import os
import sys
from pathlib import Path


def _cutile_repo() -> Path:
    """定位 cuTile 官方仓库：优先仓库内 third_party/cutile-python，其次环境变量，最后本机 /tmp。"""
    env = os.environ.get("CUTILE_PYTHON_ROOT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[2] / "third_party" / "cutile-python"
    if repo.exists():
        return repo
    return Path("/tmp/cutile-python")


REPO = _cutile_repo()
FMHA = REPO / "samples" / "AttentionFMHA.py"
TOL = 1e-2


def capability_check():
    if not FMHA.exists():
        raise SystemExit(f"official cuTile attention sample not found: {FMHA}")
    text = FMHA.read_text()
    has_attention = "Fused Multi-Head Attention" in text
    has_kv_cache = any(tok in text for tok in ("KV Cache", "kv_cache", "past_key_values"))
    print(f"[cutile_t16] official={FMHA} has_attention={has_attention} has_kv_cache_example={has_kv_cache}")
    print("[cutile_t16] N/A: cuTile has no KV-Cache append example; nearest official is "
          "flash/online-tiled AttentionFMHA. Capability pointer: S03m.")


def capability_run():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(FMHA.parent))   # 官方 sample 目录（utils 依赖）
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import AttentionFMHA as fmha
    from src.t01_vector_add.common import summarize_error

    torch.manual_seed(0)
    B, H, N, D = 2, 8, 128, 64
    q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.2
    k = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.2
    v = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda") * 0.2
    for causal in (False, True):
        out = fmha.cutile_fmha(q, k, v, qk_scale=D ** -0.5,
                               tile_m=128, tile_n=128, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=D ** -0.5)
        torch.cuda.synchronize()
        summarize_error(out, ref, f"cutile_t16_fmha_causal{causal}", tolerance=TOL)
    print("[cutile_t16] official FMHA capability run PASS; layer note: official sample "
          "has no KV-cache append (not T16 mechanism)")


if __name__ == "__main__":
    capability_check()
    capability_run()
