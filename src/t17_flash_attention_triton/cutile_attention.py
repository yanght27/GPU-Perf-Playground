"""T17 Attention —— 路径 4：cuTile Python（官方 FMHA 能力实测 + 层级 N/A 说明）。

官方依据：cuTile 官方 samples/AttentionFMHA.py（台账 S03m）。
本文件实际运行官方 `cutile_fmha`（fp16，causal/非 causal 各一次）并与 PyTorch SDPA
对照；这证明官方 attention 能力在本机 gpp-cutile（含用户新装的 torch）可运行。
T17 学习变量是 Triton tutorial 的 FA 教学版；官方示例是框架 flash 层，因此路径记
  层级 N/A，但官方能力已实测通过。
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _cutile_samples_dir() -> Path:
    """定位 cuTile 官方 sample 目录：优先仓库内 third_party/cutile-python，其次环境变量，最后本机 /tmp。"""
    env = os.environ.get("CUTILE_PYTHON_SAMPLES")
    if env:
        return Path(env)
    repo_samples = Path(__file__).resolve().parents[2] / "third_party" / "cutile-python" / "samples"
    if repo_samples.exists():
        return repo_samples
    return Path("/tmp/cutile-python/samples")


sys.path.insert(0, str(_cutile_samples_dir()))   # 官方 sample 目录（utils 依赖）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import AttentionFMHA as fmha
from src.t01_vector_add.common import summarize_error

TOL = 1e-2


def main():
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
        summarize_error(out, ref, f"cutile_t17_fmha_causal{causal}", tolerance=TOL)
    print(f"[cutile_t17] official FMHA capability run PASS; layer note: official sample is framework flash layer (not T17 Triton tutorial path)")


if __name__ == "__main__":
    main()
