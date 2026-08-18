"""T18 CUDA FA 与 PyTorch SDPA 的直接对比（同一批 q/k/v/o 二进制文件）。

官方依据：PyTorch F.scaled_dot_product_attention（台账 S15e）。工程件说明：本文件
不实现算子，只读 CUDA 程序 dump 的二进制并执行 SDPA 黄金参考比较。

CUDA 程序会把每个 case 的 q/k/v/o 写到 /tmp/t18_*；本脚本读取同一数据，
用 SDPA 计算参考并与 CUDA o 比较，满足任务书“以 SDPA 为黄金参考”。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

CASES = [
    (4, 4, 512, 64, 1),
    (4, 4, 1024, 64, 1),
    (4, 4, 2048, 64, 1),
    (2, 4, 512, 64, 0),
]
TOL = 1e-2


def run():
    for B, H, N, D, causal in CASES:
        base = Path(f"/tmp/t18_{N}_{causal}")
        q = torch.from_file(str(base) + "_q.bin", shared=False, size=int(B * H * N * D), dtype=torch.float32).cuda().reshape(B, H, N, D)
        k = torch.from_file(str(base) + "_k.bin", shared=False, size=int(B * H * N * D), dtype=torch.float32).cuda().reshape(B, H, N, D)
        v = torch.from_file(str(base) + "_v.bin", shared=False, size=int(B * H * N * D), dtype=torch.float32).cuda().reshape(B, H, N, D)
        o = torch.from_file(str(base) + "_o.bin", shared=False, size=int(B * H * N * D), dtype=torch.float32).cuda().reshape(B, H, N, D)
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                             is_causal=bool(causal), scale=D ** -0.5)
        torch.cuda.synchronize()
        summarize_error(o, ref, f"cuda_vs_sdpa_t18_B{B}_H{H}_N{N}_D{D}_causal{causal}", tolerance=TOL)


if __name__ == "__main__":
    run()
