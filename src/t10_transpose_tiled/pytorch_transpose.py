"""T10 Transpose Tile —— 路径 1：PyTorch 参考。

官方依据：torch.Tensor.t / contiguous（PyTorch 2.13 文档，台账 S15）。
"""

import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

def run(W,H):
    torch.manual_seed(0)
    a=torch.rand((H,W),device='cuda')
    ref=a.double().t().float()
    out=a.t().contiguous()
    summarize_error(out,ref,f"pytorch_t10_{H}x{W}")

if __name__=="__main__":
    run(512,512); run(513,257); run(1,128)
