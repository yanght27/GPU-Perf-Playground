"""T10 Transpose Tile —— 路径 3：Triton（tile + tl.trans）。

官方依据：Triton 语言 `tl.trans` 与 tutorial 01/03 的 tile/mask 写法（台账 S01）。
"""

import sys
from pathlib import Path
import torch, triton, triton.language as tl
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

BS=32

@triton.jit
def transpose_kernel(x_ptr,y_ptr,H,W,BS:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1)
    rm=pm*BS+tl.arange(0,BS); rn=pn*BS+tl.arange(0,BS)
    t=tl.load(x_ptr+rm[:,None]*W+rn[None,:], mask=(rm[:,None]<H)&(rn[None,:]<W), other=0.0)
    t=tl.trans(t,1,0)
    om=pn*BS+tl.arange(0,BS); on=pm*BS+tl.arange(0,BS)
    tl.store(y_ptr+om[:,None]*H+on[None,:], t, mask=(om[:,None]<W)&(on[None,:]<H))

def triton_transpose(a):
    H,W=a.shape; y=torch.empty((W,H),device='cuda',dtype=a.dtype)
    transpose_kernel[(triton.cdiv(H,BS),triton.cdiv(W,BS))](a,y,H,W,BS=BS)
    return y

def run(W,H):
    torch.manual_seed(0); a=torch.rand((H,W),device='cuda')
    ref=a.double().t().float(); out=triton_transpose(a)
    summarize_error(out,ref,f"triton_t10_{H}x{W}")

if __name__=="__main__":
    run(512,512); run(513,257); run(1,128)
