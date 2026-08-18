"""T10 Transpose Tile —— 路径 5：CUTLASS CuTe DSL（smem tile + padding）。

官方依据：CuTe DSL 03_gemm_tiled_smem.py 的 smem/barrier 写法（台账 S02）。
"""

import sys
from pathlib import Path
import torch, cutlass, cutlass.cute as cute
from cutlass.experimental import primitives as prims
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error

TS=32

@cute.kernel
def transpose_kernel(a:cutlass.Array,b:cutlass.Array,W:cutlass.Int32,H:cutlass.Int32,TS:cutlass.Constexpr[int]):
    tx,_,_=cute.arch.thread_idx(); _,ty,_=cute.arch.thread_idx()
    bx,_,_=cute.arch.block_idx(); _,by,_=cute.arch.block_idx()
    # 注意 CuTe Array 的维度顺序：s[行,列]
    tile=cutlass.Array(cutlass.Float32,(TS,TS+1),space=cutlass.AddressSpace.smem)
    col=bx*TS+tx; row=by*TS+ty
    if row<H and col<W:
        tile[ty,tx]=a[row,col]
    else:
        tile[ty,tx]=0.0
    prims.barrier_cta_sync(0)
    # 输出：block 交换，线程也交换（块内转置）
    out_col=by*TS+tx
    out_row=bx*TS+ty
    if out_row<W and out_col<H:
        b[out_row,out_col]=tile[tx,ty]
    prims.barrier_cta_sync(0)

@cute.jit
def cute_transpose_host(a,b,W:cutlass.Int32,H:cutlass.Int32,TS:cutlass.Constexpr[int]):
    block=(TS,TS,1); grid=((W+TS-1)//TS,(H+TS-1)//TS,1)
    transpose_kernel(a,b,W,H,TS).launch(grid=grid,block=block)

def cute_transpose(a):
    H,W=a.shape; b=torch.zeros((W,H),device='cuda',dtype=a.dtype)
    cute_transpose_host(cute.runtime.from_dlpack(a),cute.runtime.from_dlpack(b),W,H,TS)
    return b

def run(W,H):
    torch.manual_seed(0); a=torch.rand((H,W),device='cuda')
    ref=a.double().t().float(); out=cute_transpose(a)
    summarize_error(out,ref,f"cute_t10_{H}x{W}")

if __name__=="__main__":
    run(512,512); run(513,257); run(1,128)
