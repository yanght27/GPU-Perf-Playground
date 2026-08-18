"""T03 ReLU 向量化版 —— 路径 4：cuTile Python。
官方依据：cuTile 官方 VectorAdd_quickstart / ct.load/store（台账 S03c）。

cuTile 的 ct.load/ct.store 是 tile 级操作，编译器会自动选择向量化访存（T02 的 NCU 已显示
DRAM 88.47%、Duration 18.82us，与 CUDA float4 同梯队）。因此 T03 不改 kernel 写法，
而是把“为什么它已经是向量化”作为知识点讲清楚；代码与官方 VectorAdd_quickstart 同构。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps


def make_inputs(n: int):
    i = cp.arange(n, dtype=cp.float32)
    return cp.where(i % 7 == 0, 0.0, ((i % 97).astype(cp.float32) - 48.0) * 0.5)


@ct.kernel
def relu_kernel(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))   # tile 级 load：内部自动向量化
    # NaN 传播：与 PyTorch F.relu(NaN)=NaN 参考语义对齐（官方 isnan/where/maximum）
    out_tile = ct.where(ct.isnan(a_tile), a_tile, ct.maximum(a_tile, 0))
    ct.store(c, index=(pid,), tile=out_tile)                # tile 级 store


def cutile_relu(a, tile_size=256):
    n = a.shape[0]
    grid = (ct.cdiv(n, tile_size), 1, 1)
    c = cp.zeros_like(a)
    ct.launch(cp.cuda.get_current_stream(), grid, relu_kernel, (a, c, tile_size))
    return c


def main() -> None:
    for tag, N in [("aligned", 1 << 20), ("unaligned", 1_000_003)]:
        a = make_inputs(N)
        ref = cp.maximum(a.astype(cp.float64), 0).astype(cp.float32)
        out = cutile_relu(a)
        summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_relu_{tag}")

    x = cp.array([cp.inf, -cp.inf, cp.nan, 1e38, -1e38], dtype=cp.float32)
    ref_ext = cp.maximum(x.astype(cp.float64), 0).astype(cp.float32)
    out_ext = cutile_relu(x, tile_size=8)
    ok = cp.allclose(out_ext, ref_ext, rtol=0.0, atol=1e-5, equal_nan=True)
    print(f"[cutile_relu_extreme] {'CORRECT_PASS' if bool(ok) else 'CORRECT_FAIL'} out={out_ext.tolist()}")

    N = 1 << 20
    a = make_inputs(N)
    ITERS = 100
    for _ in range(10):
        cutile_relu(a)
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(ITERS):
        cutile_relu(a)
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end) / ITERS
    print(f"[cutile_relu] avg_ms={ms:.4f} effective_gbps={effective_gbps(N, ms / 1e3, accesses=2):.1f}")


if __name__ == "__main__":
    main()
