"""T02 ReLU 标量版 —— 路径 4：cuTile Python。
官方依据：cuTile 官方 VectorAdd_quickstart + operations maximum（台账 S03b）。

tile 骨架沿用官方 VectorAdd_quickstart（repo 29444e0c）；
ReLU 语义用官方 operations 文档中的 `ct.maximum(tile, 0)`。
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
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=ct.where(ct.isnan(a_tile), a_tile, ct.maximum(a_tile, 0)))


def cutile_relu(a, tile_size=256):
    n = a.shape[0]
    grid = (ct.cdiv(n, tile_size), 1, 1)   # 边界：N 不整除时 grid 向上取整，load 自带越界保护
    c = cp.zeros_like(a)
    ct.launch(cp.cuda.get_current_stream(), grid, relu_kernel, (a, c, tile_size))
    return c


def main() -> None:
    for tag, N in [("aligned", 1 << 20), ("unaligned", 1_000_003)]:
        a = make_inputs(N)
        ref = cp.maximum(a.astype(cp.float64), 0).astype(cp.float32)
        out = cutile_relu(a)
        summarize_error(cp.asarray(out), cp.asarray(ref), f"cutile_relu_{tag}")

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
