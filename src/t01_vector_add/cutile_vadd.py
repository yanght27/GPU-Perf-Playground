"""T01 Vector Add —— 路径 4：cuTile Python。
官方依据：cuTile 官方 VectorAdd_quickstart.py（台账 S03a）。

kernel 主体与 NVIDIA/cutile-python 官方 Quick Start 一致（samples/quickstart/
VectorAdd_quickstart.py，repo commit 29444e0c）。
"""

import sys
from pathlib import Path

import cupy as cp
import cuda.tile as ct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps

N = 1 << 20
TILE_SIZE = 256  # 每个 “processor”（类比 CUDA 的 block）处理 256 个元素


@ct.kernel  # 标记为 cuTile kernel：写的是 Python，编译后生成 GPU 代码
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    """官方示例的 tile 式加法。

    cuTile 的编程单位是 “tile”，而不是单个线程：
    一次 ct.load 就搬一整个 tile，a_tile + b_tile 是对整个 tile 的元素级加法。
    """
    pid = ct.bid(0)  # 当前 processor 在 grid 里的编号（0 维）
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))  # 从 a 的 pid 处取一个 tile
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)        # 算完整 tile 写回


def cutile_vadd(a, b, tile_size=TILE_SIZE):
    """host 端封装：算 grid、launch。"""
    n = a.shape[0]
    grid = (ct.cdiv(n, tile_size), 1, 1)  # grid 大小 = ceil(N / tile)
    c = cp.zeros_like(a)
    ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, c, tile_size))
    return c


def main() -> None:
    # 固定 seed 的 CuPy 随机输入；参考用 float64 计算（Numpy/CuPy 与 PyTorch 语义一致）
    rng = cp.random.default_rng(0)
    a32 = rng.random(N, dtype=cp.float32)
    b32 = rng.random(N, dtype=cp.float32)
    ref = (a32.astype(cp.float64) + b32.astype(cp.float64)).astype(cp.float32)

    out = cutile_vadd(a32, b32)
    summarize_error(cp.asarray(out), cp.asarray(ref), "cutile_fp32_vs_fp64")

    # CuPy event 计时：与 CUDA event 同一个原理，量 GPU 时间线
    ITERS = 100
    for _ in range(10):
        cutile_vadd(a32, b32)
    cp.cuda.Stream.null.synchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(ITERS):
        cutile_vadd(a32, b32)
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end) / ITERS
    print(f"[cutile] avg_ms={ms:.4f} effective_gbps={effective_gbps(N, ms / 1e3):.1f}")


if __name__ == "__main__":
    main()
