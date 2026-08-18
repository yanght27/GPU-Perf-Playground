"""T01 Vector Add —— 路径 3：Triton。
官方依据：Triton 官方 tutorial 01-vector-add.py（台账 S01a）。

kernel 主体与官方 tutorial 01-vector-add（tag v3.7.1，文件
python/tutorials/01-vector-add.py）一致：triton.jit、program_id、tl.arange、
mask、tl.load/tl.store。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps

N = 1 << 20


@triton.jit  # 告诉 Triton：这是一个 kernel，会被即时编译（JIT）成 GPU 代码
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """官方 tutorial 的加法 kernel，唯一学习主线。

    - 每个 “program” 处理 BLOCK_SIZE 个连续元素；
    - grid 里有多少个 program 由 host 端 launch 时决定。
    """
    pid = tl.program_id(axis=0)          # 我是第几个 program（对应 CUDA 的 block）
    block_start = pid * BLOCK_SIZE       # 我这个 program 负责的起点
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 一个 [0..BLOCK_SIZE) 的整数向量
    mask = offsets < n_elements          # N 不被 BLOCK_SIZE 整除时，把越界位置遮掉
    x = tl.load(x_ptr + offsets, mask=mask)  # 只 load 合法位置
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)  # 只 store 合法位置


def triton_vadd(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """host 端封装：分配输出，计算 grid，启动 kernel。"""
    output = torch.empty_like(x)
    n = output.numel()
    # grid：一共需要 ceil(N / BLOCK_SIZE) 个 program。
    # lambda meta 是官方写法，Triton 会为每个 kernel 元参数生成对应 grid。
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=block_size)
    return output


def main() -> None:
    torch.manual_seed(0)
    a32 = torch.rand(N, dtype=torch.float32, device="cuda")
    b32 = torch.rand(N, dtype=torch.float32, device="cuda")
    ref = (a32.double() + b32.double()).float()

    out = triton_vadd(a32, b32)
    summarize_error(out, ref, "triton_fp32_vs_fp64")

    # 官方推荐的 benchmark 工具：triton.testing.do_bench。
    # 它内部处理了 warmup、同步和分位数统计。
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: triton_vadd(a32, b32), quantiles=[0.5, 0.2, 0.8]
    )
    print(
        f"[triton] median_ms={ms:.4f} min_ms={min_ms:.4f} max_ms={max_ms:.4f} "
        f"effective_gbps={effective_gbps(N, ms / 1e3):.1f}"
    )


if __name__ == "__main__":
    main()
