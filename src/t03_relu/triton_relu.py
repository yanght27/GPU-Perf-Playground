# 官方依据：Triton language tl.where/tl.maximum（v3.7.1，台账 S01b）。
"""T03 ReLU 向量化版 —— 路径 3：Triton。

Triton 的 tl.load/tl.store 是 block 级操作，编译器会自动生成 128-bit 向量化访存。
T03 不换写法，而是把“它为什么已经向量化”用 PTX 证据讲清楚。
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.t01_vector_add.common import summarize_error, effective_gbps


def make_inputs(n: int):
    i = torch.arange(n, device="cuda", dtype=torch.float32)
    return torch.where(i % 7 == 0, torch.zeros_like(i), ((i % 97).float() - 48.0) * 0.5)


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)          # 编译器会向量化这段连续 load
    # NaN 传播：x != x 只对 NaN 为真；否则 tl.maximum 与 PyTorch 参考语义一致
    tl.store(output_ptr + offsets, tl.where(x != x, x, tl.maximum(x, 0.0)), mask=mask)


def triton_relu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n, BLOCK_SIZE=block_size)
    return output


def dump_ptx_evidence() -> None:
    """把编译出的 PTX 中向量化访存指令抓出来作为证据。"""
    n = 4096
    x = torch.rand(n, device="cuda")
    y = torch.empty_like(x)
    compiled = relu_kernel.warmup(x, y, n, BLOCK_SIZE=1024, grid=(1,))
    ptx = compiled.asm["ptx"]
    for line in ptx.splitlines():
        if "ld.global" in line or "st.global" in line:
            print("[triton_relu_ptx]", line.strip())


def main() -> None:
    for tag, N in [("aligned", 1 << 20), ("unaligned", 1_000_003)]:
        a = make_inputs(N)
        ref = torch.clamp(a.double(), min=0.0).float()
        out = triton_relu(a)
        summarize_error(out, ref, f"triton_relu_{tag}")

    x = torch.tensor([float("inf"), float("-inf"), float("nan"), 1e38, -1e38],
                     device="cuda", dtype=torch.float32)
    ref_ext = torch.clamp(x.double(), min=0.0).float()
    out_ext = triton_relu(x, block_size=8)   # 小 BLOCK 也能处理短向量
    ok = torch.allclose(out_ext, ref_ext, rtol=0.0, atol=1e-5, equal_nan=True)
    print(f"[triton_relu_extreme] {'CORRECT_PASS' if ok else 'CORRECT_FAIL'} out={out_ext.tolist()}")

    dump_ptx_evidence()

    N = 1 << 20
    a = make_inputs(N)
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: triton_relu(a), quantiles=[0.5, 0.2, 0.8]
    )
    print(
        f"[triton_relu] median_ms={ms:.4f} min_ms={min_ms:.4f} max_ms={max_ms:.4f} "
        f"effective_gbps={effective_gbps(N, ms / 1e3, accesses=2):.1f}"
    )


if __name__ == "__main__":
    main()
