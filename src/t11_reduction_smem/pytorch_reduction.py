"""T11 Reduction 共享内存规约 —— 路径 1：PyTorch 参考。

官方依据：torch.Tensor.sum / Tensor.double（PyTorch 2.13 文档，台账 S15c）。
本路径角色：黑盒参考。a.sum() 是“被测实现”，a.double().sum() 是 fp64 黄金参考。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 找到仓库根目录
from src.t01_vector_add.common import summarize_error           # 统一误差打印（工程杂务）

TOL = 1e-4  # reduction 的 fp32 求和顺序会带来约 1e-7~4e-5 的误差，容差相应放宽


def run(N, seed=0):
    torch.manual_seed(seed)                     # 固定随机种子，保证三 shape 可复现
    a = torch.rand(N, device="cuda") * 0.001    # GPU 上 N 个 [0,0.001) 的 fp32 数
    out = a.sum()                               # 路径实现：PyTorch 官方 fp32 求和
    ref = a.double().sum()                      # 黄金参考：同一数组转 fp64 再求和
    summarize_error(out, ref, f"pytorch_t11_{N}", tolerance=TOL)


if __name__ == "__main__":
    run(1048576)  # 2^20：完整覆盖
    run(999983)   # 素数：N 不被 block 整除的边界
    run(1)        # 单元素：grid 比数据大的边界
