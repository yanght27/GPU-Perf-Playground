# 工程公共逻辑：计时/误差换算。不调用第三方算子 API；各算子的官方依据见对应实现文件与 source-ledger。
"""T01 Vector Add：所有实现共享的“工程杂务”。

按学习者约定：把与算子学习无关的公共逻辑（带宽换算、误差汇总）集中到这里，
让每个实现文件只突出该工具写 kernel 的主体部分。
"""


def bytes_moved_gb(n_elements: int, dtype_size: int = 4, accesses: int = 3) -> float:
    """一次元素级算子的总访存量 = accesses * N * dtype_size。

    默认 accesses=3 对应 vector add：读 a、读 b、写 c。
    ReLU 是 accesses=2：读 x、写 y。
    """
    return float(accesses) * n_elements * dtype_size / 1e9  # GB


def effective_gbps(n_elements: int, seconds: float, dtype_size: int = 4, accesses: int = 3) -> float:
    """有效带宽 = 总移动字节数 / 耗时（GB/s）。"""
    return bytes_moved_gb(n_elements, dtype_size, accesses) / seconds


def summarize_error(actual, expected, name: str, tolerance: float = 1e-5) -> None:
    """统一打印误差结论，避免每个实现各写一套判断逻辑。

    同时兼容 torch.Tensor 和 numpy/cupy ndarray：优先用 .abs()，否则 np.abs()。
    """
    if hasattr(actual, "abs"):
        diff = (actual - expected).abs()
    else:
        import numpy as np

        diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    ok = max_abs <= tolerance
    print(
        f"[{name}] max_abs_err={max_abs:.6e} tolerance={tolerance} "
        f"{'CORRECT_PASS' if ok else 'CORRECT_FAIL'}"
    )
