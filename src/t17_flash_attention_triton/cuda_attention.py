"""T17 FlashAttention（Triton 版）—— 路径 3：CUDA C++（N/A 检查 + 归属 T18）。

官方依据：Triton tutorial 06（S01o）与 T18 任务书；CUDA 版 FA 是本教程算法的
手工映射，按增量纪律由 T18 专门实现。
本文件只做 N/A 层级说明，不实现 kernel。
"""


def main():
    print("[cuda_t17] N/A: CUDA FlashAttention is the dedicated T18 increment. "
          "T17 is scoped to the Triton official tutorial forward. Source pointer: S01o.")


if __name__ == "__main__":
    main()
