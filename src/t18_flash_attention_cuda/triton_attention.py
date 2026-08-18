"""T18 FlashAttention CUDA 版 —— 路径 3：Triton（N/A 层级说明，归属 T17）。

官方依据：Triton 官方 tutorial 06（S01o）。T17 已完成 Triton 教学版与官方完整
forward 实测；T18 只实现 CUDA 手工映射，因此本路径为显式 N/A，指向 T17 证据。
"""


def main():
    print("[triton_t18] N/A: Triton FA is T17 (implemented and verified). "
          "See docs/evidence/T17/t17-run-all.txt and t17-official-forward.txt.")


if __name__ == "__main__":
    main()
