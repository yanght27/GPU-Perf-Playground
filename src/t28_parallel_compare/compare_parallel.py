"""T28 并行架构统一对比：DP/DDP/ZeRO/FSDP/TP/PP/SP/CP 概览 + 训练框架对比。

官方依据：
- PyTorch DDP/FSDP 官方文档（S15）；
- DeepSpeed ZeRO 官方文档（S04）；
- 并行策略为 C 级分析（单卡/0.5B 无法实测多卡）。

用法：
    conda run --no-capture-output -n gpp-core python -I \
        src/t28_parallel_compare/compare_parallel.py
"""


def print_parallel_table():
    rows = [
        ("DP", "数据并行（旧）", "每卡复制模型，切数据", "梯度同步", "C"),
        ("DDP", "分布式数据并行", "每卡复制模型，梯度 all-reduce", "梯度通信", "B/C"),
        ("ZeRO", "分片优化器/梯度/参数", "按 stage 切分状态", "通信随 stage 增加", "B/C"),
        ("FSDP", "PyTorch 分片数据并行", "类似 ZeRO-3", "通信较多", "B/C"),
        ("TP", "张量并行", "把层内矩阵切分到多卡", "每层通信", "C"),
        ("PP", "流水线并行", "按层切分，卡间流水", "阶段间通信", "C"),
        ("SP", "序列并行", "把序列维度切分", "attention/通信", "C"),
        ("CP", "上下文并行", "长上下文切分", "跨卡 attention", "C"),
    ]
    print("[t28] parallel strategies overview")
    print(f"{'name':<6} {'what':<20} {'shard':<28} {'comm':<20} {'grade':<6}")
    for name, what, shard, comm, grade in rows:
        print(f"{name:<6} {what:<20} {shard:<28} {comm:<20} {grade:<6}")


def print_training_framework_table():
    rows = [
        ("T25 PyTorch", "手写训练循环", "无", "最底层"),
        ("T26 DeepSpeed", "训练优化库", "ZeRO", "显存/分布式优化"),
        ("T27 ms-swift", "上层工具链", "LoRA/SFT", "快速训练+部署"),
    ]
    print("[t28] training framework comparison (from T25-T27)")
    print(f"{'framework':<14} {'role':<18} {'optimization':<12} {'scenario':<24}")
    for name, role, opt, scenario in rows:
        print(f"{name:<14} {role:<18} {opt:<12} {scenario:<24}")


def main():
    print_parallel_table()
    print()
    print_training_framework_table()
    print("[t28] NOTE: 多卡并行策略在本机 0.5B/单卡环境只能做 C 级分析；"
          "真实多卡结论需要 N 卡环境复跑。")


if __name__ == "__main__":
    main()
