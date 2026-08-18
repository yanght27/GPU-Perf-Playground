# T28 parallel-compare

- 状态：`done`（2026-08-17 学习者全面验收通过；多卡实测标 C，恢复路径已写）
- 前置：T27
- 唯一学习变量：DP/DDP/ZeRO/FSDP/TP/PP/SP/CP 原理、通信与取舍
- 路径覆盖：PyTorch/DeepSpeed/ms-swift 证据汇总
- 环境：gpp-core / gpp-deepspeed-0.19.5 / gpp-swift-4.4.3

## 范围

- 逐项讲解 DP、DDP、ZeRO-1/2/3、FSDP、TP、PP、SP、CP：切分什么、通信什么、显存/计算/带宽权衡；
- 先回顾 PyTorch/DeepSpeed/ms-swift 在并行训练中的工具定位和命令行入口差异；
- 全部基于 PyTorch/DeepSpeed/官方论文或官方文档；
- 单卡只能实测的部分标 A，只能运行的标 B，多卡并行标 C 并写“需要 N 卡 + 官方命令”恢复路径；
- 输出 T25–T27 统一对比表与结论。

## 正确性门禁

- 结论与原始日志可追溯；C 级内容明确不冒充实测。

## 性能/工具门禁

- 证据汇总入 docs/evidence/T28/（引用 T25–T27）。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T28: <名称>`。

## 验收

- 每个并行策略都有官方来源与 A/B/C 分级；课程主线闭环；
- 结论必须能追溯到各训练工具/框架的定位、命令行与原始日志差异。

## 解锁

- T28 是主线闭环；之后只做复习、补证据或学习者指定的新方向。
