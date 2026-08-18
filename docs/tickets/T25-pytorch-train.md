# T25 pytorch-train

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T24
- 唯一学习变量：最小训练循环、checkpoint 与单卡指标
- 路径覆盖：PyTorch
- 环境：gpp-core

## 范围

- 按 PyTorch 官方 tutorial 完成最小训练循环：dataloader、forward/backward、optimizer、loss、checkpoint；
- 讲解 PyTorch 训练在 AI Infra 中的定位、和纯推理 Transformers 用法的区别；
- 核心训练命令/脚本参数逐条解释（数据、batch、epoch、checkpoint 各是什么）；
- 使用 T00 固定数据集快照与 Qwen2.5-0.5B-Instruct（小步数，以官方 Quick Start 为准）；
- 记录 step 时间、samples/s、peak GPU mem、loss 曲线；
- 不引入 DeepSpeed（那是 T26）。

## 正确性门禁

- checkpoint 保存/加载 round-trip 数值一致；固定 seed 可复现。

## 性能/工具门禁

- 显存/吞吐实测与日志入 docs/evidence/T25/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T25: <名称>`。

## 验收

- 训练循环可复现；单卡基线指标完整；
- 讲义包含 PyTorch 训练定位、与推理/框架训练的区别、核心命令逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T26”才继续；本轮不实现 T26。
