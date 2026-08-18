# T27 ms-swift-train

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T26
- 唯一学习变量：ms-swift 官方 Qwen SFT/LoRA Quick Start
- 路径覆盖：ms-swift
- 环境：gpp-swift-4.4.3（按 T00 修正）

## 范围

- 按 ms-swift 官方 docs 的 SFT/LoRA Quick Start，使用固定模型与数据集快照；
- 讲解 ms-swift 训练入口是什么、和纯 PyTorch/DeepSpeed 训练的区别；
- 核心训练命令逐条解释（模型、数据集、LoRA 参数、输出目录、checkpoint 等）；
- 训练最小步数，记录命令行、loss、显存、checkpoint；
- 解释 LoRA 与全参 SFT 的显存/吞吐差别（官方文档为据）；
- 与 T25/T26 对比入口差异。

## 正确性门禁

- 训练后模型可加载推理；固定 seed 流程可复现。

## 性能/工具门禁

- 官方 metrics 与日志入 docs/evidence/T27/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T27: <名称>`。

## 验收

- 官方 Quick Start 命令可复现；LoRA/全参取舍有依据；
- 讲义包含 ms-swift 训练定位、与 PyTorch/DeepSpeed 的区别、核心命令逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T28”才继续；本轮不实现 T28。
