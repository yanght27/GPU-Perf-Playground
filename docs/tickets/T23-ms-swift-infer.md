# T23 ms-swift-infer

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T22
- 唯一学习变量：ms-swift 官方 infer/deploy 流程
- 路径覆盖：ms-swift（gpp-swift-4.4.3，按 T00 修正）
- 环境：gpp-swift-4.4.3

## 范围

- 按 ms-swift 官方 docs 当前版本的 infer/deploy 最小流程；
- 讲解 ms-swift 是什么、和 Transformers / vLLM / SGLang / TRT-LLM 的关系与区别；
- 核心命令行逐条解释（infer/deploy 命令是什么、为什么这样写、输出怎么看）；
- 同一模型快照与统一 prompt suite；
- 记录 metrics 与显存；
- 明确 ms-swift 作为上层入口与底层 vLLM 的关系（官方文档为准）。

## 正确性门禁

- 固定 prompt 结果与 T19 语义一致（差异解释）。

## 性能/工具门禁

- 官方 metrics 与日志入 docs/evidence/T23/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T23: <名称>`。

## 验收

- 单框架证据完整、命令有官方出处；
- 讲义包含工具定位、与相邻框架区别、核心命令行逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T24”才继续；本轮不实现 T24。
