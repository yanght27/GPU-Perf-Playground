# T21 sglang-serving

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T20
- 唯一学习变量：SGLang 官方 serving 流程与单框架指标
- 路径覆盖：SGLang 容器 v0.5.17
- 环境：容器（lmsysorg/sglang:v0.5.17）

## 范围

- 按 SGLang 官方 docs Quick Start 启动服务；
- 讲解 SGLang 是什么、和 vLLM / Transformers 的区别、RadixAttention 等关键设计；
- 核心命令行逐条解释（每条命令是什么、为什么这样写、输出怎么看、常用参数）；
- 同一模型快照与同一 prompt suite；
- 记录与 T20 相同的 metrics 定义；
- 记录 RadixAttention/调度相关官方文档要点（对比留到 T24）。

## 正确性门禁

- 固定 prompt 结果与 T19 语义一致（差异解释）。

## 性能/工具门禁

- 官方 metrics 与日志入 docs/evidence/T21/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T21: <名称>`。

## 验收

- 单框架 serving 证据完整、命令有官方出处；
- 讲义包含工具定位、与相邻框架区别、核心命令行逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T22”才继续；本轮不实现 T22。
