# T22 tensorrt-llm-serving

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T21
- 唯一学习变量：TRT-LLM 官方 build/run 流程与 8GB 约束
- 路径覆盖：TRT-LLM 容器 1.2.1
- 环境：容器（nvcr.io/nvidia/tensorrt-llm/release:1.2.1）

## 范围

- 按 TRT-LLM 官方 docs/example 完成模型转换、build engine、run；
- 讲解 TRT-LLM 是什么、和 vLLM / SGLang / Transformers 的区别、engine 编译为什么是核心步骤；
- 核心命令行逐条解释（模型转换、build、run 各是什么、为什么这样写、输出怎么看）；
- 8GB 显存下按官方建议缩小 max_batch_size/max_input_len 并记录；
- 跑统一 prompt suite，记录与 T20/T21 相同 metrics；
- 记录 engine 构建时间与精度路径。

## 正确性门禁

- 固定 prompt 结果与 T19 语义一致（差异解释）。

## 性能/工具门禁

- 官方 metrics 与日志入 docs/evidence/T22/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T22: <名称>`。

## 验收

- engine 可复现构建与运行；约束有官方出处；
- 讲义包含工具定位、与相邻框架区别、核心命令行逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T23”才继续；本轮不实现 T23。
