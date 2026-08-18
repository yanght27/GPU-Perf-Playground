# T20 vllm-serving

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T19
- 唯一学习变量：vLLM 官方 serving 流程与单框架指标
- 路径覆盖：vLLM 容器 v0.27.1
- 环境：容器（vllm/vllm-openai:v0.27.1）

## 范围

- 按 vLLM 官方 docs 当前版本 serving/Quick Start 启动 OpenAI 兼容服务；
- 讲解 vLLM 是什么、解决什么问题、和 Transformers / 其他 serving 框架的区别；
- 核心命令行逐条解释（每条命令是什么、为什么这样写、输出怎么看、常用参数）；
- 模型只读挂载 assets/modelscope 固定快照；
- 跑统一 prompt suite，记录 TTFT/ITL/TPOT/吞吐/显存；
- 记录 8GB 显存下官方建议的参数约束。

## 正确性门禁

- 至少一个固定 prompt 结果与 T19 语义一致（chat template/tokenizer 相同，差异解释）。

## 性能/工具门禁

- 官方 metrics 与日志入 docs/evidence/T20/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T20: <名称>`。

## 验收

- 单框架 serving 证据完整、命令有官方出处；
- 讲义包含工具定位、与相邻框架区别、核心命令行逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T21”才继续；本轮不实现 T21。
