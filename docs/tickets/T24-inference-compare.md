# T24 inference-compare

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T23
- 唯一学习变量：TTFT/TPOT/吞吐/显存统一对比；PagedAttention、Continuous Batching、投机解码
- 路径覆盖：四框架证据汇总
- 环境：T20–T23 的容器与环境 + 统一 prompt suite

## 范围

- 统一表格对比四框架 metrics，逐项解释差异来源；
- 先回顾四框架定位与命令行差异，再解释指标差异来源；
- 概念讲解：PagedAttention（vLLM 官方文档）、Continuous Batching（官方 serving 文档）、投机解码（SGLang/vLLM 官方支持与限制）；
- 单卡/0.5B 上能实测的概念给 A/B，只能分析的给 C 并写恢复路径；
- 同 prompt 结果一致性抽查。

## 正确性门禁

- 对比表与原始证据互相可追溯；结论不得超出实测证据。

## 性能/工具门禁

- 证据汇总入 docs/evidence/T24/（引用 T20–T23 原始日志，不重复制造）。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T24: <名称>`。

## 验收

- 四框架各至少一条完整证据；三个概念有官方来源与 A/B/C 分级；
- 对比结论必须能追溯到各框架的定位、命令行与原始日志差异。

## 解锁

- 验收后，学习者明确说“解锁 T25”才继续；本轮不实现 T25。
