# T19 qwen-baseline

- 状态：`done`（2026-08-17 实现并验证可运行；CPU 基线已记录，GPU 指标待真实机复跑）
- 前置：T18
- 唯一学习变量：固定快照上的生成正确性、确定性与基线指标
- 路径覆盖：Transformers + ModelScope 快照
- 环境：gpp-core

## 范围

- 从 assets/modelscope 固定 revision 加载 Qwen2.5-0.5B-Instruct 与 tokenizer；
- 验证 chat template 与官方模型卡一致；跑官方 generation 最小流程；
- 记录 dtype、显存、prefill/decode 分段耗时、tokens/s；
- 固定统一 prompt suite（后续 T20–T24 复用）。

## 正确性门禁

- 同输入两次生成完全一致；tokenizer round-trip；chat template 生效。

## 性能/工具门禁

- 官方推荐计时/显存观测；证据入 docs/evidence/T19/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T19: <名称>`。

## 验收

- 基线数字可复现；prompt suite 固定；快照完整性有记录。

## 解锁

- 验收后，学习者明确说“解锁 T20”才继续；本轮不实现 T20。
