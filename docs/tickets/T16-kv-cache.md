# T16 kv-cache

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T15
- 唯一学习变量：增量式 KV 追加为什么只省 decode
- 路径覆盖：PT(语义参考), CU, TR, CT(官方能力检查/N/A), CUTE(官方能力检查/N/A)
- 环境：gpp-core

## 范围

- 实现 KV append 与带 cache 的 decode 步计算；
- 与无 cache 重复计算对比：kernel 数量/耗时/显存变化；
- 讲解 prefill vs decode 的计算量差异；PagedAttention 留到 T24；
- 以 Transformers 官方 KV Cache 语义为准对齐。

## 正确性门禁

- 带/不带 cache 输出一致（容差记录）；覆盖多步 decode。

## 性能/工具门禁

- 无 cache vs 有 cache 的 decode 耗时；NSYS 看重复 kernel 是否消失；证据入 docs/evidence/T16/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T16: <名称>`。

## 验收

- 能解释 KV Cache 为什么只对 decode 有效；正确性通过。

## 解锁

- 验收后，学习者明确说“解锁 T17”才继续；本轮不实现 T17。
