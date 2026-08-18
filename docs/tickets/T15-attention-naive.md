# T15 attention-naive

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T14
- 唯一学习变量：Attention 计算图到 kernel 的映射
- 路径覆盖：PT(双参考), CU, TR, CT/CUTE(按官方示例)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- S=QKᵀ/√d → causal mask → softmax → O=PV，朴素正确实现，不做 IO-Aware 分块（那是 T17/T18）；
- CUDA/TR 按官方教程或明确算法实现；CT/CUTE 有官方 attention 示例则做，否则 N/A+理由；
- PyTorch 双参考：显式 eager 公式 + F.scaled_dot_product_attention；
- 讲解 batch/head 并行与 KV Cache 需求动机（不实现，那是 T16）。

## 正确性门禁

- 覆盖有/无 causal mask、不同 seq_len/head_dim/scale、固定 seed；记录容差。

## 性能/工具门禁

- 固定 shape 实测；NSYS/NCU/SASS 原生命令；证据入 docs/evidence/T15/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T15: <名称>`。

## 验收

- 双参考正确性通过；能解释计算图各步与并行映射。

## 解锁

- 验收后，学习者明确说“解锁 T16”才继续；本轮不实现 T16。
