# T17 flash-attention-triton

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T16
- 唯一学习变量：IO-Aware Tiling + Online Softmax 的 Triton 表达
- 路径覆盖：PT(SDPA 黄金参考), TR, CU(T18 专用/N/A), CT(官方 flash 层检查/N/A), CUTE(官方 flash 层检查/N/A)
- 环境：gpp-core

## 范围

- 对齐 Triton 官方 tutorial 06（fused attention）实现 FA 风格 forward；
- 先复现官方示例参数，再换本机 shape；
- 讲解外层 K/V tile、内层 Q tile、running max/sum 与 O 修正；
- 与 T15 对比 DRAM 读写与耗时。

## 正确性门禁

- 与 F.scaled_dot_product_attention（causal/非 causal）比较，记录容差与参考 kernel 选择。

## 性能/工具门禁

- 多 seq_len（512/1024/2048，显存允许内）实测；NCU DRAM 读写量；NSYS/SASS；证据入 docs/evidence/T17/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T17: <名称>`。

## 验收

- 能推导 online 更新式；SDPA 误差在记录容差内；有实测访存/耗时下降。

## 解锁

- 验收后，学习者明确说“解锁 T18”才继续；本轮不实现 T18。
