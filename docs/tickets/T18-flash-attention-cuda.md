# T18 flash-attention-cuda

- 状态：`done`（2026-08-17 收尾；gpp-cutile 锁版本已按实际可运行环境修订，功能路径 PASS）
- 前置：T17
- 唯一学习变量：把 T17 算法手工映射到 CUDA 的 tiling/同步/规约
- 路径覆盖：PT(参考), CU, TR(T17 已完成/N-A), CT/CUTE(官方能力实测)
- 环境：gpp-core

## 范围

- CUDA C++ 实现 FA 风格 forward：K/V 外层 tile、Q 内层 tile、online softmax、__syncthreads 边界；
- 与 T17 Triton 版横向对比：同一算法在手工 CUDA 中的显式化差异；
- 复用 T05–T08 的 tiling/同步/规约知识；
- 正确性以 PyTorch SDPA 为黄金参考。

## 正确性门禁

- 与 SDPA 比较；覆盖 causal、不同 seq_len/head_dim、tile 边界。

## 性能/工具门禁

- 与 T15/T17 三版对比；NCU/NSYS/SASS 原生命令；证据入 docs/evidence/T18/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T18: <名称>`。

## 验收

- CUDA 版误差在容差内；能解释 tile 循环与同步点；证据完整。

## 解锁

- 验收后，学习者明确说“解锁 T19”才继续；本轮不实现 T19。
