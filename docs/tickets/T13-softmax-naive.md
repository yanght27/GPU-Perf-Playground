# T13 softmax-naive

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T12
- 唯一学习变量：Softmax 数值语义与行归约映射
- 路径覆盖：PT, CU, TR, CT, CUTE
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- 按行 3-pass：max → sum(exp) → normalize；
- 五路径同语义横向对比；
- 讲解数值稳定（减 max）与行归约如何映射到线程；
- CL/BL 记录 N/A 理由。

## 正确性门禁

- 与 PyTorch fp64 参考比较；覆盖极值(±1000)、全相同行、N=1、未对齐。

## 性能/工具门禁

- 3-pass 实测带宽/耗时基线；NSYS/NCU/SASS 原生命令；证据入 docs/evidence/T13/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T13: <名称>`。

## 验收

- 正确性通过；能解释数值稳定；五路径基线完整。

## 解锁

- 验收后，学习者明确说“解锁 T14”才继续；本轮不实现 T14。
