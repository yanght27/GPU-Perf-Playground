# T09 transpose-naive

- 状态：`done`（2026-08-16 学习者验收通过；讲义含“什么是转置”与五路径核心代码逐行讲解）
- 前置：T08
- 唯一学习变量：二维线程布局与读合并/写合并的取舍
- 路径覆盖：PT, CU, TR, CT, CUTE
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- Out-of-place 朴素转置：按行读、跨行写；对比两种方向；
- 五路径同语义横向对比；
- 讲解二维 blockIdx/threadIdx 到 (row,col) 的映射；
- CL/BL 记录 N/A 理由。

## 正确性门禁

- 与 PyTorch 参考比较；覆盖方阵/非方阵/奇数维/1×N。

## 性能/工具门禁

- 两方向实测带宽/耗时；NSYS/NCU/SASS 原生命令；证据入 docs/evidence/T09/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T09: <名称>`。

## 验收

- 能解释为什么转置是 Memory-Bound 以及方向差异；正确性通过。

## 解锁

- 验收后，学习者明确说“解锁 T10”才继续；本轮不实现 T10。
