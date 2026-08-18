# T02 relu-scalar

- 状态：`done`（学习者验收通过）
- 前置：T01
- 唯一学习变量：元素级 kernel 的索引、Grid 配置与边界处理
- 路径覆盖：PT, CU, TR, CT, CUTE
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- 标量 ReLU 基线，不引入向量化（那是 T03）；
- 五路径同语义横向对比：索引计算、边界分支、grid/block 选择；
- 与 T01 对比：元素级 kernel 与向量加法的差异在哪；
- CL/BL 记录 N/A 理由。

## 正确性门禁

- 与 PyTorch fp64 参考逐元素比较；覆盖负值/零/正边界与 N 未对齐。

## 性能/工具门禁

- 各路径标量版实测带宽/耗时基线；NSYS/NCU/SASS 原生命令；证据入 docs/evidence/T02/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T02: <名称>`。

## 验收

- 五路径正确性通过；能解释边界处理与 grid 配置；工具证据齐全。

## 解锁

- 验收后，学习者明确说“解锁 T03”才继续；本轮不实现 T03。
