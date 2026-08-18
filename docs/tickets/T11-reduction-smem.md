# T11 reduction-smem

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T10
- 唯一学习变量：block 内线程同步与 Shared Memory 协作
- 路径覆盖：PT, CU, TR, CT, CUTE(官方 block_smem_reduce 的纯 smem 教学版)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- 分块局部和 → shared memory 跨 warp 规约 → 写回；
- 五路径同语义横向对比；cuTile/CuTe 缺官方 reduction 示例则 N/A+理由；
- 讲解 __syncthreads() 作用域与代价、两段式归约；
- CL/BL 记录 N/A 理由。

## 正确性门禁

- 与 PyTorch fp64 sum 比较；覆盖 N 未整除、多 block 部分和、随机 seed。

## 性能/工具门禁

- shared memory 版本实测；NSYS/NCU/SASS（BAR.SYNC）原生命令；证据入 docs/evidence/T11/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T11: <名称>`。

## 验收

- 正确性通过；能解释同步边界；工具证据齐全。

## 解锁

- 验收后，学习者明确说“解锁 T12”才继续；本轮不实现 T12。
