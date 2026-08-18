# T01 vector-add

- 状态：`done`（学习者验收通过：实现、知识点、过关题均确认）
- 前置：T00
- 唯一学习变量：grid/block/thread 执行模型与五路径最小闭环
- 路径覆盖：PT, CU, TR, CT, CUTE
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- 固定输入合同 N/dtype/seed/device；
- PyTorch fp64 黄金参考；CUDA 最小 kernel 与 nvcc 编译运行流程；Triton 对齐官方 tutorial 01；cuTile/CuTe 对齐官方 Quick Start 元素级示例；
- 讲解 thread index、grid/block 配置、kernel launch、warp/SM 初步概念、Global Memory 初步概念；
- CL/BL 记录 N/A 理由。

## 正确性门禁

- 与 PyTorch fp64 逐元素比较；覆盖 N 未整除 block 的边界；五路径同输入互相对照。

## 性能/工具门禁

- NSYS 首条时间线、NCU 基础指标、SASS 的 LDG/STG 首读；证据入 docs/evidence/T01/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T01: <名称>`。

## 验收

- 五路径正确性通过；三条工具证据齐全；学习变量讲解并通过过关问题。

## 解锁

- 验收后，学习者明确说“解锁 T02”才继续；本轮不实现 T02。
