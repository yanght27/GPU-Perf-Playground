# T06 gemm-smem

- 状态：`done`（学习者验收通过）
- 前置：T05
- 唯一学习变量：共享内存 Bank Conflict 消除与 128-bit 共享内存访问
- 路径覆盖：CU, TR, CUTE(按官方示例)
- 环境：gpp-core / gpp-cute

## 范围

- 在 T05 基础上：pad/swizzle 或官方示例的 bank conflict 规避；shared memory 128-bit 加载；
- Triton 按官方 load/store 模式或 block pointer 写法（以官方 tutorial 为准）；CUTE 有官方示例则做；
- NCU 重点：shared load/store bank conflict 指标；
- 只改共享内存访问布局，不引入 cp.async/Tensor Core（那是 T07/T08）。

## 正确性门禁

- 与 T05 结果逐元素一致（同一数学输出）。

## 性能/工具门禁

- T05→T06 增量 benchmark；NCU bank conflict 指标前后对比；SASS 看 LDS/STS 宽度；证据入 docs/evidence/T06/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T06: <名称>`。

## 验收

- 有 bank conflict 量化证据与消除手段；正确性不变。

## 解锁

- 验收后，学习者明确说“解锁 T07”才继续；本轮不实现 T07。
