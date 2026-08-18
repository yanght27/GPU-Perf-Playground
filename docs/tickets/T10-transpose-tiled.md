# T10 transpose-tiled

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T09
- 唯一学习变量：Shared Memory 分块与 Bank Conflict
- 路径覆盖：PT(参考), CU, TR, CT/CUTE(按官方示例)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- Tile 转置：块读入 shared memory，转置写回；
- CUDA 按官方 sample/Programming Guide 对齐；Triton 官方转置/tile 写法；CT/CUTE 按官方示例取舍；
- pad/swizzle 消除 bank conflict（如官方示例使用）；
- NCU 重点：shared bank conflict、DRAM throughput。

## 正确性门禁

- 与 PyTorch 参考及 T09 输出一致。

## 性能/工具门禁

- T09→T10 增量 benchmark；NCU bank conflict 指标；SASS 看 LDS/STS；证据入 docs/evidence/T10/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T10: <名称>`。

## 验收

- 能用指标说明 bank conflict 是否存在/消除；正确性通过。

## 解锁

- 验收后，学习者明确说“解锁 T11”才继续；本轮不实现 T11。
