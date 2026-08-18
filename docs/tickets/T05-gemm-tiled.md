# T05 gemm-tiled

- 状态：`done`（学习者验收通过）
- 前置：T04
- 唯一学习变量：Tile 分块与数据复用
- 路径覆盖：PT(参考), CU, TR, CT, CUTE
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- 每个 block 负责一个 C tile：A/B tile 进 shared memory，复用数据减少 global 读取；
- Triton 对齐官方 tutorial 03 的 tiled matmul；cuTile/CuTe 对齐官方 matmul 示例；
- 对比 T04 朴素版：同 shape 实测提升与 DRAM 读写量变化；
- 明确什么被复用了多少次（算术强度变化）。

## 正确性门禁

- 与 PyTorch fp64 参考比较；覆盖非 tile 倍数 shape。

## 性能/工具门禁

- T04→T05 增量 benchmark；NCU 看 DRAM/L1 变化；NSYS/SASS 原生命令；证据入 docs/evidence/T05/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T05: <名称>`。

## 验收

- Tiling 版本正确性通过；能用量化数据解释数据复用收益。

## 解锁

- 验收后，学习者明确说“解锁 T06”才继续；本轮不实现 T06。
