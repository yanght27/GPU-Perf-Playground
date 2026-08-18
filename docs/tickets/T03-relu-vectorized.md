# T03 relu-vectorized

- 状态：`done`（学习者验收通过）
- 前置：T02
- 唯一学习变量：合并访问与 128-bit 向量化 load/store
- 路径覆盖：PT(参考), CU, TR, CT/CUTE(按官方示例)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- CUDA float4 向量化版本（或官方文档等价写法）；Triton 按官方 mask/block 写法；cuTile/CuTe 有官方向量化示例则实现，否则 N/A+理由；
- 对比 T02 标量版实测带宽；
- NCU 重点：MemoryWorkloadAnalysis（L1/L2/DRAM throughput）、Occupancy；
- 回答为什么 ReLU 是 Memory-Bound。

## 正确性门禁

- 与 PyTorch fp64 参考逐元素比较；覆盖未对齐长度与极值。

## 性能/工具门禁

- 标量(T02 数据) vs 向量化实测；NSYS/NCU/SASS 原生命令与解释；证据入 docs/evidence/T03/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T03: <名称>`。

## 验收

- NCU 能证明访存吞吐变化；正确性通过；Bound 判断有证据。

## 解锁

- 验收后，学习者明确说“解锁 T04”才继续；本轮不实现 T04。
