# T07 gemm-pipeline

- 状态：`done`（学习者验收通过）
- 前置：T06
- 唯一学习变量：Double Buffer / cp.async / Pipelining 隐藏访存延迟
- 路径覆盖：CU, TR, CT/CUTE(按官方示例)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- CUDA：cp.async（或官方示例等价写法）+ double buffer 重叠 load 与计算；
- Triton：num_stages/pipelining 官方写法；
- cuTile/CuTe 按官方异步示例取舍，缺失记 N/A；
- NCU 重点：stall 原因（Long Scoreboard）、latency hiding；SASS 看 LDGSTS/CP.ASYNC。

## 正确性门禁

- 与 PyTorch fp64 参考及 T06 输出一致。

## 性能/工具门禁

- T06→T07 增量 benchmark；NCU stall 指标对比；NSYS 时间线看重叠；证据入 docs/evidence/T07/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T07: <名称>`。

## 验收

- 能解释流水线为什么减少 stall；实测有增量或明确解释为何无收益。

## 解锁

- 验收后，学习者明确说“解锁 T08”才继续；本轮不实现 T08。
