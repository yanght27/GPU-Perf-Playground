# T12 reduction-shuffle

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T11
- 唯一学习变量：warp shuffle 规约、occupancy 与 latency hiding
- 路径覆盖：PT(参考), CU, TR(观察生成代码), CT(官方 ct.sum 能力对照，shuffle 机制 N/A), CUTE(官方 warp shuffle 树)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- __shfl_down_sync 等 warp 内规约 vs T11 shared memory 版本；
- Triton tl.sum 并观察生成 TTGIR/PTX 中的 shuffle（官方 CompiledKernel.asm 接口）；
- cuTile ct.sum 官方能力对照（shuffle 级 API 无官方接口，记 N/A）；CuTe 官方 warp_vector_reduce 的 bfly 树；
- 讲解 sync mask、divergence、活跃 warp 与 latency hiding；
- NCU 重点：SchedulerStats/WarpStateStats、occupancy；SASS 看 SHFL/BAR.SYNC（本实现没有 ATOM，ATOM 后置）。

## 正确性门禁

- 与 T11 输出一致（允许加法顺序不同带来的容差，记录并解释）。

## 性能/工具门禁

- T11→T12 实测对比；NCU stall 与 occupancy；SASS 指令；证据入 docs/evidence/T12/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T12: <名称>`。

## 验收

- 能解释 warp 规约 vs block 规约的差别与取舍；正确性通过。

## 解锁

- 验收后，学习者明确说“解锁 T13”才继续；本轮不实现 T13。
