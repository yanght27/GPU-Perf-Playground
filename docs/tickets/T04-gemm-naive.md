# T04 gemm-naive

- 状态：`done`（学习者验收通过）
- 前置：T03
- 唯一学习变量：朴素 GEMM 索引映射、访存/计算比与 cuBLAS 库基线
- 路径覆盖：PT, CU, TR, CT, CUTE, BL
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- C[m,n]=sum_k A[m,k]B[k,n] 朴素实现，只做正确索引与边界，不做 shared memory（那是 T05）；
- cuBLAS 按官方文档/sample 调用最小 API 作为库基线；
- 实测并量化访存/计算比：为什么朴素 GEMM 远低于 GPU 算力；
- CL 归入 T08 并记录原因。

## 正确性门禁

- 与 PyTorch fp64 参考比较；覆盖非 32 倍数与 1×K/K×1 边界。

## 性能/工具门禁

- 至少 512×512×512、1024×1024×1024 两组 shape；warmup+同步+CUDA events；cuBLAS 同 shape；NSYS/NCU/SASS；证据入 docs/evidence/T04/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T04: <名称>`。

## 验收

- 各路径正确性通过；有访存/计算比实测证据；cuBLAS 命令有官方出处。

## 解锁

- 验收后，学习者明确说“解锁 T05”才继续；本轮不实现 T05。
