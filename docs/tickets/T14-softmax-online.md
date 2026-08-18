# T14 softmax-online

- 状态：`done`（2026-08-16 学习者验收通过）
- 前置：T13
- 唯一学习变量：Online Softmax 与算子融合的访存收益
- 路径覆盖：PT(参考), CU, TR, CT/CUTE(按官方示例)
- 环境：gpp-core / gpp-cute / gpp-cutile

## 范围

- Online Softmax 单遍：running max/sum 在线更新（公式先推再实现）；
- Triton 对齐官方 tutorial 02（fused softmax）；CUDA 手工实现；CT/CUTE 按官方示例；
- 对比 T13：少写多少中间张量、DRAM 读写量实测变化；
- NCU 看 MUFU/指数路径与 L1/L2。

## 正确性门禁

- 与 PyTorch fp64 参考及 T13 输出一致（容差记录）。

## 性能/工具门禁

- 3-pass vs online 实测；NCU 读写量对比；NSYS/SASS 原生命令；证据入 docs/evidence/T14/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T14: <名称>`。

## 验收

- 能推导 online 更新式；有“少读写多少”实测证据。

## 解锁

- 验收后，学习者明确说“解锁 T15”才继续；本轮不实现 T15。
