# T08 gemm-tensorcore

- 状态：`done`（学习者验收通过）
- 前置：T07
- 唯一学习变量：Tensor Core mma 路径与 CUTLASS 官方 example
- 路径覆盖：PT(参考), CU(mma), CL, BL(高级基线)
- 环境：gpp-core / gpp-cute

## 范围

- sm_8.9 的 Tensor Core 路径：以 CUDA Programming Guide「Writing Tile Kernels」或 CUTLASS 官方 example 为权威写法（fp16/bf16 依官方示例）；
- CUTLASS C++ 官方 example 编译运行并实测；cuBLAS 同 dtype 基线；
- 讲解 CUDA Core vs Tensor Core、mma fragment、wmma 与 mma.sync 的差异；
- Hopper/Blackwell 特性只做 C 级分析并写恢复路径。

## 正确性门禁

- fp16/bf16 路径与 PyTorch 参考比较，容差单独说明；fp32 与 T07 结果对照。

## 性能/工具门禁

- 各 GEMM 版本阶梯汇总 + Tensor Core 路径实测；NCU Tensor Core pipe 利用率；SASS 看 HMMA/LDSM；证据入 docs/evidence/T08/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T08: <名称>`。

## 验收

- Tensor Core 版本正确性通过；CUTLASS 官方示例可复现；阶梯数据完整。

## 解锁

- 验收后，学习者明确说“解锁 T09”才继续；本轮不实现 T09。
