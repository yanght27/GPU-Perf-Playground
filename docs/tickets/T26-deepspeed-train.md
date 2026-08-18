# T26 deepspeed-train

- 状态：`done`（2026-08-17 学习者验收通过）
- 前置：T25
- 唯一学习变量：DeepSpeed 官方最小接入与 ZeRO 单卡实测
- 路径覆盖：DeepSpeed
- 环境：gpp-deepspeed-0.19.5

## 范围

- 按 DeepSpeed 官方 docs 最小 training 流程接入 PyTorch 循环；
- 讲解 DeepSpeed 是什么、解决什么问题、和纯 PyTorch 训练的区别；
- 核心命令/`ds_config` 参数逐条解释（ZeRO、offload、batch、checkpoint 等）；
- 单卡运行并解释 ZeRO-1/2 在单卡时的真实收益与局限；
- 记录 ds_config、step 时间、显存、吞吐；
- 与 T25 同配置对比。

## 正确性门禁

- loss 下降与 checkpoint round-trip；与 T25 数值行为可比。

## 性能/工具门禁

- 官方日志与 metrics 入 docs/evidence/T26/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- 讲义必须包含 `4.0 零基础先修：概念地图` 和 `5.x 工具/框架定位与命令行` 两个小节；
- Git 提交：`T26: <名称>`。

## 验收

- 接入流程有官方出处；单卡 ZeRO 结论有实测支撑；
- 讲义包含 DeepSpeed 定位、与 PyTorch 训练的区别、核心命令/ds_config 逐条讲解。

## 解锁

- 验收后，学习者明确说“解锁 T27”才继续；本轮不实现 T27。
