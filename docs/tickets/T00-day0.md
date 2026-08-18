# T00 day0

- 状态：`done`（学习者 2026-08-15 验收通过；机械门禁与过关问题均通过）
- 前置：无
- 唯一学习变量：环境、固定快照与证据链是否真实可复现
- 路径覆盖：全部环境 + 4 个 Docker 镜像 + 系统工具链
- 环境：全部

## 范围

- 在线核对 S01–S20，转正 source-ledger；
- 解决 gpp-swift-4.4.3 与 ms-swift 4.5.0.dev0 的版本/命名不一致（以官方仓库结论为准）；
- 生成 config/day0-lock.json（hardware、docker、containers、environments、assets 全部锁定）；
- 下载并 SHA256 校验 ModelScope Qwen2.5-0.5B-Instruct 固定 revision；
- 按 ms-swift 官方 Quick Start 选定并锁定一个官方示例数据集快照；
- 6 个环境/容器脚本 --verify-only 全部 PASS；NSYS/NCU/SASS 各留一条 smoke 证据。

## 正确性门禁

- 模型确定性生成 smoke；数据集与模型 SHA256 全量校验。

## 性能/工具门禁

- NSYS/NCU/SASS smoke 原始输出入 docs/evidence/T00/。

## 文档

- 更新 `docs/lectures/Txx-*.md`（本轮唯一主讲义）、过关问题与答案、`config/coverage-matrix.md`、`config/source-ledger.md`。
- Git 提交：`T00: <名称>`。

## 验收

- config/day0-lock.json 与实测一致且每项有官方出处；全部门禁 PASS；过关问题通过。

## 解锁

- 验收后，学习者明确说“解锁 T01”才继续；本轮不实现 T01。
