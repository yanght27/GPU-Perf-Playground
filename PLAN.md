# GPP 固定执行计划（v1.7，2026-08-15）

> 本文档是 GPU-Perf-Playground（GPP）的**唯一固定课程执行契约**。v1.1 拆细为 29 个最小增量；
> v1.2 新增最高优先级约束；v1.3 固定学习主体优先；v1.4 固定验收前覆盖完整性自查；
> v1.5 固定核心概念词典持久化；v1.6 固定零疑问讲解标准；v1.7 固定“每个算子 Ticket
> 五路径齐”。
>
> **修订规则**：只有学习者明确要求时才可修订；AI 不得自行增删、合并、跳过或提前解锁。
> 修订时更新版本号和日期并说明差异。

## 0. 最高优先级约束（学习者重申，不可妥协）

0.1 **在线对齐，禁止模型记忆**。任何算子、工具、框架的 API、命令行、环境配置、
Quick Start、工作流、实现模式，动手前必须联网核对官方文档 / 官方仓库 / 官方 sample /
官方推荐的规范参考项目，并把 URL、访问日期、版本/tag/commit、引用的官方文件写入
`config/source-ledger.md`。回答中的每一条关键命令和 API 都要能追溯到台账编号；
查不到官方来源就停止报告 blocker，不允许“凭经验先写、事后补链接”。

0.2 **增量学习闭环，不是“实现完对比一下”**。每个 Ticket 必须完整走完：
实现 → 逐行讲解 → 知识点讲解 → 覆盖检查 → 过关问题及答案（一问一答）→ 学习者验收 → 才解锁下一增量。
讲解由浅入深；`idea.md` 提到的全部内容都要被某个 Ticket 显式覆盖并在
`config/coverage-matrix.md` 标记。验收不通过，继续补当前 Ticket，不得前进。

0.3 **讲义双写**。对话里完整讲解，同时把同一内容完整落盘为
`docs/lectures/Txx-*.md`。讲义不是摘要，必须包含：唯一学习变量、官方来源、核心代码与
逐行解释、知识点讲解、实测数据、Bound 判断、过关问题及答案（一问一答）、停止点。对话结束但
讲义缺失 = 本 Ticket 未完成。

0.4 **模型资产只引用、不复制**。Qwen 模型/数据等大文件统一使用
`assets/modelscope/...` 原路径，不复制第二份；确实需要的小文本文件（如 tokenizer 配置
片段、chat template 说明）可复制或摘录，但必须注明来源路径。

0.5 **可运行、可分析、学习主体优先**。每个算子必须真实跑通正确性与性能测试，并且必须
给出 NCU/NSYS/SASS 等工具证据；但代码组织永远把“算子主体和优化主线”放在最显眼的位置：
计时、参数解析、结果打印等通用逻辑拆到 common 模块；kernel 内只留与学习变量相关的代码；
关键行必须有“为什么”注释，而不是“做了什么”注释；一屏内先看到算法主线，再看到工程细节。

0.6 **验收前覆盖完整性自查**。每个 Ticket 提交验收前，AI 必须对照任务书、
`config/coverage-matrix.md` 和下一个 Ticket 的前置知识做一次显式自查：仅凭本 Ticket 的
对话讲解 + 讲义 + 代码 + 过关题，零基础学习者是否足以掌握本增量、并具备开启下一增量的
全部前置概念。发现缺口就在本 Ticket 内继续补讲、补题、补证据，不得带着缺口提交验收；
自查结论写入讲义“知识点完整性检查”一节。

0.7 **核心概念词典持久化**。`docs/CONCEPTS.md` 是跨 Ticket 的持久概念记忆：工具定位、
Bound 三分类与判定流程、高频术语速查。每个 Ticket 结束后，凡本轮涉及且词典未收录的
核心概念，必须补充“一句话定义 + 分析方法 + 本项目证据位置 + 后续深挖 Ticket”；
禁止只在对话里讲、不留词典条目。

0.8 **零疑问讲解标准**。每个增量交付时，学习者应能仅凭“代码逐行解释 + 知识要点 +
过关题与答案”对本增量内容不再有未解释的疑问。AI 在写讲义时必须做“新手追问自查”：
凡代码或讲解中出现的每个术语（如 barrier、bank、TF32、No Eligible），要么当轮讲清
“是什么/为什么/怎么验证”，要么明确标注归属后续 Ticket；不允许出现学习者看不懂又
找不到解释的孤立术语。

## 1. 当前基线快照（已实测，2026-08-15）

| 项目 | 实测值 |
| --- | --- |
| Host / Guest | Linux / WSL2 |
| GPU | NVIDIA GPU（sm_8.9，8GB） |
| Driver | 以本机为准 |
| 系统 CUDA | 以本机为准 |
| Docker | 以本机为准 |
| 已拉取镜像 | `vllm/vllm-openai:v0.27.1`、`lmsysorg/sglang:v0.5.17`、`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`、`nvidia/cuda:13.0.2-base-ubuntu22.04` |
| Conda 环境 | `gpp-core`、`gpp-cute`、`gpp-cutile`、`gpp-deepspeed-0.19.5`、`gpp-swift-4.4.3`（Python 3.12） |
| 工具链 | 系统 CUDA 工具链（`nvcc`、`nsys`、`ncu`、`nvdisasm`、`cuobjdump`） |

关键包实测版本（供 T00 锁定前参考，T00 以官方文档在线核对后写入锁文件）：

| 环境 | 关键包 |
| --- | --- |
| gpp-core | torch 2.13.0+cu130、triton 3.7.1、transformers 5.14.1、modelscope-hub 0.1.8、numpy 2.5.2 |
| gpp-cute | torch 2.13.0+cu130、cuda-python 13.3.1、nvidia-cutlass-dsl 4.7.0、protobuf 6.33.6 |
| gpp-cutile | cuda-tile 1.5.0、cupy-cuda13x 14.1.1、cuda-toolkit 13.0.3.0、nvidia-cuda-runtime 13.0.96 |
| gpp-deepspeed-0.19.5 | torch 2.13.0+cu130、triton 3.7.1、deepspeed 0.19.5 |
| gpp-swift-4.4.3 | torch 2.11.0+cu130、triton 3.6.0、transformers 5.12.1、deepspeed 0.18.9、vllm 0.23.0、ms-swift 4.5.0.dev0 |

**T00 已解决的 Day 0 项**（证据在 `docs/evidence/T00/`）：

1. `config/day0-lock.json` 已生成；6 个 `--verify-only` 门禁全部 PASS。
2. `assets/modelscope` 模型/数据集快照已下载，revision 记录与 SHA256 清单通过校验。
3. `gpp-swift-4.4.3` 命名已核实：本机 commit `e1287928…` 与 ms-swift 官方 tag `v4.4.3` 一致，无需改名或回退。

## 2. v1.1 拆分原则（粒度修订）

1. **一个 Ticket = 一个优化层级或一个机制**。例如 ReLU 拆“标量版”与“向量化版”，GEMM 拆
   朴素/Tiling/共享内存优化/流水线/Tensor Core，Transpose 拆朴素/Tile。
2. **横向对比保留在同一 Ticket 内**：同一优化层级下，PyTorch、CUDA C++、Triton、cuTile
   Python、CuTe DSL 可以并排实现与对比，这本身就是学习变量之一（同一语义、不同工具）。
3. **每个算子 Ticket 五路径齐**：PyTorch（参考/基线）、CUDA C++、Triton、cuTile Python、
   CuTe DSL 都必须出现在该 Ticket（实现或官方能力对照）。若该优化层级官方没有对应示例，
   也必须给出“最接近的官方能力实现/同步对照 + N/A 原因”，而不是省略该路径。
4. **一个 Ticket 一个学习变量**：任务书第一行写明；讲解、实验、文档都围绕它。
5. **先正确、后性能**：黄金参考（PyTorch fp64）通过后才 benchmark；正确性失败 = 本轮停止。
6. **严格在线对齐**：动手前在线核对官方文档/仓库/sample，写入 `config/source-ledger.md`；
   禁止凭模型记忆写 API、版本、命令。
7. **证据分级**：A=本机实现并实测；B=本机运行对比；C=官方文档分析（写受限原因与恢复路径）。
   预测结果不写入报告。
8. **固定快照**：训练/推理全部复用 T00 下载校验的 ModelScope Qwen2.5-0.5B-Instruct 固定
   revision，中途不换模型/revision。
9. **节奏不再被“7 天”绑架**：以掌握为准，简单 Ticket 可一天多个，复杂 Ticket 允许多轮；
   每个 Ticket 必须通过过关问题才解锁下一个。

## 3. Ticket 地图（29 个固定增量）

状态标记：`ready` = 当前可解锁；`locked` = 等待前置；`done` = 已验收。
当前状态：**T00–T28 done；阶段二（T04–T10）验收通过；课程主线闭环**。

路径缩写：PT=PyTorch（参考/实现）、CU=CUDA C++、TR=Triton、CT=cuTile Python、
CUTE=CuTe DSL、CL=CUTLASS C++、BL=cuBLAS。

### 阶段 0：准备

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T00 | Day 0：环境与证据门禁 | 环境、快照与证据链是否真实可复现 | 全部 env + 容器 |

### 阶段 1：执行模型与元素级（T01–T03）

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T01 | Vector Add（五路径基线） | grid/block/thread 执行模型与最小闭环 | PT, CU, TR, CT, CUTE |
| T02 | ReLU 标量版（五路径基线） | 元素级 kernel 的索引、Grid 配置与边界处理 | PT, CU, TR, CT, CUTE |
| T03 | ReLU 向量化版 | 合并访问与 128-bit 向量化 load/store | PT(参考), CU, TR, CT/CUTE(按官方示例) |

### 阶段 2：GEMM（T04–T08，每个优化版本独立）

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T04 | GEMM 朴素版 + cuBLAS 基线 | 朴素索引映射、访存/计算比、库基线 | PT, CU, TR, CT, CUTE, BL |
| T05 | GEMM Shared-Memory Tiling | Tile 分块与数据复用 | PT(参考), CU, TR, CT, CUTE |
| T06 | GEMM 共享内存优化 | Bank conflict 消除与 128-bit 共享内存访问 | CU, TR, CUTE(按官方示例) |
| T07 | GEMM 异步拷贝与流水线 | Double buffer / cp.async / pipelining 隐藏访存延迟 | CU, TR(官方 tutorial 写法), CT/CUTE(按官方示例) |
| T08 | GEMM Tensor Core 与 CUTLASS | Tensor Core mma 路径与 CUTLASS 官方 example | CU(mma), CL, BL(高级基线), PT(参考) |

### 阶段 3：访存与线程协作（T09–T14）

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T09 | Transpose 朴素版（五路径基线） | 二维线程布局、读合并/写合并的取舍 | PT, CU, TR, CT, CUTE |
| T10 | Transpose Tile 版 | Shared memory 分块与 Bank Conflict | PT(参考), CU, TR, CT/CUTE(按官方示例) |
| T11 | Reduction 共享内存规约 | block 内线程同步与 shared memory 协作 | PT, CU, TR, CT, CUTE(缺官方示例记 N/A) |
| T12 | Reduction Warp Shuffle | warp shuffle 规约、occupancy 与 latency hiding | PT(参考), CU, TR(观察生成代码), CT(官方 ct.sum 能力对照), CUTE(官方 warp shuffle 树) |
| T13 | Softmax 朴素 3-pass（五路径基线） | Softmax 数值语义与行归约映射 | PT, CU, TR, CT, CUTE |
| T14 | Softmax Online/融合版 | Online Softmax 与算子融合的访存收益 | PT(参考), CU, TR, CT/CUTE(按官方示例) |

### 阶段 4：Attention（T15–T18）

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T15 | Attention 朴素前向 | QKᵀ/scale/mask/Softmax/PV 计算图到 kernel 的映射 | PT(双参考), CU, TR, CT/CUTE(按官方示例) |
| T16 | KV Cache | 增量式 KV 追加为什么只省 decode | PT(语义参考), CU, TR, CT(官方能力检查/N-A), CUTE(官方能力检查/N-A) |
| T17 | FlashAttention（Triton 官方 tutorial 版） | IO-Aware Tiling + Online Softmax 的 Triton 表达 | PT(SDPA 黄金参考), TR, CU(T18 专用/N-A), CT/CUTE(官方 flash 层检查/N-A) |
| T18 | FlashAttention（CUDA C++ 版） | 把 T17 算法手工映射到 CUDA 的 tiling/同步/规约 | PT(参考), CU, TR(T17 已完成/N-A), CT/CUTE(官方能力实测) |

### 阶段 5：模型与框架集成（T19–T28，每框架独立 + 统一对比）

| ID | 名称 | 唯一学习变量 | 路径/框架 |
| --- | --- | --- | --- |
| T19 | Qwen/Transformers 基线 | 固定快照上的生成正确性、确定性与基线指标 | Transformers（gpp-core） |
| T20 | vLLM 服务 | vLLM 官方 serving 流程与单框架指标 | vLLM 容器 v0.27.1 |
| T21 | SGLang 服务 | SGLang 官方 serving 流程与单框架指标 | SGLang 容器 v0.5.17 |
| T22 | TensorRT-LLM 服务 | TRT-LLM 官方 build/run 流程与 8GB 约束 | TRT-LLM 容器 1.2.1 |
| T23 | ms-swift 推理 | ms-swift 官方 infer/deploy 流程 | gpp-swift-4.4.3（按 T00 修正） |
| T24 | 推理统一对比 | TTFT/TPOT/吞吐/显存对比；PagedAttention、Continuous Batching、投机解码 | 统一 prompt suite + 四框架证据 |
| T25 | PyTorch 训练基线 | 最小训练循环、checkpoint 与单卡指标 | PyTorch（gpp-core） |
| T26 | DeepSpeed 训练 | DeepSpeed 官方最小接入与 ZeRO 单卡实测 | gpp-deepspeed-0.19.5 |
| T27 | ms-swift SFT | ms-swift 官方 Qwen SFT/LoRA Quick Start | gpp-swift-4.4.3（按 T00 修正） |
| T28 | 训练与并行统一对比 | DP/DDP/ZeRO/FSDP/TP/PP/SP/CP 原理、通信与取舍 | 三框架实测 + C 级多卡分析 |

## 4. 每个 Ticket 的固定验收合同

一个 Ticket 只有**全部**满足下列条件才算 `done`：

1. **唯一变量**：任务书中的学习变量被实现、讲解并被过关问题覆盖。
2. **在线对齐**：官方来源已在 `config/source-ledger.md` 记录 URL、访问日期、版本/tag/commit、
   引用的官方文件路径。
3. **环境证据**：本 Ticket 使用的环境实际运行过 `--verify-only` 门禁或等价命令，命令与输出
   进入 `docs/evidence/Txx/`。
4. **正确性**：通过黄金参考测试；数值误差、shape、边界、causal/mask（如适用）都记录。
5. **性能证据**：实测 benchmark（warmup、同步、CUDA events 或官方计时方式），只报测量值。
6. **工具取证**：T01–T18 每个算子 Ticket 必须给出并解释原生 NSYS、NCU、SASS 命令
   （`nsys profile`、`ncu`、`cuobjdump -sass`/`nvdisasm`）；便捷脚本可以给，原生命令必须展开。
   T19–T28 使用官方推荐的 metrics/日志/profiler。
7. **文档产出**：更新 `docs/lectures/Txx-*.md`（完整讲义，含代码逐行解释、知识点讲解、
   实测数据、过关问题及答案（一问一答），不是摘要）、`config/coverage-matrix.md`、
   `config/source-ledger.md`。对话中讲过的内容必须同步落盘。
8. **Bound 判断**：给出 Memory/Compute/Latency-Bound 结论 + 工具证据；证据不足时明写
   “结论为假设”，不得写成事实。
9. **Git**：每个 Ticket 一次提交 `Txx: <名称>`；提交前 `git status` 检查；大文件不入库。
10. **停止点**：本轮回答结束在 Ticket 边界，末尾只说明下一最小增量，不实现它，等待学习者解锁。
11. **覆盖完整**：本 Ticket 对应的 `idea.md` 知识点已全部讲解并在 coverage-matrix 标记；
    任何暂时无法覆盖的项必须显式写出“归哪个 Ticket 覆盖”或“为何受限”，不得静默遗漏。
12. **预验收三轴**：学习者要求预验收时，必须以初学者视角重读唯一主讲义并给出结论——
    是否学会了、是否有问题、是否能够从容开始下一节；不合格继续补，不得提交验收。

## 5. 统一验收入口（所有实现单元的外部合同）

```text
输入合同：N/M/K、seq_len、num_heads、head_dim、dtype、seed、device 等固定字段
输出合同：
  - correctness: bool, max_abs_err, max_rel_err, tolerance, reference_impl
  - timing:     measured_time, warmup, iters, timing_method
  - runtime:    env_name, framework, version, commit/tag, gpu, driver, cuda
  - evidence:   证据文件相对路径（benchmark 日志、NCU/NSYS 报告、SASS 片段）
```

存放约定：`src/` 实现、`scripts/` 一键复现与正确性/性能脚本、`docs/evidence/Txx/` 原始证据。

## 6. 证据分级与覆盖矩阵

- **A 级**：本机实现并测量（数值误差、吞吐、NCU/NSYS 报告、SASS 片段齐备）。
- **B 级**：本机只能运行与比较（例如容器内框架吞吐对比，不逐行改源码）。
- **C 级**：受单卡、sm_8.9、显存或架构约束，只能基于官方文档分析（如 TP/PP/CP、
  Hopper/Blackwell 特性）。C 级必须记录“为什么受限 + 官方恢复路径”。

`config/coverage-matrix.md` 是全量覆盖台账；`config/source-ledger.md` 是来源台账；
每完成一个 Ticket 必须更新，未核对的来源不得标注为已核实。

## 7. 目录与 Git 约定

```text
PLAN.md                         # 本文件：固定课程契约
AGENTS.md                       # AI 行为约束与每次回答模板
idea.md                         # 项目原始想法（只读参考）
docs/tickets/Txx-*.md           # 29 个固定任务书
docs/lectures/Txx-*.md          # 每轮唯一主讲义
docs/evidence/Txx/              # 原始命令输出与报告
config/day0-lock.json           # T00 生成的版本/资产锁文件
config/coverage-matrix.md       # 覆盖矩阵
config/source-ledger.md         # 官方来源台账
environments/*.sh               # 环境门禁脚本（只从仓库根运行）
src/ scripts/                   # 实现、一键复现与正确性/性能脚本
assets/ caches/ third_party/    # 大文件/依赖，Git 忽略
```

Git 规则：分支 `main`；每个 Ticket 一个提交；禁止 `push --force`、`reset --hard`、
`clean -fdx`；大文件只进 `assets/`/`caches/`（已被 `.gitignore` 忽略）。

## 8. 解锁协议（必须严格遵守）

1. 只有学习者本人在对话中明确说“解锁/开始 Txx”才能开始该 Ticket；AI 不得自动开始下一增量。
2. 一次只允许一个 Ticket 进行中；同一 Ticket 可多轮，但不允许跳到下一 Ticket。
3. Ticket 内被环境/网络阻塞：记录 blocker 证据（命令输出 + 官方链接），停下报告；
   不得绕过官方来源用记忆猜测 API。
4. 顺序：T00 → T01 → … → T28；T00 完成后 T01 变 `ready`，之后逐个解锁。
