# GPP 知识覆盖矩阵（对齐 PLAN v1.3）

> 状态：`待学` → `学习中` → `已过关`。证据级别：A=本机实现并实测；B=本机运行对比；
> C=仅官方文档分析（须写受限原因与恢复路径）。每完成一个 Ticket 立即更新，不得批量补记。
> 阶段二（T04–T10）已于 2026-08-16 验收通过；T18/T19 已于 2026-08-17 收尾（T00–T19 done）。结论记录在 `PLAN.md` 状态行，明细在各 Txx 讲义/证据。

## 1. 算子节点 × 实现路径（v1.1 细粒度）

`P`=该 Ticket 计划覆盖；`N/A`=官方能力不适用/本层不要求，须在证据中写理由；`-`=不涉及。
基线层保证五路径横向对比；优化层只做官方示例支持的路径。

| Ticket | PyTorch | CUDA C++ | Triton | cuTile Python | CuTe DSL | CUTLASS C++ | cuBLAS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T01 Vector Add 基线 | P | P | P | P | P | N/A | N/A |
| T02 ReLU 标量基线 | P | P | P | P | P | N/A | N/A |
| T03 ReLU 向量化 | P(参考) | P | P | 官方示例 | 官方示例 | N/A | N/A |
| T04 GEMM 朴素基线 | P | P | P | P | P | N/A（归 T08） | P（基线） |
| T05 GEMM Tiling | P(参考) | P | P | P | P | - | - |
| T06 GEMM 共享内存优化 | P(参考) | P | P | P(编译器对照) | 官方示例 | - | - |
| T07 GEMM 异步流水线 | P(参考) | P | P | P(latency hint) | P(cp.async 双缓冲) | - | - |
| T08 GEMM Tensor Core | P(参考) | P(mma) | - | - | - | P | P（同 dtype） |
| T09 Transpose 朴素基线 | P | P | P | P | P | N/A | N/A |
| T10 Transpose Tile | P(参考) | P | P | 官方示例 | 官方示例 | N/A | N/A |
| T11 Reduction 共享内存基线 | P | P | P | P | P(官方示例纯 smem 教学版) | N/A | N/A |
| T12 Reduction Warp Shuffle | P(参考) | P | P(观察生成代码) | P(官方 ct.sum 能力对照，shuffle N/A) | P(官方 warp shuffle 树) | N/A | N/A |
| T13 Softmax 朴素基线 | P | P | P | P | P | N/A | N/A |
| T14 Softmax Online/融合 | P(参考) | P | P | 官方示例 | 官方示例 | N/A | N/A |
| T15 Attention 朴素 | P(双参考) | P | P | 官方示例 | 官方示例 | N/A | N/A |
| T16 KV Cache | P(语义参考) | P | P | P(官方能力检查/N-A) | P(官方能力检查/N-A) | N/A | N/A |
| T17 FlashAttention Triton | P(SDPA 参考) | P(T18 专用/N-A) | P | P(官方 flash 层检查/N-A) | P(官方 flash 层检查/N-A) | N/A | N/A |
| T18 FlashAttention CUDA | P(参考) | P | P(T17 已完成/N-A) | P(官方能力实测) | P(官方能力实测) | N/A | N/A |

## 2. 核心知识点台账

| 知识点 | 目标 Ticket | 状态 | 证据级别 | 证据文件 |
| --- | --- | --- | --- | --- |
| CPU vs GPU、host/device、SPMD、CUDA 整体认知 | T01 | 已过关（基础） | A | docs/lectures/T01-vector-add.md §5.1 |
| CUDA 编译运行流程、PTX vs SASS | T01 | 已过关（基础） | A | docs/lectures/T01-vector-add.md §5.2、t01-cuda-sass.txt |
| grid / block / thread 层级与索引 | T01 | 已过关 | A | docs/evidence/T01/、lectures/T01 §5.3 |
| kernel launch、CUDA 编译运行流程 | T01 | 已过关 | A | docs/evidence/T01/t01-cuda-ncu*.txt |
| warp、SIMT、divergence、硬件结构（SM、调度） | T01/T12 | 已过关（T01 概念；T12 深挖 shuffle 并验收） | A | docs/lectures/T01-vector-add.md §5.3–5.4、lectures/T12 §4.0/§5.3 |
| 存储体系：Global/Shared/Constant/L1/L2/Register | T01/T05 | 已过关（Global+地图）；Shared/Constant 实操后置 T05/T06 | A | docs/lectures/T01-vector-add.md §5.5 |
| CUDA stream（基础）、异步、event/sync | T01 | 已过关（基础） | A | docs/lectures/T01-vector-add.md §5.6、t01-nsys-stats.txt |
| Tensor Core / CUDA Core 概念区分 | T01/T08 | 已过关（概念）；T08 实操 | A | docs/lectures/T01-vector-add.md §5.7 |
| 连续批处理概念预览 | T01/T24 | 已过关（概念）；T24 实测 | A | docs/lectures/T01-vector-add.md §5.8 |
| 五工具算子开发流程横向对比 | T01 | 已过关（第一轮） | A | docs/lectures/T01-vector-add.md §5.9、scripts/run_t01_all.sh |
| 性能工具三件套使用与分析 | T01 | 已过关（第一轮） | A | docs/lectures/T01-vector-add.md §5.10、docs/evidence/T01/ |
| 元素级索引、Grid 配置与边界处理 | T02 | 已过关 | A | docs/evidence/T02/、lectures/T02 |
| 分支与 warp divergence、predication（FMNMX） | T02 | 已过关（初探） | A | docs/evidence/T02/t02-cuda-sass.txt |
| 合并访问（coalesced access） | T03 | 已过关 | A | docs/evidence/T03/t03-cuda-sass.txt、lectures/T03 |
| 向量化加载（128-bit） | T03/T06 | 已过关（T03 global；T06 shared） | A | docs/evidence/T03/、docs/evidence/T06/t06-cuda-sass.txt |
| Memory-Bound vs Compute-Bound | T03/T04/T08 | 已过关（T01–T03 三次取证） | A | docs/evidence/T01–T03 的 NCU 报告 |
| 极值/NaN 语义（ReLU） | T03 | 已过关 | A | 各 T03 脚本 `*_extreme` 输出 |
| 朴素 GEMM 索引映射与访存/计算比 | T04 | 已过关 | A | docs/evidence/T04/、lectures/T04 |
| 算术强度（arithmetic intensity）与 Roofline | T04 | 已过关 | A | lectures/T04 §5.3–5.4、CONCEPTS.md §2 |
| 六种工具定位与选择（PyTorch/Triton/cuTile/CuTe/CUTLASS/cuBLAS） | T01–T04 | 已过关（持续更新） | A | docs/CONCEPTS.md §1 |
| Memory/Compute/Latency-Bound 三分类与判定流程 | T01–T04 | 已过关（Memory/指令瓶颈实测；Latency 定义+后置实测） | A | docs/CONCEPTS.md §2、lectures/T04 §5.7 |
| cuBLAS API 与库基线 | T04/T08 | 已过关（T04 Sgemm）；T08 高级基线 | A | src/t04_gemm_naive/cuda/gemm_naive.cu |
| NCU Roofline / 指令瓶颈判断（SM busy≠FLOP busy） | T04 | 已过关 | A | docs/evidence/T04/t04-cuda-ncu.txt |
| Tile 分块与数据复用 | T05 | 已过关 | A | docs/evidence/T05/、lectures/T05 |
| Shared memory 基础与 barrier 同步 | T05 | 已过关（基础）；T06 优化 | A | docs/evidence/T05/t05-cuda-sass.txt |
| SchedulerStats / No-Eligible stall 读法 | T05 | 已过关（初识） | A | docs/evidence/T05/t05-cuda-sections.txt |
| Bank Conflict 与 padding/swizzle | T06/T10 | 已过关（T06 四档实测；T10 转置 nopad/pad 量化） | A | docs/evidence/T06/t06-gemm*.txt、docs/evidence/T10/t10-*-ncu.txt |
| 128-bit 共享内存访问（STS.128/LDS.128） | T06 | 已过关 | A | docs/evidence/T06/t06-cuda-sass.txt |
| block 大小与 shape 的关系（Triton 配置对比） | T06 | 已过关 | A | src/t06_gemm_smem/triton_gemm.py |
| Double Buffer / cp.async / Pipelining | T07 | 已过关（技术+诚实收益判断） | A | docs/evidence/T07/t07-cuda-sass.txt、lectures/T07 |
| num_stages 软件流水线（Triton） | T07 | 已过关 | A | src/t07_gemm_pipeline/triton_gemm.py |
| Tensor Core mma 路径 | T08 | 已过关（五路径 + CUTLASS） | A | docs/evidence/T08/、lectures/T08 |
| fp16/bf16/tf32 数值格式与容差 | T08 | 已过关 | A | src/t08_gemm_tensorcore/ |
| Hopper/Blackwell 新特性 | T07/T08 | 已过关（sm_89 对应物：Tensor Core / cp.async / pipeline 实操）；专属 WGMMA/TMA 等 C 级，归 T17/T18 语境再遇时核对 | C | lectures/T07、lectures/T08、source-ledger §3 |
| blocksize/gridsize 与 shape 选择 | T06/T09/T10 | 已过关（初阶：T06 Triton 配置对比、T09/T10 三 shape 边界）；系统调参后置 T12/T17/T18 | A | src/t06_gemm_smem/triton_gemm.py、docs/evidence/T09/、docs/evidence/T10/ |
| 矩阵转置定义 / 行主序 / contiguous | T09 | 已过关（讲义重新验收补全，零基础可读） | A | lectures/T09 §4.0–4.1 |
| 二维线程布局与矩阵索引 | T09 | 已过关 | A | docs/evidence/T09/、lectures/T09 §4.2–4.5 |
| Shared Memory 分块（转置） | T10 | 已过关 | A | docs/evidence/T10/、lectures/T10 |
| Bank Conflict 转置专项（nopad/pad 量化） | T10 | 已过关 | A | docs/evidence/T10/t10-nopad-ncu.txt、t10-pad-ncu.txt、t10-blkrows8-*-ncu.txt |
| 线程同步与 shared memory 协作 | T11 | 已过关 | A | docs/evidence/T11/、lectures/T11 |
| warp 规约 vs block 规约 | T12 | 已过关 | A | docs/evidence/T12/t12-cuda-sass.txt、lectures/T12 §5.1 |
| warp shuffle（__shfl_*） | T12 | 已过关 | A | docs/evidence/T12/t12-cuda-sass.txt、t12-triton-ptx-hits.txt、lectures/T12 §5.2 |
| occupancy 与 latency hiding | T07/T12 | 已过关（T07 cp.async 初识 + T12 正式实测） | A | docs/evidence/T12/t12-cuda-ncu.txt、t12-cuda-ncu-stall.txt、lectures/T12 §5.4 |
| Softmax 数值语义与行归约 | T13 | 已过关 | A | docs/evidence/T13/t13-run-all.txt、lectures/T13 §4.0/§5.1 |
| Online Softmax | T14/T17 | 已过关（T14 实测；T17 在 FlashAttention 语境复用） | A | docs/evidence/T14/t14-run-all.txt、t14-cuda-ncu.txt、lectures/T14 §4.0/§5.1 |
| 算子融合的访存收益 | T14 | 已过关 | A | docs/evidence/T14/t14-vs-t13-bytes.txt、t14-cuda-sass.txt、lectures/T14 §5.2/§6 |
| Attention 计算图映射 | T15 | 已过关 | A | docs/evidence/T15/t15-run-all.txt、lectures/T15 §4/§5.1 |
| KV Cache 原理 | T16/T19 | 已过关（T16 实测；T19 在 Transformers 模型语境复用） | A | docs/evidence/T16/t16-run-all.txt、t16-vs-wall-bench.txt、lectures/T16 |
| FlashAttention：IO-Aware Tiling | T17/T18 | 已过关（T17 Triton 版、T18 CUDA 版均验收通过） | A | docs/evidence/T17/、docs/evidence/T18/t18-run-all.txt、docs/evidence/T18/t18-wall-median.txt、lectures/T18 |
| 固定快照生成正确性/确定性/基线指标 | T19 | 已过关（CPU 验证可运行；GPU 指标待真实机复跑） | A(CPU)/B(GPU待) | docs/evidence/T19/t19-run-all.txt、lectures/T19 |
| 固定 prompt suite（T20–T24 复用） | T19–T24 | 已过关（T19 定义并验证） | A | src/t19_qwen_baseline/prompt_suite.py、lectures/T19 |
| vLLM Serving 流程与单框架指标 | T20 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T20/t20-verify.txt、lectures/T20 |
| SGLang Serving 流程与单框架指标 | T21 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T21/t21-verify.txt、lectures/T21 |
| TensorRT-LLM Serving 流程与单框架指标 | T22 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T22/t22-verify.txt、lectures/T22 |
| ms-swift infer/deploy 流程与单框架指标 | T23 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T23/t23-verify.txt、lectures/T23 |
| PagedAttention | T24 | 学习中（概念已讲；真实数据待 T20–T23 复跑后补） | B/C | lectures/T24 §4.0.1 |
| Continuous Batching | T24 | 学习中（概念已讲；真实数据待 T20–T23 复跑后补） | B/C | lectures/T24 §4.0.2 |
| 投机解码 | T24 | 学习中（概念已讲；真实支持待框架实测） | B/C | lectures/T24 §4.0.3 |
| DP / DDP | T28 | 已过关（概念/对比已讲；多卡实测标 C） | B/C | lectures/T28 §5.1 |
| ZeRO（1/2/3） | T26/T28 | 已过关（概念/配置已讲；单卡实测待真实机） | B | lectures/T26 §4.0.2/§5.2、lectures/T28 §5.1 |
| FSDP | T28 | 已过关（概念/对比已讲；多卡实测标 C） | C | lectures/T28 §5.1 |
| TP / PP / SP / CP | T28 | 已过关（概念/对比已讲；多卡实测标 C） | C | lectures/T28 §5.1 |

## 3. 工具与框架台账

| 工具/框架 | 目标 Ticket | 状态 | 证据级别 | 证据文件 |
| --- | --- | --- | --- | --- |
| Day 0 环境与资产门禁（conda/Docker/ModelScope/SHA256） | T00 | 已过关 | A | docs/evidence/T00/ |
| NSYS 时间线（API/kernel/memcpy/stream） | T01–T18 | 已过关（API/memcpy/stream 级，T01–T18 均存证据）；kernel 时间线受 WSL2 限制=C 级 | A(API)/C(kernel) | docs/evidence/T01–T18/t*-nsys*.txt、source-ledger §3 |
| NCU 热点分析（SM/occupancy/带宽/命中率/stall/Tensor Core） | T01–T18 | 已过关（阶段二全项：occupancy T04、SchedulerStats T05、Tensor 管道 T08、DRAM/L1/bank T09/T10；T12 增加 SHFL/BAR 与 occupancy/latency 对照；T18 增加 No Eligible/Latency-Bound 对照） | A | docs/evidence/T04–T18 |
| SASS 阅读（cuobjdump/nvdisasm） | T01–T18 | 已过关（阶段二：FFMA/LDS/STS/BAR.SYNC/CP.ASYNC/HMMA；T12 新增 SHFL.DOWN；T18 新增无 HMMA 教学版指纹） | A | docs/evidence/T04–T18 |
| PyTorch 数学语义与正确性参考 | 全程 | 已过关（T01–T18 全部 fp64/SDPA 黄金参考流程） | A | src/t01_vector_add/common.py、各 Ticket 讲义 |
| Triton 开发与性能 | T01–T17 | 已过关（T01–T10：元素级/tile/pipeline/tensor core/transpose 均跑通） | A | src/t0{1..10}_*/triton_*.py |
| cuTile Python | T01–T18 | 已过关（T01–T10：Quick Start/官方 Transpose/GEMM 示例均跑通；T15–T18：官方 FMHA 能力实测） | A | src/t0{1..10}_*/cutile_*.py、src/t1{5..8}_*/cutile_*.py |
| CuTe DSL | T01–T18 | 已过关（T01–T10：vectorized_array/smem/barrier/官方 tensorop 示例均跑通；T15–T18：官方 flash_attention_v2 能力实测） | A | src/t0{1..10}_*/cute_*.py、src/t1{5..8}_*/cute_*.py |
| CUTLASS C++ | T08 | 已过关（官方 `14_ampere_tf32_tensorop_gemm` 构建+运行） | A | src/t08_gemm_tensorcore/、docs/evidence/T08/ |
| cuBLAS | T04/T08 | 已过关（T04 Sgemm 基线；T08 高级基线） | A | src/t04_gemm_naive/cuda/gemm_naive.cu、docs/evidence/T08/ |
| Transformers / Qwen2.5 基线 | T19 | 已过关（CPU 验证可运行；chat template、确定性、prefill/decode 指标已记录） | A(CPU)/B(GPU待) | src/t19_qwen_baseline/、docs/evidence/T19/t19-run-all.txt、lectures/T19 |
| vLLM | T20 | 已过关（学习者验收通过；使用流程/命令/客户端已提供） | B | docs/evidence/T20/t20-verify.txt、scripts/run_t20_all.sh、lectures/T20 |
| SGLang | T21 | 已过关（学习者验收通过；使用流程/命令/客户端已提供） | B | docs/evidence/T21/t21-verify.txt、scripts/run_t21_all.sh、lectures/T21 |
| TensorRT-LLM | T22 | 已过关（学习者验收通过；使用流程/命令/客户端已提供） | B | docs/evidence/T22/t22-verify.txt、scripts/run_t22_all.sh、lectures/T22 |
| ms-swift（推理） | T23 | 已过关（学习者验收通过；使用流程/命令/客户端已提供） | B | docs/evidence/T23/t23-verify.txt、scripts/run_t23_all.sh、lectures/T23 |
| 推理统一对比 | T24 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T24/t24-verify.txt、src/t24_inference_compare/compare_frameworks.py、lectures/T24 |
| PyTorch 最小训练循环/checkpoint/单卡指标 | T25 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T25/t25-verify.txt、t25-run-all.txt、lectures/T25 |
| PyTorch 训练 | T25 | 已过关（学习者验收通过；训练/推理/评测脚本已提供） | B | src/t25_pytorch_train/train_baseline.py、infer_checkpoint.py、eval_checkpoint.py、docs/evidence/T25/、lectures/T25 |
| DeepSpeed 最小接入与 ZeRO 单卡实测 | T26 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T26/t26-verify.txt、lectures/T26 |
| ms-swift SFT/LoRA Quick Start | T27 | 已过关（学习者验收通过；真实机复跑命令已提供） | B | docs/evidence/T27/t27-verify.txt、lectures/T27 |
| DeepSpeed | T26 | 已过关（学习者验收通过；使用流程/命令/讲义已提供） | B | src/t26_deepspeed_train/train_deepspeed.py、docs/evidence/T26/t26-verify.txt、lectures/T26 |
| ms-swift（训练） | T27 | 已过关（学习者验收通过；使用流程/命令/讲义已提供） | B | scripts/run_t27_all.sh、src/t27_ms_swift_train/run_sft.py、docs/evidence/T27/t27-verify.txt、lectures/T27 |
| 并行架构统一对比 | T28 | 已过关（对比脚本/讲义完成；多卡实测标 C） | B/C | src/t28_parallel_compare/compare_parallel.py、docs/evidence/T28/t28-verify.txt、lectures/T28 |
