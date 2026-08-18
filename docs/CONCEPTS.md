# GPP 核心概念词典（持久记忆，逐 Ticket 更新）

> 目的：把各 Ticket 涉及到的“面试/项目高频概念”集中留存。每个概念都写：一句话定义 →
> 怎么分析 → 本项目的证据在哪 → 更深的 Ticket 是哪个。不做百科全书，但涉及到的都要有。

## 1. 工具定位：每个工具“是什么、解决什么、什么时候选”

| 工具 | 是什么 | 抽象层级 | 你负责什么 / 它负责什么 | 什么时候选 |
| --- | --- | --- | --- | --- |
| PyTorch | 深度学习框架；`torch.matmul/F.relu` 等调度优化 kernel | Tensor 全自动 | 你写数学；调度/线程映射全自动 | 先建立正确语义、做黄金参考、日常模型开发 |
| CUDA C++ | NVIDIA 的 GPU 编程语言/运行时 | thread 手动 | 你写 kernel + launch + 内存管理 | 需要精细控制、教学底层、定制 kernel |
| Triton | Python DSL/编译器，生成 GPU kernel | program/block | 你写 block 级逻辑；编译器管线程与向量化 | 快速写高性能 kernel，可读性优于 CUDA |
| cuTile Python | NVIDIA 的 tile 级 Python DSL（基于 Tile IR） | tile | 你写 tile 搬运与运算；编译器管 tile 内并行 | 用 tile 语义表达矩阵算子，接 CuPy |
| CuTe DSL | CUTLASS 的 Python DSL，显式 layout/tile 控制 | 显式 tile/线程 | 你表达布局与切片；编译器生成 MLIR/SASS | 研究 layout、与 CUTLASS 生态一致 |
| CUTLASS C++ | NVIDIA 的 GEMM/Attention 模板库 | 模板/collective | 你选策略/组合 epilogue；库提供高性能模板 | 工业级高性能 GEMM 定制（T08 实操） |
| cuBLAS | NVIDIA 官方 BLAS 库 | 库调用 | 你调 API；库做几十年积累的优化 | 生产中直接要 GEMM 性能（T04 已做基线） |
| NCU | kernel 级 profiler | - | 看 Duration/SM/DRAM/L1/L2/occupancy/Roofline | 热点 kernel 为什么慢（每 Ticket 都用） |
| NSYS | 时间线 profiler | - | 看 API/memcpy/kernel/stream 顺序与重叠 | 先找哪里慢（WSL2 kernel 级受限，见台账） |
| SASS | 特定架构机器码 | - | 确认编译器最终生成什么指令 | 验证向量化/分支/FFMA 等（每 Ticket 都用） |

横向对比结论（T01–T04 实测）：同一简单算法，五个工具生成的 kernel 性能可接近
（T01/T02/T03 都 ≈18.8–35.5us 梯队）；差距主要来自抽象选择是否适合该算法
（T04 的 tile=1 cuTile 0.4 GFLOPS vs cuBLAS 7.7 TFLOPS）。

## 2. Memory-Bound / Compute-Bound / Latency-Bound：定义与分析方法

### 2.1 一句话定义

- **Memory-Bound**：时间主要花在等数据（访存），DRAM/L2 带宽利用率高，SM 算力大量空闲。
- **Compute-Bound**：时间主要花在数学计算，SM 的 FP/INT 数学管道利用率高，访存不是瓶颈。
- **Latency-Bound**：既没打满带宽也没打满算力，但有很多 stall（等长延迟依赖、等 barrier、
  等 load 返回），occupancy 不足以隐藏延迟。

### 2.2 判定流程（拿到一个 kernel 后按顺序做）

1. 算**算术强度**：`AI = FLOPs / bytes moved`。低 AI（如 ReLU≈0.13 FLOP/byte）偏向
   Memory-Bound；高 AI（大 GEMM 可达几十）偏向 Compute-Bound。峰值比
   `roofline ridge = peak FLOPs / peak bytes/s` 是分界线。
2. 跑 **NCU SpeedOfLight**：看 `DRAM Throughput`、`Compute (SM) Throughput` 和
   `Roofline Chart`（达到 FP32 峰值的百分比）。
3. 看 **stall 原因**：`SchedulerStats/WarpStateStats` 里 Long Scoreboard（等 global load）、
   Barrier、Short Scoreboard 等分布——占比高且带宽/算力都不满 → Latency-Bound。
4. 结合 **SASS** 验证：Memory-Bound 的 kernel 是大量 LDG/STG；Compute-Bound 的 GEMM 是
   大量 FFMA/HMMA；Latency-Bound 常看到依赖链和少得可怜的活跃 warp。
5. 下结论并注明证据等级（A 实测/B 运行/C 分析）。

### 2.3 本项目已实测的案例

| kernel | NCU 证据 | 判定 |
| --- | --- | --- |
| vector add（T01） | DRAM 92–93%，Compute 2–11% | Memory-Bound |
| ReLU 标量/向量（T02/T03） | DRAM 88–93%，Compute 2–15% | Memory-Bound |
| 朴素 GEMM（T04） | DRAM 5.9%，FP32 峰值 6%，SM 94.9%，L1 97.2%，LDG 58/IMAD 42/FFMA 29 | 指令/L1 管道瓶颈（既非纯 Memory 也非纯 Compute） |
| cuBLAS 1024³（T04） | 10.0 TFLOPS ≈ 接近 FP32 峰值 | Compute-Bound |

### 2.4 常见误区（本项目都踩过/讲过）

- 调用级时间反推带宽会因 L2 复用严重高估（T02 §5.9、T03 §5.9）。
- `Compute(SM) Throughput` 是**指令发射管道**利用率，不是“FLOP 用了百分之几”；
  FLOP 占比看 Roofline（T04 §5.4）。
- SM busy ≠ 算力被有效利用：朴素 GEMM 的 SM 94.9% 忙，但 FFMA 只有 29 条对 100 条杂指令。
- Memory-Bound 下单纯提高 occupancy/加 SM 不会变快；要减少字节或提高访存效率。

## 3. 其他高频概念速查（来源 Ticket）

| 概念 | 一句话 | 出处 |
| --- | --- | --- |
| SPMD/SIMT | 一份代码多数据；warp=32 线程同指令 | T01 §5.1/5.3 |
| grid/block/thread | 执行层级；全局下标公式 | T01 §5.3 |
| occupancy | 活跃 warp / 理论最大 warp；资源上限决定；T12 用 LaunchStats/Occupancy 截面 + SchedulerStats 联合判断 latency hiding 是否够 | T01 §5.4、T02 §5.5、T12 §5.4 |
| warp divergence | 同一 warp 走不同分支，两条路都执行 | T02 §5.4 |
| predication | 把 if 变成选择指令（FMNMX），避免分叉 | T02 §5.4 |
| 内存层次 | register→shared/L1→L2→global；越近越小越快 | T01 §5.5 |
| coalesced access | 相邻线程访问相邻地址 → 合并成 128B 事务 | T03 §5.2 |
| 128-bit 向量化 | LDG.E.128/ld.global.v4.b32；指令少 4 倍 | T03 §5.1–5.6 |
| alignment | float4 需 16B 对齐；cudaMalloc 保证 | T03 §5.3 |
| 尾部处理 | 主循环向量化 + 尾巴标量 | T03 §5.4 |
| NaN 语义 | ReLU(NaN)=NaN；fmaxf 是 maxNum 语义需 isnan 保护 | T03 §5.7 |
| CUDA stream/event | 命令队列/GPU 时间戳；异步与同步 | T01 §5.6 |
| 同步对照（capability baseline） | 某工具官方没有当前优化 API 时，用其官方同步实现跑同 shape 作为能力上限，并标 N/A；它不是该优化的实现 | T06 cuTile、T07 cuTile |
| conda environment / Docker image+digest / revision / SHA256 | 环境隔离、镜像指纹、版本快照、内容校验 | T00 §6.6 |
| arithmetic intensity | FLOP/byte；roofline 分界 | T04 §5.3 |
| 矩阵转置 / Aᵀ | B[j,i]=A[i,j]；H×W 变成 W×H；0 FLOP，只换访存顺序 | T09 §4.0 从零定义；T09/T10 五路径实操 |
| 行主序 / stride / contiguous | 内存一行行连续排，`A[y,x]` 在 `y*W+x`；行内 stride=1、跨行 stride=W；`a.t()` 只换 stride 视图，`a.t().contiguous()` 才复制成连续内存 | T09 §4.0、§5.2 |
| coalesced（合并访问） | 同一 warp 相邻线程访问相邻地址，一次事务用满 cache line；转置朴素版只能读/写二选一 | T09 §5.2 实测 readC/writeC |
| 数据复用/Tiling | 一个 block 复用一块数据，减少 global 读 | T04 §5.1/§5.6，T05 实操 |
| shared memory | block 内共享、容量小速度快；需 barrier 管理 | T05 §5.2–5.3 |
| `__syncthreads` / barrier | 同步 block 内所有线程的执行进度与 shared 可见性；漏掉=race；放进条件分支=死锁；树规约每轮都要放一次 | T05 §5.3、§5.7；T11 §6.2 树规约 9 个 BAR.SYNC 实测 |
| grid-stride loop | 线程按 `i += gridDim.x*blockDim.x` 跨网格取数，每线程可处理多元素，减少线程总量与 launch 开销 | T11 §5.1/§6.1 |
| reduction（定义） | 一组值经同一个二元运算折叠成一个值；可并行切分的前提是结合律；补边界用单位元（sum=0，max=-∞，min=+∞） | T11 §2 从零讲解 |
| reduction 两段式 / partial sum | 第一段每线程串行局部和（无通信），第二段 shared 树规约把局部和合成 block 部分和；跨 block 的部分和最后在 host 用 fp64 汇总 | T11 §2.4、§6.1/§6.3 |
| 求和顺序 / fp32 误差 | 同一组 fp32 数按不同顺序求和误差不同；块级树 + fp64 汇总通常比大域 fp32 sum 更接近 fp64 参考；容差要写在证据里 | T04 §5.5；T11 §6.5；T12 §5.5（shuffle 树与 shared 树结果差 1e-6 量级） |
| shared memory bank | 32 个 4B bank；同 bank 不同地址=conflict；同地址=broadcast；128-bit 冲突要看 warp 布局，用 NCU `l1tex__data_bank_conflicts_*` 实测 | T05 §5.7，T06 §5.1–5.2 实测，T10 §5.2 转置实测 |
| 边界 zero-fill / %M/%N | 越界数据变成对累加无贡献的值 | T05 §5.4 |
| `tl.dot` TF32 精度陷阱 | fp32 dot 默认可能走 TF32；用 `input_precision="ieee"` | T05 §4.2 |
| bank conflict / padding(+1) | shared 同 bank 多线程访问=conflict；转置 tile 无 pad 时 stride=32 全中 bank0（32-way），pad=+1 使 stride=33 与 32 互质而消除；NCU `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 量化 | T05 §5.6，T06 实测；T10 §5.2 实测 nopad=254,221（理论基线 253,952）/ pad=246≈0 |
| TILE_DIM / BLOCK_ROWS（tile 线程布局） | block=TILE_DIM×BLOCK_ROWS 线程，每线程搬 TILE_DIM/BLOCK_ROWS 个元素，BLOCK_ROWS 必须整除 TILE_DIM；conflict 总数=blocks×warps×循环次数×每 warp 额外冲突，与 8/16 拆分方式无关 | T10 §5.4 与官方 transpose.cu（S18f）对齐 |
| cuBLAS 列主序 | 行主序 C=A@B ⇔ 列主序 Cᵀ=BᵀAᵀ | T04 §4.6 |
| Softmax（定义/概率） | 每行 `exp(x_j)/Σexp(x_k)`：输出非负且和为 1；保持大小顺序；logits/注意力分数变成权重 | T13 §4.0 |
| Softmax 平移不变性 / 减 max | `softmax(x-m)==softmax(x)`；m=行 max 时指数 ≤0，避免 exp 上溢成 inf/NaN；下溢为 0 是合法概率 | T13 §4.0/§5.1；±1000 门禁实测 |
| 行归约映射（一行一 block） | `blockIdx.x=row`，block 内线程跨列 strided + shared 树；C 未对齐靠循环边界或 Triton mask 处理 | T13 §4.1/§5.3 |
| 3-pass Softmax | max → sum(exp) → normalize；朴素版每行读 3 遍；PyTorch 朴素中间张量使访存达 5MN 读/3MN 写；融合后理想 1 读 1 写 | T13 §4.0/§5.2；官方 tutorial 02 S01m |
| keepdims（归约广播） | 行归约后保留 (R,1) 维，才能与 (R,C) 广播；CUDA 标量天然广播，无此问题 | T13 §4.2/Q14 |
| MUFU.EX2 / FMNMX | SASS 里 expf 落 MUFU.EX2，max 落 FMNMX；与 T11/T12 的 BAR.SYNC 归约指纹叠加成 softmax kernel | T13 §5.4；t13-cuda-sass.txt |
| online softmax（running max/sum） | 维护 `m=max` 与 `l=Σexp(x-m)`；新值 >m 时 `l←l·exp(m_old-m_new)+1`；两段按“大 max 为主”合并；指数恒 ≤0 不溢出 | T14 §4.0/§5.1；CUDA onlineMerge |
| 算子融合的访存收益 | 3-pass 3R+1W → online+shared 1R+1W；本 shape L2 流量 65.73→33.64MB、DRAM read 几乎不变（L2 吸收重复读）、Duration 92.32→81.38us | T14 §5.2/§6；t14-vs-t13-bytes.txt |
| prefill / decode | prefill 一次算完 prompt 所有 token；decode 逐 token 生成；KV Cache 只对 decode 有效，因为只有 decode 反复复用历史 K/V | T16 §4.0/§5.1–5.2 |
| CUDA FA 手工映射 | 四块 shared（sQ/sK/sV/sS）+ 外层 K/V tile 循环 + 每行 rowM/rowL/rowA 状态；四个 barrier 对应四个读写依赖；教学版用 FFMA 不用 Tensor Core | T18 §4.1/§5.1；SASS FFMA 381/BAR.SYNC 5 |
| FlashAttention（IO-Aware tiling） | 外层 K/V tile、内层 Q tile；Q tile 固定后顺序消费 K/V，把 K/V 重复读从 N 次降到 N/BLOCK_M；N×N 分数矩阵只在片上出现 | T17 §4.0/§5.1；NCU L2 3.61GB→71.41MB |
| online (m,l,acc) | T14 (m,l) 推广到加权输出：新 max 时 `alpha=exp(m_old-m_new)`，`l←l*alpha+l_new`，`acc←acc*alpha+PV`，最后 O=acc/l | T17 §4.1/§5.5 |
| KV Cache / DynamicCache | 缓存历史 token 的 K/V，形状 `[B,H,seq,D]`，decode 每步只投影当前 token 并沿 seq append；省投影重复计算，不省 attention 读历史 | T16 §4.1/§5.3（S16a） |
| Attention 标准公式/计算图 | `Attention(Q,K,V)=softmax(QKᵀ/√d_k)V`；Q 查询、K 键、V 值，展开为 `S=QKᵀ/√d → mask(-inf) → softmax → O=PV`；两个 GEMM 夹一个行归约；batch/head 铺平成 grid 第二维并行 | T15 §4.0/§5.1（S22） |
| causal vs padding mask | causal：k>q 置 -inf，禁止看未来；padding：越界槽置 -inf；两者都必须存在，否则 softmax 把空槽当 0 权重 | T15 §5.3/Q3/Q7 |
| 朴素 Attention 访存 | 每 query 读整段 K/V，总访存 O(B·H·N²·D)；小 shape 被 L1/L2 吸收时呈 Latency/barrier-Bound | T15 §5.4/§7；NCU Barrier 27.24 |
| persistent program（Triton） | 用少量 program 循环处理多行；warmup 拿 regs/smem 算 occupancy 定 num_programs；warmup 的 stride 必须与真实 launch 一致，否则触发错误特化 | T14 §4.3/Q9/Q15 |
| L2 命中率与 wall GB/s | 热循环计时反推带宽会因 L2 命中被高估（T13 wall 1163.78GB/s vs NCU DRAM 79.97%）；物理带宽以 NCU 为准 | T13 §6/Q11；CONCEPTS §2.4 同口径 |

| fp32 累加误差 | 长 K 累加误差累积；参考用 fp64 | T04 §5.5 |
| reduction 树规约轮数 | block 大小 2^k 时需 k 轮，barrier=k+1（写满 1 次 + 每轮 1 次）；T12 已把 32→1 交给 shuffle | T11 §2.4/§6.1 |
| warp / lane | warp=硬件调度的 32 线程组（SIMT 同一条指令）；lane=warp 内线程编号 `threadIdx.x%32` | T01 §5.3；T12 §4.0 正式使用 |
| warp shuffle（`__shfl_down_sync`） | 同一 warp 内 lane 间直接交换寄存器值，不经过 shared/global；down 让 lane i 读 lane i+offset，5 轮树后只有 lane 0 有完整和；`_sync` 强制 mask 内 lane 收敛 | T12 §4.0/§5.2；SASS `SHFL.DOWN` 证据 docs/evidence/T12/t12-cuda-sass.txt |
| sync mask | shuffle 等 warp 原语的参与 lane 位图；调用的 lane 集合必须 == mask 集合，读目标也必须在 mask 内，否则 undefined/hang；shuffle 不是 memory barrier | T12 §5.3；官方 invalid 例子 S10g |
| bfly（butterfly） | `__shfl_xor_sync`/`shfl.sync.bfly`：lane i 与 lane i^offset 配对，5 轮后所有参与 lane 都有完整和；Triton tl.sum 自动生成 bfly，CuTe 官方示例手写 bfly | T12 §5.2；证据 t12-triton-ptx-hits.txt |
| latency hiding | 当一个 warp 等 load/依赖时，scheduler 切到另一个 resident warp 发指令，把等待时间变成有效工作；occupancy 高=备选 warp 多，但只有带宽没满时才有加速空间 | T12 §5.4/§7；NCU Active Warps 10.82 + No Eligible 89.86% + DRAM 86.99% 联合证据 |
| No Eligible（SchedulerStats） | scheduler 找不到可发射 warp 的周期占比；高不代表一定 Latency-Bound，要配合 DRAM/Compute 吞吐看 | T12 §5.4/§7；docs/evidence/T12/t12-cuda-ncu.txt |
| Warp stall 分解（Long Scoreboard/Barrier/Short Scoreboard） | 每发出 1 条指令、每个活跃 warp 平均在某 stall 状态等的周期数；Long Scoreboard=等 global load、Barrier=等 `__syncthreads`、Short Scoreboard=等 shared；读比值要记住分母是“每条指令”，总指令少的 kernel 比值会被拉高 | T12 §5.4；证据 t12-cuda-ncu-stall.txt / t11-cuda-ncu-stall-compare.txt |
| LLM / 大语言模型 | 根据前文预测下一个 token 的神经网络；生成 = 反复“预测下一个 token → 拼回去 → 再预测” | T19 §4.1 |
| Qwen2.5-0.5B-Instruct | 阿里云 Qwen2.5 系列约 5 亿参数的指令微调对话模型；本项目固定使用该快照 | T19 §4.2；assets/modelscope/qwen2.5-0.5b-instruct |
| Transformers 库 | HuggingFace 的统一模型/分词器加载与推理库；`AutoModelForCausalLM` / `AutoTokenizer` 按 config 自动选实现 | T19 §4.3；S16/S20a |
| ModelScope / 快照 / revision | 模型/数据集平台；快照是某时刻下载的固定文件集合；revision 是版本标识；固定快照保证可复现 | T19 §4.4；S14/S20 |
| tokenizer / token / round-trip | tokenizer 把文本编成 token id、把 id 解回文本；round-trip 是“文本→id→文本”能否还原的检查 | T19 §4.6 |
| chat template / 特殊标记 | 把 system/user/assistant 角色拼成模型见过的格式；Qwen 用 `<|im_start|>/<|im_end|>` 标记边界 | T19 §4.7 |
| greedy decoding / deterministic | 每次选概率最高的 token，无随机采样；同权重同输入同配置下输出确定 | T19 §4.8 |
| prefill / decode / generate | prefill=并行处理整个 prompt；decode=逐 token 自回归生成；generate=prefill+decode+停止判断 | T19 §4.9 |
| prompt suite | 固定的一组测试问题，供后续框架在同一模型上公平对比 | T19 §4.10；src/t19_qwen_baseline/prompt_suite.py |
| PagedAttention | 把 KV Cache 分成固定大小块、按需分配，减少显存碎片，提高并发 | T24 §4.0.1；vLLM 官方 |
| Continuous Batching | 请求结束立刻补新请求，不用等整批结束，提高 GPU 利用率；vLLM 叫 Continuous Batching，TRT-LLM 叫 In-Flight Batching，SGLang 也有类似机制，本质是同一个动态批处理思想 | T24 §4.0.2；vLLM/SGLang/TRT-LLM |
| Chunked Prefill | 长 prompt 拆成小块逐步 prefill，避免单请求独占 GPU | T24 §5.3；vLLM/TRT-LLM |
| Prefix Caching | 缓存相同前缀的 KV，新请求命中前缀不用重算 | T24 §5.3；vLLM |
| Speculative Decoding | 草稿模型先预测多个 token，大模型一次验证，减少串行生成次数 | T24 §4.0.3；vLLM/SGLang/TRT-LLM |
| RadixAttention | 树状复用共享前缀 KV，多请求前缀相同时省重复计算 | T24 §5.3；SGLang |
| In-Flight Batching | 动态管理请求执行，context 和 generation 阶段重叠，最大化 GPU 利用 | T24 §5.3；TRT-LLM/vLLM |
| TensorRT engine | 把模型提前编译成针对特定 GPU 的 engine，运行时更快 | T24 §5.3；TRT-LLM |
| FP8/FP4 量化 | 低精度推理，性能更高、显存更低（H100/B200 等） | T24 §5.3；TRT-LLM/vLLM |
| Disaggregated Serving | 分离 prefill 和 decode 阶段到不同 GPU，优化资源利用 | T24 §5.3；TRT-LLM/vLLM |
| LoRA | 只训练低秩适配参数，显存低、速度快 | T24 §5.3；ms-swift/TRT-LLM |
| GRPO | 强化学习算法族，用于偏好/奖励优化 | T24 §5.3；ms-swift |
## 4. 更新日志

- 2026-08-15：随 T01–T04 回溯审计创建，并修正“Bound 判定只看调用级带宽”的旧口径。
- 2026-08-16：T10 预验收——补转置场景 bank conflict/padding 词条与证据。
- 2026-08-16：T11 补强——从零定义 reduction（定义/例子/串行树/术语表），五路径核心代码逐行讲解并回写词条。
- 2026-08-16：T10 对齐官方——BLOCK_ROWS 8→16，补 tile 线程布局词条与 8/16 对照证据。
- 2026-08-16：T09/T10 讲义重新验收——补“什么是转置/行主序/contiguous/coalesced”词条，
  五路径全部改为核心代码逐行讲解。
- 2026-08-16：T12 实现——新增 warp/lane、shuffle、sync mask、bfly、latency hiding、No Eligible 词条。
- 2026-08-16：T12 预验收——补 Warp stall 分解词条与 NCU stall 证据。
- 2026-08-16：T13 实现——新增 Softmax 定义/平移不变性/行归约映射/3-pass/keepdims/MUFU/L2 口径词条。
- 2026-08-16：T14 实现——新增 online softmax、融合访存收益、persistent program 词条。
- 2026-08-16：T15 实现——新增 Attention 计算图、causal/padding mask、朴素访存词条。
- 2026-08-16：T16 实现——新增 prefill/decode、KV Cache/DynamicCache 词条。
- 2026-08-16：T17 实现——新增 FlashAttention IO-Aware tiling、online (m,l,acc) 词条。
- 2026-08-16：T18 实现——新增 CUDA FA 手工映射词条。
- 2026-08-17：T19 实现——新增 LLM/Qwen/Transformers/ModelScope/tokenizer/chat template/greedy/prefill-decode/prompt suite 词条。
- 2026-08-17：T24 阶段性——新增 PagedAttention、Continuous Batching、Chunked Prefill、Prefix Caching、Speculative Decoding、RadixAttention、In-Flight Batching、TensorRT engine、FP8/FP4、Disaggregated Serving、LoRA、GRPO 词条。
