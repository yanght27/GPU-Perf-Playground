# T06 GEMM 共享内存优化（唯一主讲义）

- Ticket：T06
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**Shared Memory Bank Conflict 消除与 128-bit 共享内存访问**
- 环境：gpp-core（PyTorch/Triton） / 系统 nvcc（CUDA） / gpp-cutile（cuTile） / gpp-cute（CuTe DSL）
- 官方来源：S01f、S02f、S10f（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`
- 本节导读：**一句话目标**——把 T05 的 tiled GEMM 用共享内存优化量化提速，掌握 bank conflict 与 128-bit 访问；**依次学到**——①shared memory 的 bank 与 conflict；②STS.128/LDS.128 是什么、为什么有用；③四档 CUDA 变体怎么实现；④NCU conflict 指标怎么读；⑤“先向量化、再消冲突”的优化顺序；**学完应能回答**——D 档为什么同时最快且无冲突？朴素 padding 为什么可能更慢？；**相关工具/技术**——PyTorch、CUDA C++、Triton、cuTile、CuTe DSL、NCU/SASS/NSYS。
- 本节内容：**要解决的问题**——T05 的 CUDA tiled 版还慢：每线程只搬 4B，且 shared 访问可能因 bank 排队；**核心手段**——①128-bit 共享内存访问：一次搬 16B，指令少 4 倍；②消除 bank conflict：padding 让线程错开 32 个 4B bank；**怎么实现**——CUDA 四档：基线/pad/float4/float4+pad；Triton 换 block 配置；CuTe 用 padding 版；**怎么验证**——正确性全 PASS；NCU conflict 指标 A=0、C=524288、D=0；SASS 见 STS.128/LDS.128；**最终结论**——先解决主要矛盾（搬移宽度），再解决剩余矛盾（conflict），一切以 NCU 数据为准。

## 1. 上一轮问题回答

T05 已验收。T05 留下一个被点名的线索：CUDA tiled 版有大量 no-eligible stall。
T06 用**四档 CUDA 实现**把“bank conflict 到底有没有、padding 到底有没有用、128-bit
到底带来什么”一次性量化清楚。

## 2. 规范实现与官方来源

| 路径 | 依据 |
| --- | --- |
| CUDA | Programming Guide「Maximize Memory Throughput」的 shared/bank/vector 原则 |
| Triton | 官方 tutorial 03 的 BM/BN/BK autotune 配置 |
| CuTe DSL | 官方 swizzle 实现与 03_gemm_tiled_smem 的 smem 用法 |

## 3. 本轮实现结果（五路径；CUDA 四档对比，正确性全部 CORRECT_PASS）

| 档 | 512³ 事件计时 | 1024³ 事件计时 | NCU(512) Duration | bank conflict 实测 |
| --- | --- | --- | --- | --- |
| A 基线 BS=16 | 0.178 ms | 5.7–10.7 ms（波动大） | 184.54 us | ld=0, st=0 |
| B pad BS=16 | 0.256 ms（更慢） | 1.97 ms | 273.15 us | ld=0, **st=524288** |
| C float4 BS=32 | 0.092 ms | 0.617 ms | 93.86 us | **ld=524288**, st=0 |
| D float4+pad BS=32 | **0.084 ms** | **0.575 ms** | **88.29 us** | **ld=0, st=0** |

其余四路径（PyTorch 参考、Triton 两配置、cuTile tile=16/32、CuTe padding）正确性全部 CORRECT_PASS；Triton 512³ 5.62TF/1024³ 6.15TF；cuTile 的 `ct.load/ct.mma` 是官方同步实现，编译器负责 shared/bank/向量化，本 Ticket 作为**能力对照**（不是手写 bank 优化），术语见 `docs/CONCEPTS.md` 的“同步对照”。

**三个反直觉但真实（NCU 实测）的结论**：
1. A 基线在这个 warp 布局下**没有** bank conflict——所以 T05 说“下一步消 conflict”
   要先纠正：真正的第一问题是“每线程只搬 4B、块太小”。
2. B 的朴素 padding 反而**制造了 store conflict**，512 更慢；1024 变快主要来自其他
   因素（block 复用/缓存状态），不能归因于“消 conflict”。
3. C 的 float4 提速 2–9 倍，但带来 ld conflict；D 把行宽 pad 到 9 后 **conflict 归零**，
   在 C 基础上再快约 6%。**正确顺序：先向量化，再为向量布局做 padding。**

## 4. 核心代码与逐行解释

### 4.1 四档共同的骨架

```cuda
__shared__ float4 As4[BS][COLS];   // C 档：无 pad；D 档：COLS+1
...
As4[ty][tx] = av;                  // 一条 STS.128 写 16 字节
__syncthreads();
for (k...) acc += a * Bs4[k][tx]...; // 一条 LDS.128 读 16 字节
```

### 4.2 C 档 float4 搬 tile 的逐行

```cuda
int tx = threadIdx.x;              // 0..7：负责一行里的第 tx 个 float4
int ty = threadIdx.y;              // 0..31：负责第 ty 行
int col4 = blockIdx.x*BS + tx*V;   // 本线程的 4 个全局列
int row  = blockIdx.y*BS + ty;     // 全局行

int a_off = row*K + bk + tx*V;
float4 av = *reinterpret_cast<const float4*>(&A[a_off]);  // LDG.E.128
As4[ty][tx] = av;                                          // STS.128
```

- block=(8,32)：256 个线程正好覆盖 32×32 tile 的 256 个 float4。
- 每线程一次搬 16B，而不是 4B：STS 指令数 / 4。
- 计算阶段每线程维护 4 个累加器（acc.x/y/z/w），一次 LDS.128 取 B 的 4 个元素。

### 4.3 D 档只改一行

```cuda
__shared__ float4 As4[BS][COLS + 1];   // 行宽 8 -> 9
__shared__ float4 Bs4[BS][COLS + 1];
```

为什么 9：warp 中 4 行 × 8 列的线程访问 `Bs4[k][tx]` 时，行宽为 8 会让 4 行的
128-bit 访问按 `ty*8` 分布，在 32 bank 上产生冲突；行宽 9 与 32 互质，各行偏移
错开，NCU 实测 ld conflict 从 524288 → 0。
### 4.4 其余四路径的最小代码与解释

**PyTorch（黄金参考）**
```python
ref = (a.double() @ b.double()).float()   # fp64 参考
out = a @ b                               # PyTorch 库基线
```
逐行：参考先在 fp64 下算真值再转 fp32；`a@b` 是框架库实现，供所有手写路径对照。

**Triton（官方 tutorial 03 的 tiled kernel，本 Ticket 只换 block 配置）**
```python
gemm_tiled_kernel[grid](..., BM=BM, BN=BN, BK=BK)  # 32x32x32 / 64x64x32 对照
```
逐行：Triton 编译器负责 shared 布局、向量化和 bank；我们只调 block 参数。
实测：512³ 32×32×32=5.62 TF；1024³ 64×64×32=6.15 TF。

**cuTile（官方 MatMul.py 同步实现，编译器替你做 shared 优化）**
```python
a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
acc = ct.mma(a, b, acc)
```
逐行：cuTile 没有暴露 bank 控制 API；`ct.load/ct.mma` 内部自动决定 shared 布局与向量化。
本 Ticket 用 tile=16×16×16 与 32×32×16 做能力对照，全部 CORRECT_PASS。
**这不是手写 bank 优化，而是官方能力上限**（见 `docs/CONCEPTS.md` 的“同步对照”）。

**CuTe DSL（官方 03_gemm_tiled_smem + padding）**
```python
a_smem = cutlass.Array(cutlass.Float32, (TS, TS + 1), space=cutlass.AddressSpace.smem)
...
prims.barrier_cta_sync(0)
```
逐行：在官方 smem tiled 教程上把 tile 行宽从 TS 改成 TS+1（padding），其余不动。
三个 shape 全部 CORRECT_PASS；它对应 CUDA 的“显式 padding”一档。


## 5. 核心知识点要点

### 5.1 bank 与 conflict 的精确模型

- shared memory = 32 个 bank × 4B；同一 bank 同一周期只能服务一个地址。
- 同一 warp 内：不同地址落同一 bank → conflict（串行，倍数降速）；同地址 → broadcast；
  32 个地址覆盖 32 bank → 无冲突。
- **128-bit（float4）访问会让模型变复杂**：一条 LDS.128 由硬件拆成 phase，
  冲突与否取决于 warp 的行/列布局，不能只凭 stride 心算，必须用 NCU 测。

### 5.2 NCU 看 conflict 的原生命令与指标

```bash
ncu -k gemmVec4 --launch-skip 26 --launch-count 1 \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,gpu__time_duration.sum \
  ./src/t06_gemm_smem/cuda/gemm_smem
```

- `..._op_ld.sum`：load 方向的 bank conflict 次数；`..._op_st.sum`：store 方向。
- 0 = 无冲突；524288 = 每次 tile 迭代的冲突次数累积。对比 C/D 两档就是
  “一个 padding 消除 52 万次冲突”的直接证据。

### 5.3 为什么顺序是“先向量化、再 padding”

- 先消冲突而不向量化（B 档）：搬 tile 还是标量，指令多、块小，冲突不是主要矛盾，
  甚至会因 padding 引入 store conflict 而变慢。
- 先向量化（C 档）：把主要矛盾（指令/搬运开销）解决，性能 2–9 倍提升；
  此时 ld conflict 才成为剩余瓶颈，再 padding 才值得。
- **优化必须按 NCU 数据排序，而不是按优化名词的顺序。**

### 5.4 block 大小与 shape 的关系（Triton 实测）

| shape | 32×32×32 | 64×64×32 |
| --- | --- | --- |
| 512³ | 5.62 TFLOPS（更好） | 2.61 TFLOPS |
| 1024³ | 2.88 TFLOPS | 6.15 TFLOPS（更好） |

同一个 kernel，不同 shape 最优 block 不同。Triton 官方 tutorial 03 的 `@triton.autotune`
配置表就是为自动搜这个；手写 CUDA 则要自己测。**没有永远最优的 block，只有当前
shape 下的最优。**

### 5.5 SASS 证据

`STS.128 / LDS.128 / BAR.SYNC`：128-bit 共享内存读写 + block 同步。
`docs/evidence/T06/t06-cuda-sass.txt`。

### 5.6 用地址算术把四档的 conflict 逐档算清（本节的“核心中的核心”）

记 `bank = (word_address) % 32`（每个 word=4B）。A/B/C/D 四个变体的区别全部来自
“行宽 stride”与“128-bit 是否参与”的组合：

| 档 | As 行宽 | Bs 行宽 | 访问方式 | 冲突来源 | NCU 实测 |
| --- | --- | --- | --- | --- | --- |
| A 基线 | 16 | 16 | 标量 LDS/STS | 一个 warp=ty0+ty1 两行，`As[ty][k]` 地址差 16 bank，恰无冲突 | ld=0, st=0 |
| B pad | 17 | 17 | 标量 LDS/STS | store `Bs[ty][tx]`：ty1 行地址 `17+tx` 在 tx≥15 后绕回 bank0..1，与 ty0 行撞 | ld=0, **st=524288** |
| C float4 | 8（float4） | 8（float4） | LDS.128/STS.128 | load `Bs4[k][tx]`：warp 4 行 × 8 列，行宽 8 使 4 行的 128-bit 访问按 `ty*8` 分布，产生 phase 冲突 | **ld=524288**, st=0 |
| D float4+pad | 9（float4） | 9（float4） | LDS.128/STS.128 | 行宽 9 与 32 互质，各行错开 | **ld=0, st=0** |

这个表的价值：**不要背“padding 一定能消 conflict”**。A 没有 conflict 是因为 warp
恰好只有两行；B 加 padding 反而改变了 store 的 bank 分布引入冲突；C 的冲突来自
128-bit 访问的 phase；D 用“互质行宽”才真正归零。**结论必须来自 NCU 指标，不来自直觉。**

### 5.7 优化方法论：为什么本轮要做 A/B/C/D 四档

- **一次只改一个变量**：A→B 只改 stride；A→C 只改搬移宽度和 block；C→D 只改 stride。
- 这样性能变化可以归因；如果一次同时改 padding+float4+block，就无法解释是哪个动作有效。
- 顺序由 NCU 数据决定：A 的 conflict=0 说明先别碰 conflict；A 的 L1/指令瓶颈说明先改
  搬移宽度（C）；C 出现 ld conflict 后再补 padding（D）。
- 这条方法论会在 T07（cp.async）、T08（tensor core）继续使用。

### 5.8 T07 前置

float4+pad 已把“搬 tile”的指令宽度拉满；下一步是让 global→shared 的搬运用
`cp.async` 与计算**重叠**（double buffer/pipeline），这就是 T07 的唯一学习变量。

## 6. 性能分析

见 §3 表。D 档 512³ 0.084 ms ≈ 3.2 TFLOPS（cuBLAS 7.7 TFLOPS 的 42%），1024³
0.575 ms ≈ 3.7 TFLOPS。事件计时受 L2/频率波动影响，**结论以 NCU 时长与冲突计数为准**。

## 7. Memory/Compute/Latency-Bound 判断

- D 档 NCU(512)：Duration 88.29 us，conflict=0。相比 C 档，瓶颈从“shared load 冲突
  引发的 LSU 排队”进一步移向计算/剩余 stall；仍未达 Compute-Bound（离 FP32 峰值远）。
- A/B 档：barrier/指令开销主导，latency-bound 色彩重。
- 判定依据与方法见 `docs/CONCEPTS.md` §2。

## 8. 知识点完整性检查

已覆盖：bank 模型、128-bit 冲突的复杂性、NCU conflict 指标、padding 的代价与收益、
向量化优先原则、block/shape 关系、SASS STS.128/LDS.128。
后置：cp.async/double buffer（T07）、Tensor Core（T08）。

## 9. 过关问题及答案（19 题，一问一答）

**A 基础**

**Q1.** shared memory 有多少 bank、每 bank 多宽？同一 warp 访问同 bank 不同地址/同地址分别叫什么？

**A1（回答）**：32 bank × 4B。不同地址落同 bank = conflict（串行）；同地址 = broadcast（无冲突）。

**Q2.** `STS.128/LDS.128` 是什么？和 `STS/LDS` 比，一次搬多少字节？

**A2（回答）**：128-bit 的 shared store/load：一次 16 字节；标量版一次 4 字节。指令数减少 4 倍。

**Q3.** D 档把 COLS 从 8 改成 9 的目的是什么？为什么 9 有效？

**A3（回答）**：改变行 stride：8 与 warp 的 4×8 访问模式产生 ld conflict；9 与 32 互质，各行偏移
   错开，NCU 实测 ld conflict 524288→0。

**Q4.** 用一句话解释 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 这个指标。

**A4（回答）**：本次 kernel 在 shared load 管道上发生的 bank conflict 总次数；0=无，数字越大越糟。

**B 理解**

**Q5.** A 档 NCU 显示 conflict=0，这验证了 T05 的哪个判断？A 慢的根因又是什么？

**A5（回答）**：验证了 T05 “退化主因不是 bank conflict，而是 block 太小、每线程只搬 4B、barrier
   太多”的预判：NCU 实测 A 档 ld/st conflict 都为 0。A 慢的根因是指令/搬移效率低，
   不是 bank conflict。

**Q6.** B 档 padding 为什么在 512 反而更慢？NCU 证据是什么？

**A6（回答）**：padding 改变了 store 的行 stride，反而让 `Bs[ty][tx]` 的 store 产生 524288 次
   conflict；512 的冲突代价大于收益，所以更慢。证据：`t06-gemmPad.txt`。

**Q7.** 为什么“先消 conflict 再向量化”在本 Ticket 被证明是错的顺序？

**A7（回答）**：因为主要瓶颈不是 conflict（A 档 conflict=0）；先向量化把主要矛盾解决后，conflict
   才成为剩余瓶颈，此时 padding 才有净收益。

**Q8.** C 档比 A 档快 2–9 倍的根本原因是什么？

**A8（回答）**：每线程一次搬 16B，STS/地址计算指令减少；同时 BS=32 块更大、barrier 次数更少，
   把 SM 的指令预算还给了 FFMA。

**Q9.** Triton 为什么不需要你手写 padding？它怎么处理 bank？

**A9（回答）**：Triton 编译器负责 shared 布局、向量化和 padding/swizzle；你只给 block 参数。官方
   tutorial 03 的 autotune 会为不同 shape 选择不同配置。

**C 应用**

**Q10.** 512³ 与 1024³ 的最优 Triton block 配置分别是什么？这说明什么原则？

**A10（回答）**：512³ 用 32×32×32；1024³ 用 64×64×32。说明没有永远最优的 block，只有当前 shape 下
    实测最优；Triton autotune 就是自动做这件事。

**Q11.** 如果 NCU 显示 ld conflict 很大，但换 padding 后性能反而下降，下一步你会先查什么？

**A11（回答）**：先确认 conflict 是否真的是主要瓶颈：看 LSU 吞吐、no-eligible、总 Duration；
    如果 padding 没收益，瓶颈可能仍是 barrier/指令/global 搬移，应回到 NCU 排序。

**Q12.** 用 float4 搬 tile 时，为什么非 4 对齐的 K/N 需要标量 fallback？

**A12（回答）**：`float4` 要求 16B 对齐；K/N 不是 4 的倍数时地址可能错位，直接 reinterpret 会
    非法访问/崩溃，所以需要按元素标量加载。

**Q13.** 把 BS 从 32 改成 64，shared 容量和 block 线程数会怎么变？要改哪些索引？

**A13（回答）**：shared 容量 ×4（每 tile 行数 ×2、列数 ×2）；block 线程数要相应改（如 16×16 或
    8×64），grid 和所有 `tx/ty` 的映射同步改，并重新测 bank/occupancy。

**Q14.** D 档 512³ 0.084 ms 约合多少 TFLOPS？离 cuBLAS 还差多少？

**A14（回答）**：2×512³/0.084ms ≈ 3.2 TFLOPS；cuBLAS 约 7.7 TFLOPS，还差约 2.4 倍（T07/T08 继续）。

**Q15.** T07 要解决的剩余瓶颈是什么？用一句话描述它的机制。

**A15（回答）**：global→shared 的搬运与计算串行，SM 在等搬运；T07 用 cp.async + double buffer 让
    搬运和计算重叠，隐藏访存延迟。

**Q16.** 用 bank 地址算术解释：A 档为什么 conflict=0，B 档为什么 store conflict=524288，
    D 档为什么又能归零？

**A16（回答）**：bank = word 地址 %32。A 档 warp 只有 ty0/ty1 两行，`As[ty][k]` 两行地址差 16 个
    bank，互不重叠，所以 ld/st 都 0。B 档行宽 17，store `Bs[ty][tx]` 在 ty1 行地址
    17+tx，tx≥15 时绕回 bank0..1 与 ty0 行冲突 → st=524288。D 档 float4 行宽 9 与 32
    互质，各行访问错开，冲突归零。**核心教训：padding 不必然消 conflict，要看 warp
    布局和 128-bit phase，结论以 NCU 为准。**

**Q17.** 为什么 T06 要做 A/B/C/D 四档而不是直接写“最终版”？这条方法论叫什么？

**A17（回答）**：因为要“一次只改一个变量”地归因：A→B 只改 stride，A→C 只改搬移宽度/block，
    C→D 只改 stride。直接写最终版会混淆每个动作的真实收益。这就是基于 NCU 数据的
    增量优化方法论，T07/T08 继续沿用。

**Q18.** cuTile 在 T06 里是否“消除了 bank conflict”？如果不是，它承担什么角色？

**A18（回答）**：不是。cuTile 没有暴露 bank 控制 API，它承担“官方同步实现的能力对照”：证明
    `ct.load/ct.mma` 由编译器自动做 shared 布局/向量化，并用 tile=16/32 正确性对照；
    bank 手写优化只能在 CUDA/CuTe 中显式演示。

**Q19.** CuTe DSL 的 padding 版与 CUDA D 档的 padding 思路有什么异同？

**A19（回答）**：相同：都通过把行宽改成与 bank 数互质来避免冲突。不同：CUDA D 档在 float4 布局上
    把 COLS 8→9；CuTe 在标量 tile 上把 (TS,TS) 改成 (TS,TS+1)；两者都需要 NCU 指标验证。
## 10. 本轮停止点

完成：CUDA 四档、Triton 配置对比、CuTe padding 版、正确性、NCU conflict 指标、
SASS/NSYS 证据、讲义 15 题。
未做：T07 cp.async 流水线。

## 11. 下一最小增量

T07 GEMM 异步拷贝与流水线：cp.async / double buffer，让搬 tile 与计算重叠。

## 附录：可复现命令

```bash
bash scripts/run_t06_all.sh
nvcc -O3 -arch=sm_89 -o src/t06_gemm_smem/cuda/gemm_smem src/t06_gemm_smem/cuda/gemm_smem.cu
./src/t06_gemm_smem/cuda/gemm_smem
ncu -k gemmVecPad --launch-skip 26 --launch-count 1 --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,gpu__time_duration.sum \
  ./src/t06_gemm_smem/cuda/gemm_smem
```
