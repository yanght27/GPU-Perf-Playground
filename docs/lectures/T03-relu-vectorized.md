# T03 ReLU 向量化版（唯一主讲义）

- Ticket：T03
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**合并访问（coalesced access）与 128-bit 向量化 load/store**
- 环境：gpp-core / 系统 nvcc / gpp-cutile / gpp-cute
- 官方来源：S01c、S02c、S03c、S10c、S10d（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——把标量 ReLU 改成 128-bit 向量化版本，理解合并访问与带宽的关系；**依次学到**——①warp 与 128B 事务；②float4/切片与 16B 对齐；③尾部处理；④各工具谁负责向量化；⑤NaN 语义；**学完应能回答**——为什么指令少 4 倍，Memory-Bound 算子却不必然提速？CuTe 为什么从 26us 追到 18.75us？；**相关工具/技术**——CUDA float4、Triton PTX、cuTile ct.load、CuTe DSL 切片、NCU/SASS。
- 本节内容：**要解决的问题**——T02 的 CuTe 标量版只有 63% DRAM 利用率、26us，说明没合并好的访存很吃亏；**核心手段**——128-bit 向量化：一条指令搬 16B；连续访问才能合并成 128B 事务；**怎么实现**——CUDA float4 主循环+标量尾巴；CuTe 用官方切片 load；其余路径用 PTX/NCU 证明已自动向量化；**怎么验证**——五路径含尾部/±Inf/NaN 全 PASS；NCU CuTe 63%→88%、26→18.75us；SASS 见 LDG.E.128；**最终结论**——向量化减少指令、救回低效实现，但不减少 DRAM 字节数。

## 1. 上一轮问题回答

T02 已验收。本轮按“验收前覆盖完整性自查”执行：T03 的每个概念（向量化、合并访问、
事务、对齐、尾部处理）都会讲清楚，并为 T04 GEMM 的“二维索引与访存/计算比”留前置。

## 2. 规范实现与官方来源

| 路径 | 依据 | 版本 |
| --- | --- | --- |
| PyTorch | `F.relu` 自动调度 `vectorized_elementwise_kernel<4>` | 2.13 |
| CUDA | Programming Guide「Maximize Memory Throughput」+ 官方 `float4`（vector_types.h） | CUDA 13.0 |
| Triton | PTX 证据 `ld.global.v4.b32 / st.global.v4.b32` | v3.7.1 |
| cuTile | 官方 `ct.load/ct.store` tile 级自动向量化 | `29444e0c` |
| CuTe DSL | 官方 `07_vectorized_array.py` 切片语法 | `564d267e` |

## 3. 本轮实现结果

- 正确性：五路径在 N=2^20、N=1,000,003（尾部 3 个元素）以及**极值 `±Inf/NaN/±1e38`**
  全部 `CORRECT_PASS`，max_abs_err=0。
- 一键复现：`bash scripts/run_t03_all.sh`。

### NCU kernel 时间（T02 标量 vs T03 向量化）

| 路径 | T02 标量 Duration | T03 向量化 Duration | T02 DRAM% | T03 DRAM% | 结论 |
| --- | --- | --- | --- | --- | --- |
| PyTorch | 18.82 us | 18.82 us（框架本就向量化） | 87.68 | 87.68 | 无变化 |
| CUDA | 18.56 us | 20.74–27.17 us（多次测量，见注） | 88.64 | 90.58–93.16 | 时间无显著变化；指令数明显减少 |
| Triton | 18.88 us | 18.82 us（编译器本就向量化） | 87.14 | 87.58 | 无变化 |
| cuTile | 18.82 us | 17.86–24.22 us（波动同量级） | 88.47 | 87.90–92.54 | 无显著变化 |
| CuTe DSL | 26.02 us | **18.75 us** | 63.18 | 87.89 | **明显变快，追平第一梯队** |

> 注：NCU 单次 Duration 有 ±2–6 us 波动（L2 状态、profiler 重放），所以 T03 不下
> “CUDA float4 一定更快”的结论；能下结论的是 SASS 指令变化（LDG.E→LDG.E.128）与
> CuTe 从 63%→88% 的 DRAM 利用率。**这是本 Ticket 最重要的证据素养。**

## 4. 核心代码与逐行解释

### 4.1 PyTorch（参考，并观察框架自动向量化）

```python
out = F.relu(a)
```

NCU kernel 名里明确写 `vectorized_elementwise_kernel<4, ...>`：`<4>` 表示每线程处理
4 个元素。**你什么都没写，但框架已经向量化。**

### 4.2 CUDA float4（本轮手写向量的主角）

```cuda
__global__ void reluVecKernel(const float4 *in, float4 *out, int nVec) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // 索引单位：一个 float4
    if (i < nVec) {
        float4 v = in[i];            // 一次读 16 字节：LDG.E.128
        v.x = fmaxf(v.x, 0.0f);      // 4 个 lane 各自 ReLU
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
        out[i] = v;                  // 一次写 16 字节：STG.E.128
    }
}
__global__ void reluTailKernel(const float *in, float *out, int start, int n) {
    int i = start + blockDim.x * blockIdx.x + threadIdx.x;   // 剩余 <4 个元素
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}
```

- `float4` 是 CUDA 官方内置向量类型（`vector_types.h`），就是 4 个连续 float。
- 索引公式没变，但 `i` 现在是“第 i 个 float4”；因此 grid 只需要 N/4/block 个线程。
- `fmaxf` 对 4 个分量各做一次 ReLU。
- 尾部处理：N=1,000,003 时，`nVec = 250,000`，还剩 3 个元素；主 kernel 处理前 1,000,000
  个，`reluTailKernel` 用标量补最后 3 个。**向量化不是“放弃边界”，而是“主循环向量化 +
  尾巴标量化”。**

### 4.3 Triton（不换写法，用 PTX 证明自动向量化）

Triton kernel 与 T02 完全相同（`tl.load/tl.store` + mask）。官方编译产物里直接看到：

```text
@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];
@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r9, %r10, %r11, %r12 };
```

`v4.b32` = 128-bit 向量访存。**Triton 的 block 抽象让编译器天然选择向量化。**

### 4.4 cuTile（tile load/store 自动向量化）

与 T02 相同的 `ct.load/ct.store`。官方 `ct.load` 是 tile 级操作：一次搬 256 个连续
float，编译器自动选择 128-bit 访存；T02 的 NCU（88.47%）已证明它与 CUDA float4 同梯队。

### 4.5 CuTe DSL（官方切片语法：T02 标量 → T03 向量化 load）

```python
@cute.kernel
def relu_vec_kernel(a_arr, c_arr, n_elements, vector_size):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = (bx * bdx + tx) * vector_size
    v = a_arr[idx:vector_size]        # 官方切片：一次向量化 load（16B）
    for i in range(vector_size):
        if idx + i < n_elements:
            val = v[i]
            c_arr[idx + i] = val if val > 0.0 else 0.0
```

- T02 是每元素一次 4B load；T03 改成官方 `a_arr[idx:vector_size]` 一次取 4 个元素。
- 为什么 store 还是标量？本 Ticket 实测：官方 CuTe DSL 当前对“向量 max/where”尚不支持
  （`cute.where` 要求 TensorSSA、`cutlass.max` 不支持 vector 类型），所以只把官方支持的
  向量化 load 用上。结果：Duration 26.02→18.75 us、DRAM 63→88%，说明 **load 侧向量化
  已经补上了主要短板**。这是官方 API 能力边界，讲义如实记录，不假装实现了不存在的 API。

## 5. 核心知识点要点（T03 全部讲透）

### 5.1 什么是向量化 load/store

普通标量：一条指令搬 4 字节（`LDG.E`）。向量化：一条指令搬 16 字节
（`LDG.E.128` / PTX `ld.global.v4.b32`）。同样的数据量，**指令条数减少 4 倍**，
地址计算、发射、解码的开销也减少；硬件一次发出更大的访存请求。

### 5.2 合并访问（coalesced access）：事务是怎么合并的

- warp = 32 线程一起发访存请求；硬件以 **32B sector / 128B cache line** 为单位服务。
- 线程 i 访问 `base + i*4`（连续 4B）时，32 个线程恰好覆盖 128B → 1 次 128B 事务，
  带宽利用率 100%。
- 线程 i 访问 `base + i*stride*4`（大 stride）时，每次 128B 事务只用其中 4B →
  有效带宽只剩 1/32。这就是“合并访问”这个词的含义：**相邻线程访问相邻地址，请求才能
  合并**。
- 向量化 float4 后：32 线程 × 16B = 512B → 4 次 128B 事务，仍然全部合并，但指令少 4 倍。

### 5.3 对齐（alignment）

- `float4` 要求 16 字节对齐；`cudaMalloc` 返回 256B 对齐的地址，所以直接把 `float*`
  强转成 `float4*` 是安全的。
- 如果你的数组起点不是 16B 对齐，128-bit 访存会非法。这是“向量化不是无条件安全”的
  第一课。

### 5.4 尾部处理

- N=1,000,003：`nVec=N/4=250,000`，向量主循环覆盖 1,000,000，剩下 3 个元素用标量
  kernel 补。通用公式：`nVec = N / vector_width; tail = N - nVec*vector_width`。
- Triton/cuTile 的 mask/check_bounds 自动处理尾块；CUDA/CuTe 手动处理。同一问题的
  四种形态（呼应 T02 §5.3 的边界知识）。

### 5.5 为什么“指令少 4 倍”，时间却没有明显变快？

因为 ReLU 是 Memory-Bound：瓶颈是 DRAM 带宽，不是指令发射。标量版的连续访问本来就已经
100% 合并，带宽已经吃满；向量化减少的是**指令开销**，在带宽瓶颈下时间基本不变。
它能带来的收益是：
1. 提高复杂 kernel 的 ILP 和地址计算效率（T07 会看到）；
2. 把本来没合并好的实现救回来（本 Ticket 的 CuTe：63%→88%）；
3. 为 Tensor Core/更宽数据类型铺路。
**结论要诚实：向量化不是万能加速器，它主要“更省指令、更充分用带宽”。**

### 5.6 如何用工具证明“它真的向量化了”

- SASS：标量是 `LDG.E`/`STG.E`；向量化是 `LDG.E.128`/`STG.E.128`
  （证据 `docs/evidence/T03/t03-cuda-sass.txt`）。
- Triton PTX：`ld.global.v4.b32` / `st.global.v4.b32`。
- NCU：看 DRAM% 与 Duration 的联动；CuTe 63%→88% 就是向量化生效的直接证据。
- 不要只信“感觉变快了”，要像本轮这样三件套互证。

### 5.7 极值语义：ReLU(NaN) 应该是 NaN

- 黄金参考 PyTorch `F.relu` 对 NaN 返回 NaN、`+Inf→+Inf`、`-Inf→0`。
- 但 CUDA 的 `fmaxf(NaN,0)` 是 IEEE maxNum 语义（返回非 NaN 操作数），会错误地把 NaN
  变成 0；C 三元表达式 `x>0 ? x : 0` 对 NaN 同样错误。
- 正确写法：先 `isnan(x) ? x : max(x,0)`；Triton `tl.where(x != x, x, ...)`；
  cuTile `ct.where(ct.isnan(a_tile), a_tile, ...)`；CuTe `val if val != val else ...`。
- 本轮五路径极值测试全部通过，证据在各自脚本的 `[xxx_relu_extreme]` 输出。
  这再次说明：**只测普通数值不够，极值/NaN 是算子正确性的一部分。**

### 5.8 各工具向量化的“主动权”对比

| 路径 | 谁来向量化 | 你需要做什么 | 风险 |
| --- | --- | --- | --- |
| PyTorch | 框架 | 什么都不做 | 不可控但已优化 |
| CUDA | 你 | 手写 float4 + 尾部 | 对齐/边界错误 |
| Triton | 编译器 | 写连续 block load | 一般自动 |
| cuTile | 编译器 | tile load/store | 一般自动 |
| CuTe DSL | 你+DSL | 切片语法；受官方 API 支持范围限制 | 能力边界 |

### 5.9 T04 前置：从这里走向 GEMM

T03 完成了“一维元素级算子”的完整训练：索引、边界、分支、访存、向量化。
T04 朴素 GEMM 会把索引从一维升级到**二维（行、列、K 维）**，并第一次出现
“每个输出要读很多输入”的**数据复用**问题——到时你会用本轮的合并访问知识判断
朴素 GEMM 的访存模式差在哪里。

## 6. 性能分析（实测）

调用级（仅看 launch 开销，不看带宽）：PyTorch 0.0204 ms；CUDA 0.0141 ms；
Triton 0.3078–0.4831 ms（本次 do_bench 波动较大，内核级以 NCU 为准）；
cuTile 0.0677 ms；CuTe Python call ≈21.4 ms（DSL launch 开销）。

NCU 内核级是本轮结论依据（见 §3 表）。L2 口径提醒沿用 T02 §5.9：调用级算带宽没有意义。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

- **Memory-Bound**：五路径 DRAM% 88–93%，Compute% 3–15%。
- 向量化没有改变 Bound 类型，只改善了带宽利用效率（CuTe 63→88%）与指令效率。
- Latency-Bound：无依赖链、无显著 stall 证据，不下结论。

## 8. 知识点完整性检查

本轮覆盖：向量化 load/store、合并访问与事务、16B 对齐、尾部处理、SASS/PTX 向量化证据、
“向量化在 Memory-Bound 下为何不必然提速”、五工具向量化主动权对比。
下一 Ticket 前置：二维索引与数据复用动机已在 §5.8 埋点。
明确后置：Shared Memory（T05）、Bank Conflict（T06）、Tensor Core（T08）。

## 9. 过关问题及答案（13 题，一问一答）

**A 基础**

**Q1.** 什么是 128-bit 向量化访存？它把几条 4B 访问合并成一条什么指令？

**A1（回答）**：一条指令读写 16 字节（4 个 float32），即 `LDG.E.128/STG.E.128`（PTX `v4.b32`）。
   它把 4 条 4B 访问合并成 1 条 128-bit 指令。

**Q2.** 用 warp 和 cache line 解释“合并访问”：32 个线程连续读 float 是几次 128B 事务？
   每线程 float4 呢？

**A2（回答）**：标量连续：32×4B=128B → 约 1 次 128B 事务；float4：32×16B=512B → 4 次 128B 事务，
   全部合并且指令数减少 4 倍。

**Q3.** 为什么 `float4` 指针要求 16 字节对齐？`cudaMalloc` 为什么天然满足？

**A3（回答）**：float4=16B，硬件要求 128-bit 访问落在 16B 边界上；cudaMalloc 返回至少 256B 对齐
   的地址，所以把 float* 强转 float4* 安全。

**Q4.** N=1,000,003 时，float4 主循环覆盖多少个元素？剩余几个？怎么处理？

**A4（回答）**：nVec=250,000 覆盖 1,000,000 个；剩 3 个；用标量 tail kernel 处理（start=1,000,000）。

**B 理解**

**Q5.** SASS 里 `LDG.E` 和 `LDG.E.128` 的区别是什么？在哪个证据文件能看到两者并存？

**A5（回答）**：`LDG.E`=32-bit 标量读，`LDG.E.128`=128-bit 向量读。两者在
   `docs/evidence/T03/t03-cuda-sass.txt` 并存：主循环用 .128，尾 kernel 用标量。

**Q6.** 为什么 CUDA float4 的指令少了 4 倍，ReLU 时间却没有明显变快？

**A6（回答）**：ReLU 是 Memory-Bound，标量连续访问已经 100% 合并、带宽已满；减少指令开销不改变
   带宽瓶颈。向量化收益体现在省指令、救回低效实现（CuTe）、为复杂 kernel 铺路。

**Q7.** CuTe 从 26.02 us 降到 18.75 us 的本质原因是什么？

**A7（回答）**：本质是 load 侧从“每元素 4B 标量读”变成官方切片 128-bit 向量读，DRAM 利用率 63%→88%。

**Q8.** Triton 源码没写 float4，为什么 PTX 里出现 `ld.global.v4.b32`？

**A8（回答）**：Triton 编译器把连续的 block 级 `tl.load/tl.store` 自动优化成 128-bit 访存；这是
   Triton 编程模型的价值，PTX 是编译产物证据。

**C 应用**

**Q9.** 如果把 vector width 从 4 换成 2（float2），SASS 指令会变成什么？尾部逻辑怎么改？

**A9（回答）**：float2 → `LDG.E.64/STG.E.64`；nVec=N/2，尾部 <2 个元素用标量补。

**Q10.** 一个 stride=2 的访问（只处理偶数下标）还能 100% 合并吗？为什么？

**A10（回答）**：不能。每 4B 有用数据之间隔着 4B 空洞，每个 128B 事务只利用一半，有效带宽约减半。

**Q11.** 为什么调用级 benchmark 显示 400+ GB/s 也不能用来证明“显存带宽提高了”？

**A11（回答）**：因为反复跑同一批数据时它们全部命中 32MB L2，调用级时间量的是 L2 速度；显存带宽
    判定只看 NCU DRAM Throughput。

**Q12.** 用一句话分别说明五条路径里“谁负责向量化”。

**A12（回答）**： PyTorch 是框架：只写 `F.relu`，向量化是框架内部实现；CUDA 是你手写 float4；Triton/cuTile 是编译器替你生成向量访问；CuTe DSL 是你用官方切片语法表达向量读，编译器生成。无论哪条路径，证据都要回到 SASS 的 `LDG.E.128/STG.E.128`。

**Q13.** 为什么 `fmaxf(x, 0)` 不能直接实现与 PyTorch 一致的 ReLU？正确写法是什么？

**A13（回答）**：因为 `fmaxf(NaN, 0)` 是 IEEE maxNum 语义，会返回非 NaN 的 0；PyTorch `F.relu(NaN)`
    返回 NaN。正确写法是 `isnan(x) ? x : fmaxf(x, 0)`（Triton `x != x`、cuTile
    `ct.isnan`、CuTe `val != val` 同构）。
## 10. 本轮停止点

完成：五路径向量化（或证明已自动向量化）、正确性含尾部、SASS/PTX/NCU/NSYS 证据、
讲义+12 题、提交 `a386730`。
未做：T04 朴素 GEMM。

## 11. 下一最小增量

T04 朴素 GEMM + cuBLAS 基线：索引从一维升二维，第一次学习“访存/计算比”与库基线，
为 T05 Tiling 铺路。

## 附录：可复现命令

```bash
bash scripts/run_t03_all.sh
nvcc -O3 -arch=sm_89 -o src/t03_relu/cuda/relu_vec src/t03_relu/cuda/relu_vec.cu
./src/t03_relu/cuda/relu_vec
ncu --set basic -k reluVecKernel -o docs/evidence/T03/t03-cuda-ncu ./src/t03_relu/cuda/relu_vec
cuobjdump -sass src/t03_relu/cuda/relu_vec
```
