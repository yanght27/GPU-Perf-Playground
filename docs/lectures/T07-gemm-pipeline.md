# T07 GEMM 异步拷贝与流水线（唯一主讲义）

- Ticket：T07
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**Double Buffer / cp.async / Pipelining 隐藏访存延迟**
- 环境：gpp-core（PyTorch/Triton） / 系统 nvcc（CUDA） / gpp-cutile（cuTile latency hint） / gpp-cute（CuTe cp.async 双缓冲）
- 官方来源：S01g、S02g、S03g、S11a（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`
- 本节导读：**一句话目标**——让“global→shared 搬 tile”和“FFMA 计算”重叠执行，理解流水线如何隐藏访存延迟；**依次学到**——①为什么 T06 还慢；②cp.async 是什么、和普通 load 有何不同；③double buffer 怎么实现；④wait_group/stage 的语义；⑤为什么本卡本配置下 CUDA 流水线没赢、Triton num_stages 却赢了；**学完应能回答**——cp.async 与同步 load 的区别？double buffer 为什么要两组 shared？本 Ticket 最重要的诚实结论是什么？；**相关工具/技术**——PyTorch、CUDA cp.async/cuda_pipeline_primitives、Triton num_stages、cuTile latency hint、CuTe DSL cp.async 双缓冲、NCU/SASS。
- 本节内容：**要解决的问题**——T06 的 float4+pad 仍要“先等 tile 搬完，再开始算”，SM 在等 global 延迟；**核心手段**——①cp.async：global→shared 的异步拷贝，不经过寄存器、不阻塞计算；②double buffer：两组 shared，搬下一块的同时算当前块；**怎么实现**——CUDA `__pipeline_memcpy_async/__pipeline_commit/__pipeline_wait_prior`；Triton `num_stages`；cuTile 官方 `latency` hint；CuTe 官方 `prims.cp_async_shared_global` 双缓冲 GEMM；**怎么验证**——正确性全 PASS；NCU Duration 对比 T06；SASS 看 `LDGSTS.E.BYPASS.128`；Triton num_stages=1/2/3 实测；**最终结论**——流水线是正确技术，但“有没有收益”取决于 shape/配置；本卡本 tile 下 CUDA 2-stage 没有超过 T06，Triton num_stages 提高 10–12%，**技术要学到手，收益要用数据说话**。

## 1. 上一轮问题回答

T06 已验收。T06 的结论停在“float4+pad 最快但仍有 stall”。T07 尝试用流水线把
“搬下一块”藏到“算当前块”的阴影里。

## 2. 规范实现与官方来源

| 路径 | 依据 |
| --- | --- |
| CUDA | Programming Guide「Writing Tile Kernels」的 cp.async + `cuda_pipeline_primitives.h` |
| Triton | 官方 tutorial 03 的 `num_stages`（autotune 使用 3–5 stages） |
| CuTe DSL | 官方 `cp_async_shared_global.py` 原语，本 Ticket 实现 STAGES=2 双缓冲 GEMM |
| cuTile | 官方 `ct.load(..., latency=1..10)` 流水线提示，latency=1/2/4 实测 |

## 3. 本轮实现结果

正确性：五路径全部 `CORRECT_PASS`——PyTorch 参考、CUDA cp.async 2-stage、Triton num_stages 1/2/3、cuTile 官方 `latency` hint 1/2/4、CuTe DSL 完整 cp.async 双缓冲 GEMM。

### NCU Duration（512 / 1024）

| 实现 | 512³ | 1024³ |
| --- | --- | --- |
| T06 vecPad（无流水线） | 88.29 us | 604.13 us |
| T07 CUDA cp.async 2-stage | 88.64 us | 615.10 us |
| T07 cuTile latency=4（事件） | 0.563 ms | 见脚本（正确性优先，不比较 kernel 级） |
| T07 Triton num_stages=1 | 5.30 TFLOPS | 7.48 TFLOPS |
| T07 Triton num_stages=2 | 5.55 TFLOPS | 8.20 TFLOPS |
| T07 Triton num_stages=3 | **5.93 TFLOPS** | **8.24 TFLOPS** |

**诚实结论**：CUDA 手写 2-stage 在本卡、本 tile 配置下**没有**超过 T06 的 float4+pad
（512 打平、1024 略慢）；Triton 的 `num_stages` 有稳定收益（512 +12%、1024 +10%）。
技术学会了，但“有没有收益”必须实测——这是本 Ticket 最重要的学习点。

## 4. 核心代码与逐行解释

### 4.1 三个原语（CUDA）

```cuda
#include <cuda_pipeline_primitives.h>
__pipeline_memcpy_async(smem, gmem, 16);  // 发一条 cp.async（16B）
__pipeline_commit();                       // 把这一批拷贝归入一个“组”
__pipeline_wait_prior(N);                  // 等：最多还有 N 组在飞
```

### 4.2 double buffer 主循环

```cuda
__shared__ float4 As4[STAGES][BS][COLS+1];   // STAGES=2：两个缓冲区
__shared__ float4 Bs4[STAGES][BS][COLS+1];

for (int tile = 0; tile < num_tiles; ++tile) {
    int buf = tile % STAGES;
    if (tile + 1 < num_tiles) {
        issue(tile + 1, (tile + 1) % STAGES);   // 异步搬下一块 → 另一组 buffer
        __pipeline_commit();
    }
    if (tile > 0) __pipeline_wait_prior(STAGES - 1);  // 等当前块就绪
    // 第一块：先搬自己，再等全部完成
    if (tile == 0) { issue(0, 0); __pipeline_commit(); __pipeline_wait_prior(0); }
    __syncthreads();
    // 计算当前 buf 的 tile（FFMA 与下一块搬运重叠）
    ...
    __syncthreads();
}
```

逐行：
- `STAGES=2` 即 double buffer：A/B 各两份 shared。
- 迭代开始先对**下一块**发 cp.async，CPU/GPU 不会等它；随后 `wait_prior(1)` 只保证
  **当前块**已完成（允许下一块仍在飞）。
- 于是计算当前块的 FFMA 时，下一块的 global→shared 在后台进行。
- 两次 `__syncthreads` 与 T05/T06 相同：搬完才算、算完再覆盖。
- SASS 证据：`LDGSTS.E.BYPASS.128`（cp.async.cg 的机器码形态）。

### 4.3 Triton 的流水线只改一个参数

```python
gemm_kernel[grid](..., num_stages=3)   # 1 → 2 → 3 实测
```

Triton 编译器自动做 multi-buffer 软件流水线；`num_stages` 是“预取几块”。
512³：1→3 stages 从 5.30 到 5.93 TFLOPS；1024³：7.48→8.24 TFLOPS。
### 4.4 cuTile 的官方流水线提示（latency）

```python
@ct.kernel
def mm_kernel(A, B, C, tm, tn, tk, latency_hint: ct.Constant[int]):
    ...
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk),
                    padding_mode=ct.PaddingMode.ZERO, latency=latency_hint)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn),
                    padding_mode=ct.PaddingMode.ZERO, latency=latency_hint)
        acc = ct.mma(a, b, acc)
```

逐行：
- `latency=1..10` 是官方 `ct.load` 的流水线提示：允许编译器提前发出后续 tile 的 load；
- 数值越大，预取越激进（也可能多占资源）；
- 本 Ticket 实测 latency=1/2/4 正确性全 PASS，512³ 0.636→0.563 ms（约 -11.5%）。

### 4.5 CuTe DSL 的 cp.async 双缓冲 GEMM

```python
a_smem = As.data_ptr() + nb*TS*(TS+1) + ty*(TS+1) + tx
prims.cp_async_shared_global(a_smem, a.data_ptr() + row*K + bk2 + tx, 4, "ca")
prims.cp_async_commit_group()
prims.cp_async_wait_group(STAGES - 1)
prims.barrier_cta_sync(0)
```

逐行：
- `As` 是 `(STAGES, TS, TS+1)` 的 shared Array：STAGES=2 即双缓冲；
- `cp_async_shared_global(dst_smem, src_gmem, 4, "ca")`：官方 cp.async 原语，
  4B 拷贝用 `ca`（16B 才可用 `cg`）；
- commit/wait 与 CUDA 的 `__pipeline_commit/__pipeline_wait_prior` 语义相同；
- 三个 shape（17×31×33、512³、1024³）全部 CORRECT_PASS。


## 5. 核心知识点要点

### 5.1 为什么需要流水线

T06 的循环是“搬→等→算→搬→等→算”。global 访问有几百周期延迟，SM 在等待时无事可做。
流水线把循环变成“边搬边算”：用多一份 buffer 换取“延迟被计算隐藏”。

### 5.2 cp.async 与普通 load 的区别

- 普通 `LDG`：global→寄存器→shared，占寄存器、阻塞在依赖链上、经过 L1。
- `cp.async`（LDGSTS）：global→shared **直接**异步拷贝，不占寄存器、不阻塞后续指令，
  可绕过 L1（cg 模式）。
- 因此它可以和计算重叠，是 sm_80+ 的官方推荐搬 tile 方式。

### 5.3 stage / wait_group 的精确语义

- 每 `commit_group` 把本线程已发出的 cp.async 打包成一组。
- `wait_prior(N)`：等“除了最近 N 组之外”的全部完成。
- double buffer：`wait_prior(1)` = 当前块完成、下一块允许在飞。
- stage 越多，预取越深，但 shared 用量线性增加（每 stage 一份 tile），occupancy 可能下降。

### 5.4 为什么本卡 CUDA 2-stage 没赢（诚实分析）

- T06 的 float4+pad 已经把指令和 bank 优化做足，且 512/1024 的数据量相对本卡带宽
  足够小，同步 load 的延迟已经部分被多 warp 隐藏；
- cp.async.cg 绕过 L1，而同步 float4 load 能吃到 L1 复用；
- 2-stage 只是“双缓冲”，要更明显需要 3–4 stages + 更大 tile/更多 warp（T08 的
  Tensor Core/CUTLASS 路线会真正需要它）。
- 结论：**在本配置下没有收益 ≠ 技术错误**；Triton 的 3-stage 证明流水线有效。

### 5.5 cuTile/CuTe 路径的记录

- cuTile：官方 `ct.load(..., latency=1..10)` 就是官方流水线提示。`cutile_gemm.py`
  用 latency=1/2/4 实测：512³ 0.636→0.563 ms（约 -11.5%），全部 CORRECT_PASS。
- CuTe DSL：`cute_gemm.py` 用官方 `prims.cp_async_shared_global` 实现完整
  STAGES=2 双缓冲 GEMM；三个 shape 全部 CORRECT_PASS。`cute_gemm.py` 是本 Ticket 唯一 CuTe 路径文件。

### 5.6 T08 前置

cp.async + 多 stage 是 Tensor Core GEMM（CUTLASS）的标准搬运管线；T08 会看到
官方 CUTLASS example 把本 Ticket 的每个原语组合成高性能 kernel。

## 6. 性能分析

见 §3 表。注意：事件计时受频率/L2 波动影响，**结论以 NCU Duration 与 Triton do_bench
为准**。T07 的 CUDA 事件计时在 1024 出现过 6ms 的离群值，NCU Duration 615us 是可信值。

## 7. Memory/Compute/Latency-Bound 判断

- T07 CUDA pipe 512：DRAM 9.72%、Compute(SM) 49.46%、occupancy 66%——比 T06 更接近
  Compute-Bound，但仍有 stall；1024：DRAM 5.34%、Compute 56.28%。
- Triton num_stages=3 的 GFLOPS 继续上升，说明瓶颈在逐步从访存等待移向计算。
- 判定方法见 `docs/CONCEPTS.md` §2。

## 8. 知识点完整性检查

已覆盖：cp.async 语义、double buffer、commit/wait、stage 取舍、流水线收益的诚实判断、
SASS LDGSTS、Triton num_stages、cuTile latency hint、CuTe cp.async 双缓冲。
后置：Tensor Core/CUTLASS（T08）。

## 9. 过关问题及答案（17 题，一问一答）

**A 基础**

**Q1.** 普通 load 与 cp.async 的区别是什么？

**A1（回答）**：普通 load：global→寄存器→shared，占寄存器、阻塞、走 L1；cp.async：global→shared
   直接异步，不占寄存器、不阻塞后续，cg 模式绕过 L1。

**Q2.** `commit_group` 和 `wait_prior(N)` 分别做什么？

**A2（回答）**：commit：把已发出的 cp.async 打包成组；wait_prior(N)：等“除最近 N 组外”全部完成。

**Q3.** double buffer 需要几份 shared？为什么？

**A3（回答）**： 两份（STAGES=2）shared tile：一份正在被 FFMA 计算，另一份同时接收下一块 K 的数据。只有存在第二份缓冲区，“搬运”和“计算”才可能重叠；只有一份时两者只能串行。

**Q4.** SASS 里哪条指令证明 cp.async 真的生成了？

**A4（回答）**： `LDGSTS.E.BYPASS.128`：LDGSTS=global 直接到 shared 的异步拷贝指令；.128=一次 16B；BYPASS=绕过 L1（对应 cp.async 的 cg 模式）。

**B 理解**

**Q5.** 流水线为什么能“隐藏延迟”？它牺牲了什么资源？

**A5（回答）**：让搬运在计算期间后台进行，把 global 延迟藏进 FFMA 时间里；代价是多占 shared
   与更复杂的同步。

**Q6.** stage 数从 2 加到 4 会带来什么好处和代价？

**A6（回答）**： 预取深度越大，越能容忍更长的 global 延迟，流水线越不容易断；代价是 shared 用量随 STAGES 线性增长，可能挤压 occupancy，所以不是越大越好。

**Q7.** 为什么 T06 的 float4+pad 已经不错时，cp.async.cg 可能反而没收益？

**A7（回答）**：同步 float4 已能吃到 L1、本 shape 带宽足够、2-stage 收益小于 cg 丢 L1 的损失；
   所以没赢是配置/形状问题，不是技术错误。

**Q8.** Triton 的 `num_stages=3` 实测比 1 快多少？这说明什么？

**A8（回答）**：512³ 5.30→5.93 TFLOPS（+12%）、1024³ 7.48→8.24 TFLOPS（+10%）；说明流水线有效。

**C 应用**

**Q9.** 写出 double buffer 主循环中两次 wait/commit 的位置，并解释为什么第一块要特殊处理。

**A9（回答）**：迭代开始先 issue 下一块+commit；然后 wait_prior(1)；第一块没有“上一块”，必须
   issue(0)+commit+wait(0)。详见 §4.2。

**Q10.** 如果 STAGES=2 但 `wait_prior` 写成 0，流水线还成立吗？会怎样？

**A10（回答）**：不成立：wait(0) 会连“下一块”一起等，搬运算完全串行，double buffer 白设。

**Q11.** 本卡 1024³ 上 CUDA pipe 615us vs T06 604us，这个结果应该怎么写进报告？

**A11（回答）**：诚实写：本配置下 cp.async 2-stage 未超过 T06（615 vs 604us），技术正确但收益
    需更大 tile/更多 stage；同时报告 Triton 3-stage 的正面证据。

**Q12.** cp.async 要求多少字节对齐？非对齐边界我们怎么处理？

**A12（回答）**：cp.async 16B 要求 16B 对齐；非对齐/越界走标量 fallback（正确性优先）。

**Q13.** cuTile 的 `latency` hint 是什么？本 Ticket 实测 latency=1→4 带来了什么变化？

**A13（回答）**：`ct.load(..., latency=1..10)` 是官方流水线提示：允许编译器提前发出后续 tile
    的 load，形成软件流水线。本 Ticket 512³ 实测 0.636→0.563 ms（约 -11.5%）。

**Q14.** 如果 shared 容量只够 1 份 tile，还能流水线吗？

**A14（回答）**：不能做多 stage 双缓冲；最多单缓冲同步流水线，或换更小 tile 腾出 shared。

**Q15.** T08 会如何复用本 Ticket 的 cp.async/多 stage 知识？

**A15（回答）**：T08 的 CUTLASS/Tensor Core GEMM 用 cp.async（或 TMA）做多 stage 搬运管线；
    本 Ticket 的三个原语会在官方 example 里反复出现。

**Q16.** cuTile 的 `latency` 参数在官方 API 中是什么？latency=4 与 latency=1 的实测差异说明了什么？

**A16（回答）**：它是官方 `ct.load` 的软件流水线提示（1–10）：允许编译器提前加载后续 tile。
    实测 latency=1→4：512³ 0.636→0.563 ms（约 -11.5%），证明官方流水线提示有效。

**Q17.** CuTe 双缓冲 GEMM 的 `As` 形状为什么是 `(STAGES, TS, TS+1)`？`"ca"` 和 `"cg"` 的区别是什么？

**A17（回答）**：STAGES 维是双缓冲的两份 tile；TS×(TS+1) 是沿用 T06 的 padding 防 bank conflict。
    `"ca"` 允许 4B/8B/16B 拷贝并缓存；`"cg"` 仅限 16B 且绕过 L1——本 CuTe 路径用 4B
    标量拷贝，所以用 `"ca"`。
## 10. 本轮停止点

完成：CUDA cp.async double buffer、Triton num_stages 对比、cuTile/CuTe 官方能力记录、
正确性、NCU/SASS/NSYS 证据、讲义 15 题。
未做：T08 Tensor Core/CUTLASS。

## 11. 下一最小增量

T08 GEMM Tensor Core 与 CUTLASS：用 mma/CUTLASS 官方 example 把 fp16/bf16 路径跑起来，
与 cuBLAS 和前面所有 CUDA 手写档做阶梯对比。

## 附录：可复现命令

```bash
bash scripts/run_t07_all.sh
nvcc -O3 -arch=sm_89 -o src/t07_gemm_pipeline/cuda/gemm_pipe \
  src/t07_gemm_pipeline/cuda/gemm_pipe.cu
./src/t07_gemm_pipeline/cuda/gemm_pipe
ncu --set full -k 'regex:gemmPipe' -o docs/evidence/T07/t07-cuda-ncu \
  ./src/t07_gemm_pipeline/cuda/gemm_pipe
cuobjdump -sass src/t07_gemm_pipeline/cuda/gemm_pipe
```
