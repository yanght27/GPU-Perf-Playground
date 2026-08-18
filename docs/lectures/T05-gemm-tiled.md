# T05 GEMM Shared-Memory Tiling（唯一主讲义）

- Ticket：T05
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**Tile 分块与数据复用**
- 环境：gpp-core / 系统 nvcc / gpp-cutile / gpp-cute
- 官方来源：S01e、S02e、S03e、S18c（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——给 GEMM 加 shared-memory tiling，掌握 Tile 分块与数据复用；**依次学到**——①tiling 省多少 global 读的账本；②shared memory 作用域与两次 barrier；③zero-fill/%M/%N 边界；④TF32 精度陷阱；⑤bank conflict 概念预告；**学完应能回答**——BS=16 时 global 读降到几分之一？漏一个 `__syncthreads` 会怎样？；**相关工具/技术**——CUDA shared memory、Triton tl.dot、cuTile ct.mma、CuTe DSL smem、NCU/SASS。
- 本节内容：**要解决的问题**——T04 朴素 GEMM 重复读 global，只有 1.18TFLOPS；**核心手段**——把 A/B 切成 tile 缓存进 shared memory，block 内复用；**怎么实现**——`src/t05_gemm_tiled/` 五路径 tiled 版，CUDA 对齐官方 matrixMul；**怎么验证**——边界+512³/1024³ 全 PASS；Triton 10GF→5.6TF、cuTile 0.4→449GF；SASS 见 LDS/STS/BAR；**最终结论**——tile 是正确方向，但 CUDA 手写版还需解决搬移宽度与同步开销（T06）。

## 1. 上一轮问题回答

T04 已验收。T04 朴素 GEMM 只有 1.18 TFLOPS，NCU 显示它死在“标量 load + 地址计算”
上；T05 用 shared memory 把 A/B 的一个 tile 缓存起来，让 block 内线程复用同一块数据。

## 2. 规范实现与官方来源

| 路径 | 权威写法 |
| --- | --- |
| CUDA | NVIDIA cuda-samples `matrixMul`（As/Bs shared tile + 两次 `__syncthreads`） |
| Triton | 官方 tutorial 03（block tile + `tl.dot` + `%M/%N` 边界折回） |
| cuTile | 官方 `MatMul.py`（`ct.load` 2D tile + `ct.mma`，tile=16） |
| CuTe DSL | 官方 `03_gemm_tiled_smem.py`（`AddressSpace.smem` + `barrier_cta_sync`） |

## 3. 本轮实现结果

正确性：五路径在 512³、1024³ 与边界 17×31×33 全部 `CORRECT_PASS`（容差 5e-3）。

### T04 朴素 vs T05 Tiled（GFLOPS）

| 路径 | 512³（T04 → T05） | 1024³（T04 → T05） |
| --- | --- | --- |
| CUDA | 1.18 TF → **1.53 TF**（+30%） | 1.15 TF → **0.39 TF**（BS=16 在 1024 反而退化，见 §5.6） |
| Triton | 10.2 GF → **5.58 TF**（+500 倍） | 10.9 GF → **8.22 TF** |
| cuTile | 0.4 GF → **449 GF**（+1100 倍） | 0.4 GF → **465 GF** |
| cuBLAS 基线 | 7.65 TF | 10.0 TF |

**诚实结论**：tile 是方向，但本 Ticket 只完成第一步。CUDA BS=16 的 tile 在 1024 退化，
SASS/NCU 证据指向 **bank conflict 与 barrier stall**（T06 的课题）；Triton/cuTile 的
编译器 tiling 已接近库，说明“同是 tiling，工程实现细节决定上限”。

## 4. 核心代码与逐行解释

### 4.1 CUDA shared-memory tiled（对齐官方 matrixMul）

```cuda
template <int BS>
__global__ void gemmTiled(const float *A, const float *B, float *C, int M, int N, int K) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BS + tx;
    int row = blockIdx.y * BS + ty;

    __shared__ float As[BS][BS];   // 本 block 缓存的 A tile
    __shared__ float Bs[BS][BS];   // 本 block 缓存的 B tile

    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BS) {
        As[ty][tx] = (row < M && bk + tx < K) ? A[row * K + bk + tx] : 0.0f;
        Bs[ty][tx] = (bk + ty < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();            // 全部线程搬完，才开始用共享内存
        #pragma unroll
        for (int k = 0; k < BS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();            // 全部线程用完，才允许覆盖 tile
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```

- `__shared__`：这块数组放在 SM 的 shared memory，block 内所有线程可见。
- 外层 `bk` 循环：把 K 切成一段段 `BS`，每段先从 global 搬进 shared，再在 shared 上算。
- **两次 `__syncthreads()` 是 tiling 的灵魂**：第一次保证“都搬完再读”；第二次保证
  “都读完再覆盖”。漏掉任何一次 = 数据竞争（race）。
- 为什么更快：每个 A/B 元素从 global 读一次，之后被 block 内多个线程从 shared 复用；
  shared 带宽远高于 global。

### 4.2 Triton（官方 tutorial 03 写法，编译器管 shared）

```python
@triton.jit
def gemm_tiled_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m_load = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M   # 边界折回
    offs_n_load = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k0 * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k0 * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K; b_ptrs += BLOCK_K * N
    tl.store(...)
```

- 你写的是 tile 级 `tl.dot`；Triton 编译器自动分配 shared memory、插入 barrier。
- `%M/%N` 与 K 的 mask 是官方对非整除 shape 的标准处理。
- `input_precision="ieee"`：fp32 精确乘加；不加会落到 TF32，误差变大（本 Ticket 实测）。

逐行补充：
- `offs_m_load = (pid_m*BM+arange)%M`：load 地址永不出界，越界的行暂时“借”第 0 行
  等合法行来凑满 tile；借来的元素随后被 mask 或 K mask 排除/乘 0，不影响结果。
- `offs_m_store` 保留真实坐标，store 时用 `c_mask` 挡住越界——所以 load 折回、store 屏蔽，
  两边配合才正确。
- `tl.cdiv(K, BK)`：K 不被 BK 整除时也要多跑一段；`k0*BK` 用来算“当前还剩多少个 k”。
- `a_ptrs += BK`：指针移动 32 个 K 元素；`b_ptrs += BK*N`：B 按行主序跨过 32 行。
  不更新指针就会永远重复同一段 K。

### 4.3 cuTile（官方 MatMul.py，tile=16×16×16）

```python
@ct.kernel
def mm_kernel(A, B, C, tm, tn, tk):
    bidx = ct.bid(0); bidy = ct.bid(1)
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bidx, bidy), tile=acc)
```

- 与 T04 的 tile=1 版本**同一份官方骨架**，只把 `tm/tn/tk` 从 1 改成 16：
  GFLOPS 从 0.4 → 449，涨了约 1100 倍。这就是“tile 大小决定搬运算价比”的最直观证据。

逐行补充：
- `ct.num_tiles(A, axis=1, shape=(tm, tk))`：把 A 按 (tm,tk) 分块后，K 方向有多少块；
  它自动处理 K 不被 tk 整除的情况（与 `ct.cdiv` 等价）。
- `ct.load(..., padding_mode=ct.PaddingMode.ZERO)`：官方边界策略就是 zero-fill；
  所以 17×31×33 也能过，不需要手写 if。
- `ct.mma(a, b, acc)`：tile 级乘加；编译器决定是否用 Tensor Core/FFMA 以及 shared 布局。
- grid 用 `(ct.cdiv(M,tm), ct.cdiv(N,tn))`：二维 tile 网格，和 CUDA 的 grid.x/y 同义。

### 4.4 CuTe DSL（官方 03_gemm_tiled_smem）

```python
a_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)
b_smem = cutlass.Array(cutlass.Float32, (TS, TS), space=cutlass.AddressSpace.smem)
...
prims.barrier_cta_sync(0)   # 对应 CUDA __syncthreads()
```

CuTe DSL 把 shared memory 与 barrier 显式暴露给你，和 CUDA 版本一一对应；这是学习
“编译器替你做了什么”的最佳对照路径。

逐行补充：
- `space=cutlass.AddressSpace.smem`：显式把数组放进 shared memory；Triton/cuTile 这一行
  由编译器生成，CuTe DSL 让你亲手写。
- `a[row, bk+tx]`：CuTe Array 支持二维下标；等价于 CUDA 的 `A[row*K + bk+tx]`。
- `prims.barrier_cta_sync(0)`：官方 barrier 原语，对应 `__syncthreads()`；参数 0 是
  barrier 编号（同一 block 内要一致）。
- 官方教程只支持整除 shape；我们加的 `row<M / col<N / bk+...<K` 是 zero-fill 策略，
  已在 §5.4 解释。

## 5. 核心知识点要点

### 5.0 先算一笔账：tiling 到底省了多少 global 读

以 BS=16、K=512、一个 block 为例：

- 朴素版：block 内 16×16=256 个线程，每个输出读 A 行 512 次 + B 列 512 次，
  合计 256×1024 = **262,144 次 global load**。
- Tiled 版：K 切成 512/16=32 段；每段搬一个 16×16 的 A tile + 一个 16×16 的 B tile，
  即 2×256×32 = **16,384 次 global load**，其余乘加都读 shared。
- 比例：262,144 / 16,384 = **16 倍**。这个 16 正好等于 BS：**tiling 把 global 流量
  降到原来的 1/BS**（一维 tile 的理想情况）。
- 为什么时间没有快 16 倍？因为 shared 访问、barrier、bank conflict 也要时间；
  NCU 里 No Eligible 70.66% 就是这些新开销的痕迹。**省了 global，不代表没有新瓶颈。**

### 5.1 Tiling 为什么能减少访存

- 朴素版每个输出元素读 2K 个 global 数据；整个 block 会重复读同一块 A/B 很多遍。
- Tiled 版：一个 `BS×BS` 的 A tile 被 block 内 BS 个线程各用 BS 次；从 global 搬一次，
  之后 BS² 次乘加都读 shared。**数据复用 = 把 global 流量换成 shared 流量。**

### 5.2 Shared memory 的三个事实

1. 作用域：一个 block 内所有线程共享；block 之间不可见。
2. 速度：比 global 快一个数量级，但容量很小（本机每 SM 约 100 KB）。
3. 需要程序员管理同步：`__syncthreads()`（Triton/cuTile 由编译器插）。

### 5.3 两次 barrier 的经典错误

- 缺第一次：线程 B 还没搬完，线程 A 就开始读 → 读到旧值。
- 缺第二次：线程 A 还没算完，线程 B 就覆盖 tile → 算错。
- 这是“race condition”在共享内存上的标准形态，面试必问。

### 5.4 边界处理：zero-fill 与 %M/%N

- CUDA/CuTe：越界位置填 0 再算，结果不受影响（0×x=0）。
- Triton：官方 `%M/%N` 把 load 地址折回有效范围，store 再按真实坐标 mask。
- 两种策略等价：**把“不该存在的数据”变成对累加无贡献的值。**

### 5.5 SASS 怎么确认 tiling 真的发生了

`LDS/STS`（shared load/store）与 `BAR.SYNC` 出现，说明数据确实走了 shared memory 且有
同步。T04 的 SASS 里没有这些。证据：`docs/evidence/T05/t05-cuda-sass.txt`。

### 5.6 为什么 CUDA BS=16 在 1024 反而退化（T06 的伏笔）

NCU（512 shape）：Duration 184.64 us、SM 92.09%、L1 94.32%、DRAM 4.45%，
但 **issued warp/scheduler 只有 0.29、No Eligible 70.66%**——大量时间在等 barrier/等
shared 数据，而不是算 FFMA。原因预告：
1. shared memory 访问有 **bank conflict**（T06 要量化并消除）；
2. 每个线程仍只搬 4B，128-bit 向量化搬 tile 会更高效（T06）；
3. BS=16 的 block 太小，同步次数太多（T06/T07 会调）。

### 5.7 bank conflict 的准确含义（T06 会动手修，现在必须懂原理）

- shared memory 物理上分 **32 个 bank**，每个 bank 每周期只能服务一个地址（4 字节）。
- 同一 warp 的 32 个线程访问 **同一个 bank 的不同地址** → 串行排队，叫 **conflict**；
  访问 **同一个地址** → 广播，无冲突；落在 **32 个不同 bank** → 无冲突。
- 本 kernel：BS=16、block=(16,16) 时，一个 warp 恰好跨 `ty=0` 和 `ty=1` 两行、
  每行 16 个 `tx`。`As[ty][k]` 的两次访问地址差 `16*4B`，落在相隔 16 的两个 bank，
  正好不冲突；`Bs[k][tx]` 同理。所以 BS=16 退化的主因**不是** bank conflict，
  而是块太小 + barrier 太频繁 + 每线程只搬 4B。T06 会用 NCU 的 conflict 指标验证
  这个判断（实测 A 档 ld/st conflict 均为 0）。
- 把概念记进 `docs/CONCEPTS.md`：bank=32×4B；conflict=同 bank 不同地址；broadcast 无冲突。

### 5.8 为什么 CUDA BS=16 在 1024 反而比朴素更慢

- 1024/16=64 → grid 64×64=4096 个 block；每个 block 做 1024/16=64 次 tile 迭代，
  每次迭代 2 次 `__syncthreads` → 每个 block 128 次 barrier。
- block 太小 → 同步开销占比高、每个 block 只搬 2KB，复用深度只有 16；
- 512 时同样 BS=16 只慢一点，1024 时 barrier/调度开销被放大，所以事件计时从
  1.87 ms 退化到 5.50 ms。
- 对策（T06）：加大 BS、float4 搬 tile、减少 barrier 次数、padding/swizzle。
- **这也是“grid/block 配置不能只靠感觉”的第二次实例**（第一次是 T02 的 256 vs 1024）。

### 5.9 更新到持久词典

`docs/CONCEPTS.md` 新增/更新：Tiling 与数据复用、shared memory、barrier 同步、
zero-fill/%M/%N 边界策略、`input_precision` 与 TF32 精度陷阱。

## 6. 性能分析（实测见 §3 表）

- 同一个 tile 思想：Triton 5.58 TF、cuTile 449 GF、CUDA 1.53 TF——**工具的实现质量差距
  第一次反超手写**。
- CUDA 手写版最能暴露问题：SASS 的 LDS/STS 告诉我们它在用 shared，但 scheduler 数据显示
  大量 no-eligible stall——T06 的优化目标已经明确。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

- CUDA tiled 512：DRAM 4.45%、Compute(SM) 92.09%，但 Roofline FP32 峰值仍低、
  No Eligible 70.66%——**瓶颈是 barrier/shared 访存导致的 latency-bound**，不是带宽。
- Triton/cuTile tiled：GFLOPS 大幅提高，趋向 Compute-Bound。
- 判定方法统一见 `docs/CONCEPTS.md` §2。

## 8. 知识点完整性检查

已覆盖：Tiling、数据复用、shared memory、barrier/race、zero-fill 与 %M/%N 边界、
SASS LDS/STS/BAR、TF32 精度、各路径实现质量对比。
T06 前置已埋好：bank conflict、128-bit 共享访存、block 大小选择。
后置：Tensor Core（T08）。

## 9. 过关问题及答案（15 题，一问一答）

**A 基础**

**Q1.** 用“数据复用”解释：为什么 shared-memory tiling 比朴素 GEMM 少读 global？

**A1（回答）**：朴素版每个输出读 2K 次 global；tiled 版每个 tile 从 global 读一次，之后 BS² 次乘加
   都读 shared，把“重复的 global 流量”换成“便宜的 shared 流量”。

**Q2.** `__syncthreads()` 两次分别防什么？漏掉一次会发生什么？

**A2（回答）**：第一次：全部线程搬完才允许读；第二次：全部线程算完才允许覆盖。漏一次 = race，
   读到旧值或半新半旧数据，结果随机。

**Q3.** shared memory 的作用域是什么？一个 block 能读另一个 block 的 shared 吗？

**A3（回答）**：作用域是一个 block；跨 block 不可见，跨 block 协作只能回 global 或用特殊原语。

**Q4.** 非整除边界时，zero-fill 为什么不会影响累加结果？

**A4（回答）**：填 0 后，越界位置参与 `0*x` 或 `x*0`，对累加贡献为 0，等价于没参与。

**B 理解**

**Q5.** Triton 源码里没有 `__shared__`，它怎么做到 tiling？怎么证明它用了 shared？

**A5（回答）**：Triton 编译器把 block 级 `tl.dot/tl.load` 翻译成 shared tile + barrier；
   证据：Triton PTX/IR 以及性能从 10 GF 涨到 5.6 TF。官方 tutorial 03 就是它的权威写法。

**Q6.** `tl.dot` 不加 `input_precision="ieee"` 会发生什么？为什么？

**A6（回答）**： 默认可能用 TF32 算 dot：TF32 的尾数只有 10 位（格式共 19 位 = 1 符号 + 8 指数 + 10 尾数），舍入误差比 fp32 大；`input_precision="ieee"` 强制走 fp32 精确乘加。本 Ticket 512/1024 实测误差从 6e-3/8e-3 降到 1e-5 量级。

**Q7.** SASS 里出现 LDS/STS/BAR.SYNC 说明什么？T04 的 SASS 为什么没有？

**A7（回答）**：LDS/STS=shared 读写，BAR.SYNC=block 内同步；说明 tiling 真发生了。T04 直接
   LDG 重复读 global，所以没有这些指令。

**Q8.** cuTile 只把 tile 从 1 改成 16，为什么 GFLOPS 涨了约 1100 倍？

**A8（回答）**：tile=1 时每个 block 只算 1 个输出，固定开销（调度、地址、原语）占比 99%+；
   tile=16 时一个 block 算 256 个输出，搬运的 tile 数据被大量复用，固定开销被摊薄。

**C 应用**

**Q9.** CUDA tiled 512 的 NCU：DRAM 4.45%、SM 92.09%、No Eligible 70.66%，它是什么 Bound？
   下一步最该优化什么？

**A9（回答）**：带宽和算力都不满，No Eligible 70.66% 说明 warp 常在等 barrier/shared——latency-bound
   /同步瓶颈。下一步 T06：消除 bank conflict、128-bit 化 shared 访问、调 block 大小。

**Q10.** 把 CUDA 的 BS 从 16 改成 32，shared memory 用量变成几倍？为什么可能减少同步次数？

**A10（回答）**：shared 用量从 2×16×16×4=2KB 变成 2×32×32×4=8KB（4 倍）；K 循环次数减半、
    同步次数减半，但 bank conflict 风险变化要实测（T06）。

**Q11.** 边界策略 zero-fill 与 %M/%N 的等价性体现在哪？

**A11（回答）**：两者都保证“越界位置对累加无贡献”：zero-fill 让乘加结果为 0；%M/%N 让 load 落到
    合法元素、store 再用 mask 挡住越界输出。

**Q12.** 如果要进一步减少 shared 访存的 bank conflict，T06 最可能做哪两件事？

**A12（回答）**：最可能：padding/swizzle 消除 bank conflict；把每线程 4B 的 shared 搬移改成
    128-bit（float4），并在不同 BS 下实测 NCU。

**Q13.** 用 BS=16、K=512 算账：一个 block 在朴素版和 tiled 版各需要多少次 global load？
    比值是多少？这个比值由哪个参数决定？

**A13（回答）**：朴素版：256 线程 × (512+512)=262,144 次；tiled 版：32 段 × 2 tile × 256=16,384 次；
    比值 16=BS。global 流量降到 1/BS。

**Q14.** shared memory 有多少个 bank、每个 bank 多宽？同一 warp 访问“同 bank 不同地址”
    和“同地址”分别叫什么、代价是什么？

**A14（回答）**：32 个 bank，每 bank 4B。同 bank 不同地址 = bank conflict（串行化，性能下降）；
    同地址 = broadcast（硬件一次广播给整个 warp，无冲突）。本 kernel 的两种访问
    都是 32 线程相邻 4B，各占一个 bank，所以**不是** bank conflict 主导；BS=16 的
    退化主因是 block 太小、barrier 太多、每线程只搬 4B。

**Q15.** 如果把 `__syncthreads()` 放进 `if (row < M)` 里面，会发生什么？为什么？

**A15（回答）**：死锁（或未定义行为）：barrier 要求 block 内**所有**线程都到达；放进条件分支后，
    不满足条件的线程永远不来，满足条件的线程永远等。规则：`__syncthreads()` 必须
    被 block 内所有线程在同一个控制流位置执行。
## 10. 本轮停止点

完成：五路径 tiled、正确性含边界、benchmark 对比、NCU/SASS/NSYS 证据、讲义 12 题。
未做：T06 共享内存优化（bank conflict/128-bit）。

## 11. 下一最小增量

T06 GEMM 共享内存优化：bank conflict 消除、128-bit shared 访存、block 大小调优，
用 NCU 的 shared 指标量化收益。

## 附录：可复现命令

```bash
bash scripts/run_t05_all.sh
nvcc -O3 -arch=sm_89 -o src/t05_gemm_tiled/cuda/gemm_tiled \
  src/t05_gemm_tiled/cuda/gemm_tiled.cu
ncu --set full -k 'regex:gemmTiled' --launch-skip 26 -o docs/evidence/T05/t05-cuda-ncu-512 \
  ./src/t05_gemm_tiled/cuda/gemm_tiled
cuobjdump -sass src/t05_gemm_tiled/cuda/gemm_tiled
```
