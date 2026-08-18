# T04 朴素 GEMM + cuBLAS 基线（唯一主讲义）

- Ticket：T04
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**朴素 GEMM 的二维索引映射、访存/计算比，以及为什么要有 cuBLAS 库基线**
- 环境：gpp-core / 系统 nvcc+cuBLAS / gpp-cutile / gpp-cute
- 官方来源：S01d、S02d、S03d、S09a、S10e（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——写出朴素 GEMM 并建立 cuBLAS 库基线，理解访存/计算比与 Roofline；**依次学到**——①二维索引与行主序；②cuBLAS 列主序转换；③算术强度；④NCU 的 SM busy≠FLOP busy；⑤Bound 三分类；**学完应能回答**——为什么 SM 94.9% 忙但 FP32 只用 6%？朴素 GEMM 是什么 Bound？；**相关工具/技术**——PyTorch、CUDA C++、cuBLAS、Triton、cuTile、CuTe DSL、NCU Roofline。
- 本节内容：**要解决的问题**——T03 只会一维元素级算子；GEMM 是 AI 核心，需要二维索引并解释“为什么朴素实现慢库那么多”；**核心手段**——三重循环朴素 GEMM + cublasSgemm 一行基线；**怎么实现**——`src/t04_gemm_naive/` 五路径 + 边界 shape；**怎么验证**——fp64 参考全 PASS；cuBLAS 7.7TFLOPS vs CUDA 朴素 1.18TFLOPS；NCU：DRAM 5.9%、FP32 峰值 6%、SM 94.9%；**最终结论**——朴素 GEMM 是指令/L1 瓶颈，tiling 是下一步方向。

## 1. 上一轮问题回答

阶段一（T01–T03）预验收通过。本讲义默认读者已完成：一维索引/边界/Grid 配置、
合并访问与 128-bit 向量化、Memory-Bound 判断、NSYS/NCU/SASS 基本用法。T04 把索引从
一维升级到二维，并第一次引入“库基线”与“访存/计算比”的定量讨论。

## 2. 规范实现与官方来源

| 路径 | 依据 | 版本 |
| --- | --- | --- |
| PyTorch | `torch.einsum`/`torch.matmul`（fp64 参考） | 2.13 |
| CUDA | 二维 grid/thread + 官方头 `cublas_v2.h` 的 `cublasSgemm` | CUDA 13.0 |
| Triton | 官方 tutorial 03 的指针算术说明；T04 用最直白的逐元素朴素版 | v3.7.1 |
| cuTile | 官方 `MatMul.py` 的 `ct.mma` 写法，tile 取 1×1×1 退化为朴素 | `29444e0c` |
| CuTe DSL | 官方 GEMM tutorials 系列；T04 用 Array 索引写逐元素朴素版 | `564d267e` |

## 3. 本轮实现结果

正确性：五路径在 512³、1024³ 与边界形状 17×31×33、1×128×1 全部 `CORRECT_PASS`
（fp32 对 fp64 参考 max_abs_err ≤ 5e-3；1024³ 的 1.8e-3 是 fp32 累加误差，属预期）。

### 实测性能（GFLOPS = 2·M·N·K / 时间）

| 路径 | 512³ | 1024³ |
| --- | --- | --- |
| PyTorch matmul（库基线） | ~0.036 ms，7.5 TFLOPS | ~0.20 ms，10.9 TFLOPS |
| cuBLAS Sgemm | ~0.035 ms，7.7 TFLOPS | ~0.21 ms，10.0 TFLOPS |
| CUDA 朴素 | ~0.23 ms，1.18 TFLOPS | ~1.87 ms，1.15 TFLOPS |
| Triton 朴素（每输出一 program） | ~26.3 ms，10.2 GFLOPS | ~197 ms，10.9 GFLOPS |
| cuTile 朴素（tile=1×1×1） | ~625 ms，0.4 GFLOPS | ~5.0 s，0.4 GFLOPS |
| CuTe DSL 朴素（Python 调用） | ~13.2 ms | ~16.4 ms |

**第一课结论**：同样是“朴素写法”，抽象层级越低的工具越容易写得非常慢；库（cuBLAS/
cuDNN 等）经过几十年优化，是工业界默认基线。我们的目标是 T05–T08 逐步逼近它。

## 4. 核心代码与逐行解释

### 4.1 PyTorch：黄金参考与语义

```python
ref = (a.double() @ b.double()).float()      # fp64 参考
out = torch.einsum("mk,kn->mn", a, b)        # 显式朴素语义
out = a @ b                                  # PyTorch 库实现（cuBLAS 后端）
```

### 4.2 CUDA 朴素 kernel（本轮主角）

```cuda
__global__ void gemmNaive(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 输出行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 输出列
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
```

- 二维 launch：`dim3 block(16,16)`、`dim3 grid((N+15)/16,(M+15)/16)`。x 方向对应列，
  y 方向对应行——这是 T04 从一维索引升级二维的第一步。
- `row*K+k`：A 是行主序（row-major），第 row 行第 k 列的线性地址。
- `k*N+col`：B 行主序，第 k 行第 col 列。
- 内层 `for(k)`：每个输出元素要读 A 的一整行和 B 的一整列，串行累加。
- **没有 shared memory**：A/B 的每个元素被反复从 global memory 读（T05 要解决的）。

### 4.3 Triton 朴素版

```python
@triton.jit
def gemm_naive_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
    pid = tl.program_id(0)
    row = pid // N; col = pid % N      # 一维 grid 手算二维坐标
    acc = 0.0
    for k in tl.range(0, K):
        a = tl.load(a_ptr + row * K + k)
        b = tl.load(b_ptr + k * N + col)
        acc += a * b
    tl.store(c_ptr + row * N + col, acc)
```
一个 program 算一个输出元素，所以 512² 有 262,144 个 program，K 循环 512 次，
每次 2 次标量 load——这是“抽象上的朴素”，所以只有 ~10 GFLOPS。

### 4.4 cuTile 朴素版（官方 mma 原语 + tile=1）

```python
@ct.kernel
def mm_kernel(A, B, C, tm, tn, tk):
    bidx = ct.bid(0); bidy = ct.bid(1)
    nk = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(nk):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)      # 官方矩阵乘累加
    ct.store(C, index=(bidx, bidy), tile=acc)
```
这是官方 `MatMul.py` 的完整形态，只是 `tm=tn=tk=1`：一个 processor 一个输出元素，
`ct.mma` 退化为标量乘加。0.4 GFLOPS 说明**tile 太小 = 每个 block 搬运开销远大于计算**。

### 4.5 CuTe DSL 朴素版（显式 Array 索引）

```python
@cute.kernel
def gemm_naive_kernel(a_arr, b_arr, c_arr, M, N, K):
    tx,_,_=cute.arch.thread_idx(); _,ty,_=cute.arch.thread_idx()
    bx,_,_=cute.arch.block_idx(); _,by,_=cute.arch.block_idx()
    bdx,_,_=cute.arch.block_dim(); _,bdy,_=cute.arch.block_dim()
    col = bx*bdx+tx; row = by*bdy+ty
    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc += a_arr[row*K+k]*b_arr[k*N+col]
        c_arr[row*N+col] = acc
```
与 CUDA 版同构；注意这里把 grid 的 x 轴分给列、y 轴分给行，与 CUDA 代码一致。

### 4.6 cuBLAS 基线（column-major 转换是必考知识点）

```c
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, dB, N, dA, K, &beta, dBlas, N);
```

- cuBLAS 按 **列主序** 存矩阵。行主序的 `C[M×N]=A[M×K]@B[K×N]` 等价于列主序的
  `C^T = B^T @ A^T`。
- 所以把矩阵尺寸传成 `N,M,K`，第一矩阵传 B（leading dim N），第二矩阵传 A
  （leading dim K），输出传 C（leading dim N）。**不转换会得到转置错误的结果**。

## 5. 核心知识点要点（T04 全部讲透）

### 5.1 GEMM 是什么、为什么是 AI 的基础

- 矩阵乘 `C[m,n] = Σ_k A[m,k]·B[k,n]`；Transformer 的 QKᵀ、Attention 输出、MLP 全连接
  全都是 GEMM。
- 两个大矩阵相乘时，每个输入元素会被使用很多次（A 的每行被 N 个输出复用，B 的每列被
  M 个输出复用），所以 GEMM 的优化核心是**数据复用**（T05 Tiling 的动机）。

### 5.2 行主序二维索引

一维数组里存二维矩阵：`A[i,j] → A[i*cols + j]`。T04 的两个公式
`row*K+k`（A）、`k*N+col`（B）就是它的直接应用。**二维线程布局**（blockIdx.x/y）
与二维数据索引是对应的：x→列、y→行。

### 5.3 访存/计算比（arithmetic intensity）

- 朴素 GEMM：每个输出元素做 K 次乘加（2K FLOP），但要读 K 次 A + K 次 B = 2K 次 load。
- 从 DRAM 视角，每个 A/B 元素被重复读 M 或 N 次（因为没缓存复用），所以
  **算术强度极低**。
- 更直观：CUDA 朴素 512³ 只跑出 1.18 TFLOPS，而本卡 FP32 峰值约 10 TFLOPS 量级——
  大部分时间在等数据/算地址，不是在算乘加。

### 5.4 怎么用 NCU 区分“SM 忙”和“算力被用满”（本轮最重要的工具课）

NCU 对 CUDA 朴素 kernel 的实测：

| 指标 | 值 | 怎么读 |
| --- | --- | --- |
| Compute (SM) Throughput | 94.88% | SM 的**指令发射管道**很忙 |
| GPU Speed Of Light Roofline | 只达到 FP32 峰值的 6% | **有用的浮点数学**只用了 6% |
| DRAM Throughput | 5.92% | 显存远没打满 |
| L1/TEX Throughput | 97.20% | L1/LSU 管道被标量 load 打满 |
| Achieved Occupancy | 91.74% | warp 够多 |
| SASS 指令计数 | 58 个 LDG、42 个 IMAD、29 个 FFMA | 一半以上的指令是**取数和算地址** |

结论：朴素 GEMM 是 **指令/L1 管道瓶颈**——SM 在忙，但忙的是“取数、算地址、发射”，
真正做乘加的 FFMA 只占少部分。这就是它只有 1.18 TFLOPS 的原因。
**“SM 忙” ≠ “算力被有效利用”，要看 Roofline/GFLOPS。**

### 5.5 fp32 累加误差为什么随 K 增大

K=1024 时朴素 CUDA 对 fp64 参考的 max_abs_err 达 1.8e-3（K=512 是 2.7e-4）。
1024 次 fp32 加法会累积舍入。所以 T04 门禁容差设为 5e-3 并记录；真正高精度训练/推理
会用 fp32 累加 + 分块顺序（Kahan/TF32 是后话）。

### 5.6 各路径为什么这么慢（横向对比）

| 路径 | 慢在哪 |
| --- | --- |
| Triton 朴素 | 每输出一个 program、K 次标量 load，program 开销巨大 |
| cuTile tile=1 | 每个 block 只算 1 个元素，tile 调度开销 >> 计算 |
| CuTe/CUDA 朴素 | 指令流里地址计算和标量 load 占一半以上 |
| PyTorch/cuBLAS | 分块 + 向量化 + 寄存器复用 + 线程协作，接近峰值 |

### 5.7 Bound 三分类的正式定义与判定流程（秋招高频，务必背下）

- **Memory-Bound**：带宽先到顶（DRAM% 高、Compute% 低）。例：vector add、ReLU。
- **Compute-Bound**：数学管道先到顶（Roofline 接近 FP32/FP16 峰值）。例：cuBLAS 1024³。
- **Latency-Bound**：带宽和算力都不满，stall 高（等 load/barrier/依赖链），occupancy
  不足以隐藏延迟。例：本卡 NSYS 缺失不影响这个定义；未来 T07/T12 会实测 stall。
- 判定顺序：① 算术强度 `AI=FLOPs/bytes`；② NCU SpeedOfLight 的 DRAM%/Compute%/Roofline；
  ③ SchedulerStats 看 stall；④ SASS 数 LDG/FFMA 验证。完整流程和本项目案例表见
  `docs/CONCEPTS.md` §2。
- T04 朴素 GEMM 的诚实结论：DRAM 5.9%、FP32 峰值 6% → 不是典型的 Memory/Compute；
  SM 94.9% + L1 97.2% + LDG/IMAD 多 → **指令/L1 管道瓶颈**（可归为 Latency/Instruction
  Bound 一类）。这说明真实世界不止两分类。

### 5.8 工具定位与 CUTLASS 的边界

- 六种“写算子”工具的定义与选择标准已集中到 `docs/CONCEPTS.md` §1；T04 新增的对比结论：
  库（cuBLAS）> 手写朴素 CUDA > 不适合的 DSL tile 配置。
- **CUTLASS C++ 为什么归 T08**：官方 CUTLASS C++ examples 从分块/collective 起步，没有
  “朴素单线程 GEMM”这一档；直接在这里用会跳过 T05–T07 的 tiling 学习路径，所以 T04
  记录 N/A（归 T08），CuTe DSL 路径负责本 Ticket 的“显式低层”视角。

## 6. 性能分析（实测数据见 §3 表）

- cuBLAS ≈ PyTorch matmul：两者都是库，差距在个位数百分比。
- CUDA 朴素比 cuBLAS 慢 6–7 倍；Triton/cuTile 的“极端朴素”慢 3 个数量级。
- 1024³ 比 512³ 计算量增加 8 倍：cuBLAS 时间 0.035→0.21 ms ≈6 倍，CUDA 朴素 ≈8 倍，
  说明库在大 shape 下更接近线性扩展。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

- CUDA 朴素 GEMM：**既不是 DRAM-Bound（5.92%），也不是纯 FP32 Compute-Bound（峰值 6%）**，
  而是**指令发射/L1 访存管道瓶颈**（SM 94.88%、L1 97.20%）。
- 结论要诚实：Bound 分类不是只有 Memory/Compute 两档；本轮学到第三类——
  “instruction/latency-bound”式的低效实现。T05 用 Tiling 提高数据复用、减少标量 load，
  会显著改变 NCU 剖面。
- cuBLAS 在 1024³ 达到 ~10 TFLOPS，接近本卡 FP32 峰值，是 Compute-Bound 的库实现。

## 8. 知识点完整性检查

已覆盖：二维索引与线程布局、三重循环、行主序、column-major 转换、算术强度、库基线、
NCU Roofline 读法、fp32 累加误差、五路径横向对比、边界 shape。
T05 前置：数据复用动机、Tile 概念预告、官方 tutorial 03 定位——已在本讲义 §5.1/§5.6
说明。Shared Memory 实操归 T05。
`config/coverage-matrix.md` 已同步。

## 9. 过关问题及答案（14 题，一问一答）

**A 基础**

**Q1.** 写出 GEMM 定义式；为什么说 Transformer 里到处都是 GEMM？

**A1（回答）**：`C[m,n]=Σ_k A[m,k]B[k,n]`。Attention 的 QKᵀ、V 加权、MLP 全连接都是矩阵乘，GEMM
   性能几乎决定模型速度。

**Q2.** 行主序下 A[row,k] 和 B[k,col] 的线性地址公式是什么？

**A2（回答）**： 行主序的线性地址 = 行号 × 该行元素数 + 列号。A 的形状是 M×K，所以 `A[row,k]` 在 `row*K+k`；B 的形状是 K×N，所以 `B[k,col]` 在 `k*N+col`。两个公式的“列宽”分别是 K 和 N，别写混。

**Q3.** 二维 launch 的 blockIdx.x/y 与 threadIdx.x/y 分别如何映射到输出行列？

**A3（回答）**：x 方向→列：`col=blockIdx.x*blockDim.x+threadIdx.x`；y 方向→行：
   `row=blockIdx.y*blockDim.y+threadIdx.y`。

**Q4.** cuBLAS 是列主序，行主序 C=A@B 应该怎样调用 `cublasSgemm`（尺寸、顺序、leading dim）？

**A4（回答）**：传 `(N,M,K)`，`dB` 当第一矩阵（lda=N）、`dA` 当第二矩阵（lda=K）、输出 `dBlas`
   lda=N；op 均为 CUBLAS_OP_N。因为列主序的 `C^T=B^T A^T`。

**B 理解**

**Q5.** 朴素 GEMM 每个输出元素要读多少次 A/B？整个 C 算完，A 的每个元素被读了几次？这叫
   什么现象？

**A5（回答）**：每个输出读 2K 次（A 行 K 次 + B 列 K 次）。A 的每个元素被 N 个输出读，B 的每个
   元素被 M 个输出读——这是**数据复用**，但朴素版没有用缓存/共享内存承接，等于重复
   从 global 读。

**Q6.** NCU 显示 Compute(SM)=94.88%，为什么 Roofline 却说只用了 FP32 峰值的 6%？两个数矛盾吗？

**A6（回答）**：不矛盾。Compute(SM) 是“指令发射管道利用率”，包括取数、算地址、发射等所有指令；
   Roofline 是“FP32 乘加峰值占比”。朴素 kernel 里 58 个 LDG + 42 个 IMAD 对 29 个
   FFMA——SM 忙的是杂活，真正数学只占 6%。

**Q7.** 为什么 K=1024 的 max_abs_err 比 K=512 大？

**A7（回答）**： fp32 每做一次加法都可能产生约 2^-24 量级的相对舍入；K 次累加把 K 次舍入串成一条链，误差会逐步累积，所以 K=1024 的误差通常比 K=512 大。这正是 T04 用 fp64 做黄金参考、并记录容差的原因。

**Q8.** cuTile tile=1 的官方写法为什么只有 0.4 GFLOPS？

**A8（回答）**：tile=1 时每个 block 只算 1 个元素，却要完整走一遍 tile 调度/边界检查/load 原语，
   固定开销远超一个乘加；tile 太小是病根（T05 会放大 tile）。

**C 应用**

**Q9.** 如果 M=17、N=31，block=(16,16)，grid 应该怎么算？没有 `if(row<M && col<N)` 会怎样？

**A9（回答）**：`grid=ceil(N/16)=2, ceil(M/16)=2`（x 对应 N/列）。没有边界判断时，row≥17 或
   col≥31 的线程会读写越界地址。

**Q10.** 把 CUDA 朴素 kernel 的 block 从 (16,16) 改成 (32,8)，索引公式需要改吗？为什么？

**A10（回答）**：不需要。公式用 blockDim.x/y 而非写死 16，自动适应任何 block 形状。

**Q11.** 朴素 GEMM 是 Memory-Bound 还是 Compute-Bound？用 NCU 数据说明。

**A11（回答）**：都不是纯的：DRAM 5.92%（不 Memory-Bound），FP32 峰值 6%（不 Compute-Bound），
    SM 94.88%/L1 97.20%——是**指令/L1 管道瓶颈**的低效实现。

**Q12.** 要让 CUDA 朴素 1.18 TFLOPS 逼近 cuBLAS 7.7 TFLOPS，下一个 Ticket（T05）最关键的
    一个动作是什么？为什么它能减少什么？

**A12（回答）**：Tiling：把 A/B 的 tile 装进 shared memory，让一个 block 内的线程复用同一块数据，
    把“K 次重复 global load”变成“一次 global load + 多次 shared 读”。这样标量 load
    和地址计算大幅减少，SM 的指令预算回到 FFMA 上。

**Q13.** 用一句话分别定义 Memory-Bound、Compute-Bound、Latency-Bound；判定一个 kernel 属于
    哪一类，你按什么顺序做、各看什么指标？

**A13（回答）**：Memory-Bound=带宽先到顶；Compute-Bound=数学管道先到顶（Roofline 接近峰值）；
    Latency-Bound=两者都不满、stall 高/依赖长/occupancy 不足以隐藏延迟。判定顺序：
    ① 算 AI=FLOPs/bytes；② NCU SpeedOfLight（DRAM%、Compute%、Roofline）；
    ③ SchedulerStats 看 stall；④ SASS 数 LDG/FFMA 佐证。完整案例见 CONCEPTS.md §2。

**Q14.** PyTorch、Triton、cuTile、CuTe DSL、CUTLASS C++、cuBLAS 分别是什么？为什么 T04 用
    cuBLAS 而不是 CUTLASS C++ 做基线？

**A14（回答）**：PyTorch=框架自动调度；Triton=block 级 Python DSL 编译器；cuTile=tile 级 Python DSL；
    CuTe DSL=CUTLASS 的显式 layout Python DSL；CUTLASS C++=高性能 GEMM 模板库；
    cuBLAS=官方 BLAS 库。T04 用 cuBLAS 是因为它的 `cublasSgemm` 一行就给出行业最强
    基线，而 CUTLASS C++ 的官方例子都从分块/tensor-core 起步，属于 T05–T08 的进阶
    内容（官方能力适用边界）。
## 10. 本轮停止点

完成：五路径朴素 GEMM + cuBLAS 基线、正确性（含边界）、benchmark、NCU full/SASS/NSYS
证据、讲义+12 题、提交 `fd236ef`。
未做：T05 GEMM Tiling（shared memory）。

## 11. 下一最小增量

T05 GEMM Shared-Memory Tiling：每个 block 把 A/B 的一个 tile 装进 shared memory 复用，
用 NCU 看 DRAM/L1 与 GFLOPS 的变化，正式学习 Tile 与数据复用。

## 附录：可复现命令

```bash
nvcc -O3 -arch=sm_89 -o src/t04_gemm_naive/cuda/gemm_naive \
  src/t04_gemm_naive/cuda/gemm_naive.cu -lcublas
./src/t04_gemm_naive/cuda/gemm_naive 512 512 512
ncu --set full -k gemmNaive -o docs/evidence/T04/t04-cuda-ncu \
  ./src/t04_gemm_naive/cuda/gemm_naive 512 512 512
cuobjdump -sass src/t04_gemm_naive/cuda/gemm_naive
nsys profile --trace=cuda,nvtx,osrt -o docs/evidence/T04/t04-nsys \
  ./src/t04_gemm_naive/cuda/gemm_naive 512 512 512
```
