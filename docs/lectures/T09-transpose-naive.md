# T09 Transpose 朴素版（唯一主讲义）

- Ticket：T09
- 状态：`done`（2026-08-16 学习者验收通过）
- 唯一学习变量：**二维线程布局与读合并/写合并的取舍**
- 环境：gpp-core（PyTorch/Triton） / 系统 nvcc（CUDA） / gpp-cutile（cuTile） / gpp-cute（CuTe DSL）
- 官方来源：S01i、S02j、S03i、S15、S18e（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`
- 本节导读：**一句话目标**——先搞清“转置”这个数学操作在内存里意味着什么，再写朴素转置，理解二维线程索引与“读合并 vs 写合并”只能二选一；**依次学到**——①什么是矩阵转置、行主序内存里它改变了什么；②PyTorch/CUDA/Triton/cuTile/CuTe 五路径**核心代码**逐行讲解（工程件不占篇幅）；③二维 block/thread 到 (row,col) 的映射；④转置为什么天然有一个方向不合并；⑤readC/writeC 实测与 Memory-Bound 判定；⑥T10 shared memory tile 的动机；**学完应能回答**——B[j,i]=A[i,j] 在行主序数组里对应哪两个下标？readC 和 writeC 各在哪个访存方向合并？为什么转置是 Memory-Bound？T10 靠什么让两个方向都高效？；**相关工具/技术**——PyTorch、CUDA C++、Triton、cuTile、CuTe DSL、NCU/SASS。
- 本节内容：**要解决的问题**——矩阵转置本身没有计算量，但内存访问顺序被翻转，朴素实现必然一个方向合并、另一个方向跨行；**核心手段**——二维索引 `out[x*H+y]=in[y*W+x]`；**怎么实现**——CUDA 两个 kernel（读合并/写合并）、Triton 每元素一个 program、cuTile 官方 Transpose.py tile=1、CuTe 二维 Array 映射；**怎么验证**——三个 shape（方阵/非方阵/1×N）全 PASS；NCU MemoryWorkload；SASS LDG/STG；**最终结论**——Transpose 是 Memory-Bound，朴素版只能优化一半访存，shared memory tile（T10）才能把“跨行”搬进块内解决。

## 1. 上一轮问题回答

T08 已验收。T09 进入“访存类算子”阶段：不再增加新硬件特性，而是把 T01–T03 学到的
合并访问知识用于二维矩阵。

本轮重新验收补两件事：①从零定义“转置是什么”；②把五个路径的**核心代码逐行**讲完
（工程件不逐行，只说明职责）。

## 2. 规范实现与官方来源

| 路径 | 依据 |
| --- | --- |
| PyTorch | `torch.Tensor.t()` / `contiguous()`（S15，PyTorch 2.13 文档） |
| CUDA | cuda-samples `transpose.cu` 的 `transposeNaive`，加边界保护（S18e） |
| Triton | 官方语言 API `tl.load`/`tl.store`/`tl.program_id`（S01i，官方无专属 transpose tutorial） |
| cuTile | 官方 `Transpose.py`（S03i，T09 用 tile=1 退化为朴素） |
| CuTe DSL | 官方 thread/block 索引写法 03/07 tutorial（S02j，官方无专属 transpose tutorial） |

## 3. 本轮实现结果

正确性：512×512、513×257、1×128 三 shape，五路径全部 CORRECT_PASS。

CUDA 两方向实测（512×512，证据见 `docs/evidence/T09/`）：

| kernel | avg_ms | NCU(512) DRAM% | 说明 |
| --- | --- | --- | --- |
| readC（读合并、写跨行） | 0.0147 | 45.5% | 读方向 coalesced |
| writeC（写合并、读跨行） | 0.0099 | 65.5% | 写方向 coalesced（专用 NCU：`t09-cuda-ncu-writeC.txt`） |

PyTorch 0.127 ms（含 contiguous 额外内核）；Triton/cuTile/CuTe 正确性通过。
**结论：转置时间由“不合并的那一半访存”主导；T10 用 shared memory 消除它。**
注意：WSL2 普通计时波动大，毫秒级小数以证据文件为准，判断结论以 NCU 指标为准。

## 4. 核心代码与逐行解释

> 本节只贴**核心算子代码**并逐行讲。seed、warmup、CUDA event 计时、`verify` 等通用
> 工程逻辑不逐行占用篇幅：它们属于 T01 建立的公共工程件（`src/t01_vector_add/common.py`
> 与各文件后半段），不是 T09 的学习变量。每小节末尾用一句话说明它们在哪、干什么。

### 4.0 零基础先修：Transpose 到底是什么

**定义**：矩阵转置是把矩阵沿左上到右下的**主对角线**翻面。原来在第 i 行第 j 列的数，
放到第 j 行第 i 列：

```text
B[j, i] = A[i, j]        // 行号、列号互换
```

**形状变化**：输入 A 是 H 行 × W 列，输出 B=Aᵀ 是 **W 行 × H 列**。

**完整例子**（2 行 3 列 → 3 行 2 列）：

```text
A = | 1 2 3 |        B = Aᵀ = | 1 4 |
    | 4 5 6 |                  | 2 5 |
                               | 3 6 |

A[0,0]=1 -> B[0,0]=1（对角线不动）
A[0,1]=2 -> B[1,0]=2；A[0,2]=3 -> B[2,0]=3
A[1,0]=4 -> B[0,1]=4；A[1,1]=5 -> B[1,1]=5；A[1,2]=6 -> B[2,1]=6
```

**行主序（row-major）内存**：GPU 上的矩阵是一段连续内存，一行一行排。A 是 H×W，
`A[y,x]` 的地址是 `y*W+x`；B 是 W×H，`B[x,y]` 的地址是 `x*H+y`：

```text
A 内存：1,2,3, 4,5,6        ← A 行内连续（1→2→3，stride=1）
B 内存：1,4, 2,5, 3,6        ← B 行内连续（1→4，每行 2 个元素）
```

所以转置在内存里做的事就是：**把行内连续（stride=1）和跨行（stride=W/H）两个方向对调**。
它没有任何加减乘除，**0 FLOP，纯访存**——这就是为什么 T09 一整节都在讨论“读合并还是写合并”。

**为什么 AI Infra 天天见转置**：Attention 的 `Q@Kᵀ`（T15 会用到）、PyTorch 权重
shape 变换、T04 学过的 cuBLAS“列主序=行主序的转置”都是它。

**PyTorch 两个词的区别（第一个易错点）**：

```python
# （教学示意，理解 strides 用，不是本项目 src 文件）
a = torch.arange(6).reshape(2, 3)    # shape=(2,3)，strides=(3,1)：行内步长 1
a.t()                                 # shape=(3,2)，strides=(1,3)：只把 strides 对调，view！
a.t().contiguous()                    # shape=(3,2)，strides=(2,1)：真的复制一块新内存
```

- `a.t()`：只交换 strides 的“视图”，一个字节都不搬，因此结果通常不是 contiguous；
- `a.t().contiguous()`：按转置后的顺序真正复制出行主序的转置副本。
本 Ticket 的 CUDA/Triton/cuTile/CuTe 都实现**后者**（输出是行主序连续数组），
所以 PyTorch 参考也写 `a.t().contiguous()`。

**五个路径表达同一个公式**：

| 路径 | 表达式 |
| --- | --- |
| PyTorch | `a.t().contiguous()` |
| CUDA | `out[x * H + y] = in[y * W + x]` |
| Triton | `tl.store(y_ptr + col*H + row, tl.load(x_ptr + row*W + col))` |
| cuTile | `ct.transpose(tile)` + store 时 `index=(bidy, bidx)`（index 交换） |
| CuTe DSL | `b[col, row] = a[row, col]` |

### 4.1 路径 1：PyTorch 参考

核心代码（`src/t09_transpose_naive/pytorch_transpose.py` 第 16–20 行）：

```python
    torch.manual_seed(0)
    a = torch.rand((H, W), device="cuda", dtype=torch.float32)
    ref = (a.double().t()).float()
    out = a.t().contiguous()
    summarize_error(out, ref, f"pytorch_transpose_{H}x{W}")
```

- 第 1 行：固定随机种子，让输入可复现。
- 第 2 行：在 GPU 上生成 H 行 × W 列的 fp32 随机矩阵。
- 第 3 行：**黄金参考**：升到 fp64 再做转置，避免误差，最后降回 fp32。
- 第 4 行：**核心算子**：`.t()` 换视图、`.contiguous()` 真正复制出行主序转置矩阵。
- 第 5 行：与参考逐元素比较，打印 `max_abs_err` 与 PASS/FAIL。
- 工程件说明：同文件其余部分是 warmup 10 次 + `torch.cuda.Event` 计时 50 次，不参与转置语义。

### 4.2 路径 2：CUDA C++（核心 = 两个 kernel）

核心代码（`src/t09_transpose_naive/cuda/transpose.cu` 第 20–29 行）：

```cuda
// 反方向：output 按行连续写（合并），input 按列读（跨行，不合并）
__global__ void transposeWriteCoalesced(const float *in, float *out, int W, int H)
{
    // 写合并方向：相邻线程对应 output 的相邻列 y；input 读地址 stride=W（跨行读）
    int y = blockIdx.x * blockDim.x + threadIdx.x;   // output 列
    int x = blockIdx.y * blockDim.y + threadIdx.y;   // output 行
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];   // B[x,y] = A[y,x]
    }
}
```

第一个 kernel `transposeReadCoalesced`（文件第 10–18 行）是同一公式的镜像：`x` 由
`threadIdx.x` 给（读方向连续），`y` 由 `threadIdx.y` 给。这里逐行讲 writeC：

- 第 1 行：注释说明这个方向的取舍：输出写合并、输入读跨行。
- 第 2 行：`__global__` 声明 GPU kernel；参数是输入/输出指针与宽 W、高 H。
- 第 4 行：`blockIdx.x*blockDim.x+threadIdx.x`：block 编号 × block 大小 + 块内线程编号，
  得到全局**输出列 y**。变化最快的 `threadIdx.x` 给 y → 相邻线程拿到相邻列。
- 第 5 行：`blockIdx.y*blockDim.y+threadIdx.y`：得到全局**输出行 x**。
- 第 6 行：边界保护：grid 向上取整产生的越界线程不读写。
- 第 7 行：核心公式。输入 `in[y*W+x]` 行主序（读跨 W）；输出 `out[x*H+y]` 行主序
  （写合并）。两个 kernel 公式完全相同，**谁连续谁合并**——这就是 T09 唯一学习变量。
- 工程件说明：`makeInputs/verify/timeKernel/main` 负责确定性输入、CPU 校验与 CUDA event
  计时，不改变 kernel 语义。

### 4.3 路径 3：Triton（核心 = kernel + 一行 launch）

```python
@triton.jit
def transpose_kernel(x_ptr, y_ptr, H, W):
    pid = tl.program_id(0)
    row = pid // W
    col = pid % W
    if row < H and col < W:
        v = tl.load(x_ptr + row * W + col)   # A[row,col]
        tl.store(y_ptr + col * H + row, v)   # B[col,row]
```

```python
    transpose_kernel[(H * W,)](a, y, H, W)
```

- 第 1 行：`@triton.jit` 把该 Python 函数编译成 GPU kernel。
- 第 2 行：kernel 参数是输入/输出指针、行数 H、列数 W。
- 第 3 行：`tl.program_id(0)`：当前 program 在一维 grid 中的编号，类似 CUDA 的 `blockIdx.x`；
  本 Ticket 一个 program 只处理一个元素。
- 第 4 行：`pid // W`：整除，一维编号 → 行号。
- 第 5 行：`pid % W`：取模，一维编号 → 列号。这两行与 CUDA 的 2D grid/block 映射等价。
- 第 6 行：边界保护。
- 第 7 行：`tl.load(指针 + row*W + col)`：读 `A[row,col]`（Triton 的指针加法自动乘以元素字节数）。
- 第 8 行：`tl.store(指针 + col*H + row, v)`：写 `B[col,row]`，完成转置。
- 第 9 行（launch）：`[(H*W,)]` 是一维 grid，共 H×W 个 program；每个 program 一个元素。
- 工程件说明：`triton_transpose()` 只负责分配输出与 launch，`run()` 只负责参考与校验。

### 4.4 路径 4：cuTile（核心 = tile kernel）

```python
@ct.kernel
def transpose_kernel(x, y, tm: ct.Constant[int], tn: ct.Constant[int]):
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)
```

- 第 1 行：`@ct.kernel` 声明 tile kernel。
- 第 2 行：`tm,tn` 是 tile 行/列数的编译期常量；T09 传 `(1,1)` 退化为朴素版。
- 第 3–4 行：`ct.bid(0)/ct.bid(1)`：当前 block 在 grid 两维的编号（对应 blockIdx.x/y）。
- 第 5 行：`ct.load(x, index=(bidx,bidy), shape=(tm,tn))`：以 `(bidx,bidy)` 为原点读一个
  tile。tile=1 时就是一个元素，所以是朴素版。
- 第 6 行：`ct.transpose`：对 tile 做维度交换（官方 `Transpose.py` 的核心原语）。
- 第 7 行：`ct.store(..., index=(bidy,bidx), ...)`：写回时 index 两维**交换**，完成转置。
- 工程件说明：`cutile_transpose()` 计算 grid 并 `ct.launch`；`run()` 负责 CuPy 输入、
  fp64 参考与校验。官方依据 S03i（官方 sample 的 tile=1 退化写法）。

### 4.5 路径 5：CuTe DSL（核心 = device kernel）

```python
@cute.kernel
def transpose_kernel(a: cutlass.Array, b: cutlass.Array,
                     W: cutlass.Int32, H: cutlass.Int32):
    tx, _, _ = cute.arch.thread_idx()
    _, ty, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    _, by, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    _, bdy, _ = cute.arch.block_dim()
    col = bx * bdx + tx
    row = by * bdy + ty
    if row < H and col < W:
        b[col, row] = a[row, col]
```

- 第 1 行：`@cute.kernel` 声明 device kernel。
- 第 2–3 行：参数 `a,b` 是 `cutlass.Array` 多维数组；`W,H` 是 32 位整数。
- 第 4–5 行：`thread_idx()` 返回三维线程索引；两次调用分别取 `tx`、`ty`
  （对应 CUDA 的 threadIdx.x/y）。
- 第 6–7 行：`block_idx()` 取 `bx,by`（对应 blockIdx.x/y）。
- 第 8–9 行：`block_dim()` 取 block 尺寸 `bdx,bdy`（对应 blockDim.x/y）。
- 第 10 行：`col = bx*bdx + tx`：全局列 = block 起点 + 块内偏移。
- 第 11 行：`row = by*bdy + ty`：全局行。
- 第 12 行：边界保护。
- 第 13 行：`b[col,row] = a[row,col]`：CuTe 的二维下标写法，就是转置定义本身。
- 工程件说明：`@cute.jit` 的 host 包装负责 `block=(16,16,1)`、grid 向上取整与 launch；
  `cute.runtime.from_dlpack` 把 PyTorch tensor 零拷贝交给 CuTe 运行时。

## 5. 核心知识点要点

### 5.1 二维索引映射（用具体数字走一遍）

模板公式：

```text
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```

记法：**blockDim = 每块多大；blockIdx = 第几块；threadIdx = 块内第几个线程**。
“第几块 × 每块多大 + 块内偏移”= 全局坐标。

例：`block=(16,16)`、`grid=(ceil(W/16), ceil(H/16))`。线程 `(threadIdx.x=3, threadIdx.y=2)`
在 `blockIdx=(0,0)` 时：`col=0*16+3=3`，`row=0*16+2=2`——它负责 A 的第 2 行第 3 列。
grid.x 方向铺满列、grid.y 方向铺满行；这是后面所有 2D 算子的模板。

### 5.2 转置为什么必然有一半不合并（用地址算一遍）

两个词先钉死：

- **合并访问（coalesced）**：同一 warp（32 个线程一组）的相邻线程访问相邻地址，
  一次内存事务搬回整段 cache line（128B=32 个 fp32），带宽用满。
- **stride（步长）**：相邻两次访问的地址间隔。行内相邻 stride=1；列方向相邻 stride=W。

用 `W=512` 的四线程例子（tx=0,1,2,3，ty=0）：

```text
readC：读 in[y*W+x] = in[0*512+tx] → 地址 0,1,2,3        ← 读合并
       写 out[x*H+y] = out[tx*512+0] → 地址 0,512,1024,1536 ← 写跨行

writeC：读 in[y*W+x] = in[tx*512+0] → 地址 0,512,1024,1536 ← 读跨行
        写 out[x*H+y] = out[0*512+tx] → 地址 0,1,2,3        ← 写合并
```

- 输入行内连续（stride=1）、跨行 stride=W；
- 输出行内连续（stride=1）、跨行 stride=H；
- 转置把这两个方向对调，所以朴素实现**只能保证一侧合并**：读合并则写跨行，
  写合并则读跨行。这不是代码写错，是转置的数学本质。T09 把两种选择都实现出来实测，
  就是 readC 与 writeC。

### 5.3 Memory-Bound（用实测数据把结论算出来）

转置 0 FLOP，只有读 N + 写 N = 2N 次 4B 访存。512×512 时：

```text
N = 512*512 = 262,144
移动字节 = 2 * N * 4B = 2,097,152 B ≈ 2 MiB
```

把 T09 实测时间换算成有效带宽（这是实测换算，不是预测）：

```text
writeC：2,097,152 B / 0.0099 ms ≈ 212 GB/s
readC ：2,097,152 B / 0.0147 ms ≈ 143 GB/s
```

NCU 同步印证：DRAM 45.5–65.5%、Compute 10–14%——**Memory-Bound**。L1 hit 52–61%
说明 stride 那侧在浪费 cache line：128B 能装 32 个 fp32，跨行访问往往只用其中 1 个，
其余 31 个被“浪费”搬进 cache。

### 5.4 T10 的动机

既然朴素版必然一半跨行，就换一条搬运路线：把 32×32 tile **按合并方向读进 shared memory**，
在块内完成行列对调（shared 容量小、带宽高，跨行代价小得多），再**按合并方向写回 global**。
这样 global 读和写都合并；新问题是块内转置访问 shared 可能产生 bank conflict（T10 的
学习变量，用 padding 消除）。

### 5.5 本轮术语速查（零疑问自查表）

| 术语 | 一句话解释 | 详细位置 |
| --- | --- | --- |
| 转置 / Aᵀ | B[j,i]=A[i,j]，H×W 变成 W×H | §4.0 |
| 行主序 | 内存里一行一行连续排，`A[y,x]` 在 `y*W+x` | §4.0 |
| stride | 同一方向相邻元素的地址间隔 | §4.0、§5.2 |
| contiguous | 内存与逻辑顺序一致；`a.t()` 只是换 stride 的视图，`contiguous()` 才真正复制 | §4.0、§4.1 |
| 黄金参考 | 用 fp64 算出的高精度标准答案，fp32 实现与它比误差 | §4.1 |
| warp / 合并访问 | 32 线程一组、相邻线程访问相邻地址 | §5.2 |
| grid/block/thread | 任务网格 → 每块 → 块内线程 的三级组织 | §4.2（y/x 两套三件套） |
| warmup / CUDA event | 先空跑稳定状态；GPU 时间戳计时（工程件，非学习变量） | §4 开头说明 |
| program / @triton.jit | Triton 的一次 kernel 实例与编译装饰器 | §4.3 |
| tile / ct.load / ct.store | cuTile 的“一块数据”与搬入搬出原语 | §4.4 |
| cutlass.Array / DLPack | CuTe 的多维数组与跨框架零拷贝协议 | §4.5 |

## 6. 性能分析

见 §3。两个 CUDA kernel 时间差来自“读合并 vs 写合并”在硬件上的代价差异；不要把它
当成固定结论，shape 不同会变化。本机 WSL2 普通计时噪声大，结论以 NCU 的 DRAM%/L1 hit
为准（证据 `docs/evidence/T09/t09-cuda-ncu.txt`、`t09-cuda-ncu-writeC.txt`）。

## 7. Memory/Compute/Latency-Bound 判断

**Memory-Bound**：readC DRAM 45.5%、writeC DRAM 65.5%，Compute 10–14%，L1 hit 52–61%。
证据：`docs/evidence/T09/t09-cuda-ncu.txt`、`t09-cuda-ncu-writeC.txt`。

## 8. 知识点完整性检查

已覆盖：转置定义与形状变化、行主序下标、`.t()` 与 `.contiguous()` 的区别、
五路径**核心代码**逐行讲解、二维索引、读/写合并取舍、三个 shape、NCU/SASS、Bound 判定。
工程件（seed/计时/verify）不逐行，已在 §4 开头统一说明职责。
后置：T10 shared memory tile 与 bank conflict。

## 9. 过关问题及答案（18 题，一问一答）

**A 基础**

**Q1.** 什么是矩阵转置？输入 H×W 时输出是什么形状？写出 B 与 A 的元素关系。

**A1（回答）**：转置是把矩阵沿主对角线翻面：`B[j,i] = A[i,j]`；输入 H×W 时输出是 W×H。

**Q2.** 行主序内存里 `A[y,x]` 的线性地址是什么？转置后 `B[x,y]` 的线性地址是什么？

**A2（回答）**：`A[y,x]` 在 `y*W+x`；`B[x,y]` 在 `x*H+y`（因为 B 每行有 H 个元素）。

**Q3.** PyTorch 的 `a.t()` 和 `a.t().contiguous()` 有什么区别？哪个才得到“行主序的转置副本”？

**A3（回答）**：`a.t()` 只交换 strides，是一个不搬数据的视图，通常不是 contiguous；
   `a.t().contiguous()` 才会按转置后的顺序复制出一块连续内存，是真正“转置副本”。
   本 Ticket 的 CUDA/Triton/cuTile/CuTe 输出都是行主序连续数组，所以 PyTorch 参考用它。

**Q4.** 写出 2D 线程索引到 (row,col) 的公式。

**A4（回答）**：`row = blockIdx.y*blockDim.y + threadIdx.y; col = blockIdx.x*blockDim.x + threadIdx.x`。

**Q5.** readC 和 writeC 哪个方向合并？为什么另一个不合并？

**A5（回答）**：readC 读合并（相邻 threadIdx.x 读 `in[y*W+x]` 的 x 连续）但写跨 H；
   writeC 写合并（相邻 threadIdx.x 写 `out[x*H+y]` 的 y 连续）但读跨 W。
   因为转置交换连续维度，连续输入必对应跨行输出。

**Q6.** 为什么转置是 Memory-Bound？

**A6（回答）**：无 FLOP，只有 2N 次访存；NCU 显示 DRAM 利用率高（45–66%）、Compute 低（10–14%）。

**B 理解**

**Q7.** Triton 的 `row = pid // W; col = pid % W` 与 CUDA 的 grid/block 映射有什么关系？

**A7（回答）**：都是把一维编号手工映射成二维坐标：Triton 用整除/取模把 pid 切成 (row,col)；
   CUDA 用 grid.y/grid.x × blockDim + threadIdx 直接得到 (row,col)。两者一一对应。

**Q8.** cuTile 官方 Transpose 的 `ct.transpose` 做了什么？tile=1 时为什么是朴素版？

**A8（回答）**：`ct.transpose` 对 tile 做维度交换；tile=1 时每个 tile 只有 1 个元素，
   没有块内复用，整体退化为逐元素朴素转置。

**Q9.** CuTe DSL 里的 `tx,ty,bx,by,bdx,bdy` 分别对应 CUDA 的什么？

**A9（回答）**：`tx,ty` = threadIdx.x/y；`bx,by` = blockIdx.x/y；`bdx,bdy` = blockDim.x/y；
   CuTe 把它们拆成三元组取前两位。

**Q10.** NCU L1 hit 52% 说明了什么？

**A10（回答）**：stride 访问让 128B cache line 的利用率只有约一半，说明“不合并”的那一半访存在浪费带宽。

**Q11.** 两个 CUDA kernel 公式相同，为什么性能不同？

**A11（回答）**：公式相同但线程映射不同：readC 让读合并，writeC 让写合并；GPU 对读写合并的敏感度
    和具体 shape 决定哪个更快（本例 writeC 更快）。

**C 应用**

**Q12.** 如果矩阵按列主序存储，readC/writeC 的哪个方向会反过来？

**A12（回答）**： 会反过来。列主序下“列内相邻元素连续、行间跨 N”，原来 readC 的读合并会变成跨行读，writeC 的写合并也会换边。结论：哪个下标变化最快、哪个方向连续，合并方向就跟谁。

**Q13.** 1×128 的转置为什么两个方向差别很小？

**A13（回答）**：只有一个“行”或“列”，跨行 stride 极小或访问总量小，stride 惩罚接近消失，
    两个方向几乎一样。

**Q14.** 为什么 shared memory tile 能同时让读和写合并？它增加了什么新问题？

**A14（回答）**：tile 读入 shared 时用合并方向，写出时也合并；新问题：块内转置访问 shared 可能
    bank conflict，需要 padding/swizzle（T10）。

**Q15.** 把 block 从 (16,16) 改成 (32,8)，索引公式要改吗？网格呢？

**A15（回答）**：公式不变（公式里用 blockDim，自动适配）；grid 的 x/y 块数要按新 block 尺寸重新
    向上取整：(W+31)/32 和 (H+7)/8。

**Q16.** 转置的总数据量是多少？如果带宽固定，理论最小时间怎么估？

**A16（回答）**：输入 N 个元素 + 输出 N 个元素 = 2N 次 4B 访存（512×512 时 2×262144×4≈2.1MB）；
    理论最小时间 ≈ 2N×4B / DRAM 带宽。

**Q17.** 在 NCU 里如何确认 stride 访问的代价？

**A17（回答）**：看 L1/TEX hit rate、DRAM throughput，以及 SASS 的 LDG/STG 地址模式。

**Q18.** T10 最可能先做哪个动作？

**A18（回答）**：先实现 shared memory tile 转置，再用 NCU 测 bank conflict。
## 10. 本轮停止点

完成：五路径朴素转置、三 shape、两方向 benchmark、NCU/SASS/NSYS、讲义 18 题；
本次重新验收补齐“什么是转置”、行主序/contiguous 定义，并把五路径统一为**核心代码逐行讲解**。
未做：T10 shared memory tile（已单独成 Ticket，本轮不夹带）。

## 11. 下一最小增量

T10 Transpose Tile：shared memory 分块转置 + bank conflict 消除。

## 附录：可复现命令

```bash
bash scripts/run_t09_all.sh
nvcc -O3 -arch=sm_89 -o src/t09_transpose_naive/cuda/transpose src/t09_transpose_naive/cuda/transpose.cu
# readC：--set full 会包含 SpeedOfLight / MemoryWorkloadAnalysis 等 section
ncu --set full -k 'regex:transposeReadCoalesced' -o docs/evidence/T09/t09-cuda-ncu \
  ./src/t09_transpose_naive/cuda/transpose
# writeC：单独取 SpeedOfLight 一节（本机证据 t09-cuda-ncu-writeC.txt）
ncu --section SpeedOfLight -k 'regex:transposeWriteCoalesced' \
  ./src/t09_transpose_naive/cuda/transpose > docs/evidence/T09/t09-cuda-ncu-writeC.txt
cuobjdump -sass ./src/t09_transpose_naive/cuda/transpose > docs/evidence/T09/t09-cuda-sass.txt
```
