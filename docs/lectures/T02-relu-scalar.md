# T02 ReLU 标量版（五路径基线，唯一主讲义）

- Ticket：T02
- 状态：`done`（学习者验收通过）
- 唯一学习变量：元素级 kernel 的**索引、Grid 配置与边界处理**（不引入向量化，那是 T03）
- 环境：gpp-core / 系统 nvcc / gpp-cutile / gpp-cute
- 官方来源：S01b、S02b、S03b、S10b、S15b、S18b（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——把 vector add 换成带分支的 ReLU，掌握元素级索引、Grid 配置与边界处理；**依次学到**——①ceil 除法与最后一个 block；②warp divergence 与 predication；③block=256 vs 1024 的 occupancy 差异；④各工具边界表达；**学完应能回答**——N=1,000,003 时 grid 与尾块怎么算？源码 if 为什么在 SASS 里没有分支？；**相关工具/技术**——PyTorch、CUDA C++、Triton、cuTile、CuTe DSL、NCU/SASS。
- 本节内容：**要解决的问题**——T01 只处理了“每个线程算一个元素”的加法，还没处理条件分支和不整除边界；**核心手段**——ReLU(x)=max(0,x) 作为最小带分支算子；边界用 if/mask/自动 check_bounds；分支用 SASS 看 predication；**怎么实现**——`src/t02_relu/` 五路径标量版，N=2^20 与 N=1,000,003 双测；**怎么验证**——五路径 CORRECT_PASS；NCU block256 occupancy 84%、block1024 57.8%；SASS 见 FMNMX；**最终结论**——元素级 kernel 骨架固定，配置和边界都要实测；简单分支常被谓词化，不必然 divergence。

## 1. 上一轮问题回答

T01 已验收。本轮继续遵守“零基础、知识点完整覆盖”：凡是 T02 代码实际碰到、且 T01 没
讲透的概念（Grid 配置怎么选、边界处理在各工具中的形态、if 分支与 divergence、occupancy
如何从 NCU 读），全部在本讲义讲清，并为 T03 向量化做铺垫。

## 2. 规范实现与官方来源

| 路径 | 官方依据 | 版本/commit |
| --- | --- | --- |
| PyTorch | `torch.nn.functional.relu` 官方文档 | 2.13 |
| CUDA C++ | cuda-samples `vectorAdd.cu` 的 host/device 骨架 + Programming Guide「Kernels/Thread Hierarchy/Control Flow」 | `b7c5481c` |
| Triton | 官方 tutorial 01 骨架 + language 参考 `tl.maximum`（Math Ops） | v3.7.1 |
| cuTile | 官方 `VectorAdd_quickstart` 骨架 + operations 文档 `maximum` | `29444e0c` |
| CuTe DSL | 官方 `07_vectorized_array.py` 的索引/切片骨架 | `564d267e` |

## 3. 本轮实现结果

**ReLU(x) = max(x, 0)**。输入特意设计成“每个 warp 内同时有正、负、零”（i%7==0 处为精确 0），
以检验语义和边界。

| 路径 | 对齐 N=2^20 | 未对齐 N=1,000,003 | max_abs_err | 结论 |
| --- | --- | --- | --- | --- |
| PyTorch | PASS | PASS | 0.0 | CORRECT_PASS |
| CUDA C++ | PASS | PASS | 0.0 | CORRECT_PASS |
| Triton | PASS | PASS | 0.0 | CORRECT_PASS |
| cuTile | PASS | PASS | 0.0 | CORRECT_PASS |
| CuTe DSL | PASS | PASS | 0.0 | CORRECT_PASS |

一键复现：`bash scripts/run_t02_all.sh`。证据：`docs/evidence/T02/`。

### 实测时间

| 路径 | 调用级（含 launch 开销） | 调用级带宽(2N×4B) | NCU Duration | NCU DRAM% | NCU Compute% |
| --- | --- | --- | --- | --- | --- |
| PyTorch | 0.0280 ms | 299.5 GB/s | 18.82 us | 87.68 | 2.98 |
| CUDA(block=256) | 0.0137 ms | 612.8 GB/s | 18.56 us | 88.64 | 14.31 |
| CUDA(block=1024) | 0.0174 ms | 482.2 GB/s | 18.98 us | 86.84 | 13.99 |
| Triton | 0.0440 ms | 190.5 GB/s | 18.88 us | 87.14 | 2.99 |
| cuTile | 0.0393 ms | 213.4 GB/s | 18.82 us | 88.47 | 14.62 |
| CuTe DSL | ~14.3 ms（Python call） | 不比较 | 26.02 us | 63.18 | 7.77 |

> ⚠️ 重要修正：上表中的“调用级带宽”是用**调用耗时**算的，它会因为 L2 缓存复用而高得
> 离谱（例如 612 GB/s 超过了本卡显存理论带宽 ≈256 GB/s）。所以它只用来比较“各工具调用
> 开销”，**不能当作显存带宽**。判 Memory-Bound 只能看 NCU 的 DRAM%（见 §5.9）。

**三个稳定结论**：
1. PyTorch/CUDA/Triton/cuTile 生成的 ReLU kernel 几乎一样快（≈18.8 us）；
   CuTe 标量循环版慢一些（26 us），因为它没有向量化——这正好是 T03 的动机。
2. ReLU 只读 1 个数组、写 1 个数组（2N 次访存），所以 kernel 时间大约是 vector add
   （3N 次访存）的一半（18.8 vs 35.5 us）。**访存量直接决定 Memory-Bound 算子的时间**。
3. block=256 的 occupancy 比 block=1024 高，实测也更快（见 §5.5）。

## 4. 核心代码与逐行解释

### 4.1 PyTorch

```python
def pytorch_relu(a):
    return F.relu(a)      # torch.nn.functional.relu：官方 ReLU
```

- `F.relu` 是官方算子入口；PyTorch 会调度到向量化 CUDA kernel（T03 用 NCU 看的
  `vectorized_elementwise_kernel` 就是它）。
- 黄金参考：`torch.clamp(a.double(), min=0.0).float()`。先 fp64 算真值再转 fp32，
  与 T01 同理，把“实现错误”和“fp32 舍入”分开。

### 4.2 CUDA C++

```cuda
__global__ void reluKernel(const float *input, float *output, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // 与 T01 相同：全局下标
    if (i < n) {                                      // 边界保护
        output[i] = input[i] > 0.0f ? input[i] : 0.0f; // ReLU 分支
    }
}
```

- 前两行就是 T01 的元素级骨架：**这是所有逐元素算子的公共模板**，T02 的新东西只有
  第三行的分支。
- `input[i] > 0.0f ? input[i] : 0.0f`：C 的三元表达式，翻译成人话：
  “如果这个数是正数就保持原样，否则输出 0”。
- 编译运行：`nvcc -O3 -arch=sm_89 -o src/t02_relu/cuda/relu src/t02_relu/cuda/relu.cu`。

### 4.3 Triton

```python
@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements            # 边界保护：Triton 用 mask 而非 if
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)
```

- 注意对比：CUDA 用 `if (i<n)` 保护边界；Triton 用 **mask**。两者语义等价，但 Triton
  操作的是整块向量，所以边界保护也写成向量 mask。
- `tl.maximum(x, 0.0)` 是官方 Math Op（逐元素 max），等价于 `tl.where(x > 0, x, 0.0)`。

### 4.4 cuTile Python

```python
@ct.kernel
def relu_kernel(a, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=ct.maximum(a_tile, 0))
```

- `ct.maximum(a_tile, 0)`：官方 operations 文档中的 tile 级逐元素 max。
- 边界处理：cuTile 的 `ct.load` 对“最后一个不满的 tile”自带安全处理，我们实测
  N=1,000,003 也能 PASS；host 侧仍需 `grid = ct.cdiv(n, tile_size)` 向上取整。

### 4.5 CuTe DSL

```python
@cute.kernel
def relu_kernel(a_arr, c_arr, n_elements, vector_size):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = (bx * bdx + tx) * vector_size
    for i in range(vector_size):
        if idx + i < n_elements:                 # 显式边界保护
            val = a_arr[idx + i]
            c_arr[idx + i] = val if val > 0.0 else 0.0
```

- 本轮刻意写成“每线程循环 4 次、每次处理 1 个元素”的**标量**形式，便于对照 CUDA 的
  `if(i<n)`；T03 会把它改成切片向量化（`a_arr[idx:vector_size]`）。
- `if idx+i < n_elements` 与 CUDA 的 `if (i<n)` 完全同构——不同工具，同一个边界问题。

## 5. 核心知识点要点（本轮全部讲透）

### 5.1 ReLU 是什么、为什么重要

- 数学：ReLU(x) = max(0, x)。负数归零，正数保持。
- 在神经网络里它是**激活函数**：给网络引入非线性；计算极便宜（一次比较），所以成为
  卷积网络/Transformer 的默认选择之一。
- 从算子角度，它是“比 vector add 多一个分支”的最简单算子，正好用来学：
  索引 → 边界 → 分支 → 访存。

### 5.2 元素级 kernel 的公共骨架（T01 的公式在本轮复用了）

```text
全局下标 i = blockDim.x * blockIdx.x + threadIdx.x
边界保护：i < n
工作：output[i] = f(input[i])
```

T02 新学的是：**把 f 换成什么，都还是这个骨架**。以后你看到 GEMM、Softmax 也先问
“它的 f 是什么、它的索引怎么算”。

### 5.3 边界处理：ceil 除法与“最后一个 block 只干一部分活”

- `grid = (n + block - 1) / block` 就是整数向上取整。例：N=1,000,003、block=256 →
  grid=3907；前 3906 个 block 处理 999,936 个元素，最后一个 block 只有 67 个有效元素。
- 所以边界保护不是可有可无：最后那些线程的 i 已经 ≥N，没有 `if(i<n)` 就会读写非法地址。
- 各工具的等价形态：CUDA `if(i<n)`；Triton `mask`；cuTile `ct.load` 自动处理；
  CuTe DSL 显式 `if idx+i < n`。**同一个问题，不同抽象层的表达方式**。

### 5.4 if 分支与 warp divergence：源码分支 ≠ 必然分叉

- warp 是 32 个线程一组、同一条指令前进（SIMT）。如果 32 个线程进入 if 的不同分支，
  硬件只能两条路都执行、各取所需，浪费一半算力，这就是 **divergence**。
- 但“源码里有 if”不一定产生分支指令。我们看 SASS：编译器把 `x>0 ? x : 0` 编译成了
  一条 **FMNMX**（floating max with RZ=0），根本没有 if/else 两条路——这是**谓词化/
  选择指令**，天然无 divergence。
- 结论：**简单的数据相关分支，编译器常常能用 predication 消除；真正危险的是长代码块
  的分支**。这是读 SASS 才能学到的知识，本轮证据在
  `docs/evidence/T02/t02-cuda-sass.txt`。

### 5.5 Grid 配置怎么选（本轮实测 block=256 vs 1024）

NCU 实测（同一 ReLU kernel）：

| block | grid | 理论 occupancy | 实际 occupancy | Duration | 解释 |
| --- | --- | --- | --- | --- | --- |
| 256 | 4096 | 100% | 84.12% | 18.56 us | 每 SM 可放 6 个 block=1536 线程，满 |
| 1024 | 1024 | 66.67% | 57.84% | 18.98 us | 每 SM 只能放 1 个 block=1024 线程，512 个线程位浪费 |

规则（先记住，T07/T12 再深挖）：
1. block 必须是 32 的倍数（warp 大小）；
2. block 最大 1024；
3. 每个 SM 最多 1536 线程：1024 线程的 block 放不下第二个，所以理论 occupancy 只有 66.7%；
4. 对 Memory-Bound 的小算子，occupancy 够用即可，不必迷信“越高越快”，但本例 256 恰好
   又高又快。**最终以 NCU 实测为准，不以感觉为准。**

### 5.6 Memory-Bound 的第二次证据，以及“访存量决定时间”

- ReLU 每元素：读 1 次、写 1 次 = 2N 次访存；vector add 是 3N 次。
- NCU kernel 时间：ReLU ≈18.8 us，vector add ≈35.5 us，比例 ≈ 2:3，和访存次数比例一致。
  这不是巧合：两者都几乎打满 DRAM（87–93%），时间 ≈ 移动的字节数 / 带宽。
- 计算强度依旧很低（Compute% 2–15%），所以 ReLU 仍是 **Memory-Bound**。
- 优化方向预告：T03 用 128-bit 向量化减少“指令条数和地址计算开销”，但不会减少总字节数；
  真正减少字节数要靠算子融合（T06/T14）。

### 5.7 五工具横向对比（T02 的新增观察）

| 维度 | PyTorch | CUDA | Triton | cuTile | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 边界表达 | 全自动 | `if(i<n)` | mask | `ct.load` 自动 | 显式 if |
| ReLU 表达 | `F.relu` | 三元表达式 | `tl.maximum` | `ct.maximum` | 标量 if |
| 向量化 | 自动 | 无（标量） | 自动 | 自动 | 本轮故意标量 |
| kernel 时间 | 18.82 us | 18.56 us | 18.88 us | 18.82 us | 26.02 us |

CuTe 慢的原因就是“每线程 4 次标量访问”没有合并成 128-bit 访存——T03 会把五条路径
都向量化，看差距是否消失。

### 5.8 性能工具怎么用（T02 的实战解读）

- **SASS 读分支**：`S2R` 读 CTAID/TID（block/thread 索引）→ `IMAD` 算全局下标 →
  `ISETP` 边界比较 → `@P0 EXIT`（越界线程直接退出）→ `LDG.E` 读 → `FMNMX` 做 max →
  `STG.E` 写。一套下来，§5.2 的骨架在硬件上长什么样就清楚了。
- **NCU 读 occupancy**：`Theoretical Occupancy` 由 block 大小和每 SM 资源上限算出；
  `Achieved Occupancy` 是实际运行值。两者差距大时，说明还有别的限制（本 T02 不展开）。
- **NSYS 看 API**：`cudaMalloc` 依旧是最贵的 API；kernel 时间条仍受 WSL2 限制（C 级）。

### 5.9 修正一个初学者必踩的坑：为什么“调用级 GB/s”会超过显存理论带宽

- 我们的输入/输出只有 8.39 MB，而本卡 **L2 缓存有 32 MB**（T01 §5.4 实测）。benchmark
  循环反复读同一批数组时，数据几乎全在 L2 里，`cudaEvent` 量到的是“L2 到 SM”的速度，
  不是 DRAM 带宽，所以出现 612 GB/s 这种超过显存理论带宽（≈256 GB/s）的数字。
- NCU 实测证据（`docs/evidence/T02/t02-cuda-ncu-dram-metrics.txt`）：
  - 单次 kernel `dram__bytes.sum = 4.20 MB`（只有读；写还在 L2 缓存里没落到 DRAM）；
  - `gpu__time_duration = 18.94 us` → 4.2 MB / 18.94 us ≈ 222 GB/s；
  - `DRAM% = 86.84%` → 反推理论峰值 ≈ 222/0.868 ≈ 255 GB/s，与 128-bit GDDR6 ≈256 GB/s
    一致。
- **以后判 Memory-Bound 只用 NCU 的 `dram__throughput` / `DRAM Throughput`，不用调用级
  时间反推带宽。** T01 讲义里的“491 GB/s”同样是 L2 复用的产物，本讲义在此正式修正口径。

### 5.10 T03 前置知识：合并访问（coalesced access）预告

- 一个 warp 的 32 个线程发访存请求时，硬件不是发 32 次，而是尽量**合并**成尽量少的
  32 字节 sector / 128 字节 cache line 事务。
- **合并访问**：线程 0 读地址 0、线程 1 读地址 4、线程 2 读地址 8……（连续 4B），
  32 个线程的 128B 恰好合并成一次 128B 事务。T01/T02 的 `a[i]` 就是这种模式。
- **非合并访问**：每个线程读相隔很远的地址（例如 `a[i*stride]`），硬件要发很多次小事务，
  有效带宽暴跌。
- **向量化如何放大收益**：每个线程一次读 16B（float4），一个 warp 一次读 32×16B=512B，
  只需 4 个 128B 事务，地址计算指令也变少——这就是 T03 要做的 128-bit 向量化。
- 本轮 CuTe 标量版（每线程 4 次 4B 读）DRAM 利用率只有 63%，正是因为它没有把 4 次小访问
  表达成 1 次 16B 访问。T03 改切片语法后，预期会看到 DRAM% 回到 ~88%、时间追上 18.8 us。

## 6. 性能分析

实测数据表见 §3。三条稳定结论：

1. PyTorch/CUDA/Triton/cuTile 的 ReLU kernel 时间接近（约 18.8 us），CuTe 标量版
   慢在未向量化（26 us）——这是 T03 的动机。
2. ReLU 每元素读 1 写 1（2N 次访存），kernel 时间约为 vector add（3N 次访存）的一半
   （18.8 us vs 35.5 us），再次验证“访存量决定 Memory-Bound 算子的时间”。
3. block=256 的 occupancy 高于 block=1024，实测也更快（详见 §5.5）。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

**Memory-Bound**。NCU 实测 DRAM% 为 86.84–88.64，Compute% 只有 2.98–14.62（表见 §3）；
判断依据与 vector add 相同：每元素只有 1 次比较/选择，访存量主导时间。CuTe 标量版
DRAM% 只有 63.18，属于“低效访存实现拖慢同一算子”，不是反例。L2 缓存在反复 benchmark
时会让调用级 GB/s 虚高，所以判 Bound 只看 NCU DRAM%（详见 §5.9）。

## 8. 知识点完整性检查

本轮已覆盖：元素级索引、Grid 配置选择、ceil 除法与边界处理、warp divergence 与
predication、occupancy 的 NCU 读取、Memory-Bound 第二次取证、五工具边界/分支表达对比、
SASS 分支降低为 FMNMX。
未覆盖且明确后置：128-bit 向量化（T03）、Shared Memory（T05）、Bank Conflict（T06）、
Tensor Core（T08）。
`config/coverage-matrix.md` 已同步。

## 9. 过关问题及答案（12 题，一问一答）

**A 基础**

**Q1.** 写出 ReLU 的数学定义；为什么它既是“激活函数”又是“最简单的带分支算子”？

**A1（回答）**：ReLU(x)=max(0,x)。神经网络用它引入非线性，且计算只有一次比较；从算子角度看它比
   vector add 只多一个“按元素分支”，最适合在学会索引之后第一次处理条件逻辑。

**Q2.** N=1,000,003、block=256 时 grid 是多少？最后一个 block 实际只处理多少个元素？
   没有边界保护会发生什么？

**A2（回答）**：grid=ceil(1000003/256)=3907。前 3906 个 block 覆盖 3906×256=999936 个元素；最后一个
   block 只有 1000003−999936=67 个有效元素。没有边界保护时，最后 block 的 189 个线程会
   读写数组外的地址，结果未定义/程序崩溃。

**Q3.** 为什么 block 通常取 32 的倍数，且最大不超过 1024？

**A3（回答）**：warp=32 线程，block 是 32 的倍数才不会浪费整 warp；硬件规定每 block 最多 1024 线程，
   超过无法 launch。

**B 理解**

**Q4.** 源码里写了 `if (x > 0)`，为什么 SASS 里却没有真正的分支？这规避了什么性能问题？

**A4（回答）**：编译器把它降成了 FMNMX（max(x,0) 的谓词/选择指令），没有 if/else 两条控制流，因此
   不会产生 warp divergence。这告诉我们：源码 if 不必然等于硬件分支，短小的数据相关
   条件常被谓词化。

**Q5.** ReLU 和 vector add 都 Memory-Bound，为什么 ReLU 的 kernel 时间只有 vector add 的一半？

**A5（回答）**：ReLU 每元素读 1 写 1（2N 次访存），vector add 读 2 写 1（3N 次访存）；两者都接近
   带宽上限，所以时间正比于移动字节数：18.8 us ≈ 35.5 us × 2/3。

**Q6.** NCU 里 Theoretical Occupancy 和 Achieved Occupancy 分别是什么？block=1024 时为什么
   theoretical 只有 66.67%？

**A6（回答）**：Theoretical = 按 block 大小和 SM 资源上限（1536 线程）算出的最多驻留线程比例；
   Achieved = 实际运行时的比例。block=1024 时每 SM 只能放 1 个 block（1024<1536 但放不
   下第二个），最多用 1024/1536=66.67%；实际调度还有尾波，所以 achieved 57.84%。

**Q7.** 四种“边界保护”分别长什么样：CUDA、Triton、cuTile、CuTe DSL？

**A7（回答）**：CUDA：`if (i < n)`；Triton：`mask = offsets < n` 传入 load/store；cuTile：host 侧
   `ct.cdiv` 向上取整，`ct.load` 自动处理不满 tile；CuTe DSL：`if idx + i < n_elements`
   显式判断。

**C 应用**

**Q8.** 如果要把 ReLU 换成 Leaky ReLU（负数乘 0.01），只改哪一行？各路径分别怎么改？

**A8（回答）**：CUDA：`output[i] = input[i] > 0 ? input[i] : 0.01f * input[i]`；Triton：
   `tl.where(x > 0, x, 0.01 * x)`；cuTile：`ct.where` 或等价表达式（以官方 operations 文档
   为准）；CuTe：`val if val > 0 else 0.01*val`；PyTorch：`F.leaky_relu(a, 0.01)`。
   只改“f”那一行，索引/边界骨架不动。

**Q9.** 一个 Memory-Bound kernel 的 occupancy 从 84% 提到 100%，带宽利用率已经 93%，它还会
   明显变快吗？为什么？

**A9（回答）**：不会明显变快。瓶颈是 DRAM 带宽（已 93%），occupancy 再高只是让更多 warp 在“等数据”
   上排队；要提速必须减少访存字节数或提高访存效率（向量化、复用数据）。

**Q10.** 根据 NCU 表，CuTe 标量版为什么最慢？T03 的哪个改动最可能让它追上来？

**A10（回答）**：因为 CuTe 本轮是“每线程 4 次独立标量访问”，没有合并成 128-bit 访存，DRAM 利用率只
   有 63%。T03 把它改成切片语法 `a_arr[idx:vector_size]` 后，会生成 `ld.global.v4`，
   最有希望追到 18.8 us 梯队。

**Q11.** 调用级 benchmark 算出 612 GB/s，为什么不能说明这台 GPU 的显存带宽是 612 GB/s？
    判定 Memory-Bound 应该用哪个指标？

**A11（回答）**：因为反复跑同一批数组时，8.39 MB 数据几乎全部命中 32 MB 的 L2，调用级时间量的是
   L2→SM 的速度；显存理论带宽只有 ≈256 GB/s（128-bit GDDR6）。判 Memory-Bound 要用
   NCU 的 `DRAM Throughput`（dram__throughput），它直接测显存通道利用率。

**Q12.** 用自己的话说“合并访问”是什么？一个 warp 的 32 个线程连续读 float（4B）时，硬件
    大约会合并成多少次 128B 事务？每线程改成 float4 后呢？

**A12（回答）**：合并访问 = 一个 warp 的 32 个线程访问**连续、对齐**的地址，使硬件把多次请求合并成
   尽量少的 cache-line 事务。标量 float：32 线程×4B=128B → 约 1 次 128B 事务；float4：
   32×16B=512B → 4 次 128B 事务，且地址计算指令减少，访存效率更高。
## 10. 本轮停止点

完成：五路径标量 ReLU、统一正确性（含未对齐 N）、grid 配置对比（256 vs 1024）、
NCU/SASS/NSYS 证据、讲义与过关题。
未做：T03 向量化实现（下一增量）。

## 11. 下一最小增量

T03 ReLU 向量化版：把标量逐元素访问换成 128-bit 向量化 load/store，用 NCU 看带宽利用率
和指令条数的变化，正式回答“Memory-Bound 算子为什么靠向量化提速”。

## 附录：可复现命令

```bash
bash scripts/run_t02_all.sh
nvcc -O3 -arch=sm_89 -o src/t02_relu/cuda/relu src/t02_relu/cuda/relu.cu
./src/t02_relu/cuda/relu 256      # 与 1024 对比
ncu --set basic -k reluKernel -o docs/evidence/T02/t02-cuda-ncu ./src/t02_relu/cuda/relu
cuobjdump -sass src/t02_relu/cuda/relu
nsys profile --trace=cuda,nvtx,osrt -o docs/evidence/T02/t02-nsys ./src/t02_relu/cuda/relu
```
