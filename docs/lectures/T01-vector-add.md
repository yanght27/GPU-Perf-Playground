# T01 Vector Add（五路径基线，唯一主讲义 v2）

- Ticket：T01
- 状态：`done`（学习者验收通过）
- 唯一学习变量：CUDA 的 **grid / block / thread 执行模型**，以及用五种工具写出同一个向量加法
- 环境：gpp-core（PyTorch/Triton）、系统 nvcc（CUDA）、gpp-cutile（cuTile）、gpp-cute（CuTe DSL）
- 官方来源：S01a、S02a、S03a、S10a、S15a、S18a（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——用五种工具写出同一个向量加法，并真正理解 GPU 的 grid/block/thread 执行模型；**依次学到**——①五工具各自是什么；②CUDA kernel 编译运行流程；③全局下标公式；④warp/SM/内存层次；⑤NCU/SASS 第一次分析；**学完应能回答**——`i = blockDim.x*blockIdx.x+threadIdx.x` 为什么一一映射？vector add 为什么 Memory-Bound？；**相关工具/技术**——PyTorch、CUDA C++、Triton、cuTile、CuTe DSL、NCU/SASS。
- 本节内容：**要解决的问题**——没有 AI Infra 经验，需要从零建立 CUDA 整体认知与五工具开发流程；**核心手段**——用最简单的 `c=a+b` 作为载体，五条路径写同一算法；CUDA 路径讲 grid/block/thread、launch、内存拷贝；**怎么实现**——`src/t01_vector_add/` 五路径 + `scripts/run_t01_all.sh`；**怎么验证**——五路径与 fp64 参考误差为 0，NCU 五路径 kernel 时长≈35.5us，SASS 见 LDG/STG；**最终结论**——工具抽象不同，但 kernel 本身几乎一样快；差异主要在开发流程与 launch 开销。

## 1. 上一轮问题回答

上一轮学习者指出：核心知识讲解和过关问题太少、太简，小白理解不了。本版已重写 §5
（知识点）与 §9（过关问题及答案，一问一答），并按“当前增量涉及什么就讲清什么，同时为后续
增量铺路”的原则标注知识地图。

## 2. 规范实现与官方来源

| 路径 | 对齐的官方文件 | 版本/commit |
| --- | --- | --- |
| PyTorch | 官方文档 `torch.Tensor.add` | 2.13 |
| CUDA C++ | NVIDIA cuda-samples `cpp/0_Introduction/vectorAdd/vectorAdd.cu` + Programming Guide「Kernels / Thread Hierarchy」 | cuda-samples `b7c5481c` |
| Triton | 官方 tutorial `python/tutorials/01-vector-add.py` | tag `v3.7.1` |
| cuTile | 官方 Quick Start `samples/quickstart/VectorAdd_quickstart.py` | repo `29444e0c` |
| CuTe DSL | CUTLASS 官方示例 `examples/python/CuTeDSL/experimental/primitives/tutorial/07_vectorized_array.py` | CUTLASS `564d267e` |

## 3. 本轮实现结果

N = 2^20 = 1,048,576 个 float32。所有路径 max_abs_err = 0（正确性 PASS）。

| 路径 | 黄金参考 | max_abs_err | 结论 |
| --- | --- | --- | --- |
| PyTorch fp32 | PyTorch fp64 再转 fp32 | 0.0 | CORRECT_PASS |
| CUDA C++ | CPU double 参考 | 0.0 | CORRECT_PASS |
| Triton | PyTorch fp64 再转 fp32 | 0.0 | CORRECT_PASS |
| cuTile | CuPy fp64 再转 fp32 | 0.0 | CORRECT_PASS |
| CuTe DSL | PyTorch fp64 再转 fp32 | 0.0 | CORRECT_PASS |

一键复现：`bash scripts/run_t01_all.sh`。证据目录：`docs/evidence/T01/`。

## 4. 核心代码与逐行解释

### 4.1 PyTorch：先建立“什么是对的”

```python
def pytorch_vadd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.add(a, b)
```

- `torch.Tensor`：多维数组 + 所在设备（CPU 或 `cuda:0`）的抽象。小白阶段可以把它理解成
  “放在显卡内存里的 numpy 数组”。
- `torch.add(a, b)`：我们只写数学语义，PyTorch 的 dispatcher 在幕后挑选/生成 CUDA kernel。
- 黄金参考 `(a.double() + b.double()).float()`：用 fp64 先算“接近数学真值”的结果，再转回
  fp32 比较，从而区分“实现错误”和“fp32 舍入误差”。

### 4.2 CUDA C++：第一次看见 kernel

```cuda
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
// host 侧启动：
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```

- `__global__`：这个函数是 **kernel**：在 GPU 上执行、由 CPU 启动。
- `blockDim.x * blockIdx.x + threadIdx.x`：本轮最重要的公式，详见 §5.3。
- `if (i < n)`：边界保护，N 不被线程数整除时防止越界。
- `<<<grid, block>>>`：launch 配置，决定产生多少并行工作。
- 配套动作：`cudaMalloc`（CPU 与 GPU 内存独立）→ `cudaMemcpy`（拷入/拷出）→ 编译运行。

### 4.3 Triton：从“线程”上升到“program + 向量”

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

- `tl.program_id(axis=0)` 对应 CUDA 的 `blockIdx.x`；Triton 帮你管理 block 内线程，
  所以你看不到 `threadIdx`。
- `tl.arange` 生成一个下标向量：Triton 操作“整块数据”而不是单个元素。
- `mask` + `tl.load/tl.store`：Triton 的边界保护方式。

### 4.4 cuTile Python：从“向量”上升到“tile”

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)
```

- `ct.bid(0)` 概念上仍是 block 编号。
- `ct.load(..., shape=(tile_size,))` 一次取一个 tile（数据块）；`a_tile + b_tile` 是
  整个 tile 的元素级加法，线程映射由编译器自动决定。

### 4.5 CuTe DSL：显式布局与向量化切片

```python
@cute.kernel
def vector_add_kernel(a_arr, b_arr, c_arr, vector_size: cutlass.Constexpr[int]):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bdx, _, _ = cute.arch.block_dim()
    idx = (bx * bdx + tx) * vector_size
    c_arr[idx:vector_size] = a_arr[idx:vector_size] + b_arr[idx:vector_size]
```

- `thread_idx/block_idx/block_dim` 显式对应 `threadIdx/blockIdx/blockDim`。
- 每个线程一次处理 `vector_size=4` 个元素（128-bit 向量访存），切片语法会被编译成
  `ld.global.v4/st.global.v4`（T03 展开）。

## 5. 核心知识要点（零基础版，T01 全部讲清）

### 5.1 先建立 CUDA 整体认知：GPU 到底和 CPU 有什么不同

CPU 的目标是**低延迟**：少量很强的大核，擅长处理复杂、有分支、互相依赖的逻辑。
GPU 的目标是**高吞吐**：几千个较简单的小执行单元同时工作，擅长“同一种简单计算，对
海量数据各做一份”。

- vector add 就是最典型场景：1M 个元素互不依赖，最理想的做法是让很多执行单元同时算。
- CUDA 的写法是 **SPMD（Single Program, Multiple Data）**：你只写**一份** kernel 代码，
  它会被几百万个线程同时执行；每个线程用内建索引算出“我该处理第几个元素”。
- CPU 是 host（主机），GPU 是 device（设备）。`cudaMalloc/cudaMemcpy/kernel launch` 都是
  CPU 向 GPU 下达的命令；两者的内存是独立的物理空间。

一句话总结：**CUDA 编程 = 写一份“每个数据点该做什么”的 kernel + 告诉 GPU 用多少
线程、按什么层级组织起来执行。**

### 5.2 CUDA 编译运行流程（每一步都有对应命令）

以我们的 `vector_add.cu` 为例：

```text
vector_add.cu
   │  ① nvcc 预处理/分离 host 与 device 代码
   ▼
device 部分 ──② 编译成 PTX（虚拟 ISA）──③ 再编译成 SASS（sm_89 真实机器指令）
   │                                        │
   │  SASS 打进 cubin                        │
   ▼                                        ▼
host 部分（C++） + 内嵌 cubin（fatbin） → 链接成 Linux 可执行文件 vector_add
   │  ④ 运行可执行文件
   ▼
CUDA runtime/driver 把 cubin 加载进 GPU context
   │  ⑤ CPU 发出 kernel launch + 参数
   ▼
GPU 硬件调度器把 grid 里的 block 分派到各个 SM 执行
```

- 我们用的原生命令：`nvcc -O3 -arch=sm_89 -o vector_add vector_add.cu`。
  `-arch=sm_89` 的意思是“请为 Ada 架构的 8.9 计算能力生成代码”。
- **PTX**：跨代兼容的“中间汇编”，像 Java 字节码；**SASS**：具体到某代 GPU 的机器指令。
  同一份 PTX 可以在未来架构上被驱动 JIT 成 SASS，所以我们能同时看到 `cuobjdump -sass`
  和 `nvdisasm` 两种视角。
- 验证证据：`docs/evidence/T01/t01-cuda-sass.txt`（SASS）、`t01-cuda-nvdisasm.txt`（反汇编）。
- Triton/cuTile/CuTe 没有显式 `nvcc`，它们在**运行时**把 Python kernel JIT 编译成
  PTX/cubin 并缓存；所以这些路径第一次运行较慢，benchmark 前必须 **warmup**。

### 5.3 Block / Grid / Warp / Thread：本轮核心学习变量

CUDA 把一次 kernel 的执行组织成三层：

| 层级 | 英文 | 是谁 | 内建变量 | 本例数值 |
| --- | --- | --- | --- | --- |
| 线程 | thread | 最小执行单位，算一个数据点 | `threadIdx.x` | 每个 block 256 个 |
| 块 | block | 一组线程，被整个调度到一个 SM | `blockIdx.x`, `blockDim.x` | 4096 个 block |
| 网格 | grid | 本次 launch 的全部 block | `gridDim.x` | grid=4096 |
| 束 | warp | block 内每 32 个线程组成的调度单位 | 无直接变量 | 每 block 8 个 warp |

**为什么公式能“一个元素只被一个线程处理”**：

```text
全局编号 i = blockDim.x * blockIdx.x + threadIdx.x
例如：第 100 个 block、第 3 个线程 → i = 256*100 + 3 = 25603
```

- `blockIdx` 保证不同 block 的起始段不重叠；
- `threadIdx` 保证 block 内每个线程各取一个偏移；
- 两者拼起来就是 [0, N) 的一对一编号。**没有这个公式，你就不知道每个线程该干哪份活。**

**warp 与 SIMT**：GPU 不是让 32 个线程自由行动，而是把一个 warp 的 32 个线程
“同一步调”地发射同一条指令（SIMT）。如果 warp 内不同线程走了 if 的不同分支，硬件只能
两条路都执行、再各取所需——这叫 divergence，会浪费算力（T05 会碰到）。
本 kernel 的 `if (i < n)` 在绝大多数 warp 中要么全真要么全假，divergence 影响很小。

**把数字放回本机**：每个 SM 最多同时驻留 1536 个线程，本 kernel 每 block 256 线程 →
每个 SM 最多放 6 个 block；36 个 SM 一次最多驻留 216 个 block；grid 有 4096 个 block →
大约 4096/216 ≈ 19 波（wave）。这些数字不是背的，而是解释“为什么 grid 不是越大越好、
也不是随便设”的起点（T02 继续）。

### 5.4 GPU 硬件结构：这些 thread/block 最终跑在哪

把 GPU 想象成一栋楼：

```text
GPU 芯片
 ├─ 多个 SM（流式多处理器）          ← 一栋楼里的“车间”，本机 36 个
 │   ├─ CUDA Core（算术单元）         ← 车间里的“工人”，做加减乘除
 │   ├─ warp scheduler                ← “班组长”，每次挑一个 warp 发指令
 │   ├─ register file（每 SM 64K×32bit）← 工人的“随身草稿纸”，最快
 │   ├─ shared memory / L1（每 SM 约 100 KB）← 车间公共黑板
 ├─ L2 cache（本机实测 32 MB）        ← 整栋楼的公共缓存
 └─ Global Memory / DRAM（本机 8 GB） ← 仓库（离车间最远、容量最大）
```

本机实测参数（torch 读取，A 级证据）：

| 参数 | 实测值 |
| --- | --- |
| GPU | RTX 4070 Laptop，sm_8.9（Ada） |
| SM 数量 | 36 |
| warp size | 32 |
| 每 SM 最大线程数 | 1536 |
| 每 block 最大线程数 | 1024 |
| 每 SM 寄存器 | 65536 × 32bit |
| 每 SM shared memory | 102400 B（约 100 KB） |
| L2 cache | 33554432 B（32 MB） |
| 显存 | 8585216000 B（≈8187 MiB） |

为什么这个结构决定性能：数据在**离计算单元近的地方**，访问快但容量小；在**远的地方**，
容量大但访问慢。性能优化 80% 的功夫是在安排数据离 SM 更近（T05 shared memory、T07 tiling
都会回到这张表）。

### 5.5 存储体系：Global / Shared / Constant Memory 到底指什么

| 存储器 | 作用域 | 容量（数量级） | 速度（数量级） | 谁分配 | 本轮/T 后的安排 |
| --- | --- | --- | --- | --- | --- |
| register | 单个线程私有 | 每 SM 64K×32bit | 最快 | 编译器自动 | 自动使用，无需手写 |
| local memory | 线程私有，寄存器不够时溢出到显存 | 受显存限制 | 慢 | 编译器自动 | 知道它存在即可 |
| shared memory | block 内所有线程共享 | 每 SM 约 100 KB | 接近 L1，几十周期 | `__shared__` | **T05/T06 深讲** |
| global memory | 所有线程可见 | 8 GB | 数百周期 | `cudaMalloc` | **本轮在用** |
| constant memory | 只读、全局可见 | 64 KB | 同地址广播时很快 | `__constant__` | 概念现在懂，用到再讲 |
| L1/L2 cache | 硬件缓存，程序员不显式分配 | L1/共享合用；L2 32 MB | 介于两者之间 | 硬件管理 | NCU 里观察 |

本轮 vector add 只用到了：**register + global memory + L1/L2 自动缓存**。
- `A[i]`、`B[i]`、`C[i]` 都是 global memory 访问（SASS 里的 `LDG.E/STG.E`）。
- 相邻线程访问相邻地址时，硬件能把多次小访问合并成一次大访问（coalesced access），
  这是 T02/T03 的核心。
- shared/constant 本轮**不实现**，但你要建立这张“距离-容量-速度”地图，后面每次优化
  都是在它上面移动数据。

### 5.6 CUDA 流（stream）：GPU 是异步的

CPU 启动 kernel 后**不会等它算完**，而是把命令放进一个队列立即返回——这个队列就是
**stream**。好处是 CPU 可以继续准备下一批工作，GPU 按队列顺序执行。

- 默认 stream（stream 0）：不做任何指定时，所有命令都在同一个队列里顺序执行。
- 事件（event）：在 stream 里插一个“时间戳标记”，可以量两个标记之间的 GPU 时间；
  我们 benchmark 里的 `cudaEventRecord/cudaEventElapsedTime` 和
  `torch.cuda.Event` 就是这个。
- `cudaDeviceSynchronize()`：让 CPU 停下来等 GPU 把队列清空——所以 benchmark 和正确性
  检查里都要先同步，否则你读到的还是旧数据。
- 多 stream 可以让 memcpy 和 kernel 重叠执行，本轮不展开（NSYS 时间线到集成 Ticket
  会再遇到）。

### 5.7 Tensor Core 与 CUDA Core：两种“计算单元”

- **CUDA Core**：通用的标量算术单元，一次算一个（或几个）数。vector add 的 `a+b`
  就是 CUDA Core 在做。
- **Tensor Core**：专门做**矩阵小块乘加（D = A×B + C）**的硬件单元，一次能算一个
  4×4×4 或更大的小矩阵乘。它的出现让 fp16/bf16 GEMM 吞吐暴涨。
- 为什么 vector add 不用 Tensor Core？因为 Tensor Core 擅长“矩阵乘法”这种有大量
  复用乘加的计算，不擅长“逐元素加一个数”；把 vector add 硬塞进矩阵乘法反而绕路。
- T08（优化 GEMM / CUTLASS）会实际用到 Tensor Core；现在只要建立“两种单元、各有所长”
  的区分。

### 5.8 连续批处理（Continuous Batching）的概念预览

这是推理服务（vLLM/SGLang，T24）里的概念，idea 要求早期建立认知，这里先给直觉：

- 生成式模型逐个 token 地输出。**静态 batching**：一批请求必须同时开始、同时结束，
  短请求要等长请求，GPU 空闲。
- **Continuous Batching**：服务端维护“正在处理的请求池”，任何请求一生成完一个 token，
  立刻让出位置给新请求；新请求随时插队进入。GPU 每步都在处理尽量多的 token。
- 为什么和 CUDA 相关：每个 decode 步就是一批小矩阵乘（GEMM）；批得越满，GPU 利用率越高。
- 本轮不实现，只在概念层建立“GPU 需要被喂满工作”的直觉；T16 KV Cache、T24 框架对比
  会回头用它。

### 5.9 五种算子开发流程横向对比（本轮实际跑出来的差别）

> 工具“是什么、解决什么、什么时候选”的定位对比已集中到 `docs/CONCEPTS.md` §1，
> 后续每个 Ticket 都会更新那里的实测结论；本表只讲 T01 的开发流程差异。

| 维度 | PyTorch | CUDA C++ | Triton | cuTile | CuTe DSL |
| --- | --- | --- | --- | --- | --- |
| 写的是什么 | 数学表达式 | kernel + host 代码 | Python kernel | tile kernel | 显式布局 kernel |
| 线程管理 | 全自动 | 手写公式 | 手写 program，块内自动 | 全自动 | 半手动 |
| 编译方式 | 预先编译在框架里 | `nvcc` 显式编译 | Python JIT | Python JIT | Python JIT |
| 首次运行 | 无额外编译 | 编译一次 | JIT+缓存 | JIT+缓存 | JIT+缓存 |
| 需要什么数组对象 | torch.Tensor | C 指针 | torch.Tensor | CuPy ndarray | DLPack/Tensor |
| 官方计时方式 | CUDA event | CUDA event | `triton.testing.do_bench` | CuPy event | 见 NCU |
| 本机调用级耗时 | ~0.025 ms | ~0.025 ms | ~0.070 ms | ~0.058 ms | ~17 ms（launch 开销大） |

学习结论：**kernel 本身几乎一样快；工具差异主要在抽象层级和启动成本**。这也是后面
每个算子都做五路径对比的意义：同一算法在不同工具中的“表达成本”不同。

### 5.10 性能工具各回答什么问题（T01 版）

- **NSYS（时间线相机）**：什么时刻发生了什么——CPU 侧 CUDA API、memcpy、kernel 时间条、
  stream 重叠。本机 WSL2 只能拿到 API 级时间线（C 级限制），恢复路径见 §7.1 与附录。
- **NCU（kernel 显微镜）**：一个 kernel 里 SM 在忙什么——Duration、DRAM 带宽利用率、
  计算利用率、occupancy、stall 原因。本机完整可用（A 级）。
- **SASS（机器码）**：kernel 最终变成了哪些硬件指令——`LDG.E/STG.E/ISETP` 直接对应
  “读全局内存/写全局内存/边界比较”。
- 分析顺序：先用 NSYS 找“哪段最耗时”，再用 NCU 钻进去看“为什么”，最后读 SASS 确认
  “编译器到底生成了什么”。T01 先建立这个流程，后面每个 Ticket 都重复它。

## 6. 性能分析（实测数据）

N=2^20 float32，一次加法总访存 = 3×N×4B ≈ 12.58 MB。

> ⚠️ 修正（T02 复查时发现）：调用级时间算出的 GB/s 会被 L2 缓存复用抬高（甚至超过显存
> 理论带宽 ≈256 GB/s），只能用于比较工具调用开销；**判 Memory-Bound 只看 NCU DRAM%**。
> 详细解释见 T02 讲义 §5.9。

| 路径 | 调用级 | 调用级带宽 | NCU Duration | NCU DRAM% | NCU Compute% |
| --- | --- | --- | --- | --- | --- |
| PyTorch | 0.0256 ms | 491 GB/s | 35.46 us | 92.64 | 1.93 |
| CUDA | 0.0255 ms | 494 GB/s | 35.33 us | 92.95 | 10.04 |
| Triton | 0.0702 ms | 179 GB/s | 35.97 us | 93.03 | 2.19 |
| cuTile | 0.0584 ms | 215 GB/s | 35.46 us | 92.71 | 10.98 |
| CuTe DSL | ~17.4 ms（Python call） | 不用于比较 | 35.55 us | 92.45 | 8.11 |

- 调用级数字会波动（复现脚本某次 CUDA 显示 0.0145 ms），**稳定结论看 NCU 行**：
  五个 kernel 都是 ≈35.5 us。
- NCU Duration ≈35.5 us 时带宽 = 12.58 MB / 35.5 us ≈ 354 GB/s 量级（按 kernel 实际
  访存）；DRAM 利用率 92–93% 说明带宽几乎打满。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

- 结论：**Memory-Bound**。
- 证据链（A 级实测）：
  1. 算术强度极低：每元素 1 次加法，3 次数值访问（读 a、读 b、写 c）。
  2. NCU：DRAM Throughput 92–93%，SM Compute Throughput 只有 2–11%。
  3. 解读：显存带宽是瓶颈，SM 大部分时间在等数据；给更多计算单元也不会更快。
- 不是 Latency-Bound：本 kernel 每个元素之间没有依赖链，warp 可以靠切换隐藏延迟；
  我们也没有看到 Long Scoreboard 之类的 stall 证据，所以不强行下结论。
- 下一步优化方向（不实现，只预告）：T02 继续练元素级索引与边界；T03 用 128-bit
  向量化访存减少指令条数、更充分地吃满带宽。

### 7.1 工具取证与解释（原生命令）

```bash
# NSYS：先看整体时间线（本机 API 级；kernel 级受 WSL2 限制）
nsys profile --trace=cuda,nvtx,osrt -o docs/evidence/T01/t01-nsys \
  ./src/t01_vector_add/cuda/vector_add
nsys stats --report cuda_api_gpu_sum docs/evidence/T01/t01-nsys.nsys-rep

# NCU：再看 kernel 内部
ncu --set basic -k vectorAdd -o docs/evidence/T01/t01-cuda-ncu \
  ./src/t01_vector_add/cuda/vector_add
ncu --import docs/evidence/T01/t01-cuda-ncu.ncu-rep --page details

# SASS：最后看机器码
cuobjdump -sass src/t01_vector_add/cuda/vector_add > docs/evidence/T01/t01-cuda-sass.txt
nvcc -cubin -arch=sm_89 -o docs/evidence/T01/t01-cuda.cubin \
  src/t01_vector_add/cuda/vector_add.cu
nvdisasm docs/evidence/T01/t01-cuda.cubin > docs/evidence/T01/t01-cuda-nvdisasm.txt
```

关键实测证据：`ISETP.GE.AND`（边界判断）→ `LDG.E`（读 a/b）→ `STG.E`（写 c）。
五路径的 NCU kernel 报告：`docs/evidence/T01/t01-*-ncu-kernel.ncu-rep`。

## 8. 知识点完整性检查（当前增量 + 前置知识地图）

本轮已讲解并通过代码/工具验证：

| 知识点 | 本轮深度 | 证据 | 后续深挖 |
| --- | --- | --- | --- |
| CPU vs GPU、host/device、SPMD | 概念+代码 | 五路径运行 | - |
| CUDA 编译运行流程、PTX vs SASS | 流程+命令 | SASS 证据 | - |
| grid/block/thread 映射 | **核心，逐行+习题** | 正确性+NCU | T09 二维 |
| warp、SIMT、divergence | 概念+本 kernel 行为 | 本机 warp=32 | T05/T12 |
| SM 硬件结构、occupancy | 概念+本机实测参数 | NCU occupancy | T07/T12 |
| 存储体系六种内存 | 地图+Global 实测 | SASS LDG/STG | T05/T06 |
| CUDA stream、异步、event | 概念+benchmark 用法 | NSYS API 报告 | T19+ |
| Tensor Core vs CUDA Core | 概念区分 | - | T08 |
| 连续批处理 | 概念预览 | - | T16/T24 |
| 五工具开发流程对比 | **核心，逐路径** | run_t01_all | 每个算子重复 |
| 性能工具三件套使用 | 命令+指标解读 | NCU/SASS/NSYS | 每个算子重复 |
| Memory-Bound 判断 | 判断方法+证据 | NCU 92% | T02/T03 |

明确“只讲到概念层、不假装精通”的：shared/constant memory、多 stream 重叠、
Tensor Core 编程、连续批处理实现——分别归 T05/T06、集成 Ticket、T08、T24。

## 9. 过关问题及答案（11 题，一问一答）

**A 组：基础概念（必须全对）**

**Q1.** 用你自己的话解释 `i = blockDim.x * blockIdx.x + threadIdx.x` 三项各是什么；为什么
   N 个元素恰好每个被处理一次？

**A1（回答）**：`blockDim.x`：每个 block 的线程数（本例 256）；`blockIdx.x`：当前 block 在 grid 里的
   编号（0..4095）；`threadIdx.x`：当前线程在 block 内的编号（0..255）。block 号决定
   “从第 256*i 个元素开始”，线程号决定“段内偏移”，两者拼成唯一的全局编号 i；每个线程
   取一个不同的 i，所以恰好一一覆盖 [0,N)，不会漏也不会重复。

**Q2.** 什么是 warp？为什么说 GPU 以 warp 而不是单个线程为单位调度？

**A2（回答）**：warp 是 32 个线程组成的硬件调度单位：GPU 每次取一条指令，让这 32 个线程一起执行
   （SIMT）。以 warp 为单位调度能大幅减少取指/译码开销；这也是为什么 block 大小通常取
   32 的倍数，以及 if 分支在 warp 内会引发 divergence 浪费。

**Q3.** 画出（文字即可）`vector_add.cu` 从源码到 GPU 执行完的完整流程，并说明 PTX 和 SASS 的区别。

**A3（回答）**：流程：`.cu` → nvcc 分离 host/device → device 代码编译为 PTX（虚拟指令）→ 再编译为
   SASS（sm_89 真实指令）并打进 cubin → cubin 嵌入 host 可执行文件（fatbin）→ 运行时
   driver 加载模块 → CPU launch → GPU 调度 grid 的 block 到 SM。PTX 是跨架构中间表示，
   SASS 是具体架构机器码；所以 `-arch=sm_89` 决定生成哪代 SASS。

**Q4.** 为什么 CPU 上的数组不能直接给 kernel 用？`cudaMalloc` 和 `cudaMemcpy` 各自解决什么问题？

**A4（回答）**：CPU 内存和 GPU 显存是两块独立物理空间，CPU 指针在 GPU 上无意义。`cudaMalloc` 在显存
   分配空间；`cudaMemcpy` 把数据跨过 PCIe/系统总线从 CPU 拷到 GPU（算完再拷回）。
   如果把 CPU 指针直接传给 kernel，会读到非法地址。

**Q5.** 按“离计算单元由近到远”排列 register、shared memory、L1/L2 cache、global memory，
   并说明 vector add 本轮实际用到哪些、T05 将使用哪个。

**A5（回答）**：由近到远：register → shared memory/L1 → L2 → global memory(DRAM)。vector add 用到
   register（编译器分配的中间值）和 global memory（A/B/C 数组），经过 L1/L2 自动缓存；
   T05 会显式使用 shared memory 做 GEMM tiling。

**B 组：理解与联系（能解释就过关）**

**Q6.** PyTorch、Triton、cuTile 分别用什么抽象“替你管理线程”？这和 CUDA 手写 `threadIdx` 相比，
   你牺牲了什么、得到了什么？

**A6（回答）**：PyTorch 用 Tensor 抽象：你只写数学，全部调度/线程映射自动；Triton 用
   `program_id + arange`：你负责分块，编译器负责块内线程映射；cuTile 用 tile 抽象：
   你负责 tile 的搬运和运算，编译器负责 tile 内部并行。相对 CUDA 手写 threadIdx，
   你牺牲了细粒度控制，换来更少出错、更快开发；CUDA 手写则让你看清硬件真相。

**Q7.** `torch.cuda.Event` / `cudaEventRecord` 为什么能测 GPU 时间？为什么 benchmark 前要
   warmup 和 `synchronize`？

**A7（回答）**：Event 是在 GPU stream 里插入的时间戳；两个 event 之间的 elapsed time 是 GPU 时间线
   上的间隔，不包含 CPU 的 Python 开销。warmup 是为了让 JIT 编译/缓存/分配都稳定下来；
   synchronize 是因为 kernel 异步执行，不同步 CPU 会读到未完成的数据或提前结束计时。

**Q8.** NCU 显示 DRAM 93%、Compute 10%，判断 vector add 是哪种 Bound？如果 GPU 算力翻倍，
   它会变快吗？为什么？

**A8（回答）**：Memory-Bound：带宽利用率 93% 而计算单元只有 ~10%，瓶颈是访存。算力翻倍不会变快，
   因为数据喂不饱；应该减少访存次数或提高每次访存有效宽度（T03 向量化、T05 shared memory）。

**Q9.** Tensor Core 为什么不适合 vector add？它适合什么？我们到哪个 Ticket 才会真正用它？

**A9（回答）**：Tensor Core 是矩阵小块乘加单元，适合有大量乘加复用的 GEMM；vector add 每个元素只做
   一次加法、没有矩阵结构，用 Tensor Core 反而要绕路。T08 优化 GEMM 才会真正用 mma。

**C 组：应用与迁移（做出来才算掌握）**

**Q10.** 如果 N=1,000,000，block=256，grid 应该设多少？为什么公式里要向上取整？把
    `threadsPerBlock` 改成 1024 后，grid 变成多少？哪个配置可能更利于 occupancy？

**A10（回答）**：grid = ceil(1,000,000/256) = 3907（3906×256=999,936，差 64 个元素，必须再开一个
    block；边界判断 `if(i<n)` 让最后一个 block 只算 64 个）。block=1024 时
    grid=ceil(1,000,000/1024)=977。本机每 SM 最多 1536 线程：block=256 时每 SM 可放
    6 个 block，block=1024 时每 SM 只能放 1 个 block（1536/1024=1.5 取整为 1），
    通常 block=256 对这类小 kernel 更利于 occupancy；但最终结论要用 NCU occupancy 实测。

**Q11.** 把五条路径按“抽象层级从高到低”排序，并各用一句话说出你在该路径里负责什么、
    工具替你负责什么。

**A11（回答）**：抽象从高到低：PyTorch（你写算式）→ cuTile（你写 tile）→ Triton（你写 block）→
    CuTe DSL（你写线程+向量切片）→ CUDA C++（你写线程）。各自职责一句话见 §4/§5.9 表。
## 10. 本轮停止点

- 完成：五路径实现与官方对齐、正确性、benchmark、NCU/SASS/NSYS 证据、一键复现脚本、
  讲义 v2（知识点扩充）、11 道过关题。
- 实测 vs 受限：所有性能数字为 A 级实测；NSYS kernel 时间线仍为 WSL2 C 级。
- 未做：没有进入 T02 的实现。

## 11. 下一最小增量

T02 ReLU 标量版（五路径基线）：把学习变量换成“元素级索引、Grid 配置与边界处理”，
在 vector add 之上第一次接触分支（`x>0 ? x : 0`）和“配置如何影响 Memory-Bound 算子”。

## 附录：可复现命令

```bash
bash scripts/run_t01_all.sh

nvcc -O3 -arch=sm_89 -o src/t01_vector_add/cuda/vector_add \
  src/t01_vector_add/cuda/vector_add.cu
./src/t01_vector_add/cuda/vector_add

ncu --set basic -k vectorAdd -o docs/evidence/T01/t01-cuda-ncu \
  ./src/t01_vector_add/cuda/vector_add
cuobjdump -sass src/t01_vector_add/cuda/vector_add
nsys profile --trace=cuda,nvtx,osrt -o docs/evidence/T01/t01-nsys \
  ./src/t01_vector_add/cuda/vector_add
```
