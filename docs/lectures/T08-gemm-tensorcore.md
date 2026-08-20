# T08 GEMM Tensor Core 与 CUTLASS（唯一主讲义）

- Ticket：T08
- 状态：`done`（学习者验收通过）
- 唯一学习变量：**Tensor Core mma 路径与 CUTLASS 官方 example**
- 环境：gpp-core（PyTorch/Triton） / 系统 nvcc（CUDA WMMA） / gpp-cutile（cuTile tf32） / gpp-cute（CuTe DSL）+ CUTLASS 官方示例
- 官方来源：S01h、S02h、S02i、S03h、S18d（`config/source-ledger.md`）
- 跨 Ticket 术语：`docs/CONCEPTS.md`
- 本节导读：**一句话目标**——让 GEMM 真正用上 Tensor Core，并用 CUTLASS 官方 example 建立 fp16/tf32 基线；**依次学到**——①Tensor Core 与 CUDA Core 的区别；②fp16/bf16/tf32 的数值与格式；③CUDA WMMA 的 fragment/mma_sync；④Triton/cuTile/CuTe 怎么触发 tensor core；⑤CUTLASS 在工业界的定位；**学完应能回答**——为什么 fp16 GEMM 能到 18 TFLOPS 而 fp32 手写只有 3 TFLOPS？WMMA 与 CUTLASS 的关系？；**相关工具/技术**——PyTorch fp16/bf16、CUDA WMMA、Triton tl.dot、cuTile ct.tfloat32、CuTe DSL tensorop、CUTLASS、cuBLAS 后端、NCU Tensor 管道。
- 本节内容：**要解决的问题**——T01–T07 的手写 fp32 GEMM 离峰值还远，AI 推理/训练实际用的是 fp16/bf16/tf32 且靠 Tensor Core；**核心手段**——Tensor Core 一次算 16×16×16 的小矩阵乘加（HMMA），fp16/bf16 输入 + fp32 累加；**怎么实现**——CUDA WMMA（官方 cuda-samples 写法）、Triton `tl.dot`（fp16）、cuTile `ct.tfloat32`、CuTe DSL 官方 ampere tensorop、CUTLASS 官方 tf32 example；**怎么验证**——各路径与 fp32/量化后参考比较；NCU Tensor 管道占比；SASS 出现 `HMMA.16816`；**最终结论**——Tensor Core 是 fp16/bf16 时代的 GEMM 底座；不同工具都只是它的不同封装，CUTLASS 是最完整的工业模板库。

## 1. 上一轮问题回答

T07 已验收。T07 的 cp.async 流水线是为 T08 铺路：Tensor Core 的 mma 也需要数据及时到达，
多 stage 搬运是标准配置。

## 2. 规范实现与官方来源

| 路径 | 官方依据 |
| --- | --- |
| PyTorch | `a @ b`（fp16/bf16，cuBLAS 后端） |
| CUDA | NVIDIA cuda-samples `bf16TensorCoreGemm` 的 WMMA 写法 |
| Triton | 官方 tutorial 03 的 fp16 matmul |
| cuTile | 官方 `MatMul.py` 的 `ct.tfloat32` |
| CuTe DSL | 官方 `cute/ampere/kernel/dense_gemm/tensorop_gemm.py` |
| CUTLASS | 官方 `14_ampere_tf32_tensorop_gemm`（CMake 构建） |

## 3. 本轮实现结果

正确性：
- PyTorch fp16 512/1024、bf16 512/1024：CORRECT_PASS（bf16 容差 0.08）
- CUDA WMMA bf16：对 **bf16 量化输入后的 fp64 参考** max_err 2e-4/1.3e-3，CORRECT_PASS
- Triton fp16：CORRECT_PASS；cuTile tf32：CORRECT_PASS
- CuTe DSL 官方 ampere tensorop：512³ 20.64 us，PASS
- CUTLASS 官方 tf32：512³ 1.48 TF、1024³ 4.78 TF，Passed

性能（512³ / 1024³）：
- PyTorch fp16：8.96 / 12.3 TFLOPS
- Triton fp16：9.84 / 18.5 TFLOPS（NCU：Tensor 管道最高，23.3%）
- CUDA WMMA bf16：7.81 / 9.28 TFLOPS（SASS：`HMMA.16816.F32.BF16`）
- cuTile tf32：正确性通过；CUTLASS tf32：1.48 / 4.78 TF

## 4. 核心代码与逐行解释

### 4.1 CUDA WMMA（官方 cuda-samples 的 simple_wmma 形态）

```cuda
wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> acc_frag;

wmma::fill_fragment(acc_frag, 0.0f);
for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, A + tile_row*16*K + k, K);
    wmma::load_matrix_sync(b_frag, Bc + tile_col*16*K + k, K);  // B 用列主序（官方要求）
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);          // 一条 HMMA
}
wmma::store_matrix_sync(C + ..., acc_frag, N, wmma::mem_row_major);
```

- fragment = 每个线程手里的一小块矩阵寄存器布局；`16,16,16` 是 mma shape。
- `load_matrix_sync` 由整个 warp 协作加载 tile；`mma_sync` 是 Tensor Core 指令；
- A row-major + B col-major 是官方 sample 的约定；B 内存要按 `Bc[n,k]=B[k,n]` 布局。
- SASS 证据：`HMMA.16816.F32.BF16`。

### 4.2 Triton / cuTile / CuTe / CUTLASS 各怎么触发 Tensor Core

**Triton（官方 tutorial 03 的 fp16 tiled kernel）**

```python
acc = tl.zeros((BM, BN), dtype=tl.float32)
for k0 in range(0, tl.cdiv(K, BK)):
    a = tl.load(ap, mask=ok[None, :] < K - k0 * BK, other=0.0)   # fp16 tile
    b = tl.load(bp, mask=ok[:, None] < K - k0 * BK, other=0.0)
    acc += tl.dot(a, b)      # 编译器把 fp16×fp16→fp32 映射为 Tensor Core mma
```

逐行：`tl.load` 搬 fp16 tile；`tl.dot` 不指定 `input_precision="ieee"`（T05 的 fp32 精度开关），
编译器自动选择 mma；累加器保持 fp32。NCU 证据：`Tensor is the highest-utilized pipeline (23.3%)`。

**cuTile（官方 MatMul.py 的 tf32 开关）**

```python
dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype   # 官方示例原句
a = ct.load(A, index=(bidx,k), shape=(tm,tk), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
b = ct.load(B, index=(k,bidy), shape=(tk,tn), padding_mode=ct.PaddingMode.ZERO).astype(dtype)
acc = ct.mma(a, b, acc)   # tf32 mma：Tensor Core
```

逐行：`ct.tfloat32` 把 fp32 输入转成 tf32 格式；`ct.mma` 依据 dtype 选择 Tensor Core 路径；
`acc` 保持 fp32。

**CuTe DSL（官方 Ampere tensorop_gemm.py）**

```bash
python tensorop_gemm.py \
  --mnkl 512,512,512,1 --atom_layout_mnk 2,2,1 \
  --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \
  --a_major m --b_major n --c_major n
```

逐项：`mnkl`=M/N/K/batch；`atom_layout_mnk=2,2,1` 配置 mma atom 布局；
`ab_dtype/acc_dtype` 指定 fp16 输入 + fp32 累加；`a_major/b_major/c_major` 指定矩阵主序。
本机实测 512³ `Execution time 28.77us`、PASS（路径文件 `src/t08_gemm_tensorcore/cute_gemm.py`）。

**CUTLASS（官方 14_ampere_tf32_tensorop_gemm）**

- 构建：`cmake -S third_party/cutlass -B /tmp/cutlass-build -DCUTLASS_NVCC_ARCHS=89 -DCUTLASS_ENABLE_EXAMPLES=ON`
- 运行：`./14_ampere_tf32_tensorop_gemm --m=1024 --n=1024 --k=1024 --iterations=20`
- 代码本质：实例化 `cutlass::gemm::device::Gemm`（tf32 × tf32 + fp32 累加），模板参数决定
  tile shape、stage 数、mma 配置和 epilogue。
- 实测：512³ 1.48 TF、1024³ 4.78 TF，Passed。

## 5. 核心知识点要点

### 5.1 Tensor Core 与 CUDA Core 的区别

- CUDA Core：一个指令算标量/向量 FFMA；
- Tensor Core：一个指令算 `16×16×16` 小矩阵乘加（HMMA），吞吐是标量的几十倍；
- fp16/bf16/tf32 输入 + fp32 累加是主流配置。

### 5.2 fp16 / bf16 / tf32 数值格式

- fp16：5 位指数 + 10 位尾数，精度较高但范围小；
- bf16：8 位指数 + 7 位尾数，范围和 fp32 一样，精度低；
- tf32：19 位（10 位尾数），范围同 fp32，用于 fp32 GEMM 加速。
- 所以 bf16 容差要比 fp16 大；本 Ticket 分别用 0.02/0.08 实测。

### 5.3 WMMA 三件套与 fragment

load_matrix_sync → mma_sync → store_matrix_sync；fragment 是寄存器里的矩阵分块。
B 用 col-major 是官方 sample 约定，不是随意选择。

### 5.4 工具层定位

| 工具 | Tensor Core 入口 |
| --- | --- |
| CUDA | WMMA / mma.sync（手写） |
| Triton | tl.dot（编译器选 mma） |
| cuTile | ct.mma + tfloat32 |
| CuTe DSL | 官方 tensorop_gemm 的 atom_layout |
| CUTLASS | 模板化完整 GEMM（工业最强模板） |
| cuBLAS | 库调用 |

### 5.5 NCU 怎么看 Tensor Core 用得怎么样

- Triton NCU：`Tensor is the highest-utilized pipeline (23.3%)`；
- SASS：`HMMA.16816.F32.BF16`；
- 对比 T04 手写 fp32 只有 FFMA、FP32 峰值 6%，Tensor Core 路径的 GFLOPS 显著更高。

### 5.6 CUTLASS 定位与 T09 前置

CUTLASS 把 T05–T07 学的 tiling/smem/cp.async/mma 模板化；官方 example 一行模板实例
就是完整 kernel。T09 回到 Transpose，用二维索引/共享内存继续积累“访存类算子”经验。

## 6. 性能分析

见 §3。注意 PyTorch fp16 与 Triton fp16 的差距来自库实现与 tile 配置；WMMA 的简单
全局版只 7.8–9.3 TF，因为它没有 T05–T07 的 smem 分块/流水线——这正是 CUTLASS 存在的意义。

## 7. Memory/Compute/Latency-Bound 判断

- Tensor Core 路径明显偏 **Compute-Bound**（Triton 1024³ 18.5 TF；Tensor 管道最高）；
- WMMA 简单版仍有访存等待（Compute 17.8% in NCU full），属实现不充分而非硬件上限；
- 判定流程见 `docs/CONCEPTS.md` §2。

## 8. 知识点完整性检查

已覆盖：Tensor Core 原理、fp16/bf16/tf32、WMMA、HMMA SASS、四工具 Tensor Core 入口、
CUTLASS 定位、NCU Tensor 管道。
后置：T09 Transpose 二维索引。

## 9. 过关问题及答案（17 题，一问一答）

**A 基础**

**Q1.** Tensor Core 一个 mma 指令在 WMMA 里算什么 shape？与 CUDA Core 的 FFMA 有什么区别？

**A1（回答）**： WMMA 的 `mma_sync` 是 16×16×16：一次计算 `D(16×16) += A(16×16) × B(16×16)`，即 M=16、N=16、K=16。CUDA Core 的 FFMA 一次只做一个标量乘加，Tensor Core 一次做 16×16×16 的小矩阵乘加，吞吐高一个数量级。

**Q2.** fp16/bf16/tf32 各有多少尾数位？为什么 bf16 容差更大？

**A2（回答）**： fp16 尾数 10 位；bf16 尾数 7 位；tf32 尾数 10 位（tf32 的“19 位”指格式总位数：1 符号 + 8 指数 + 10 尾数，不是 19 位尾数）。尾数越少，能表示的相对精度越低，量化舍入越大，所以 bf16 的容差最大。

**Q3.** WMMA 三件套是什么？fragment 是什么？

**A3（回答）**： WMMA 三件套：`load_matrix_sync` 把内存数据装进 fragment，`mma_sync` 在 fragment 上做矩阵乘加，`store_matrix_sync` 把结果 fragment 写回内存。fragment 是每个线程寄存器里持有的矩阵分块，线程间分布方式由硬件约定。

**Q4.** 官方 cuda-samples 的 WMMA 为什么 B 用 col-major？我们的 B 内存做了什么变换？

**A4（回答）**：官方 simple_wmma 假设 B 列主序；我们把行主序 B 转成 `Bc[n,k]=B[k,n]` 并 leading dim K。

**B 理解**

**Q5.** Triton 源码里没有写 mma，为什么 NCU 显示 Tensor 管道最高？

**A5（回答）**：Triton 编译器把 fp16 `tl.dot` 映射为 mma；NCU Tensor 管道 23.3% 是证据。

**Q6.** cuTile 如何触发 tf32 Tensor Core？

**A6（回答）**：官方写法 `dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype`，再 `ct.mma`。

**Q7.** SASS 的 `HMMA.16816.F32.BF16` 各字段什么意思？

**A7（回答）**： `HMMA` = Half-precision Matrix Multiply-Accumulate（矩阵乘加指令）；`16816` 是 shape 16×8×16（M=16、N=8、K=16）；`.F32` 表示累加器是 fp32；`.BF16` 表示 A/B 输入是 bf16。

**Q8.** WMMA 简单版为什么只有 7.8–9.3 TF，而 Triton fp16 能到 18.5 TF？

**A8（回答）**：WMMA 简单版无 smem 分块/多 stage/双缓冲，访存与 tile 配置弱；Triton 编译器自动
   做了 T05–T07 的优化。

**C 应用**

**Q9.** 如果 B 不转列主序直接加载，会发生什么？

**A9（回答）**：fragment 布局与内存布局不匹配，得到错误结果（本项目第一次运行就是 CORRECT_FAIL，
   换列主序后才 PASS）。

**Q10.** bf16 参考应该用“量化前 fp32 输入”还是“量化后输入”？为什么本 Ticket 用后者？

**A10（回答）**：量化后输入：先证明 WMMA 指令正确，再单独讨论 fp16/bf16 的量化误差；否则无法区分
    实现错误与格式精度。

**Q11.** CUTLASS 官方 example 一行模板实例替代了我们 T05–T07 的哪些工作？

**A11（回答）**：tiling、shared memory、bank 处理、cp.async 多 stage、mma 配置和 epilogue。

**Q12.** 为什么说 Tensor Core 路径是 Compute-Bound，而 T04 朴素 fp32 是指令/L1 瓶颈？

**A12（回答）**：Tensor Core 把大部分指令变成 HMMA 数学，计算管道成为瓶颈；T04 的 FFMA 占比低、
    LDG/IMAD 高，是指令/访存瓶颈。

**Q13.** 把 WMMA tile 从 16×16 改成 32×8 会怎样？官方 sample 是怎么组织的？

**A13（回答）**：WMMA 标准 shape 是 16×16×16（不同指令集另有 8/32 变体）；官方 sample 用多 warp
    组织更大的 tile，而不是改单条 mma shape。

**Q14.** 在 NCU 里哪个输出最能证明 Tensor Core 被使用？

**A14（回答）**：NCU 的 `Tensor is the highest-utilized pipeline` 或 SASS 的 `HMMA`。

**Q15.** T09 要学什么？它与 Tensor Core 有什么关系？

**A15（回答）**：T09 Transpose：二维索引/共享内存/bank conflict 的组合；与 Tensor Core 无直接关系，
    是阶段二继续打基础。

**Q16.** cuTile 里 `ct.tfloat32` 起了什么作用？如果输入本来就是 fp16，还需要它吗？

**A16（回答）**：`ct.tfloat32` 把 fp32 输入转换成 tf32 格式，从而让 `ct.mma` 走 Tensor Core；如果输入
    本来就是 fp16，按官方示例的写法 `dtype = A.dtype` 就不需要转换，fp16 直接进 mma。

**Q17.** CuTe 官方命令里的 `--atom_layout_mnk 2,2,1` 和 `--acc_dtype Float32` 分别配置了什么？

**A17（回答）**：`atom_layout_mnk` 配置 mma atom 的 M/N/K 布局（2,2,1 是官方 Ampere 示例配置）；
    `acc_dtype Float32` 指定累加器用 fp32，避免 fp16 累加误差。
## 10. 本轮停止点

完成：五路径 Tensor Core、CUTLASS/CuTe 官方示例、WMMA SASS HMMA、NCU Tensor 证据、
讲义 15 题。
未做：T09 Transpose。

## 11. 下一最小增量

T09 Transpose 朴素版：二维线程布局与矩阵索引映射，为共享内存分块和 bank conflict 再做一轮练习。

## 附录：可复现命令

```bash
bash scripts/run_t08_all.sh
nvcc -O3 -arch=sm_89 -o src/t08_gemm_tensorcore/cuda/wmma_bf16 \
  src/t08_gemm_tensorcore/cuda/wmma_bf16.cu
cuobjdump -sass src/t08_gemm_tensorcore/cuda/wmma_bf16 | grep HMMA
```
