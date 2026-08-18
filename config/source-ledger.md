# GPP 官方来源台账（对齐 PLAN v1.3）

> 规则：动手写代码/命令前必须在线核对并填写。未核对的一律保持“待核对”，禁止把记忆中的
> API 写成已核实。每条来源包含：URL、访问日期、版本/tag/commit、引用的官方文件路径、
> 服务哪个 Ticket、证据存档位置。

## 1. 已核实官方来源（T00，访问日期 2026-08-15）

| ID | 类型 | 来源 | 核实 URL | 版本/tag/commit | 官方文件引用 | 服务 Ticket |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | 官方项目+文档 | Triton | https://triton-lang.org/main/index.html （200）；https://github.com/triton-lang/triton | PyPI 最新 3.7.1；tag `v3.7.1`=`f797708c…` | `python/tutorials/`（T01 用 01-vector-add） | T01–T18 |
| S02 | 官方项目+文档 | CUTLASS | https://docs.nvidia.com/cutlass/latest/ （200）；https://github.com/NVIDIA/cutlass | main HEAD=`564d267e4c992c456d12ad02665f9acedf7708f1`；`python/CuTeDSL/requirements-cu13.txt` 锁定 `nvidia-cutlass-dsl[cu13]==4.7.0` | `examples/`、`python/CuTeDSL/` | T01–T18 |
| S03 | 官方项目+文档 | cuTile Python | https://docs.nvidia.com/cuda/cutile-python/quickstart.html （200）；https://github.com/NVIDIA/cutile-python | PyPI `cuda-tile` 最新 1.5.0；repo HEAD=`29444e0c…` | README `pip install cuda-tile[tileiras]`、docs/ | T01–T18 |
| S04 | 官方项目+文档 | DeepSpeed | https://deepspeed.readthedocs.io/en/latest/ （200）；https://github.com/deepspeedai/DeepSpeed | PyPI `deepspeed` 最新 0.19.5；repo HEAD=`9bd89f9d…` | docs/、README | T26/T28 |
| S05 | 官方项目+文档 | vLLM | https://docs.vllm.ai/en/v0.27.1/ （200）；https://github.com/vllm-project/vllm | PyPI `vllm` 最新 0.27.1；Docker Hub `vllm/vllm-openai:v0.27.1` digest 与本地镜像一致 | docs/serving/、quickstart | T20/T24 |
| S06 | 官方项目+文档 | SGLang | https://docs.sglang.io/ （200）；https://github.com/sgl-project/sglang | Docker Hub `lmsysorg/sglang:v0.5.17` digest 与本地镜像一致；repo HEAD=`e99ecb6e…` | docs/start/、docs/backend/ | T21/T24 |
| S07 | 官方项目+文档 | TensorRT-LLM | https://nvidia.github.io/TensorRT-LLM/ 与 release-notes （200）；https://github.com/NVIDIA/TensorRT-LLM | GitHub tag `v1.2.1`=`376f7e1b…`；本地 NGC 镜像 `release:1.2.1` | examples/、docs/ | T22/T24 |
| S08 | 官方项目+文档 | ms-swift | https://swift.readthedocs.io/zh-cn/latest/ （200，站点版本串 4.5.0.dev0）；https://github.com/modelscope/ms-swift | GitHub tag `v4.4.3`=`e1287928be4451b9ed5e2fb00a24ad3c8f61287b`（本环境安装 commit 与官方 tag 完全一致）；README Quick Start 行 163–180 使用 `AI-ModelScope/alpaca-gpt4-data-zh#500` | `README.md`、docs/source/、examples/train/ | T23/T27/T28 |
| S09 | 官方文档 | cuBLAS | https://docs.nvidia.com/cuda/cublas/ （200，cuBLAS 13.3 文档） | CUDA 13.3 文档线 | API Reference | T04/T08 |
| S10 | 官方文档 | CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ （200）；https://docs.nvidia.com/cuda/cuda-programming-guide/ | 当前线上版本 | Thread Hierarchy、Memory、Synchronization | T01–T18 |
| S11 | 官方文档 | CUDA「Writing Tile Kernels」 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html （200） | 当前线上版本 | 全文 | T08 |
| S12 | 官方文档 | Nsight Compute Profiling Guide | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html （200，NsightCompute 13.3） | 13.3 | Metric/Section Reference | T01–T18 |
| S13 | 官方文档 | Nsight Systems User Guide | https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | 官方 UserGuide | Timeline/CLI | T01–T18 |
| S14 | 官方 CLI+模型页 | ModelScope | `ms --help`/`ms download --help`（本地 gpp-core 的 modelscope-hub 0.1.8 官方 CLI）；模型页 https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct | CLI：`ms download --repo-type model|dataset --revision <rev> --local-dir <dir> <repo_id>` | download/inference API | T00/T19 |
| S15 | 官方文档 | PyTorch | https://pytorch.org/docs/2.13/ （200）；https://pytorch.org/get-started/locally/ | torch 2.13.0（PyPI 最新），本机 `+cu130` 轮子 | torch.cuda、SDPA、DDP/FSDP | 全程 |
| S16 | 官方文档 | HuggingFace Transformers | https://huggingface.co/docs/transformers/ （200） | 线上 latest；本机 gpp-core 锁 5.14.1 | generation、chat templates | T19 |
| S17 | 官方项目 | FlashAttention | https://github.com/Dao-AILab/flash-attention | HEAD=`145b1010…` | algorithm 说明、实现参考 | T15/T17/T18 |
| S18 | 官方仓库 | NVIDIA cuda-samples | https://github.com/NVIDIA/cuda-samples | HEAD=`b7c5481c…` | reduction、transpose、gemm 相关 sample | T03–T18 |
| S19 | 官方文档 | NVIDIA Container Toolkit | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/ （200） | 本机 nvidia-ctk 1.20.0 | install/config、GPU 容器 | T00 |
| S20 | 官方模型卡 | Qwen2.5-0.5B-Instruct | https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct（API 返回 apache-2.0、qwen2、chat） | 本地锁定 revision `master` 下载快照（文件 SHA256 见 day0-lock） | model card、tokenizer_config.json chat_template | T00/T19–T27 |

## 2. 本地实测基线（T00 已核实并写入 config/day0-lock.json）

| ID | 项目 | 实测值 | 核对状态 |
| --- | --- | --- | --- |
| L01 | GPU / driver / system CUDA | NVIDIA GPU（sm_8.9，8GB）/ 以本机为准 / 以本机为准 | 已实测并锁定 |
| L02 | Docker / nvidia-ctk | 29.1.3 / 1.20.0 | 已实测并锁定 |
| L03 | vLLM 镜像 | `vllm/vllm-openai:v0.27.1`（digest 与 Docker Hub 一致） | 已实测并锁定 |
| L04 | SGLang 镜像 | `lmsysorg/sglang:v0.5.17`（digest 与 Docker Hub 一致） | 已实测并锁定 |
| L05 | TensorRT-LLM 镜像 | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`（tag 对应官方 GitHub v1.2.1） | 已实测并锁定 |
| L06 | conda 环境与关键包 | 见 `config/day0-lock.json` | 已实测并锁定 |

## 2.5 T01 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T01 用途 |
| --- | --- | --- | --- | --- |
| S15a | PyTorch Tensor.add | https://pytorch.org/docs/2.13/generated/torch.Tensor.add.html（200） | 2.13 | PyTorch 路径与黄金参考 |
| S10a | CUDA Programming Guide：Kernels / Thread Hierarchy | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels 、#thread-hierarchy（200） | 当前线上版 | grid/block/thread 讲解 |
| S18a | NVIDIA cuda-samples vectorAdd | 本地 sparse clone `cuda-samples/cpp/0_Introduction/vectorAdd/vectorAdd.cu`（GitHub 同路径） | `b7c5481c` | CUDA C++ 路径的权威写法 |
| S01a | Triton tutorial 01-vector-add | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/01-vector-add.py | `v3.7.1` | Triton 路径逐行对齐 |
| S03a | cuTile VectorAdd_quickstart | 本地 `cutile-python/samples/quickstart/VectorAdd_quickstart.py`（GitHub 同路径） | `29444e0c` | cuTile 路径逐行对齐 |
| S02a | CUTLASS CuTe DSL 07_vectorized_array | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/07_vectorized_array.py` | `564d267e` | CuTe DSL 路径逐行对齐 |

## 2.6 T02 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T02 用途 |
| --- | --- | --- | --- | --- |
| S15b | PyTorch torch.nn.functional.relu | https://pytorch.org/docs/2.13/generated/torch.nn.functional.relu.html（200） | 2.13 | PyTorch 路径与语义 |
| S10b | CUDA Programming Guide：Kernels / Thread Hierarchy / Control Flow | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels 、#thread-hierarchy（200） | 当前线上版 | 边界、分支、grid 讲解 |
| S18b | NVIDIA cuda-samples vectorAdd（T02 沿用其 host/device 骨架） | 本地 `cuda-samples/cpp/0_Introduction/vectorAdd/vectorAdd.cu` | `b7c5481c` | CUDA 路径骨架 |
| S01b | Triton language：tl.where / tl.maximum（Math Ops） | https://triton-lang.org/main/python-api/generated/triton.language.maximum.html 等（在线核对） | v3.7.1 文档线 | Triton ReLU 语义 |
| S03b | cuTile operations：maximum | 本地 `cutile-python/docs/source/operations.rst`（maximum 条目） | `29444e0c` | cuTile ReLU 语义 |
| S02b | CUTLASS CuTe DSL 07_vectorized_array（T02 沿用其索引/切片骨架） | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/07_vectorized_array.py` | `564d267e` | CuTe DSL 骨架 |

## 2.7 T03 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T03 用途 |
| --- | --- | --- | --- | --- |
| S10c | CUDA Programming Guide「Maximize Memory Throughput」（合并访问） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput（200） | 当前线上版 | 合并访问/事务讲解 |
| S10d | CUDA 内置向量类型 float4 | 本地官方头文件（CUDA 安装目录下 `include/vector_types.h`） | CUDA 13.0 | float4 定义 |
| S01c | Triton 编译产物 PTX（ld/st.global.v4.b32） | 本机 `src/t03_relu/triton_relu.py` 的 `dump_ptx_evidence()`，对照 Triton v3.7.1 | v3.7.1 | Triton 自动向量化证据 |
| S03c | cuTile `ct.load/ct.store`（tile 级自动向量化） | 本地 `cutile-python/src/cuda/tile/_stub.py` load 签名、docs operations | `29444e0c` | cuTile 向量化说明 |
| S02c | CUTLASS CuTe DSL `07_vectorized_array.py`（切片向量化 load） | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/07_vectorized_array.py` | `564d267e` | CuTe 向量化 load |

## 2.8 T04 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T04 用途 |
| --- | --- | --- | --- | --- |
| S09a | cuBLAS 文档与 `cublasSgemm` | https://docs.nvidia.com/cuda/cublas/ （200，cuBLAS 13.3）+ 本地官方头文件（CUDA 安装目录下 `include/cublas_v2.h`） | CUDA 13.0 | cuBLAS 基线 |
| S10e | CUDA Programming Guide（二维 grid/thread 与循环） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html（200） | 当前线上版 | 二维索引讲解 |
| S01d | Triton tutorial 03-matrix-multiplication | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/03-matrix-multiplication.py | `v3.7.1` | T04 朴素写法对照；T05 分块权威来源 |
| S03d | cuTile MatMul sample | 本地 `cutile-python/samples/MatMul.py` | `29444e0c` | T04 官方 `ct.mma` 写法（tile=1 朴素化） |
| S02d | CUTLASS CuTe DSL GEMM tutorials | 本地 `third_party/cutlass/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py`（代表 tutorial 系列） | `564d267e` | T04 朴素 Array 索引写法；T05 权威来源 |

## 2.9 T05 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T05 用途 |
| --- | --- | --- | --- | --- |
| S18c | NVIDIA cuda-samples matrixMul（shared-memory tiled） | 本地 sparse clone `cuda-samples/cpp/0_Introduction/matrixMul/matrixMul.cu` | `b7c5481c` | CUDA tiled 骨架 |
| S01e | Triton tutorial 03-matrix-multiplication | 本地 `triton03.py`（GitHub tag v3.7.1 同路径） | `v3.7.1` | Triton tiled 骨架与 %M/%N 边界技巧 |
| S03e | cuTile MatMul.py（tile=16×16×16） | 本地 `cutile-python/samples/MatMul.py` | `29444e0c` | cuTile `ct.load/ct.mma` tiled 写法 |
| S02e | CUTLASS CuTe DSL 03_gemm_tiled_smem.py | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/03_gemm_tiled_smem.py` | `564d267e` | CuTe shared-memory tiled 权威写法 |

## 2.10 T06 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T06 用途 |
| --- | --- | --- | --- | --- |
| S10f | CUDA Programming Guide「Maximize Memory Throughput」（shared memory/bank/vector 原则） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput（200） | 当前线上版 | bank conflict 与 128-bit 依据 |
| S01f | Triton tutorial 03 autotune 配置表（BM/BN/BK/num_warps） | 本地 `triton03.py` | `v3.7.1` | Triton block 配置对照 |
| S02f | CUTLASS CuTe DSL swizzle/smem 官方实现 | 本地 `third_party/cutlass/python/CuTeDSL/cutlass/base_dsl/swizzle.py` 与 tutorial 系列 | `564d267e` | CuTe padding/swizzle 参考 |

## 2.11 T07 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T07 用途 |
| --- | --- | --- | --- | --- |
| S11a | CUDA Programming Guide「Writing Tile Kernels」（cp.async/pipeline） | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html（200） | 当前线上版 | cp.async + 流水线权威写法 |
| S01g | Triton tutorial 03 num_stages 配置 | 本地 `triton03.py` | `v3.7.1` | Triton 软件流水线 |
| S02g | CuTe DSL cp_async_shared_global.py（官方 cp.async 原语） | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/cp_async_shared_global.py` | `564d267e` | T07 CuTe 双缓冲 GEMM 的原语来源 |
| S03g | cuTile `ct.load(..., latency=1..10)` 官方流水线提示 | 本地 `cutile-python/src/cuda/tile/_stub.py`（load latency 参数） | `29444e0c` | T07 cuTile latency 1/2/4 实测 |

## 2.12 T08 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T08 用途 |
| --- | --- | --- | --- | --- |
| S18d | NVIDIA cuda-samples bf16TensorCoreGemm（WMMA 官方写法） | 本地 `cuda-samples/cpp/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu` | `b7c5481c` | CUDA WMMA bf16 路径 |
| S02h | CUTLASS 官方 14_ampere_tf32_tensorop_gemm | 本地 `third_party/cutlass/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu` | `564d267e` | CUTLASS tf32 tensor core 基线 |
| S02i | CuTe DSL 官方 ampere tensorop_gemm | 本地 `third_party/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` | `564d267e` | CuTe DSL fp16 tensor core |
| S01h | Triton tutorial 03 fp16 matmul | 本地 `triton03.py` | `v3.7.1` | Triton fp16 tl.dot |
| S03h | cuTile MatMul tf32 | 本地 `cutile-python/samples/MatMul.py` | `29444e0c` | cuTile tf32 mma |

## 2.13 T09 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T09 用途 |
| --- | --- | --- | --- | --- |
| S18e | NVIDIA cuda-samples transpose（transposeNaive） | 本地 `cuda-samples/cpp/6_Performance/transpose/transpose.cu` | `b7c5481c` | CUDA 朴素转置两方向 |
| S03i | cuTile Transpose.py | 本地 `cutile-python/samples/Transpose.py` | `29444e0c` | cuTile tile=1 朴素转置 |
| S01i | Triton 语言 tl.load/tl.store/tl.program_id（无专属 transpose tutorial，记录） | https://triton-lang.org/main/ | v3.7.1 | Triton 二维映射 |
| S02j | CuTe DSL thread/block 索引写法（03/07 tutorial） | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/` | `564d267e` | CuTe 二维映射 |

## 2.14 T10 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T10 用途 |
| --- | --- | --- | --- | --- |
| S18f | NVIDIA cuda-samples transpose.cu（transposeCoalesced / transposeNoBankConflicts） | 本地 `cuda-samples/cpp/6_Performance/transpose/transpose.cu` | `b7c5481c` | CUDA tile 转置与 padding |
| S03j | cuTile Transpose.py（tile=32） | 本地 `cutile-python/samples/Transpose.py` | `29444e0c` | cuTile tile 转置 |
| S01j | Triton `tl.trans` 语言 API | https://triton-lang.org/main/python-api/generated/triton.language.trans.html | v3.7.1 | Triton tile 转置 |
| S02k | CuTe DSL 03_gemm_tiled_smem.py（smem/barrier 写法） | 本地 `third_party/cutlass/examples/python/CuTeDSL/experimental/primitives/tutorial/03_gemm_tiled_smem.py` | `564d267e` | CuTe smem tile 转置 |

## 2.15 T11 使用的官方文件（在线核对后引用）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T11 用途 |
| --- | --- | --- | --- | --- |
| S18g | NVIDIA cuda-samples reduction（reduce0–7/cg_reduce 全家族） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu （200） | `b7c5481c` | CUDA 两段式 shared 规约（reduce2/reduce6 形态） |
| S01k | Triton `tl.sum` 语言 API | https://triton-lang.org/main/python-api/generated/triton.language.sum.html （200） | v3.7.1 | Triton block 内求和 |
| S03k | cuTile 官方 reduction 测试（ct.sum 轴规约与 keepdims） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/test/test_reduction.py （200）；本地 `cutile-python/test/test_reduction.py` | `29444e0c` | cuTile 1-D→2-D padded 分块求和 |
| S02l | CuTe DSL 官方 block_smem_reduce.py（block 级 shared reduction 原语家族） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/experimental/primitives/reduction/block_smem_reduce.py （200） | `564d267e` | CuTe smem+barrier 纯树规约（官方示例的简化纯 smem 版） |
| S15c | PyTorch `torch.Tensor.sum` / `Tensor.double` | https://pytorch.org/docs/2.13/generated/torch.Tensor.sum.html （200） | 2.13.0+cu130 | PyTorch 路径与 fp64 黄金参考 |


## 2.16 T12 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T12 用途 |
| --- | --- | --- | --- | --- |
| S10g | CUDA Programming Guide：Warp Shuffle Functions（`__shfl_down_sync` 签名、mask 语义、invalid 例子、warp reduction 例子） | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions （200） | 当前线上版（CUDA 13.3 文档线） | CUDA shuffle 语义与 mask 红线 |
| S10h | CUDA Programming Guide：Hardware Implementation → Hardware Multithreading（warp 调度与资源分区） | https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#advanced-kernels-hardware-implementation-hardware-multithreading （200） | 当前线上版 | latency hiding 的硬件依据 |
| S10i | CUDA Programming Guide：Kernel Launch and Occupancy | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy （200） | 当前线上版 | occupancy 定义与资源限制 |
| S18h | NVIDIA cuda-samples reduction_kernel.cu 的 reduce7（grid-stride → warpReduceSum → warp 和写 shared → 最终 warp shuffle） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu （200，已存本地 reduction_kernel.cu） | `b7c5481c` | CUDA reduceShfl 的权威结构 |
| S01k | Triton `tl.sum`（沿用 T11） | https://triton-lang.org/main/python-api/generated/triton.language.sum.html （200） | v3.7.1 | Triton 高层规约语义 |
| S01l | Triton 官方编译器源码 `python/triton/compiler/compiler.py`（CompiledKernel.asm 字典，含 ttir/ttgir/llir/ptx/cubin） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/triton/compiler/compiler.py （200，本机包同版本） | `v3.7.1` | 观察 Triton 生成 PTX/TTGIR 的官方接口 |
| S03k | cuTile 官方 reduction 测试（沿用 T11；grep shfl/shuffle 无命中 → shuffle 机制 N/A） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/test/test_reduction.py （200）；本地 `cutile-python/test/test_reduction.py` | `29444e0c` | cuTile ct.sum 能力对照与 N/A 依据 |
| S02m | CUTLASS CuTe DSL 官方 warp_vector_reduce.py（f32 add 走 5 轮 shuffle_sync_bfly 树） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/experimental/primitives/reduction/warp_vector_reduce.py （200，已存本地 warp_vector_reduce.py；仓库本地同 commit 副本） | `564d267e` | CuTe warp shuffle 树权威写法 |
| S15c | PyTorch `torch.Tensor.sum` / `Tensor.double`（沿用 T11） | https://pytorch.org/docs/2.13/generated/torch.Tensor.sum.html （200） | 2.13.0+cu130 | PyTorch 参考与 fp64 黄金参考 |
| S12 | Nsight Compute Profiling Guide（沿用；LaunchStats/Occupancy/SpeedOfLight/MemoryWorkloadAnalysis/SchedulerStats/WarpStateStats 截面，含 WarpStateStats body 的 Long Scoreboard/Barrier/Short Scoreboard stall 指标） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html （200） | Nsight Compute 2025.3.x（本机） | NCU 截面与指标解释 |
| S13 | Nsight Systems User Guide（沿用；WSL2 kernel 时间线受限备注在 §3） | https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | 官方 UserGuide | NSYS API 级取证 |


## 2.17 T13 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T13 用途 |
| --- | --- | --- | --- | --- |
| S01m | Triton 官方 tutorial 02-fused-softmax（naive_softmax 与 softmax_kernel 的 mask/BLOCK 技巧） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/02-fused-softmax.py （200，已存本地 triton02.py） | `v3.7.1` | PyTorch naive 写法与 Triton 行 kernel |
| S15d | PyTorch `torch.softmax` / `Tensor.max` / `torch.exp` / `Tensor.sum` | https://pytorch.org/docs/2.13/generated/torch.softmax.html 、.../torch.Tensor.max.html 、.../torch.exp.html 、.../torch.Tensor.sum.html （200） | 2.13.0+cu130 | PyTorch 路径语义与 fp64 黄金参考 |
| S18g | NVIDIA cuda-samples reduction reduce2（shared 树规约，沿用） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu （200） | `b7c5481c` | CUDA 行内 max/sum shared 树骨架 |
| S10j | CUDA Math API：Single Precision Functions（`expf/fmaxf`） | https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html （200） | 当前线上版（CUDA 13.x 文档线） | CUDA `expf`/`fmaxf` 语义与 SASS MUFU.EX2/FMNMX 解释 |
| S03l | cuTile 官方 test/test_softmax.py（softmax_per_row：ct.max/ct.exp/ct.sum） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/test/test_softmax.py （200）；本地 `cutile-python/test/test_softmax.py` | `29444e0c` | cuTile 行 softmax 权威写法 |
| S02n | CUTLASS CuTe DSL 官方 tutorial 06_softmax.py（Kernel 2: Block-level with Shared Memory Reductions） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/experimental/primitives/tutorial/06_softmax.py （200，已存本地 cute_softmax_tutorial.py；仓库本地同 commit 副本） | `564d267e` | CuTe block smem 3-pass softmax 权威写法 |
| S12 / S13 | Nsight Compute / Nsight Systems（沿用） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ；https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | Nsight Compute 2025.3.x / Nsight Systems 官方 UserGuide | NCU 截面/stall 与 NSYS API 取证 |


## 2.18 T14 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T14 用途 |
| --- | --- | --- | --- | --- |
| S21 | Online normalizer calculation for softmax（Milakov & Gimelshein, arXiv:1805.02867） | https://arxiv.org/abs/1805.02867 （200） | arXiv 1805.02867 | online max/sum rescale 更新式与 (m,l) 合并式 |
| S01m | Triton 官方 tutorial 02-fused-softmax（softmax_kernel 与 persistent-program host，沿用） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/02-fused-softmax.py （200） | `v3.7.1` | Triton 官方 fused kernel 与 occupancy 配置 |
| S02n | CUTLASS CuTe DSL 官方 tutorial 06_softmax.py（Kernel 5: Online Naive，沿用） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/experimental/primitives/tutorial/06_softmax.py （200，本地 `cute_softmax_tutorial.py`） | `564d267e` | CuTe 官方 online kernel 5 |
| S03l | cuTile 官方 test/test_softmax.py（softmax_per_row，沿用） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/test/test_softmax.py （200） | `29444e0c` | cuTile tile 级融合 softmax；online 原语 N/A 依据 |
| S15d | PyTorch torch.softmax 等（沿用） | https://pytorch.org/docs/2.13/generated/torch.softmax.html （200） | 2.13.0+cu130 | PyTorch 融合实现参考与 fp64 黄金参考 |
| S10j | CUDA Math API Single Precision Functions（沿用） | https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html （200） | 当前线上版 | CUDA expf 语义与 MUFU.EX2 解释 |
| S18g / S10g | cuda-samples reduction reduce7 与 Programming Guide shuffle（沿用） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu （200）；https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions （200） | `b7c5481c` / 当前线上版 | CUDA (m,l) 对的 warp shuffle 合并 |
| S12 / S13 | Nsight Compute / Nsight Systems（沿用） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ；https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | Nsight Compute 2025.3.x / 官方 UserGuide | NCU L2/DRAM 读写量与 NSYS API 取证 |


## 2.19 T15 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T15 用途 |
| --- | --- | --- | --- | --- |
| S22 | Attention Is All You Need（Vaswani et al. 2017）：scaled dot-product attention 标准公式与 Q/K/V 定义 | https://arxiv.org/abs/1706.03762 （200） | arXiv 1706.03762 | 标准公式来源（讲义 §4.0/§5.1） |
| S01o | Triton 官方 tutorial 06-fused-attention（qk/scale/mask/softmax/pv 步骤；外层 flash 分块明确不用） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/06-fused-attention.py （200，已存本地 triton06.py） | `v3.7.1` | Triton 朴素 attention 四步语义 |
| S15e | PyTorch `torch.matmul` / `torch.nn.functional.softmax` / `torch.nn.functional.scaled_dot_product_attention` | https://pytorch.org/docs/2.13/generated/torch.matmul.html ；.../torch.nn.functional.softmax.html ；.../torch.nn.functional.scaled_dot_product_attention.html （200） | 2.13.0+cu130 | PyTorch eager + SDPA 双参考与 fp64 黄金参考 |
| S03m | cuTile 官方 samples/AttentionFMHA.py（FMHA + online softmax + tiling，flash 层） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/samples/AttentionFMHA.py （200）；本地 `cutile-python/samples/AttentionFMHA.py` | `29444e0c` | cuTile 官方 attention 层级检查；朴素层 N/A 依据 |
| S02o | CUTLASS CuTe DSL 官方 ampere flash_attention_v2.py（flash 层） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py （200）；本地 `third_party/cutlass/.../flash_attention_v2.py` | `564d267e` | CuTe 官方 attention 层级检查；朴素层 N/A 依据 |
| S18g / S10g | cuda-samples reduction reduce7 与 Programming Guide shuffle（沿用） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu ；https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions （200） | `b7c5481c` / 当前线上版 | CUDA 行 softmax 的 warp shuffle 规约 |
| S10j | CUDA Math API Single Precision Functions（沿用） | https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html （200） | 当前线上版 | CUDA expf |
| S12 / S13 | Nsight Compute / Nsight Systems（沿用） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ；https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | Nsight Compute 2025.3.x / 官方 UserGuide | NCU/NSYS 取证 |


## 2.20 T16 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T16 用途 |
| --- | --- | --- | --- | --- |
| S02p | CUTLASS CuTe DSL 官方 Blackwell MLA decode KV-cache 示例（page table + variable-length KV sequences） | https://raw.githubusercontent.com/NVIDIA/cutlass/564d267e4c992c456d12ad02665f9acedf7708f1/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py （200）；本地 `third_party/cutlass/.../mla_decode_fp16.py` | `564d267e` | CuTe KV-cache 官方能力检查（sm_100 限定，本机 N/A） |
| S03m | cuTile 官方 samples/AttentionFMHA.py（T15 已核） | https://raw.githubusercontent.com/NVIDIA/cutile-python/29444e0c/samples/AttentionFMHA.py （200） | `29444e0c` | cuTile KV-cache 最接近官方能力检查（无 KV-cache 示例，N/A） |
| S16a | HuggingFace Transformers `cache_utils.DynamicCache`（[B,H,seq,D] 增量 append 语义） | https://raw.githubusercontent.com/huggingface/transformers/v5.14.1/src/transformers/cache_utils.py （200）；本机 gpp-core transformers 5.14.1 同文件；文档 https://huggingface.co/docs/transformers/en/generation_strategies （200） | `v5.14.1` | PyTorch 路径官方缓存语义 |
| S01o | Triton 官方 tutorial 06-fused-attention（qk/softmax/pv 步骤，沿用） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/06-fused-attention.py （200） | `v3.7.1` | Triton decode attention 四步语义 |
| S15e | PyTorch matmul/softmax/SDPA（沿用） | https://pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html （200） | 2.13.0+cu130 | PyTorch 参考 attention |
| S18g / S10g | cuda-samples reduction reduce7 与 Programming Guide shuffle（沿用） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu ；https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions （200） | `b7c5481c` / 当前线上版 | CUDA decode softmax 规约 |
| S10j | CUDA Math API Single Precision Functions（沿用） | https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html （200） | 当前线上版 | CUDA expf |
| S12 / S13 | Nsight Compute / Nsight Systems（沿用） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ；https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | Nsight Compute 2025.3.x / 官方 UserGuide | NCU/NSYS 取证 |


## 2.21 T17 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T17 用途 |
| --- | --- | --- | --- | --- |
| S01o | Triton 官方 tutorial 06-fused-attention（FA2 数值结构：Q tile/K-V tile/running m,l/acc rescale/causal mask/tl.dot） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/06-fused-attention.py （200，已存本地 triton06.py） | `v3.7.1` | Triton FA forward 权威结构 |
| S17a | FlashAttention 论文（Dao et al., arXiv:2205.14135）的 IO-Aware tiling 与 online softmax 算法 | https://arxiv.org/abs/2205.14135 （200） | arXiv 2205.14135 | FA 算法与 tile 复杂度依据 |
| S15e | PyTorch `F.scaled_dot_product_attention`（沿用） | https://pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html （200） | 2.13.0+cu130 | SDPA 黄金参考 |
| S12 / S13 | Nsight Compute / Nsight Systems（沿用） | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html ；https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/nsight/nsight-systems/UserGuide/index.html （200） | Nsight Compute 2025.3.x / 官方 UserGuide | NCU L2/DRAM 对比与 NSYS 取证 |


## 2.22 T18 使用的官方文件（在线核对后引用，访问日期 2026-08-16）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T18 用途 |
| --- | --- | --- | --- | --- |
| S01o | Triton 官方 tutorial 06-fused-attention（T17 已核，FA2 结构来源） | https://raw.githubusercontent.com/triton-lang/triton/v3.7.1/python/tutorials/06-fused-attention.py （200） | `v3.7.1` | T18 CUDA 手工映射的算法蓝本 |
| S17a | FlashAttention 论文（T17 已核） | https://arxiv.org/abs/2205.14135 （200） | arXiv 2205.14135 | FA tiling/online 算法依据 |
| S15e | PyTorch SDPA（沿用） | https://pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html （200） | 2.13.0+cu130 | SDPA 黄金参考与同数据对照 |
| S18g | NVIDIA cuda-samples reduction_kernel.cu（reduce7 行规约思路） | https://raw.githubusercontent.com/NVIDIA/cuda-samples/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu （200） | `b7c5481c` | CUDA FA 每行 online merge 的 shared/串行规约参考 |
| S10g | CUDA Programming Guide：Warp Shuffle / Synchronization Functions | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions （200） | 当前线上版（CUDA 13.3 文档线） | `__syncthreads`/shared 可见性与行归约依据 |
| S03m / S02o | cuTile AttentionFMHA / CuTe flash_attention_v2（沿用） | T17 同 URL | `29444e0c` / `564d267e` | 五路径官方能力实测 |
| S10 / S12 / S13 | CUDA Programming Guide / Nsight Compute / Nsight Systems（沿用） | 同前 | 当前线上版 / Nsight 2025.3.x | shared/barrier 语义与 NCU/NSYS 取证 |


## 2.23 T19 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T19 用途 |
| --- | --- | --- | --- | --- |
| S20a | Qwen2.5-0.5B-Instruct 官方 README Quick Start | 本地固定快照 `assets/modelscope/qwen2.5-0.5b-instruct/README.md`（ModelScope 页 URL 见 S20） | revision `master` | 官方 messages 结构、`apply_chat_template`、`model.generate` 最小流程 |
| S16a | Transformers `AutoModelForCausalLM` / `AutoTokenizer` / `apply_chat_template` / `GenerationMixin.generate` | https://huggingface.co/docs/transformers/ （S16 已核，200） | transformers 5.14.1 | 模型/分词器加载与生成 API |
| S14 | ModelScope CLI 下载与 revision（沿用） | 本机 gpp-core `modelscope-hub 0.1.8` | 0.1.8 | 固定快照来源与 revision 记录 |


## 2.24 T20 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T20 用途 |
| --- | --- | --- | --- | --- |
| S05a | vLLM Quickstart | https://docs.vllm.ai/en/v0.27.1/getting_started/quickstart/ （200） | v0.27.1 | Offline Batched Inference、Online Serving、`vllm serve Qwen/Qwen2.5-1.5B-Instruct`、Attention Backends |
| S05b | vLLM OpenAI-compatible server | https://docs.vllm.ai/en/v0.27.1/online_serving/ （200） | v0.27.1 | `/v1/models`、`/v1/chat/completions`、流式响应 |
| S05c | vLLM Docker 镜像 | `vllm/vllm-openai:v0.27.1`（本地镜像，digest 见 `config/day0-lock.json`） | v0.27.1 | 容器启动命令与入口 `vllm serve` |
| S05d | vLLM Quickstart: On Attention Backends | https://docs.vllm.ai/en/v0.27.1/getting_started/quickstart/#on-attention-backends （200） | v0.27.1 | `--attention-backend FLASH_ATTN` / `FLASHINFER` 手动指定注意力后端 |


## 2.25 T21 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T21 用途 |
| --- | --- | --- | --- | --- |
| S06a | SGLang Quickstart（官方站点 + GitHub v0.5.17） | https://docs.sglang.io/docs/get-started/quickstart （200）；https://raw.githubusercontent.com/sgl-project/sglang/v0.5.17/docs/docs/get-started/quickstart.mdx （200） | latest / v0.5.17 | Docker 启动、`sglang.launch_server`、curl/OpenAI SDK/requests/`/generate` 示例 |
| S06b | SGLang OpenAI API | https://docs.sglang.io/docs/basic_usage/openai_api （200）；https://raw.githubusercontent.com/sgl-project/sglang/v0.5.17/docs/docs/basic_usage/openai_api.mdx （200） | latest / v0.5.17 | `/v1/chat/completions`、OpenAI 兼容调用 |
| S06c | SGLang Docker 镜像 | `lmsysorg/sglang:v0.5.17`（本地镜像存在） | v0.5.17 | 容器启动命令 |


## 2.26 T22 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T22 用途 |
| --- | --- | --- | --- | --- |
| S07a | TRT-LLM Qwen 示例 README | https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v1.2.1/examples/models/core/qwen/README.md （200） | v1.2.1 | `convert_checkpoint.py`、`trtllm-build`、`run.py` 官方流程 |
| S07b | TRT-LLM `trtllm-serve` 命令文档 | https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/v1.2.1/docs/source/commands/trtllm-serve/trtllm-serve.rst （200） | v1.2.1 | OpenAI 兼容端点、`trtllm-serve <model> --tokenizer --host --port` |
| S07c | TRT-LLM Docker 镜像 | `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`（本地镜像存在） | v1.2.1 | 容器内工具路径与启动方式 |
| S07d | TRT-LLM Overview | https://nvidia.github.io/TensorRT-LLM/overview.html （200） | latest | TensorRT-LLM 是什么、为什么有用、关键能力 |
| S07e | TRT-LLM Quick Start Guide | https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html （200） | latest | `trtllm-serve` 直接起服务、curl 示例、LLM API 离线推理 |


## 2.27 T23 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T23 用途 |
| --- | --- | --- | --- | --- |
| S08a | ms-swift README（Inference / Deployment 命令） | https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.3/README.md （200） | v4.4.3 | `swift infer`、`swift deploy`、`--infer_backend vllm` |
| S08b | ms-swift 官方文档 | https://swift.readthedocs.io/zh-cn/latest/ （200） | latest | 部署/推理/训练整体说明 |
| S08c | ms-swift 快速开始 | https://swift.readthedocs.io/zh-cn/latest/GetStarted/Quick-start.html （200） | latest | `swift sft`、`swift infer`、`swift deploy` 快速使用 |
| S08d | ms-swift 命令行参数 | https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html （200） | latest | `--infer_backend`、`--host/--port`、训练参数等 |
| S08e | ms-swift 推理和部署 | https://swift.readthedocs.io/zh-cn/latest/Instruction/Inference-and-deployment.html （200） | latest | 服务端/客户端、`swift deploy` |
| S08f | ms-swift Best Practices（Qwen3.8） | https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_8-Best-Practice.html （200） | latest | 最佳实践参考 |
| S08g | 本机 gpp-swift-4.4.3 环境 | `conda run -n gpp-swift-4.4.3 swift deploy --help` | ms-swift 4.5.0.dev0（源码安装） | `--host/--port/--served_model_name/--infer_backend` 参数核对 |


## 2.28 T24 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T24 用途 |
| --- | --- | --- | --- | --- |
| S05 | vLLM 官方文档/blog（PagedAttention、Continuous Batching、Speculative Decoding） | https://blog.vllm.ai/2023/06/20/vllm.html （200）；https://docs.vllm.ai/en/latest/features/spec_decode.html （200）；https://docs.vllm.com.cn/en/latest/ （200） | latest | PagedAttention、Continuous Batching、投机解码、量化、多硬件 |
| S06 | SGLang 官方文档（RadixAttention） | https://docs.sglang.io/ （200）；https://sglang-zh.llamafactory.cn/index.html （200） | latest | RadixAttention、Jump-ahead decoding、FlashInfer、前端语言 |
| S07d | TRT-LLM Overview | https://nvidia.github.io/TensorRT-LLM/overview.html （200） | latest | TensorRT-LLM 优化能力 |
| S08e | ms-swift 推理和部署 | https://swift.readthedocs.io/zh-cn/latest/Instruction/Inference-and-deployment.html （200）；https://swift.readthedocs.io/zh-cn/latest/GetStarted/Quick-start.html （200） | latest | ms-swift 部署封装、全链路能力 |


## 2.29 T25 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T25 用途 |
| --- | --- | --- | --- | --- |
| S15 | PyTorch 官方训练文档 | https://pytorch.org/tutorials/beginner/basics/intro.html （200） | latest | forward/backward/optimizer 训练循环 |
| S16 | Transformers AutoModelForCausalLM / AutoTokenizer | https://huggingface.co/docs/transformers/ （S16 已核） | transformers 5.14.1 | 模型/分词器加载 |
| S20 | Qwen2.5-0.5B-Instruct 固定快照 | `assets/modelscope/qwen2.5-0.5b-instruct` | revision `master` | 固定模型 |
| S14 | alpaca-gpt4-data-zh 固定数据集 | `assets/modelscope/alpaca-gpt4-data-zh/train.csv` | revision `master` | SFT 数据 |


## 2.30 T26 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T26 用途 |
| --- | --- | --- | --- | --- |
| S04 | DeepSpeed 官方文档/README | https://deepspeed.readthedocs.io/en/latest/ （200）；https://github.com/deepspeedai/DeepSpeed （200） | 0.19.5 | ZeRO 概念、DeepSpeed 定位 |
| S04a | DeepSpeed `deepspeed.initialize` 与 ZeRO 配置 | https://deepspeed.readthedocs.io/en/latest/initialize.html （200） | 0.19.5 | 最小接入、ds_config |
| S04b | DeepSpeed Getting Started | https://www.deepspeed.ai/getting-started/ （200） | latest | 安装、initialize、launcher 启动流程 |
| S04c | DeepSpeed 中文文档 | https://docs.deepspeed.org.cn/en/latest/index.html （200） | latest | 训练/推理/ZeRO 概念中文参考 |


## 2.31 T27 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T27 用途 |
| --- | --- | --- | --- | --- |
| S08a | ms-swift README（SFT 命令） | https://raw.githubusercontent.com/modelscope/ms-swift/v4.4.3/README.md （200） | v4.4.3 | `swift sft` 命令、LoRA 参数 |
| S08b | ms-swift 命令行参数 | https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html （200） | latest | 训练参数说明 |
| S08c | ms-swift 快速开始 | https://swift.readthedocs.io/zh-cn/latest/GetStarted/Quick-start.html （200） | latest | SFT/LoRA 快速使用 |


## 2.32 T28 使用的官方文件（在线核对后引用，访问日期 2026-08-17）

| ID | 官方文件 | 精确 URL / 本地官方仓库路径 | 版本/commit | T28 用途 |
| --- | --- | --- | --- | --- |
| S15 | PyTorch 官方 DDP/FSDP 文档 | https://pytorch.org/docs/stable/notes/ddp.html （200）；https://pytorch.org/docs/stable/fsdp.html （200） | latest | DDP/FSDP 概念 |
| S04 | DeepSpeed ZeRO 官方文档 | https://deepspeed.readthedocs.io/en/latest/ （200） | 0.19.5 | ZeRO 分片 |
| S08 | ms-swift 官方文档 | https://swift.readthedocs.io/zh-cn/latest/ （200） | latest | ms-swift 分布式/训练 |
| S28a | Megatron-LM 官方仓库 | https://github.com/NVIDIA/Megatron-LM （200） | latest | TP/PP/SP/CP 参考 |


## 3. 环境受限备注（C 级证据，必须诚实标注）

- NSYS：本机 WSL2 上 `nsys profile` 可采集 CUDA API/memcpy 时间线，但 SQLite 中无
  `CUPTI_ACTIVITY_KIND_KERNEL` 表，kernel 时间线缺失。官方论坛确认这是 WSL2 已知限制：
  https://forums.developer.nvidia.com/t/nsys-is-not-collecting-kernel-data/244647
  恢复路径：同一命令在原生 Linux 或 Windows 侧 Nsight Systems 运行。NCU 正常（本机已取证）。
- 显存门禁：`torch` 报告 8187 MiB，`nvidia-smi` 标称 8188 MiB；锁文件以 8187 MiB 为下限。

## 4. 记录格式（后续追加）

```text
Sxx | <官方文档/官方仓库/官方示例/高质量参考> | <精确URL> | <YYYY-MM-DD> | <version/tag/commit> | <官方文件相对路径:行号(可选)> | <Txx> | <docs/evidence/Txx/...>
```
