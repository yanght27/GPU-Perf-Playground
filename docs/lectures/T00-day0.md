# T00 Day 0：环境与证据门禁（唯一主讲义）

- Ticket：T00
- 状态：`done`（学习者 2026-08-15 验收通过）
- 唯一学习变量：环境、固定快照与证据链是否**真实、可复现、被官方来源锁定**
- 环境：5 个 conda 环境 + 4 个 Docker 镜像 + 系统 CUDA/NSYS/NCU/SASS
- 官方来源：S01–S20（`config/source-ledger.md`，2026-08-15 在线核实）
- 跨 Ticket 术语：`docs/CONCEPTS.md`（工具定位、Bound 判定、高频概念速查）
- 本节导读：**一句话目标**——把环境、版本、资产、容器全部锁定，让后续每个增量的证据可复现；**依次学到**——①conda/Docker 是什么、为什么需要锁；②revision/SHA256 怎么校验快照；③A/B/C 证据级别；④NSYS/NCU/SASS 第一次使用；**学完应能回答**——为什么先锁环境再学算子？smoke 是干什么的？；**相关工具/技术**——conda、Docker/NVIDIA Container Toolkit、ModelScope、SHA256、NSYS/NCU/SASS。
- 本节内容：**要解决的问题**——之前环境散落各处，版本漂移会让“同一份代码”结果不一致；**核心手段**——day0-lock.json 锁硬件/镜像/环境/资产 revision+文件哈希，6 个门禁脚本做机器检查；**怎么实现**——`environments/*.sh --verify-only` + `containers.sh --verify-only` + 最小 vector_add smoke；**怎么验证**——6 门禁全 PASS、NCU/SASS 出指标、NSYS 记录 WSL2 限制；**最终结论**——先有可复现的 Day 0，后续每个 Ticket 的证据才可信。

## 1. 上一轮问题回答

无（T00 是课程第一个增量）。

## 2. 规范实现与官方来源

本轮没有“算子实现”，但同样严格在线对齐：每个版本/命令/镜像都在动手前核对了官方来源，
全部记录在 `config/source-ledger.md`。关键结论：

| 事项 | 官方依据 | 结论 |
| --- | --- | --- |
| ModelScope 下载命令 | gpp-core 内 modelscope-hub 0.1.8 官方 CLI：`ms download --help`（S14） | 用 `ms download --repo-type model|dataset --revision <rev> --local-dir <dir> <repo_id>`；已修正 `environments/gpp-core.sh` 的旧式 `modelscope download --model` 写法 |
| ms-swift 环境名与版本 | ms-swift 官方仓库 tag `v4.4.3` = `e1287928…`（S08） | 环境名 `gpp-swift-4.4.3` 正确；该 tag 内版本文件声明 `4.5.0.dev0`，`direct_url.json` commit 与官方 tag 完全一致，无需改名/回退 |
| CUTLASS 锁定 commit | 官方 main HEAD `564d267e…`；`python/CuTeDSL/requirements-cu13.txt` 要求 `nvidia-cutlass-dsl[cu13]==4.7.0`（S02） | `third_party/cutlass` 已检出该 commit，与 gpp-cute 安装的 4.7.0 一致 |
| 容器镜像 | Docker Hub/官方 NGC 标签与本地 digest 比对（S05/S06/S07） | 4 个镜像均 amd64，GPU passthrough PASS |
| 模型快照 | ModelScope 官方模型页 + `ms download`（S14/S20） | Qwen2.5-0.5B-Instruct 本地快照 11 个文件，SHA256 全量锁定 |
| 训练数据快照 | ms-swift v4.4.3 官方 README Quick Start（S08，行 163–180 使用 `AI-ModelScope/alpaca-gpt4-data-zh#500`） | 采用同一官方数据集 `AI-ModelScope/alpaca-gpt4-data-zh`，本地快照 4 个文件 |

## 3. 本轮实现结果（实际命令与输出）

### 3.1 6 个机械门禁全部 PASS

| 命令 | 结果 | 证据 |
| --- | --- | --- |
| `bash environments/gpp-core.sh --verify-only` | `gpp_core: PASS`；模型+数据集完整性 PASS；matmul=1024.0 | `docs/evidence/T00/gpp-core-verify.txt` |
| `bash environments/gpp-cute.sh --verify-only` | `gpp_cute: PASS`；cutlass-dsl 4.7.0；commit 校验通过 | `docs/evidence/T00/gpp-cute-verify.txt` |
| `bash environments/gpp-cutile.sh --verify-only` | `gpp_cutile: PASS`；cuda-tile 1.5.0 / cupy 14.1.1；matmul=1024.0 | `docs/evidence/T00/gpp-cutile-verify.txt` |
| `bash environments/gpp-deepspeed-0.19.5.sh --verify-only` | `gpp_deepspeed: PASS`；deepspeed 0.19.5；matmul=1024.0 | `docs/evidence/T00/gpp-deepspeed-verify.txt` |
| `bash environments/gpp-swift-4.4.3.sh --verify-only` | `gpp_swift: PASS`；ms-swift commit=`e1287928…`=官方 tag v4.4.3 | `docs/evidence/T00/gpp-swift-verify.txt` |
| `bash environments/containers.sh --verify-only` | `container_gpu_passthrough: PASS`；4 镜像 amd64、容器内 `nvidia-smi` 均为 4070/8.9/8188 | `docs/evidence/T00/containers-verify.txt` |

### 3.2 锁文件与资产

- `config/day0-lock.json`：硬件（4070 / sm_8.9 / ≥8187 MiB / driver 581.94）、Docker
  29.1.3 + nvidia-ctk 1.20.0、4 个镜像、5 个环境（python + 精确包版本 + CUTLASS/ms-swift
  commit）、2 个资产（revision + 每个文件的 size/sha256）全部锁定。
- `assets/modelscope/`：Qwen2.5-0.5B-Instruct 11 个文件（含 988 MB `model.safetensors`），
  数据集 4 个文件；两者 revision 记录与 SHA256 清单均通过 gpp-core 门禁。
- 说明：显存 `nvidia-smi` 标称 8188 MiB，`torch` 实测 8187 MiB，锁文件以 8187 为下限。

### 3.3 工具链 smoke

- `nvcc -O3 -arch=sm_89` 编译最小 `vector_add` 并运行通过：`c[0]=3.000000`。
- NCU：成功采集，关键指标 `DRAM Throughput 93.04%`、`Compute (SM) Throughput 10.04%`、
  `Achieved Occupancy 90.73%`、`Duration 35.36 us`（详见 `t00-smoke-ncu-details.txt`）。
- SASS：`cuobjdump -sass` 与 `nvdisasm`（先 `nvcc -cubin` 生成 cubin）均成功，看到
  `LDG.E` / `STG.E` / `BRA` 等指令。
- NSYS：命令可运行并生成 API 级时间线（`cudaMalloc`/`cudaMemcpy`/`cudaLaunchKernel`），
  但 WSL2 下未采集到 CUDA kernel 表——这是**已核实的 WSL2 限制**（官方论坛 S13 备注），
  恢复路径是原生 Linux 或 Windows 侧 Nsight Systems。本轮如实标注为 C 级，未伪造证据。

### 3.4 三个名词解释（避免把“受限”误解为“失败”）

- **A/B/C 证据级别**：A = 本机真实实现并测到了数据；B = 本机只能运行和比较；C = 受本机
  条件限制，只能依据官方资料做分析。C 不是“做错了”，而是“当前环境确实拿不到这类证据，
  我们明确记录原因和恢复路径”。T00 里 NCU/SASS 是 A 级；NSYS 的 API 时间线是 A 级、
  kernel 时间线是 C 级。
- **kernel 缺失**：NSYS 时间线里应当有一行 `vector_add` 这个 GPU kernel 的执行条；本机
  WSL2 上 NSYS 只录到了 CPU 侧 CUDA API（分配内存、拷贝、启动 kernel），没有录到 GPU 侧
  kernel 条。原因是 WSL2 的 GPU 驱动在 Windows 侧、用户程序在 Linux 侧，时间戳同步受限
  （官方论坛已确认）。这不是我们代码的问题：同一个二进制拿到原生 Linux 或 Windows 侧
  Nsight Systems 就能看到 kernel 条。
- **smoke 与 T01 的关系**：smoke = “最小可用性验证”，相当于装完水管后先开一下水龙头。
  T00 的任务书要求验证“编译、运行、GPU、NSYS、NCU、SASS”六个门禁，就必须有一个最小的
  CUDA 程序作为被验证对象，所以临时写了 12 行的 `vector_add`。它**不是 T01 的课程实现**：
  没有五路径、没有正确性测试、没有 benchmark。T01 会从零正式实现 vector add 并讲解
  grid/block/thread；smoke 只是 T00 的验收证据，学完 T01 后它就没有用处了。

## 4. 核心代码与逐行解释（T00 最小 smoke：vector_add）

```cuda
// 唯一作用：证明 nvcc → 运行 → NSYS/NCU/SASS 这条证据链可用。
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程 id
    if (i < n) c[i] = a[i] + b[i];                   // 边界保护
}
```

完整源码与二进制在 `docs/evidence/T00/vector_add.cu`（这个 kernel 会在 T01 被正式课程版本取代）。

### 4.1 逐行解释

- `__global__`：声明该函数在 GPU 上执行、由 host 通过 launch 语法启动；这是 T01 要展开
  学习的 CUDA 编程模型入口。
- `blockIdx.x * blockDim.x + threadIdx.x`：把“block 编号 × 每 block 线程数 + block 内
  线程编号”线性化成全局下标 `i`。T00 只要求 smoke 通过，T01 会系统讲 grid/block/thread。
- `if (i < n)`：`n` 通常不是 block 大小的整数倍，最后一组线程会越界；边界保护是元素级
  kernel 的固定动作（T02 会再深入）。
- 内存访问 `a[i]/b[i]/c[i]` 都是连续的，因此是合并访问（coalesced），这正是后面所有
  带宽优化讨论的起点。

## 5. 核心知识点要点讲解

### 5.1 为什么要 Day 0：先锁环境，再学知识

学习闭环的每一步都要求“证据可复现”。如果 PyTorch/Triton/CUDA 版本漂移，同一段代码可能
今天能跑、明天报错，或性能数字无法互相比较。所以 T00 用一个 `day0-lock.json` 把：

- 硬件（GPU 型号、SM、显存下限）
- 每个 conda 环境的 Python 与包精确版本
- 官方仓库 commit（CUTLASS、ms-swift）
- 容器镜像 tag 与 digest
- 模型/数据集 revision 与每个文件 SHA256

全部固定下来。后续每个 Ticket 先跑 `--verify-only`，等价于“上一步的证据还成立吗”。

### 5.2 版本一致性的两个典型问题（本轮都实际遇到并解决）

1. **ms-swift 名字像 4.4.3，包却显示 4.5.0.dev0**：不是装错了。在线核对官方仓库后，
   tag `v4.4.3` 对应的 commit 就是本机安装的 `e1287928…`，该 tag 里版本文件声明的字符串
   是 `4.5.0.dev0`。结论：环境名正确，锁 commit 即可。
2. **nvidia-smi 8188 MiB vs torch 8187 MiB**：同一块卡两种统计口径差 1 MiB。门禁不能写
   死 8188，否则 torch 视角永远失败；锁文件用“实测下限 8187”作为可复现标准。

### 5.3 ModelScope 固定快照与 SHA256

“固定 revision 快照”不是口头约定：`ms download --revision master --local-dir ...` 之后，
锁文件记录目录内**每一个文件**的路径、字节数和 SHA256。以后每次验证都会重算哈希，
任何文件被改动、缺失或多出都会被门禁抓住。这是 B/C 级证据可信度的基础。

### 5.4 Docker GPU 透传（passthrough）

容器里能跑 CUDA 不是 Docker 自带能力，而是靠 NVIDIA Container Toolkit 注入驱动接口。
`containers.sh --verify-only` 证明了三件事：daemon 是 WSL 原生引擎、镜像架构是 amd64、
`docker run --rm --gpus all --entrypoint nvidia-smi <image>` 在容器内能看到本机 GPU。

### 5.5 工具 smoke 教了什么

- **NCU 的第一组数字**：`DRAM Throughput 93.04%` vs `Compute 10.04%`——向量加几乎没有
  计算量，时间几乎都花在读写内存上，这就是“Memory-Bound”的第一手证据（T02/T03 展开）。
- **SASS 是编译产物**：`nvdisasm` 直接吃 host 可执行文件会报 section 错误，正确路径是
  `nvcc -cubin` 生成设备端 cubin，再 `nvdisasm <file>.cubin`。T01 开始会反复用。
- **NSYS 的 WSL2 限制**：API 时间线可用，kernel 时间线缺失；恢复路径见 §3.3。后续每个
  算子 Ticket 我会同时给 NSYS 命令，并明确区分“本机拿到了什么证据 / 什么只能 C 级”。

### 5.6 新手必懂的环境术语（零疑问补讲）

- **conda environment**：一套独立的 Python 目录（含解释器、site-packages），不同环境
  互不污染。T00 之后所有命令都写 `conda run -n <env>`，就是为了“哪个工具在哪个环境跑”
  可复现。
- **Docker image / container**：image 是只读模板（相当于安装盘），container 是运行实例
  （相当于装好的机器）。锁文件锁 image tag + digest，保证每个人跑的是同一份模板。
- **image digest**：镜像内容的 SHA256 指纹。同一个 tag 可能被重新推送覆盖，digest 不会，
  所以门禁同时记录 digest。
- **revision**：模型/数据集仓库的版本号（branch/tag/commit）。我们固定 revision 下载，
  再对每个文件算 SHA256，保证“快照”和“校验”两层都锁死。
- **SHA256**：内容哈希。文件改动一个字节，哈希就完全不同；门禁每次重算并比对。
- **chat template 标记**：Qwen 对话模板里的固定字符串（如 `add_generation_prompt`、
  `system/user/assistant`），用来确认下载的 tokenizer 配置是“带对话模板”的官方文件。
- **A/B/C 证据级别与 smoke**：见本讲义 §3.4；一句话——A=本机实测，B=本机只能运行比较，
  C=受环境限制只能记录官方分析，smoke=最小可用性验证。

## 6. 性能分析（本轮仅 smoke，不是优化实验）

NCU `--set basic` 实测（`vector_add`，N=2^20，grid=4096，block=256）：

| 指标 | 值 | 含义 |
| --- | --- | --- |
| Duration | 35.36 us | 单次 kernel 执行时间 |
| DRAM Throughput | 93.04% | 显存带宽几乎打满 |
| Compute (SM) Throughput | 10.04% | 计算单元很闲 |
| Achieved Occupancy | 90.73% | 活跃 warp 占理论上限比例 |
| Registers/Thread | 16 | 内核寄存器占用 |

NSYS API 级时间线显示：`cudaMalloc` 是 smoke 程序最大时间项（冷启动/首次分配），
`cudaMemcpy` 与 `cudaLaunchKernel` 相对很小；kernel 级数据因 WSL2 限制缺失。

## 7. Memory-Bound / Compute-Bound / Latency-Bound 判断

- 结论：smoke 的 vector_add 是典型 **Memory-Bound**。
- 证据：DRAM Throughput 93.04% vs Compute 10.04%；算术强度 ≈ 每 3 次访存只有 1 次加法。
- 诚实标注：这是 NCU 实测（A 级），不是预测；NSYS kernel 时间线缺失（C 级，原因已记录）。

## 8. 知识点完整性检查（对照 coverage-matrix）

- 本轮把“环境与资产门禁”列为已过关；NSYS/NCU/SASS 三行更新为“学习中”，证据指向
  `docs/evidence/T00/`。
- 其余算子/框架知识点保持 `待学`，归属 T01–T28，没有静默遗漏。
- `idea.md` 要求“版本、导入、编译、GPU、NSYS、NCU、SASS 环境门禁”：全部已执行并留证据。

## 9. 过关问题及答案（8 题，一问一答）

**Q1.** 为什么 Day 0 要同时锁“包版本 + 官方仓库 commit + 资产 SHA256”，只锁包版本够不够？

**A1（回答）**：不够。包版本只锁定“发布号”，但 ms-swift/CUTLASS 这类从 git commit 安装/使用的项目，
   同名版本可能对应不同源码；模型/数据集没有 SHA256 就无法发现文件损坏或缺失。三者合起来
   才能保证同一份代码在同一证据基线上复现。

**Q2.** `gpp-swift-4.4.3` 里的 ms-swift 显示 `4.5.0.dev0`，为什么仍然认为环境正确？

**A2（回答）**：因为本机 `direct_url.json` 的 commit `e1287928…` 与 ms-swift 官方仓库 tag `v4.4.3`
   完全一致；该 tag 的版本文件声明的字符串就是 `4.5.0.dev0`。所以“环境名 4.4.3”锁的是
   tag，包显示的 dev0 是源码内版本串，二者不矛盾。

**Q3.** 容器能访问 GPU 的本质原因是什么？`--gpus all` 靠谁工作？

**A3（回答）**：本质是 NVIDIA Container Toolkit 在容器创建时把驱动/GPU 设备接口注入容器，使容器内
   CUDA runtime 能通过宿主驱动访问 GPU。`docker run --gpus all` 需要 daemon 配置了
   `nvidia` runtime；本机 `containers.sh` 已实测 4 个镜像透传 PASS。

**Q4.** smoke 的 NCU 数据显示 DRAM 93%、Compute 10%，这说明 vector_add 是什么 Bound？为什么？

**A4（回答）**： Memory-Bound。向量加每个元素只有 1 次加法，却要读 a、b 和写 c 共 3 次访存，算术强度极低；NCU 实测 DRAM 吞吐 93.04%、SM 计算吞吐仅 10.04%，瓶颈在带宽不在算力。

**Q5.** conda environment 解决什么问题？为什么 T00 之后所有命令都写 `conda run -n <env>`？

**A5（回答）**：conda environment 是独立的 Python 目录，解决“不同工具依赖互相冲突/污染”的问题。
   写 `conda run -n <env>` 是为了每一步都可复现：谁在哪个环境跑、装了哪个版本，不靠
   “当前 shell 恰好激活了什么”这种不稳定状态。

**Q6.** Docker image 和 container 的区别是什么？为什么锁 image 时还要锁 digest？

**A6（回答）**：image 是只读模板，container 是运行实例。tag 可被重新推送覆盖，digest 是内容哈希，
   锁 digest 才能保证今天跑的镜像和昨天验证的是同一份。

**Q7.** 模型快照的 revision 和文件 SHA256 各锁住什么？只锁其中一个会漏掉什么问题？

**A7（回答）**：revision 锁“下载哪个版本的内容”，SHA256 锁“文件下载后有没有损坏/被改”。只锁 revision
   无法发现文件损坏；只锁 SHA256 不知道文件对应哪个官方版本。

**Q8.** 用一句话分别解释 A/B/C 证据级别；为什么“C 级”不等于“失败”？

**A8（回答）**：A=本机实现并实测；B=本机只能运行比较；C=受环境限制只能依据官方资料分析并记录恢复
   路径。C 是诚实标注“当前环境拿不到这类证据”，而不是做错或失败。
## 10. 本轮停止点

- 已完成：在线核实 S01–S20；锁定 CUTLASS commit 并检出；生成 `config/day0-lock.json`；
  下载/校验模型与数据集；修正 modelscope 官方 CLI；6 个 `--verify-only` 全 PASS；
  NSYS/NCU/SASS smoke 证据齐备。
- 理论 vs 实测：版本对应关系、SHA256、门禁设计是理论/工程事实；NCU 数字是实测；
  NSYS kernel 时间线受 WSL2 阻塞（C 级）。
- 未做：没有进入 T01 的任何算子课程内容。

## 11. 下一最小增量

T01 Vector Add（五路径基线）：在 T00 已证明可复现的 gpp-core/gpp-cute/gpp-cutile 上，
正式学习 grid/block/thread 执行模型，并把 PyTorch/CUDA/Triton/cuTile/CuTe 五条开发路径
跑成同一个“实现→测试→实测”闭环。

## 附录：可复现原生命令

```bash
# 6 个环境/容器门禁（仓库根目录执行）
bash environments/gpp-core.sh --verify-only
bash environments/gpp-cute.sh --verify-only
bash environments/gpp-cutile.sh --verify-only
bash environments/gpp-deepspeed-0.19.5.sh --verify-only
bash environments/gpp-swift-4.4.3.sh --verify-only
bash environments/containers.sh --verify-only

# NCU smoke
ncu --set basic -o docs/evidence/T00/t00-smoke-ncu ./docs/evidence/T00/vector_add
ncu --import docs/evidence/T00/t00-smoke-ncu.ncu-rep --page details --print-details all

# NSYS smoke（WSL2：API 级可用，kernel 级缺失，见正文）
nsys profile --trace=cuda,nvtx,osrt -o docs/evidence/T00/t00-smoke-nsys ./docs/evidence/T00/vector_add
nsys stats --report cuda_api_gpu_sum docs/evidence/T00/t00-smoke-nsys.nsys-rep

# SASS smoke
nvcc -cubin -arch=sm_89 -o docs/evidence/T00/t00-smoke.cubin docs/evidence/T00/vector_add.cu
cuobjdump -sass docs/evidence/T00/vector_add
nvdisasm docs/evidence/T00/t00-smoke.cubin
```
