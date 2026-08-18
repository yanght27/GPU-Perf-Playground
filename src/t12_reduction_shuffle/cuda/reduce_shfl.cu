// T12 Reduction Warp Shuffle —— 路径 2：CUDA C++（本 Ticket 核心路径）。
// 官方依据：
//   - CUDA C++ Programming Guide「Warp Shuffle Functions」（台账 S10g）：__shfl_down_sync
//     从高 lane 取数、mask 语义、offset 不越界 wrap；
//   - NVIDIA cuda-samples reduction_kernel.cu 的 reduce7 形态（台账 S18h）：
//     grid-stride 每线程多元素 -> warp 内 shuffle 树 -> 每 warp 一个部分和写 shared ->
//     最后一个 warp 再 shuffle 折叠 8 个 warp 和 -> block 写一个部分和。
// 与官方 reduce7 的差异：最后 32->1 时，本实现让最终 warp 的 32 条 lane 全部参与
// （tid>=8 的 lane 补加法单位元 0），从而满足“mask 中所有被读 lane 都参与”的约束；
// 官方 reduce7 用 ballot 掩码只让 8 条 lane 参与，本实现采用等价的零填充写法。

#include <cuda_runtime.h>
#include <cstdio>
#include <random>

#define THREADS 256
#define WARPS (THREADS / 32)
#define MAX_BLOCKS 1024

// warp 内 32 个 lane 的求和：offset 从 16 折半到 1，共 5 轮。
// mask=0xffffffff 表示 warp 里 32 条 lane 全部参与本函数（调用点在统一路径上）。
__device__ __forceinline__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceShfl(const float *__restrict__ in, float *__restrict__ partial, int n)
{
    __shared__ float sdata[WARPS];   // 只要 8 个 float：每 warp 一个部分和

    unsigned tid  = threadIdx.x;
    unsigned lane = tid & 31;        // = tid % 32：线程在 warp 里的 lane id
    unsigned wid  = tid >> 5;        // = tid / 32：属于第几个 warp

    // 第 1 段：grid-stride 串行累加（与 T11 完全相同，保证对比只差第 2 段）
    unsigned i      = blockIdx.x * THREADS + tid;
    unsigned stride = gridDim.x * THREADS;
    float sum = 0.0f;
    for (; i < n; i += stride)
        sum += in[i];

    // 第 2 段 A：每个 warp 自己在寄存器里做 5 轮 shuffle 树，不碰 shared
    sum = warpReduceSum(sum);

    // 每 warp 的 lane 0 把自己的 warp 和写进 shared，一次写 8 个数
    if (lane == 0)
        sdata[wid] = sum;
    __syncthreads();   // 8 个 warp 和全部写好后，最终 warp 才能读；整个 kernel 只有这一次 barrier

    // 第 2 段 B：最后一个 warp 的 32 条 lane 一起把 8 个 warp 和折叠成 1 个
    // 先让每条 lane 都拿到有效数：前 8 条 lane 读 sdata，其余 lane 补 0（加法单位元），
    // 这样后面 5 轮 shuffle 可以使用 full mask 且读取目标一定 active。
    if (wid == 0) {
        float v = (lane < WARPS) ? sdata[lane] : 0.0f;
        v = warpReduceSum(v);
        if (lane == 0)
            partial[blockIdx.x] = v;
    }
}

static void makeInputs(float *h, int n, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 0.001f);
    for (int i = 0; i < n; ++i) h[i] = dist(gen);
}

static void runCase(int n)
{
    float *hIn = new float[n];
    makeInputs(hIn, n, 0);
    double ref = 0.0;
    for (int i = 0; i < n; ++i) ref += (double)hIn[i];  // CPU fp64 黄金参考

    float *dIn, *dPartial;
    cudaMalloc(&dIn, (size_t)n * sizeof(float));
    cudaMalloc(&dPartial, (size_t)MAX_BLOCKS * sizeof(float));
    cudaMemcpy(dIn, hIn, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + THREADS - 1) / THREADS;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    if (blocks < 1) blocks = 1;

    auto launch = [&] { reduceShfl<<<blocks, THREADS>>>(dIn, dPartial, n); };
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    ms /= 20.0f;

    float *hPartial = new float[blocks];
    cudaMemcpy(hPartial, dPartial, (size_t)blocks * sizeof(float), cudaMemcpyDeviceToHost);
    double got = 0.0;                       // 多 block 部分和在 host 上按 fp64 汇总
    for (int i = 0; i < blocks; ++i) got += (double)hPartial[i];

    double err = got > ref ? got - ref : ref - got;
    double bytes = (double)n * sizeof(float);
    printf("[cuda_reduce_shfl] N=%d blocks=%d avg_ms=%.4f read_GBps=%.2f got=%.9f ref=%.9f max_abs_err=%.6e tolerance=1e-04 %s\n",
           n, blocks, ms, bytes / (ms * 1e-3) / 1e9, got, ref, err,
           err <= 1e-4 ? "CORRECT_PASS" : "CORRECT_FAIL");

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(dIn);
    cudaFree(dPartial);
    delete[] hIn;
    delete[] hPartial;
}

int main()
{
    int shapes[3] = {1 << 20, 999983, 1};  // 2^20、素数（非整除）、单元素边界
    for (int si = 0; si < 3; ++si) runCase(shapes[si]);
    return 0;
}
