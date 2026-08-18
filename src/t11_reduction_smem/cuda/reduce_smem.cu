// T11 Reduction 共享内存规约 —— 路径 2：CUDA C++。
// 官方依据：cuda-samples reduction_kernel.cu 的 reduce2（shared 顺序地址树规约）与
// reduce6（grid-stride 每线程多元素）形态（commit b7c5481c，台账 S18g）；
// __syncthreads 语义依据 CUDA C++ Programming Guide（台账 S10）。

#include <cuda_runtime.h>
#include <cstdio>
#include <random>

#define THREADS 256
#define MAX_BLOCKS 1024

// 每个线程先 grid-stride 累加自己负责的输入，再写 shared，
// 然后 block 内做 8 轮 tree reduction，每轮都要 __syncthreads。
__global__ void reduceSmem(const float *__restrict__ in, float *__restrict__ partial, int n)
{
    __shared__ float sdata[THREADS];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * THREADS + tid;
    unsigned stride = gridDim.x * THREADS;

    float sum = 0.0f;
    for (; i < n; i += stride)
        sum += in[i];

    sdata[tid] = sum;
    __syncthreads();  // 所有线程都把自己的部分和写进 shared 之后才能读别人

    for (unsigned s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();  // 本轮写入完成后，下一轮才能读 sdata[tid+s]
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
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

    auto launch = [&] { reduceSmem<<<blocks, THREADS>>>(dIn, dPartial, n); };
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
    printf("[cuda_reduce_smem] N=%d blocks=%d avg_ms=%.4f read_GBps=%.2f got=%.9f ref=%.9f max_abs_err=%.6e tolerance=1e-04 %s\n",
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
