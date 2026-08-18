// T13 Softmax 朴素 3-pass —— 路径 2：CUDA C++（本 Ticket 核心路径）。
// 官方依据：
//   - 算法结构 3-pass（max -> sum(exp) -> normalize）对齐 Triton 官方
//     tutorial 02-fused-softmax 的 naive_softmax（台账 S01m）；
//   - 每行一个 block 的 shared 树规约沿用 NVIDIA cuda-samples reduction 的
//     reduce2 形态（台账 S18g）；expf 语义依据 CUDA Math API Single Precision
//     Functions（台账 S10j）。
// 每行三次读 global（max 一遍、exp-sum 一遍、normalize 一遍）+ 一次写 global。

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <random>

#define THREADS 128

__global__ void softmaxNaive(const float *__restrict__ x, float *__restrict__ y, int C)
{
    __shared__ float sdata[THREADS];
    int row = blockIdx.x;                     // 一行一个 block
    int tid = threadIdx.x;

    // pass 1：求行最大值 m（先减 m，exp 才不溢出）
    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += THREADS) {
        float v = x[row * C + c];
        local_max = fmaxf(local_max, v);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float m = sdata[0];

    // pass 2：再读一遍输入，求 exp(x-m) 的行和
    float local_sum = 0.0f;
    for (int c = tid; c < C; c += THREADS) {
        float v = x[row * C + c];
        local_sum += expf(v - m);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float denom = sdata[0];

    // pass 3：读第三遍，写 softmax 输出
    for (int c = tid; c < C; c += THREADS) {
        float v = x[row * C + c];
        y[row * C + c] = expf(v - m) / denom;
    }
}

static void fillRandom(float *h, int rows, int cols, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < rows * cols; ++i) h[i] = dist(gen);
}

static void makeCase(float *h, int rows, int cols, int which)
{
    fillRandom(h, rows, cols, 0);
    if (which == 0) {                         // 主 shape：全相同行 + ±1000 极值
        for (int c = 0; c < cols; ++c) h[0 * cols + c] = 7.0f;
        h[1 * cols + 0] = -1000.0f; h[1 * cols + 1] = 1000.0f;
        h[2 * cols + 0] = 1000.0f;  h[2 * cols + 1] = -1000.0f;
    } else if (which == 1) {                  // 未对齐列数 + 全相同负值行
        for (int c = 0; c < cols; ++c) h[0 * cols + c] = -7.0f;
        h[1 * cols + 0] = 1000.0f; h[1 * cols + 1] = -1000.0f;
        h[2 * cols + 0] = -1000.0f; h[2 * cols + 1] = 1000.0f;
    } else {                                  // N=1：单元素边界
        h[0] = 1000.0f;
    }
}

static void runCase(int rows, int cols, int which)
{
    size_t n = (size_t)rows * cols;
    float *hX = new float[n];
    float *hY = new float[n];
    makeCase(hX, rows, cols, which);

    float *dX, *dY;
    cudaMalloc(&dX, n * sizeof(float));
    cudaMalloc(&dY, n * sizeof(float));
    cudaMemcpy(dX, hX, n * sizeof(float), cudaMemcpyHostToDevice);

    auto launch = [&] { softmaxNaive<<<rows, THREADS>>>(dX, dY, cols); };
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    int iters = (which == 0) ? 50 : 1;
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    cudaMemcpy(hY, dY, n * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (int r = 0; r < rows; ++r) {
        double m = -1e300;                    // fp64 CPU 稳定 softmax 参考
        for (int c = 0; c < cols; ++c) m = std::fmax(m, (double)hX[r * cols + c]);
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += std::exp((double)hX[r * cols + c] - m);
        for (int c = 0; c < cols; ++c) {
            double ref = std::exp((double)hX[r * cols + c] - m) / sum;
            double diff = std::fabs(ref - (double)hY[r * cols + c]);
            if (diff > max_err) max_err = diff;
        }
    }
    double bytes_moved = (double)n * 4.0 * 4.0;  // 3 读 + 1 写
    printf("[cuda_softmax_naive] R=%d C=%d avg_ms=%.4f effective_GBps=%.2f max_abs_err=%.6e tolerance=1e-05 %s\n",
           rows, cols, ms, bytes_moved / (ms * 1e-3) / 1e9, max_err,
           max_err <= 1e-5 ? "CORRECT_PASS" : "CORRECT_FAIL");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dX); cudaFree(dY);
    delete[] hX; delete[] hY;
}

int main()
{
    runCase(1024, 4096, 0);  // 主基线 shape（列大，3 遍 global 读接近带宽瓶颈）
    runCase(37, 999, 1);     // 行/列都未对齐 + 极值 + 全相同行
    runCase(1, 1, 2);        // N=1 边界
    return 0;
}
