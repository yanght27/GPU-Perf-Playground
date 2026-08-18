// T14 Softmax Online/融合版 —— 路径 2：CUDA C++（本 Ticket 核心路径）。
// 官方依据：
//   - Online normalizer calculation for softmax（arXiv:1805.02867）的 running max/sum
//     rescale 更新式（台账 S21）；官方实现形态对齐 CuTe DSL tutorial 06_softmax.py 的
//     Kernel 5/6/8 online 变体（台账 S02n）与 Triton tutorial 02 的 fused softmax（S01m）；
//   - expf 语义 CUDA Math API（S10j）；warp shuffle 合并沿用 cuda-samples
//     reduction 的 reduce7 形态（S18g）与 Programming Guide shuffle 语义（S10g）。
// 与 T13 的差别：整行先一次读进 shared（global 读 1 遍），online 归约在 shared/shuffle
// 上做，最后从 shared 读回归一化写 global（写 1 遍）=> 1R+1W；
// T13 是 3R+1W。CUDA 官方 cuda-samples 没有 softmax sample，故结构取最接近的官方
// online/fused 示例（CuTe/Triton），并在台账中记 N/A 原因。

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <random>

#define THREADS 128
#define WARPS (THREADS / 32)
#define MAX_C 4096   // T13/T14 统一门禁的最大行宽；整行缓存进 shared

__device__ __forceinline__ void onlineMerge(float &m, float &l, float om, float ol)
{
    // 合并两段 (m,l) 与 (om,ol)：大 max 为主，小 max 一侧的和按比例 rescale。
    if (om > m) {
        l = ol + l * expf(m - om);
        m = om;
    } else {
        l = l + ol * expf(om - m);
    }
}

__global__ void softmaxOnline(const float *__restrict__ x, float *__restrict__ y, int C)
{
    __shared__ float srow[MAX_C];        // 整行缓存：16KB，global 只读一次
    __shared__ float smax[WARPS];        // 每 warp 一个 online (m,l) 对
    __shared__ float ssum[WARPS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 融合第 1 步：把整行一次读进 shared（唯一一次 global 读）
    for (int c = tid; c < C; c += THREADS)
        srow[c] = x[row * C + c];
    __syncthreads();                     // 整行就绪后才能开始归约

    // 融合第 2 步：每线程对 shared 里的行做 online 单遍扫描
    float m = -FLT_MAX, l = 0.0f;
    for (int c = tid; c < C; c += THREADS) {
        float v = srow[c];
        if (v > m) {                     // 出现新 max：旧和按 exp(m_old-m_new) 缩放，+1 计入新元素
            l = l * expf(m - v) + 1.0f;
            m = v;
        } else {
            l += expf(v - m);
        }
    }

    // 融合第 3 步：先 warp 内 shuffle 合并 32 个 (m,l)，再把 4 个 warp 合并
    for (int offset = 16; offset > 0; offset >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, offset);
        float ol = __shfl_down_sync(0xffffffff, l, offset);
        onlineMerge(m, l, om, ol);
    }
    if (lane == 0) { smax[wid] = m; ssum[wid] = l; }
    __syncthreads();

    if (wid == 0) {                      // 最终 warp 把 4 个 warp 对合并；其余 lane 补单位元
        float mm = (lane < WARPS) ? smax[lane] : -FLT_MAX;
        float ll = (lane < WARPS) ? ssum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float om = __shfl_down_sync(0xffffffff, mm, offset);
            float ol = __shfl_down_sync(0xffffffff, ll, offset);
            onlineMerge(mm, ll, om, ol);
        }
        if (lane == 0) { smax[0] = mm; ssum[0] = ll; }
    }
    __syncthreads();

    float row_max = smax[0], row_sum = ssum[0];

    // 融合第 4 步：从 shared 读回归一化，写 global（唯一一次 global 写）
    for (int c = tid; c < C; c += THREADS)
        y[row * C + c] = expf(srow[c] - row_max) / row_sum;
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
    if (which == 0) {
        for (int c = 0; c < cols; ++c) h[0 * cols + c] = 7.0f;
        h[1 * cols + 0] = -1000.0f; h[1 * cols + 1] = 1000.0f;
        h[2 * cols + 0] = 1000.0f;  h[2 * cols + 1] = -1000.0f;
    } else if (which == 1) {
        for (int c = 0; c < cols; ++c) h[0 * cols + c] = -7.0f;
        h[1 * cols + 0] = 1000.0f; h[1 * cols + 1] = -1000.0f;
        h[2 * cols + 0] = -1000.0f; h[2 * cols + 1] = 1000.0f;
    } else {
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

    auto launch = [&] { softmaxOnline<<<rows, THREADS>>>(dX, dY, cols); };
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
        double m = -1e300;
        for (int c = 0; c < cols; ++c) m = std::fmax(m, (double)hX[r * cols + c]);
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += std::exp((double)hX[r * cols + c] - m);
        for (int c = 0; c < cols; ++c) {
            double ref = std::exp((double)hX[r * cols + c] - m) / sum;
            double diff = std::fabs(ref - (double)hY[r * cols + c]);
            if (diff > max_err) max_err = diff;
        }
    }
    double bytes_moved = (double)n * 4.0 * 2.0;  // 1R + 1W
    printf("[cuda_softmax_online] R=%d C=%d avg_ms=%.4f effective_GBps=%.2f max_abs_err=%.6e tolerance=1e-05 %s\n",
           rows, cols, ms, bytes_moved / (ms * 1e-3) / 1e9, max_err,
           max_err <= 1e-5 ? "CORRECT_PASS" : "CORRECT_FAIL");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dX); cudaFree(dY);
    delete[] hX; delete[] hY;
}

int main()
{
    runCase(1024, 4096, 0);
    runCase(37, 999, 1);
    runCase(1, 1, 2);
    return 0;
}
