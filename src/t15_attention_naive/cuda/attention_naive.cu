// T15 Attention 朴素前向 —— 路径 2：CUDA C++（本 Ticket 核心路径）。
// 官方依据：算法结构对齐 PyTorch eager 公式与 Triton 官方 tutorial 06-fused-attention
// 的 qk/pv/softmax 步骤（台账 S01o、S15e）；行 max/sum 规约沿用 cuda-samples
// reduction 的 reduce7 warp-shuffle 形态（S18g、S10g）；expf 语义 CUDA Math API（S10j）。
// 明确不做 IO-Aware KV 分块（那是 T17/T18）：每个 query 行一个 block，重复读整段 K/V。
// 内存布局：[B,H,N,D] 行主序；bh = b*H+h 为 batch-head 平铺下标。

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <random>

#define THREADS 128
#define MAX_N 128   // 朴素版最大序列长（block 线程上限）
#define MAX_D 128

__global__ void attentionNaive(const float *__restrict__ Q,
                               const float *__restrict__ K,
                               const float *__restrict__ V,
                               float *__restrict__ O,
                               float scale, int N, int D, int causal)
{
    __shared__ float s[MAX_N];        // 一个 query 的原始 S 行（scores，全程不覆盖）

    int q  = blockIdx.x;              // 查询位置：每个 query 行一个 block
    int bh = blockIdx.y;              // batch*heads 平铺下标
    int tid = threadIdx.x;

    const float *qrow = Q + ((size_t)bh * N + q) * D;

    // 计算图第 1 步：S_k = Q_q · K_k * scale
    s[tid] = -FLT_MAX;
    if (tid < N) {
        float dot = 0.0f;
        const float *krow = K + ((size_t)bh * N + tid) * D;
        for (int d = 0; d < D; ++d)
            dot += qrow[d] * krow[d];
        s[tid] = dot * scale;
    }

    // 计算图第 2 步：causal mask（k > q 的位置设为 -inf）
    if (causal && tid > q)
        s[tid] = -FLT_MAX;
    __syncthreads();

    // 计算图第 3 步：行 softmax（T12 shuffle 树 + 4 warp 交接，避开全块树）
    int lane = tid & 31;
    int wid  = tid >> 5;
    float val = s[tid];

    for (int off = 16; off > 0; off >>= 1) {          // warp 内 max
        float o = __shfl_down_sync(0xffffffff, val, off);
        val = fmaxf(val, o);
    }
    __shared__ float smax[THREADS / 32], ssum[THREADS / 32];
    if (lane == 0) smax[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float mm = (lane < THREADS / 32) ? smax[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_down_sync(0xffffffff, mm, off);
            mm = fmaxf(mm, o);
        }
        if (lane == 0) smax[0] = mm;
    }
    __syncthreads();
    float m = smax[0];

    val = expf(s[tid] - m);                           // warp 内 sum
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffff, val, off);
        val += o;
    }
    if (lane == 0) ssum[wid] = val;
    __syncthreads();
    if (wid == 0) {
        float ss = (lane < THREADS / 32) ? ssum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_down_sync(0xffffffff, ss, off);
            ss += o;
        }
        if (lane == 0) ssum[0] = ss;
    }
    __syncthreads();
    float denom = ssum[0];

    s[tid] = expf(s[tid] - m) / denom;                // s 现在成为 attention 权重 p
    __syncthreads();                                  // 所有权重就绪后，输出线程才能读整行

    // 计算图第 4 步：O_q = Σ_k p_k V_k
    if (tid < D) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += s[k] * V[((size_t)bh * N + k) * D + tid];
        O[((size_t)bh * N + q) * D + tid] = acc;
    }
}

static void makeCase(float *q, float *k, float *v, int B, int H, int N, int D, int which)
{
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, which == 0 ? 0.5f : (which == 1 ? 0.3f : 1.0f));
    size_t total = (size_t)B * H * N * D;
    for (size_t i = 0; i < total; ++i) { q[i] = dist(gen); k[i] = dist(gen); v[i] = dist(gen); }
}

static void runCase(int B, int H, int N, int D, int causal, double scale, int which)
{
    size_t bh = (size_t)B * H;
    size_t n = bh * N * D;
    float *hQ = new float[n], *hK = new float[n], *hV = new float[n], *hO = new float[n];
    makeCase(hQ, hK, hV, B, H, N, D, which);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, n * sizeof(float)); cudaMalloc(&dK, n * sizeof(float));
    cudaMalloc(&dV, n * sizeof(float)); cudaMalloc(&dO, n * sizeof(float));
    cudaMemcpy(dQ, hQ, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N, (unsigned)bh);
    auto launch = [&] { attentionNaive<<<grid, THREADS>>>(dQ, dK, dV, dO, (float)scale, N, D, causal); };
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    int iters = (which == 0) ? 20 : 1;
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= iters;

    cudaMemcpy(hO, dO, n * sizeof(float), cudaMemcpyDeviceToHost);

    // fp64 eager 参考：S = QK^T*scale -> mask -> softmax -> PV
    double max_err = 0.0;
    for (size_t b = 0; b < (size_t)B; ++b)
    for (size_t h = 0; h < (size_t)H; ++h)
    for (int qi = 0; qi < N; ++qi) {
        double sm[MAX_N];
        for (int ki = 0; ki < N; ++ki) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += (double)hQ[((b*H+h)*N+qi)*D+d] * (double)hK[((b*H+h)*N+ki)*D+d];
            sm[ki] = dot * scale;
            if (causal && ki > qi) sm[ki] = -1e300;
        }
        double m = -1e300;
        for (int ki = 0; ki < N; ++ki) m = std::fmax(m, sm[ki]);
        double sum = 0.0;
        for (int ki = 0; ki < N; ++ki) sum += std::exp(sm[ki] - m);
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int ki = 0; ki < N; ++ki)
                acc += std::exp(sm[ki] - m) / sum * (double)hV[((b*H+h)*N+ki)*D+d];
            double diff = std::fabs(acc - (double)hO[((b*H+h)*N+qi)*D+d]);
            if (diff > max_err) max_err = diff;
        }
    }

    printf("[cuda_attention_naive] B=%d H=%d N=%d D=%d causal=%d scale=%.6f avg_ms=%.4f max_abs_err=%.6e tolerance=1e-04 %s\n",
           B, H, N, D, causal, scale, ms, max_err,
           max_err <= 1e-4 ? "CORRECT_PASS" : "CORRECT_FAIL");

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO;
}

int main()
{
    runCase(2, 2, 64, 32, 1, 1.0 / sqrt(32.0), 0);   // 主 shape：causal
    runCase(1, 1, 37, 17, 0, 0.5, 1);                // 未对齐 + non-causal + 显式 scale
    runCase(1, 1, 1, 8, 1, 1.0 / sqrt(8.0), 2);      // N=1 边界
    return 0;
}
