// T16 KV Cache —— 路径 2：CUDA C++（核心路径）。
// 官方依据：Transformers DynamicCache 的 [B,H,seq,D] 增量 append 语义（S16a）；
// 注意力计算图沿用 T15（S01o/S15e）；softmax 规约沿用 cuda-samples reduce7
// warp shuffle 形态（S18g/S10g）。
// 布局：X/Q/K/V 均为 [BH,T,D] 行主序，BH = B*H。LPRE 个 token 做 prefill，
// 之后 DEC 步 decode。有 cache：每步只投影 1 个 K/V token 并写入 cache；
// 无 cache：每步重投影 0..t 的全部历史 K/V。

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#define THREADS 512
#define B 2
#define H 2
#define T 512
#define D 64
#define LPRE 256
#define DEC (T - LPRE)
#define BH (B * H)

// 线性投影 Y[pos] = X[pos] @ W：每个线程算一个输出元素。
// count=1 且 dst 为 cache 时就是“投影+append”一步。
__global__ void projectLinear(const float *__restrict__ X,
                              const float *__restrict__ W,
                              float *__restrict__ Y,
                              int start, int count, int nT)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * D * BH;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int dout = idx % D;
        int tmp  = idx / D;
        int pos  = tmp % count;
        int bh   = tmp / count;
        float acc = 0.0f;
        const float *xrow = X + ((size_t)bh * T + start + pos) * D;
        const float *wcol = W + dout;
        for (int din = 0; din < D; ++din)
            acc += xrow[din] * wcol[din * D];
        Y[((size_t)bh * T + start + pos) * D + dout] = acc;
    }
}

// 单 query decode 注意力：Q 在位置 q，K/V 前缀长度 N=q+1（cache 或无 cache 的完整前缀）。
__global__ void attentionDecode(const float *__restrict__ Q,
                                const float *__restrict__ K,
                                const float *__restrict__ V,
                                float *__restrict__ O,
                                int q, int N, int nD)
{
    __shared__ volatile float s[THREADS];
    __shared__ float smax[THREADS / 32], ssum[THREADS / 32];

    int bh = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;
    const float *qrow = Q + ((size_t)bh * T + q) * D;

    s[tid] = -FLT_MAX;
    if (tid < N) {
        const float *krow = K + ((size_t)bh * T + tid) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qrow[d] * krow[d];
        s[tid] = dot * (1.0f / sqrtf((float)D));
    }
    __syncthreads();

    float val = s[tid];
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_down_sync(0xffffffff, val, off);
        val = fmaxf(val, o);
    }
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

    val = expf(s[tid] - m);
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

    s[tid] = expf(s[tid] - m) / denom;
    __syncthreads();

    if (tid < D) {
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += s[k] * V[((size_t)bh * T + k) * D + tid];
        O[((size_t)bh * T + q) * D + tid] = acc;
    }
}

static void makeData(std::vector<float> &X, std::vector<float> &WQ,
                     std::vector<float> &WK, std::vector<float> &WV)
{
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    X.resize((size_t)BH * T * D);
    WQ.resize((size_t)D * D); WK.resize((size_t)D * D); WV.resize((size_t)D * D);
    for (auto &v : X) v = dist(gen);
    float ws = 1.0f / sqrtf((float)D);
    for (auto &v : WQ) v = dist(gen) * ws;
    for (auto &v : WK) v = dist(gen) * ws;
    for (auto &v : WV) v = dist(gen) * ws;
}

static void cpuProject(const std::vector<float> &X, const std::vector<float> &W,
                       std::vector<double> &Y)
{
    Y.assign((size_t)BH * T * D, 0.0);
    for (int bh = 0; bh < BH; ++bh)
    for (int t = 0; t < T; ++t)
    for (int dout = 0; dout < D; ++dout) {
        double acc = 0.0;
        for (int din = 0; din < D; ++din)
            acc += (double)X[((size_t)bh * T + t) * D + din] * (double)W[(size_t)din * D + dout];
        Y[((size_t)bh * T + t) * D + dout] = acc;
    }
}

static std::vector<float> runDecode(bool use_cache,
                                    const float *dX, const float *dWQ,
                                    const float *dWK, const float *dWV,
                                    size_t n, float *ms)
{
    size_t nBH = (size_t)BH * T * D;
    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, nBH * sizeof(float));
    cudaMalloc(&dK, nBH * sizeof(float));
    cudaMalloc(&dV, nBH * sizeof(float));
    cudaMalloc(&dO, nBH * sizeof(float));
    // Q 只投影一次：两种模式相同
    int blocks = ((size_t)T * D * BH + 255) / 256;
    projectLinear<<<blocks, 256>>>(dX, dWQ, dQ, 0, T, T);

    if (use_cache) {   // prefill：一次性投影 LPRE 个历史 token 写入 cache（不进 decode 计时）
        int bp = ((size_t)LPRE * D * BH + 255) / 256;
        projectLinear<<<bp, 256>>>(dX, dWK, dK, 0, LPRE, T);
        projectLinear<<<bp, 256>>>(dX, dWV, dV, 0, LPRE, T);
        cudaDeviceSynchronize();
    }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int t = LPRE; t < T; ++t) {
        if (use_cache) {                       // 只投影当前 1 个 token 并 append 进 cache
            int b1 = (D * BH + 255) / 256;
            projectLinear<<<b1, 256>>>(dX, dWK, dK, t, 1, T);
            projectLinear<<<b1, 256>>>(dX, dWV, dV, t, 1, T);
            attentionDecode<<<dim3(1, BH), THREADS>>>(dQ, dK, dV, dO, t, t + 1, D);
        } else {                               // 重投影 0..t 的全部历史 K/V
            int bp = ((size_t)(t + 1) * D * BH + 255) / 256;
            projectLinear<<<bp, 256>>>(dX, dWK, dK, 0, t + 1, T);
            projectLinear<<<bp, 256>>>(dX, dWV, dV, 0, t + 1, T);
            attentionDecode<<<dim3(1, BH), THREADS>>>(dQ, dK, dV, dO, t, t + 1, D);
        }
    }
    cudaEventRecord(e); cudaEventSynchronize(e);
    cudaEventElapsedTime(ms, s, e);
    std::vector<float> hO(n);
    cudaMemcpy(hO.data(), dO, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return hO;
}

static double compareRef(const std::vector<float> &O,
                         const std::vector<double> &Q, const std::vector<double> &K,
                         const std::vector<double> &V)
{
    double max_err = 0.0;
    for (int bh = 0; bh < BH; ++bh)
    for (int t = LPRE; t < T; ++t)
    for (int dout = 0; dout < D; ++dout) {
        double m = -1e300;
        for (int k = 0; k <= t; ++k) {
            double dot = 0.0;
            for (int din = 0; din < D; ++din)
                dot += Q[((size_t)bh * T + t) * D + din] * K[((size_t)bh * T + k) * D + din];
            m = std::fmax(m, dot / sqrt((double)D));
        }
        double sum = 0.0;
        for (int k = 0; k <= t; ++k) {
            double dot = 0.0;
            for (int din = 0; din < D; ++din)
                dot += Q[((size_t)bh * T + t) * D + din] * K[((size_t)bh * T + k) * D + din];
            sum += std::exp(dot / sqrt((double)D) - m);
        }
        double acc = 0.0;
        for (int k = 0; k <= t; ++k) {
            double dot = 0.0;
            for (int din = 0; din < D; ++din)
                dot += Q[((size_t)bh * T + t) * D + din] * K[((size_t)bh * T + k) * D + din];
            double p = std::exp(dot / sqrt((double)D) - m) / sum;
            acc += p * V[((size_t)bh * T + k) * D + dout];
        }
        double diff = std::fabs(acc - (double)O[((size_t)bh * T + t) * D + dout]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int main()
{
    std::vector<float> X, WQ, WK, WV;
    makeData(X, WQ, WK, WV);
    std::vector<double> Q, K, V;
    cpuProject(X, WQ, Q); cpuProject(X, WK, K); cpuProject(X, WV, V);
    size_t n = (size_t)BH * T * D;

    float *dX, *dWQ, *dWK, *dWV;
    cudaMalloc(&dX, X.size() * sizeof(float));
    cudaMalloc(&dWQ, WQ.size() * sizeof(float));
    cudaMalloc(&dWK, WK.size() * sizeof(float));
    cudaMalloc(&dWV, WV.size() * sizeof(float));
    cudaMemcpy(dX, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dWQ, WQ.data(), WQ.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dWK, WK.data(), WK.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dWV, WV.data(), WV.size() * sizeof(float), cudaMemcpyHostToDevice);

    float ms_cache = 0, ms_no = 0;
    auto O_cache = runDecode(true, dX, dWQ, dWK, dWV, n, &ms_cache);
    auto O_no = runDecode(false, dX, dWQ, dWK, dWV, n, &ms_no);

    double err_cache = compareRef(O_cache, Q, K, V);
    double err_no = compareRef(O_no, Q, K, V);
    double diff_cn = 0.0;
    for (size_t i = 0; i < n; ++i) diff_cn = std::fmax(diff_cn, std::fabs((double)O_cache[i] - (double)O_no[i]));
    printf("[cuda_t16] cache_vs_ref_err=%.6e no_cache_vs_ref_err=%.6e cache_vs_nocache_diff=%.6e tolerance=1e-04 %s\n",
           err_cache, err_no, diff_cn, (err_cache <= 1e-4 && err_no <= 1e-4 && diff_cn <= 1e-4) ? "CORRECT_PASS" : "CORRECT_FAIL");
    printf("[cuda_t16_timing] no_cache_ms=%.3f cache_ms=%.3f speedup=%.2fx cache_bytes=%zu no_cache_projected_kv_elems=%zu\n",
           ms_no, ms_cache, ms_no / ms_cache, (size_t)2 * BH * T * D * 4,
           (size_t)2 * BH * D * (LPRE * DEC + DEC * (DEC + 1) / 2));

    cudaFree(dX); cudaFree(dWQ); cudaFree(dWK); cudaFree(dWV);
    return 0;
}
