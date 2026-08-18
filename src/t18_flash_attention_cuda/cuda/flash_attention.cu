// T18 FlashAttention CUDA C++（FA 风格手工映射，教学版）。
// 官方依据：T17 的 Triton FA2 算法结构（S01o、S17a）；行内 online merge 与
// shared tiling 复用 cuda-samples reduction/shuffle 思路（S18g/S10g）。
// 说明：本教学版用 CUDA Core fp32 显式实现外层 K/V tile + 内层 Q tile + online
// softmax + acc rescale，不调用 Tensor Core/WMMA；Tensor Core 加速不在 T18 任务书
// 最小增量内。BM=64, BN=32, D=64, THREADS=128：每 query 行由 2 个线程分担
// scores 半块与 acc 半块。

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <string>

#define BM 64
#define BN 32
#define D 64
#define THREADS 128

__global__ void flashAttentionCUDA(const float *__restrict__ Q,
                                   const float *__restrict__ K,
                                   const float *__restrict__ V,
                                   float *__restrict__ O,
                                   int BH, int N, float scale, int causal)
{
    __shared__ float sQ[BM * D];
    __shared__ float sK[BN * D];
    __shared__ float sV[BN * D];
    __shared__ float sS[BM * BN];
    __shared__ float rowM[BM], rowL[BM], rowA[BM];

    int q_tile = blockIdx.x;                 // Q tile：第几组 BM 行
    int bh = blockIdx.y;                     // batch*head
    int tid = threadIdx.x;
    int half = tid / BM;                     // 0/1：该行由两个线程分担
    int row = tid % BM;                      // tile 内行号
    int q = q_tile * BM + row;               // 全局 query 行

    // 载入 Q tile 到 shared（一个 Q block 固定不变；idx 拆成 tile 行/列）
    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int rq = idx / D, dq = idx % D;
        int gq = q_tile * BM + rq;
        sQ[idx] = (gq < N) ? Q[((size_t)bh * N + gq) * D + dq] : 0.0f;
    }

    // online 状态初始化
    if (tid < BM) { rowM[tid] = -FLT_MAX; rowL[tid] = 0.0f; }
    __syncthreads();

    float acc[D / 2];
    for (int j = 0; j < D / 2; ++j) acc[j] = 0.0f;

    // 外层 K/V tile 循环（FA 的核心：每个 K/V tile 只读一次）
    for (int start_n = 0; start_n < N; start_n += BN) {
        for (int idx = tid; idx < BN * D; idx += THREADS) {
            int k = idx / D, d = idx % D;
            int gk = start_n + k;
            sK[idx] = (gk < N) ? K[((size_t)bh * N + gk) * D + d] : 0.0f;
            sV[idx] = (gk < N) ? V[((size_t)bh * N + gk) * D + d] : 0.0f;
        }
        __syncthreads();

        float m = -FLT_MAX, l = 0.0f;
        for (int j = 0; j < BN / 2; ++j) {
            int k = half * (BN / 2) + j;    // 本线程负责的 16 个 key 列
            int gk = start_n + k;
            float s = -FLT_MAX;
            if (q < N && gk < N) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += sQ[row * D + d] * sK[k * D + d];
                s = dot * scale;
                if (causal && gk > q) s = -FLT_MAX;
            }
            sS[row * BN + k] = s;
            if (s == -FLT_MAX) continue;   // masked/padding：对 max/sum 贡献 0
            if (s > m) { l = l * expf(m - s) + 1.0f; m = s; }
            else       { l += expf(s - m); }
        }
        __syncthreads();

        // 每个 query 行由 1 个线程顺序合并 BN 个 score（简单、可逐行核对）
        if (tid < BM) {
            float rm = -FLT_MAX, rl = 0.0f;
            for (int kk = 0; kk < BN; ++kk) {
                float ss = sS[tid * BN + kk];
                if (ss == -FLT_MAX) continue;
                if (ss > rm) { rl = rl * expf(rm - ss) + 1.0f; rm = ss; }
                else         { rl += expf(ss - rm); }
            }
            float old_m = rowM[tid], old_l = rowL[tid];
            if (rm == -FLT_MAX) {
                rowA[tid] = 1.0f; rowM[tid] = old_m; rowL[tid] = old_l;
            } else {
                float alpha = expf(old_m - rm);
                rowM[tid] = rm; rowL[tid] = old_l * alpha + rl; rowA[tid] = alpha;
            }
        }
        __syncthreads();

        float alpha = rowA[row];
        float mnew = rowM[row];
        for (int j = 0; j < D / 2; ++j) acc[j] *= alpha;
        // 每个线程负责本行 D/2 个输出列，但要读取全部 BN 个 score
        for (int k = 0; k < BN; ++k) {
            float p = expf(sS[row * BN + k] - mnew);
            for (int d = 0; d < D / 2; ++d)
                acc[d] += p * sV[k * D + half * (D / 2) + d];
        }
        __syncthreads();   // 等所有线程用完 sK/sV 再载入下一 tile
    }

    // 最终归一化并写回
    float lsum = rowL[row];
    for (int j = 0; j < D / 2; ++j) {
        float out = acc[j] / lsum;
        int d = half * (D / 2) + j;
        if (q < N) O[((size_t)bh * N + q) * D + d] = out;
    }
}

static void makeCase(std::vector<float> &q, std::vector<float> &k,
                     std::vector<float> &v, int B, int H, int N)
{
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, 0.2f);
    size_t n = (size_t)B * H * N * D;
    q.resize(n); k.resize(n); v.resize(n);
    for (size_t i = 0; i < n; ++i) { q[i] = dist(gen); k[i] = dist(gen); v[i] = dist(gen); }
}

static void cpuRef(const std::vector<float> &q, const std::vector<float> &k,
                   const std::vector<float> &v, int B, int H, int N,
                   int causal, float scale, std::vector<float> &ref)
{
    size_t bh = (size_t)B * H;
    ref.assign(bh * N * D, 0.0f);
    for (size_t z = 0; z < bh; ++z)
    for (int qi = 0; qi < N; ++qi) {
        double m = -1e300;
        std::vector<double> s(N);
        for (int ki = 0; ki < N; ++ki) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d)
                dot += (double)q[(z * N + qi) * D + d] * (double)k[(z * N + ki) * D + d];
            s[ki] = dot * scale;
            if (causal && ki > qi) s[ki] = -1e300;
            m = std::fmax(m, s[ki]);
        }
        double sum = 0.0;
        for (int ki = 0; ki < N; ++ki) sum += std::exp(s[ki] - m);
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int ki = 0; ki < N; ++ki)
                acc += std::exp(s[ki] - m) / sum * (double)v[(z * N + ki) * D + d];
            ref[(z * N + qi) * D + d] = (float)acc;
        }
    }
}

static void runCase(int B, int H, int N, int causal, float scale, bool wall_only)
{
    size_t bh = (size_t)B * H;
    size_t n = bh * N * D;
    std::vector<float> q, k, v, ref;
    makeCase(q, k, v, B, H, N);
    if (!wall_only) cpuRef(q, k, v, B, H, N, causal, scale, ref);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, n * sizeof(float)); cudaMalloc(&dK, n * sizeof(float));
    cudaMalloc(&dV, n * sizeof(float)); cudaMalloc(&dO, n * sizeof(float));
    cudaMemcpy(dQ, q.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, k.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((N + BM - 1) / BM, (unsigned)bh);
    auto launch = [&] { flashAttentionCUDA<<<grid, THREADS>>>(dQ, dK, dV, dO, (int)bh, N, scale, causal); };
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= 20.0f;

    if (wall_only) {
        printf("[cuda_fa_t18_wall] B=%d H=%d N=%d D=%d causal=%d avg_ms=%.4f\n",
               B, H, N, D, causal, ms);
        cudaEventDestroy(s); cudaEventDestroy(e);
        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        return;
    }

    std::vector<float> o(n);
    cudaMemcpy(o.data(), dO, n * sizeof(float), cudaMemcpyDeviceToHost);
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) max_err = std::fmax(max_err, std::fabs((double)o[i] - (double)ref[i]));
    printf("[cuda_fa_t18] B=%d H=%d N=%d D=%d causal=%d avg_ms=%.4f max_abs_err=%.6e tolerance=1e-02 %s\n",
           B, H, N, D, causal, ms, max_err, max_err <= 1e-2 ? "CORRECT_PASS" : "CORRECT_FAIL");

    std::string tag = "/tmp/t18_" + std::to_string(N) + "_" + std::to_string(causal);
    auto write_bin = [&](const std::string &suffix, const void *p, size_t bytes) {
        std::ofstream f(tag + suffix, std::ios::binary); f.write((const char *)p, bytes);
    };
    write_bin("_q.bin", q.data(), n * sizeof(float));
    write_bin("_k.bin", k.data(), n * sizeof(float));
    write_bin("_v.bin", v.data(), n * sizeof(float));
    write_bin("_o.bin", o.data(), n * sizeof(float));

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

int main(int argc, char **argv)
{
    // --wall-only：跳过 CPU fp64 参考与二进制 dump，只跑多次 wall 计时。
    // 默认路径保持“正确性 + 计时 + dump”完整流程（run_t18_all.sh 使用）。
    bool wall_only = (argc > 1 && std::string(argv[1]) == "--wall-only");
    runCase(4, 4, 512, 1, 1.0f / sqrtf((float)D), wall_only);
    runCase(4, 4, 1024, 1, 1.0f / sqrtf((float)D), wall_only);
    runCase(4, 4, 2048, 1, 1.0f / sqrtf((float)D), wall_only);
    runCase(2, 4, 512, 0, 1.0f / sqrtf((float)D), wall_only);
    return 0;
}
