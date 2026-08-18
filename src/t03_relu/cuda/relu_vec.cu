// T03 ReLU 向量化版 —— 路径 2：CUDA C++ float4。
// 官方依据：CUDA vector_types.h + Programming Guide「Maximize Memory Throughput」（台账 S10c/S10d）。
// 学习变量：合并访问与 128-bit 向量化 load/store；对齐与尾部处理。
// float4 是 CUDA 官方内置向量类型（vector_types.h）；内存访问语义见
// CUDA Programming Guide「Maximize Memory Throughput」。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// 主 kernel：每个线程一次处理 4 个 float（16 字节）。
// in/out 都按 float4 解释，要求指针 16 字节对齐（cudaMalloc 保证满足）。
__global__ void reluVecKernel(const float4 *in, float4 *out, int nVec)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;   // 索引单位变成“向量”，不再是单元素
    if (i < nVec) {
        float4 v = in[i];           // 一条 128-bit load：ld.global.v4.b32
        // 用 isnan 传播 NaN，与 PyTorch F.relu(NaN)=NaN 的参考语义一致；
        // fmaxf 是 IEEE maxNum 语义，会把 NaN 抹成 0，不能单独用。
        v.x = isnan(v.x) ? v.x : fmaxf(v.x, 0.0f);
        v.y = isnan(v.y) ? v.y : fmaxf(v.y, 0.0f);
        v.z = isnan(v.z) ? v.z : fmaxf(v.z, 0.0f);
        v.w = isnan(v.w) ? v.w : fmaxf(v.w, 0.0f);
        out[i] = v;                 // 一条 128-bit store：st.global.v4.b32
    }
}

// 尾部 kernel：N 不被 4 整除时，剩余 <4 个元素用标量补。
__global__ void reluTailKernel(const float *in, float *out, int start, int n)
{
    int i = start + blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = isnan(x) ? x : (x > 0.0f ? x : 0.0f);
    }
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i)
        h[i] = (i % 7 == 0) ? 0.0f : (float)((i % 97) - 48) * 0.5f;
}

static bool verify(const float *input, const float *output, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        double e = (double)input[i] > 0.0 ? (double)input[i] : 0.0;
        float d = std::fabs((float)e - output[i]);
        if (d > max_abs) max_abs = d;
    }
    printf("[cuda_relu_vec] max_abs_err=%.6e tolerance=1e-05 %s\n",
           max_abs, max_abs <= 1e-5f ? "CORRECT_PASS" : "CORRECT_FAIL");
    return max_abs <= 1e-5f;
}

static float runAligned(const float *d_in, float *d_out, int n, int block)
{
    int nVec = n / 4;                              // 主循环只处理整组 float4
    int grid = (nVec + block - 1) / block;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    const int ITERS = 100;
    for (int k = 0; k < 10; ++k)                   // warmup
        reluVecKernel<<<grid, block>>>((const float4 *)d_in, (float4 *)d_out, nVec);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int k = 0; k < ITERS; ++k)
        reluVecKernel<<<grid, block>>>((const float4 *)d_in, (float4 *)d_out, nVec);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e); ms /= ITERS;
    printf("[cuda_relu_vec] n=%d grid=%d block=%d avg_ms=%.4f effective_gbps=%.1f\n",
           n, grid, block, ms, 2.0 * n * sizeof(float) / 1e9 / (ms / 1e3));
    return ms;
}

int main(int argc, char **argv)
{
    const int N = 1 << 20;
    const int N_ODD = 1000003;                     // 测试尾部
    const int block = (argc > 1) ? atoi(argv[1]) : 256;

    float *h_in = new float[N], *h_out = new float[N];
    makeInputs(h_in, N);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void **)&d_in, N * sizeof(float));
    cudaMalloc((void **)&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    runAligned(d_in, d_out, N, block);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_in, h_out, N);

    // 尾部用例：向量主循环 + 标量尾 kernel
    cudaFree(d_in); cudaFree(d_out);
    float *h_odd = new float[N_ODD], *h_odd_out = new float[N_ODD];
    makeInputs(h_odd, N_ODD);
    cudaMalloc((void **)&d_in, N_ODD * sizeof(float));
    cudaMalloc((void **)&d_out, N_ODD * sizeof(float));
    cudaMemcpy(d_in, h_odd, N_ODD * sizeof(float), cudaMemcpyHostToDevice);
    int nVec = N_ODD / 4, tailStart = nVec * 4;
    reluVecKernel<<<(nVec + block - 1) / block, block>>>((const float4 *)d_in, (float4 *)d_out, nVec);
    int tail = N_ODD - tailStart;
    if (tail > 0)
        reluTailKernel<<<(tail + block - 1) / block, block>>>(d_in, d_out, tailStart, N_ODD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odd_out, d_out, N_ODD * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_odd, h_odd_out, N_ODD);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out; delete[] h_odd; delete[] h_odd_out;

    // 极值用例：±Inf、NaN、±1e38，语义与 PyTorch F.relu 对齐（NaN 传播）
    const int NE = 5;
    float h_e[NE] = {INFINITY, -INFINITY, NAN, 1e38f, -1e38f};
    float h_e_out[NE] = {0, 0, 0, 0, 0};
    float *d_e = nullptr, *d_e_out = nullptr;
    cudaMalloc((void **)&d_e, NE * sizeof(float));
    cudaMalloc((void **)&d_e_out, NE * sizeof(float));
    cudaMemcpy(d_e, h_e, NE * sizeof(float), cudaMemcpyHostToDevice);
    reluTailKernel<<<1, 32>>>(d_e, d_e_out, 0, NE);
    cudaDeviceSynchronize();
    cudaMemcpy(h_e_out, d_e_out, NE * sizeof(float), cudaMemcpyDeviceToHost);
    float expect[NE] = {INFINITY, 0.0f, NAN, 1e38f, 0.0f};
    bool ok = true;
    for (int i = 0; i < NE; ++i) {
        if (isnan(expect[i])) ok = ok && isnan(h_e_out[i]);
        else ok = ok && (h_e_out[i] == expect[i]);
    }
    printf("[cuda_relu_vec_extreme] %s out=%g %g %g %g %g\n",
           ok ? "CORRECT_PASS" : "CORRECT_FAIL",
           (double)h_e_out[0], (double)h_e_out[1], (double)h_e_out[2],
           (double)h_e_out[3], (double)h_e_out[4]);
    cudaFree(d_e); cudaFree(d_e_out);
    return 0;
}
