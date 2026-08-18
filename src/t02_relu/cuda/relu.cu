// T02 ReLU 标量版 —— 路径 2：CUDA C++。
// 官方依据：cuda-samples vectorAdd 骨架（台账 S18b）+ CUDA Programming Guide（S10b）。
// 学习变量：元素级索引、Grid 配置、边界处理、以及 if 分支在 GPU 上的行为。
// 基础框架沿用官方 cuda-samples vectorAdd（commit b7c5481c）的 host/device 流程。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ReLU(x) = x > 0 ? x : 0
__global__ void reluKernel(const float *input, float *output, int n)
{
    // 和 T01 完全相同的全局下标公式：这是元素级 kernel 的公共骨架
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // 边界保护：N 不一定是线程总数的整数倍
    if (i < n) {
        // 每个线程独立做一次判断；标量版没有向量化
        output[i] = isnan(input[i]) ? input[i] : (input[i] > 0.0f ? input[i] : 0.0f);
    }
}

// 生成所有路径共用的确定性输入：i%7==0 处精确为 0，其余有正有负
static void makeInputs(float *h_A, int n)
{
    for (int i = 0; i < n; ++i) {
        h_A[i] = (i % 7 == 0) ? 0.0f : (float)((i % 97) - 48) * 0.5f;
    }
}

static bool verify(const float *input, const float *output, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        double expected = (double)input[i] > 0.0 ? (double)input[i] : 0.0;
        float e = std::fabs((float)expected - output[i]);
        if (e > max_abs) max_abs = e;
    }
    printf("[cuda_relu] max_abs_err=%.6e tolerance=1e-05 %s\n",
           max_abs, max_abs <= 1e-5f ? "CORRECT_PASS" : "CORRECT_FAIL");
    return max_abs <= 1e-5f;
}

static void runKernel(const float *d_in, float *d_out, int n, int block, float *ms)
{
    int grid = (n + block - 1) / block;      // ceil 除法：不整除时必须多开一个 block
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int ITERS = 100;
    for (int k = 0; k < 10; ++k) reluKernel<<<grid, block>>>(d_in, d_out, n);  // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int k = 0; k < ITERS; ++k) reluKernel<<<grid, block>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(ms, start, stop);
    *ms /= ITERS;
    printf("[cuda_relu] n=%d grid=%d block=%d avg_ms=%.4f effective_gbps=%.1f\n",
           n, grid, block, *ms, 2.0 * n * sizeof(float) / 1e9 / (*ms / 1e3));
}

int main(int argc, char **argv)
{
    const int N = 1 << 20;
    const int N_ODD = 1000003;   // 特意取一个“别扭”的长度，逼出边界路径
    const int block = (argc > 1) ? atoi(argv[1]) : 256;  // 用于对比不同 grid 配置

    float *h_in = new float[N];
    float *h_out = new float[N];
    makeInputs(h_in, N);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void **)&d_in, N * sizeof(float));
    cudaMalloc((void **)&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    float ms = 0.0f;
    runKernel(d_in, d_out, N, block, &ms);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_in, h_out, N);

    // 边界用例：N_ODD 不是任何线程配置的整数倍
    cudaFree(d_in);
    cudaFree(d_out);
    float *h_odd = new float[N_ODD];
    float *h_odd_out = new float[N_ODD];
    makeInputs(h_odd, N_ODD);
    cudaMalloc((void **)&d_in, N_ODD * sizeof(float));
    cudaMalloc((void **)&d_out, N_ODD * sizeof(float));
    cudaMemcpy(d_in, h_odd, N_ODD * sizeof(float), cudaMemcpyHostToDevice);
    int grid_odd = (N_ODD + block - 1) / block;
    reluKernel<<<grid_odd, block>>>(d_in, d_out, N_ODD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odd_out, d_out, N_ODD * sizeof(float), cudaMemcpyDeviceToHost);
    verify(h_odd, h_odd_out, N_ODD);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in; delete[] h_out; delete[] h_odd; delete[] h_odd_out;
    return 0;
}
