// T01 Vector Add —— 路径 2：CUDA C++
// 官方依据：NVIDIA cuda-samples vectorAdd（台账 S18a）+ CUDA Programming Guide（S10a）。
// 核心逻辑与 NVIDIA 官方 cuda-samples 的 vectorAdd 一致（官方文件：
// cpp/0_Introduction/vectorAdd/vectorAdd.cu，commit b7c5481c）。
// 本文件去掉了官方 sample 的 helper 依赖，只保留“最能说明问题”的主线。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// GPU 上执行的函数：每个线程算一个元素。
// __global__ = “这是 kernel，由 CPU 启动、在 GPU 上跑”。
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    // 全局下标 = 当前 block 在 grid 中的编号 * 每个 block 的线程数 + 当前线程在 block 中的编号
    // blockIdx.x / blockDim.x / threadIdx.x 是 CUDA 内建变量（grid、block、thread 三层结构）。
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // N 不一定是线程总数的整数倍：超出范围的下标不能访问。
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 20;                 // 和 Python 路径一致的规模
    const int threadsPerBlock = 256;       // 一个 block 放 256 个线程（官方 sample 同款配置）
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 向上取整

    // 1) 分配 CPU 内存并生成确定性的输入（公式固定，任何机器都能复现）
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    double *h_ref = new double[N];         // CPU 上的 double 黄金参考
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)((i % 97) * 0.5);
        h_B[i] = (float)(((i + 3) % 89) * 0.25);
        h_ref[i] = (double)h_A[i] + (double)h_B[i];
    }

    // 2) 在 GPU 显存里分配同样三个数组（CPU 和 GPU 内存是分开的两块物理空间）
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));
    cudaMalloc((void **)&d_C, N * sizeof(float));

    // 3) 把输入从 CPU 内存拷到 GPU 显存
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4) 启动 kernel：<<<grid 有多少 block, 每个 block 有多少线程>>>
    //    这就是 CUDA 的 “launch 配置”，T01 的核心学习变量。
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 5) 计时：CUDA event 只量 GPU 上的 kernel 时间（不含 memcpy）。
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int ITERS = 100;
    for (int k = 0; k < 10; ++k) vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int k = 0; k < ITERS; ++k)
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;
    double gbps = 3.0 * N * sizeof(float) / 1e9 / (ms / 1e3);

    // 6) 把结果拷回 CPU，与 double 参考比较
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    float max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = std::fabs((float)h_ref[i] - h_C[i]);
        if (e > max_abs) max_abs = e;
    }
    printf("[cuda] max_abs_err=%.6e tolerance=1e-05 %s\n",
           max_abs, max_abs <= 1e-5f ? "CORRECT_PASS" : "CORRECT_FAIL");
    printf("[cuda] avg_ms=%.4f effective_gbps=%.1f\n", ms, gbps);
    printf("[cuda] grid=%d block=%d\n", blocksPerGrid, threadsPerBlock);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_ref;
    return 0;
}
