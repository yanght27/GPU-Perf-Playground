// T04 朴素 GEMM —— 路径 2：CUDA C++（无 shared memory）。
// 官方依据：CUDA Programming Guide 二维索引（台账 S10e）；cuBLAS 基线（S09a）。
// 学习变量：二维索引映射、三重循环、访存/计算比；并用官方 cuBLAS 做库基线。
// cuBLAS API 依据：docs.nvidia.com/cuda/cublas（13.3）与本地官方头文件 cublas_v2.h。

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// 朴素 kernel：每个线程负责 C 的一个元素，沿 K 串行累加。
// grid/block 都是二维：blockIdx.x/y、threadIdx.x/y 是 T04 新学的二维索引。
__global__ void gemmNaive(const float *A, const float *B, float *C,
                          int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 输出行 m
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 输出列 n
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];    // A 行主序：row*K+k；B 行主序：k*N+col
        }
        C[row * N + col] = acc;
    }
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i)
        h[i] = (float)((i % 97) - 48) * 0.1f;
}

// CPU double 参考
static void cpuRef(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)A[m * K + k] * (double)B[k * N + n];
            C[m * N + n] = (float)acc;
        }
}

static bool verify(const float *got, const float *ref, int n, const char *tag)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = std::fabs(got[i] - ref[i]);
        if (e > max_abs) max_abs = e;
    }
    printf("[%s] max_abs_err=%.6e tolerance=5e-03 %s\n",
           tag, max_abs, max_abs <= 5e-3f ? "CORRECT_PASS" : "CORRECT_FAIL");
    return max_abs <= 5e-3f;
}

static float timeNaive(const float *dA, const float *dB, float *dC, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    const int ITERS = 20;
    for (int i = 0; i < 5; ++i)
        gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < ITERS; ++i)
        gemmNaive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, s, e);
    return ms / ITERS;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;

    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float *hA = new float[M * K], *hB = new float[K * N];
    float *hC = new float[M * N], *hRef = new float[M * N];
    makeInputs(hA, M * K);
    makeInputs(hB, K * N);
    cpuRef(hA, hB, hRef, M, N, K);

    float *dA, *dB, *dC, *dBlas;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC); cudaMalloc(&dBlas, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    float msNaive = timeNaive(dA, dB, dC, M, N, K);
    cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost);
    verify(hC, hRef, M * N, "cuda_naive_gemm");
    double gflops = 2.0 * M * N * K / 1e9;
    printf("[cuda_naive_gemm] M=%d N=%d K=%d avg_ms=%.4f gflops=%.2f\n",
           M, N, K, msNaive, gflops / (msNaive / 1e3));

    // cuBLAS 基线：官方 cublasSgemm。cuBLAS 按列主序，行主序 C=A@B 等价于
    // 列主序 C^T = B^T @ A^T，所以传 n,m,k 并把 B 当第一矩阵、A 当第二矩阵。
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dBlas, N);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dBlas, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float msBlas = 0.0f;
    cudaEventElapsedTime(&msBlas, s, e);
    msBlas /= 20.0f;
    cudaMemcpy(hC, dBlas, szC, cudaMemcpyDeviceToHost);
    verify(hC, hRef, M * N, "cublas_sgemm");
    printf("[cublas_sgemm] M=%d N=%d K=%d avg_ms=%.4f gflops=%.2f\n",
           M, N, K, msBlas, gflops / (msBlas / 1e3));

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dBlas);
    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    return 0;
}
