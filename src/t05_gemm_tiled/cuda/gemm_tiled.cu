// T05 GEMM Shared-Memory Tiling —— 路径 2：CUDA C++。
// 官方依据：NVIDIA cuda-samples matrixMul（台账 S18c）。
// 结构对齐 NVIDIA 官方 cuda-samples matrixMul（BLOCK_SIZE 模板、As/Bs shared tile、
// 两次 __syncthreads）；在此基础上加了非整除边界的 zero-fill 保护。
// 官方文件：cpp/0_Introduction/matrixMul/matrixMul.cu，commit b7c5481c。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

template <int BS>
__global__ void gemmTiled(const float *A, const float *B, float *C,
                          int M, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BS + tx;   // 输出列
    int row = blockIdx.y * BS + ty;   // 输出行

    __shared__ float As[BS][BS];      // A 的 tile 缓存
    __shared__ float Bs[BS][BS];      // B 的 tile 缓存

    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BS) {
        // 每个线程往 shared memory 搬一个元素；越界处填 0（边界保护）
        As[ty][tx] = (row < M && bk + tx < K) ? A[row * K + bk + tx] : 0.0f;
        Bs[ty][tx] = (bk + ty < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();              // 1) 全 block 都搬完才开始算
        #pragma unroll
        for (int k = 0; k < BS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();              // 2) 都算完才能覆盖下一块
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i)
        h[i] = (float)((i % 97) - 48) * 0.1f;
}

static void cpuRef(const float *A, const float *B, float *C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += (double)A[m*K+k] * (double)B[k*N+n];
            C[m*N+n] = (float)acc;
        }
}

static bool verify(const float *got, const float *ref, int n, const char *tag)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) max_abs = std::fmax(max_abs, std::fabs(got[i]-ref[i]));
    printf("[%s] max_abs_err=%.6e tolerance=5e-03 %s\n",
           tag, max_abs, max_abs <= 5e-3f ? "CORRECT_PASS" : "CORRECT_FAIL");
    return max_abs <= 5e-3f;
}

template <int BS>
static void runShape(int M, int N, int K, float *hA, float *hB, float *hRef,
                     float *hC, float *dA, float *dB, float *dC)
{
    dim3 block(BS, BS);
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    cudaMemcpy(dA, hA, (size_t)M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, (size_t)K*N*sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < 5; ++i)
        gemmTiled<BS><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i)
        gemmTiled<BS><<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= 20;
    cudaMemcpy(hC, dC, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost);
    verify(hC, hRef, M*N, "cuda_gemm_tiled");
    double gflops = 2.0*M*N*K/1e9;
    printf("[cuda_gemm_tiled] M=%d N=%d K=%d BS=%d avg_ms=%.4f gflops=%.1f\n",
           M, N, K, BS, ms, gflops/(ms/1e3));
}

int main()
{
    int shapes[][3] = {{17,31,33},{512,512,512},{1024,1024,1024}};
    for (auto &s : shapes) {
        int M=s[0], N=s[1], K=s[2];
        float *hA=new float[M*K], *hB=new float[K*N], *hRef=new float[M*N], *hC=new float[M*N];
        makeInputs(hA, M*K); makeInputs(hB, K*N); cpuRef(hA,hB,hRef,M,N,K);
        float *dA,*dB,*dC;
        cudaMalloc(&dA,(size_t)M*K*sizeof(float)); cudaMalloc(&dB,(size_t)K*N*sizeof(float));
        cudaMalloc(&dC,(size_t)M*N*sizeof(float));
        runShape<16>(M,N,K,hA,hB,hRef,hC,dA,dB,dC);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hRef; delete[] hC;
    }
    return 0;
}
