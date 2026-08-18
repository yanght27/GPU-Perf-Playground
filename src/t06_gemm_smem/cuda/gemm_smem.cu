// T06 GEMM 共享内存优化 —— 路径 2：CUDA C++。
// 官方依据：CUDA Programming Guide「Maximize Memory Throughput」（台账 S10f）；bank/vector 原则。
// 三档对比：A) T05 基线(BS=16, stride=16 -> 2-way bank conflict)
//          B) padding 版(BS=16, stride=17 -> 无 conflict)
//          C) float4 版(BS=32, 128-bit STS/LDS)
// 依据：CUDA Programming Guide「Maximize Memory Throughput」的 bank/vector 原则。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// A) 基线：As[16][16]，stride=16；As[ty][k] 中 ty 与 ty+2 落到同一 bank。
template <int BS>
__global__ void gemmBase(const float *A, const float *B, float *C, int M, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BS + tx;
    int row = blockIdx.y * BS + ty;
    __shared__ float As[BS][BS], Bs[BS][BS];
    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BS) {
        As[ty][tx] = (row < M && bk + tx < K) ? A[row * K + bk + tx] : 0.0f;
        Bs[ty][tx] = (bk + ty < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// B) padding：As[16][17]；stride=17 与 32 互质，16 个 ty 各占不同 bank。
template <int BS>
__global__ void gemmPad(const float *A, const float *B, float *C, int M, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BS + tx;
    int row = blockIdx.y * BS + ty;
    __shared__ float As[BS][BS + 1];
    __shared__ float Bs[BS][BS + 1];
    float acc = 0.0f;
    for (int bk = 0; bk < K; bk += BS) {
        As[ty][tx] = (row < M && bk + tx < K) ? A[row * K + bk + tx] : 0.0f;
        Bs[ty][tx] = (bk + ty < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BS; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// C) float4：BS=32，block=(8,32)。每线程一条 128-bit 指令搬 4 个元素进 shared。
template <int BS>   // BS == 32
__global__ void gemmVec4(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int V = 4;
    constexpr int COLS = BS / V;                    // 8 个 float4 每行
    int tx = threadIdx.x;                           // 0..7：列方向的第 tx 个 float4
    int ty = threadIdx.y;                           // 0..31：行
    int col4 = blockIdx.x * BS + tx * V;            // 本线程负责的 4 个全局列
    int row  = blockIdx.y * BS + ty;                // 全局行

    __shared__ float4 As4[BS][COLS];
    __shared__ float4 Bs4[BS][COLS];

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int bk = 0; bk < K; bk += BS) {
        // 128-bit 从 global 搬 A/B 各 4 个连续元素；非整除时给 0
        int a_off = row * K + bk + tx * V;
        float4 av;
        if (a_off % V == 0 && row < M && bk + tx * V + 3 < K)
            av = *reinterpret_cast<const float4 *>(&A[a_off]);
        else if (row < M) {
            av = make_float4(bk + tx*V + 0 < K ? A[a_off + 0] : 0.0f,
                             bk + tx*V + 1 < K ? A[a_off + 1] : 0.0f,
                             bk + tx*V + 2 < K ? A[a_off + 2] : 0.0f,
                             bk + tx*V + 3 < K ? A[a_off + 3] : 0.0f);
        } else av = make_float4(0,0,0,0);
        int b_off = (bk + ty) * N + col4;
        float4 bv;
        if (b_off % V == 0 && bk + ty < K && col4 + 3 < N)
            bv = *reinterpret_cast<const float4 *>(&B[b_off]);
        else if (bk + ty < K) {
            bv = make_float4(col4 + 0 < N ? B[b_off + 0] : 0.0f,
                             col4 + 1 < N ? B[b_off + 1] : 0.0f,
                             col4 + 2 < N ? B[b_off + 2] : 0.0f,
                             col4 + 3 < N ? B[b_off + 3] : 0.0f);
        } else bv = make_float4(0,0,0,0);
        As4[ty][tx] = av;                            // STS.128
        Bs4[ty][tx] = bv;                            // STS.128
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            int kidx = k / V, lane = k % V;          // 从 float4 里挑第 lane 个分量
            float a = (lane == 0) ? As4[ty][kidx].x :
                      (lane == 1) ? As4[ty][kidx].y :
                      (lane == 2) ? As4[ty][kidx].z : As4[ty][kidx].w;
            float b0 = Bs4[k][tx].x;
            float b1 = Bs4[k][tx].y;
            float b2 = Bs4[k][tx].z;
            float b3 = Bs4[k][tx].w;
            acc.x += a * b0;
            acc.y += a * b1;
            acc.z += a * b2;
            acc.w += a * b3;
        }
        __syncthreads();
    }
    int c_off = row * N + col4;
    if (row < M && c_off % V == 0 && col4 + 3 < N) {
        *reinterpret_cast<float4 *>(&C[c_off]) = acc;
    } else if (row < M) {
        float *out = &C[c_off];
        if (col4 + 0 < N) out[0] = acc.x;
        if (col4 + 1 < N) out[1] = acc.y;
        if (col4 + 2 < N) out[2] = acc.z;
        if (col4 + 3 < N) out[3] = acc.w;
    }
}

// D) float4 + padding：Bs4 行宽 9（与 32 互质），消除 C) 中实测到的 LDS bank conflict。
template <int BS>   // BS == 32
__global__ void gemmVecPad(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int V = 4;
    constexpr int COLS = BS / V;                    // 8 个 float4 每行
    int tx = threadIdx.x;                           // 0..7
    int ty = threadIdx.y;                           // 0..31
    int col4 = blockIdx.x * BS + tx * V;
    int row  = blockIdx.y * BS + ty;

    __shared__ float4 As4[BS][COLS + 1];            // padding：stride=9
    __shared__ float4 Bs4[BS][COLS + 1];

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int bk = 0; bk < K; bk += BS) {
        int a_off = row * K + bk + tx * V;
        float4 av;
        if (a_off % V == 0 && row < M && bk + tx * V + 3 < K)
            av = *reinterpret_cast<const float4 *>(&A[a_off]);
        else if (row < M)
            av = make_float4(bk + tx*V + 0 < K ? A[a_off + 0] : 0.0f,
                             bk + tx*V + 1 < K ? A[a_off + 1] : 0.0f,
                             bk + tx*V + 2 < K ? A[a_off + 2] : 0.0f,
                             bk + tx*V + 3 < K ? A[a_off + 3] : 0.0f);
        else av = make_float4(0,0,0,0);
        int b_off = (bk + ty) * N + col4;
        float4 bv;
        if (b_off % V == 0 && bk + ty < K && col4 + 3 < N)
            bv = *reinterpret_cast<const float4 *>(&B[b_off]);
        else if (bk + ty < K)
            bv = make_float4(col4 + 0 < N ? B[b_off + 0] : 0.0f,
                             col4 + 1 < N ? B[b_off + 1] : 0.0f,
                             col4 + 2 < N ? B[b_off + 2] : 0.0f,
                             col4 + 3 < N ? B[b_off + 3] : 0.0f);
        else bv = make_float4(0,0,0,0);
        As4[ty][tx] = av;
        Bs4[ty][tx] = bv;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            int kidx = k / V, lane = k % V;
            float a = (lane == 0) ? As4[ty][kidx].x :
                      (lane == 1) ? As4[ty][kidx].y :
                      (lane == 2) ? As4[ty][kidx].z : As4[ty][kidx].w;
            float b0 = Bs4[k][tx].x;
            float b1 = Bs4[k][tx].y;
            float b2 = Bs4[k][tx].z;
            float b3 = Bs4[k][tx].w;
            acc.x += a * b0; acc.y += a * b1; acc.z += a * b2; acc.w += a * b3;
        }
        __syncthreads();
    }
    int c_off = row * N + col4;
    if (row < M && c_off % V == 0 && col4 + 3 < N)
        *reinterpret_cast<float4 *>(&C[c_off]) = acc;
    else if (row < M) {
        float *out = &C[c_off];
        if (col4 + 0 < N) out[0] = acc.x;
        if (col4 + 1 < N) out[1] = acc.y;
        if (col4 + 2 < N) out[2] = acc.z;
        if (col4 + 3 < N) out[3] = acc.w;
    }
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 97) - 48) * 0.1f;
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
static void verify(const float *got, const float *ref, int n, const char *tag)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) max_abs = std::fmax(max_abs, std::fabs(got[i]-ref[i]));
    printf("[%s] max_abs_err=%.6e tolerance=5e-03 %s\n",
           tag, max_abs, max_abs <= 5e-3f ? "CORRECT_PASS" : "CORRECT_FAIL");
}

template <typename F>
static float timeKernel(F launch, int iters = 20)
{
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0.0f; cudaEventElapsedTime(&ms, s, e);
    return ms / iters;
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
        cudaMemcpy(dA,hA,(size_t)M*K*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB,(size_t)K*N*sizeof(float),cudaMemcpyHostToDevice);

        auto run = [&](auto kernel, const char *tag) {
            cudaMemset(dC, 0, (size_t)M*N*sizeof(float));
            kernel();
            cudaMemcpy(hC,dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost);
            verify(hC,hRef,M*N,tag);
        };

        dim3 b16(16,16); dim3 g16((N+15)/16,(M+15)/16);
        float ms = timeKernel([&]{ gemmBase<16><<<g16,b16>>>(dA,dB,dC,M,N,K); });
        run([&]{ gemmBase<16><<<g16,b16>>>(dA,dB,dC,M,N,K); }, "t06_base16");
        printf("[t06_base16] M=%d N=%d K=%d avg_ms=%.4f\n", M,N,K,ms);

        ms = timeKernel([&]{ gemmPad<16><<<g16,b16>>>(dA,dB,dC,M,N,K); });
        run([&]{ gemmPad<16><<<g16,b16>>>(dA,dB,dC,M,N,K); }, "t06_pad16");
        printf("[t06_pad16] M=%d N=%d K=%d avg_ms=%.4f\n", M,N,K,ms);

        // float4 版只在 N 方向需要 4 的倍数做快速路径；其余走标量尾，仍正确
        dim3 b32(8,32); dim3 g32((N+3)/4/8, (M+31)/32);
        ms = timeKernel([&]{ gemmVec4<32><<<g32,b32>>>(dA,dB,dC,M,N,K); });
        run([&]{ gemmVec4<32><<<g32,b32>>>(dA,dB,dC,M,N,K); }, "t06_vec4_32");
        printf("[t06_vec4_32] M=%d N=%d K=%d avg_ms=%.4f\n", M,N,K,ms);

        ms = timeKernel([&]{ gemmVecPad<32><<<g32,b32>>>(dA,dB,dC,M,N,K); });
        run([&]{ gemmVecPad<32><<<g32,b32>>>(dA,dB,dC,M,N,K); }, "t06_vecpad_32");
        printf("[t06_vecpad_32] M=%d N=%d K=%d avg_ms=%.4f\n", M,N,K,ms);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hRef; delete[] hC;
    }
    return 0;
}
