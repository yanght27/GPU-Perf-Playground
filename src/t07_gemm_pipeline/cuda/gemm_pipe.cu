// T07 GEMM cp.async 流水线 —— 路径 2：CUDA C++。
// 官方依据：CUDA Programming Guide「Writing Tile Kernels」+ cuda_pipeline_primitives.h（台账 S11a）。
// 依据：CUDA Programming Guide「Writing Tile Kernels」的 cp.async + multistage 模式。
// 在 T06 的 float4+pad tile 基础上，用 2 组 shared buffer + cp.async 让
// “下一块 global→shared 搬运”与“当前块 FFMA 计算”重叠。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <cuda_pipeline_primitives.h>

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem)
{
    __pipeline_memcpy_async(smem, gmem, 16);
}
__device__ __forceinline__ void cp_async_commit() { __pipeline_commit(); }
template <int N> __device__ __forceinline__ void cp_async_wait() { __pipeline_wait_prior(N); }

template <int BS, int STAGES>  // BS=32, STAGES=2
__global__ void gemmPipe(const float *A, const float *B, float *C, int M, int N, int K)
{
    constexpr int V = 4;
    constexpr int COLS = BS / V;             // 8
    int tx = threadIdx.x;                    // 0..7：float4 列
    int ty = threadIdx.y;                    // 0..31：行
    int col4 = blockIdx.x * BS + tx * V;
    int row  = blockIdx.y * BS + ty;

    __shared__ float4 As4[STAGES][BS][COLS + 1];
    __shared__ float4 Bs4[STAGES][BS][COLS + 1];

    int num_tiles = (K + BS - 1) / BS;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    auto issue = [&](int tile, int buf) {
        int bk = tile * BS;
        int a_off = row * K + bk + tx * V;
        int b_off = (bk + ty) * N + col4;
        bool a_ok = (row < M && bk + tx * V + 3 < K && a_off % V == 0);
        bool b_ok = (bk + ty < K && col4 + 3 < N && b_off % V == 0);
        if (a_ok) cp_async16(&As4[buf][ty][tx], &A[a_off]);
        else {
            // 非 16B 对齐或越界：同步标量写（正确性优先）
            float4 v = make_float4(0,0,0,0);
            if (row < M) {
                v.x = bk+tx*V+0 < K ? A[a_off+0] : 0.0f;
                v.y = bk+tx*V+1 < K ? A[a_off+1] : 0.0f;
                v.z = bk+tx*V+2 < K ? A[a_off+2] : 0.0f;
                v.w = bk+tx*V+3 < K ? A[a_off+3] : 0.0f;
            }
            As4[buf][ty][tx] = v;
        }
        if (b_ok) cp_async16(&Bs4[buf][ty][tx], &B[b_off]);
        else {
            float4 v = make_float4(0,0,0,0);
            if (bk + ty < K) {
                v.x = col4+0 < N ? B[b_off+0] : 0.0f;
                v.y = col4+1 < N ? B[b_off+1] : 0.0f;
                v.z = col4+2 < N ? B[b_off+2] : 0.0f;
                v.w = col4+3 < N ? B[b_off+3] : 0.0f;
            }
            Bs4[buf][ty][tx] = v;
        }
    };

    // 主循环：先发下一块，再算当前块
    for (int tile = 0; tile < num_tiles; ++tile) {
        int buf = tile % STAGES;
        if (tile + 1 < num_tiles) {
            issue(tile + 1, (tile + 1) % STAGES);   // 异步搬运下一块
            cp_async_commit();
        }
        if (tile > 0) cp_async_wait<STAGES - 1>();  // 等“最多 STAGES-1 组在飞”，即当前块已就绪
        if (tile == 0) {
            // 第一块没有任何异步在飞，先搬自己再等
            issue(0, 0);
            cp_async_commit();
            cp_async_wait<0>();
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            int kidx = k / V, lane = k % V;
            float a = (lane == 0) ? As4[buf][ty][kidx].x :
                      (lane == 1) ? As4[buf][ty][kidx].y :
                      (lane == 2) ? As4[buf][ty][kidx].z : As4[buf][ty][kidx].w;
            float b0 = Bs4[buf][k][tx].x;
            float b1 = Bs4[buf][k][tx].y;
            float b2 = Bs4[buf][k][tx].z;
            float b3 = Bs4[buf][k][tx].w;
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
static void verify(const float *got, const float *ref, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) max_abs = std::fmax(max_abs, std::fabs(got[i]-ref[i]));
    printf("[cuda_gemm_pipe] max_abs_err=%.6e tolerance=5e-03 %s\n",
           max_abs, max_abs <= 5e-3f ? "CORRECT_PASS" : "CORRECT_FAIL");
}

int main()
{
    int shapes[][3] = {{17,31,33},{512,512,512},{1024,1024,1024}};
    for (auto &s : shapes) {
        int M=s[0], N=s[1], K=s[2];
        float *hA=new float[M*K], *hB=new float[K*N], *hRef=new float[M*N], *hC=new float[M*N];
        makeInputs(hA,M*K); makeInputs(hB,K*N); cpuRef(hA,hB,hRef,M,N,K);
        float *dA,*dB,*dC;
        cudaMalloc(&dA,(size_t)M*K*sizeof(float)); cudaMalloc(&dB,(size_t)K*N*sizeof(float));
        cudaMalloc(&dC,(size_t)M*N*sizeof(float));
        cudaMemcpy(dA,hA,(size_t)M*K*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB,(size_t)K*N*sizeof(float),cudaMemcpyHostToDevice);
        dim3 block(8,32); dim3 grid((N+3)/4/8,(M+31)/32);
        auto launch = [&]{ gemmPipe<32,2><<<grid,block>>>(dA,dB,dC,M,N,K); };
        for (int i=0;i<5;++i) launch();
        cudaDeviceSynchronize();
        cudaEvent_t st,en; cudaEventCreate(&st); cudaEventCreate(&en);
        cudaEventRecord(st);
        for (int i=0;i<20;++i) launch();
        cudaEventRecord(en); cudaEventSynchronize(en);
        float ms=0; cudaEventElapsedTime(&ms,st,en); ms/=20;
        cudaMemset(dC,0,(size_t)M*N*sizeof(float));
        launch(); cudaDeviceSynchronize();
        cudaMemcpy(hC,dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost);
        verify(hC,hRef,M*N);
        printf("[cuda_gemm_pipe] M=%d N=%d K=%d avg_ms=%.4f gflops=%.1f\n",
               M,N,K,ms, 2.0*M*N*K/1e9/(ms/1e3));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hRef; delete[] hC;
    }
    return 0;
}
