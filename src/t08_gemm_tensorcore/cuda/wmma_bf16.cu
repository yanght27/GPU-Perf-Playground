// T08 Tensor Core —— 路径 2：CUDA C++（官方 WMMA bf16 Tensor Core）。
// 官方依据：NVIDIA cuda-samples bf16TensorCoreGemm 的 simple_wmma_bf16gemm
//（commit b7c5481c，台账 S18d），使用 nvcuda::wmma::mma_sync 与 load/store_matrix_sync。

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

using namespace nvcuda;

// 一个 warp 算一个 16x16x16 tile：最直接的 WMMA Tensor Core 路径
__global__ void wmma_bf16_gemm(const __nv_bfloat16 *A, const __nv_bfloat16 *B,
                               float *C, int M, int N, int K)
{
    int tile_row = blockIdx.y;   // 16 行为一个 tile
    int tile_col = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + tile_row * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + tile_col * 16 * K + k, K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(C + tile_row * 16 * N + tile_col * 16, acc_frag, N, wmma::mem_row_major);
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

int main()
{
    int shapes[2][3] = {{512,512,512}, {1024,1024,1024}};
    for (int si = 0; si < 2; ++si) {
        int M=shapes[si][0], N=shapes[si][1], K=shapes[si][2];
        float *hA=new float[M*K], *hB=new float[K*N], *hRef=new float[M*N], *hC=new float[M*N];
        makeInputs(hA,M*K); makeInputs(hB,K*N);
        __nv_bfloat16 *hAb=new __nv_bfloat16[M*K], *hBb=new __nv_bfloat16[K*N];
        for (int i=0;i<M*K;++i) hAb[i]=__float2bfloat16(hA[i]);
        // B 按 col_major fragment 的官方布局：Bc[n,k] = B[k,n]，leading dim K
        for (int k=0;k<K;++k) for (int n=0;n<N;++n) hBb[n*K+k]=__float2bfloat16(hB[k*N+n]);
        // 参考：用“bf16 量化后的输入”在 fp64 下计算，隔离 WMMA 正确性
        for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
            double acc=0.0;
            for (int k=0;k<K;++k) acc += (double)__bfloat162float(hAb[m*K+k]) * (double)__bfloat162float(hBb[n*K+k]);
            hRef[m*N+n]=(float)acc;
        }
        __nv_bfloat16 *dA,*dB; float *dC;
        cudaMalloc(&dA,(size_t)M*K*sizeof(__nv_bfloat16)); cudaMalloc(&dB,(size_t)K*N*sizeof(__nv_bfloat16));
        cudaMalloc(&dC,(size_t)M*N*sizeof(float));
        cudaMemcpy(dA,hAb,(size_t)M*K*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hBb,(size_t)K*N*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice);
        dim3 block(32); dim3 grid(N/16, M/16);
        auto launch=[&]{ wmma_bf16_gemm<<<grid,block>>>(dA,dB,dC,M,N,K); };
        for (int i=0;i<5;++i) launch();
        cudaDeviceSynchronize();
        cudaEvent_t st,en; cudaEventCreate(&st); cudaEventCreate(&en);
        cudaEventRecord(st); for(int i=0;i<20;++i) launch(); cudaEventRecord(en);
        cudaEventSynchronize(en); float ms=0; cudaEventElapsedTime(&ms,st,en); ms/=20;
        cudaMemcpy(hC,dC,(size_t)M*N*sizeof(float),cudaMemcpyDeviceToHost);
        float max_abs=0; for(int i=0;i<M*N;++i) max_abs=std::fmax(max_abs,std::fabs(hC[i]-hRef[i]));
        printf("[cuda_wmma_bf16] M=%d N=%d K=%d max_abs_err=%.6e tolerance=0.5 %s avg_ms=%.4f gflops=%.1f\n",
               M,N,K,max_abs, max_abs<=0.5f?"CORRECT_PASS":"CORRECT_FAIL", ms, 2.0*M*N*K/1e9/(ms/1e3));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        delete[] hA; delete[] hB; delete[] hRef; delete[] hC; delete[] hAb; delete[] hBb;
    }
    return 0;
}
