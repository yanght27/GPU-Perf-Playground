// T09 Transpose 朴素版 —— 路径 2：CUDA C++。
// 官方依据：NVIDIA cuda-samples cpp/6_Performance/transpose/transpose.cu 的
// transposeNaive（commit b7c5481c，台账 S18e）。
// 两个方向：readCoalesced（读合并、写跨行）与 writeCoalesced（读跨行、写合并）。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// 官方 naive 方向：input 按行连续读（合并），output 按列写（跨行，不合并）
__global__ void transposeReadCoalesced(const float *in, float *out, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];   // B[x,y] = A[y,x]：读合并
    }
}

// 反方向：output 按行连续写（合并），input 按列读（跨行，不合并）
__global__ void transposeWriteCoalesced(const float *in, float *out, int W, int H)
{
    // 写合并方向：相邻线程对应 output 的相邻列 y；input 读地址 stride=W（跨行读）
    int y = blockIdx.x * blockDim.x + threadIdx.x;   // output 列
    int x = blockIdx.y * blockDim.y + threadIdx.y;   // output 行
    if (x < W && y < H) {
        out[x * H + y] = in[y * W + x];   // B[x,y] = A[y,x]
    }
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 97) - 48) * 0.1f;
}
static bool verify(const char *tag, const float *got, const float *in, int W, int H)
{
    float max_abs = 0.0f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            max_abs = std::fmax(max_abs, std::fabs(got[x * H + y] - in[y * W + x]));
    printf("[cuda_transpose:%s] max_abs_err=%.6e tolerance=1e-05 %s\n",
           tag, max_abs, max_abs <= 1e-5f ? "CORRECT_PASS" : "CORRECT_FAIL");
    return max_abs <= 1e-5f;
}

template <typename F>
static float timeKernel(F launch)
{
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 20; ++i) launch();
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms / 20;
}

int main()
{
    int shapes[3][2] = {{512,512},{513,257},{1,128}};
    for (int si=0; si<3; ++si) {
        int W=shapes[si][0], H=shapes[si][1];
        float *hIn=new float[W*H], *hOut=new float[W*H], *hRef=new float[W*H];
        makeInputs(hIn,W*H);
        float *dIn,*dOut;
        cudaMalloc(&dIn,(size_t)W*H*sizeof(float)); cudaMalloc(&dOut,(size_t)W*H*sizeof(float));
        cudaMemcpy(dIn,hIn,(size_t)W*H*sizeof(float),cudaMemcpyHostToDevice);
        dim3 block(16,16); dim3 grid((W+15)/16,(H+15)/16);
        auto a=[&]{ transposeReadCoalesced<<<grid,block>>>(dIn,dOut,W,H); };
        auto b=[&]{ transposeWriteCoalesced<<<grid,block>>>(dIn,dOut,W,H); };
        float ms1=timeKernel(a);
        cudaMemcpy(hOut,dOut,(size_t)W*H*sizeof(float),cudaMemcpyDeviceToHost);
        verify("readC",hOut,hIn,W,H);
        printf("[cuda_transpose_readC] W=%d H=%d avg_ms=%.4f\n", W,H,ms1);
        float ms2=timeKernel(b);
        cudaMemcpy(hOut,dOut,(size_t)W*H*sizeof(float),cudaMemcpyDeviceToHost);
        verify("writeC",hOut,hIn,W,H);
        printf("[cuda_transpose_writeC] W=%d H=%d avg_ms=%.4f\n", W,H,ms2);
        cudaFree(dIn); cudaFree(dOut);
        delete[] hIn; delete[] hOut; delete[] hRef;
    }
    return 0;
}
