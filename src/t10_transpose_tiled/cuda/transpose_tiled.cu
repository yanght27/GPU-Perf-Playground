// T10 Transpose Tile —— 路径 2：CUDA C++。
// 官方依据：cuda-samples transpose.cu 的 transposeCoalesced 与 transposeNoBankConflicts
//（commit b7c5481c，台账 S18f）。本文件加边界保护以支持非整除 shape。

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define TILE_DIM 32
// 与官方 transpose.cu 一致：block 为 TILE_DIM x BLOCK_ROWS=32x16=512 线程，
// 每个线程负责 TILE_DIM/BLOCK_ROWS=2 个元素；BLOCK_ROWS 必须整除 TILE_DIM。
#define BLOCK_ROWS 16

// 官方 transposeCoalesced：shared tile 无 padding，读共享时 32 线程同 bank
template <bool PAD>
__global__ void transposeTile(const float *in, float *out, int W, int H)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + (PAD ? 1 : 0)];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < W && y + i < H)
            tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * W + x];
        else
            tile[threadIdx.y + i][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // 输出坐标：交换 block/thread 方向。
    // 输出矩阵是 W 行 x H 列，行主序索引 out[行 * H + 列]：
    // ox 是输出列（0..H-1，block/thread 都换方向）；oy 是输出行（0..W-1）。
    int ox = blockIdx.y * TILE_DIM + threadIdx.x;
    int oy = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (oy + i < W && ox < H)
            out[(oy + i) * H + ox] = tile[threadIdx.x][threadIdx.y + i];
    }
}

static void makeInputs(float *h, int n)
{
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 97) - 48) * 0.1f;
}
static void verify(const char *tag, const float *got, const float *in, int W, int H)
{
    float max_abs = 0.0f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            max_abs = std::fmax(max_abs, std::fabs(got[x * H + y] - in[y * W + x]));
    printf("[cuda_transpose_tile:%s] max_abs_err=%.6e tolerance=1e-05 %s\n",
           tag, max_abs, max_abs <= 1e-5f ? "CORRECT_PASS" : "CORRECT_FAIL");
}

template <bool PAD>
static float timeTile(const float *dIn, float *dOut, int W, int H)
{
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((W + TILE_DIM - 1) / TILE_DIM, (H + TILE_DIM - 1) / TILE_DIM);
    auto launch = [&]{ transposeTile<PAD><<<grid, block>>>(dIn, dOut, W, H); };
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
    for (int si = 0; si < 3; ++si) {
        int W=shapes[si][0], H=shapes[si][1];
        float *hIn=new float[W*H], *hOut=new float[W*H];
        makeInputs(hIn,W*H);
        float *dIn,*dOut;
        cudaMalloc(&dIn,(size_t)W*H*sizeof(float)); cudaMalloc(&dOut,(size_t)W*H*sizeof(float));
        cudaMemcpy(dIn,hIn,(size_t)W*H*sizeof(float),cudaMemcpyHostToDevice);
        for (bool pad : {false, true}) {
            float ms = pad ? timeTile<true>(dIn,dOut,W,H) : timeTile<false>(dIn,dOut,W,H);
            cudaMemcpy(hOut,dOut,(size_t)W*H*sizeof(float),cudaMemcpyDeviceToHost);
            verify(pad?"pad":"nopad", hOut, hIn, W, H);
            printf("[cuda_transpose_tile:%s] W=%d H=%d avg_ms=%.4f\n", pad?"pad":"nopad", W,H,ms);
        }
        cudaFree(dIn); cudaFree(dOut); delete[] hIn; delete[] hOut;
    }
    return 0;
}
