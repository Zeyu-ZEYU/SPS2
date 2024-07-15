#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_WIDTH 32 // Tile width, can be tuned for different scenarios

__global__ void bmm_half_kernel(const at::Half *A, const at::Half *B, at::Half *C, int bs, int a, int b, int c)
{
    __shared__ half As[TILE_WIDTH][TILE_WIDTH];
    __shared__ half Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z, tx = threadIdx.x, ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    half Cvalue = __float2half(0.0);

    for (int t = 0; t < (b - 1) / TILE_WIDTH + 1; ++t)
    {
        if (Row < a && t * TILE_WIDTH + tx < b)
            As[ty][tx] = __float2half(A[bz * a * b + Row * b + t * TILE_WIDTH + tx]);
        else
            As[ty][tx] = __float2half(0.0);

        if (Col < c && t * TILE_WIDTH + ty < b)
            Bs[ty][tx] = __float2half(B[bz * b * c + (t * TILE_WIDTH + ty) * c + Col]);
        else
            Bs[ty][tx] = __float2half(0.0);

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Cvalue = __hadd(Cvalue, __hmul(As[ty][k], Bs[k][tx]));

        __syncthreads();
    }

    if (Row < a && Col < c)
        C[bz * a * c + Row * c + Col] = __half2float(Cvalue);
}

torch::Tensor bmm_half(torch::Tensor A, torch::Tensor B)
{
    const auto bs = A.size(0);
    const auto a = A.size(1);
    const auto b = A.size(2);
    const auto c = B.size(2);

    auto C = torch::zeros({bs, a, c}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((c + TILE_WIDTH - 1) / TILE_WIDTH, (a + TILE_WIDTH - 1) / TILE_WIDTH, bs);

    bmm_half_kernel<<<dimGrid, dimBlock>>>(A.data_ptr<at::Half>(), B.data_ptr<at::Half>(), C.data_ptr<at::Half>(), bs, a, b, c);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, py_module)
{
    py_module.def("call", &bmm_half, "Tiled batch matrix multiplication for half precision with CUDA.");
}
