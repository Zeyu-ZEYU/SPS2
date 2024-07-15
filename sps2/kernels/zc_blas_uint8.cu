#include <torch/extension.h>
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include "helper_cuda.h"

// Set this to 0 to use more than 64 KB of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 KB of shared memory
// per multiprocessor when setting it to 0.
#define SHARED_MEMORY_LIMIT_64K 1

#define SHMEM_STRIDE_FOR_C 128
#define SHMEM_OFFSET_FOR_C 64

#if SHARED_MEMORY_LIMIT_64K
#define HALF_K_CHUNK_TILES 4
#define HALF_K_TILE_CHUNK_LEN 64
#define HALF_SHMEM_PADDING_SKEW 16
#define HALF_K_TILE_CHUNK_PADDED_LEN 80
#define UINT8_K_CHUNK_TILES 8
#define UINT8_K_TILE_CHUNK_LEN 128
#define UINT8_SHMEM_PADDING_SKEW 32
#define UINT8_K_TILE_CHUNK_PADDED_LEN 160
#else
#define HALF_K_CHUNK_TILES 8
#define HALF_K_TILE_CHUNK_LEN 128
#define HALF_SHMEM_PADDING_SKEW 16
#define HALF_K_TILE_CHUNK_PADDED_LEN 144
#define UINT8_K_CHUNK_TILES 16
#define UINT8_K_TILE_CHUNK_LEN 256
#define UINT8_SHMEM_PADDING_SKEW 32
#define UINT8_K_TILE_CHUNK_PADDED_LEN 288
#endif

#define checkKernelErrors(expr)                                   \
    do                                                            \
    {                                                             \
        expr;                                                     \
                                                                  \
        cudaError_t __err = cudaGetLastError();                   \
        if (__err != cudaSuccess)                                 \
        {                                                         \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                   cudaGetErrorString(__err));                    \
            abort();                                              \
        }                                                         \
    } while (0)

using namespace nvcuda;

typedef unsigned int uint;
typedef const unsigned int cuint;

__global__ void bmm_uint8_kernel(const uint8_t *A, const uint8_t *B, int *C,
                                 cuint batch_size, cuint M, cuint N, cuint K)
{
    extern __shared__ uint8_t shmem_uint8[][UINT8_K_TILE_CHUNK_PADDED_LEN];
    const int4 zero_mem = make_int4(0, 0, 0, 0);

    // Warp and lane identification.
    cuint warpId = threadIdx.x / 32;
    cuint warpThreadId = threadIdx.x % 32;

    const size_t glob_c_col_len = batch_size * M;
    cuint max_col_blocks_per_batch = (M + 128 - 1) / 128;
    cuint max_row_blocks = (N + 128 - 1) / 128;
    cuint num_blocks = gridDim.x;

    // The first half of the warps in the CTA copy the A matrix, the rest copy
    // the B matrix.
    size_t shmem_idx_for_ab = warpId < 4 ? 32 * warpId
                                         : 128 + 32 * (warpId - 4);
    shmem_idx_for_ab += warpThreadId;

    // This pointer is used to access the C matrix tiles this warp computes.
    int *shmem_ptr_c_warp = (int *)&shmem_uint8[0][0] +
                            (warpId / 2) * SHMEM_STRIDE_FOR_C * 16 * 2 +
                            (warpId % 2) * SHMEM_OFFSET_FOR_C;
    // Used for loading C from the shared memory to the global memory.
    int *shmem_ptr_c_thread = shmem_ptr_c_warp + warpThreadId * SHMEM_STRIDE_FOR_C;

    // Get the indices of the current block in the C matrix.
    uint block_id = blockIdx.x;
    size_t blk_glob_c_idx_i = block_id / max_row_blocks * 128 -
                              (block_id / max_row_blocks / max_col_blocks_per_batch) *
                                  (max_col_blocks_per_batch * 128 - M);
    size_t blk_glob_c_idx_j = block_id % max_row_blocks * 128;
    int exceed_col_boundry = (block_id / max_row_blocks) % max_col_blocks_per_batch + 1 == max_col_blocks_per_batch &&
                                     max_col_blocks_per_batch * 128 != M
                                 ? 1
                                 : 0;
    int exceed_row_boundry = block_id % max_row_blocks + 1 == max_row_blocks &&
                                     max_row_blocks * 128 != N
                                 ? 1
                                 : 0;
    // Execute per block.
    while (blk_glob_c_idx_i < glob_c_col_len)
    {
        // The first 4 warps copy A, and the last 4 warps copy B.
        uint warp_real_copy_lines = 0;
        if (warpId < 4)
        {
            if (exceed_col_boundry)
            {
                int block_tail_lines = 128 - (max_col_blocks_per_batch * 128 - M);
                if (warpId < block_tail_lines / 32)
                    warp_real_copy_lines = 32;
                else if (warpId == block_tail_lines / 32)
                    warp_real_copy_lines = block_tail_lines % 32;
            }
            else
                warp_real_copy_lines = 32;
        }
        else
        {
            if (exceed_row_boundry)
            {
                int block_tail_lines = 128 - (max_row_blocks * 128 - N);
                if (warpId - 4 < block_tail_lines / 32)
                    warp_real_copy_lines = 32;
                else if (warpId - 4 == block_tail_lines / 32)
                    warp_real_copy_lines = block_tail_lines % 32;
            }
            else
                warp_real_copy_lines = 32;
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const uint8_t *warp_gmem_ab_ptr = (warpId < 4) ? (&A[blk_glob_c_idx_i * K] +
                                                          32 * K * warpId)
                                                       : (&B[blk_glob_c_idx_j * K] +
                                                          32 * K * (warpId - 4));

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K dimension.
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag[2][4];
        // Init acc_frag with 0.
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                wmma::fill_fragment(acc_frag[i][j], 0);
            }
        }

#pragma unroll
        // Must assume K is 128, so we choose tile_k < 128 / 16.
        for (int tile_k = 0; tile_k < 8; tile_k += UINT8_K_CHUNK_TILES)
        {
            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            const int4 *gmem_ab_ptr;
            uint gmem_ab_ptr_step_len;
            if (warpThreadId < warp_real_copy_lines)
            {
                gmem_ab_ptr = (const int4 *)(warp_gmem_ab_ptr + warpThreadId * K + tile_k * 16);
                gmem_ab_ptr_step_len = 1;
            }
            else
            {
                gmem_ab_ptr = &zero_mem;
                gmem_ab_ptr_step_len = 0;
            }

            // Copy slices of the A and B matrices to shared memory.
#pragma unroll
            for (int i = 0; i < UINT8_K_CHUNK_TILES * 16 * 1 / 16; i++)
            {
                *((int4 *)&shmem_uint8[shmem_idx_for_ab][0] + i) = *gmem_ab_ptr;

                gmem_ab_ptr += gmem_ab_ptr_step_len;
            }
            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < UINT8_K_CHUNK_TILES; k_step++)
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, uint8_t, wmma::row_major> a_frag[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, uint8_t, wmma::col_major> b_frag[4];

#pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    size_t shmem_idx_a = (warpId / 2) * 16 * 2 + (i * 16);
                    const uint8_t *tile_ptr = &shmem_uint8[shmem_idx_a][k_step * 16];

                    wmma::load_matrix_sync(a_frag[i], tile_ptr, UINT8_K_TILE_CHUNK_PADDED_LEN);

#pragma unroll
                    for (int j = 0; j < 4; j++)
                    {
                        if (i == 0)
                        {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = 128 +
                                                 (4 * 16) * (warpId % 2) +
                                                 (j * 16);
                            const uint8_t *tile_ptr = &shmem_uint8[shmem_idx_b][k_step * 16];

                            wmma::load_matrix_sync(b_frag[j], tile_ptr, UINT8_K_TILE_CHUNK_PADDED_LEN);
                        }

                        wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // Store the C fragments to shared memory.
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 4; j++)
            {
                int *tile_ptr = shmem_ptr_c_warp + i * SHMEM_STRIDE_FOR_C * 16 + j * 16;

                wmma::store_matrix_sync(tile_ptr, acc_frag[i][j], SHMEM_STRIDE_FOR_C, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Now that shared memory contains all the C tiles, stream them to global
        // memory.
        int is_working_thread = 1;
        if (exceed_col_boundry)
        {
            int block_tail_lines = 128 - (max_col_blocks_per_batch * 128 - M);
            int warp_thread_block_line_id = (threadIdx.x / 64) * 32 + threadIdx.x % 32;
            if (warp_thread_block_line_id >= block_tail_lines)
                is_working_thread = 0;
        }
        // Default is (64 * sizeof(int) / 16).
        int int4_copy_count = 16;
        int single_value_copy_count = 0;
        if (is_working_thread && exceed_row_boundry)
        {
            int block_tail_lines = 128 - (max_row_blocks * 128 - N);
            if (block_tail_lines < 64)
            {
                if (warpId % 2 == 0)
                {
                    int4_copy_count = block_tail_lines * sizeof(int) / 16;
                    single_value_copy_count = block_tail_lines - int4_copy_count * 16 / sizeof(int);
                }
                else
                    int4_copy_count = 0;
            }
            else
            {
                if (warpId % 2 == 1)
                {
                    int4_copy_count = (block_tail_lines - 64) * sizeof(int) / 16;
                    single_value_copy_count = (block_tail_lines - 64) - int4_copy_count * 16 / sizeof(int);
                }
            }
        }
        const size_t gmem_idx_for_c = blk_glob_c_idx_i * N + blk_glob_c_idx_j +
                                      (warpId / 2) * 32 * N + (warpId % 2) * 64 + warpThreadId * N;
        if (is_working_thread)
        {
            int *dst_gmem_ptr_c_thread = &C[gmem_idx_for_c];
            for (int i = 0; i < int4_copy_count; i++)
            {
                *((int4 *)dst_gmem_ptr_c_thread + i) = *((int4 *)shmem_ptr_c_thread + i);
            }
            int *shmem_ptr_for_singv = shmem_ptr_c_thread + int4_copy_count * 16 / sizeof(int);
            int *dst_gmem_ptr_for_singv = dst_gmem_ptr_c_thread + int4_copy_count * 16 / sizeof(int);
            for (int i = 0; i < single_value_copy_count; i++)
            {
                *(dst_gmem_ptr_for_singv + i) = *(shmem_ptr_for_singv + i);
            }
        }
        __syncthreads();

        // Update the indices of the block in the C matrix.
        block_id += num_blocks;
        blk_glob_c_idx_i = block_id / max_row_blocks * 128 -
                           (block_id / max_row_blocks / max_col_blocks_per_batch) *
                               (max_col_blocks_per_batch * 128 - M);
        blk_glob_c_idx_j = block_id % max_row_blocks * 128;
        exceed_col_boundry = (block_id / max_row_blocks) % max_col_blocks_per_batch + 1 == max_col_blocks_per_batch &&
                                     max_col_blocks_per_batch * 128 != M
                                 ? 1
                                 : 0;
        exceed_row_boundry = block_id % max_row_blocks + 1 == max_row_blocks &&
                                     max_row_blocks * 128 != N
                                 ? 1
                                 : 0;
    } // Execute the next block.
}

torch::Tensor bmm_uint8(torch::Tensor A, torch::Tensor B)
{
    // Assumption:
    // A is in the shape of (batch_size, M, K)
    // B is in the shape of (batch_size, N, K)
    // This function is for the batched matrix multiplication: (batch_size, M, K) * (batch_size, K, N)
    // The output is a matrix C in the shape of (batch_size, M, N)
    // K is fixed to be 128
    const auto batch_size = A.size(0);
    const auto M = A.size(1);
    const auto N = B.size(1);
    const auto K = A.size(2);
    assert(K == 128);

    auto C = torch::zeros({batch_size, M, N}, torch::dtype(torch::kInt).device(torch::kCUDA));

    cuint DEVICE_ID = 0;
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, DEVICE_ID));

    const size_t SHMEM_SIZE_FOR_AB = sizeof(uint8_t) * (128 + 128) * UINT8_K_TILE_CHUNK_PADDED_LEN;
    const size_t SHMEM_SIZE_FOR_C = sizeof(int) * (128 * 128);
    const size_t SHMEM_SIZE = MAX(SHMEM_SIZE_FOR_AB, SHMEM_SIZE_FOR_C);

    // dim3 dimGrid(batch_size, (M + BLOCK_COL_LEN - 1) / BLOCK_COL_LEN, (N + BLOCK_ROW_LEN - 1) / BLOCK_ROW_LEN);
    // dim3 dimBlock(THREADS_PER_BLOCK);

    assert(deviceProp.sharedMemPerMultiprocessor >= SHMEM_SIZE);
    checkCudaErrors(cudaFuncSetAttribute(bmm_uint8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE));
    checkKernelErrors((bmm_uint8_kernel<<<deviceProp.multiProcessorCount, 8 * 32,
                                          SHMEM_SIZE>>>(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(),
                                                        C.data_ptr<int>(), batch_size, M, N, K)));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, py_module)
{
    py_module.def("bmm_uint8", &bmm_uint8, "Matrix multiplication for half.");
}
