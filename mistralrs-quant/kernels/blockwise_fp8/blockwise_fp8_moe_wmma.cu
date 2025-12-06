/**
 * @brief WMMA-based Blockwise FP8 MoE GEMM kernel.
 *
 * This kernel uses Tensor Core WMMA operations for efficient FP8 MoE GEMM.
 * Weights are dequantized on-the-fly before WMMA operations.
 *
 * Based on the moe_gemm_wmma.cu kernel structure.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

using namespace nvcuda::wmma;

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

// WMMA tile sizes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
using VecT = float4;

constexpr int VEC_SIZE = 8;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32;

constexpr int M_BLK = WARPS_M * WMMA_M;
constexpr int N_BLK = WARPS_N * WMMA_N;
constexpr int K_BLK = WMMA_K;

// Helper: FP8 to float
__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
    return __half2float(__nv_cvt_fp8_to_halfraw(val.__x, __NV_E4M3));
}

// Count tokens per expert kernel
static __global__ void count_tokens_per_expert_kernel(
    const int32_t* expert_ids,
    int32_t* expert_counts,
    int size_m
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size_m) {
        int32_t expert_id = expert_ids[i];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Calculate expert offsets
static void calculate_expert_offsets_fp8(
    const int32_t* d_expert_ids,
    int size_m,
    int32_t* d_expert_offsets,
    int num_experts,
    cudaStream_t stream
) {
    int32_t* d_expert_counts;
    cudaMallocAsync(&d_expert_counts, num_experts * sizeof(int32_t), stream);
    cudaMemsetAsync(d_expert_counts, 0, num_experts * sizeof(int32_t), stream);

    int threads = 256;
    int blocks = (size_m + threads - 1) / threads;
    count_tokens_per_expert_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_ids, d_expert_counts, size_m
    );

    thrust::device_ptr<const int32_t> d_counts_ptr(d_expert_counts);
    thrust::device_ptr<int32_t> d_offsets_ptr(d_expert_offsets);

    thrust::inclusive_scan(
        thrust::cuda::par.on(stream),
        d_counts_ptr,
        d_counts_ptr + num_experts,
        d_offsets_ptr + 1
    );

    cudaMemsetAsync(d_expert_offsets, 0, sizeof(int32_t), stream);
    cudaFreeAsync(d_expert_counts, stream);
}

/**
 * @brief WMMA-based Blockwise FP8 MoE GEMM kernel
 *
 * @param input            [size_m, size_k]
 * @param weights          [num_experts, size_n, size_k] in FP8
 * @param weight_scales    [num_experts, size_n/block_y, size_k/block_x]
 * @param sorted_token_ids [size_m]
 * @param expert_offsets   [num_experts + 1]
 * @param topk_weights     [size_m] optional
 * @param output           [size_m, size_n]
 */
template<typename T>
__global__ void blockwise_fp8_moe_gemm_wmma_kernel(
    const T* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weights,
    const float* __restrict__ weight_scales,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    T* __restrict__ output,
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k,
    const int weight_block_size_y,
    const int weight_block_size_x,
    const int scale_n_blocks,
    const int scale_k_blocks
) {
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const __nv_fp8_e4m3* expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;
    const float* expert_scale = weight_scales + (size_t)expert_id * scale_n_blocks * scale_k_blocks;

    extern __shared__ uint8_t smem_bytes[];

    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    T* B_sh = reinterpret_cast<T*>(A_sh + M_BLK * K_BLK);
    uint8_t* C_ptr = reinterpret_cast<uint8_t*>(B_sh + N_BLK * K_BLK);

    size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
    if (offset != 0) {
        C_ptr += (alignof(float) - offset);
    }
    float* C_sh = reinterpret_cast<float*>(C_ptr);

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int laneId = threadId % 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
    const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE;
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // Load B tile (weights) with on-the-fly dequantization
            for (int i = threadId; i < N_BLK * K_BLK; i += BLOCK_THREADS) {
                int n_local = i / K_BLK;
                int k_local = i % K_BLK;

                int n_global = n_base + n_local;
                int k_global = k_base + k_local;

                if (n_global < size_n && k_global < size_k) {
                    // Get scale for this block
                    int scale_y = n_global / weight_block_size_y;
                    int scale_x = k_global / weight_block_size_x;
                    float scale = expert_scale[scale_y * scale_k_blocks + scale_x];

                    __nv_fp8_e4m3 w_fp8 = expert_w[(size_t)n_global * size_k + k_global];
                    float w_dequant = fp8_to_float(w_fp8) * scale;
                    B_sh[n_local * K_BLK + k_local] = static_cast<T>(w_dequant);
                } else {
                    B_sh[n_local * K_BLK + k_local] = T(0);
                }
            }

            // Load A tile (inputs)
            for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE;
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;

                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg;
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = token_index / (topk_weights ? 1 : topk);
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) =
                        *reinterpret_cast<const VecT*>(&input[(size_t)input_index * size_k + k_global]);
                } else {
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = zero_vec;
                }
            }

            __syncthreads();

            // Compute using WMMA
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

            const T* A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
            const T* B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

            load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
            load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

            mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        // Store results
        float* C_sh_ptr = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

        __syncthreads();

        // Write to global memory
        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
            int m_local_c = i / N_BLK;
            int n_local_c = i % N_BLK;

            int m_seg = m_base + m_local_c;
            int n_global = n_base + n_local_c;

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c];
                    if (topk_weights) {
                        val *= topk_weights[token_index];
                    }
                    output[(size_t)token_index * size_n + n_global] = static_cast<T>(val);
                }
            }
        }

        __syncthreads();
    }
}

// Transposed variant for stacked format [E, K, N]
template<typename T>
__global__ void blockwise_fp8_moe_gemm_wmma_transposed_kernel(
    const T* __restrict__ input,
    const __nv_fp8_e4m3* __restrict__ weights,      // [num_experts, size_k, size_n] transposed
    const float* __restrict__ weight_scales,        // [num_experts, size_k/block_y, size_n/block_x]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_offsets,
    const float* __restrict__ topk_weights,
    T* __restrict__ output,
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k,
    const int weight_block_size_y,                  // Block size along K
    const int weight_block_size_x,                  // Block size along N
    const int scale_k_blocks,
    const int scale_n_blocks
) {
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id < 0 || expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;

    if (num_rows_in_segment == 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    // Transposed layout: [E, K, N]
    const __nv_fp8_e4m3* expert_w = weights + (size_t)expert_id * (size_t)size_k * (size_t)size_n;
    const float* expert_scale = weight_scales + (size_t)expert_id * scale_k_blocks * scale_n_blocks;

    extern __shared__ uint8_t smem_bytes[];

    T* A_sh = reinterpret_cast<T*>(smem_bytes);
    T* B_sh = reinterpret_cast<T*>(A_sh + M_BLK * K_BLK);
    uint8_t* C_ptr = reinterpret_cast<uint8_t*>(B_sh + N_BLK * K_BLK);

    size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
    if (offset != 0) {
        C_ptr += (alignof(float) - offset);
    }
    float* C_sh = reinterpret_cast<float*>(C_ptr);

    const int threadId = threadIdx.x;
    const int warpId = threadId / 32;
    const int warp_m_idx = warpId / WARPS_N;
    const int warp_n_idx = warpId % WARPS_N;

    const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
    const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE;
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        fill_fragment(c_frag, 0.0f);

        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // Load B tile with transposed dequantization
            for (int i = threadId; i < N_BLK * K_BLK; i += BLOCK_THREADS) {
                int n_local = i / K_BLK;
                int k_local = i % K_BLK;

                int n_global = n_base + n_local;
                int k_global = k_base + k_local;

                if (n_global < size_n && k_global < size_k) {
                    // Transposed: weight[k, n] = expert_w[k * size_n + n]
                    int scale_y = k_global / weight_block_size_y;
                    int scale_x = n_global / weight_block_size_x;
                    float scale = expert_scale[scale_y * scale_n_blocks + scale_x];

                    __nv_fp8_e4m3 w_fp8 = expert_w[(size_t)k_global * size_n + n_global];
                    float w_dequant = fp8_to_float(w_fp8) * scale;
                    B_sh[n_local * K_BLK + k_local] = static_cast<T>(w_dequant);
                } else {
                    B_sh[n_local * K_BLK + k_local] = T(0);
                }
            }

            // Load A tile (inputs)
            for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
                int idx = i * VEC_SIZE;
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;

                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;

                if (m_seg < num_rows_in_segment && k_global < size_k) {
                    int token_pair_index = segment_start + m_seg;
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = token_index / (topk_weights ? 1 : topk);
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) =
                        *reinterpret_cast<const VecT*>(&input[(size_t)input_index * size_k + k_global]);
                } else {
                    *reinterpret_cast<VecT*>(&A_sh[m_local * K_BLK + k_local]) = zero_vec;
                }
            }

            __syncthreads();

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

            const T* A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
            const T* B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

            load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
            load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

            mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();
        }

        float* C_sh_ptr = C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
        store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

        __syncthreads();

        const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
        for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
            int m_local_c = i / N_BLK;
            int n_local_c = i % N_BLK;

            int m_seg = m_base + m_local_c;
            int n_global = n_base + n_local_c;

            if (m_seg < num_rows_in_segment && n_global < size_n) {
                int token_pair_index = segment_start + m_seg;
                if (token_pair_index < size_m) {
                    int token_index = sorted_token_ids[token_pair_index];
                    float val = C_sh[m_local_c * N_BLK + n_local_c];
                    if (topk_weights) {
                        val *= topk_weights[token_index];
                    }
                    output[(size_t)token_index * size_n + n_global] = static_cast<T>(val);
                }
            }
        }

        __syncthreads();
    }
}

// C interface functions
extern "C" void launch_blockwise_fp8_moe_gemm_wmma_f16(
    const __half* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __half* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
    int32_t* expert_offsets;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    calculate_expert_offsets_fp8(expert_ids, size_m, expert_offsets, num_experts, stream);

    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    size_t A_sh_bytes = M_BLK * K_BLK * sizeof(__half);
    size_t B_sh_bytes = N_BLK * K_BLK * sizeof(__half);
    size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16;
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes;

    blockwise_fp8_moe_gemm_wmma_kernel<__half><<<grid, block, smem_bytes, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k,
        weight_block_size_y, weight_block_size_x,
        scale_n_blocks, scale_k_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFreeAsync(expert_offsets, stream);
}

extern "C" void launch_blockwise_fp8_moe_gemm_wmma_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __nv_bfloat16* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_n_blocks, int scale_k_blocks,
    cudaStream_t stream
) {
    int32_t* expert_offsets;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    calculate_expert_offsets_fp8(expert_ids, size_m, expert_offsets, num_experts, stream);

    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    size_t A_sh_bytes = M_BLK * K_BLK * sizeof(__nv_bfloat16);
    size_t B_sh_bytes = N_BLK * K_BLK * sizeof(__nv_bfloat16);
    size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16;
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes;

    blockwise_fp8_moe_gemm_wmma_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k,
        weight_block_size_y, weight_block_size_x,
        scale_n_blocks, scale_k_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFreeAsync(expert_offsets, stream);
}

// Transposed variants
extern "C" void launch_blockwise_fp8_moe_gemm_wmma_transposed_f16(
    const __half* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __half* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
    int32_t* expert_offsets;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    calculate_expert_offsets_fp8(expert_ids, size_m, expert_offsets, num_experts, stream);

    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    size_t A_sh_bytes = M_BLK * K_BLK * sizeof(__half);
    size_t B_sh_bytes = N_BLK * K_BLK * sizeof(__half);
    size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16;
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes;

    blockwise_fp8_moe_gemm_wmma_transposed_kernel<__half><<<grid, block, smem_bytes, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k,
        weight_block_size_y, weight_block_size_x,
        scale_k_blocks, scale_n_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFreeAsync(expert_offsets, stream);
}

extern "C" void launch_blockwise_fp8_moe_gemm_wmma_transposed_bf16(
    const __nv_bfloat16* input,
    const __nv_fp8_e4m3* weights,
    const float* weight_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    __nv_bfloat16* output,
    int num_experts, int topk,
    int size_m, int size_n, int size_k,
    int weight_block_size_y, int weight_block_size_x,
    int scale_k_blocks, int scale_n_blocks,
    cudaStream_t stream
) {
    int32_t* expert_offsets;
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
    calculate_expert_offsets_fp8(expert_ids, size_m, expert_offsets, num_experts, stream);

    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    size_t A_sh_bytes = M_BLK * K_BLK * sizeof(__nv_bfloat16);
    size_t B_sh_bytes = N_BLK * K_BLK * sizeof(__nv_bfloat16);
    size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
    size_t AB_bytes = A_sh_bytes + B_sh_bytes;
    size_t pad = (16 - (AB_bytes % 16)) % 16;
    size_t smem_bytes = AB_bytes + pad + C_sh_bytes;

    blockwise_fp8_moe_gemm_wmma_transposed_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
        input, weights, weight_scales,
        sorted_token_ids, expert_offsets, topk_weights, output,
        num_experts, topk, size_m, size_n, size_k,
        weight_block_size_y, weight_block_size_x,
        scale_k_blocks, scale_n_blocks
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFreeAsync(expert_offsets, stream);
}
