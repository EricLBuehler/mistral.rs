#ifndef NO_WMMA_KERNEL
/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  Each block computes a tile of the output corresponding to:
 *    - One expert segment (group of tokens routed to the same expert)
 *    - One N-dimension tile (a sub-block of the expert's output features)
 *
 *  The kernel loads input activations and expert weights in tiles using shared
 * memory, performs matrix multiplication using Tensor Cores (WMMA), and
 * accumulates results into a shared C tile. The final results are written
 * atomically into the global output buffer to support multi-expert (top-k > 1)
 * routing where tokens appear in multiple experts’ outputs.
 *
 *
 * Original Implementation:
 * https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm_wmma.cu
 *
 *  @note
 *   - Uses 4 warps per block (2×2 warp tiling) with block tile = 32×32×16.
 *   - Shared memory tiles are padded and zeroed for tail handling.
 *
 */

#include "moe_utils.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef NO_WMMA_KERNEL
#include <mma.h>
using namespace nvcuda::wmma;
#endif

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
using VecT = float4;

// Vectorized load size (float4 = 128 bits = 8 half/bfloat16 values)
constexpr int VEC_SIZE = 8;
constexpr int NUM_VECS = WMMA_M * WMMA_N / 8;

// We use 4 Warps (128 threads) per block
constexpr int WARPS_M = 2;                          // 2 warps along M
constexpr int WARPS_N = 2;                          // 2 warps along N
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 4 warps
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * 32; // 128 threads

constexpr int M_BLK = WARPS_M * WMMA_M; // 32
constexpr int N_BLK = WARPS_N * WMMA_N; // 32
constexpr int K_BLK = WMMA_K;           // 16

/**
 *  @brief  WMMA-based grouped MoE GEMM kernel.
 *
 *  @tparam T               Data type: half or nv_bfloat16
 *
 *  @param input            [size_m or size_m/topk, size_k]
 *  @param weights          [num_experts, size_n, size_k] compacted expert
 * weights
 *  @param sorted_token_ids [size_m] mapping of per-token row indices (sorted by
 * expert)
 *  @param expert_offsets   [num_experts] array of {start, len} tokens indices
 * for each expert
 *  @param topk_weights     [size_m] optional per-token scaling weights (nullptr
 * if unused)
 *  @param output           [size_m, size_n] global output buffer (must be
 * zero-initialized)
 *  @param num_experts      Total number of experts
 *  @param topk             Number of experts each token is routed to
 *  @param size_m           Number of tokens
 *  @param size_n           Output hidden dimension (per expert)
 *  @param size_k           Input hidden dimension
 */
template <typename T>
__global__ void moe_gemm_grouped_kernel(
    const T *__restrict__ input,   // [size_m, size_k]
    const T *__restrict__ weights, // [num_experts, size_n, size_k]
    const int32_t *__restrict__ sorted_token_ids, // [size_m]
    const int32_t *__restrict__ expert_offsets,   // [num_experts]
    const float *__restrict__ topk_weights,       // [size_m]
    T *__restrict__ output, // [size_m, size_n] (Zero-initialized)
    const int num_experts, const int topk, const int32_t size_m,
    const int32_t size_n, const int32_t size_k) {
  // Get Segment and N-Tile for this Block
  const int expert_id = blockIdx.x;
  const int n_tile_idx = blockIdx.y;
  if (expert_id < 0 || expert_id >= num_experts)
    return;
  const int segment_start = expert_offsets[expert_id];
  const int segment_end = expert_offsets[expert_id + 1];
  const int num_rows_in_segment = segment_end - segment_start;

  if (num_rows_in_segment == 0)
    return;

  const int n_base = n_tile_idx * N_BLK;
  if (n_base >= size_n)
    return;

  const T *expert_w =
      weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

  extern __shared__ uint8_t smem_bytes[];

  // A tile: [M_BLK, K_BLK] (row-major)
  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  // B tile: [N_BLK, K_BLK] (row-major)
  T *B_sh = reinterpret_cast<T *>(A_sh + M_BLK * K_BLK);
  uint8_t *C_ptr = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);

  // align next pointer to float alignment
  size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
  if (offset != 0) {
    C_ptr += (alignof(float) - offset);
  }
  float *C_sh = reinterpret_cast<float *>(
      C_ptr); // shared scratch for final per-block tile writes

  const int threadId = threadIdx.x;
  const int warpId = threadId / 32;
  const int laneId = threadId % 32;
  const int warp_m_idx = warpId / WARPS_N;
  const int warp_n_idx = warpId % WARPS_N;

  const int B_ELEMS_PER_BLOCK = N_BLK * K_BLK;
  const int VEC_ELEMS_B = B_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
  const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
  const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
    // We'll accumulate full-K results in per-warp fragments (initialized here)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    // For every k_block we will load B_sh and A_sh for this m_base subsequently
    for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
      // Load B Tile (Weights) into B_sh
      for (int i = threadId; i < VEC_ELEMS_B; i += BLOCK_THREADS) {
        int idx = i * VEC_SIZE; // element index (0..511)
        int n_local = idx / K_BLK;
        int k_local = idx % K_BLK;

        int n_global = n_base + n_local;
        int k_global = k_base + k_local;

        // this should be always satisfied since k dim aligned to 8
        if (n_global < size_n && k_global < size_k) {
          *reinterpret_cast<VecT *>(&B_sh[n_local * K_BLK + k_local]) =
              *reinterpret_cast<const VecT *>(
                  &expert_w[(size_t)n_global * size_k + k_global]);
        } else {
          *reinterpret_cast<VecT *>(&B_sh[n_local * K_BLK + k_local]) =
              zero_vec;
        }
      }

      // Load A Tile (Inputs) into A_sh for this m_base and this k_base
      for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
        int idx = i * VEC_SIZE; // element index
        int m_local = idx / K_BLK;
        int k_local = idx % K_BLK;

        int m_seg = m_base + m_local; // row index within segment
        int k_global = k_base + k_local;

        if (m_seg < num_rows_in_segment && k_global < size_k) {
          int token_pair_index = segment_start + m_seg;
          int token_index = sorted_token_ids[token_pair_index];
          int input_index = token_index / (topk_weights ? 1 : topk);
          *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) =
              *reinterpret_cast<const VecT *>(
                  &input[(size_t)input_index * size_k + k_global]);
        } else {
          // in case m dim in this segment not aligned to 8
          *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) =
              zero_vec;
        }
      }

      __syncthreads();

      // Compute (Warp-level) : update c_frag for this k_block
      fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
      fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

      // Point this warp to its tile in shared memory
      const T *A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
      const T *B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

      load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
      load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

      // Accumulate into c_frag (which persists across k_base iterations)
      mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads(); // Fix shared memory mismatch on V100
    } // end k_base loop (we have a fully-accumulated c_frag for this m_base
      // tile)

    // Store the accumulated c_frag to C_sh (shared) once per warp
    // Point this warp to its 16x16 tile *within* the 32x32 C_sh
    float *C_sh_ptr =
        C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
    // store the full accumulated 16x16 tile (note ld = N_BLK, result in
    // row-major in C_sh)
    store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

    __syncthreads();

    // Cooperative Store from C_sh to Global
    // 128 threads write [M_BLK, N_BLK] = [32, 32] = 1024 elements
    const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
    for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
      int m_local_c = i / N_BLK; // row in C_sh (0..31)
      int n_local_c = i % N_BLK; // col in C_sh (0..31)

      int m_seg = m_base + m_local_c;    // row index within segment
      int n_global = n_base + n_local_c; // col index in output

      if (m_seg < num_rows_in_segment && n_global < size_n) {
        int token_pair_index = segment_start + m_seg;
        if (token_pair_index < size_m) {
          int token_index = sorted_token_ids[token_pair_index];
          float val = C_sh[m_local_c * N_BLK + n_local_c];
          if (topk_weights) {
            val *= topk_weights[token_index];
          }
          vllm::from_float(output[(size_t)token_index * size_n + n_global],
                           val);
        }
      }
    }
  } // end m_base loop
}

/**
 *  @brief  WMMA-based grouped MoE GEMM kernel for transposed weights [E, K, N].
 *
 *  @tparam T               Data type: half or nv_bfloat16
 *
 *  @param input            [size_m or size_m/topk, size_k]
 *  @param weights          [num_experts, size_k, size_n] transposed expert
 * weights
 *  @param sorted_token_ids [size_m] mapping of per-token row indices (sorted by
 * expert)
 *  @param expert_offsets   [num_experts] array of {start, len} tokens indices
 * for each expert
 *  @param topk_weights     [size_m] optional per-token scaling weights (nullptr
 * if unused)
 *  @param output           [size_m, size_n] global output buffer (must be
 * zero-initialized)
 *  @param num_experts      Total number of experts
 *  @param topk             Number of experts each token is routed to
 *  @param size_m           Number of tokens
 *  @param size_n           Output hidden dimension (per expert)
 *  @param size_k           Input hidden dimension
 */
template <typename T>
__global__ void moe_gemm_grouped_transposed_kernel(
    const T *__restrict__ input,   // [size_m, size_k]
    const T *__restrict__ weights, // [num_experts, size_k, size_n] - transposed
                                   // layout
    const int32_t *__restrict__ sorted_token_ids, // [size_m]
    const int32_t *__restrict__ expert_offsets,   // [num_experts]
    const float *__restrict__ topk_weights,       // [size_m]
    T *__restrict__ output, // [size_m, size_n] (Zero-initialized)
    const int num_experts, const int topk, const int32_t size_m,
    const int32_t size_n, const int32_t size_k) {
  // Get Segment and N-Tile for this Block
  const int expert_id = blockIdx.x;
  const int n_tile_idx = blockIdx.y;
  if (expert_id < 0 || expert_id >= num_experts)
    return;
  const int segment_start = expert_offsets[expert_id];
  const int segment_end = expert_offsets[expert_id + 1];
  const int num_rows_in_segment = segment_end - segment_start;

  if (num_rows_in_segment == 0)
    return;

  const int n_base = n_tile_idx * N_BLK;
  if (n_base >= size_n)
    return;

  // For transposed layout [E, K, N]: base is expert * K * N
  const T *expert_w =
      weights + (size_t)expert_id * (size_t)size_k * (size_t)size_n;

  extern __shared__ uint8_t smem_bytes[];

  // A tile: [M_BLK, K_BLK] (row-major)
  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  // B tile: [N_BLK, K_BLK] (row-major) - we load transposed weights into this
  // layout
  T *B_sh = reinterpret_cast<T *>(A_sh + M_BLK * K_BLK);
  uint8_t *C_ptr = reinterpret_cast<uint8_t *>(B_sh + N_BLK * K_BLK);

  // align next pointer to float alignment
  size_t offset = reinterpret_cast<uintptr_t>(C_ptr) % alignof(float);
  if (offset != 0) {
    C_ptr += (alignof(float) - offset);
  }
  float *C_sh = reinterpret_cast<float *>(
      C_ptr); // shared scratch for final per-block tile writes

  const int threadId = threadIdx.x;
  const int warpId = threadId / 32;
  const int laneId = threadId % 32;
  const int warp_m_idx = warpId / WARPS_N;
  const int warp_n_idx = warpId % WARPS_N;

  const int A_ELEMS_PER_BLOCK = M_BLK * K_BLK;
  const int VEC_ELEMS_A = A_ELEMS_PER_BLOCK / VEC_SIZE; // 512 / 8 = 64
  VecT zero_vec;
  zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

  for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
    // We'll accumulate full-K results in per-warp fragments (initialized here)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    // For every k_block we will load B_sh and A_sh for this m_base subsequently
    for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
      // Load B Tile (Weights) into B_sh - transposed layout
      // For transposed [E, K, N]: weight[k, n] = expert_w[k * size_n + n]
      // We need to load into B_sh[n_local, k_local] for compute
      for (int i = threadId; i < N_BLK * K_BLK; i += BLOCK_THREADS) {
        int n_local = i / K_BLK;
        int k_local = i % K_BLK;

        int n_global = n_base + n_local;
        int k_global = k_base + k_local;

        if (n_global < size_n && k_global < size_k) {
          // Transposed access: weight[k, n] = expert_w[k * size_n + n]
          B_sh[n_local * K_BLK + k_local] =
              expert_w[(size_t)k_global * size_n + n_global];
        } else {
          B_sh[n_local * K_BLK + k_local] = T(0);
        }
      }

      // Load A Tile (Inputs) into A_sh for this m_base and this k_base
      for (int i = threadId; i < VEC_ELEMS_A; i += BLOCK_THREADS) {
        int idx = i * VEC_SIZE; // element index
        int m_local = idx / K_BLK;
        int k_local = idx % K_BLK;

        int m_seg = m_base + m_local; // row index within segment
        int k_global = k_base + k_local;

        if (m_seg < num_rows_in_segment && k_global < size_k) {
          int token_pair_index = segment_start + m_seg;
          int token_index = sorted_token_ids[token_pair_index];
          int input_index = token_index / (topk_weights ? 1 : topk);
          *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) =
              *reinterpret_cast<const VecT *>(
                  &input[(size_t)input_index * size_k + k_global]);
        } else {
          // in case m dim in this segment not aligned to 8
          *reinterpret_cast<VecT *>(&A_sh[m_local * K_BLK + k_local]) =
              zero_vec;
        }
      }

      __syncthreads();

      // Compute (Warp-level) : update c_frag for this k_block
      fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, T, row_major> a_frag;
      fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, T, col_major> b_frag;

      // Point this warp to its tile in shared memory
      const T *A_sh_ptr = A_sh + (warp_m_idx * WMMA_M * K_BLK);
      const T *B_sh_ptr = B_sh + (warp_n_idx * WMMA_N * K_BLK);

      load_matrix_sync(a_frag, A_sh_ptr, K_BLK);
      load_matrix_sync(b_frag, B_sh_ptr, K_BLK);

      // Accumulate into c_frag (which persists across k_base iterations)
      mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads(); // Fix shared memory mismatch on V100
    } // end k_base loop (we have a fully-accumulated c_frag for this m_base
      // tile)

    // Store the accumulated c_frag to C_sh (shared) once per warp
    // Point this warp to its 16x16 tile *within* the 32x32 C_sh
    float *C_sh_ptr =
        C_sh + (warp_m_idx * WMMA_M * N_BLK) + (warp_n_idx * WMMA_N);
    // store the full accumulated 16x16 tile (note ld = N_BLK, result in
    // row-major in C_sh)
    store_matrix_sync(C_sh_ptr, c_frag, N_BLK, mem_row_major);

    __syncthreads();

    // Cooperative Store from C_sh to Global
    // 128 threads write [M_BLK, N_BLK] = [32, 32] = 1024 elements
    const int C_ELEMS_PER_BLOCK = M_BLK * N_BLK;
    for (int i = threadId; i < C_ELEMS_PER_BLOCK; i += BLOCK_THREADS) {
      int m_local_c = i / N_BLK; // row in C_sh (0..31)
      int n_local_c = i % N_BLK; // col in C_sh (0..31)

      int m_seg = m_base + m_local_c;    // row index within segment
      int n_global = n_base + n_local_c; // col index in output

      if (m_seg < num_rows_in_segment && n_global < size_n) {
        int token_pair_index = segment_start + m_seg;
        if (token_pair_index < size_m) {
          int token_index = sorted_token_ids[token_pair_index];
          float val = C_sh[m_local_c * N_BLK + n_local_c];
          if (topk_weights) {
            val *= topk_weights[token_index];
          }
          vllm::from_float(output[(size_t)token_index * size_n + n_global],
                           val);
        }
      }
    }
  } // end m_base loop
}

extern "C" void
moe_gemm_wmma(const void *input,               // [size_m, size_k]
              const void *weights,             // [num_experts, size_n, size_k]
              const int32_t *sorted_token_ids, // [size_m] (Device)
              const int32_t *expert_ids,       // [size_m * topk]
              const float *topk_weights, // [size_m] (Device, can be nullptr)
              void *output,              // [size_m, size_n]
              int num_experts, int topk, int size_m, int size_n, int size_k,
              int data_type, // 0 = half, 1 = bfloat16
              cudaStream_t stream) {
  int32_t *expert_offsets;
  cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
  calculate_expert_offsets(expert_ids, size_m, expert_offsets, num_experts,
                           stream);

  int grid_n = CEILDIV(size_n, N_BLK);
  dim3 grid(num_experts, grid_n, 1);
  dim3 block(BLOCK_THREADS, 1, 1);

  // Shared memory: A_sh[M_BLK, K_BLK] + B_sh[N_BLK, K_BLK]
  size_t A_sh_bytes = M_BLK * K_BLK * 2; // (32*16 * 2) = 1024
  size_t B_sh_bytes = N_BLK * K_BLK * 2; // (32*16 * 2) = 1024
  size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
  size_t AB_bytes = A_sh_bytes + B_sh_bytes;
  size_t pad = (16 - (AB_bytes % 16)) % 16;
  size_t smem_bytes = AB_bytes + pad + C_sh_bytes; // ~6KB total needed

  if (data_type == 0) { // half
    moe_gemm_grouped_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<half *>(output),
        num_experts, topk, size_m, size_n, size_k);
  } else if (data_type == 1) { // bfloat16
#ifndef NO_BF16_KERNEL
    moe_gemm_grouped_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
        num_experts, topk, size_m, size_n, size_k);
#endif
  }

  cudaFreeAsync(expert_offsets, stream);
}

// Transposed weight variant: weights are [num_experts, size_k, size_n] instead
// of [num_experts, size_n, size_k]
extern "C" void moe_gemm_wmma_transposed(
    const void *input,   // [size_m, size_k]
    const void *weights, // [num_experts, size_k, size_n] - transposed layout
    const int32_t *sorted_token_ids, // [size_m] (Device)
    const int32_t *expert_ids,       // [size_m * topk]
    const float *topk_weights,       // [size_m] (Device, can be nullptr)
    void *output,                    // [size_m, size_n]
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int data_type, // 0 = half, 1 = bfloat16
    cudaStream_t stream) {
  int32_t *expert_offsets;
  cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int32_t), stream);
  calculate_expert_offsets(expert_ids, size_m, expert_offsets, num_experts,
                           stream);

  int grid_n = CEILDIV(size_n, N_BLK);
  dim3 grid(num_experts, grid_n, 1);
  dim3 block(BLOCK_THREADS, 1, 1);

  // Shared memory: A_sh[M_BLK, K_BLK] + B_sh[N_BLK, K_BLK]
  size_t A_sh_bytes = M_BLK * K_BLK * 2; // (32*16 * 2) = 1024
  size_t B_sh_bytes = N_BLK * K_BLK * 2; // (32*16 * 2) = 1024
  size_t C_sh_bytes = M_BLK * N_BLK * sizeof(float);
  size_t AB_bytes = A_sh_bytes + B_sh_bytes;
  size_t pad = (16 - (AB_bytes % 16)) % 16;
  size_t smem_bytes = AB_bytes + pad + C_sh_bytes; // ~6KB total needed

  if (data_type == 0) { // half
    moe_gemm_grouped_transposed_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<half *>(output),
        num_experts, topk, size_m, size_n, size_k);
  } else if (data_type == 1) { // bfloat16
#ifndef NO_BF16_KERNEL
    moe_gemm_grouped_transposed_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const nv_bfloat16 *>(input),
        reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<nv_bfloat16 *>(output),
        num_experts, topk, size_m, size_n, size_k);
#endif
  }

  cudaFreeAsync(expert_offsets, stream);
}
#else

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Stub implementation when WMMA is not available (SM < 70)
// These are needed for linking but should not be called at runtime on old GPUs
extern "C" void
moe_gemm_wmma(const void *input,
              const void *weights,
              const int32_t *sorted_token_ids,
              const int32_t *expert_ids,
              const float *topk_weights,
              void *output,
              int num_experts, int topk, int size_m, int size_n, int size_k,
              int data_type,
              cudaStream_t stream) {
  // WMMA requires SM 7.0+ (Volta), this stub should never be called at runtime
  fprintf(stderr, "ERROR: moe_gemm_wmma called but WMMA is not available (requires SM >= 70)\n");
}

extern "C" void moe_gemm_wmma_transposed(
    const void *input,
    const void *weights,
    const int32_t *sorted_token_ids,
    const int32_t *expert_ids,
    const float *topk_weights,
    void *output,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int data_type,
    cudaStream_t stream) {
  // WMMA requires SM 7.0+ (Volta), this stub should never be called at runtime
  fprintf(stderr, "ERROR: moe_gemm_wmma_transposed called but WMMA is not available (requires SM >= 70)\n");
}

#endif