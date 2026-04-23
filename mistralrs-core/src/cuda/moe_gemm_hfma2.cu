/**
 * @brief CUDA kernel for Mixture-of-Experts (MoE) GEMM using half2/bf162
 * vectorized arithmetic as a fallback for WMMA.
 *
 * Adapted from candle-kernels for mistral.rs
 */

#include "moe_utils.h"
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace vllm_rs {

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int K_BLK = 16;
constexpr int BLOCK_THREADS = 128;
constexpr int THREADS_PER_GROUP = 16;

template <typename T2>
static __device__ __forceinline__ float hfma2_dot2(T2 a, T2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same_v<T2, nv_bfloat162>) {
    nv_bfloat162 res = __hmul2(a, b);
    return __bfloat162float(res.x) + __bfloat162float(res.y);
  }
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  if constexpr (std::is_same_v<T2, half2>) {
    half2 res = __hmul2(a, b);
    return __half2float(res.x) + __half2float(res.y);
  }
#endif

  if constexpr (std::is_same_v<T2, nv_bfloat162>) {
    return __bfloat162float(a.x) * __bfloat162float(b.x) +
           __bfloat162float(a.y) * __bfloat162float(b.y);
  } else {
    return vllm::to_float(a.x) * vllm::to_float(b.x) +
           vllm::to_float(a.y) * vllm::to_float(b.y);
  }
}

template <typename T, typename T2>
__global__ void moe_gemm_hfma2_kernel(
    const T *__restrict__ input,                  // [num_tokens, size_k]
    const T *__restrict__ weights,                // [num_experts, size_n, size_k]
    const int32_t *__restrict__ sorted_token_ids, // [size_m]
    const int32_t *__restrict__ expert_offsets,   // [num_experts + 1]
    const float *__restrict__ topk_weights,       // [size_m]
    T *__restrict__ output,                       // [size_m, size_n]
    const int num_experts, const int topk, const int32_t size_m,
    const int32_t size_n, const int32_t size_k) {
  const int expert_id = blockIdx.x;
  const int n_tile_idx = blockIdx.y;
  if (expert_id >= num_experts)
    return;

  const int segment_start = expert_offsets[expert_id];
  const int segment_end = expert_offsets[expert_id + 1];
  const int num_rows_in_segment = segment_end - segment_start;
  if (num_rows_in_segment <= 0)
    return;

  const int n_base = n_tile_idx * N_BLK;
  if (n_base >= size_n)
    return;

  const T *expert_w =
      weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

  constexpr int K_PAD = K_BLK + 2;
  extern __shared__ uint8_t smem_bytes[];
  T *A_sh = reinterpret_cast<T *>(smem_bytes);
  T *B_sh = reinterpret_cast<T *>(A_sh + M_BLK * K_PAD);

  const int tid = threadIdx.x;

  for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
      constexpr int A_ELEMS = M_BLK * K_BLK;
      for (int i = tid; i < A_ELEMS / 2; i += BLOCK_THREADS) {
        int idx = i * 2;
        int m_local = idx / K_BLK;
        int k_local = idx % K_BLK;
        int m_seg = m_base + m_local;
        int k_global = k_base + k_local;

        T *dest = &A_sh[m_local * K_PAD + k_local];
        T2 val;

        if (m_seg < num_rows_in_segment && (k_global + 1) < size_k) {
          int token_pair_index = segment_start + m_seg;
          int token_index = sorted_token_ids[token_pair_index];
          int input_index = topk_weights ? token_index : (token_index / topk);
          val = *reinterpret_cast<const T2 *>(
              &input[(size_t)input_index * size_k + k_global]);
        } else if (m_seg < num_rows_in_segment && k_global < size_k) {
          reinterpret_cast<T *>(&val)[0] =
              input[(size_t)(topk_weights ? sorted_token_ids[segment_start + m_seg]
                                          : (sorted_token_ids[segment_start + m_seg] / topk)) *
                        size_k +
                    k_global];
          reinterpret_cast<T *>(&val)[1] = T(0.0f);
        } else {
          val = T2{T(0.0f), T(0.0f)};
        }
        *reinterpret_cast<T2 *>(dest) = val;
      }

      constexpr int B_ELEMS = N_BLK * K_BLK;
      for (int i = tid; i < B_ELEMS / 2; i += BLOCK_THREADS) {
        int idx = i * 2;
        int n_local = idx / K_BLK;
        int k_local = idx % K_BLK;
        int n_global = n_base + n_local;
        int k_global = k_base + k_local;

        T *dest = &B_sh[n_local * K_PAD + k_local];
        T2 val;

        if (n_global < size_n && (k_global + 1) < size_k) {
          val = *reinterpret_cast<const T2 *>(
              &expert_w[(size_t)n_global * size_k + k_global]);
        } else if (n_global < size_n && k_global < size_k) {
          reinterpret_cast<T *>(&val)[0] =
              expert_w[(size_t)n_global * size_k + k_global];
          reinterpret_cast<T *>(&val)[1] = T(0.0f);
        } else {
          val = T2{T(0.0f), T(0.0f)};
        }
        *reinterpret_cast<T2 *>(dest) = val;
      }
      __syncthreads();

      const int group_id = tid / THREADS_PER_GROUP;
      const int lane_id = tid % THREADS_PER_GROUP;
      const int m_start = group_id * 4;
      const int n_start = lane_id * 2;

      const T2 *A2_sh = reinterpret_cast<const T2 *>(A_sh);
      const T2 *B2_sh = reinterpret_cast<const T2 *>(B_sh);

      for (int k2 = 0; k2 < K_BLK / 2; ++k2) {
        T2 a2[4];
        for (int r = 0; r < 4; ++r) {
          a2[r] = A2_sh[(m_start + r) * (K_PAD / 2) + k2];
        }

        T2 b2[2];
        for (int c = 0; c < 2; ++c) {
          b2[c] = B2_sh[(n_start + c) * (K_PAD / 2) + k2];
        }

        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 2; ++c) {
            acc[r * 2 + c] += hfma2_dot2(a2[r], b2[c]);
          }
        }
      }
      __syncthreads();
    }

    const int group_id = tid / THREADS_PER_GROUP;
    const int lane_id = tid % THREADS_PER_GROUP;
    const int m_start = group_id * 4;
    const int n_start = lane_id * 2;

    for (int r = 0; r < 4; ++r) {
      int m_seg = m_base + m_start + r;
      if (m_seg >= num_rows_in_segment)
        continue;

      int token_pair_index = segment_start + m_seg;
      int token_index = sorted_token_ids[token_pair_index];

      for (int c = 0; c < 2; ++c) {
        int n_global = n_base + n_start + c;
        if (n_global >= size_n)
          continue;

        float val = acc[r * 2 + c];
        if (topk_weights) {
          val *= topk_weights[token_index];
        }
        vllm::from_float(output[(size_t)token_index * size_n + n_global], val);
      }
    }
  }
}

} // namespace vllm_rs

extern "C" void mistralrs_moe_gemm_hfma2(
    const void *input, const void *weights, const int32_t *sorted_token_ids,
    const int32_t *expert_ids, const float *topk_weights, void *output,
    int32_t *expert_offsets, int num_experts, int topk, int size_m, int size_n,
    int size_k, int data_type, cudaStream_t stream) {
  using namespace vllm_rs;

  calculate_expert_offsets(expert_ids, size_m, expert_offsets, num_experts,
                           stream);

  int grid_n = CEILDIV(size_n, N_BLK);
  dim3 grid(num_experts, grid_n, 1);
  dim3 block(BLOCK_THREADS, 1, 1);

  constexpr int K_PAD = K_BLK + 2;
  size_t smem_bytes =
      (M_BLK * K_PAD + N_BLK * K_PAD) * 2; // sizeof(half) or sizeof(bfloat16)

  if (data_type == 0) {
    moe_gemm_hfma2_kernel<half, half2><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half *>(input),
        reinterpret_cast<const half *>(weights), sorted_token_ids,
        expert_offsets, topk_weights, reinterpret_cast<half *>(output),
        num_experts, topk, size_m, size_n, size_k);
  }
#ifndef NO_BF16_KERNEL
  else if (data_type == 1) {
    moe_gemm_hfma2_kernel<nv_bfloat16, nv_bfloat162>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const nv_bfloat16 *>(input),
            reinterpret_cast<const nv_bfloat16 *>(weights), sorted_token_ids,
            expert_offsets, topk_weights,
            reinterpret_cast<nv_bfloat16 *>(output), num_experts, topk, size_m,
            size_n, size_k);
  }
#endif
}
