// Adapted from: 
// moe_gemm_grouped_kernel_wna16_prefill: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemm.cu
// moe_gemv_kernel_wna16: https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/moe_gemv.cu

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda::wmma;

namespace {

constexpr int WARP_SIZE = 32;
constexpr int M_TILE = 32;
constexpr int N_TILE = 32;
constexpr int K_TILE = 16;
constexpr int THREADS = 128;

template <typename T>
__device__ inline T from_float(float value);

template <>
__device__ inline half from_float<half>(float value) {
  return __float2half(value);
}

#ifndef NO_BF16_KERNEL
template <>
__device__ inline nv_bfloat16 from_float<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}
#endif

template <typename T>
__device__ inline float to_float(T value);

template <>
__device__ inline float to_float<half>(half value) {
  return __half2float(value);
}

#ifndef NO_BF16_KERNEL
template <>
__device__ inline float to_float<nv_bfloat16>(nv_bfloat16 value) {
  return __bfloat162float(value);
}
#endif

__global__ void count_and_scan(const int32_t* ids, int32_t* counts,
                               int32_t* offsets, int num_experts, int size_m) {
  extern __shared__ int32_t counts_s[];
  const int tid = threadIdx.x;
  for (int expert = tid; expert < num_experts; expert += blockDim.x) {
    counts_s[expert] = 0;
  }
  __syncthreads();
  for (int index = tid; index < size_m; index += blockDim.x) {
    const int expert = ids[index];
    if (expert >= 0 && expert < num_experts) atomicAdd(&counts_s[expert], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int32_t offset = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      counts[expert] = counts_s[expert];
      offsets[expert] = offset;
      offset += counts_s[expert];
    }
    offsets[num_experts] = offset;
  }
}

template <typename T, int BITS, int ROWS_PER_BLOCK>
__global__ void moe_gemv_kernel(
    const T* __restrict__ input, const uint32_t* __restrict__ weights,
    const float* __restrict__ scales, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids, const float* __restrict__ topk_weights,
    T* __restrict__ output, int num_experts, int topk, int size_m, int size_n,
    int size_k, int group_size, int zero_point) {
  const int assignment = blockIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  if (assignment >= size_m || row >= size_n) return;

  const int token_id = sorted_ids[assignment];
  const int expert = expert_ids[assignment];
  if (expert < 0 || expert >= num_experts) return;
  const int input_id = topk_weights ? token_id : token_id / topk;

  extern __shared__ unsigned char raw[];
  T* input_s = reinterpret_cast<T*>(raw);
  const T* input_row = input + static_cast<size_t>(input_id) * size_k;
  for (int k = threadIdx.x; k < size_k; k += blockDim.x) input_s[k] = input_row[k];
  __syncthreads();

  constexpr int PACK_FACTOR = 32 / BITS;
  const int packed_k = (size_k + PACK_FACTOR - 1) / PACK_FACTOR;
  const int scale_k = (size_k + group_size - 1) / group_size;
  const uint32_t mask = (1u << BITS) - 1u;
  const uint32_t* weight_row = weights +
      (static_cast<size_t>(expert) * size_n + row) * packed_k;
  const float* scale_row = scales +
      (static_cast<size_t>(expert) * size_n + row) * scale_k;

  float sum = 0.0f;
  for (int packed = lane; packed < packed_k; packed += WARP_SIZE) {
    const uint32_t word = weight_row[packed];
#pragma unroll
    for (int q_index = 0; q_index < PACK_FACTOR; ++q_index) {
      const int k = packed * PACK_FACTOR + q_index;
      if (k < size_k) {
        const int q = static_cast<int>((word >> (q_index * BITS)) & mask) - zero_point;
        sum += to_float(input_s[k]) * (static_cast<float>(q) * scale_row[k / group_size]);
      }
    }
  }
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  if (lane == 0) {
    if (topk_weights) sum *= topk_weights[token_id];
    output[static_cast<size_t>(token_id) * size_n + row] = from_float<T>(sum);
  }
}

template <typename T, int BITS>
__global__ void moe_gemm_kernel(
    const T* __restrict__ input, const uint32_t* __restrict__ weights,
    const float* __restrict__ scales, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ offsets, const float* __restrict__ topk_weights,
    T* __restrict__ output, int num_experts, int topk, int size_m, int size_n,
    int size_k, int group_size, int zero_point) {
  const int expert = blockIdx.x;
  const int n_base = blockIdx.y * N_TILE;
  if (expert >= num_experts || n_base >= size_n) return;
  const int start = offsets[expert];
  const int end = offsets[expert + 1];
  if (start >= end) return;

  constexpr int PACK_FACTOR = 32 / BITS;
  const int packed_k = (size_k + PACK_FACTOR - 1) / PACK_FACTOR;
  const int scale_k = (size_k + group_size - 1) / group_size;
  const uint32_t mask = (1u << BITS) - 1u;
  const uint32_t* expert_w = weights + static_cast<size_t>(expert) * size_n * packed_k;
  const float* expert_s = scales + static_cast<size_t>(expert) * size_n * scale_k;

  extern __shared__ unsigned char raw[];
  T* a_s = reinterpret_cast<T*>(raw);
  T* b_s = a_s + M_TILE * K_TILE;
  float* c_s = reinterpret_cast<float*>(b_s + N_TILE * K_TILE);
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int warp_m = warp / 2;
  const int warp_n = warp % 2;

  for (int m_base = 0; m_base < end - start; m_base += M_TILE) {
    fragment<accumulator, 16, 16, K_TILE, float> acc;
    fill_fragment(acc, 0.0f);
    for (int k_base = 0; k_base < size_k; k_base += K_TILE) {
      for (int index = tid; index < N_TILE * K_TILE; index += THREADS) {
        const int n = index / K_TILE;
        const int k = index % K_TILE;
        const int ng = n_base + n;
        const int kg = k_base + k;
        if (ng < size_n && kg < size_k) {
          const uint32_t word = expert_w[static_cast<size_t>(ng) * packed_k + kg / PACK_FACTOR];
          const int q = static_cast<int>((word >> ((kg % PACK_FACTOR) * BITS)) & mask) - zero_point;
          b_s[n * K_TILE + k] = from_float<T>(static_cast<float>(q) * expert_s[static_cast<size_t>(ng) * scale_k + kg / group_size]);
        } else {
          b_s[n * K_TILE + k] = from_float<T>(0.0f);
        }
      }
      for (int index = tid; index < M_TILE * K_TILE; index += THREADS) {
        const int m = index / K_TILE;
        const int k = index % K_TILE;
        const int route = start + m_base + m;
        const int kg = k_base + k;
        if (route < end && kg < size_k) {
          const int token_id = sorted_ids[route];
          const int input_id = topk_weights ? token_id : token_id / topk;
          a_s[m * K_TILE + k] = input[static_cast<size_t>(input_id) * size_k + kg];
        } else {
          a_s[m * K_TILE + k] = from_float<T>(0.0f);
        }
      }
      __syncthreads();
      fragment<matrix_a, 16, 16, K_TILE, T, row_major> a;
      fragment<matrix_b, 16, 16, K_TILE, T, col_major> b;
      load_matrix_sync(a, a_s + warp_m * 16 * K_TILE, K_TILE);
      load_matrix_sync(b, b_s + warp_n * 16 * K_TILE, K_TILE);
      mma_sync(acc, a, b, acc);
      __syncthreads();
    }
    store_matrix_sync(c_s + warp_m * 16 * N_TILE + warp_n * 16, acc, N_TILE, mem_row_major);
    __syncthreads();
    for (int index = tid; index < M_TILE * N_TILE; index += THREADS) {
      const int m = index / N_TILE;
      const int n = index % N_TILE;
      const int route = start + m_base + m;
      const int ng = n_base + n;
      if (route < end && route < size_m && ng < size_n) {
        const int token_id = sorted_ids[route];
        float value = c_s[m * N_TILE + n];
        if (topk_weights) value *= topk_weights[token_id];
        output[static_cast<size_t>(token_id) * size_n + ng] = from_float<T>(value);
      }
    }
    __syncthreads();
  }
}

template <typename T, int BITS, int ROWS_PER_BLOCK>
void launch_gemv_rows(const void* input, const uint32_t* weights, const float* scales,
                 const int32_t* sorted_ids, const int32_t* expert_ids,
                 const float* topk_weights, void* output, int num_experts,
                 int topk, int size_m, int size_n, int size_k, int group_size,
                 int zero_point, cudaStream_t stream) {
  constexpr int rows = ROWS_PER_BLOCK;
  dim3 grid((size_n + rows - 1) / rows, size_m);
  dim3 block(rows * WARP_SIZE);
  moe_gemv_kernel<T, BITS, ROWS_PER_BLOCK><<<grid, block, static_cast<size_t>(size_k) * sizeof(T), stream>>>(
      static_cast<const T*>(input), weights, scales, sorted_ids, expert_ids,
      topk_weights, static_cast<T*>(output), num_experts, topk, size_m, size_n,
      size_k, group_size, zero_point);
}

template <typename T, int BITS>
void launch_gemv(const void* input, const uint32_t* weights, const float* scales,
                 const int32_t* sorted_ids, const int32_t* expert_ids,
                 const float* topk_weights, void* output, int num_experts,
                 int topk, int size_m, int size_n, int size_k, int group_size,
                 int zero_point, cudaStream_t stream) {
  const int rows = size_n <= 512 ? 16 : size_n <= 2048 ? 8 : size_n <= 4096 ? 4 : 2;
  switch (rows) {
    case 16:
      launch_gemv_rows<T, BITS, 16>(input, weights, scales, sorted_ids, expert_ids,
                                     topk_weights, output, num_experts, topk, size_m, size_n,
                                     size_k, group_size, zero_point, stream);
      break;
    case 8:
      launch_gemv_rows<T, BITS, 8>(input, weights, scales, sorted_ids, expert_ids,
                                    topk_weights, output, num_experts, topk, size_m, size_n,
                                    size_k, group_size, zero_point, stream);
      break;
    case 4:
      launch_gemv_rows<T, BITS, 4>(input, weights, scales, sorted_ids, expert_ids,
                                    topk_weights, output, num_experts, topk, size_m, size_n,
                                    size_k, group_size, zero_point, stream);
      break;
    default:
      launch_gemv_rows<T, BITS, 2>(input, weights, scales, sorted_ids, expert_ids,
                                    topk_weights, output, num_experts, topk, size_m, size_n,
                                    size_k, group_size, zero_point, stream);
      break;
  }
}

template <typename T, int BITS>
void launch_gemm(const void* input, const uint32_t* weights, const float* scales,
                 const int32_t* sorted_ids, const int32_t* expert_ids,
                 const float* topk_weights, void* output, int32_t* counts,
                 int32_t* offsets, int num_experts, int topk, int size_m,
                 int size_n, int size_k, int group_size, int zero_point,
                 bool prefill, cudaStream_t stream) {
  count_and_scan<<<1, 1024, static_cast<size_t>(num_experts) * sizeof(int32_t), stream>>>(
      expert_ids, counts, offsets, num_experts, size_m);
  if (!prefill) {
    launch_gemv<T, BITS>(input, weights, scales, sorted_ids, expert_ids, topk_weights,
                         output, num_experts, topk, size_m, size_n, size_k,
                         group_size, zero_point, stream);
    return;
  }
  dim3 grid(num_experts, (size_n + N_TILE - 1) / N_TILE);
  dim3 block(THREADS);
  const size_t smem = (M_TILE * K_TILE + N_TILE * K_TILE) * sizeof(T) +
                      M_TILE * N_TILE * sizeof(float);
  moe_gemm_kernel<T, BITS><<<grid, block, smem, stream>>>(
      static_cast<const T*>(input), weights, scales, sorted_ids, offsets,
      topk_weights, static_cast<T*>(output), num_experts, topk, size_m, size_n,
      size_k, group_size, zero_point);
}

}  // namespace

extern "C" void moe_gemv_wna16(
    const void* input, const uint32_t* weights, const void* weight_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* topk_weights, void* output, int num_experts, int topk,
    int size_m, int size_n, int size_k, int bits, int group_size, int zero_point,
    int data_type, int64_t stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* scales = static_cast<const float*>(weight_scales);
  if (data_type == 0) {
    if (bits == 4) launch_gemv<half, 4>(input, weights, scales, sorted_token_ids, expert_ids,
                                         topk_weights, output, num_experts, topk, size_m, size_n,
                                         size_k, group_size, zero_point, cuda_stream);
    else if (bits == 8) launch_gemv<half, 8>(input, weights, scales, sorted_token_ids, expert_ids,
                                              topk_weights, output, num_experts, topk, size_m, size_n,
                                              size_k, group_size, zero_point, cuda_stream);
  }
#ifndef NO_BF16_KERNEL
  else if (data_type == 1) {
    if (bits == 4) launch_gemv<nv_bfloat16, 4>(input, weights, scales, sorted_token_ids, expert_ids,
                                               topk_weights, output, num_experts, topk, size_m, size_n,
                                               size_k, group_size, zero_point, cuda_stream);
    else if (bits == 8) launch_gemv<nv_bfloat16, 8>(input, weights, scales, sorted_token_ids, expert_ids,
                                                   topk_weights, output, num_experts, topk, size_m, size_n,
                                                   size_k, group_size, zero_point, cuda_stream);
  }
#endif
}

extern "C" void moe_gemm_wmma_wna16(
    const void* input, const uint32_t* weights, const void* weight_scales,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* topk_weights, void* output, int32_t* expert_counts,
    int32_t* expert_offsets, int num_experts, int topk, int size_m, int size_n,
    int size_k, int bits, int group_size, int zero_point, int data_type,
    bool is_prefill, int64_t stream) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto* scales = static_cast<const float*>(weight_scales);
  if (data_type == 0) {
    if (bits == 4) launch_gemm<half, 4>(input, weights, scales, sorted_token_ids, expert_ids,
                                         topk_weights, output, expert_counts, expert_offsets,
                                         num_experts, topk, size_m, size_n, size_k, group_size,
                                         zero_point, is_prefill, cuda_stream);
    else if (bits == 8) launch_gemm<half, 8>(input, weights, scales, sorted_token_ids, expert_ids,
                                              topk_weights, output, expert_counts, expert_offsets,
                                              num_experts, topk, size_m, size_n, size_k, group_size,
                                              zero_point, is_prefill, cuda_stream);
  }
#ifndef NO_BF16_KERNEL
  else if (data_type == 1) {
    if (bits == 4) launch_gemm<nv_bfloat16, 4>(input, weights, scales, sorted_token_ids, expert_ids,
                                               topk_weights, output, expert_counts, expert_offsets,
                                               num_experts, topk, size_m, size_n, size_k, group_size,
                                               zero_point, is_prefill, cuda_stream);
    else if (bits == 8) launch_gemm<nv_bfloat16, 8>(input, weights, scales, sorted_token_ids, expert_ids,
                                                   topk_weights, output, expert_counts, expert_offsets,
                                                   num_experts, topk, size_m, size_n, size_k, group_size,
                                                   zero_point, is_prefill, cuda_stream);
  }
#endif
}
