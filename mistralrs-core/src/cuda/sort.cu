#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <limits>
#include <stdint.h>

__global__ void softcap_f32_kernel(const float *__restrict__ x,
                                   float *__restrict__ dst, const int n,
                                   const float cap) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = tanhf(x[idx] / cap) * cap;
}

extern "C" void softcap_f32(const void *x, void *dst, const int n,
                            const float cap, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  softcap_f32_kernel<<<grid, block, 0, custream>>>(
      reinterpret_cast<const float *>(x), reinterpret_cast<float *>(dst), n,
      cap);
}

__global__ void copy_f32_kernel(const float *__restrict__ x,
                                float *__restrict__ dst, const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = x[idx];
}

__global__ void apply_sparse_penalties_f32_kernel(
    float *__restrict__ logits, const uint32_t *__restrict__ token_ids,
    const float *__restrict__ counts, const int n, const int n_tokens,
    const float frequency_penalty, const float presence_penalty,
    const float repetition_penalty) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_tokens) {
    return;
  }

  const uint32_t token_id = token_ids[idx];
  if (token_id >= static_cast<uint32_t>(n)) {
    return;
  }

  const float count = counts[idx];
  if (count <= 0.0f) {
    return;
  }

  float value = logits[token_id];
  value -= count * frequency_penalty + presence_penalty;

  if (repetition_penalty != 1.0f) {
    value = value > 0.0f ? value / repetition_penalty
                         : value * repetition_penalty;
  }

  logits[token_id] = value;
}

extern "C" void apply_sparse_penalties_f32(
    const void *x, void *dst, const uint32_t *token_ids, const float *counts,
    const int n, const int n_tokens, const float frequency_penalty,
    const float presence_penalty, const float repetition_penalty,
    int64_t stream) {
  if (n <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = 256;
  const int copy_grid = (n + block - 1) / block;
  copy_f32_kernel<<<copy_grid, block, 0, custream>>>(
      reinterpret_cast<const float *>(x), reinterpret_cast<float *>(dst), n);

  if (n_tokens <= 0) {
    return;
  }

  const int penalty_grid = (n_tokens + block - 1) / block;
  apply_sparse_penalties_f32_kernel<<<penalty_grid, block, 0, custream>>>(
      reinterpret_cast<float *>(dst), token_ids, counts, n, n_tokens,
      frequency_penalty, presence_penalty, repetition_penalty);
}

template <typename T>
__device__ __forceinline__ float rms_residual_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float
rms_residual_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float
rms_residual_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T rms_residual_from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half
rms_residual_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
rms_residual_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__global__ void rms_norm_residual_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, const T *__restrict__ scale,
    T *__restrict__ dst, const int ncols, const float eps) {
  __shared__ float reduce[1024];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * ncols;
  const float scale_value = scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float value = rms_residual_to_float(x[row_offset + col]);
    sum += value * value;
  }
  reduce[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(reduce[0] / static_cast<float>(ncols) + eps);
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float normed = rms_residual_to_float(x[row_offset + col]) * inv_rms *
                         rms_residual_to_float(weight[col]);
    const float value =
        (rms_residual_to_float(residual[row_offset + col]) + normed) *
        scale_value;
    dst[row_offset + col] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_residual(const void *x, const void *residual,
                              const void *weight, const void *scale, void *dst,
                              const int nrows, const int ncols,
                              const float eps, int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = ncols < 1024 ? 32 : 1024;
  rms_norm_residual_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(scale),
      reinterpret_cast<T *>(dst), ncols, eps);
}

extern "C" void rms_norm_residual_f32(const void *x, const void *residual,
                                      const void *weight, const void *scale,
                                      void *dst, const int nrows,
                                      const int ncols, const float eps,
                                      int64_t stream) {
  launch_rms_norm_residual<float>(x, residual, weight, scale, dst, nrows, ncols,
                                  eps, stream);
}

extern "C" void rms_norm_residual_f16(const void *x, const void *residual,
                                      const void *weight, const void *scale,
                                      void *dst, const int nrows,
                                      const int ncols, const float eps,
                                      int64_t stream) {
  launch_rms_norm_residual<__half>(x, residual, weight, scale, dst, nrows,
                                   ncols, eps, stream);
}

extern "C" void rms_norm_residual_bf16(const void *x, const void *residual,
                                       const void *weight, const void *scale,
                                       void *dst, const int nrows,
                                       const int ncols, const float eps,
                                       int64_t stream) {
  launch_rms_norm_residual<__nv_bfloat16>(x, residual, weight, scale, dst,
                                          nrows, ncols, eps, stream);
}

template <typename T> inline __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T, bool ascending>
__global__ void bitonic_sort_kernel(T *arr, uint32_t *dst, int j, int k) {
  unsigned int i, ij;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ij = i ^ j;

  if (ij > i) {
    if constexpr (ascending) {
      if ((i & k) == 0) {
        if (arr[i] > arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      } else {
        if (arr[i] < arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      }
    }

    if constexpr (!ascending) {
      if ((i & k) != 0) {
        if (arr[i] > arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      } else {
        if (arr[i] < arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      }
    }
  }
  __syncthreads();
}

int next_power_of_2(int x) {
  int n = 1;
  while (n < x) {
    n *= 2;
  }
  return n;
}

#define ASORT_OP(T, RUST_NAME, ASC)                                            \
  extern "C" void RUST_NAME(void *x1, void *dst1, const int nrows,             \
                            const int ncols, bool inplace, int64_t stream) {   \
    T *x = reinterpret_cast<T *>(x1);                                          \
    uint32_t *dst = reinterpret_cast<uint32_t *>(dst1);                        \
    const cudaStream_t custream = (cudaStream_t)stream;                        \
    int ncols_pad = next_power_of_2(ncols);                                    \
    T *x_row_padded;                                                           \
    uint32_t *dst_row_padded;                                                  \
    cudaMallocAsync((void **)&x_row_padded, ncols_pad * sizeof(T), custream);  \
    cudaMallocAsync((void **)&dst_row_padded, ncols_pad * sizeof(uint32_t),    \
                    custream);                                                 \
    uint32_t *indices_padded =                                                 \
        (uint32_t *)malloc(ncols_pad * sizeof(uint32_t));                      \
    for (int i = 0; i < ncols_pad; i++) {                                      \
      indices_padded[i] = i;                                                   \
    }                                                                          \
    T *values_padded = (T *)malloc((ncols_pad - ncols) * sizeof(T));           \
    for (int i = 0; i < ncols_pad - ncols; i++) {                              \
      values_padded[i] =                                                       \
          ASC ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min(); \
    }                                                                          \
    int max_threads_per_block = 1024;                                          \
    int threads_per_block =                                                    \
        max_threads_per_block > ncols_pad ? ncols_pad : max_threads_per_block; \
    int blocks_per_row =                                                       \
        (ncols_pad + threads_per_block - 1) / threads_per_block;               \
    for (int row = 0; row < nrows; row++) {                                    \
      T *x_row = x + row * ncols;                                              \
      uint32_t *dst_row = dst + row * ncols;                                   \
      cudaMemcpyAsync(x_row_padded, x_row, ncols * sizeof(T),                  \
                      cudaMemcpyDeviceToDevice, custream);                     \
      if (ncols_pad - ncols > 0)                                               \
        cudaMemcpyAsync(x_row_padded + ncols, values_padded,                   \
                        (ncols_pad - ncols) * sizeof(T),                       \
                        cudaMemcpyHostToDevice, custream);                     \
      cudaMemcpyAsync(dst_row_padded, indices_padded,                          \
                      ncols_pad * sizeof(uint32_t), cudaMemcpyHostToDevice,    \
                      custream);                                               \
      for (int k = 2; k <= ncols_pad; k <<= 1) {                               \
        for (int j = k >> 1; j > 0; j = j >> 1) {                              \
          bitonic_sort_kernel<T, ASC>                                          \
              <<<blocks_per_row, threads_per_block, 0, custream>>>(            \
                  x_row_padded, dst_row_padded, j, k);                         \
        }                                                                      \
      }                                                                        \
      if (inplace)                                                             \
        cudaMemcpyAsync(x_row, x_row_padded, ncols * sizeof(T),                \
                        cudaMemcpyDeviceToDevice, custream);                   \
      cudaMemcpyAsync(dst_row, dst_row_padded, ncols * sizeof(uint32_t),       \
                      cudaMemcpyDeviceToDevice, custream);                     \
    }                                                                          \
    cudaFreeAsync(x_row_padded, custream);                                     \
    cudaFreeAsync(dst_row_padded, custream);                                   \
    free(indices_padded);                                                      \
    free(values_padded);                                                       \
  }

ASORT_OP(__nv_bfloat16, asort_asc_bf16, true)
ASORT_OP(__nv_bfloat16, asort_desc_bf16, false)

ASORT_OP(__half, asort_asc_f16, true)
ASORT_OP(__half, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(int64_t, asort_asc_i64, true)

ASORT_OP(float, asort_desc_f32, false)
ASORT_OP(double, asort_desc_f64, false)
ASORT_OP(uint8_t, asort_desc_u8, false)
ASORT_OP(uint32_t, asort_desc_u32, false)
ASORT_OP(int64_t, asort_desc_i64, false)

// ============================================================================
// Optimized parallel topk kernel for small k (MoE routing)
//
// Much faster than full sort for small k:
// - Processes all rows in parallel (one block per row)
// - Uses simple "find max k times" algorithm: O(n*k) for small k
// - Single kernel launch for all rows
// ============================================================================

template <typename T>
__device__ __forceinline__ T warp_reduce_max_with_idx(T val, int idx,
                                                      int &max_idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other_val = __shfl_down_sync(0xffffffff, val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
  max_idx = idx;
  return val;
}

// One block per row, finds top-k elements
// For n <= 1024 (typical MoE expert count), single block is sufficient
// Writes values and indices to SEPARATE buffers (no post-processing needed)
template <typename T>
__global__ void topk_kernel(const T *__restrict__ input, // [nrows, ncols]
                            T *__restrict__ values_out,  // [nrows, k]
                            uint32_t *__restrict__ indices_out, // [nrows, k]
                            const int nrows, const int ncols, const int k) {
  const int row = blockIdx.x;
  if (row >= nrows)
    return;

  const T *row_in = input + row * ncols;
  T *row_values = values_out + row * k;
  uint32_t *row_indices = indices_out + row * k;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Shared memory for this row's data and mask
  extern __shared__ char smem[];
  T *s_data = (T *)smem;
  bool *s_used = (bool *)(s_data + ncols);

  // Load data into shared memory
  for (int i = tid; i < ncols; i += block_size) {
    s_data[i] = row_in[i];
    s_used[i] = false;
  }
  __syncthreads();

  // Find top-k elements
  for (int ki = 0; ki < k; ki++) {
    // Find max among unused elements
    T local_max = (T)(-INFINITY);
    int local_idx = -1;

    for (int i = tid; i < ncols; i += block_size) {
      float candidate = (float)s_data[i];
      if (!s_used[i] && candidate == candidate &&
          candidate > (float)local_max) {
        local_max = s_data[i];
        local_idx = i;
      }
    }

    // Warp reduction to find max
    int warp_max_idx;
    T warp_max = warp_reduce_max_with_idx(local_max, local_idx, warp_max_idx);

    // Block reduction (if more than 1 warp)
    __shared__ T warp_maxes[32];
    __shared__ int warp_indices[32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_indices[warp_id] = warp_max_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < 32) {
      T val = (tid < num_warps) ? warp_maxes[tid] : (T)(-INFINITY);
      int idx = (tid < num_warps) ? warp_indices[tid] : -1;
      int final_idx;
      T final_max = warp_reduce_max_with_idx(val, idx, final_idx);

      if (tid == 0) {
        if (final_idx < 0) {
          final_idx = 0;
          final_max = (T)0;
        }
        row_values[ki] = final_max;
        row_indices[ki] = (uint32_t)final_idx;
        s_used[final_idx] = true;
      }
    }
    __syncthreads();
  }
}

// Wrapper for f32 - writes to separate values and indices buffers
extern "C" void topk_f32(const float *input,
                         float *values_out,     // [nrows, k]
                         uint32_t *indices_out, // [nrows, k]
                         int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  // One block per row
  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(float) + ncols * sizeof(bool);

  topk_kernel<float><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

// Wrapper for bf16 - writes to separate values and indices buffers
extern "C" void topk_bf16(const __nv_bfloat16 *input,
                          __nv_bfloat16 *values_out, // [nrows, k]
                          uint32_t *indices_out,     // [nrows, k]
                          int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(__nv_bfloat16) + ncols * sizeof(bool);

  topk_kernel<__nv_bfloat16><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

// Wrapper for f16 - writes to separate values and indices buffers
extern "C" void topk_f16(const __half *input,
                         __half *values_out,    // [nrows, k]
                         uint32_t *indices_out, // [nrows, k]
                         int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(__half) + ncols * sizeof(bool);

  topk_kernel<__half><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

// ============================================================================
// FUSED topk + softmax kernel
// Finds top-k elements AND computes softmax weights in ONE kernel
// Eliminates intermediate tensor allocation entirely
// ============================================================================

template <typename T>
__global__ void topk_softmax_kernel(
    const T *__restrict__ input,        // [nrows, ncols] - router logits
    T *__restrict__ weights_out,        // [nrows, k] - softmax weights (NOT raw
                                        // logits)
    uint32_t *__restrict__ indices_out, // [nrows, k]
    const int nrows, const int ncols, const int k) {
  const int row = blockIdx.x;
  if (row >= nrows)
    return;

  const T *row_in = input + row * ncols;
  T *row_weights = weights_out + row * k;
  uint32_t *row_indices = indices_out + row * k;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Shared memory layout: [data][used][topk_vals][topk_idx][softmax_ws]
  extern __shared__ char smem[];
  T *s_data = (T *)smem;
  bool *s_used = (bool *)(s_data + ncols);
  T *s_topk_vals = (T *)(s_used + ncols);
  int *s_topk_idx = (int *)(s_topk_vals + k);
  float *s_softmax_ws =
      (float *)(s_topk_idx + k); // Dynamic workspace for softmax

  // Load data into shared memory
  for (int i = tid; i < ncols; i += block_size) {
    s_data[i] = row_in[i];
    s_used[i] = false;
  }
  __syncthreads();

  // Find top-k elements (same as before)
  for (int ki = 0; ki < k; ki++) {
    T local_max = (T)(-INFINITY);
    int local_idx = -1;

    for (int i = tid; i < ncols; i += block_size) {
      float candidate = (float)s_data[i];
      if (!s_used[i] && candidate == candidate &&
          candidate > (float)local_max) {
        local_max = s_data[i];
        local_idx = i;
      }
    }

    // Warp reduction
    int warp_max_idx;
    T warp_max = warp_reduce_max_with_idx(local_max, local_idx, warp_max_idx);

    __shared__ T warp_maxes[32];
    __shared__ int warp_indices[32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_indices[warp_id] = warp_max_idx;
    }
    __syncthreads();

    if (tid < 32) {
      T val = (tid < num_warps) ? warp_maxes[tid] : (T)(-INFINITY);
      int idx = (tid < num_warps) ? warp_indices[tid] : -1;
      int final_idx;
      T final_max = warp_reduce_max_with_idx(val, idx, final_idx);

      if (tid == 0) {
        if (final_idx < 0) {
          final_idx = 0;
          final_max = (T)0;
        }
        s_topk_vals[ki] = final_max;
        s_topk_idx[ki] = final_idx;
        s_used[final_idx] = true;
      }
    }
    __syncthreads();
  }

  // Now compute softmax over the k values IN-PLACE
  // softmax(x) = exp(x - max) / sum(exp(x - max))
  if (tid == 0) {
    // Find max of topk values
    float max_val = (float)s_topk_vals[0];
    for (int i = 1; i < k; i++) {
      float v = (float)s_topk_vals[i];
      if (v > max_val)
        max_val = v;
    }

    // Compute exp(x - max) and sum using shared memory workspace
    float sum_exp = 0.0f;
    for (int i = 0; i < k; i++) {
      s_softmax_ws[i] = expf((float)s_topk_vals[i] - max_val);
      sum_exp += s_softmax_ws[i];
    }

    // Normalize and write output
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < k; i++) {
      row_weights[i] = (T)(s_softmax_ws[i] * inv_sum);
      row_indices[i] = (uint32_t)s_topk_idx[i];
    }
  }
}

// Wrappers for fused topk+softmax
extern "C" void topk_softmax_f32(const float *input, float *weights_out,
                                 uint32_t *indices_out, int nrows, int ncols,
                                 int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  int block_size = (ncols <= 64)    ? 64
                   : (ncols <= 128) ? 128
                   : (ncols <= 256) ? 256
                                    : 512;
  size_t smem_size = ncols * sizeof(float) + ncols * sizeof(bool) +
                     k * sizeof(float) + k * sizeof(int) + k * sizeof(float);
  topk_softmax_kernel<float><<<nrows, block_size, smem_size, custream>>>(
      input, weights_out, indices_out, nrows, ncols, k);
}

extern "C" void topk_softmax_bf16(const __nv_bfloat16 *input,
                                  __nv_bfloat16 *weights_out,
                                  uint32_t *indices_out, int nrows, int ncols,
                                  int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  int block_size = (ncols <= 64)    ? 64
                   : (ncols <= 128) ? 128
                   : (ncols <= 256) ? 256
                                    : 512;
  size_t smem_size = ncols * sizeof(__nv_bfloat16) + ncols * sizeof(bool) +
                     k * sizeof(__nv_bfloat16) + k * sizeof(int) +
                     k * sizeof(float);
  topk_softmax_kernel<__nv_bfloat16>
      <<<nrows, block_size, smem_size, custream>>>(
          input, weights_out, indices_out, nrows, ncols, k);
}

extern "C" void topk_softmax_f16(const __half *input, __half *weights_out,
                                 uint32_t *indices_out, int nrows, int ncols,
                                 int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  int block_size = (ncols <= 64)    ? 64
                   : (ncols <= 128) ? 128
                   : (ncols <= 256) ? 256
                                    : 512;
  size_t smem_size = ncols * sizeof(__half) + ncols * sizeof(bool) +
                     k * sizeof(__half) + k * sizeof(int) + k * sizeof(float);
  topk_softmax_kernel<__half><<<nrows, block_size, smem_size, custream>>>(
      input, weights_out, indices_out, nrows, ncols, k);
}

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float block_reduce_sum_f32(float val) {
  __shared__ float warp_sums[32];
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (blockDim.x + 31) / 32;

  val = warp_reduce_sum_f32(val);
  if (lane_id == 0) {
    warp_sums[warp_id] = val;
  }
  __syncthreads();

  val = (tid < num_warps) ? warp_sums[tid] : 0.0f;
  if (warp_id == 0) {
    val = warp_reduce_sum_f32(val);
  }
  return val;
}

// Large-vocabulary top-k for token sampling. The MoE top-k kernel above stages
// a full row in shared memory, which is not viable for 100k+ vocabularies. This
// kernel scans fixed-size chunks, emits per-chunk top-k candidates, and computes
// each chunk's contribution to the full softmax denominator.
__global__ void topk_large_stage1_f32(
    const float *__restrict__ input, float *__restrict__ block_values,
    uint32_t *__restrict__ block_indices, float *__restrict__ block_maxes,
    float *__restrict__ block_sums, const int ncols, const int k,
    const int chunk_size, const float inv_temperature) {
  const int chunk = blockIdx.x;
  const int start = chunk * chunk_size;
  const int end = min(start + chunk_size, ncols);
  const int width = max(0, end - start);
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  extern __shared__ char smem[];
  bool *s_used = reinterpret_cast<bool *>(smem);

  for (int i = tid; i < chunk_size; i += block_size) {
    s_used[i] = false;
  }
  __syncthreads();

  for (int ki = 0; ki < k; ++ki) {
    float local_max = -INFINITY;
    int local_idx = -1;

    for (int local = tid; local < width; local += block_size) {
      const float candidate = input[start + local];
      if (!s_used[local] && candidate == candidate && candidate > local_max) {
        local_max = candidate;
        local_idx = start + local;
      }
    }

    int warp_max_idx;
    float warp_max =
        warp_reduce_max_with_idx<float>(local_max, local_idx, warp_max_idx);

    __shared__ float warp_maxes[32];
    __shared__ int warp_indices[32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_indices[warp_id] = warp_max_idx;
    }
    __syncthreads();

    if (tid < 32) {
      float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
      int idx = (tid < num_warps) ? warp_indices[tid] : -1;
      int final_idx;
      float final_max =
          warp_reduce_max_with_idx<float>(val, idx, final_idx);

      if (tid == 0) {
        block_values[chunk * k + ki] = final_max;
        block_indices[chunk * k + ki] =
            final_idx >= 0 ? static_cast<uint32_t>(final_idx) : 0;
        if (final_idx >= start && final_idx < end) {
          s_used[final_idx - start] = true;
        }
      }
    }
    __syncthreads();
  }

  const float block_max =
      width > 0 ? block_values[chunk * k] * inv_temperature : -INFINITY;
  float local_sum = 0.0f;
  if (block_max != -INFINITY) {
    for (int local = tid; local < width; local += block_size) {
      const float candidate = input[start + local];
      if (candidate == candidate) {
        local_sum += expf(candidate * inv_temperature - block_max);
      }
    }
  }

  const float block_sum = block_reduce_sum_f32(local_sum);
  if (tid == 0) {
    block_maxes[chunk] = block_max;
    block_sums[chunk] = block_sum;
  }
}

__global__ void topk_large_stage2_f32(
    const float *__restrict__ block_values,
    const uint32_t *__restrict__ block_indices,
    const float *__restrict__ block_maxes,
    const float *__restrict__ block_sums, float *__restrict__ values_out,
    uint32_t *__restrict__ indices_out, float *__restrict__ softmax_info_out,
    const int nblocks, const int k) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int n_candidates = nblocks * k;

  extern __shared__ char smem[];
  bool *s_used = reinterpret_cast<bool *>(smem);

  for (int i = tid; i < n_candidates; i += block_size) {
    s_used[i] = false;
  }
  __syncthreads();

  float local_global_max = -INFINITY;
  for (int block = tid; block < nblocks; block += block_size) {
    local_global_max = fmaxf(local_global_max, block_maxes[block]);
  }

  int unused_idx;
  float warp_global_max =
      warp_reduce_max_with_idx<float>(local_global_max, tid, unused_idx);

  __shared__ float warp_maxes[32];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_global_max;
  }
  __syncthreads();

  __shared__ float s_global_max;
  if (tid < 32) {
    float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
    int final_idx;
    float final_max = warp_reduce_max_with_idx<float>(val, tid, final_idx);
    if (tid == 0) {
      s_global_max = final_max;
    }
  }
  __syncthreads();

  float local_denom = 0.0f;
  if (s_global_max != -INFINITY) {
    for (int block = tid; block < nblocks; block += block_size) {
      local_denom += block_sums[block] * expf(block_maxes[block] - s_global_max);
    }
  }
  const float denom = block_reduce_sum_f32(local_denom);
  if (tid == 0) {
    softmax_info_out[0] = denom;
    softmax_info_out[1] = s_global_max;
  }
  __syncthreads();

  for (int ki = 0; ki < k; ++ki) {
    float local_max = -INFINITY;
    int local_pos = -1;

    for (int pos = tid; pos < n_candidates; pos += block_size) {
      const float candidate = block_values[pos];
      if (!s_used[pos] && candidate == candidate && candidate > local_max) {
        local_max = candidate;
        local_pos = pos;
      }
    }

    int warp_max_pos;
    float warp_max =
        warp_reduce_max_with_idx<float>(local_max, local_pos, warp_max_pos);

    __shared__ float merge_warp_maxes[32];
    __shared__ int merge_warp_indices[32];

    if (lane_id == 0) {
      merge_warp_maxes[warp_id] = warp_max;
      merge_warp_indices[warp_id] = warp_max_pos;
    }
    __syncthreads();

    if (tid < 32) {
      float val = (tid < num_warps) ? merge_warp_maxes[tid] : -INFINITY;
      int idx = (tid < num_warps) ? merge_warp_indices[tid] : -1;
      int final_pos;
      float final_max =
          warp_reduce_max_with_idx<float>(val, idx, final_pos);

      if (tid == 0) {
        values_out[ki] = final_max;
        indices_out[ki] =
            final_pos >= 0 ? block_indices[final_pos] : static_cast<uint32_t>(0);
        if (final_pos >= 0) {
          s_used[final_pos] = true;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void topk_large_stage2_f32_packed(
    const float *__restrict__ block_values,
    const uint32_t *__restrict__ block_indices,
    const float *__restrict__ block_maxes,
    const float *__restrict__ block_sums, float *__restrict__ packed_out,
    const int nblocks, const int k) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  const int n_candidates = nblocks * k;

  extern __shared__ char smem[];
  bool *s_used = reinterpret_cast<bool *>(smem);

  for (int i = tid; i < n_candidates; i += block_size) {
    s_used[i] = false;
  }
  __syncthreads();

  float local_global_max = -INFINITY;
  for (int block = tid; block < nblocks; block += block_size) {
    local_global_max = fmaxf(local_global_max, block_maxes[block]);
  }

  int unused_idx;
  float warp_global_max =
      warp_reduce_max_with_idx<float>(local_global_max, tid, unused_idx);

  __shared__ float warp_maxes[32];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_global_max;
  }
  __syncthreads();

  __shared__ float s_global_max;
  if (tid < 32) {
    float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
    int final_idx;
    float final_max = warp_reduce_max_with_idx<float>(val, tid, final_idx);
    if (tid == 0) {
      s_global_max = final_max;
    }
  }
  __syncthreads();

  float local_denom = 0.0f;
  if (s_global_max != -INFINITY) {
    for (int block = tid; block < nblocks; block += block_size) {
      local_denom += block_sums[block] * expf(block_maxes[block] - s_global_max);
    }
  }
  const float denom = block_reduce_sum_f32(local_denom);
  if (tid == 0) {
    packed_out[2 * k] = denom;
    packed_out[2 * k + 1] = s_global_max;
  }
  __syncthreads();

  for (int ki = 0; ki < k; ++ki) {
    float local_max = -INFINITY;
    int local_pos = -1;

    for (int pos = tid; pos < n_candidates; pos += block_size) {
      const float candidate = block_values[pos];
      if (!s_used[pos] && candidate == candidate && candidate > local_max) {
        local_max = candidate;
        local_pos = pos;
      }
    }

    int warp_max_pos;
    float warp_max =
        warp_reduce_max_with_idx<float>(local_max, local_pos, warp_max_pos);

    __shared__ float merge_warp_maxes[32];
    __shared__ int merge_warp_indices[32];

    if (lane_id == 0) {
      merge_warp_maxes[warp_id] = warp_max;
      merge_warp_indices[warp_id] = warp_max_pos;
    }
    __syncthreads();

    if (tid < 32) {
      float val = (tid < num_warps) ? merge_warp_maxes[tid] : -INFINITY;
      int idx = (tid < num_warps) ? merge_warp_indices[tid] : -1;
      int final_pos;
      float final_max =
          warp_reduce_max_with_idx<float>(val, idx, final_pos);

      if (tid == 0) {
        packed_out[ki] = final_max;
        packed_out[k + ki] =
            final_pos >= 0 ? static_cast<float>(block_indices[final_pos])
                           : 0.0f;
        if (final_pos >= 0) {
          s_used[final_pos] = true;
        }
      }
    }
    __syncthreads();
  }
}

extern "C" void topk_large_f32(
    const float *input, float *block_values, uint32_t *block_indices,
    float *block_maxes, float *block_sums, float *values_out,
    uint32_t *indices_out, float *softmax_info_out, int ncols, int k,
    int chunk_size, int nblocks, float inv_temperature, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);

  topk_large_stage1_f32<<<nblocks, block_size, stage1_smem, custream>>>(
      input, block_values, block_indices, block_maxes, block_sums, ncols, k,
      chunk_size, inv_temperature);
  topk_large_stage2_f32<<<1, block_size, stage2_smem, custream>>>(
      block_values, block_indices, block_maxes, block_sums, values_out,
      indices_out, softmax_info_out, nblocks, k);
}

extern "C" void topk_large_f32_packed(
    const float *input, float *block_values, uint32_t *block_indices,
    float *block_maxes, float *block_sums, float *packed_out, int ncols, int k,
    int chunk_size, int nblocks, float inv_temperature, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);

  topk_large_stage1_f32<<<nblocks, block_size, stage1_smem, custream>>>(
      input, block_values, block_indices, block_maxes, block_sums, ncols, k,
      chunk_size, inv_temperature);
  topk_large_stage2_f32_packed<<<1, block_size, stage2_smem, custream>>>(
      block_values, block_indices, block_maxes, block_sums, packed_out, nblocks,
      k);
}
