#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <limits>
#include <stdint.h>
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
      if (!s_used[i] && (float)s_data[i] > (float)local_max) {
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
      if (!s_used[i] && (float)s_data[i] > (float)local_max) {
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
