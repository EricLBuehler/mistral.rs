#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <type_traits>

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
    value =
        value > 0.0f ? value / repetition_penalty : value * repetition_penalty;
  }

  logits[token_id] = value;
}

__global__ void apply_sparse_logits_bias_f32_kernel(
    float *__restrict__ logits, const uint32_t *__restrict__ token_ids,
    const float *__restrict__ biases, const int n, const int n_tokens) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_tokens) {
    return;
  }

  const uint32_t token_id = token_ids[idx];
  if (token_id >= static_cast<uint32_t>(n)) {
    return;
  }

  logits[token_id] += biases[idx];
}

extern "C" void
apply_sparse_penalties_f32(const void *x, void *dst, const uint32_t *token_ids,
                           const float *counts, const int n, const int n_tokens,
                           const float frequency_penalty,
                           const float presence_penalty,
                           const float repetition_penalty, int64_t stream) {
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

extern "C" void apply_sparse_logits_bias_f32(
    const void *x, void *dst, const uint32_t *token_ids, const float *biases,
    const int n, const int n_tokens, int64_t stream) {
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

  const int bias_grid = (n_tokens + block - 1) / block;
  apply_sparse_logits_bias_f32_kernel<<<bias_grid, block, 0, custream>>>(
      reinterpret_cast<float *>(dst), token_ids, biases, n, n_tokens);
}

template <typename T>
__device__ __forceinline__ float rms_residual_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float rms_residual_to_float<__half>(__half value) {
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
__device__ __forceinline__ __half rms_residual_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
rms_residual_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T> struct alignas(16) rms_vec8 {
  T data[8];
};

template <typename T>
__device__ __forceinline__ float rms_vec8_sum_squares(const rms_vec8<T> &vec) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float value = rms_residual_to_float(vec.data[i]);
    sum += value * value;
  }
  return sum;
}

template <typename T>
__host__ __forceinline__ bool rms_vec8_aligned(const void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(rms_vec8<T>) == 0;
}

template <typename T>
__host__ __forceinline__ bool rms_vec8_supported(const void *a, const void *b,
                                                 const void *c, const void *d,
                                                 const int ncols) {
  return ncols % 8 == 0 && rms_vec8_aligned<T>(a) && rms_vec8_aligned<T>(b) &&
         rms_vec8_aligned<T>(c) && rms_vec8_aligned<T>(d);
}

__host__ __forceinline__ int rms_vec8_block_size(const int vec_cols) {
  const int rounded = ((vec_cols + 31) / 32) * 32;
  return std::min(std::max(rounded, 32), 1024);
}

__device__ __forceinline__ float rms_block_sum(float value,
                                               float *warp_sums) {
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }

  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  const int num_warps = (blockDim.x + 31) >> 5;
  value = threadIdx.x < num_warps ? warp_sums[lane] : 0.0f;

  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }

  if (threadIdx.x == 0) {
    warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

template <typename T>
__global__ void rms_norm_residual_vec8_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, const T *__restrict__ scale,
    T *__restrict__ dst, const int ncols, const float eps) {
  using Vec = rms_vec8<T>;
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_cols = ncols / 8;
  const int row_offset = row * vec_cols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ weight_vec = reinterpret_cast<const Vec *>(weight);
  Vec *__restrict__ dst_vec = reinterpret_cast<Vec *>(dst);

  float sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    sum += rms_vec8_sum_squares(x_vec[row_offset + col]);
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);

  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    const Vec weight_value = weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float normed = rms_residual_to_float(x_value.data[i]) * inv_rms *
                           rms_residual_to_float(weight_value.data[i]);
      const float value =
          (rms_residual_to_float(residual_value.data[i]) + normed) *
          scale_value;
      out.data[i] = rms_residual_from_float<T>(value);
    }
    dst_vec[idx] = out;
  }
}

template <typename T>
__global__ void rms_norm_residual_kernel(const T *__restrict__ x,
                                         const T *__restrict__ residual,
                                         const T *__restrict__ weight,
                                         const T *__restrict__ scale,
                                         T *__restrict__ dst, const int ncols,
                                         const float eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * ncols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float value = rms_residual_to_float(x[row_offset + col]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);
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
                              const int nrows, const int ncols, const float eps,
                              int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    if (rms_vec8_supported<T>(x, residual, weight, dst, ncols)) {
      const int block = rms_vec8_block_size(ncols / 8);
      rms_norm_residual_vec8_kernel<T><<<nrows, block, 0, custream>>>(
          reinterpret_cast<const T *>(x),
          reinterpret_cast<const T *>(residual),
          reinterpret_cast<const T *>(weight),
          reinterpret_cast<const T *>(scale), reinterpret_cast<T *>(dst),
          ncols, eps);
      return;
    }
  }

  const int block = ncols < 1024 ? 32 : 1024;
  rms_norm_residual_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(scale),
      reinterpret_cast<T *>(dst), ncols, eps);
}

template <typename T>
__global__ void rms_norm_residual_then_rms_norm_vec8_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ residual_weight, const T *__restrict__ scale,
    const T *__restrict__ norm_weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float residual_eps,
    const float norm_eps) {
  using Vec = rms_vec8<T>;
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_cols = ncols / 8;
  const int row_offset = row * vec_cols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ residual_weight_vec =
      reinterpret_cast<const Vec *>(residual_weight);
  const Vec *__restrict__ norm_weight_vec =
      reinterpret_cast<const Vec *>(norm_weight);
  Vec *__restrict__ residual_dst_vec = reinterpret_cast<Vec *>(residual_dst);
  Vec *__restrict__ norm_dst_vec = reinterpret_cast<Vec *>(norm_dst);

  float sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    sum += rms_vec8_sum_squares(x_vec[row_offset + col]);
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) +
             residual_eps);

  float residual_sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    const Vec residual_weight_value = residual_weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float normed =
          rms_residual_to_float(x_value.data[i]) * inv_rms *
          rms_residual_to_float(residual_weight_value.data[i]);
      const float value =
          (rms_residual_to_float(residual_value.data[i]) + normed) *
          scale_value;
      out.data[i] = rms_residual_from_float<T>(value);
      residual_sum += value * value;
    }
    residual_dst_vec[idx] = out;
  }
  const float norm_inv_rms =
      rsqrtf(rms_block_sum(residual_sum, reduce) / static_cast<float>(ncols) +
             norm_eps);

  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec residual_value = residual_dst_vec[idx];
    const Vec norm_weight_value = norm_weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = rms_residual_to_float(residual_value.data[i]) *
                          norm_inv_rms *
                          rms_residual_to_float(norm_weight_value.data[i]);
      out.data[i] = rms_residual_from_float<T>(value);
    }
    norm_dst_vec[idx] = out;
  }
}

template <typename T>
__global__ void rms_norm_residual_then_rms_norm_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ residual_weight, const T *__restrict__ scale,
    const T *__restrict__ norm_weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float residual_eps,
    const float norm_eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * ncols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float value = rms_residual_to_float(x[row_offset + col]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) +
             residual_eps);
  float residual_sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    const float normed = rms_residual_to_float(x[idx]) * inv_rms *
                         rms_residual_to_float(residual_weight[col]);
    const float value =
        (rms_residual_to_float(residual[idx]) + normed) * scale_value;
    residual_dst[idx] = rms_residual_from_float<T>(value);
    residual_sum += value * value;
  }
  const float norm_inv_rms =
      rsqrtf(rms_block_sum(residual_sum, reduce) / static_cast<float>(ncols) +
             norm_eps);
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    const float value = rms_residual_to_float(residual_dst[idx]) *
                        norm_inv_rms * rms_residual_to_float(norm_weight[col]);
    norm_dst[idx] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_residual_then_rms_norm(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    if (rms_vec8_supported<T>(x, residual, residual_weight, residual_dst,
                              ncols) &&
        rms_vec8_aligned<T>(norm_weight) &&
        rms_vec8_aligned<T>(norm_dst)) {
      const int block = rms_vec8_block_size(ncols / 8);
      rms_norm_residual_then_rms_norm_vec8_kernel<T>
          <<<nrows, block, 0, custream>>>(
              reinterpret_cast<const T *>(x),
              reinterpret_cast<const T *>(residual),
              reinterpret_cast<const T *>(residual_weight),
              reinterpret_cast<const T *>(scale),
              reinterpret_cast<const T *>(norm_weight),
              reinterpret_cast<T *>(residual_dst),
              reinterpret_cast<T *>(norm_dst), ncols, residual_eps, norm_eps);
      return;
    }
  }

  const int block = ncols < 1024 ? 32 : 1024;
  rms_norm_residual_then_rms_norm_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(residual_weight),
      reinterpret_cast<const T *>(scale),
      reinterpret_cast<const T *>(norm_weight), reinterpret_cast<T *>(residual_dst),
      reinterpret_cast<T *>(norm_dst), ncols, residual_eps, norm_eps);
}

template <typename T>
__global__ void rms_norm_strided_4d_kernel(
    const T *__restrict__ x, const T *__restrict__ weight, T *__restrict__ dst,
    const int64_t stride_b, const int64_t stride_h, const int64_t stride_s,
    const int64_t stride_d, const int batch, const int heads,
    const int seq_len, const int head_dim, const float eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int seq = row % seq_len;
  const int tmp = row / seq_len;
  const int head = tmp % heads;
  const int batch_idx = tmp / heads;
  const int64_t src_base = static_cast<int64_t>(batch_idx) * stride_b +
                           static_cast<int64_t>(head) * stride_h +
                           static_cast<int64_t>(seq) * stride_s;
  const int64_t dst_base = static_cast<int64_t>(row) * head_dim;

  float sum = 0.0f;
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = rms_residual_to_float(x[src_base + col * stride_d]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(head_dim) + eps);
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = rms_residual_to_float(x[src_base + col * stride_d]) *
                        inv_rms * rms_residual_to_float(weight[col]);
    dst[dst_base + col] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_strided_4d(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  if (batch <= 0 || heads <= 0 || seq_len <= 0 || head_dim <= 0) {
    return;
  }

  const int total_rows = batch * heads * seq_len;
  int block = 32;
  while (block < head_dim && block < 1024) {
    block <<= 1;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  rms_norm_strided_4d_kernel<T><<<total_rows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(weight),
      reinterpret_cast<T *>(dst), stride_b, stride_h, stride_s, stride_d, batch,
      heads, seq_len, head_dim, eps);
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

extern "C" void rms_norm_residual_then_rms_norm_f32(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<float>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

extern "C" void rms_norm_residual_then_rms_norm_f16(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<__half>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

extern "C" void rms_norm_residual_then_rms_norm_bf16(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<__nv_bfloat16>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

extern "C" void rms_norm_strided_4d_f32(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<float>(x, weight, dst, stride_b, stride_h,
                                    stride_s, stride_d, batch, heads, seq_len,
                                    head_dim, eps, stream);
}

extern "C" void rms_norm_strided_4d_f16(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<__half>(x, weight, dst, stride_b, stride_h,
                                     stride_s, stride_d, batch, heads, seq_len,
                                     head_dim, eps, stream);
}

extern "C" void rms_norm_strided_4d_bf16(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<__nv_bfloat16>(x, weight, dst, stride_b, stride_h,
                                            stride_s, stride_d, batch, heads,
                                            seq_len, head_dim, eps, stream);
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

constexpr int MOE_ROUTER_SCORE_RAW = 0;
constexpr int MOE_ROUTER_SCORE_SOFTMAX = 1;
constexpr int MOE_ROUTER_SCORE_SIGMOID = 2;
constexpr int MOE_ROUTER_WEIGHT_SCORE = 0;
constexpr int MOE_ROUTER_WEIGHT_SOFTMAX = 1;
constexpr int MOE_ROUTER_WEIGHT_SIGMOID = 2;
constexpr int MOE_ROUTER_ROWS_PER_BLOCK = 4;

template <typename T> __device__ __forceinline__ float router_to_float(T x) {
  return static_cast<float>(x);
}

template <> __device__ __forceinline__ float router_to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float
router_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float router_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

__device__ __forceinline__ float router_warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
  }
  return val;
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_softmax_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
  float max_val = -INFINITY;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      max_val = max(max_val, vals[i]);
    }
  }

  max_val = router_warp_reduce_max(max_val);
  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] = expf(vals[i] - max_val);
      sum += vals[i];
    } else {
      vals[i] = 0.0f;
    }
  }

  sum = router_warp_reduce_sum(sum);
  const float inv_sum = 1.0f / sum;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] *= inv_sum;
    }
  }
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_sigmoid_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    vals[i] = (!use_limit || idx < limit) ? 1.0f / (1.0f + expf(-vals[i]))
                                          : -INFINITY;
  }
}

template <typename T, int n_experts, bool has_bias, bool has_expert_scale>
__launch_bounds__(MOE_ROUTER_ROWS_PER_BLOCK * 32, 1) __global__
    void moe_router_topk_kernel(
        const T *__restrict__ logits, float *__restrict__ weights,
        uint32_t *__restrict__ ids, const float *__restrict__ selection_bias,
        const float *__restrict__ expert_scale, const int n_rows,
        const int top_k, const int score_mode, const int weight_mode,
        const bool renormalize, const bool clamp_logits, const float clamp_min,
        const float clamp_max, const float norm_min, const float output_scale) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }

  logits += row * n_experts;
  weights += row * top_k;
  ids += row * top_k;

  constexpr int values_per_thread = n_experts > 32 ? n_experts / 32 : 1;

  float raw[values_per_thread];
  float score[values_per_thread];
  float selection[values_per_thread];
  float output_weights[values_per_thread];
  uint32_t output_ids[values_per_thread];

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    float value = (n_experts % 32 == 0 || expert < n_experts)
                      ? router_to_float(logits[expert])
                      : -INFINITY;
    if (clamp_logits && expert < n_experts) {
      value = fminf(fmaxf(value, clamp_min), clamp_max);
    }
    if (value != value) {
      value = -INFINITY;
    }
    raw[i] = value;
    score[i] = value;
    selection[i] = value;
    output_weights[i] = 0.0f;
    output_ids[i] = 0;
  }

  if (score_mode == MOE_ROUTER_SCORE_SOFTMAX) {
    router_softmax_warp<values_per_thread, false>(score, n_experts, lane);
  } else if (score_mode == MOE_ROUTER_SCORE_SIGMOID) {
    router_sigmoid_warp<values_per_thread, false>(score, n_experts, lane);
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    selection[i] = score[i];
    if constexpr (has_bias) {
      if (expert < n_experts) {
        selection[i] += selection_bias[expert];
      }
    }
    if (selection[i] != selection[i]) {
      selection[i] = -INFINITY;
    }
  }

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    float best_selection = selection[0];
    float best_score = score[0];
    float best_raw = raw[0];
    int best_expert = lane;

#pragma unroll
    for (int i = 1; i < values_per_thread; i++) {
      const int expert = lane + i * 32;
      if ((n_experts % 32 == 0 || expert < n_experts) &&
          selection[i] > best_selection) {
        best_selection = selection[i];
        best_score = score[i];
        best_raw = raw[i];
        best_expert = expert;
      }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      const float other_selection =
          __shfl_xor_sync(0xffffffff, best_selection, mask, 32);
      const float other_score =
          __shfl_xor_sync(0xffffffff, best_score, mask, 32);
      const float other_raw = __shfl_xor_sync(0xffffffff, best_raw, mask, 32);
      const int other_expert =
          __shfl_xor_sync(0xffffffff, best_expert, mask, 32);
      if (other_selection > best_selection ||
          (other_selection == best_selection && other_expert < best_expert)) {
        best_selection = other_selection;
        best_score = other_score;
        best_raw = other_raw;
        best_expert = other_expert;
      }
    }

    float out = best_score;
    if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
      out = best_raw;
    } else if (weight_mode == MOE_ROUTER_WEIGHT_SIGMOID) {
      out = 1.0f / (1.0f + expf(-best_raw));
    }

    if ((k_idx & 31) == lane) {
      output_weights[k_idx / 32] = out;
      output_ids[k_idx / 32] = static_cast<uint32_t>(best_expert);
    }

    if ((best_expert & 31) == lane) {
      selection[best_expert / 32] = -INFINITY;
    }
  }

  if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
    router_softmax_warp<values_per_thread, true>(output_weights, top_k, lane);
  }

  if (renormalize) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      const int idx = lane + i * 32;
      if (idx < top_k) {
        sum += output_weights[i];
      }
    }
    sum = router_warp_reduce_sum(sum);
    sum = fmaxf(sum, norm_min);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      output_weights[i] *= inv_sum;
    }
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (idx < top_k) {
      float scale = output_scale;
      if constexpr (has_expert_scale) {
        scale *= expert_scale[output_ids[i]];
      }
      weights[idx] = output_weights[i] * scale;
      ids[idx] = output_ids[i];
    }
  }
}

template <typename T, bool has_bias, bool has_expert_scale>
void launch_moe_router_topk(const T *logits, float *weights, uint32_t *ids,
                            const float *selection_bias,
                            const float *expert_scale, const int n_rows,
                            const int n_experts, const int top_k,
                            const int score_mode, const int weight_mode,
                            const bool renormalize, const bool clamp_logits,
                            const float clamp_min, const float clamp_max,
                            const float norm_min, const float output_scale,
                            int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block_dims(32, MOE_ROUTER_ROWS_PER_BLOCK, 1);
  dim3 grid_dims((n_rows + MOE_ROUTER_ROWS_PER_BLOCK - 1) /
                     MOE_ROUTER_ROWS_PER_BLOCK,
                 1, 1);

#define LAUNCH_MOE_ROUTER_TOPK(EXPERTS)                                        \
  moe_router_topk_kernel<T, EXPERTS, has_bias, has_expert_scale>               \
      <<<grid_dims, block_dims, 0, custream>>>(                                \
          logits, weights, ids, selection_bias, expert_scale, n_rows, top_k,   \
          score_mode, weight_mode, renormalize, clamp_logits, clamp_min,       \
          clamp_max, norm_min, output_scale);                                  \
  break

  switch (n_experts) {
  case 1:
    LAUNCH_MOE_ROUTER_TOPK(1);
  case 2:
    LAUNCH_MOE_ROUTER_TOPK(2);
  case 4:
    LAUNCH_MOE_ROUTER_TOPK(4);
  case 8:
    LAUNCH_MOE_ROUTER_TOPK(8);
  case 16:
    LAUNCH_MOE_ROUTER_TOPK(16);
  case 32:
    LAUNCH_MOE_ROUTER_TOPK(32);
  case 64:
    LAUNCH_MOE_ROUTER_TOPK(64);
  case 128:
    LAUNCH_MOE_ROUTER_TOPK(128);
  case 256:
    LAUNCH_MOE_ROUTER_TOPK(256);
  case 512:
    LAUNCH_MOE_ROUTER_TOPK(512);
  case 576:
    LAUNCH_MOE_ROUTER_TOPK(576);
  default:
    break;
  }

#undef LAUNCH_MOE_ROUTER_TOPK
}

template <typename T>
void moe_router_topk_dispatch(const T *logits, float *weights, uint32_t *ids,
                              const float *selection_bias,
                              const float *expert_scale, const int n_rows,
                              const int n_experts, const int top_k,
                              const int score_mode, const int weight_mode,
                              const bool renormalize, const bool clamp_logits,
                              const float clamp_min, const float clamp_max,
                              const float norm_min, const float output_scale,
                              int64_t stream) {
  if (selection_bias != nullptr && expert_scale != nullptr) {
    launch_moe_router_topk<T, true, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (selection_bias != nullptr) {
    launch_moe_router_topk<T, true, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (expert_scale != nullptr) {
    launch_moe_router_topk<T, false, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else {
    launch_moe_router_topk<T, false, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  }
}

extern "C" void moe_router_topk_f32(const float *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<float>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void
moe_router_topk_bf16(const __nv_bfloat16 *logits, float *weights, uint32_t *ids,
                     const float *selection_bias, const float *expert_scale,
                     int n_rows, int n_experts, int top_k, int score_mode,
                     int weight_mode, bool renormalize, bool clamp_logits,
                     float clamp_min, float clamp_max, float norm_min,
                     float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__nv_bfloat16>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void moe_router_topk_f16(const __half *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__half>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
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
// kernel scans fixed-size chunks, emits per-chunk top-k candidates, and
// computes each chunk's contribution to the full softmax denominator.
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
      float final_max = warp_reduce_max_with_idx<float>(val, idx, final_idx);

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
    const float *__restrict__ block_maxes, const float *__restrict__ block_sums,
    float *__restrict__ values_out, uint32_t *__restrict__ indices_out,
    float *__restrict__ softmax_info_out, const int nblocks, const int k) {
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
      local_denom +=
          block_sums[block] * expf(block_maxes[block] - s_global_max);
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
      float final_max = warp_reduce_max_with_idx<float>(val, idx, final_pos);

      if (tid == 0) {
        values_out[ki] = final_max;
        indices_out[ki] = final_pos >= 0 ? block_indices[final_pos]
                                         : static_cast<uint32_t>(0);
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
    const float *__restrict__ block_maxes, const float *__restrict__ block_sums,
    float *__restrict__ packed_out, const int nblocks, const int k) {
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
      local_denom +=
          block_sums[block] * expf(block_maxes[block] - s_global_max);
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
      float final_max = warp_reduce_max_with_idx<float>(val, idx, final_pos);

      if (tid == 0) {
        packed_out[ki] = final_max;
        packed_out[k + ki] = final_pos >= 0
                                 ? static_cast<float>(block_indices[final_pos])
                                 : 0.0f;
        if (final_pos >= 0) {
          s_used[final_pos] = true;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void top1_large_stage1_f32(const float *__restrict__ input,
                                      float *__restrict__ block_values,
                                      uint32_t *__restrict__ block_indices,
                                      const int ncols,
                                      const int chunk_size) {
  const int chunk = blockIdx.x;
  const int start = chunk * chunk_size;
  const int end = min(start + chunk_size, ncols);
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  float local_max = -INFINITY;
  int local_idx = -1;
  for (int idx = start + tid; idx < end; idx += block_size) {
    const float candidate = input[idx];
    if (candidate == candidate && candidate > local_max) {
      local_max = candidate;
      local_idx = idx;
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
    float final_max = warp_reduce_max_with_idx<float>(val, idx, final_idx);

    if (tid == 0) {
      block_values[chunk] = final_max;
      block_indices[chunk] =
          final_idx >= 0 ? static_cast<uint32_t>(final_idx) : 0;
    }
  }
}

__global__ void top1_large_stage2_f32_packed(
    const float *__restrict__ block_values,
    const uint32_t *__restrict__ block_indices, float *__restrict__ packed_out,
    const int nblocks) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  float local_max = -INFINITY;
  int local_pos = -1;
  for (int pos = tid; pos < nblocks; pos += block_size) {
    const float candidate = block_values[pos];
    if (candidate == candidate && candidate > local_max) {
      local_max = candidate;
      local_pos = pos;
    }
  }

  int warp_max_pos;
  float warp_max =
      warp_reduce_max_with_idx<float>(local_max, local_pos, warp_max_pos);

  __shared__ float warp_maxes[32];
  __shared__ int warp_indices[32];

  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_max;
    warp_indices[warp_id] = warp_max_pos;
  }
  __syncthreads();

  if (tid < 32) {
    float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
    int pos = (tid < num_warps) ? warp_indices[tid] : -1;
    int final_pos;
    float final_max = warp_reduce_max_with_idx<float>(val, pos, final_pos);

    if (tid == 0) {
      packed_out[0] = final_max;
      packed_out[1] = final_pos >= 0
                          ? static_cast<float>(block_indices[final_pos])
                          : 0.0f;
    }
  }
}

extern "C" void topk_large_f32(const float *input, float *block_values,
                               uint32_t *block_indices, float *block_maxes,
                               float *block_sums, float *values_out,
                               uint32_t *indices_out, float *softmax_info_out,
                               int ncols, int k, int chunk_size, int nblocks,
                               float inv_temperature, int64_t stream) {
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

extern "C" void topk_large_f32_packed(const float *input, float *block_values,
                                      uint32_t *block_indices,
                                      float *block_maxes, float *block_sums,
                                      float *packed_out, int ncols, int k,
                                      int chunk_size, int nblocks,
                                      float inv_temperature, int64_t stream) {
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

extern "C" void top1_large_f32_packed(const float *input, float *block_values,
                                      uint32_t *block_indices,
                                      float *packed_out, int ncols,
                                      int chunk_size, int nblocks,
                                      int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;

  top1_large_stage1_f32<<<nblocks, block_size, 0, custream>>>(
      input, block_values, block_indices, ncols, chunk_size);
  top1_large_stage2_f32_packed<<<1, block_size, 0, custream>>>(
      block_values, block_indices, packed_out, nblocks);
}
