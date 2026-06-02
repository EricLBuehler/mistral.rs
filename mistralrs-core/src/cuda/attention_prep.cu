#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

template <typename T> __device__ __forceinline__ float ap_to_float(T value) {
  return static_cast<float>(value);
}

template <> __device__ __forceinline__ float ap_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float
ap_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T ap_from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half ap_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
ap_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T, bool IS_NEOX>
__device__ __forceinline__ void write_norm_rope_row(
    const T *__restrict__ src, const T *__restrict__ weight,
    const T *__restrict__ cos, const T *__restrict__ sin, T *__restrict__ dst,
    const int64_t src_base, const int64_t src_stride_d, const int head_dim,
    const int rot_dim, const float eps, volatile float *__restrict__ reduce) {
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = ap_to_float(src[src_base + col * src_stride_d]);
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

  const float inv_rms = rsqrtf(reduce[0] / static_cast<float>(head_dim) + eps);

  for (int rot_offset = tid; rot_offset < rot_dim; rot_offset += blockDim.x) {
    int x_idx;
    int y_idx;
    if constexpr (IS_NEOX) {
      x_idx = rot_offset;
      y_idx = rot_dim + rot_offset;
    } else {
      x_idx = 2 * rot_offset;
      y_idx = x_idx + 1;
    }

    const float x = ap_to_float(src[src_base + x_idx * src_stride_d]) *
                    inv_rms * ap_to_float(weight[x_idx]);
    const float y = ap_to_float(src[src_base + y_idx * src_stride_d]) *
                    inv_rms * ap_to_float(weight[y_idx]);
    const float c = ap_to_float(cos[rot_offset]);
    const float s = ap_to_float(sin[rot_offset]);

    dst[x_idx] = ap_from_float<T>(x * c - y * s);
    dst[y_idx] = ap_from_float<T>(y * c + x * s);
  }

  const int rotated_width = rot_dim * 2;
  for (int col = rotated_width + tid; col < head_dim; col += blockDim.x) {
    const float value = ap_to_float(src[src_base + col * src_stride_d]) *
                        inv_rms * ap_to_float(weight[col]);
    dst[col] = ap_from_float<T>(value);
  }
}

template <typename T>
__device__ __forceinline__ void
write_norm_row(const T *__restrict__ src, const T *__restrict__ weight,
               T *__restrict__ dst, const int64_t src_base,
               const int64_t src_stride_d, const int head_dim, const float eps,
               volatile float *__restrict__ reduce) {
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = ap_to_float(src[src_base + col * src_stride_d]);
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

  const float inv_rms = rsqrtf(reduce[0] / static_cast<float>(head_dim) + eps);
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = ap_to_float(src[src_base + col * src_stride_d]) *
                        inv_rms * ap_to_float(weight[col]);
    dst[col] = ap_from_float<T>(value);
  }
}

template <typename T, bool IS_NEOX>
__global__ void qk_rms_norm_rope_kernel(
    const T *__restrict__ q, const T *__restrict__ k,
    const T *__restrict__ q_weight, const T *__restrict__ k_weight,
    const T *__restrict__ cos, const T *__restrict__ sin, T *__restrict__ q_out,
    T *__restrict__ k_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const int cos_batch_stride, const float q_eps,
    const float k_eps) {
  __shared__ float reduce[1024];

  const int q_rows = batch * q_heads * seq_len;
  const int row = blockIdx.x;
  const bool is_q = row < q_rows;
  const int local_row = is_q ? row : row - q_rows;
  const int heads = is_q ? q_heads : k_heads;

  const int seq = local_row % seq_len;
  const int tmp = local_row / seq_len;
  const int head = tmp % heads;
  const int batch_idx = tmp / heads;

  const int cos_row =
      cos_batch_stride == 0 ? seq : batch_idx * cos_batch_stride + seq;
  const T *cos_row_ptr = cos + static_cast<int64_t>(cos_row) * rot_dim;
  const T *sin_row_ptr = sin + static_cast<int64_t>(cos_row) * rot_dim;

  if (is_q) {
    const int64_t src_base = static_cast<int64_t>(batch_idx) * q_stride_b +
                             static_cast<int64_t>(head) * q_stride_h +
                             static_cast<int64_t>(seq) * q_stride_s;
    T *dst = q_out + static_cast<int64_t>(row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(q, q_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, q_stride_d, head_dim, rot_dim,
                                    q_eps, reduce);
  } else {
    const int64_t src_base = static_cast<int64_t>(batch_idx) * k_stride_b +
                             static_cast<int64_t>(head) * k_stride_h +
                             static_cast<int64_t>(seq) * k_stride_s;
    T *dst = k_out + static_cast<int64_t>(local_row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(k, k_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, k_stride_d, head_dim, rot_dim,
                                    k_eps, reduce);
  }
}

template <typename T, bool IS_NEOX>
__global__ void qk_rms_norm_rope_positions_kernel(
    const T *__restrict__ q, const T *__restrict__ k,
    const T *__restrict__ q_weight, const T *__restrict__ k_weight,
    const T *__restrict__ cos, const T *__restrict__ sin,
    const uint32_t *__restrict__ positions, T *__restrict__ q_out,
    T *__restrict__ k_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps) {
  __shared__ float reduce[1024];

  const int q_rows = batch * q_heads * seq_len;
  const int row = blockIdx.x;
  const bool is_q = row < q_rows;
  const int local_row = is_q ? row : row - q_rows;
  const int heads = is_q ? q_heads : k_heads;

  const int seq = local_row % seq_len;
  const int tmp = local_row / seq_len;
  const int head = tmp % heads;
  const int batch_idx = tmp / heads;

  const uint32_t pos = positions[batch_idx] + static_cast<uint32_t>(seq);
  const T *cos_row_ptr = cos + static_cast<int64_t>(pos) * rot_dim;
  const T *sin_row_ptr = sin + static_cast<int64_t>(pos) * rot_dim;

  if (is_q) {
    const int64_t src_base = static_cast<int64_t>(batch_idx) * q_stride_b +
                             static_cast<int64_t>(head) * q_stride_h +
                             static_cast<int64_t>(seq) * q_stride_s;
    T *dst = q_out + static_cast<int64_t>(row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(q, q_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, q_stride_d, head_dim, rot_dim,
                                    q_eps, reduce);
  } else {
    const int64_t src_base = static_cast<int64_t>(batch_idx) * k_stride_b +
                             static_cast<int64_t>(head) * k_stride_h +
                             static_cast<int64_t>(seq) * k_stride_s;
    T *dst = k_out + static_cast<int64_t>(local_row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(k, k_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, k_stride_d, head_dim, rot_dim,
                                    k_eps, reduce);
  }
}

template <typename T, bool IS_NEOX>
__global__ void qkv_rms_norm_rope_positions_kernel(
    const T *__restrict__ q, const T *__restrict__ k, const T *__restrict__ v,
    const T *__restrict__ q_weight, const T *__restrict__ k_weight,
    const T *__restrict__ v_weight, const T *__restrict__ cos,
    const T *__restrict__ sin, const uint32_t *__restrict__ positions,
    T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out,
    const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d,
    const int64_t v_stride_b, const int64_t v_stride_h,
    const int64_t v_stride_s, const int64_t v_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps,
    const float v_eps) {
  __shared__ float reduce[1024];

  const int q_rows = batch * q_heads * seq_len;
  const int kv_rows = batch * k_heads * seq_len;
  const int row = blockIdx.x;

  if (row < q_rows) {
    const int seq = row % seq_len;
    const int tmp = row / seq_len;
    const int head = tmp % q_heads;
    const int batch_idx = tmp / q_heads;
    const uint32_t pos = positions[batch_idx] + static_cast<uint32_t>(seq);
    const T *cos_row_ptr = cos + static_cast<int64_t>(pos) * rot_dim;
    const T *sin_row_ptr = sin + static_cast<int64_t>(pos) * rot_dim;
    const int64_t src_base = static_cast<int64_t>(batch_idx) * q_stride_b +
                             static_cast<int64_t>(head) * q_stride_h +
                             static_cast<int64_t>(seq) * q_stride_s;
    T *dst = q_out + static_cast<int64_t>(row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(q, q_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, q_stride_d, head_dim, rot_dim,
                                    q_eps, reduce);
    return;
  }

  const int kv_row = row - q_rows;
  const bool is_k = kv_row < kv_rows;
  const int local_row = is_k ? kv_row : kv_row - kv_rows;
  const int seq = local_row % seq_len;
  const int tmp = local_row / seq_len;
  const int head = tmp % k_heads;
  const int batch_idx = tmp / k_heads;

  if (is_k) {
    const uint32_t pos = positions[batch_idx] + static_cast<uint32_t>(seq);
    const T *cos_row_ptr = cos + static_cast<int64_t>(pos) * rot_dim;
    const T *sin_row_ptr = sin + static_cast<int64_t>(pos) * rot_dim;
    const int64_t src_base = static_cast<int64_t>(batch_idx) * k_stride_b +
                             static_cast<int64_t>(head) * k_stride_h +
                             static_cast<int64_t>(seq) * k_stride_s;
    T *dst = k_out + static_cast<int64_t>(local_row) * head_dim;
    write_norm_rope_row<T, IS_NEOX>(k, k_weight, cos_row_ptr, sin_row_ptr, dst,
                                    src_base, k_stride_d, head_dim, rot_dim,
                                    k_eps, reduce);
  } else {
    const int64_t src_base = static_cast<int64_t>(batch_idx) * v_stride_b +
                             static_cast<int64_t>(head) * v_stride_h +
                             static_cast<int64_t>(seq) * v_stride_s;
    T *dst = v_out + static_cast<int64_t>(local_row) * head_dim;
    write_norm_row<T>(v, v_weight, dst, src_base, v_stride_d, head_dim, v_eps,
                      reduce);
  }
}

template <typename T, bool IS_NEOX>
void launch_qk_rms_norm_rope(const void *q, const void *k, const void *q_weight,
                             const void *k_weight, const void *cos,
                             const void *sin, void *q_out, void *k_out,
                             const int64_t q_stride_b, const int64_t q_stride_h,
                             const int64_t q_stride_s, const int64_t q_stride_d,
                             const int64_t k_stride_b, const int64_t k_stride_h,
                             const int64_t k_stride_s, const int64_t k_stride_d,
                             const int batch, const int q_heads,
                             const int k_heads, const int seq_len,
                             const int head_dim, const int rot_dim,
                             const int cos_batch_stride, const float q_eps,
                             const float k_eps, int64_t stream) {
  if (batch <= 0 || q_heads <= 0 || seq_len <= 0 || head_dim <= 0 ||
      rot_dim <= 0) {
    return;
  }

  const int total_rows = batch * (q_heads + k_heads) * seq_len;
  if (total_rows <= 0) {
    return;
  }

  int block = 32;
  while (block < head_dim && block < 1024) {
    block <<= 1;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  qk_rms_norm_rope_kernel<T, IS_NEOX><<<total_rows, block, 0, custream>>>(
      reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k),
      reinterpret_cast<const T *>(q_weight),
      reinterpret_cast<const T *>(k_weight), reinterpret_cast<const T *>(cos),
      reinterpret_cast<const T *>(sin), reinterpret_cast<T *>(q_out),
      reinterpret_cast<T *>(k_out), q_stride_b, q_stride_h, q_stride_s,
      q_stride_d, k_stride_b, k_stride_h, k_stride_s, k_stride_d, batch,
      q_heads, k_heads, seq_len, head_dim, rot_dim, cos_batch_stride, q_eps,
      k_eps);
}

template <typename T, bool IS_NEOX>
void launch_qk_rms_norm_rope_positions(
    const void *q, const void *k, const void *q_weight, const void *k_weight,
    const void *cos, const void *sin, const void *positions, void *q_out,
    void *k_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps, int64_t stream) {
  if (batch <= 0 || q_heads <= 0 || seq_len <= 0 || head_dim <= 0 ||
      rot_dim <= 0) {
    return;
  }

  const int total_rows = batch * (q_heads + k_heads) * seq_len;
  if (total_rows <= 0) {
    return;
  }

  int block = 32;
  while (block < head_dim && block < 1024) {
    block <<= 1;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  qk_rms_norm_rope_positions_kernel<T, IS_NEOX>
      <<<total_rows, block, 0, custream>>>(
          reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k),
          reinterpret_cast<const T *>(q_weight),
          reinterpret_cast<const T *>(k_weight),
          reinterpret_cast<const T *>(cos), reinterpret_cast<const T *>(sin),
          reinterpret_cast<const uint32_t *>(positions),
          reinterpret_cast<T *>(q_out), reinterpret_cast<T *>(k_out),
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps);
}

template <typename T, bool IS_NEOX>
void launch_qkv_rms_norm_rope_positions(
    const void *q, const void *k, const void *v, const void *q_weight,
    const void *k_weight, const void *v_weight, const void *cos,
    const void *sin, const void *positions, void *q_out, void *k_out,
    void *v_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d,
    const int64_t v_stride_b, const int64_t v_stride_h,
    const int64_t v_stride_s, const int64_t v_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps, const float v_eps,
    int64_t stream) {
  if (batch <= 0 || q_heads <= 0 || k_heads <= 0 || seq_len <= 0 ||
      head_dim <= 0 || rot_dim <= 0) {
    return;
  }

  const int total_rows = batch * (q_heads + k_heads + k_heads) * seq_len;
  if (total_rows <= 0) {
    return;
  }

  int block = 32;
  while (block < head_dim && block < 1024) {
    block <<= 1;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  qkv_rms_norm_rope_positions_kernel<T, IS_NEOX>
      <<<total_rows, block, 0, custream>>>(
          reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k),
          reinterpret_cast<const T *>(v), reinterpret_cast<const T *>(q_weight),
          reinterpret_cast<const T *>(k_weight),
          reinterpret_cast<const T *>(v_weight),
          reinterpret_cast<const T *>(cos), reinterpret_cast<const T *>(sin),
          reinterpret_cast<const uint32_t *>(positions),
          reinterpret_cast<T *>(q_out), reinterpret_cast<T *>(k_out),
          reinterpret_cast<T *>(v_out), q_stride_b, q_stride_h, q_stride_s,
          q_stride_d, k_stride_b, k_stride_h, k_stride_s, k_stride_d,
          v_stride_b, v_stride_h, v_stride_s, v_stride_d, batch, q_heads,
          k_heads, seq_len, head_dim, rot_dim, q_eps, k_eps, v_eps);
}

extern "C" void qk_rms_norm_rope(
    const void *q, const void *k, const void *q_weight, const void *k_weight,
    const void *cos, const void *sin, void *q_out, void *k_out,
    const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const int cos_batch_stride, const float q_eps,
    const float k_eps, const int is_neox, const int dtype, int64_t stream) {
  if (is_neox) {
    if (dtype == 0) {
      launch_qk_rms_norm_rope<__half, true>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    } else if (dtype == 1) {
      launch_qk_rms_norm_rope<__nv_bfloat16, true>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    } else if (dtype == 2) {
      launch_qk_rms_norm_rope<float, true>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    }
  } else {
    if (dtype == 0) {
      launch_qk_rms_norm_rope<__half, false>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    } else if (dtype == 1) {
      launch_qk_rms_norm_rope<__nv_bfloat16, false>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    } else if (dtype == 2) {
      launch_qk_rms_norm_rope<float, false>(
          q, k, q_weight, k_weight, cos, sin, q_out, k_out, q_stride_b,
          q_stride_h, q_stride_s, q_stride_d, k_stride_b, k_stride_h,
          k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len, head_dim,
          rot_dim, cos_batch_stride, q_eps, k_eps, stream);
    }
  }
}

extern "C" void qk_rms_norm_rope_positions(
    const void *q, const void *k, const void *q_weight, const void *k_weight,
    const void *cos, const void *sin, const void *positions, void *q_out,
    void *k_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps, const int is_neox,
    const int dtype, int64_t stream) {
  if (is_neox) {
    if (dtype == 0) {
      launch_qk_rms_norm_rope_positions<__half, true>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    } else if (dtype == 1) {
      launch_qk_rms_norm_rope_positions<__nv_bfloat16, true>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    } else if (dtype == 2) {
      launch_qk_rms_norm_rope_positions<float, true>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    }
  } else {
    if (dtype == 0) {
      launch_qk_rms_norm_rope_positions<__half, false>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    } else if (dtype == 1) {
      launch_qk_rms_norm_rope_positions<__nv_bfloat16, false>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    } else if (dtype == 2) {
      launch_qk_rms_norm_rope_positions<float, false>(
          q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out,
          q_stride_b, q_stride_h, q_stride_s, q_stride_d, k_stride_b,
          k_stride_h, k_stride_s, k_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, stream);
    }
  }
}

extern "C" void qkv_rms_norm_rope_positions(
    const void *q, const void *k, const void *v, const void *q_weight,
    const void *k_weight, const void *v_weight, const void *cos,
    const void *sin, const void *positions, void *q_out, void *k_out,
    void *v_out, const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_s, const int64_t q_stride_d,
    const int64_t k_stride_b, const int64_t k_stride_h,
    const int64_t k_stride_s, const int64_t k_stride_d,
    const int64_t v_stride_b, const int64_t v_stride_h,
    const int64_t v_stride_s, const int64_t v_stride_d, const int batch,
    const int q_heads, const int k_heads, const int seq_len, const int head_dim,
    const int rot_dim, const float q_eps, const float k_eps, const float v_eps,
    const int is_neox, const int dtype, int64_t stream) {
  if (is_neox) {
    if (dtype == 0) {
      launch_qkv_rms_norm_rope_positions<__half, true>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    } else if (dtype == 1) {
      launch_qkv_rms_norm_rope_positions<__nv_bfloat16, true>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    } else if (dtype == 2) {
      launch_qkv_rms_norm_rope_positions<float, true>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    }
  } else {
    if (dtype == 0) {
      launch_qkv_rms_norm_rope_positions<__half, false>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    } else if (dtype == 1) {
      launch_qkv_rms_norm_rope_positions<__nv_bfloat16, false>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    } else if (dtype == 2) {
      launch_qkv_rms_norm_rope_positions<float, false>(
          q, k, v, q_weight, k_weight, v_weight, cos, sin, positions, q_out,
          k_out, v_out, q_stride_b, q_stride_h, q_stride_s, q_stride_d,
          k_stride_b, k_stride_h, k_stride_s, k_stride_d, v_stride_b,
          v_stride_h, v_stride_s, v_stride_d, batch, q_heads, k_heads, seq_len,
          head_dim, rot_dim, q_eps, k_eps, v_eps, stream);
    }
  }
}
