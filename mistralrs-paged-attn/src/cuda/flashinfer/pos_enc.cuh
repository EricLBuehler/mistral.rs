/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include "layout.cuh"
#include "math.cuh"
#include "page.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

struct RopeQuantizeAppendPagedKVCacheParams {
  uint32_t nnz;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t rope_dim;
  uint32_t no_rope_dim;
  size_t q_rope_in_stride_n;
  size_t q_rope_in_stride_h;
  size_t q_nope_in_stride_n;
  size_t q_nope_in_stride_h;
  size_t q_rope_out_stride_n;
  size_t q_rope_out_stride_h;
  size_t q_nope_out_stride_n;
  size_t q_nope_out_stride_h;
  size_t k_rope_in_stride;
  size_t k_rope_in_stride_h;
  size_t k_nope_in_stride;
  size_t k_nope_in_stride_h;
  size_t v_in_stride;
  size_t v_in_stride_h;
  float quant_scale_q;
  float quant_scale_kv;
};

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kRoPELlama = 1U,
  // Apply ALiBi bias
  kALiBi = 2U
};

/*!
 * \brief Convert PosEncodingMode to string
 * \param pos_encoding_mode A PosEncodingMode value
 */
inline std::string PosEncodingModeToString(const PosEncodingMode& pos_encoding_mode) {
  switch (pos_encoding_mode) {
    case PosEncodingMode::kNone:
      return "None";
    case PosEncodingMode::kRoPELlama:
      return "Llama";
    case PosEncodingMode::kALiBi:
      return "ALiBi";
    default:
      return "Unknown";
  }
}

__device__ __forceinline__ float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] =
          vec[i] * cos +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i] +
          ((threadIdx.x * vec_size < rotary_dim / 2) ? -permuted_vec[i] : permuted_vec[i]) * sin[i];
    }
  }
  return vec;
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim] with interleave,
 *   return thread-local vector.
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_interleave(
    const T* x, const vec_t<float, vec_size>& freq, int32_t offset,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      __sincosf(embed, &sin, &cos);
      vec[i] = vec[i] * cos + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin;
    }
  }
  return vec;
}

template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin_interleave(
    const T* x, const vec_t<float, vec_size>& cos, const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i];
    }
  }
  return vec;
}

/*
HACK (ByronHsu): in the interleave mode with cos_sin_cache, we actually only use the first half of
cos and sin

For example,
In the below example, the vec_size is 4
the computation in the kernel is:
    [x1, x2, x3, x4...] * [cos1, cos1, cos2, cos2] + [-x2, x1, -x4, x3...] * [sin1, sin1, sin2,
sin2] the data we loaded are:
    - loaded vec = [x1, x2, x3, x4]
    - loaded cos = [cos1, cos2, cos3, cos4]
    - loaded sin = [sin1, sin2, sin3, sin4]
But only the first half of cos and sin is used in the computation.

However, we argue the additional overhead is acceptable:
    1. loading additional elements of cos and sin is not adding much overhead. The arithmetic
intensity is the same as non-interleave mode. Each elements of cos and sin is load twice
    2. we don't want two code paths of cos and sin vector for interleave and non-interleave mode.
*/
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size>
vec_apply_llama_rope_cos_sin_interleave_reuse_half(const T* x, const vec_t<float, vec_size>& cos,
                                                   const vec_t<float, vec_size>& sin,
                                                   const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      // i / 2 is to get the index of the first half of cos and sin
      vec[i] = vec[i] * cos[i / 2] +
               ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

/*
HACK (raayandhar): RoPE + quant kernels expect `no_rope_dim` to be a multiple of `rope_dim`.
That meant that every chunk covered a full vector load/store.

Now, since BatchQKApplyRotaryPosIdsCosSinCache is routed to the RopeQuantize kernel, the
"no rope" slice can be smaller than `rope_dim`.
e.g. consider head_dim = 192, rope_dim = 128, no_rope_dim = 64.

We want to write 64 values only, but the kernel will load and write 128, and cause
an error.

We guard these partial chunks by zero-padding unused lanes, and only writing
back the elements that actually exist in the chunk.
*/
template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void scale_store_partial_chunk(const DType* in_ptr, QuantType* out_ptr,
                                                          uint32_t lane_elem_offset,
                                                          uint32_t chunk_valid, float scale) {
  if (chunk_valid == 0 || lane_elem_offset >= chunk_valid) {
    return;
  }
  vec_t<float, vec_size> vec;
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_load(in_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp.cast_load(in_ptr + elem_idx);
        vec[i] = tmp[0];
      } else {
        vec[i] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    vec[i] = vec[i] * scale;
  }
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_store(out_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp[0] = vec[i];
        tmp.cast_store(out_ptr + elem_idx);
      }
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsCosSinCacheKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const int half_rotary_dim = rotary_dim / 2;

    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if (tx * vec_size < rotary_dim) {
      int sin_offset = rotary_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rotary_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rotary_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rotary_dim) + (sin_offset + vec_idx));
    }

    // not to unroll the loop, because num head might be large and might lead to worse performance
#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(q_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(k_ptr, cos, sin,
                                                                                  rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename IdType,
          typename QuantType>
__global__ void RopeQuantizeKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out,
    float* __restrict__ cos_sin_cache, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h, size_t q_nope_in_stride_n,
    size_t q_nope_in_stride_h, size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h, size_t k_rope_in_stride,
    size_t k_rope_in_stride_h, size_t k_nope_in_stride, size_t k_nope_in_stride_h,
    size_t k_rope_out_stride, size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {  // generalized kernel
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  // Calculate flexible boundaries for block allocation
  uint32_t rope_chunk_size = rope_dim;  // Process entire rope_dim per chunk
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rope_dim = rope_dim / 2;
    // Load cos/sin for RoPE processing blocks only
    // 1. if interleave:
    //  - cos = cos_sin_cache[pos_id][tx * vec_size // 2]
    //  - sin = cos_sin_cache[pos_id][(rot_dim // 2) + tx * vec_size // 2]
    // 2. if not interleave
    //  - cos = cos_cache[pos_id][(tx * vec_size) % (rot_dim // 2)]
    //  - sin = sin_cache[pos_id][(rot_dim // 2) + (tx * vec_size) % (rot_dim // 2)]
    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;  // Use half_rotary_dim
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // Q RoPE processing: num_qo_heads * rope_chunks blocks
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // K RoPE processing: num_kv_heads * rope_chunks blocks
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_rope_in_stride, k_rope_in_stride_h);
      QuantType* k_rope_out_ptr =
          k_rope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_rope_out_stride,
                                            k_rope_out_stride_h);

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }
      k_rope_vec.cast_store(k_rope_out_ptr + tx * vec_size);

    } else if (by < k_nope_end) {
      // K Non-RoPE processing: num_kv_heads * no_rope_chunks blocks
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;  // Use same chunk size

      DType* k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                              k_nope_in_stride, k_nope_in_stride_h);
      QuantType* k_nope_out_ptr =
          k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset, k_nope_out_stride,
                                            k_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      // Handle tail chunks where no_rope_dim is not a multiple of rope_dim.
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          k_nope_in_ptr, k_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_kv);

    } else {
      // Q Non-RoPE processing: num_qo_heads * no_rope_chunks blocks
      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;  // Use same chunk size

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim) ? min(rope_chunk_size, no_rope_dim - elem_offset) : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      // Handle tail chunks where no_rope_dim is not a multiple of rope_dim.
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          q_nope_in_ptr, q_nope_out_ptr, lane_elem_offset, chunk_valid, quant_scale_q);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsHeadParallelismKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

    if (by < num_qo_heads) {
      uint32_t qo_head_idx = by;
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    } else {
      uint32_t kv_head_idx = by - num_qo_heads;
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryPosIdsKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, size_t q_stride_n,
    size_t q_stride_h, size_t k_stride_n, size_t k_stride_h, size_t q_rope_stride_n,
    size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h, float smooth_a,
    float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  // NOTE: q and q_rope may be the same ptr, so do k and k_rope
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  vec_t<float, vec_size> cos, sin;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    if (tx * vec_size < rotary_dim) {
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        float embed = float(pos) * freq[i];
        __sincosf(embed, &sin[i], &cos[i]);
      }
    }

#pragma unroll 1
    for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      DType* q_ptr = q + get_elem_offset_impl(idx, qo_head_idx, 0, q_stride_n, q_stride_h);
      DType* q_rope_ptr =
          q_rope + get_elem_offset_impl(idx, qo_head_idx, 0, q_rope_stride_n, q_rope_stride_h);
      vec_t<float, vec_size> q_vec;
      if constexpr (interleave) {
        q_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      } else {
        q_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_ptr, cos, sin, rotary_dim);
      }
      q_vec.cast_store(q_rope_ptr + tx * vec_size);
    }

#pragma unroll 1
    for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
      DType* k_ptr = k + get_elem_offset_impl(idx, kv_head_idx, 0, k_stride_n, k_stride_h);
      DType* k_rope_ptr =
          k_rope + get_elem_offset_impl(idx, kv_head_idx, 0, k_rope_stride_n, k_rope_stride_h);
      vec_t<float, vec_size> k_vec;
      if constexpr (interleave) {
        k_vec = vec_apply_llama_rope_cos_sin_interleave<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      } else {
        k_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_ptr, cos, sin, rotary_dim);
      }
      k_vec.cast_store(k_rope_ptr + tx * vec_size);
    }
  }
}

template <bool interleave, uint32_t head_dim, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType>
__global__ void BatchQKApplyRotaryKernel(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    float smooth_a, float smooth_b, float rope_rcp_scale, float rope_rcp_theta) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t bdy = blockDim.y;
  vec_t<float, vec_size> freq;
  if (tx * vec_size < rotary_dim) {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      if constexpr (interleave) {
        freq[i] = __powf(rope_rcp_theta, float(2 * ((tx * vec_size + i) / 2)) / float(rotary_dim));
      } else {
        freq[i] = __powf(rope_rcp_theta,
                         float(2 * ((tx * vec_size + i) % (rotary_dim / 2))) / float(rotary_dim));
      }

      float smooth = freq[i] * smooth_a + smooth_b;
      smooth = max(0.0f, min(1.0f, smooth));  // clamp to [0, 1]
      freq[i] = (1 - smooth) * (freq[i] * rope_rcp_scale) + smooth * freq[i];
    }
  }

  if (bx < batch_size * num_qo_heads) {
    // apply rotary to q
    const uint32_t batch_idx = bx / num_qo_heads;
    const uint32_t qo_head_idx = bx % num_qo_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> q_vec;
      if (i * bdy + ty < seq_len) {
        DType* q_ptr = q + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                                q_stride_n, q_stride_h);
        DType* q_rope_ptr =
            q_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, qo_head_idx, 0,
                                          q_rope_stride_n, q_rope_stride_h);
        if constexpr (interleave) {
          q_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          q_vec =
              vec_apply_llama_rope<vec_size, bdx>(q_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        q_vec.cast_store(q_rope_ptr + tx * vec_size);
      }
    }
  } else {
    // apply rotary to k
    uint32_t batch_idx = (bx - batch_size * num_qo_heads) / num_kv_heads;
    uint32_t kv_head_idx = (bx - batch_size * num_qo_heads) % num_kv_heads;
    const uint32_t seq_len = indptr[batch_idx + 1] - indptr[batch_idx];
    const uint32_t offset = offsets[batch_idx];
#pragma unroll 2
    for (uint32_t i = 0; i < (seq_len + bdy - 1) / bdy; ++i) {
      vec_t<float, vec_size> k_vec;
      if (i * bdy + ty < seq_len) {
        DType* k_ptr = k + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                                k_stride_n, k_stride_h);
        DType* k_rope_ptr =
            k_rope + get_elem_offset_impl(indptr[batch_idx] + i * bdy + ty, kv_head_idx, 0,
                                          k_rope_stride_n, k_rope_stride_h);
        if constexpr (interleave) {
          k_vec = vec_apply_llama_rope_interleave<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty,
                                                                 rotary_dim);
        } else {
          k_vec =
              vec_apply_llama_rope<vec_size, bdx>(k_ptr, freq, offset + i * bdy + ty, rotary_dim);
        }
        k_vec.cast_store(k_rope_ptr + tx * vec_size);
      }
    }
  }
}

/*!
 * \brief Unified CUDA kernel to apply RoPE, quantize to FP8, and append to paged cache.
 *
 * Templated on CacheT to support both GQA/MHA (paged_kv_t) and MLA (paged_kv_mla_t).
 * Cache-only behaviors are selected with constexpr on the CacheT.
 */
template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType, typename RoPEIdType,
          typename PagedKVIdType, typename QuantType, typename CacheT>
__global__ void RopeQuantizeAppendPagedKVCacheKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, DType* v_in,
    QuantType* q_rope_out, QuantType* q_nope_out, CacheT paged_kv_like,
    PagedKVIdType* __restrict__ batch_indices, PagedKVIdType* __restrict__ positions,
    float* __restrict__ cos_sin_cache, RoPEIdType* __restrict__ pos_ids,
    const RopeQuantizeAppendPagedKVCacheParams params) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  // Local aliases for params for readability
  const uint32_t nnz = params.nnz;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t rope_dim = params.rope_dim;
  const uint32_t no_rope_dim = params.no_rope_dim;
  const size_t q_rope_in_stride_n = params.q_rope_in_stride_n;
  const size_t q_rope_in_stride_h = params.q_rope_in_stride_h;
  const size_t q_nope_in_stride_n = params.q_nope_in_stride_n;
  const size_t q_nope_in_stride_h = params.q_nope_in_stride_h;
  const size_t q_rope_out_stride_n = params.q_rope_out_stride_n;
  const size_t q_rope_out_stride_h = params.q_rope_out_stride_h;
  const size_t q_nope_out_stride_n = params.q_nope_out_stride_n;
  const size_t q_nope_out_stride_h = params.q_nope_out_stride_h;
  const size_t k_rope_in_stride = params.k_rope_in_stride;
  const size_t k_rope_in_stride_h = params.k_rope_in_stride_h;
  const size_t k_nope_in_stride = params.k_nope_in_stride;
  const size_t k_nope_in_stride_h = params.k_nope_in_stride_h;
  const size_t v_in_stride = params.v_in_stride;
  const size_t v_in_stride_h = params.v_in_stride_h;
  const float quant_scale_q = params.quant_scale_q;
  const float quant_scale_kv = params.quant_scale_kv;

  // Calculate flexible boundaries for block allocation
  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  // For MLA, num_kv_heads is effectively 1
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  // Deduce MLA vs GQA/MHA from CacheT
  constexpr bool IS_MLA = std::is_same<CacheT, paged_kv_mla_t<QuantType, PagedKVIdType>>::value;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const RoPEIdType pos = pos_ids[idx];

    // Compute page location for this token
    uint32_t page_iter, entry_idx;
    paged_kv_like.page_size.divmod(
        paged_kv_like.indptr[batch_indices[idx]] * paged_kv_like.page_size + positions[idx],
        page_iter, entry_idx);

    const int half_rope_dim = rope_dim / 2;
    // Load cos/sin for RoPE processing blocks only
    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;  // Force integer division
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // ============ Q RoPE processing ============
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // ============ K RoPE processing & Cache Append ============
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr;
      if constexpr (IS_MLA) {
        // MLA: 2D K
        k_rope_in_ptr = k_rope_in + idx * k_rope_in_stride + elem_offset;
      } else {
        // GQA/MHA: 3D K
        k_rope_in_ptr = k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                         k_rope_in_stride, k_rope_in_stride_h);
      }

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec = vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }

      if constexpr (IS_MLA) {
        QuantType* kpe_ptr =
            paged_kv_like.get_kpe_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
        k_rope_vec.cast_store(kpe_ptr);
      } else {
        QuantType* k_ptr = paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx, tx * vec_size);
        k_rope_vec.cast_store(k_ptr);
      }

    } else if (by < k_nope_end) {
      // ============ K Non-RoPE processing & Cache Append ============
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* k_nope_in_ptr;
      if constexpr (IS_MLA) {
        k_nope_in_ptr = k_nope_in + idx * k_nope_in_stride + elem_offset;
      } else {
        k_nope_in_ptr = k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                                         k_nope_in_stride, k_nope_in_stride_h);
      }

      vec_t<float, vec_size> k_nope_vec;
      k_nope_vec.cast_load(k_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_nope_vec[i] = k_nope_vec[i] * quant_scale_kv;
      }

      if constexpr (IS_MLA) {
        QuantType* ckv_ptr =
            paged_kv_like.get_ckv_ptr(page_iter, entry_idx, elem_offset + tx * vec_size);
        k_nope_vec.cast_store(ckv_ptr);
      } else {
        QuantType* k_ptr = paged_kv_like.get_k_ptr(page_iter, k_head_idx, entry_idx,
                                                   rope_dim + elem_offset + tx * vec_size);
        k_nope_vec.cast_store(k_ptr);
      }

    } else if (by < k_nope_end + (IS_MLA ? 0u : num_kv_heads)) {
      // ============ V processing & Cache Append (GQA/MHA only) ============
      if constexpr (!IS_MLA) {
        uint32_t kv_head_idx = by - k_nope_end;
        DType* v_in_ptr =
            v_in + get_elem_offset_impl(idx, kv_head_idx, 0, v_in_stride, v_in_stride_h);
        // Cover the full head dimension (rope_dim + no_rope_dim) in chunks of rope_chunk_size
        uint32_t head_dim_total = rope_dim + no_rope_dim;
        uint32_t v_chunks = (head_dim_total + rope_chunk_size - 1) / rope_chunk_size;
#pragma unroll 1
        for (uint32_t j = 0; j < v_chunks; ++j) {
          uint32_t v_elem_offset = j * rope_chunk_size;
          if (v_elem_offset + tx * vec_size < head_dim_total) {
            vec_t<float, vec_size> v_vec;
            v_vec.cast_load(v_in_ptr + v_elem_offset + tx * vec_size);
#pragma unroll
            for (uint32_t i = 0; i < vec_size; ++i) {
              v_vec[i] = v_vec[i] * quant_scale_kv;
            }
            QuantType* v_ptr = paged_kv_like.get_v_ptr(page_iter, kv_head_idx, entry_idx,
                                                       v_elem_offset + tx * vec_size);
            v_vec.cast_store(v_ptr);
          }
        }
      }

    } else {
      // ============ Q Non-RoPE processing ============
      // MLA has no V section, so Q-nope starts immediately after K-nope.
      // GQA/MHA has a V section of length num_kv_heads blocks.
      uint32_t q_nope_start = k_nope_end + (IS_MLA ? 0u : num_kv_heads);
      uint32_t q_head_idx = (by - q_nope_start) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - q_nope_start) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset, q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      vec_t<float, vec_size> q_nope_vec;
      q_nope_vec.cast_load(q_nope_in_ptr + tx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_nope_vec[i] = q_nope_vec[i] * quant_scale_q;
      }
      q_nope_vec.cast_store(q_nope_out_ptr + tx * vec_size);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DType, typename IdType, typename QuantType>
cudaError_t RopeQuantize(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* k_rope_out, QuantType* q_nope_out, QuantType* k_nope_out, float* cos_sin_cache,
    IdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride, size_t k_rope_out_stride_h,
    size_t k_nope_out_stride, size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  // Use nested macros for runtime->compile-time dispatch for required constexpr values
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunk_size = rope_dim;
    uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t no_rope_chunks = (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;
    void* args[] = {(void*)&q_rope_in,
                    (void*)&k_rope_in,
                    (void*)&q_nope_in,
                    (void*)&k_nope_in,
                    (void*)&q_rope_out,
                    (void*)&k_rope_out,
                    (void*)&q_nope_out,
                    (void*)&k_nope_out,
                    (void*)&cos_sin_cache,
                    (void*)&pos_ids,
                    (void*)&nnz,
                    (void*)&num_qo_heads,
                    (void*)&num_kv_heads,
                    (void*)&rope_dim,
                    (void*)&no_rope_dim,
                    (void*)&q_rope_in_stride_n,
                    (void*)&q_rope_in_stride_h,
                    (void*)&q_nope_in_stride_n,
                    (void*)&q_nope_in_stride_h,
                    (void*)&q_rope_out_stride_n,
                    (void*)&q_rope_out_stride_h,
                    (void*)&q_nope_out_stride_n,
                    (void*)&q_nope_out_stride_h,
                    (void*)&k_rope_in_stride,
                    (void*)&k_rope_in_stride_h,
                    (void*)&k_nope_in_stride,
                    (void*)&k_nope_in_stride_h,
                    (void*)&k_rope_out_stride,
                    (void*)&k_rope_out_stride_h,
                    (void*)&k_nope_out_stride,
                    (void*)&k_nope_out_stride_h,
                    (void*)&quant_scale_q,
                    (void*)&quant_scale_kv};
    auto kernel = RopeQuantizeKernel<INTERLEAVE, vec_size, 1, DType, IdType, QuantType>;
    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out, k_rope_out,
        q_nope_out, k_nope_out, cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim,
        no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n, q_nope_in_stride_h,
        q_rope_out_stride_n, q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
        k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
        k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride, k_nope_out_stride_h,
        quant_scale_q, quant_scale_kv));
  });

  return cudaSuccess;
}

/*!
 * \brief Host function to apply RoPE, quantize to FP8, and append K/V to paged cache (GQA/MHA)
 */
template <typename DType, typename RoPEIdType, typename PagedKVIdType, typename QuantType>
cudaError_t RopeQuantizeAppendPagedKVCache(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, DType* v_in,
    QuantType* q_rope_out, QuantType* q_nope_out, paged_kv_t<QuantType, PagedKVIdType> paged_kv,
    PagedKVIdType* batch_indices, PagedKVIdType* positions, float* cos_sin_cache,
    RoPEIdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rope_dim, uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t v_in_stride, size_t v_in_stride_h, float quant_scale_q,
    float quant_scale_kv, bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunks = 1;
    uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;

    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_kv_heads +
                              num_qo_heads * no_rope_chunks;

    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = RopeQuantizeAppendPagedKVCacheKernel<INTERLEAVE, vec_size, /*bdx=*/1, DType,
                                                       RoPEIdType, PagedKVIdType, QuantType,
                                                       paged_kv_t<QuantType, PagedKVIdType>>;
    RopeQuantizeAppendPagedKVCacheParams params;
    params.nnz = nnz;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = num_kv_heads;
    params.rope_dim = rope_dim;
    params.no_rope_dim = no_rope_dim;
    params.q_rope_in_stride_n = q_rope_in_stride_n;
    params.q_rope_in_stride_h = q_rope_in_stride_h;
    params.q_nope_in_stride_n = q_nope_in_stride_n;
    params.q_nope_in_stride_h = q_nope_in_stride_h;
    params.q_rope_out_stride_n = q_rope_out_stride_n;
    params.q_rope_out_stride_h = q_rope_out_stride_h;
    params.q_nope_out_stride_n = q_nope_out_stride_n;
    params.q_nope_out_stride_h = q_nope_out_stride_h;
    params.k_rope_in_stride = k_rope_in_stride;
    params.k_rope_in_stride_h = k_rope_in_stride_h;
    params.k_nope_in_stride = k_nope_in_stride;
    params.k_nope_in_stride_h = k_nope_in_stride_h;
    params.v_in_stride = v_in_stride;
    params.v_in_stride_h = v_in_stride_h;
    params.quant_scale_q = quant_scale_q;
    params.quant_scale_kv = quant_scale_kv;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in, v_in, q_rope_out, q_nope_out,
        paged_kv, batch_indices, positions, cos_sin_cache, pos_ids, params));
  });

  return cudaSuccess;
}

/*!
 * \brief Host function to apply RoPE, quantize to FP8, and append to MLA paged cache
 */
template <typename DType, typename RoPEIdType, typename PagedKVIdType, typename QuantType>
cudaError_t RopeQuantizeAppendPagedMLACache(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in, QuantType* q_rope_out,
    QuantType* q_nope_out, paged_kv_mla_t<QuantType, PagedKVIdType> paged_kv_mla,
    PagedKVIdType* batch_indices, PagedKVIdType* positions, float* cos_sin_cache,
    RoPEIdType* pos_ids, uint32_t nnz, uint32_t num_qo_heads, uint32_t rope_dim,
    uint32_t no_rope_dim, size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_nope_in_stride, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl = false, cudaStream_t stream = nullptr) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunks = 1;
    uint32_t no_rope_chunks = (no_rope_dim + rope_dim - 1) / rope_dim;
    constexpr uint32_t num_kv_heads = 1;
    uint32_t total_blocks_y = num_qo_heads * rope_chunks + num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks + num_qo_heads * no_rope_chunks;

    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    auto kernel = RopeQuantizeAppendPagedKVCacheKernel<INTERLEAVE, vec_size, /*bdx=*/1, DType,
                                                       RoPEIdType, PagedKVIdType, QuantType,
                                                       paged_kv_mla_t<QuantType, PagedKVIdType>>;
    DType* v_in_nullptr = nullptr;
    uint32_t num_kv_heads_1 = 1;
    size_t k_rope_in_stride_h_dup = k_rope_in_stride;
    size_t k_nope_in_stride_h_dup = k_nope_in_stride;
    RopeQuantizeAppendPagedKVCacheParams params;
    params.nnz = nnz;
    params.num_qo_heads = num_qo_heads;
    params.num_kv_heads = 1u;
    params.rope_dim = rope_dim;
    params.no_rope_dim = no_rope_dim;
    params.q_rope_in_stride_n = q_rope_in_stride_n;
    params.q_rope_in_stride_h = q_rope_in_stride_h;
    params.q_nope_in_stride_n = q_nope_in_stride_n;
    params.q_nope_in_stride_h = q_nope_in_stride_h;
    params.q_rope_out_stride_n = q_rope_out_stride_n;
    params.q_rope_out_stride_h = q_rope_out_stride_h;
    params.q_nope_out_stride_n = q_nope_out_stride_n;
    params.q_nope_out_stride_h = q_nope_out_stride_h;
    params.k_rope_in_stride = k_rope_in_stride;
    params.k_rope_in_stride_h = k_rope_in_stride_h_dup;
    params.k_nope_in_stride = k_nope_in_stride;
    params.k_nope_in_stride_h = k_nope_in_stride_h_dup;
    params.v_in_stride = 0;
    params.v_in_stride_h = 0;
    params.quant_scale_q = quant_scale_q;
    params.quant_scale_kv = quant_scale_kv;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel,
                                            // inputs
                                            q_rope_in, k_rope_in, q_nope_in, k_nope_in,
                                            v_in_nullptr,
                                            // q outputs
                                            q_rope_out, q_nope_out,
                                            // cache + indices
                                            paged_kv_mla, batch_indices, positions,
                                            // rope tables
                                            cos_sin_cache, pos_ids,
                                            // params
                                            params));
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIdsCosSinCache(
    DType* q, DType* k, DType* q_rope, DType* k_rope, float* cos_sin_cache, IdType* pos_ids,
    uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim,
    uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, cudaStream_t stream = nullptr) {
  if (head_dim < rotary_dim) {
    std::ostringstream err_msg;
    err_msg << "head_dim (" << head_dim << ") must be >= rotary_dim (" << rotary_dim << ")";
    FLASHINFER_ERROR(err_msg.str());
  }

  // We have better performance with this kernel with these head_dim instead of RopeQuantize
  if (head_dim == 64 || head_dim == 128 || head_dim == 256 || head_dim == 512) {
    int dev_id = 0;
    int num_sms = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        // operate on 16 Bytes at a time
        constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
        // how many threads needed per head_dim
        constexpr uint32_t bdx = HEAD_DIM / vec_size;
        // how many threads needed per block
        uint32_t num_threads = std::max(128U, bdx);
        // how many tokens can we process in a block
        uint32_t bdy = num_threads / bdx;
        // how many blocks needed to process all tokens
        uint32_t nblks_x = (nnz + bdy - 1) / bdy;
        void* args[] = {(void*)&q,
                        (void*)&k,
                        (void*)&q_rope,
                        (void*)&k_rope,
                        (void*)&cos_sin_cache,
                        (void*)&pos_ids,
                        (void*)&nnz,
                        (void*)&num_qo_heads,
                        (void*)&num_kv_heads,
                        (void*)&rotary_dim,
                        (void*)&q_stride_n,
                        (void*)&q_stride_h,
                        (void*)&k_stride_n,
                        (void*)&k_stride_h,
                        (void*)&q_rope_stride_n,
                        (void*)&q_rope_stride_h,
                        (void*)&k_rope_stride_n,
                        (void*)&k_rope_stride_h};
        auto kernel_0 = BatchQKApplyRotaryPosIdsCosSinCacheKernel<INTERLEAVE, HEAD_DIM, vec_size,
                                                                  bdx, DType, IdType>;

        int num_blocks_per_sm_0 = 0;
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
        uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;

        if ((nnz + bdy - 1) / bdy >= num_ctas_0) {
          dim3 nblks(nblks_x);
          dim3 nthrs(bdx, bdy);
          FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
        } else {
          dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
          dim3 nthrs(bdx, bdy);
          auto kernel_1 = BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel<
              INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
          FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
        }
      });
    });
    return cudaSuccess;
  }

  const uint32_t rope_dim = rotary_dim;
  const uint32_t no_rope_dim = head_dim - rotary_dim;

  // Route to RopeQuantize kernel
  DType* q_rope_in = q;
  DType* k_rope_in = k;
  DType* q_nope_in = q + rotary_dim;
  DType* k_nope_in = k + rotary_dim;
  DType* q_rope_out = q_rope;
  DType* k_rope_out = k_rope;
  DType* q_nope_out = q_rope + rotary_dim;
  DType* k_nope_out = k_rope + rotary_dim;

  return RopeQuantize<DType, IdType, DType>(
      q_rope_in, k_rope_in, q_nope_in, k_nope_in, q_rope_out, k_rope_out, q_nope_out, k_nope_out,
      cos_sin_cache, pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim, q_stride_n,
      q_stride_h, q_stride_n, q_stride_h, q_rope_stride_n, q_rope_stride_h, q_rope_stride_n,
      q_rope_stride_h, k_stride_n, k_stride_h, k_stride_n, k_stride_h, k_rope_stride_n,
      k_rope_stride_h, k_rope_stride_n, k_rope_stride_h, /*quant_scale_q=*/1.0f,
      /*quant_scale_kv=*/1.0f, interleave, /*enable_pdl=*/false, stream);
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      uint32_t nblks_x = (nnz + bdy - 1) / bdy;

      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      auto kernel_0 =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;

      int num_blocks_per_sm_0 = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm_0, kernel_0, num_threads, /*smem_size=*/0));
      uint32_t num_ctas_0 = num_blocks_per_sm_0 * num_sms;
      if (nblks_x >= num_ctas_0) {
        dim3 nblks(nblks_x);
        dim3 nthrs(bdx, bdy);

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_0, nblks, nthrs, args, 0, stream));
      } else {
        dim3 nblks(nblks_x, num_qo_heads + num_kv_heads);
        dim3 nthrs(bdx, bdy);
        auto kernel_1 = BatchQKApplyRotaryPosIdsHeadParallelismKernel<INTERLEAVE, HEAD_DIM,
                                                                      vec_size, bdx, DType, IdType>;

        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel_1, nblks, nthrs, args, 0, stream));
      }
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotary(DType* q, DType* k, DType* q_rope, DType* k_rope,
                               IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n,
                               size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
                               size_t q_rope_stride_n, size_t q_rope_stride_h,
                               size_t k_rope_stride_n, size_t k_rope_stride_h, bool interleave,
                               float rope_scale, float rope_theta, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = 0.f;
  float smooth_b = 0.f;

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyRotaryInPlace(DType* __restrict__ q, DType* __restrict__ k,
                                      IdType* __restrict__ indptr, IdType* __restrict__ offsets,
                                      uint32_t batch_size, uint32_t num_qo_heads,
                                      uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
                                      size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
                                      size_t k_stride_h, bool interleave, float rope_scale,
                                      float rope_theta, cudaStream_t stream = nullptr) {
  return BatchQKApplyRotary<DType, IdType>(
      q, k, q, k, indptr, offsets, batch_size, num_qo_heads, num_kv_heads, rotary_dim, head_dim,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, q_stride_n, q_stride_h, k_stride_n,
      k_stride_h, interleave, rope_scale, rope_theta, stream);
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31Rotary(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* __restrict__ indptr,
    IdType* __restrict__ offsets, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rotary_dim, uint32_t head_dim, size_t q_stride_n, size_t q_stride_h, size_t k_stride_n,
    size_t k_stride_h, size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n,
    size_t k_rope_stride_h, bool interleave, float rope_scale, float rope_theta,
    float low_freq_factor, float high_freq_factor, float old_context_length,
    cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks(batch_size * (num_qo_heads + num_kv_heads));
      dim3 nthrs(bdx, bdy);
      auto kernel = BatchQKApplyRotaryKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&indptr,
                      (void*)&offsets,
                      (void*)&batch_size,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

template <typename DType, typename IdType>
cudaError_t BatchQKApplyLlama31RotaryPosIds(
    DType* q, DType* k, DType* q_rope, DType* k_rope, IdType* pos_ids, uint32_t nnz,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t rotary_dim, uint32_t head_dim,
    size_t q_stride_n, size_t q_stride_h, size_t k_stride_n, size_t k_stride_h,
    size_t q_rope_stride_n, size_t q_rope_stride_h, size_t k_rope_stride_n, size_t k_rope_stride_h,
    bool interleave, float rope_scale, float rope_theta, float low_freq_factor,
    float high_freq_factor, float old_context_length, cudaStream_t stream = nullptr) {
  float rope_rcp_scale = 1.0f / rope_scale;
  float rope_rcp_theta = 1.0f / rope_theta;
  float smooth_a = old_context_length / (2 * M_PI * high_freq_factor - 2 * M_PI * low_freq_factor);
  float smooth_b = -1.0f / (high_freq_factor / low_freq_factor - 1.0f);

  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      uint32_t num_threads = std::max(128U, bdx);
      uint32_t bdy = num_threads / bdx;
      dim3 nblks((nnz + bdy - 1) / bdy);
      dim3 nthrs(bdx, bdy);
      auto kernel =
          BatchQKApplyRotaryPosIdsKernel<INTERLEAVE, HEAD_DIM, vec_size, bdx, DType, IdType>;
      void* args[] = {(void*)&q,
                      (void*)&k,
                      (void*)&q_rope,
                      (void*)&k_rope,
                      (void*)&pos_ids,
                      (void*)&nnz,
                      (void*)&num_qo_heads,
                      (void*)&num_kv_heads,
                      (void*)&rotary_dim,
                      (void*)&q_stride_n,
                      (void*)&q_stride_h,
                      (void*)&k_stride_n,
                      (void*)&k_stride_h,
                      (void*)&q_rope_stride_n,
                      (void*)&q_rope_stride_h,
                      (void*)&k_rope_stride_n,
                      (void*)&k_rope_stride_h,
                      (void*)&smooth_a,
                      (void*)&smooth_b,
                      (void*)&rope_rcp_scale,
                      (void*)&rope_rcp_theta};
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
    });
  });

  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
