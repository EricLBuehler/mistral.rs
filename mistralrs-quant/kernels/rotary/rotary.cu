#ifndef NO_BF16_KERNEL
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
    scalar_t *__restrict__ arr, const scalar_t *__restrict__ cos_ptr,
    const scalar_t *__restrict__ sin_ptr, int rot_offset, int rot_dim) {
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = rot_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template <typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
    scalar_t *__restrict__ query, // [num_tokens, num_heads, head_size]
    scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ cos_cache, // [num_tokens, rot_dim]
    const scalar_t *__restrict__ sin_cache, // [num_tokens, rot_dim]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;

  const scalar_t *cos_ptr = cos_cache + token_idx * rot_dim;
  const scalar_t *sin_ptr = sin_cache + token_idx * rot_dim;

  const int nq = num_heads * rot_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / rot_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % rot_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, rot_dim);
  }

  const int nk = num_kv_heads * rot_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / rot_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % rot_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, rot_dim);
  }
}

} // namespace vllm

#define CALL_ROTARY(T, IS_NEOX)                                                \
  vllm::rotary_embedding_kernel<T, IS_NEOX><<<grid, block, 0, stream>>>(       \
      reinterpret_cast<T *>(query), reinterpret_cast<T *>(key),                \
      reinterpret_cast<T *>(cos_cache), reinterpret_cast<T *>(sin_cache),      \
      rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);

extern "C" void
rotary_embedding(void *query,     // [num_tokens, num_heads, head_size]
                 void *key,       // [num_tokens, num_kv_heads, head_size]
                 void *cos_cache, // [num_tokens, rot_dim]
                 void *sin_cache, // [num_tokens, rot_dim]
                 int32_t is_neox,

                 int32_t head_size, int64_t num_tokens, int32_t rot_dim,
                 int32_t num_heads, int32_t num_kv_heads, int64_t query_stride,
                 int64_t key_stride,

                 uint32_t dtype // 0 => f16; 1 => bf16; 2 => f32
) {

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim, 512));
  const cudaStream_t stream = 0;
  const bool is_neox_bool = is_neox;

  if (is_neox_bool) {
    if (dtype == 0) {
      CALL_ROTARY(half, true);
    } else if (dtype == 1) {
#ifndef NO_BF16_KERNEL
      CALL_ROTARY(__nv_bfloat16, true);
#else
      fprintf(stderr, "ERROR: rotary_embedding BF16 requires BF16 support (SM 8.0+)\n");
#endif
    } else if (dtype == 2) {
      CALL_ROTARY(float, true);
    }
  } else {
    if (dtype == 0) {
      CALL_ROTARY(half, false);
    } else if (dtype == 1) {
#ifndef NO_BF16_KERNEL
      CALL_ROTARY(__nv_bfloat16, false);
#else
      fprintf(stderr, "ERROR: rotary_embedding BF16 requires BF16 support (SM 8.0+)\n");
#endif
    } else if (dtype == 2) {
      CALL_ROTARY(float, false);
    }
  }
}