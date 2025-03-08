#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    scalar_t *__restrict__ key_cache,   // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
    scalar_t *__restrict__ value_cache, // [num_blocks, num_heads, head_size,
                                        // block_size]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
        block_offset * x + x_offset;
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size + head_offset * block_size +
        block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

#define CALL_RESHAPE_AND_CACHE(T)                                              \
  vllm::reshape_and_cache_kernel<T><<<grid, block, 0, stream>>>(               \
      reinterpret_cast<T *>(key), reinterpret_cast<T *>(value),                \
      reinterpret_cast<T *>(key_cache), reinterpret_cast<T *>(value_cache),    \
      slot_mapping, key_stride, value_stride, num_heads, head_size,            \
      block_size, x);

} // namespace vllm

extern "C" void reshape_and_cache(
    void *key,         // [num_tokens, num_heads, head_size]
    void *value,       // [num_tokens, num_heads, head_size]
    void *key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t *slot_mapping, // [num_tokens]

    int32_t num_tokens, int32_t num_heads, int32_t head_size,
    int32_t block_size, int32_t x, int32_t key_stride, int32_t value_stride,
    cudaStream_t stream,

    uint32_t dtype // 0 => f16; 1 => bf16; 2 => f32
) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  if (dtype == 0) {
    CALL_RESHAPE_AND_CACHE(uint16_t);
  } else if (dtype == 1) {
    CALL_RESHAPE_AND_CACHE(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_RESHAPE_AND_CACHE(float);
  }
  CUDA_CHECK(cudaGetLastError());
}