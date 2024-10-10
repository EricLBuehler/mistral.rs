#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "cuda_compat.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include "quantization/fp8/quant_utils.cuh"

namespace vllm {

template<typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  cache_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const float *key_scale,
  const float *value_scale,
  const int x) {
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

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *key_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *value_scale);
    }
  }
}

//  CACHE_T is the stored data type of kv-cache.
// KV_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE><<<grid, block, 0, stream>>>(      \
    reinterpret_cast<KV_T*>(key),                                        \
    reinterpret_cast<KV_T*>(value),                                      \
    reinterpret_cast<CACHE_T*>(key_cache),                                  \
    reinterpret_cast<CACHE_T*>(value_cache),                                \
    slot_mapping,                                                     \
    key_stride,                                                       \
    value_stride,                                                     \
    num_heads,                                                        \
    head_size,                                                        \
    block_size,                                                       \
    key_scale,                                                        \
    value_scale,                                                      \
    x);


} // namespace vllm

extern "C" void reshape_and_cache(
  void *key,              // [num_tokens, num_heads, head_size]
  void *value,            // [num_tokens, num_heads, head_size]
  void *key_cache,        // [num_blocks, num_heads, head_size/x, block_size, x]
  void *value_cache,      // [num_blocks, num_heads, head_size, block_size]
  int64_t* slot_mapping,  // [num_tokens]

  int32_t num_tokens,
  int32_t num_heads,
  int32_t head_size,
  int32_t block_size,
  int32_t x,
  int32_t key_stride,
  int32_t value_stride,
  
  const float *key_scale,
  const float *value_scale,

  uint32_t dtype,      // 0 => f16; 1 => bf16; 2 => f32
  uint32_t kv_dtype      // 0 => same as dtype; 1 => fp8e4m3
  )
{
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = 0;

  DISPATCH_BY_KV_CACHE_DTYPE(dtype, kv_dtype, CALL_RESHAPE_AND_CACHE)
}