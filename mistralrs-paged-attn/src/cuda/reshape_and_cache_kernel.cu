#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

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

// atomicMax for non-negative floats (abs values), used for per-cell scale maxes.
__device__ __forceinline__ float atomicMaxFloatShared(float *address, float val) {
  int *addr_i = reinterpret_cast<int *>(address);
  int old_i = *addr_i;
  float old_f = __int_as_float(old_i);
  while (old_f < val) {
    int assumed = old_i;
    int new_i = __float_as_int(val);
    int prev = atomicCAS(addr_i, assumed, new_i);
    if (prev == assumed) {
      return __int_as_float(prev);
    }
    old_i = prev;
    old_f = __int_as_float(old_i);
  }
  return old_f;
}

// f16 keys arrive as raw uint16_t bit patterns; convert like the FP8 path does.
template <typename scalar_t>
__device__ __forceinline__ float to_float_val(scalar_t v) {
  return (float)v;
}
template <>
__device__ __forceinline__ float to_float_val<uint16_t>(uint16_t v) {
  return __half2float(*reinterpret_cast<__half *>(&v));
}
template <>
__device__ __forceinline__ float to_float_val<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    cache_t *__restrict__ key_cache,    // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
    cache_t *__restrict__ value_cache,  // [num_blocks, num_heads, head_size,
                                        // block_size]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float *k_scale, const float *v_scale) {
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
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      key_cache[tgt_key_idx] =
          vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key,
                                                              *k_scale);
      value_cache[tgt_value_idx] =
          vllm::fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value,
                                                              *v_scale);
    }
  }
}

#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                        \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                      \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<KV_T *>(key), reinterpret_cast<KV_T *>(value),      \
          reinterpret_cast<CACHE_T *>(key_cache),                              \
          reinterpret_cast<CACHE_T *>(value_cache), slot_mapping, key_stride,  \
          value_stride, num_heads, head_size, block_size, x,                   \
          reinterpret_cast<const float *>(k_scale),                            \
          reinterpret_cast<const float *>(v_scale));

// F4 (4-bit, per-token scales) KV cache write.
//
// Layouts (see cache_engine calculate_*_block_shape):
//   key_cache   [num_blocks, num_heads, head_size/32, block_size, 32] u8,
//               2 values/byte packed along the x (32-dim cell) axis
//   value_cache [num_blocks, num_heads, head_size/2, block_size] u8,
//               2 values/byte packed along the head axis
//   k_scale     [num_blocks, num_heads, head_size/32, block_size] f32,
//               one scale per 32-value cell (one token's x-row)
//   v_scale     [num_blocks, num_heads, block_size] f32,
//               one scale per token per head (256 values)
// Values are symmetric 4-bit: q = round(v/scale) clamped to [-8, 7], stored
// biased by 8 (nibble 0..15); dequant is (nibble - 8) * scale.
//
// One block per token: 128 threads, each handling 2 K pairs and 2 V pairs
// (a pair = (even, odd) head offsets, which always land in the same cell and
// share one byte), so byte writes are race-free.
template <typename scalar_t>
__global__ void reshape_and_cache_f4_kernel(
    const scalar_t *__restrict__ key,   // [num_tokens, num_heads, head_size]
    const scalar_t *__restrict__ value, // [num_tokens, num_heads, head_size]
    uint8_t *__restrict__ key_cache,    // [num_blocks, num_heads, hd/32, bs, 32]
    uint8_t *__restrict__ value_cache,  // [num_blocks, num_heads, hd/2, bs]
    float *__restrict__ k_scale,        // [num_blocks, num_heads, hd/32, bs]
    float *__restrict__ v_scale,        // [num_blocks, num_heads, bs]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size) {
  constexpr int CELL = 32;            // values per K cell
  constexpr int K_BYTES_PER_CELL = 16; // 32 values packed into 16 bytes
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t token_offset = slot_idx % block_size;

  const int tid = threadIdx.x;
  constexpr int THREADS = 128;
  const int k_pairs_per_head = head_size / 2; // pairs of head offsets
  const int num_k_pairs = num_heads * k_pairs_per_head; // 2 * 128 = 256
  const int num_v_pairs = num_heads * k_pairs_per_head; // 256

  // Pair index this thread owns: 2 pairs per thread.
  const int k_pair0 = tid * 2;
  const int k_pair1 = tid * 2 + 1;
  const int v_pair0 = tid * 2;
  const int v_pair1 = tid * 2 + 1;
  const bool has_k_pair0 = k_pair0 < num_k_pairs;
  const bool has_k_pair1 = k_pair1 < num_k_pairs;
  const bool has_v_pair0 = v_pair0 < num_v_pairs;
  const bool has_v_pair1 = v_pair1 < num_v_pairs;

  // Per-cell maxes: K cells (head, xrow) and V cells (head), dynamic shared.
  extern __shared__ float smem[];
  float *k_cell_max = smem;
  float *v_cell_max = smem + num_heads * (head_size / CELL);
  for (int i = tid; i < num_heads * (head_size / CELL); i += THREADS) {
    k_cell_max[i] = 0.0f;
  }
  for (int i = tid; i < num_heads; i += THREADS) {
    v_cell_max[i] = 0.0f;
  }
  __syncthreads();

  const int64_t key_base = token_idx * key_stride;
  const int64_t value_base = token_idx * value_stride;

  // Phase 1: per-cell abs-max (atomic max into shared memory).
  auto cell_max_k = [&](int head, int ho) {
    const int cell = head * (head_size / CELL) + ho / CELL;
    float av = fabsf(to_float_val(key[key_base + head * head_size + ho]));
    atomicMaxFloatShared(&k_cell_max[cell], av);
  };
  auto cell_max_v = [&](int head, int ho) {
    float av = fabsf(to_float_val(value[value_base + head * head_size + ho]));
    atomicMaxFloatShared(&v_cell_max[head], av);
  };
  auto handle_pair = [&](int head, int ho, auto fn) {
    fn(head, ho);
    fn(head, ho + 1);
  };
  if (has_k_pair0) {
    int head = (k_pair0 * 2) / head_size;
    int ho = (k_pair0 * 2) % head_size;
    handle_pair(head, ho, cell_max_k);
  }
  if (has_k_pair1) {
    int head = (k_pair1 * 2) / head_size;
    int ho = (k_pair1 * 2) % head_size;
    handle_pair(head, ho, cell_max_k);
  }
  if (has_v_pair0) {
    int head = v_pair0 / k_pairs_per_head;
    int ho = (v_pair0 % k_pairs_per_head) * 2;
    handle_pair(head, ho, cell_max_v);
  }
  if (has_v_pair1) {
    int head = v_pair1 / k_pairs_per_head;
    int ho = (v_pair1 % k_pairs_per_head) * 2;
    handle_pair(head, ho, cell_max_v);
  }
  __syncthreads();

  // Phase 2: quantize and pack.
  auto quant = [](float v, float scale) -> int {
    float s = scale > 0.0f ? scale : 1.0f;
    int q = (int)llrintf(v / s);
    q = q < -8 ? -8 : (q > 7 ? 7 : q);
    return q + 8;
  };
  auto write_pair_k = [&](int head, int ho) {
    const int xrow = ho / CELL;
    const int xoff = ho % CELL;
    const int64_t cell_byte = ((block_idx * num_heads + head) * (head_size / CELL) + xrow) *
                                  (block_size * K_BYTES_PER_CELL) +
                              token_offset * K_BYTES_PER_CELL + xoff / 2;
    const float scale = k_cell_max[head * (head_size / CELL) + xrow] / 8.0f;
    const int lo = quant(to_float_val(key[key_base + head * head_size + ho]), scale);
    const int hi = quant(to_float_val(key[key_base + head * head_size + ho + 1]), scale);
    key_cache[cell_byte] = (uint8_t)(lo | (hi << 4));
    k_scale[((block_idx * num_heads + head) * (head_size / CELL) + xrow) * block_size +
            token_offset] = scale;
  };
  auto write_pair_v = [&](int head, int ho) {
    const int64_t byte = ((block_idx * num_heads + head) * (head_size / 2) + ho / 2) *
                             block_size +
                         token_offset;
    const float scale = v_cell_max[head] / 8.0f;
    const int lo = quant(to_float_val(value[value_base + head * head_size + ho]), scale);
    const int hi = quant(to_float_val(value[value_base + head * head_size + ho + 1]), scale);
    value_cache[byte] = (uint8_t)(lo | (hi << 4));
    v_scale[(block_idx * num_heads + head) * block_size + token_offset] = scale;
  };
  if (has_k_pair0) {
    int head = (k_pair0 * 2) / head_size;
    int ho = (k_pair0 * 2) % head_size;
    write_pair_k(head, ho);
  }
  if (has_k_pair1) {
    int head = (k_pair1 * 2) / head_size;
    int ho = (k_pair1 * 2) % head_size;
    write_pair_k(head, ho);
  }
  if (has_v_pair0) {
    int head = v_pair0 / k_pairs_per_head;
    int ho = (v_pair0 % k_pairs_per_head) * 2;
    write_pair_v(head, ho);
  }
  if (has_v_pair1) {
    int head = v_pair1 / k_pairs_per_head;
    int ho = (v_pair1 % k_pairs_per_head) * 2;
    write_pair_v(head, ho);
  }
}

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

    uint32_t dtype,       // 0 => f16; 1 => bf16; 2 => f32
    uint32_t cache_dtype, // 0 => f16; 1 => bf16; 2 => f32; 3 => fp8_e4m3
    float *k_scale, float *v_scale) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  if (cache_dtype == 4) {
    // F4 cache: 128-thread blocks, one per token, dynamic shared for cell maxes.
    dim3 f4_block(128);
    int shared_bytes =
        (num_heads * (head_size / 32) + num_heads) * (int)sizeof(float);
    if (dtype == 0) {
      vllm::reshape_and_cache_f4_kernel<uint16_t>
          <<<grid, f4_block, shared_bytes, stream>>>(
              reinterpret_cast<uint16_t *>(key), reinterpret_cast<uint16_t *>(value),
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache), k_scale, v_scale,
              slot_mapping, key_stride, value_stride, num_heads, head_size,
              block_size);
    } else if (dtype == 1) {
      vllm::reshape_and_cache_f4_kernel<__nv_bfloat16>
          <<<grid, f4_block, shared_bytes, stream>>>(
              reinterpret_cast<__nv_bfloat16 *>(key),
              reinterpret_cast<__nv_bfloat16 *>(value),
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache), k_scale, v_scale,
              slot_mapping, key_stride, value_stride, num_heads, head_size,
              block_size);
    } else if (dtype == 2) {
      vllm::reshape_and_cache_f4_kernel<float>
          <<<grid, f4_block, shared_bytes, stream>>>(
              reinterpret_cast<float *>(key), reinterpret_cast<float *>(value),
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache), k_scale, v_scale,
              slot_mapping, key_stride, value_stride, num_heads, head_size,
              block_size);
    }
  } else if (cache_dtype == 3) {
    // FP8 E4M3 cache
    if (dtype == 0) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dtype == 1) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dtype == 2) {
      CALL_RESHAPE_AND_CACHE(float, uint8_t,
                             vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    // Non-FP8 cache
    if (dtype == 0) {
      CALL_RESHAPE_AND_CACHE(uint16_t, uint16_t,
                             vllm::Fp8KVCacheDataType::kAuto);
    } else if (dtype == 1) {
      CALL_RESHAPE_AND_CACHE(__nv_bfloat16, __nv_bfloat16,
                             vllm::Fp8KVCacheDataType::kAuto);
    } else if (dtype == 2) {
      CALL_RESHAPE_AND_CACHE(float, float, vllm::Fp8KVCacheDataType::kAuto);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}