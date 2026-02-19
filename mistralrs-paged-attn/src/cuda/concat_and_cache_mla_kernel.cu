#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#include <algorithm>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

template <typename scalar_t>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t *__restrict__ ckv,  // [num_tokens, kv_lora_rank]
    const scalar_t *__restrict__ k_pe, // [num_tokens, kpe_head_dim]
    scalar_t *__restrict__ ckv_cache,  // [num_blocks, block_size, kv_lora_rank]
    scalar_t *__restrict__ kpe_cache,  // [num_blocks, block_size, kpe_head_dim]
    const int64_t *__restrict__ slot_mapping, // [num_tokens]
    const int ckv_stride, const int kpe_stride, const int kv_lora_rank,
    const int kpe_head_dim, const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int64_t ckv_dst_base =
      block_idx * block_size * kv_lora_rank + block_offset * kv_lora_rank;
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const int64_t src_idx = token_idx * ckv_stride + i;
    ckv_cache[ckv_dst_base + i] = ckv[src_idx];
  }

  const int64_t kpe_dst_base =
      block_idx * block_size * kpe_head_dim + block_offset * kpe_head_dim;
  for (int i = threadIdx.x; i < kpe_head_dim; i += blockDim.x) {
    const int64_t src_idx = token_idx * kpe_stride + i;
    kpe_cache[kpe_dst_base + i] = k_pe[src_idx];
  }
}

extern "C" void
concat_and_cache_mla(void *ckv,       // [num_tokens, kv_lora_rank]
                     void *k_pe,      // [num_tokens, kpe_head_dim]
                     void *ckv_cache, // [num_blocks, block_size, kv_lora_rank]
                     void *kpe_cache, // [num_blocks, block_size, kpe_head_dim]
                     int64_t *slot_mapping, // [num_tokens]
                     int32_t num_tokens, int32_t kv_lora_rank,
                     int32_t kpe_head_dim, int32_t block_size,
                     int32_t ckv_stride, int32_t kpe_stride,
                     cudaStream_t stream, uint32_t dtype) {
  dim3 grid(num_tokens);
  int max_dim = kv_lora_rank > kpe_head_dim ? kv_lora_rank : kpe_head_dim;
  dim3 block(std::min(max_dim, 512));

  if (dtype == 0) {
    concat_and_cache_mla_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half *>(ckv), reinterpret_cast<__half *>(k_pe),
        reinterpret_cast<__half *>(ckv_cache),
        reinterpret_cast<__half *>(kpe_cache), slot_mapping, ckv_stride,
        kpe_stride, kv_lora_rank, kpe_head_dim, block_size);
  } else if (dtype == 1) {
    concat_and_cache_mla_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(ckv),
        reinterpret_cast<__nv_bfloat16 *>(k_pe),
        reinterpret_cast<__nv_bfloat16 *>(ckv_cache),
        reinterpret_cast<__nv_bfloat16 *>(kpe_cache), slot_mapping, ckv_stride,
        kpe_stride, kv_lora_rank, kpe_head_dim, block_size);
  } else if (dtype == 2) {
    concat_and_cache_mla_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(ckv), reinterpret_cast<float *>(k_pe),
        reinterpret_cast<float *>(ckv_cache),
        reinterpret_cast<float *>(kpe_cache), slot_mapping, ckv_stride,
        kpe_stride, kv_lora_rank, kpe_head_dim, block_size);
  }
  CUDA_CHECK(cudaGetLastError());
}
