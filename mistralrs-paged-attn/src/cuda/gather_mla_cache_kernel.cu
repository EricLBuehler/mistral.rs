#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

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
__global__ void gather_mla_cache_kernel(
    const scalar_t
        *__restrict__ ckv_cache, // [num_blocks, block_size, kv_lora_rank]
    const scalar_t
        *__restrict__ kpe_cache,    // [num_blocks, block_size, kpe_head_dim]
    scalar_t *__restrict__ ckv_out, // [num_tokens, kv_lora_rank]
    scalar_t *__restrict__ kpe_out, // [num_tokens, kpe_head_dim]
    const int32_t *__restrict__ block_table,  // [batch, max_blocks]
    const int32_t *__restrict__ cu_seq_lens,  // [batch + 1]
    const int32_t *__restrict__ token_to_seq, // [num_tokens]
    const int32_t num_tokens, const int32_t block_size,
    const int32_t block_table_stride, const int32_t kv_lora_rank,
    const int32_t kpe_head_dim) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }
  const int32_t batch_id = token_to_seq[token_id];
  const int32_t batch_start = cu_seq_lens[batch_id];
  const int32_t batch_offset = token_id - batch_start;
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int64_t cache_base = static_cast<int64_t>(block_id) * block_size + slot;
  const int64_t ckv_offset = cache_base * kv_lora_rank;
  const int64_t kpe_offset = cache_base * kpe_head_dim;
  const int64_t out_ckv_offset = static_cast<int64_t>(token_id) * kv_lora_rank;
  const int64_t out_kpe_offset = static_cast<int64_t>(token_id) * kpe_head_dim;

  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    ckv_out[out_ckv_offset + i] = ckv_cache[ckv_offset + i];
  }
  for (int i = threadIdx.x; i < kpe_head_dim; i += blockDim.x) {
    kpe_out[out_kpe_offset + i] = kpe_cache[kpe_offset + i];
  }
}

extern "C" void
gather_mla_cache(void *ckv_cache, void *kpe_cache, void *ckv_out, void *kpe_out,
                 const int32_t *block_table, const int32_t *cu_seq_lens,
                 const int32_t *token_to_seq, int32_t num_tokens,
                 int32_t block_size, int32_t block_table_stride,
                 int32_t kv_lora_rank, int32_t kpe_head_dim,
                 cudaStream_t stream, uint32_t dtype) {
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(256);

  if (dtype == 0) {
    gather_mla_cache_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<__half *>(ckv_cache),
        reinterpret_cast<__half *>(kpe_cache),
        reinterpret_cast<__half *>(ckv_out),
        reinterpret_cast<__half *>(kpe_out), block_table, cu_seq_lens,
        token_to_seq, num_tokens, block_size, block_table_stride, kv_lora_rank,
        kpe_head_dim);
  } else if (dtype == 1) {
    gather_mla_cache_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(ckv_cache),
        reinterpret_cast<__nv_bfloat16 *>(kpe_cache),
        reinterpret_cast<__nv_bfloat16 *>(ckv_out),
        reinterpret_cast<__nv_bfloat16 *>(kpe_out), block_table, cu_seq_lens,
        token_to_seq, num_tokens, block_size, block_table_stride, kv_lora_rank,
        kpe_head_dim);
  } else if (dtype == 2) {
    gather_mla_cache_kernel<float><<<grid, block, 0, stream>>>(
        reinterpret_cast<float *>(ckv_cache),
        reinterpret_cast<float *>(kpe_cache),
        reinterpret_cast<float *>(ckv_out), reinterpret_cast<float *>(kpe_out),
        block_table, cu_seq_lens, token_to_seq, num_tokens, block_size,
        block_table_stride, kv_lora_rank, kpe_head_dim);
  }
  CUDA_CHECK(cudaGetLastError());
}
