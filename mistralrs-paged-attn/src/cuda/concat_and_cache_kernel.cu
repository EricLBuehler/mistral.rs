#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

template <typename scalar_t, typename cache_t>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size                      //
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
    return;
    }
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                    int src_stride, int dst_stride, int size, int offset) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            const int64_t src_idx = token_idx * src_stride + i;
            const int64_t dst_idx =
                block_idx * block_stride + block_offset * entry_stride + i + offset;
            dst[dst_idx] = src[src_idx];
        }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}



#define CALL_CONCAT_AND_CACHE_MLA(T)                                     \
    concat_and_cache_mla_kernel<T><<<grid, block, 0, stream>>>(      \
        reinterpret_cast<T*>(kv_c),                                        \
        reinterpret_cast<T*>(k_pe),                                      \
        reinterpret_cast<T*>(kv_cache),                                  \
        slot_mapping,                                                       \
        block_stride,                                                     \
        entry_stride,                                                       \
        kv_c_stride,                                                     \
        k_pe_stride,                                                        \
        kv_lora_rank,                                                        \
        pe_dim,                                                       \
        block_size                                                               \
    );

extern "C" void concat_and_cache_mla(
    void* kv_c,          // [num_tokens, kv_lora_rank]
    void* k_pe,          // [num_tokens, pe_dim]
    void* kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    int64_t* slot_mapping,  // [num_tokens] or [num_actual_tokens]

    int num_tokens,
    int kv_lora_rank,
    int pe_dim,
    int block_size,

    int kv_c_stride,
    int k_pe_stride,
    int block_stride,
    int entry_stride,
    cudaStream_t stream,

    uint32_t dtype      // 0 => f16; 1 => bf16; 2 => f32

) {
  dim3 grid(num_tokens);
  dim3 block(std::min(kv_lora_rank, 512));

  if (dtype == 0){
    CALL_CONCAT_AND_CACHE_MLA(__half);
  } else if (dtype == 1) {
    CALL_CONCAT_AND_CACHE_MLA(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_CONCAT_AND_CACHE_MLA(float);
  }
  CUDA_CHECK(cudaGetLastError());
}
