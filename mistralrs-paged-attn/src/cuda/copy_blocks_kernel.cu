#include <stdint.h>

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__device__ void
copy_blocks_internal_kernel(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                            const int64_t *__restrict__ block_mapping,
                            const int numel_per_block_key,
                            const int numel_per_block_value) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t *key_cache = reinterpret_cast<scalar_t *>(key_cache_ptrs[layer_idx]);
  scalar_t *value_cache =
      reinterpret_cast<scalar_t *>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset_key = src_block_number * numel_per_block_key;
  const int64_t dst_block_offset_key = dst_block_number * numel_per_block_key;
  for (int i = threadIdx.x; i < numel_per_block_key; i += blockDim.x) {
    int64_t src_offset = src_block_offset_key + i;
    int64_t dst_offset = dst_block_offset_key + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  const int64_t src_block_offset_value =
      src_block_number * numel_per_block_value;
  const int64_t dst_block_offset_value =
      dst_block_number * numel_per_block_value;
  for (int i = threadIdx.x; i < numel_per_block_value; i += blockDim.x) {
    int64_t src_offset = src_block_offset_value + i;
    int64_t dst_offset = dst_block_offset_value + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

extern "C" __global__ void
copy_blocks_kernel_f32(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block_key,
                       const int numel_per_block_value) {
  copy_blocks_internal_kernel<float>(key_cache_ptrs, value_cache_ptrs,
                                     block_mapping, numel_per_block_key,
                                     numel_per_block_value);
}

extern "C" __global__ void
copy_blocks_kernel_f16(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block_key,
                       const int numel_per_block_value) {
  copy_blocks_internal_kernel<int16_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block_key,
                                       numel_per_block_value);
}

extern "C" __global__ void
copy_blocks_kernel_bf16(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                        const int64_t *__restrict__ block_mapping,
                        const int numel_per_block_key,
                        const int numel_per_block_value) {
  copy_blocks_internal_kernel<int16_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block_key,
                                       numel_per_block_value);
}

//
extern "C" void copy_blocks_f32(int64_t *key_cache_ptrs,
                                int64_t *value_cache_ptrs,
                                const int64_t *__restrict__ block_mapping,
                                const int num_layers, const int num_pairs,
                                int numel_per_block_key,
                                int numel_per_block_value, int64_t stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  int num_threads = numel_per_block_key;
  if (numel_per_block_value > num_threads) {
    num_threads = numel_per_block_value;
  }
  if (num_threads > 1024) {
    num_threads = 1024;
  }
  copy_blocks_kernel_f32<<<dim3(num_layers, num_pairs, 1), num_threads, 0,
                           stream>>>(key_cache_ptrs, value_cache_ptrs,
                                     block_mapping, numel_per_block_key,
                                     numel_per_block_value);
}

extern "C" void copy_blocks_f16(int64_t *key_cache_ptrs,
                                int64_t *value_cache_ptrs,
                                const int64_t *__restrict__ block_mapping,
                                const int num_layers, const int num_pairs,
                                int numel_per_block_key,
                                int numel_per_block_value, int64_t stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  int num_threads = numel_per_block_key;
  if (numel_per_block_value > num_threads) {
    num_threads = numel_per_block_value;
  }
  if (num_threads > 1024) {
    num_threads = 1024;
  }
  copy_blocks_kernel_f16<<<dim3(num_layers, num_pairs, 1), num_threads, 0,
                           stream>>>(key_cache_ptrs, value_cache_ptrs,
                                     block_mapping, numel_per_block_key,
                                     numel_per_block_value);
}

extern "C" void copy_blocks_bf16(int64_t *key_cache_ptrs,
                                 int64_t *value_cache_ptrs,
                                 const int64_t *__restrict__ block_mapping,
                                 const int num_layers, const int num_pairs,
                                 int numel_per_block_key,
                                 int numel_per_block_value, int64_t stream_) {
  cudaStream_t stream = (cudaStream_t)stream_;
  int num_threads = numel_per_block_key;
  if (numel_per_block_value > num_threads) {
    num_threads = numel_per_block_value;
  }
  if (num_threads > 1024) {
    num_threads = 1024;
  }
  copy_blocks_kernel_bf16<<<dim3(num_layers, num_pairs, 1), num_threads, 0,
                            stream>>>(key_cache_ptrs, value_cache_ptrs,
                                      block_mapping, numel_per_block_key,
                                      numel_per_block_value);
}
