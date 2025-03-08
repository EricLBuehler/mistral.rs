#include <stdint.h>

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__device__ void
copy_blocks_internal_kernel(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                            const int64_t *__restrict__ block_mapping,
                            const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t *key_cache = reinterpret_cast<scalar_t *>(key_cache_ptrs[layer_idx]);
  scalar_t *value_cache =
      reinterpret_cast<scalar_t *>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Monomorphize the generics ourselves
extern "C" __global__ void
copy_blocks_kernel_u8(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                      const int64_t *__restrict__ block_mapping,
                      const int numel_per_block) {
  copy_blocks_internal_kernel<uint8_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block);
}

extern "C" __global__ void
copy_blocks_kernel_u32(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block) {
  copy_blocks_internal_kernel<uint32_t>(key_cache_ptrs, value_cache_ptrs,
                                        block_mapping, numel_per_block);
}

extern "C" __global__ void
copy_blocks_kernel_i64(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block) {
  copy_blocks_internal_kernel<int64_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block);
}

extern "C" __global__ void
copy_blocks_kernel_f32(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block) {
  copy_blocks_internal_kernel<float>(key_cache_ptrs, value_cache_ptrs,
                                     block_mapping, numel_per_block);
}

extern "C" __global__ void
copy_blocks_kernel_f64(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block) {
  copy_blocks_internal_kernel<double>(key_cache_ptrs, value_cache_ptrs,
                                      block_mapping, numel_per_block);
}

// f16, bf16 are special cases: We use a 16-bit integer to simulate the bit
// width. SAFETY: This is technically UB due to aliasing, but it is OK because
// the width is compatible.
extern "C" __global__ void
copy_blocks_kernel_f16(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                       const int64_t *__restrict__ block_mapping,
                       const int numel_per_block) {
  copy_blocks_internal_kernel<int16_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block);
}

extern "C" __global__ void
copy_blocks_kernel_bf16(int64_t *key_cache_ptrs, int64_t *value_cache_ptrs,
                        const int64_t *__restrict__ block_mapping,
                        const int numel_per_block) {
  copy_blocks_internal_kernel<int16_t>(key_cache_ptrs, value_cache_ptrs,
                                       block_mapping, numel_per_block);
}