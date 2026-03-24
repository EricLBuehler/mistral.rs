#include "pagedattention.cuh"
using namespace vllm;
extern "C" void paged_attention_v1_bf16(
    void *out,          // [num_seqs, num_heads, head_size]
    void *query,        // [num_seqs, num_heads, head_size]
    void *key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
    void *value_cache,  // [num_blocks, num_heads, head_size, block_size]
    void *alibi_slopes, // [num_heads]
    int32_t num_kv_heads, float scale, float softcapping,
    uint32_t *block_tables, // [num_seqs, max_num_blocks_per_seq]
    uint32_t *context_lens, // [num_seqs]
    int32_t block_size, int32_t max_context_len,

    int32_t num_seqs, int32_t num_heads, int32_t head_size,
    int32_t max_num_blocks_per_seq, int32_t q_stride, int32_t kv_block_stride,
    int32_t kv_head_stride, cudaStream_t stream,

    uint32_t cache_dtype, // 0 => f16; 1 => bf16; 2 => f32; 3 => fp8_e4m3
    float *k_scale, float *v_scale, const float *sinks) {

  if (cache_dtype == 3) {
    // FP8 cache
    CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, uint8_t,
                                vllm::Fp8KVCacheDataType::kFp8E4M3);
  } else {
    // Non-FP8 cache
    CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat16,
                                vllm::Fp8KVCacheDataType::kAuto);
  }
  CUDA_CHECK(cudaGetLastError());
}