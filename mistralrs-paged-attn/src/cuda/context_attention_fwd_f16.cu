#include "context_attention_fwd.cuh"
using namespace vllm;
extern "C" void context_attention_fwd_f16(
    void *out,          // [total_new_tokens, num_heads, head_size]
    void *query,        // [total_new_tokens, num_heads, head_size]
    void *key,          // [total_new_tokens, num_kv_heads, head_size]
    void *value,        // [total_new_tokens, num_kv_heads, head_size]
    void *key_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    void *value_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    int32_t num_kv_heads, float scale,
    uint32_t *block_tables, // [num_seqs, max_num_blocks_per_seq]
    int32_t *context_lens,  // [num_seqs]
    int32_t *query_lens,    // [num_seqs]
    int32_t *query_start_locs, // [num_seqs+1]
    int32_t *seq_ids,       // [total_new_tokens]
    int32_t max_num_blocks_per_seq,
    int32_t total_new_tokens, int32_t num_heads, int32_t head_size,
    int32_t block_size,
    int32_t kv_block_stride, int32_t kv_head_stride,
    cudaStream_t stream,
    uint32_t cache_dtype, // 0 => f16; 1 => bf16; 2 => f32; 3 => fp8_e4m3
    float *k_scale, float *v_scale,
    int32_t sliding_window, const float *sinks,
    int32_t max_total_kv_len) {

  if (cache_dtype == 3) {
    CALL_CTX_ATTN_LAUNCHER_BLOCK_SIZE(uint16_t, uint8_t,
                                      vllm::Fp8KVCacheDataType::kFp8E4M3);
  } else {
    CALL_CTX_ATTN_LAUNCHER_BLOCK_SIZE(uint16_t, uint16_t,
                                      vllm::Fp8KVCacheDataType::kAuto);
  }
  CTX_ATTN_CUDA_CHECK(cudaGetLastError());
}
