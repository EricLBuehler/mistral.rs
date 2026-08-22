#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fa3Fp8DecodeScheduleParams {
    const int32_t *cu_seqlens_q;
    const int32_t *seqused_k;
    int32_t *scheduler_metadata;
    int32_t batch_size;
    int32_t total_q;
    int32_t num_q_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t page_size;
    int32_t max_seqlen_k;
    int32_t num_splits;
    int32_t num_sm;
    int32_t device_id;
} Fa3Fp8DecodeScheduleParams;

typedef struct Fa3Fp8DecodeParams {
    Fa3Fp8DecodeScheduleParams schedule;
    const void *q;
    const void *k;
    const void *v;
    void *out;
    float *softmax_lse;
    float *out_accum;
    float *softmax_lse_accum;
    const int32_t *page_table;
    const float *q_descale;
    const float *k_descale;
    const float *v_descale;
    int64_t q_row_stride;
    int64_t q_head_stride;
    int64_t k_token_stride;
    int64_t k_head_stride;
    int64_t k_page_stride;
    int64_t v_token_stride;
    int64_t v_head_stride;
    int64_t v_page_stride;
    int64_t out_row_stride;
    int64_t out_head_stride;
    int64_t page_table_batch_stride;
    int64_t q_descale_batch_stride;
    int64_t q_descale_head_stride;
    int64_t k_descale_batch_stride;
    int64_t k_descale_head_stride;
    int64_t v_descale_batch_stride;
    int64_t v_descale_head_stride;
    int32_t num_pages;
    int32_t max_pages_per_sequence;
    float softmax_scale;
    int32_t scheduler_metadata_prepared;
} Fa3Fp8DecodeParams;

size_t fa3_fp8_decode_scheduler_metadata_i32(int32_t batch_size, int32_t num_splits);
size_t fa3_fp8_decode_out_accum_f32(const Fa3Fp8DecodeParams *params);
size_t fa3_fp8_decode_lse_accum_f32(const Fa3Fp8DecodeParams *params);
int fa3_fp8_decode_materialize_paged_metadata(
    const int32_t *paged_kv_indptr, const int32_t *paged_kv_indices,
    const int32_t *paged_kv_last_page_len, int32_t *page_table,
    int32_t *seqused_k, int32_t batch_size, int32_t page_table_batch_stride,
    int32_t page_size, cudaStream_t stream);
int fa3_fp8_decode_prepare(const Fa3Fp8DecodeScheduleParams *params, cudaStream_t stream);
int fa3_fp8_decode_run(const Fa3Fp8DecodeParams *params, cudaStream_t stream);
int fa3_bf16_to_e4m3_static(const void *input, void *output, int32_t rows, int32_t columns,
                            int64_t input_row_stride, int64_t output_row_stride,
                            const float *descale, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
