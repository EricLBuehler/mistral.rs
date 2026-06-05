#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <exception>
#include <stdint.h>
#include <stdio.h>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/default_prefill_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/prefill.cuh"
#include "flashinfer/attention/variants.cuh"

using namespace flashinfer;

namespace mistralrs_flashinfer {

template <typename DType>
__global__ void reshape_and_cache_flashinfer_kernel(
    const DType *__restrict__ key, const DType *__restrict__ value,
    DType *__restrict__ key_cache, DType *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t key_stride,
    int32_t value_stride) {
  const int32_t token_idx = blockIdx.x;
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }

  const int64_t block_idx = slot / block_size;
  const int64_t block_offset = slot % block_size;
  const int32_t n = num_heads * head_size;

  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t head_idx = i / head_size;
    const int32_t dim_idx = i % head_size;
    const int64_t dst_idx =
        ((block_idx * num_heads + head_idx) * block_size + block_offset) *
            head_size +
        dim_idx;
    key_cache[dst_idx] = key[token_idx * key_stride + i];
    value_cache[dst_idx] = value[token_idx * value_stride + i];
  }
}

template <typename DType>
__global__ void gather_kv_cache_flashinfer_kernel(
    const DType *__restrict__ key_cache, const DType *__restrict__ value_cache,
    DType *__restrict__ k_out, DType *__restrict__ v_out,
    const int32_t *__restrict__ block_table,
    const int32_t *__restrict__ cu_seq_lens, int32_t num_tokens,
    int32_t block_size, int32_t block_table_stride, int32_t num_kv_heads,
    int32_t head_size) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

  int32_t seq_id = 0;
  while (cu_seq_lens[seq_id + 1] <= token_id) {
    seq_id++;
  }

  const int32_t seq_start = cu_seq_lens[seq_id];
  const int32_t seq_offset = token_id - seq_start;
  const int32_t table_idx = seq_offset / block_size;
  const int32_t slot = seq_offset % block_size;
  const int32_t block_idx =
      block_table[seq_id * block_table_stride + table_idx];
  const int32_t n = num_kv_heads * head_size;

  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t head_idx = i / head_size;
    const int32_t dim_idx = i % head_size;
    const int64_t cache_idx =
        ((int64_t(block_idx) * num_kv_heads + head_idx) * block_size + slot) *
            head_size +
        dim_idx;
    const int64_t out_idx = int64_t(token_id) * n + i;
    k_out[out_idx] = key_cache[cache_idx];
    v_out[out_idx] = value_cache[cache_idx];
  }
}

template <typename DType, uint32_t HEAD_DIM, bool USE_SLIDING_WINDOW,
          bool USE_LOGITS_SOFT_CAP>
cudaError_t run_flashinfer_decode(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, cudaStream_t stream) {
  using Params = BatchDecodeParams<DType, DType, DType, int32_t>;
  using AttentionVariant =
      DefaultAttention<false, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, false>;

  paged_kv_t<DType, int32_t> paged_kv(
      num_kv_heads, page_size, HEAD_DIM, batch_size, QKVLayout::kHND,
      static_cast<DType *>(key_cache), static_cast<DType *>(value_cache),
      const_cast<int32_t *>(kv_indices), const_cast<int32_t *>(kv_indptr),
      const_cast<int32_t *>(kv_last_page_len));

  Params params(static_cast<DType *>(q), /*q_rope_offset=*/nullptr, paged_kv,
                static_cast<DType *>(o), /*lse=*/nullptr,
                /*maybe_alibi_slopes=*/nullptr, num_qo_heads, q_stride_n,
                q_stride_h, window_left, logits_soft_cap, sm_scale, 1.0f, 1.0f);
  params.request_indices = const_cast<int32_t *>(request_indices);
  params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
  params.o_indptr = const_cast<int32_t *>(o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size_ptr);
  params.block_valid_mask = const_cast<bool *>(block_valid_mask);
  params.padded_batch_size = padded_batch_size;

  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PosEncodingMode::kNone,
                                            AttentionVariant, Params>(
          params, static_cast<DType *>(tmp_v), static_cast<float *>(tmp_s),
          /*enable_pdl=*/false, stream);
  if (status != cudaSuccess) {
    fprintf(stderr, "FlashInfer decode failed: %s\n",
            cudaGetErrorString(status));
  }
  return status;
}

template <typename DType, uint32_t HEAD_DIM, bool USE_SLIDING_WINDOW,
          bool USE_LOGITS_SOFT_CAP>
cudaError_t run_flashinfer_tensor_core_decode(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *qo_tile_indices,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, cudaStream_t stream) {
  using Params = BatchPrefillPagedParams<DType, DType, DType, int32_t>;
  using AttentionVariant =
      DefaultAttention<false, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, false>;

  paged_kv_t<DType, int32_t> paged_kv(
      num_kv_heads, page_size, HEAD_DIM, batch_size, QKVLayout::kHND,
      static_cast<DType *>(key_cache), static_cast<DType *>(value_cache),
      const_cast<int32_t *>(kv_indices), const_cast<int32_t *>(kv_indptr),
      const_cast<int32_t *>(kv_last_page_len));

  Params params(
      static_cast<DType *>(q), paged_kv, /*maybe_custom_mask=*/nullptr,
      const_cast<int32_t *>(q_indptr), /*maybe_mask_indptr=*/nullptr,
      /*maybe_q_rope_offset=*/nullptr, static_cast<DType *>(o),
      /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr, num_qo_heads, q_stride_n,
      q_stride_h, window_left, logits_soft_cap, sm_scale, 1.0f, 1.0f);
  params.request_indices = const_cast<int32_t *>(request_indices);
  params.qo_tile_indices = const_cast<int32_t *>(qo_tile_indices);
  params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
  params.merge_indptr = const_cast<int32_t *>(o_indptr);
  params.o_indptr = const_cast<int32_t *>(o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size_ptr);
  params.block_valid_mask = const_cast<bool *>(block_valid_mask);
  params.padded_batch_size = padded_batch_size;
  params.max_total_num_rows = batch_size;

  cudaError_t status = BatchPrefillWithPagedKVCacheDispatched<
      16, HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone,
      /*use_fp16_qk_reduction=*/false, MaskMode::kNone, AttentionVariant,
      Params>(params, static_cast<DType *>(tmp_v),
              static_cast<float *>(tmp_s),
              /*enable_pdl=*/false, stream);
  if (status != cudaSuccess) {
    fprintf(stderr, "FlashInfer tensor-core decode failed: %s\n",
            cudaGetErrorString(status));
  }
  return status;
}

template <typename DType, uint32_t HEAD_DIM, bool USE_SLIDING_WINDOW,
          bool USE_LOGITS_SOFT_CAP>
void run_flashinfer_prefill(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *request_indices,
    const int32_t *qo_tile_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, int32_t batch_size,
    int32_t padded_batch_size, int32_t total_q, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, bool causal, cudaStream_t stream) {
  using Params = BatchPrefillPagedParams<DType, DType, DType, int32_t>;
  using AttentionVariant =
      DefaultAttention<false, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, false>;

  paged_kv_t<DType, int32_t> paged_kv(
      num_kv_heads, page_size, HEAD_DIM, batch_size, QKVLayout::kHND,
      static_cast<DType *>(key_cache), static_cast<DType *>(value_cache),
      const_cast<int32_t *>(kv_indices), const_cast<int32_t *>(kv_indptr),
      const_cast<int32_t *>(kv_last_page_len));

  Params params(
      static_cast<DType *>(q), paged_kv, /*maybe_custom_mask=*/nullptr,
      const_cast<int32_t *>(q_indptr), /*maybe_mask_indptr=*/nullptr,
      /*maybe_q_rope_offset=*/nullptr, static_cast<DType *>(o),
      /*lse=*/nullptr, /*maybe_alibi_slopes=*/nullptr, num_qo_heads, q_stride_n,
      q_stride_h, window_left, logits_soft_cap, sm_scale, 1.0f, 1.0f);
  params.request_indices = const_cast<int32_t *>(request_indices);
  params.qo_tile_indices = const_cast<int32_t *>(qo_tile_indices);
  params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
  params.o_indptr = const_cast<int32_t *>(o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size_ptr);
  params.block_valid_mask = const_cast<bool *>(block_valid_mask);
  params.padded_batch_size = padded_batch_size;
  params.max_total_num_rows = total_q;

  cudaError_t status;
  if (causal) {
    status = BatchPrefillWithPagedKVCacheDispatched<
        64, HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone,
        /*use_fp16_qk_reduction=*/false, MaskMode::kCausal, AttentionVariant,
        Params>(params, static_cast<DType *>(nullptr),
                static_cast<float *>(nullptr),
                /*enable_pdl=*/false, stream);
  } else {
    status = BatchPrefillWithPagedKVCacheDispatched<
        64, HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone,
        /*use_fp16_qk_reduction=*/false, MaskMode::kNone, AttentionVariant,
        Params>(params, static_cast<DType *>(nullptr),
                static_cast<float *>(nullptr),
                /*enable_pdl=*/false, stream);
  }
  if (status != cudaSuccess) {
    fprintf(stderr, "FlashInfer prefill failed: %s\n",
            cudaGetErrorString(status));
  }
}

template <typename DType, uint32_t HEAD_DIM>
cudaError_t dispatch_flashinfer_decode_softcap(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *qo_tile_indices,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, bool use_tensor_cores, cudaStream_t stream) {
  if constexpr (!std::is_same_v<DType, float> && HEAD_DIM <= 256) {
    if (use_tensor_cores) {
      if (window_left >= 0) {
        if (logits_soft_cap > 0.0f) {
          return run_flashinfer_tensor_core_decode<DType, HEAD_DIM, true, true>(
              q, key_cache, value_cache, kv_indptr, kv_indices,
              kv_last_page_len, q_indptr, qo_tile_indices, request_indices,
              kv_tile_indices, o_indptr, kv_chunk_size_ptr, block_valid_mask, o,
              tmp_v, tmp_s, batch_size, padded_batch_size, num_qo_heads,
              num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
              window_left, logits_soft_cap, stream);
        } else {
          return run_flashinfer_tensor_core_decode<DType, HEAD_DIM, true, false>(
              q, key_cache, value_cache, kv_indptr, kv_indices,
              kv_last_page_len, q_indptr, qo_tile_indices, request_indices,
              kv_tile_indices, o_indptr, kv_chunk_size_ptr, block_valid_mask, o,
              tmp_v, tmp_s, batch_size, padded_batch_size, num_qo_heads,
              num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
              window_left, logits_soft_cap, stream);
        }
      } else if (logits_soft_cap > 0.0f) {
        return run_flashinfer_tensor_core_decode<DType, HEAD_DIM, false, true>(
            q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
            q_indptr, qo_tile_indices, request_indices, kv_tile_indices,
            o_indptr, kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s,
            batch_size, padded_batch_size, num_qo_heads, num_kv_heads,
            page_size, q_stride_n, q_stride_h, sm_scale, window_left,
            logits_soft_cap, stream);
      } else {
        return run_flashinfer_tensor_core_decode<DType, HEAD_DIM, false, false>(
            q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
            q_indptr, qo_tile_indices, request_indices, kv_tile_indices,
            o_indptr, kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s,
            batch_size, padded_batch_size, num_qo_heads, num_kv_heads,
            page_size, q_stride_n, q_stride_h, sm_scale, window_left,
            logits_soft_cap, stream);
      }
    }
  } else {
    (void)use_tensor_cores;
  }

  if (window_left >= 0) {
    if (logits_soft_cap > 0.0f) {
      return run_flashinfer_decode<DType, HEAD_DIM, true, true>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
          block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
          num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
          sm_scale, window_left, logits_soft_cap, stream);
    } else {
      return run_flashinfer_decode<DType, HEAD_DIM, true, false>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
          block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
          num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
          sm_scale, window_left, logits_soft_cap, stream);
    }
  } else if (logits_soft_cap > 0.0f) {
    return run_flashinfer_decode<DType, HEAD_DIM, false, true>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
        num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
        window_left, logits_soft_cap, stream);
  } else {
    return run_flashinfer_decode<DType, HEAD_DIM, false, false>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
        num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
        window_left, logits_soft_cap, stream);
  }
}

template <typename DType>
cudaError_t dispatch_flashinfer_decode_head_dim(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *qo_tile_indices,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, bool use_tensor_cores, cudaStream_t stream) {
  if (head_size == 64) {
    return dispatch_flashinfer_decode_softcap<DType, 64>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, use_tensor_cores,
        stream);
  } else if (head_size == 128) {
    return dispatch_flashinfer_decode_softcap<DType, 128>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, use_tensor_cores,
        stream);
  } else if (head_size == 256) {
    return dispatch_flashinfer_decode_softcap<DType, 256>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, use_tensor_cores,
        stream);
  } else if (head_size == 512) {
    return dispatch_flashinfer_decode_softcap<DType, 512>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, use_tensor_cores,
        stream);
  } else {
    fprintf(stderr, "FlashInfer decode received unsupported head_size %d\n",
            head_size);
    return cudaErrorInvalidValue;
  }
}

template <typename DType, uint32_t HEAD_DIM>
void dispatch_flashinfer_prefill_softcap(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *request_indices,
    const int32_t *qo_tile_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, int32_t batch_size,
    int32_t padded_batch_size, int32_t total_q, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, bool causal, cudaStream_t stream) {
  if (window_left >= 0) {
    if (logits_soft_cap > 0.0f) {
      run_flashinfer_prefill<DType, HEAD_DIM, true, true>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
          total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n,
          q_stride_h, sm_scale, window_left, logits_soft_cap, causal, stream);
    } else {
      run_flashinfer_prefill<DType, HEAD_DIM, true, false>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
          total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n,
          q_stride_h, sm_scale, window_left, logits_soft_cap, causal, stream);
    }
  } else if (logits_soft_cap > 0.0f) {
    run_flashinfer_prefill<DType, HEAD_DIM, false, true>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
        total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
        sm_scale, window_left, logits_soft_cap, causal, stream);
  } else {
    run_flashinfer_prefill<DType, HEAD_DIM, false, false>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
        total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
        sm_scale, window_left, logits_soft_cap, causal, stream);
  }
}

template <typename DType>
void dispatch_flashinfer_prefill_head_dim(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *request_indices,
    const int32_t *qo_tile_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, int32_t batch_size,
    int32_t padded_batch_size, int32_t total_q, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, bool causal, cudaStream_t stream) {
  if (head_size == 64) {
    dispatch_flashinfer_prefill_softcap<DType, 64>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
        total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
        sm_scale, window_left, logits_soft_cap, causal, stream);
  } else if (head_size == 128) {
    dispatch_flashinfer_prefill_softcap<DType, 128>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
        total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
        sm_scale, window_left, logits_soft_cap, causal, stream);
  } else if (head_size == 256) {
    dispatch_flashinfer_prefill_softcap<DType, 256>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
        kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
        total_q, num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
        sm_scale, window_left, logits_soft_cap, causal, stream);
  } else {
    fprintf(stderr, "FlashInfer prefill received unsupported head_size %d\n",
            head_size);
  }
}

} // namespace mistralrs_flashinfer

extern "C" void reshape_and_cache_flashinfer(
    void *key, void *value, void *key_cache, void *value_cache,
    int64_t *slot_mapping, int32_t num_tokens, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t key_stride,
    int32_t value_stride, uint32_t dtype, cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  if (dtype == 0) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__half>
        <<<grid, block, 0, stream>>>(
            static_cast<__half *>(key), static_cast<__half *>(value),
            static_cast<__half *>(key_cache),
            static_cast<__half *>(value_cache), slot_mapping, num_heads,
            head_size, block_size, key_stride, value_stride);
  } else if (dtype == 1) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 *>(key),
                                     static_cast<__nv_bfloat16 *>(value),
                                     static_cast<__nv_bfloat16 *>(key_cache),
                                     static_cast<__nv_bfloat16 *>(value_cache),
                                     slot_mapping, num_heads, head_size,
                                     block_size, key_stride, value_stride);
  } else if (dtype == 2) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<float>
        <<<grid, block, 0, stream>>>(
            static_cast<float *>(key), static_cast<float *>(value),
            static_cast<float *>(key_cache), static_cast<float *>(value_cache),
            slot_mapping, num_heads, head_size, block_size, key_stride,
            value_stride);
  } else {
    fprintf(stderr,
            "reshape_and_cache_flashinfer received unsupported dtype %u\n",
            dtype);
  }
}

extern "C" int32_t flashinfer_decode(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *qo_tile_indices,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, uint32_t dtype, bool use_tensor_cores,
    cudaStream_t stream) {
  try {
    if (dtype == 0) {
      return mistralrs_flashinfer::dispatch_flashinfer_decode_head_dim<__half>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
          padded_batch_size, num_qo_heads, num_kv_heads, head_size, page_size,
          q_stride_n, q_stride_h, sm_scale, window_left, logits_soft_cap,
          use_tensor_cores, stream);
    } else if (dtype == 1) {
      return mistralrs_flashinfer::dispatch_flashinfer_decode_head_dim<__nv_bfloat16>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
          padded_batch_size, num_qo_heads, num_kv_heads, head_size, page_size,
          q_stride_n, q_stride_h, sm_scale, window_left, logits_soft_cap,
          use_tensor_cores, stream);
    } else if (dtype == 2) {
      return mistralrs_flashinfer::dispatch_flashinfer_decode_head_dim<float>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, qo_tile_indices, request_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, tmp_v, tmp_s, batch_size,
          padded_batch_size, num_qo_heads, num_kv_heads, head_size, page_size,
          q_stride_n, q_stride_h, sm_scale, window_left, logits_soft_cap,
          use_tensor_cores, stream);
    }
    fprintf(stderr, "FlashInfer decode received unsupported dtype %u\n", dtype);
  } catch (const std::exception &e) {
    fprintf(stderr, "FlashInfer decode failed: %s\n", e.what());
    return cudaErrorUnknown;
  } catch (...) {
    fprintf(stderr, "FlashInfer decode failed with unknown exception\n");
    return cudaErrorUnknown;
  }
  return cudaErrorInvalidValue;
}

extern "C" int32_t flashinfer_prefill(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *q_indptr, const int32_t *request_indices,
    const int32_t *qo_tile_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, int32_t batch_size,
    int32_t padded_batch_size, int32_t total_q, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, uint32_t dtype, bool causal, cudaStream_t stream) {
  try {
    if (dtype == 0) {
      mistralrs_flashinfer::dispatch_flashinfer_prefill_head_dim<__half>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
          total_q, num_qo_heads, num_kv_heads, head_size, page_size, q_stride_n,
          q_stride_h, sm_scale, window_left, logits_soft_cap, causal, stream);
    } else if (dtype == 1) {
      mistralrs_flashinfer::dispatch_flashinfer_prefill_head_dim<__nv_bfloat16>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          q_indptr, request_indices, qo_tile_indices, kv_tile_indices, o_indptr,
          kv_chunk_size_ptr, block_valid_mask, o, batch_size, padded_batch_size,
          total_q, num_qo_heads, num_kv_heads, head_size, page_size, q_stride_n,
          q_stride_h, sm_scale, window_left, logits_soft_cap, causal, stream);
    } else {
      fprintf(stderr, "FlashInfer prefill received unsupported dtype %u\n",
              dtype);
      return 1;
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "FlashInfer prefill failed: %s\n", e.what());
    return 1;
  } catch (...) {
    fprintf(stderr, "FlashInfer prefill failed with unknown exception\n");
    return 1;
  }
  return 0;
}

extern "C" void gather_kv_cache_flashinfer(
    void *key_cache, void *value_cache, void *k_out, void *v_out,
    const int32_t *block_table, const int32_t *cu_seq_lens, int32_t num_tokens,
    int32_t num_seqs, int32_t block_size, int32_t block_table_stride,
    int32_t num_kv_heads, int32_t head_size, uint32_t dtype,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_kv_heads * head_size, 512));

  if (dtype == 0) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__half>
        <<<grid, block, 0, stream>>>(
            static_cast<__half *>(key_cache),
            static_cast<__half *>(value_cache), static_cast<__half *>(k_out),
            static_cast<__half *>(v_out), block_table, cu_seq_lens, num_tokens,
            block_size, block_table_stride, num_kv_heads, head_size);
  } else if (dtype == 1) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__nv_bfloat16>
        <<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 *>(key_cache),
                                     static_cast<__nv_bfloat16 *>(value_cache),
                                     static_cast<__nv_bfloat16 *>(k_out),
                                     static_cast<__nv_bfloat16 *>(v_out),
                                     block_table, cu_seq_lens, num_tokens,
                                     block_size, block_table_stride,
                                     num_kv_heads, head_size);
  } else if (dtype == 2) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<float>
        <<<grid, block, 0, stream>>>(
            static_cast<float *>(key_cache), static_cast<float *>(value_cache),
            static_cast<float *>(k_out), static_cast<float *>(v_out),
            block_table, cu_seq_lens, num_tokens, block_size,
            block_table_stride, num_kv_heads, head_size);
  } else {
    fprintf(stderr,
            "gather_kv_cache_flashinfer received unsupported dtype %u\n",
            dtype);
  }
}
