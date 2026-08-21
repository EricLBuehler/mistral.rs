#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <exception>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/variants.cuh"

using namespace flashinfer;

namespace mistralrs_flashinfer {

template <typename DType, typename CacheType>
__global__ void reshape_and_cache_flashinfer_kernel(
    const DType *__restrict__ key, const DType *__restrict__ value,
    CacheType *__restrict__ key_cache, CacheType *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t key_stride,
    int32_t value_stride, float k_scale, float v_scale) {
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
    if constexpr (std::is_same_v<DType, CacheType>) {
      key_cache[dst_idx] = key[token_idx * key_stride + i];
      value_cache[dst_idx] = value[token_idx * value_stride + i];
    } else {
      key_cache[dst_idx] = static_cast<CacheType>(
          static_cast<float>(key[token_idx * key_stride + i]) / k_scale);
      value_cache[dst_idx] = static_cast<CacheType>(
          static_cast<float>(value[token_idx * value_stride + i]) / v_scale);
    }
  }
}

template <typename CacheType, typename OutType>
__global__ void gather_kv_cache_flashinfer_kernel(
    const CacheType *__restrict__ key_cache,
    const CacheType *__restrict__ value_cache, OutType *__restrict__ k_out,
    OutType *__restrict__ v_out,
    const int32_t *__restrict__ block_table,
    const int32_t *__restrict__ cu_seq_lens, int32_t num_tokens,
    int32_t block_size, int32_t block_table_stride, int32_t num_kv_heads,
    int32_t head_size, float k_scale, float v_scale) {
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
    if constexpr (std::is_same_v<CacheType, OutType>) {
      k_out[out_idx] = key_cache[cache_idx];
      v_out[out_idx] = value_cache[cache_idx];
    } else {
      k_out[out_idx] = static_cast<OutType>(
          static_cast<float>(key_cache[cache_idx]) * k_scale);
      v_out[out_idx] = static_cast<OutType>(
          static_cast<float>(value_cache[cache_idx]) * v_scale);
    }
  }
}

template <typename DType, typename CacheType, uint32_t HEAD_DIM,
          bool USE_SLIDING_WINDOW, bool USE_LOGITS_SOFT_CAP>
cudaError_t run_flashinfer_decode(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, float k_scale, float v_scale, cudaStream_t stream) {
  using Params = BatchDecodeParams<DType, CacheType, DType, int32_t>;
  using AttentionVariant =
      DefaultAttention<false, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, false>;

  paged_kv_t<CacheType, int32_t> paged_kv(
      num_kv_heads, page_size, HEAD_DIM, batch_size, QKVLayout::kHND,
      static_cast<CacheType *>(key_cache), static_cast<CacheType *>(value_cache),
      const_cast<int32_t *>(kv_indices), const_cast<int32_t *>(kv_indptr),
      const_cast<int32_t *>(kv_last_page_len));

  Params params(static_cast<DType *>(q), /*q_rope_offset=*/nullptr, paged_kv,
                static_cast<DType *>(o), /*lse=*/nullptr,
                /*maybe_alibi_slopes=*/nullptr, num_qo_heads, q_stride_n,
                q_stride_h, window_left, logits_soft_cap, sm_scale * k_scale,
                1.0f, 1.0f);
  params.v_scale = v_scale;
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

template <typename DType, typename CacheType, uint32_t HEAD_DIM>
cudaError_t dispatch_flashinfer_decode_softcap(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t page_size, int32_t q_stride_n,
    int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, float k_scale, float v_scale,
    cudaStream_t stream) {
  if (window_left >= 0) {
    if (logits_soft_cap > 0.0f) {
      return run_flashinfer_decode<DType, CacheType, HEAD_DIM, true, true>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
          block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
          num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
          sm_scale, window_left, logits_soft_cap, k_scale, v_scale, stream);
    } else {
      return run_flashinfer_decode<DType, CacheType, HEAD_DIM, true, false>(
          q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
          request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
          block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
          num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h,
          sm_scale, window_left, logits_soft_cap, k_scale, v_scale, stream);
    }
  } else if (logits_soft_cap > 0.0f) {
    return run_flashinfer_decode<DType, CacheType, HEAD_DIM, false, true>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
        num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
        window_left, logits_soft_cap, k_scale, v_scale, stream);
  } else {
    return run_flashinfer_decode<DType, CacheType, HEAD_DIM, false, false>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,
        num_qo_heads, num_kv_heads, page_size, q_stride_n, q_stride_h, sm_scale,
        window_left, logits_soft_cap, k_scale, v_scale, stream);
  }
}

template <typename DType, typename CacheType>
cudaError_t dispatch_flashinfer_decode_head_dim(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, float k_scale, float v_scale,
    cudaStream_t stream) {
  if (head_size == 64) {
    return dispatch_flashinfer_decode_softcap<DType, CacheType, 64>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, k_scale, v_scale,
        stream);
  } else if (head_size == 128) {
    return dispatch_flashinfer_decode_softcap<DType, CacheType, 128>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, k_scale, v_scale,
        stream);
  } else if (head_size == 256) {
    return dispatch_flashinfer_decode_softcap<DType, CacheType, 256>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, k_scale, v_scale,
        stream);
  } else if (head_size == 512) {
    return dispatch_flashinfer_decode_softcap<DType, CacheType, 512>(
        q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,
        request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,
        block_valid_mask, o, tmp_v, tmp_s, batch_size,
        padded_batch_size, num_qo_heads, num_kv_heads, page_size, q_stride_n,
        q_stride_h, sm_scale, window_left, logits_soft_cap, k_scale, v_scale,
        stream);
  } else {
    fprintf(stderr, "FlashInfer decode received unsupported head_size %d\n",
            head_size);
    return cudaErrorInvalidValue;
  }
}

} // namespace mistralrs_flashinfer

extern "C" void reshape_and_cache_flashinfer(
    void *key, void *value, void *key_cache, void *value_cache,
    int64_t *slot_mapping, int32_t num_tokens, int32_t num_heads,
    int32_t head_size, int32_t block_size, int32_t key_stride,
    int32_t value_stride, float k_scale, float v_scale, uint32_t dtype,
    uint32_t cache_dtype,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  if (dtype == 0 && cache_dtype == 0) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__half, __half>
        <<<grid, block, 0, stream>>>(
            static_cast<__half *>(key), static_cast<__half *>(value),
            static_cast<__half *>(key_cache),
            static_cast<__half *>(value_cache), slot_mapping, num_heads,
            head_size, block_size, key_stride, value_stride, k_scale, v_scale);
  } else if (dtype == 1 && cache_dtype == 1) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__nv_bfloat16,
                                                               __nv_bfloat16>
        <<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 *>(key),
                                     static_cast<__nv_bfloat16 *>(value),
                                     static_cast<__nv_bfloat16 *>(key_cache),
                                     static_cast<__nv_bfloat16 *>(value_cache),
                                     slot_mapping, num_heads, head_size,
                                     block_size, key_stride, value_stride,
                                     k_scale, v_scale);
  } else if (dtype == 2 && cache_dtype == 2) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<float, float>
        <<<grid, block, 0, stream>>>(
            static_cast<float *>(key), static_cast<float *>(value),
            static_cast<float *>(key_cache), static_cast<float *>(value_cache),
            slot_mapping, num_heads, head_size, block_size, key_stride,
            value_stride, k_scale, v_scale);
  } else if (cache_dtype == 3 && dtype == 0) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__half,
                                                               __nv_fp8_e4m3>
        <<<grid, block, 0, stream>>>(
            static_cast<__half *>(key), static_cast<__half *>(value),
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache), slot_mapping, num_heads,
            head_size, block_size, key_stride, value_stride, k_scale, v_scale);
  } else if (cache_dtype == 3 && dtype == 1) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<__nv_bfloat16,
                                                               __nv_fp8_e4m3>
        <<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(key),
            static_cast<__nv_bfloat16 *>(value),
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache), slot_mapping, num_heads,
            head_size, block_size, key_stride, value_stride, k_scale, v_scale);
  } else if (cache_dtype == 3 && dtype == 2) {
    mistralrs_flashinfer::reshape_and_cache_flashinfer_kernel<float,
                                                               __nv_fp8_e4m3>
        <<<grid, block, 0, stream>>>(
            static_cast<float *>(key), static_cast<float *>(value),
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache), slot_mapping, num_heads,
            head_size, block_size, key_stride, value_stride, k_scale, v_scale);
  } else {
    fprintf(stderr,
            "reshape_and_cache_flashinfer received unsupported dtype pair %u/%u\n",
            dtype, cache_dtype);
  }
}

extern "C" int32_t flashinfer_decode(
    void *q, void *key_cache, void *value_cache, const int32_t *kv_indptr,
    const int32_t *kv_indices, const int32_t *kv_last_page_len,
    const int32_t *request_indices, const int32_t *kv_tile_indices,
    const int32_t *o_indptr, const int32_t *kv_chunk_size_ptr,
    const bool *block_valid_mask, void *o, void *tmp_v, void *tmp_s,
    int32_t batch_size, int32_t padded_batch_size, int32_t num_qo_heads,
    int32_t num_kv_heads, int32_t head_size, int32_t page_size,
    int32_t q_stride_n, int32_t q_stride_h, float sm_scale, int32_t window_left,
    float logits_soft_cap, float k_scale, float v_scale, uint32_t dtype,
    uint32_t cache_dtype,
    cudaStream_t stream) {
#define CALL_FLASHINFER_DECODE(DTYPE, CACHE_DTYPE)                             \
  return mistralrs_flashinfer::dispatch_flashinfer_decode_head_dim<           \
      DTYPE, CACHE_DTYPE>(                                                     \
      q, key_cache, value_cache, kv_indptr, kv_indices, kv_last_page_len,      \
      request_indices, kv_tile_indices, o_indptr, kv_chunk_size_ptr,           \
      block_valid_mask, o, tmp_v, tmp_s, batch_size, padded_batch_size,        \
      num_qo_heads, num_kv_heads, head_size, page_size, q_stride_n, q_stride_h, \
      sm_scale, window_left, logits_soft_cap, k_scale, v_scale, stream)
  try {
    if (dtype == 0 && cache_dtype == 0) {
      CALL_FLASHINFER_DECODE(__half, __half);
    } else if (dtype == 1 && cache_dtype == 1) {
      CALL_FLASHINFER_DECODE(__nv_bfloat16, __nv_bfloat16);
    } else if (dtype == 2 && cache_dtype == 2) {
      CALL_FLASHINFER_DECODE(float, float);
    } else if (dtype == 0 && cache_dtype == 3) {
      CALL_FLASHINFER_DECODE(__half, __nv_fp8_e4m3);
    } else if (dtype == 1 && cache_dtype == 3) {
      CALL_FLASHINFER_DECODE(__nv_bfloat16, __nv_fp8_e4m3);
    } else if (dtype == 2 && cache_dtype == 3) {
      CALL_FLASHINFER_DECODE(float, __nv_fp8_e4m3);
    }
    fprintf(stderr, "FlashInfer decode received unsupported dtype pair %u/%u\n",
            dtype, cache_dtype);
  } catch (const std::exception &e) {
    fprintf(stderr, "FlashInfer decode failed: %s\n", e.what());
    return cudaErrorUnknown;
  } catch (...) {
    fprintf(stderr, "FlashInfer decode failed with unknown exception\n");
    return cudaErrorUnknown;
  }
#undef CALL_FLASHINFER_DECODE
  return cudaErrorInvalidValue;
}

extern "C" void gather_kv_cache_flashinfer(
    void *key_cache, void *value_cache, void *k_out, void *v_out,
    const int32_t *block_table, const int32_t *cu_seq_lens, int32_t num_tokens,
    int32_t num_seqs, int32_t block_size, int32_t block_table_stride,
    int32_t num_kv_heads, int32_t head_size, uint32_t out_dtype,
    uint32_t cache_dtype, float k_scale, float v_scale, cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_kv_heads * head_size, 512));

  if (out_dtype == 0 && cache_dtype == 0) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__half, __half>
        <<<grid, block, 0, stream>>>(
            static_cast<__half *>(key_cache),
            static_cast<__half *>(value_cache), static_cast<__half *>(k_out),
            static_cast<__half *>(v_out), block_table, cu_seq_lens, num_tokens,
            block_size, block_table_stride, num_kv_heads, head_size, k_scale,
            v_scale);
  } else if (out_dtype == 1 && cache_dtype == 1) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__nv_bfloat16,
                                                            __nv_bfloat16>
        <<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 *>(key_cache),
                                     static_cast<__nv_bfloat16 *>(value_cache),
                                     static_cast<__nv_bfloat16 *>(k_out),
                                     static_cast<__nv_bfloat16 *>(v_out),
                                     block_table, cu_seq_lens, num_tokens,
                                     block_size, block_table_stride,
                                     num_kv_heads, head_size, k_scale, v_scale);
  } else if (out_dtype == 2 && cache_dtype == 2) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<float, float>
        <<<grid, block, 0, stream>>>(
            static_cast<float *>(key_cache), static_cast<float *>(value_cache),
            static_cast<float *>(k_out), static_cast<float *>(v_out),
            block_table, cu_seq_lens, num_tokens, block_size,
            block_table_stride, num_kv_heads, head_size, k_scale, v_scale);
  } else if (cache_dtype == 3 && out_dtype == 0) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__nv_fp8_e4m3,
                                                            __half>
        <<<grid, block, 0, stream>>>(
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache),
            static_cast<__half *>(k_out), static_cast<__half *>(v_out),
            block_table, cu_seq_lens, num_tokens, block_size,
            block_table_stride, num_kv_heads, head_size, k_scale, v_scale);
  } else if (cache_dtype == 3 && out_dtype == 1) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__nv_fp8_e4m3,
                                                            __nv_bfloat16>
        <<<grid, block, 0, stream>>>(
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache),
            static_cast<__nv_bfloat16 *>(k_out),
            static_cast<__nv_bfloat16 *>(v_out), block_table, cu_seq_lens,
            num_tokens, block_size, block_table_stride, num_kv_heads, head_size,
            k_scale, v_scale);
  } else if (cache_dtype == 3 && out_dtype == 2) {
    mistralrs_flashinfer::gather_kv_cache_flashinfer_kernel<__nv_fp8_e4m3,
                                                            float>
        <<<grid, block, 0, stream>>>(
            static_cast<__nv_fp8_e4m3 *>(key_cache),
            static_cast<__nv_fp8_e4m3 *>(value_cache),
            static_cast<float *>(k_out), static_cast<float *>(v_out), block_table,
            cu_seq_lens, num_tokens, block_size, block_table_stride,
            num_kv_heads, head_size, k_scale, v_scale);
  } else {
    fprintf(stderr,
            "gather_kv_cache_flashinfer received unsupported dtype pair %u/%u\n",
            out_dtype, cache_dtype);
  }
}
