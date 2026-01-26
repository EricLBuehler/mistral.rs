#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// FlashInfer MLA decode requires __grid_constant__ which is only available on compute >= 70
// For older architectures, we provide a stub that prints an error message
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define FLASHINFER_MLA_UNSUPPORTED 1
#endif

// Check at host compile time using the list of target architectures
// __CUDA_ARCH_LIST__ contains all architectures being compiled for
#ifndef FLASHINFER_MLA_UNSUPPORTED
// We need to guard the include because the headers use __grid_constant__
#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/variants.cuh"

using namespace flashinfer;

namespace {
constexpr uint32_t HEAD_DIM_CKV = 512;
constexpr uint32_t HEAD_DIM_KPE = 64;

template <typename DType>
void run_mla_decode(
    void *q_nope, void *q_pe, void *ckv_cache, void *kpe_cache,
    const int32_t *kv_indptr, const int32_t *kv_indices,
    const int32_t *kv_last_page_len, void *o, int32_t batch_size,
    int32_t num_qo_heads, int32_t page_size, float sm_scale,
    int32_t window_left, float logits_soft_cap, float rope_scale,
    float rope_theta, const int32_t *request_indices,
    const int32_t *kv_tile_indices, const int32_t *o_indptr,
    const int32_t *kv_chunk_size_ptr, cudaStream_t stream) {
  using Params = BatchDecodeParamsMLA<DType, DType, DType, int32_t>;
  using AttentionVariant = DefaultAttention<false, false, false, false>;

  paged_kv_mla_t<DType, int32_t> paged_kv(
      page_size, HEAD_DIM_CKV, HEAD_DIM_KPE, batch_size,
      static_cast<DType *>(ckv_cache), static_cast<DType *>(kpe_cache),
      const_cast<int32_t *>(kv_indices), const_cast<int32_t *>(kv_indptr),
      const_cast<int32_t *>(kv_last_page_len));

  Params params(static_cast<DType *>(q_nope), static_cast<DType *>(q_pe),
                /*q_rope_offset=*/nullptr, paged_kv,
                static_cast<DType *>(o), /*lse=*/nullptr, num_qo_heads,
                window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta);
  params.request_indices = const_cast<int32_t *>(request_indices);
  params.kv_tile_indices = const_cast<int32_t *>(kv_tile_indices);
  params.o_indptr = const_cast<int32_t *>(o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t *>(kv_chunk_size_ptr);
  params.padded_batch_size = batch_size;

  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatchedMLA<HEAD_DIM_CKV, HEAD_DIM_KPE,
                                               AttentionVariant, Params>(
          params, nullptr, nullptr, /*enable_pdl=*/false, stream);
  if (status != cudaSuccess) {
    fprintf(stderr,
            "FlashInfer MLA decode failed: %s\n",
            cudaGetErrorString(status));
  }
}
}  // namespace

extern "C" void flashinfer_mla_decode(
    void *q_nope, void *q_pe, void *ckv_cache, void *kpe_cache,
    const int32_t *kv_indptr, const int32_t *kv_indices,
    const int32_t *kv_last_page_len, void *o, int32_t batch_size,
    int32_t num_qo_heads, int32_t page_size, float sm_scale,
    int32_t window_left, float logits_soft_cap, float rope_scale,
    float rope_theta, const int32_t *request_indices,
    const int32_t *kv_tile_indices, const int32_t *o_indptr,
    const int32_t *kv_chunk_size_ptr, uint32_t dtype,
    cudaStream_t stream) {
  if (dtype == 0) {
    run_mla_decode<__half>(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr,
                           kv_indices, kv_last_page_len, o, batch_size,
                           num_qo_heads, page_size, sm_scale, window_left,
                           logits_soft_cap, rope_scale, rope_theta,
                           request_indices, kv_tile_indices, o_indptr,
                           kv_chunk_size_ptr, stream);
  } else if (dtype == 1) {
    run_mla_decode<__nv_bfloat16>(q_nope, q_pe, ckv_cache, kpe_cache,
                                  kv_indptr, kv_indices, kv_last_page_len, o,
                                  batch_size, num_qo_heads, page_size, sm_scale,
                                  window_left, logits_soft_cap, rope_scale,
                                  rope_theta, request_indices, kv_tile_indices,
                                  o_indptr, kv_chunk_size_ptr, stream);
  } else if (dtype == 2) {
    run_mla_decode<float>(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr,
                          kv_indices, kv_last_page_len, o, batch_size,
                          num_qo_heads, page_size, sm_scale, window_left,
                          logits_soft_cap, rope_scale, rope_theta,
                          request_indices, kv_tile_indices, o_indptr,
                          kv_chunk_size_ptr, stream);
  } else {
    fprintf(stderr, "FlashInfer MLA decode received unsupported dtype %u\n", dtype);
  }
}

#else

extern "C" void flashinfer_mla_decode(
    void *q_nope, void *q_pe, void *ckv_cache, void *kpe_cache,
    const int32_t *kv_indptr, const int32_t *kv_indices,
    const int32_t *kv_last_page_len, void *o, int32_t batch_size,
    int32_t num_qo_heads, int32_t page_size, float sm_scale,
    int32_t window_left, float logits_soft_cap, float rope_scale,
    float rope_theta, const int32_t *request_indices,
    const int32_t *kv_tile_indices, const int32_t *o_indptr,
    const int32_t *kv_chunk_size_ptr, uint32_t dtype,
    cudaStream_t stream) {
  (void)q_nope; (void)q_pe; (void)ckv_cache; (void)kpe_cache;
  (void)kv_indptr; (void)kv_indices; (void)kv_last_page_len; (void)o;
  (void)batch_size; (void)num_qo_heads; (void)page_size; (void)sm_scale;
  (void)window_left; (void)logits_soft_cap; (void)rope_scale; (void)rope_theta;
  (void)request_indices; (void)kv_tile_indices; (void)o_indptr;
  (void)kv_chunk_size_ptr; (void)dtype; (void)stream;
  fprintf(stderr, "ERROR: flashinfer_mla_decode requires GPU with compute capability >= 7.0 (Volta+)\n");
}

#endif
