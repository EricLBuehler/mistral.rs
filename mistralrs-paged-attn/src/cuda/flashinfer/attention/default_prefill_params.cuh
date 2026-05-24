/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_PREFILL_PARAMS_CUH_
#define FLASHINFER_PREFILL_PARAMS_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "../page.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SinglePrefillParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  uint8_t* maybe_custom_mask;
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t qo_len;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  uint32_t head_dim;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  uint32_t partition_kv;

  __host__ SinglePrefillParams()
      : q(nullptr),
        k(nullptr),
        v(nullptr),
        maybe_custom_mask(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        group_size(),
        qo_len(0),
        kv_len(0),
        num_qo_heads(0),
        num_kv_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        k_stride_n(0),
        k_stride_h(0),
        v_stride_n(0),
        v_stride_h(0),
        head_dim(0),
        window_left(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        partition_kv(false) {}

  __host__ SinglePrefillParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* maybe_custom_mask,
                               DTypeO* o, float* lse, float* maybe_alibi_slopes,
                               uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len,
                               uint32_t kv_len, uint32_t q_stride_n, uint32_t q_stride_h,
                               uint32_t kv_stride_n, uint32_t kv_stride_h, uint32_t head_dim,
                               int32_t window_left, float logits_soft_cap, float sm_scale,
                               float rope_scale, float rope_theta)
      : q(q),
        k(k),
        v(v),
        maybe_custom_mask(maybe_custom_mask),
        o(o),
        lse(lse),
        maybe_alibi_slopes(maybe_alibi_slopes),
        group_size(num_qo_heads / num_kv_heads),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        qo_len(qo_len),
        kv_len(kv_len),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        k_stride_n(kv_stride_n),
        k_stride_h(kv_stride_h),
        v_stride_n(kv_stride_n),
        v_stride_h(kv_stride_h),
        head_dim(head_dim),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1. / rope_scale),
        rope_rcp_theta(1. / rope_theta),
        partition_kv(false) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return qo_len;
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillRaggedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* kv_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;  // maybe_q_rope_offset is only used for fused-rope attention
  IdType* maybe_k_rope_offset;  // maybe_k_rope_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t k_stride_n;
  uint32_t k_stride_h;
  uint32_t v_stride_n;
  uint32_t v_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  IdType* kv_chunk_size_ptr;
  bool* block_valid_mask;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;
  uint32_t* maybe_prefix_len_ptr;
  uint16_t* maybe_token_pos_in_items_ptr;
  uint32_t token_pos_in_items_len;
  uint16_t* maybe_max_item_len_ptr;

  __host__ BatchPrefillRaggedParams()
      : q(nullptr),
        k(nullptr),
        v(nullptr),
        maybe_custom_mask(nullptr),
        q_indptr(nullptr),
        kv_indptr(nullptr),
        maybe_mask_indptr(nullptr),
        maybe_q_rope_offset(nullptr),
        maybe_k_rope_offset(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        group_size(),
        num_qo_heads(0),
        num_kv_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        k_stride_n(0),
        k_stride_h(0),
        v_stride_n(0),
        v_stride_h(0),
        window_left(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false),
        maybe_prefix_len_ptr(nullptr),
        maybe_token_pos_in_items_ptr(nullptr),
        token_pos_in_items_len(0),
        maybe_max_item_len_ptr(nullptr) {}

  __host__ BatchPrefillRaggedParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, uint8_t* maybe_custom_mask,
                                    IdType* q_indptr, IdType* kv_indptr, IdType* maybe_mask_indptr,
                                    IdType* maybe_q_rope_offset, IdType* maybe_k_rope_offset,
                                    DTypeO* o, float* lse, float* maybe_alibi_slopes,
                                    uint32_t num_qo_heads, uint32_t num_kv_heads,
                                    uint32_t q_stride_n, uint32_t q_stride_h, uint32_t kv_stride_n,
                                    uint32_t kv_stride_h, int32_t window_left,
                                    float logits_soft_cap, float sm_scale, float rope_scale,
                                    float rope_theta)
      : q(q),
        k(k),
        v(v),
        maybe_custom_mask(maybe_custom_mask),
        q_indptr(q_indptr),
        kv_indptr(kv_indptr),
        maybe_mask_indptr(maybe_mask_indptr),
        maybe_q_rope_offset(maybe_q_rope_offset),
        maybe_k_rope_offset(maybe_k_rope_offset),
        o(o),
        lse(lse),
        maybe_alibi_slopes(maybe_alibi_slopes),
        group_size(num_qo_heads / num_kv_heads),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        k_stride_n(kv_stride_n),
        k_stride_h(kv_stride_h),
        v_stride_n(kv_stride_n),
        v_stride_h(kv_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        kv_chunk_size_ptr(nullptr),
        block_valid_mask(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false),
        maybe_prefix_len_ptr(nullptr),
        maybe_token_pos_in_items_ptr(nullptr),
        token_pos_in_items_len(0),
        maybe_max_item_len_ptr(nullptr) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }
};

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q;
  paged_kv_t<DTypeKV, IdType> paged_kv;
  uint8_t* maybe_custom_mask;
  IdType* q_indptr;
  IdType* maybe_mask_indptr;
  IdType* maybe_q_rope_offset;  // maybe_q_rope_offset is only used for fused-rope attention
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint_fastdiv group_size;
  uint32_t num_qo_heads;
  IdType q_stride_n;
  IdType q_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;

  IdType* request_indices;
  IdType* qo_tile_indices;
  IdType* kv_tile_indices;
  IdType* merge_indptr;
  IdType* o_indptr;
  bool* block_valid_mask;
  IdType* kv_chunk_size_ptr;
  uint32_t max_total_num_rows;
  uint32_t* total_num_rows;
  uint32_t padded_batch_size;
  bool partition_kv;
  uint32_t* maybe_prefix_len_ptr;
  uint16_t* maybe_token_pos_in_items_ptr;
  uint32_t token_pos_in_items_len;
  uint16_t* maybe_max_item_len_ptr;

  __host__ BatchPrefillPagedParams()
      : q(nullptr),
        paged_kv(),
        maybe_custom_mask(nullptr),
        q_indptr(nullptr),
        maybe_mask_indptr(nullptr),
        maybe_q_rope_offset(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        group_size(),
        num_qo_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        window_left(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        block_valid_mask(nullptr),
        kv_chunk_size_ptr(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false),
        maybe_prefix_len_ptr(nullptr),
        maybe_token_pos_in_items_ptr(nullptr),
        token_pos_in_items_len(0),
        maybe_max_item_len_ptr(nullptr) {}

  __host__ BatchPrefillPagedParams(DTypeQ* q, paged_kv_t<DTypeKV, IdType> paged_kv,
                                   uint8_t* maybe_custom_mask, IdType* q_indptr,
                                   IdType* maybe_mask_indptr, IdType* maybe_q_rope_offset,
                                   DTypeO* o, float* lse, float* maybe_alibi_slopes,
                                   uint32_t num_qo_heads, IdType q_stride_n, IdType q_stride_h,
                                   int32_t window_left, float logits_soft_cap, float sm_scale,
                                   float rope_scale, float rope_theta)
      : q(q),
        paged_kv(paged_kv),
        maybe_custom_mask(maybe_custom_mask),
        q_indptr(q_indptr),
        maybe_mask_indptr(maybe_mask_indptr),
        maybe_q_rope_offset(maybe_q_rope_offset),
        o(o),
        lse(lse),
        maybe_alibi_slopes(maybe_alibi_slopes),
        group_size(num_qo_heads / paged_kv.num_heads),
        num_qo_heads(num_qo_heads),
        q_stride_n(q_stride_n),
        q_stride_h(q_stride_h),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        request_indices(nullptr),
        qo_tile_indices(nullptr),
        kv_tile_indices(nullptr),
        merge_indptr(nullptr),
        o_indptr(nullptr),
        block_valid_mask(nullptr),
        kv_chunk_size_ptr(nullptr),
        max_total_num_rows(0),
        total_num_rows(nullptr),
        padded_batch_size(0),
        partition_kv(false),
        maybe_prefix_len_ptr(nullptr),
        maybe_token_pos_in_items_ptr(nullptr),
        token_pos_in_items_len(0),
        maybe_max_item_len_ptr(nullptr) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
