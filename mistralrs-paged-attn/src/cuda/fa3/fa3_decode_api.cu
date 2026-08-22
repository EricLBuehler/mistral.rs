#include "fa3_decode_api.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cuda_check.h"
#include "flash.h"

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#include <cmath>

namespace {

constexpr int kHeadDim = 256;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kMetadataThreads = 256;
constexpr int kQuantThreads = 256;

int round_up_4(int value) { return (value + 3) / 4 * 4; }

int validate_schedule(const Fa3Fp8DecodeScheduleParams *p) {
    if (p == nullptr || p->cu_seqlens_q == nullptr || p->seqused_k == nullptr ||
        p->scheduler_metadata == nullptr || p->batch_size <= 0 || p->total_q <= 0 ||
        p->total_q > p->batch_size || p->num_q_heads <= 0 || p->num_kv_heads <= 0 ||
        p->num_q_heads % p->num_kv_heads != 0 || p->head_dim != kHeadDim ||
        p->page_size <= 0 || p->max_seqlen_k <= 0 || p->num_splits <= 1 ||
        p->num_splits > 256 || p->num_sm <= 0 || p->device_id < 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

int validate_run(const Fa3Fp8DecodeParams *p) {
    const int schedule_status = p == nullptr ? cudaErrorInvalidValue
                                              : validate_schedule(&p->schedule);
    if (schedule_status != cudaSuccess) {
        return schedule_status;
    }
    if (p->q == nullptr || p->k == nullptr || p->v == nullptr || p->out == nullptr ||
        p->softmax_lse == nullptr || p->out_accum == nullptr ||
        p->softmax_lse_accum == nullptr || p->page_table == nullptr ||
        p->q_row_stride <= 0 || p->q_head_stride <= 0 || p->k_token_stride <= 0 ||
        p->k_head_stride <= 0 || p->k_page_stride <= 0 || p->v_token_stride <= 0 ||
        p->v_head_stride <= 0 || p->v_page_stride <= 0 || p->out_row_stride <= 0 ||
        p->out_head_stride <= 0 || p->page_table_batch_stride <= 0 ||
        p->q_descale_batch_stride < 0 || p->q_descale_head_stride < 0 ||
        p->k_descale_batch_stride < 0 || p->k_descale_head_stride < 0 ||
        p->v_descale_batch_stride < 0 || p->v_descale_head_stride < 0 ||
        p->num_pages <= 0 || p->max_pages_per_sequence <= 0 ||
        int64_t(p->max_pages_per_sequence) * p->schedule.page_size <
            p->schedule.max_seqlen_k ||
        !std::isfinite(p->softmax_scale) || p->softmax_scale <= 0.0f ||
        p->scheduler_metadata_prepared == 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

void fill_schedule(const Fa3Fp8DecodeScheduleParams &src, Flash_fwd_params &dst) {
    dst = {};
    dst.b = src.batch_size;
    dst.b_k = src.batch_size;
    dst.seqlen_q = 1;
    dst.seqlen_k = src.max_seqlen_k;
    dst.seqlen_q_rounded = kBlockM;
    dst.seqlen_k_rounded = (dst.seqlen_k + kBlockN - 1) / kBlockN * kBlockN;
    dst.total_q = src.total_q;
    dst.h = src.num_q_heads;
    dst.h_k = src.num_kv_heads;
    dst.d = src.head_dim;
    dst.d_rounded = kHeadDim;
    dst.dv = src.head_dim;
    dst.dv_rounded = kHeadDim;

    dst.p_dropout = 1.0f;
    dst.rp_dropout = 1.0f;
    dst.p_dropout_in_uint8_t = 255;
    dst.window_size_left = -1;
    dst.window_size_right = -1;
    dst.is_e4m3 = true;
    dst.is_bf16 = true;
    dst.is_causal = false;
    dst.is_local = false;
    dst.arch = 90;
    dst.device_id = src.device_id;

    dst.cu_seqlens_q = const_cast<int32_t *>(src.cu_seqlens_q);
    dst.seqused_k = const_cast<int32_t *>(src.seqused_k);
    dst.page_size = src.page_size;
    dst.pagedkv_tma = false;

    dst.num_splits = src.num_splits;
    dst.pack_gqa = true;
    dst.prepare_varlen_pdl = false;
    dst.varlen_sort_batches = false;
    dst.head_swizzle = false;
    dst.skip_scheduler_metadata_computation = true;
    dst.cp_world_size = 1;
    dst.cp_rank = 0;
    dst.num_sm = src.num_sm;

    const int batch_rounded = round_up_4(src.batch_size);
    dst.prepare_seqlen_q_ptr = src.scheduler_metadata;
    dst.num_splits_dynamic_ptr = src.scheduler_metadata + batch_rounded;
    dst.tile_count_semaphore = src.scheduler_metadata + 2 * batch_rounded;
    dst.tile_count_semaphore_offset = 2 * batch_rounded;
}

void fill_run(const Fa3Fp8DecodeParams &src, Flash_fwd_params &dst) {
    fill_schedule(src.schedule, dst);
    dst.q_ptr = const_cast<void *>(src.q);
    dst.k_ptr = const_cast<void *>(src.k);
    dst.v_ptr = const_cast<void *>(src.v);
    dst.o_ptr = src.out;
    dst.softmax_lse_ptr = src.softmax_lse;
    dst.oaccum_ptr = src.out_accum;
    dst.softmax_lseaccum_ptr = src.softmax_lse_accum;

    dst.q_row_stride = src.q_row_stride;
    dst.q_head_stride = src.q_head_stride;
    dst.k_row_stride = src.k_token_stride;
    dst.k_head_stride = src.k_head_stride;
    dst.k_batch_stride = src.k_page_stride;
    dst.v_row_stride = src.v_token_stride;
    dst.v_head_stride = src.v_head_stride;
    dst.v_batch_stride = src.v_page_stride;
    dst.v_dim_stride = 1;
    dst.o_row_stride = src.out_row_stride;
    dst.o_head_stride = src.out_head_stride;

    dst.total_k = src.num_pages * src.schedule.page_size;
    dst.scale_softmax = src.softmax_scale;
    dst.page_table = const_cast<int32_t *>(src.page_table);
    dst.page_table_batch_stride = src.page_table_batch_stride;
    dst.num_pages = src.num_pages;

    dst.q_descale_ptr = const_cast<float *>(src.q_descale);
    dst.k_descale_ptr = const_cast<float *>(src.k_descale);
    dst.v_descale_ptr = const_cast<float *>(src.v_descale);
    dst.q_descale_batch_stride = src.q_descale_batch_stride;
    dst.q_descale_head_stride = src.q_descale_head_stride;
    dst.k_descale_batch_stride = src.k_descale_batch_stride;
    dst.k_descale_head_stride = src.k_descale_head_stride;
    dst.v_descale_batch_stride = src.v_descale_batch_stride;
    dst.v_descale_head_stride = src.v_descale_head_stride;

    dst.oaccum_split_stride =
        int64_t(src.schedule.num_q_heads) * src.schedule.total_q * src.schedule.head_dim;
    dst.oaccum_head_stride = int64_t(src.schedule.total_q) * src.schedule.head_dim;
    dst.oaccum_row_stride = src.schedule.head_dim;
    dst.lseaccum_split_stride =
        int64_t(src.schedule.num_q_heads) * src.schedule.total_q;
    dst.lseaccum_head_stride = src.schedule.total_q;
    dst.is_fp32 = false;
}

__global__ void materialize_paged_metadata_kernel(
    const int32_t *__restrict__ paged_kv_indptr,
    const int32_t *__restrict__ paged_kv_indices,
    const int32_t *__restrict__ paged_kv_last_page_len,
    int32_t *__restrict__ page_table, int32_t *__restrict__ seqused_k,
    int32_t page_table_batch_stride, int32_t page_size) {
    const int32_t batch = blockIdx.x;
    const int32_t begin = paged_kv_indptr[batch];
    const int32_t end = paged_kv_indptr[batch + 1];
    const int32_t page_count = end - begin;
    int32_t *row = page_table + int64_t(batch) * page_table_batch_stride;
    for (int32_t page = threadIdx.x; page < page_count; page += blockDim.x) {
        row[page] = paged_kv_indices[begin + page];
    }
    if (threadIdx.x == 0) {
        seqused_k[batch] = page_count == 0
                                ? 0
                                : (page_count - 1) * page_size +
                                      paged_kv_last_page_len[batch];
    }
}

struct alignas(16) Bf16Vector16 {
    __nv_bfloat162 values[8];
};

struct alignas(16) Fp8Vector16 {
    __nv_fp8x2_storage_t values[8];
};

template <int Mode>
__global__ void bf16_to_e4m3_static_kernel(const __nv_bfloat16 *__restrict__ input,
                                            __nv_fp8_e4m3 *__restrict__ output,
                                            int32_t columns, int64_t input_row_stride,
                                            int64_t output_row_stride,
                                            const float *__restrict__ descale) {
    const int32_t row = blockIdx.x;
    const float inv_descale = 1.0f / *descale;
    input += int64_t(row) * input_row_stride;
    output += int64_t(row) * output_row_stride;
    if constexpr (Mode == 2) {
        const auto *input_vectors = reinterpret_cast<const Bf16Vector16 *>(input);
        auto *output_vectors = reinterpret_cast<Fp8Vector16 *>(output);
        for (int32_t vector = threadIdx.x; vector < columns / 16; vector += blockDim.x) {
            const Bf16Vector16 source = input_vectors[vector];
            Fp8Vector16 destination;
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                float2 value = __bfloat1622float2(source.values[pair]);
                value.x *= inv_descale;
                value.y *= inv_descale;
                destination.values[pair] =
                    __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
            }
            output_vectors[vector] = destination;
        }
    } else if constexpr (Mode == 1) {
        const auto *input_pairs = reinterpret_cast<const __nv_bfloat162 *>(input);
        auto *output_pairs = reinterpret_cast<__nv_fp8x2_storage_t *>(output);
        for (int32_t pair = threadIdx.x; pair < columns / 2; pair += blockDim.x) {
            float2 value = __bfloat1622float2(input_pairs[pair]);
            value.x *= inv_descale;
            value.y *= inv_descale;
            output_pairs[pair] = __nv_cvt_float2_to_fp8x2(value, __NV_SATFINITE, __NV_E4M3);
        }
    } else {
        for (int32_t column = threadIdx.x; column < columns; column += blockDim.x) {
            const float value = __bfloat162float(input[column]) * inv_descale;
            output[column].__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

} // namespace

extern "C" size_t fa3_fp8_decode_scheduler_metadata_i32(int32_t batch_size,
                                                          int32_t num_splits) {
    if (batch_size <= 0 || num_splits <= 1) {
        return 0;
    }
    return size_t(2 * round_up_4(batch_size) + 1);
}

extern "C" size_t fa3_fp8_decode_out_accum_f32(const Fa3Fp8DecodeParams *params) {
    if (params == nullptr || params->schedule.num_splits <= 0 ||
        params->schedule.num_q_heads <= 0 || params->schedule.total_q <= 0 ||
        params->schedule.head_dim <= 0) {
        return 0;
    }
    return size_t(params->schedule.num_splits) * params->schedule.num_q_heads *
           params->schedule.total_q * params->schedule.head_dim;
}

extern "C" size_t fa3_fp8_decode_lse_accum_f32(const Fa3Fp8DecodeParams *params) {
    if (params == nullptr || params->schedule.num_splits <= 0 ||
        params->schedule.num_q_heads <= 0 || params->schedule.total_q <= 0) {
        return 0;
    }
    return size_t(params->schedule.num_splits) * params->schedule.num_q_heads *
           params->schedule.total_q;
}

extern "C" int fa3_fp8_decode_materialize_paged_metadata(
    const int32_t *paged_kv_indptr, const int32_t *paged_kv_indices,
    const int32_t *paged_kv_last_page_len, int32_t *page_table,
    int32_t *seqused_k, int32_t batch_size, int32_t page_table_batch_stride,
    int32_t page_size, cudaStream_t stream) {
    if (paged_kv_indptr == nullptr || paged_kv_indices == nullptr ||
        paged_kv_last_page_len == nullptr || page_table == nullptr ||
        seqused_k == nullptr || batch_size <= 0 ||
        page_table_batch_stride <= 0 || page_size <= 0) {
        return cudaErrorInvalidValue;
    }
    materialize_paged_metadata_kernel<<<batch_size, kMetadataThreads, 0, stream>>>(
        paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, page_table,
        seqused_k, page_table_batch_stride, page_size);
    return cudaPeekAtLastError();
}

extern "C" int fa3_fp8_decode_prepare(const Fa3Fp8DecodeScheduleParams *params,
                                        cudaStream_t stream) {
    const int validation = validate_schedule(params);
    if (validation != cudaSuccess) {
        return validation;
    }
    Flash_fwd_params upstream{};
    fill_schedule(*params, upstream);
    upstream.skip_scheduler_metadata_computation = false;
    try {
        prepare_varlen_num_blocks(upstream, stream, true, kBlockM, kBlockN, false);
    } catch (const FlashAttentionCudaError &error) {
        return error.status;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaPeekAtLastError();
}

extern "C" int fa3_fp8_decode_run(const Fa3Fp8DecodeParams *params,
                                    cudaStream_t stream) {
    const int validation = validate_run(params);
    if (validation != cudaSuccess) {
        return validation;
    }
    Flash_fwd_params upstream{};
    fill_run(*params, upstream);
    try {
        run_mha_fwd_<90, cutlass::float_e4m3_t, kHeadDim, kHeadDim, true, true, false, true>(
            upstream, stream);
        run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(upstream, stream, true);
    } catch (const FlashAttentionCudaError &error) {
        return error.status;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaPeekAtLastError();
}

extern "C" int fa3_bf16_to_e4m3_static(const void *input, void *output,
                                         int32_t rows, int32_t columns,
                                         int64_t input_row_stride,
                                         int64_t output_row_stride,
                                         const float *descale,
                                         cudaStream_t stream) {
    if (input == nullptr || output == nullptr || descale == nullptr || rows <= 0 ||
        columns <= 0 || input_row_stride < columns || output_row_stride < columns) {
        return cudaErrorInvalidValue;
    }
    const bool vector_aligned = reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
                                reinterpret_cast<uintptr_t>(output) % 16 == 0 &&
                                columns % 16 == 0 && input_row_stride % 8 == 0 &&
                                output_row_stride % 16 == 0;
    if (vector_aligned) {
        bf16_to_e4m3_static_kernel<2><<<rows, kQuantThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output),
            columns, input_row_stride, output_row_stride, descale);
    } else if (columns % 2 == 0 && input_row_stride % 2 == 0 && output_row_stride % 2 == 0) {
        bf16_to_e4m3_static_kernel<1><<<rows, kQuantThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output),
            columns, input_row_stride, output_row_stride, descale);
    } else {
        bf16_to_e4m3_static_kernel<0><<<rows, kQuantThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input), static_cast<__nv_fp8_e4m3 *>(output),
            columns, input_row_stride, output_row_stride, descale);
    }
    return cudaPeekAtLastError();
}
