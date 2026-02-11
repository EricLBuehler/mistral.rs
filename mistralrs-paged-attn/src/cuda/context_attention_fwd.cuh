/*
 * Context attention forward kernel for prefix caching.
 *
 * When a prefix cache hit occurs, only new tokens need to be computed.
 * Each query token attends to:
 *   Phase 1: All cached context tokens (read from paged KV cache via block table)
 *   Phase 2: Preceding new tokens (read from input K/V tensors, with causal mask)
 *
 * This is essentially paged_attention generalized for multiple query tokens per
 * sequence, following vLLM's context_attention_fwd (prefix_prefill) algorithm.
 *
 * Grid: (num_heads, total_new_tokens)
 * Each thread block handles one query token for one head.
 */

#include <stdint.h>
#include <stdio.h>
#include <float.h>

#include "attention/attention_dtypes.h"
#include "attention/attention_utils.cuh"

#ifndef USE_ROCM
#include "quantization/fp8/nvidia/quant_utils.cuh"
#else
#include "quantization/fp8/amd/quant_utils.cuh"
#endif

#include <algorithm>

#ifndef CTX_ATTN_CUDA_CHECK
#define CTX_ATTN_CUDA_CHECK(call)                                              \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(err);                                                               \
    }                                                                          \
  } while (0)
#endif

#ifndef USE_ROCM
#define CTX_WARP_SIZE 32
#else
#define CTX_WARP_SIZE warpSize
#endif
#define CTX_MAX(a, b) ((a) > (b) ? (a) : (b))
#define CTX_MIN(a, b) ((a) < (b) ? (a) : (b))
#define CTX_DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace vllm {

/// Main context attention kernel.
///
/// Grid: (num_heads, total_new_tokens)
/// Block: (NUM_THREADS)
///
/// For each query token, this kernel:
/// 1. Computes QK dot products with all context tokens (from paged cache)
/// 2. Computes QK dot products with preceding new tokens (from input, causal)
/// 3. Computes softmax over all logits
/// 4. Accumulates weighted V from both context cache and new input
/// 5. Writes the output
template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType kv_dt,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void context_attention_fwd_kernel(
    scalar_t *__restrict__ out,      // [total_new_tokens, num_heads, head_size]
    const scalar_t *__restrict__ q,  // [total_new_tokens, num_heads, head_size]
    const scalar_t *__restrict__ k,  // [total_new_tokens, num_kv_heads, head_size]
    const scalar_t *__restrict__ v,  // [total_new_tokens, num_kv_heads, head_size]
    const cache_t *__restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t *__restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const uint32_t *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens,      // [num_seqs] - cached tokens per seq
    const int *__restrict__ query_lens,        // [num_seqs] - new tokens per seq
    const int *__restrict__ query_start_locs,  // [num_seqs+1] - cumulative start offsets
    const int *__restrict__ seq_ids,           // [total_new_tokens] - batch index per token
    const int max_num_blocks_per_seq,
    const int kv_block_stride,
    const int kv_head_stride,
    const float *k_scale,
    const float *v_scale,
    const int sliding_window,             // 0 = disabled
    const float *__restrict__ sinks       // [num_heads] or nullptr
) {
    const int head_idx = blockIdx.x;
    const int token_flat_idx = blockIdx.y;  // flat index into total_new_tokens
    const int num_heads = gridDim.x;

    // Look up which sequence this token belongs to and its local position
    const int batch_idx = seq_ids[token_flat_idx];
    const int q_start = query_start_locs[batch_idx];
    const int q_local_idx = token_flat_idx - q_start;  // position within new tokens
    const int ctx_len = context_lens[batch_idx];
    const int q_len = query_lens[batch_idx];

    // Total KV tokens this query attends to:
    // All context tokens + new tokens [0..q_local_idx] (causal)
    const int total_kv_len = ctx_len + q_local_idx + 1;

    if (q_local_idx >= q_len) return;

    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // Thread decomposition (same as paged_attention_v1)
    constexpr int THREAD_GROUP_SIZE = CTX_MAX(CTX_WARP_SIZE / BLOCK_SIZE, 1);
    constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
    constexpr int NUM_TOKENS_PER_THREAD_GROUP =
        CTX_DIVIDE_ROUND_UP(BLOCK_SIZE, CTX_WARP_SIZE);
    constexpr int NUM_WARPS = NUM_THREADS / CTX_WARP_SIZE;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / CTX_WARP_SIZE;
    const int lane = thread_idx % CTX_WARP_SIZE;

    constexpr int VEC_SIZE = CTX_MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
    using K_vec = typename vllm::Vec<scalar_t, VEC_SIZE>::Type;
    using Q_vec = typename vllm::Vec<scalar_t, VEC_SIZE>::Type;

    constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
    constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

    const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
    const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

    // Load query into registers via shared memory
    const scalar_t *q_ptr = q + token_flat_idx * num_heads * HEAD_SIZE
                              + head_idx * HEAD_SIZE;
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
         i += NUM_THREAD_GROUPS) {
        const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
        q_vecs[thread_group_offset][i] =
            *reinterpret_cast<const Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
    }
    __syncthreads();

    // Shared memory for logits and reduction
    extern __shared__ char shared_mem[];
    float *logits = reinterpret_cast<float *>(shared_mem);
    __shared__ float red_smem[2 * NUM_WARPS];

    constexpr int x = 16 / sizeof(cache_t);
    float qk_max = -FLT_MAX;

    // ===== PHASE 1: Context tokens from paged KV cache =====
    const int num_context_blocks = CTX_DIVIDE_ROUND_UP(ctx_len, BLOCK_SIZE);
    const uint32_t *block_table = block_tables + batch_idx * max_num_blocks_per_seq;

    for (int block_idx = warp_idx; block_idx < num_context_blocks;
         block_idx += NUM_WARPS) {
        const int64_t physical_block_number =
            static_cast<int64_t>(block_table[block_idx]);

        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset =
                (thread_group_idx + i * CTX_WARP_SIZE) % BLOCK_SIZE;
            const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
            K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
            for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                const cache_t *k_ptr_base =
                    k_cache + physical_block_number * kv_block_stride +
                    kv_head_idx * kv_head_stride + physical_block_offset * x;
                const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
                const int offset1 = (vec_idx * VEC_SIZE) / x;
                const int offset2 = (vec_idx * VEC_SIZE) % x;

                if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
                    k_vecs[j] = *reinterpret_cast<const K_vec *>(
                        k_ptr_base + offset1 * BLOCK_SIZE * x + offset2);
                } else {
                    using Cache_K_vec = typename vllm::Vec<cache_t, VEC_SIZE>::Type;
                    Cache_K_vec fp8_k_vec = *reinterpret_cast<const Cache_K_vec *>(
                        k_ptr_base + offset1 * BLOCK_SIZE * x + offset2);
                    k_vecs[j] = vllm::fp8::scaled_convert<K_vec, Cache_K_vec, kv_dt>(
                        fp8_k_vec, *k_scale);
                }
            }

            float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                                   q_vecs[thread_group_offset], k_vecs);

            if (thread_group_offset == 0) {
                const bool mask = token_idx >= ctx_len;
                // Apply sliding window mask on context tokens
                bool sw_mask = false;
                if (sliding_window > 0) {
                    // Global position of this query token
                    const int q_global_pos = ctx_len + q_local_idx;
                    sw_mask = (q_global_pos - token_idx) >= sliding_window;
                }
                const bool final_mask = mask || sw_mask;
                logits[token_idx] = final_mask ? 0.f : qk;
                qk_max = final_mask ? qk_max : fmaxf(qk_max, qk);
            }
        }
    }

    // ===== PHASE 2: New tokens from input K/V (causal) =====
    // New tokens: indices [0..q_local_idx] in the input arrays for this batch.
    // These are at positions [q_start..q_start + q_local_idx] in the flat arrays.
    const int num_new_kv = q_local_idx + 1;  // causal: attend to self and before
    const int new_kv_blocks = CTX_DIVIDE_ROUND_UP(num_new_kv, BLOCK_SIZE);

    for (int block_idx = warp_idx; block_idx < new_kv_blocks;
         block_idx += NUM_WARPS) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int local_offset =
                (thread_group_idx + i * CTX_WARP_SIZE) % BLOCK_SIZE;
            const int new_token_idx = block_idx * BLOCK_SIZE + local_offset;

            K_vec k_vecs[NUM_VECS_PER_THREAD];
            if (new_token_idx < num_new_kv) {
                // Load K from input tensor
                const int flat_kv_idx = q_start + new_token_idx;
                const scalar_t *k_input_ptr =
                    k + flat_kv_idx * num_kv_heads * HEAD_SIZE
                      + kv_head_idx * HEAD_SIZE;

#pragma unroll
                for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                    const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
                    k_vecs[j] = *reinterpret_cast<const Q_vec *>(
                        k_input_ptr + vec_idx * VEC_SIZE);
                }
            } else {
                // Zero out for masking
#pragma unroll
                for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
                    memset(&k_vecs[j], 0, sizeof(K_vec));
                }
            }

            float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                                   q_vecs[thread_group_offset], k_vecs);

            if (thread_group_offset == 0) {
                const bool mask = new_token_idx >= num_new_kv;
                // Store in logits array after context tokens
                const int logit_idx = ctx_len + new_token_idx;
                logits[logit_idx] = mask ? 0.f : qk;
                qk_max = mask ? qk_max : fmaxf(qk_max, qk);
            }
        }
    }

    // ===== Softmax =====
    // Reduce qk_max across warps
#pragma unroll
    for (int mask = CTX_WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
        qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }
    if (lane == 0) {
        red_smem[warp_idx] = qk_max;
    }
    __syncthreads();
    qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
    }
    qk_max = VLLM_SHFL_SYNC(qk_max, 0);

    // Include sink in max
    if (sinks != nullptr) {
        qk_max = fmaxf(qk_max, sinks[head_idx]);
    }

    // Compute exp and sum
    float exp_sum = 0.f;
    for (int i = thread_idx; i < total_kv_len; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }

    // Block-level sum reduction
    // Warp-level reduction
#pragma unroll
    for (int mask = CTX_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        exp_sum += VLLM_SHFL_XOR_SYNC(exp_sum, mask);
    }
    if (lane == 0) {
        red_smem[warp_idx] = exp_sum;
    }
    __syncthreads();
    if (lane < NUM_WARPS) {
        exp_sum = red_smem[lane];
    }
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        exp_sum += VLLM_SHFL_XOR_SYNC(exp_sum, mask);
    }
    exp_sum = VLLM_SHFL_SYNC(exp_sum, 0);

    // Include sink in exp sum
    if (sinks != nullptr) {
        exp_sum += __expf(sinks[head_idx] - qk_max);
    }

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < total_kv_len; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // ===== V accumulation: Phase 1 (context from cache) =====
    constexpr int V_VEC_SIZE = CTX_MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
    using V_vec = typename vllm::Vec<scalar_t, V_VEC_SIZE>::Type;
    using L_vec = typename vllm::Vec<scalar_t, V_VEC_SIZE>::Type;
    using Float_L_vec = typename vllm::FloatVec<L_vec>::Type;

    constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
    constexpr int NUM_ROWS_PER_ITER = CTX_WARP_SIZE / NUM_V_VECS_PER_ROW;
    constexpr int NUM_ROWS_PER_THREAD =
        CTX_DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

    float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        accs[i] = 0.f;
    }

    scalar_t zero_value;
    vllm::zero(zero_value);

    // V from context cache
    for (int block_idx = warp_idx; block_idx < num_context_blocks;
         block_idx += NUM_WARPS) {
        const int64_t physical_block_number =
            static_cast<int64_t>(block_table[block_idx]);
        const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        L_vec logits_vec;
        vllm::from_float(logits_vec,
                   *reinterpret_cast<Float_L_vec *>(logits + token_idx));

        const cache_t *v_ptr = v_cache + physical_block_number * kv_block_stride +
                               kv_head_idx * kv_head_stride;
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
            if (row_idx < HEAD_SIZE) {
                const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
                V_vec v_vec;

                if constexpr (kv_dt == vllm::Fp8KVCacheDataType::kAuto) {
                    v_vec = *reinterpret_cast<const V_vec *>(v_ptr + offset);
                } else {
                    using Cache_V_vec = typename vllm::Vec<cache_t, V_VEC_SIZE>::Type;
                    Cache_V_vec fp8_v_vec =
                        *reinterpret_cast<const Cache_V_vec *>(v_ptr + offset);
                    v_vec = vllm::fp8::scaled_convert<V_vec, Cache_V_vec, kv_dt>(
                        fp8_v_vec, *v_scale);
                }
                if (block_idx == num_context_blocks - 1) {
                    scalar_t *v_vec_ptr = reinterpret_cast<scalar_t *>(&v_vec);
#pragma unroll
                    for (int j = 0; j < V_VEC_SIZE; j++) {
                        v_vec_ptr[j] =
                            token_idx + j < ctx_len ? v_vec_ptr[j] : zero_value;
                    }
                }
                accs[i] += vllm::dot(logits_vec, v_vec);
            }
        }
    }

    // ===== V accumulation: Phase 2 (new tokens from input) =====
    // We process new tokens in blocks of BLOCK_SIZE for alignment with logits_vec
    for (int block_idx = warp_idx; block_idx < new_kv_blocks;
         block_idx += NUM_WARPS) {
        const int base_new_idx = block_idx * BLOCK_SIZE;
        const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
        const int new_token_idx = base_new_idx + physical_block_offset;

        // Load logits for this block of new tokens (stored after context)
        const int logit_base = ctx_len + base_new_idx;
        L_vec logits_vec;
        vllm::from_float(logits_vec,
                   *reinterpret_cast<Float_L_vec *>(logits + logit_base + physical_block_offset));

#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
            if (row_idx < HEAD_SIZE) {
                // Build V_vec from input V tensor for this block of tokens
                V_vec v_vec;
                scalar_t *v_vec_ptr = reinterpret_cast<scalar_t *>(&v_vec);
#pragma unroll
                for (int j = 0; j < V_VEC_SIZE; j++) {
                    const int tidx = new_token_idx + j;
                    if (tidx < num_new_kv) {
                        const int flat_idx = q_start + tidx;
                        v_vec_ptr[j] = v[flat_idx * num_kv_heads * HEAD_SIZE
                                         + kv_head_idx * HEAD_SIZE + row_idx];
                    } else {
                        v_vec_ptr[j] = zero_value;
                    }
                }
                accs[i] += vllm::dot(logits_vec, v_vec);
            }
        }
    }

    // ===== Warp reduction and output =====
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        float acc = accs[i];
#pragma unroll
        for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
            acc += VLLM_SHFL_XOR_SYNC(acc, mask);
        }
        accs[i] = acc;
    }

    __syncthreads();

    // Cross-warp reduction
    float *out_smem = reinterpret_cast<float *>(shared_mem);
#pragma unroll
    for (int i = NUM_WARPS; i > 1; i /= 2) {
        int mid = i / 2;
        if (warp_idx >= mid && warp_idx < i) {
            float *dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
            for (int j = 0; j < NUM_ROWS_PER_THREAD; j++) {
                const int row_idx = lane / NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER;
                if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
                    dst[row_idx] = accs[j];
                }
            }
        }
        __syncthreads();
        if (warp_idx < mid) {
            const float *src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
            for (int j = 0; j < NUM_ROWS_PER_THREAD; j++) {
                const int row_idx = lane / NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER;
                if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
                    accs[j] += src[row_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (warp_idx == 0) {
        scalar_t *out_ptr =
            out + token_flat_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
            if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
                vllm::from_float(*(out_ptr + row_idx), accs[i]);
            }
        }
    }
}

} // namespace vllm

// ===== Launcher =====

#define LAUNCH_CTX_ATTN(HEAD_SIZE)                                              \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                         \
      ((void *)vllm::context_attention_fwd_kernel<T, CACHE_T, KV_DT,           \
                                                   HEAD_SIZE, BLOCK_SIZE,       \
                                                   NUM_THREADS>),               \
      shared_mem_size);                                                         \
  vllm::context_attention_fwd_kernel<T, CACHE_T, KV_DT, HEAD_SIZE,             \
                                     BLOCK_SIZE, NUM_THREADS>                   \
      <<<grid, block, shared_mem_size, stream>>>(                               \
          reinterpret_cast<T *>(out), reinterpret_cast<const T *>(query),       \
          reinterpret_cast<const T *>(key),                                     \
          reinterpret_cast<const T *>(value),                                   \
          reinterpret_cast<const CACHE_T *>(key_cache),                         \
          reinterpret_cast<const CACHE_T *>(value_cache), num_kv_heads, scale,  \
          block_tables, context_lens, query_lens, query_start_locs, seq_ids,    \
          max_num_blocks_per_seq, kv_block_stride, kv_head_stride,              \
          k_scale, v_scale, sliding_window, sinks);

template <typename T, typename CACHE_T, vllm::Fp8KVCacheDataType KV_DT,
          int BLOCK_SIZE, int NUM_THREADS = 128>
inline void context_attention_fwd_launcher(
    void *out, void *query, void *key, void *value,
    void *key_cache, void *value_cache,
    int num_kv_heads, float scale,
    uint32_t *block_tables, int *context_lens, int *query_lens,
    int *query_start_locs, int *seq_ids,
    int max_num_blocks_per_seq,
    int total_new_tokens, int num_heads, int head_size,
    int kv_block_stride, int kv_head_stride,
    cudaStream_t stream,
    const float *k_scale, const float *v_scale,
    int sliding_window, const float *sinks,
    int max_total_kv_len) {

    constexpr int NUM_WARPS = NUM_THREADS / CTX_WARP_SIZE;
    // Shared memory: max(logits, outputs)
    // logits: max_total_kv_len floats
    // outputs: (NUM_WARPS / 2) * head_size floats
    int logits_size = max_total_kv_len * sizeof(float);
    int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
    int shared_mem_size = std::max(logits_size, outputs_size);

    dim3 grid(num_heads, total_new_tokens);
    dim3 block(NUM_THREADS);

    switch (head_size) {
    case 64:
        LAUNCH_CTX_ATTN(64);
        break;
    case 80:
        LAUNCH_CTX_ATTN(80);
        break;
    case 96:
        LAUNCH_CTX_ATTN(96);
        break;
    case 112:
        LAUNCH_CTX_ATTN(112);
        break;
    case 128:
        LAUNCH_CTX_ATTN(128);
        break;
    case 192:
        LAUNCH_CTX_ATTN(192);
        break;
    case 256:
        LAUNCH_CTX_ATTN(256);
        break;
    default:
        break;
    }
}

#define CALL_CTX_ATTN_LAUNCHER(T, CACHE_T, KV_DT, BLOCK_SIZE)                  \
  context_attention_fwd_launcher<T, CACHE_T, KV_DT, BLOCK_SIZE>(               \
      out, query, key, value, key_cache, value_cache, num_kv_heads, scale,      \
      block_tables, context_lens, query_lens, query_start_locs, seq_ids,        \
      max_num_blocks_per_seq, total_new_tokens, num_heads, head_size,           \
      kv_block_stride, kv_head_stride, stream, k_scale, v_scale,               \
      sliding_window, sinks, max_total_kv_len);

#define CALL_CTX_ATTN_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DT)                   \
  switch (block_size) {                                                         \
  case 8:                                                                       \
    CALL_CTX_ATTN_LAUNCHER(T, CACHE_T, KV_DT, 8);                              \
    break;                                                                      \
  case 16:                                                                      \
    CALL_CTX_ATTN_LAUNCHER(T, CACHE_T, KV_DT, 16);                             \
    break;                                                                      \
  case 32:                                                                      \
    CALL_CTX_ATTN_LAUNCHER(T, CACHE_T, KV_DT, 32);                             \
    break;                                                                      \
  default:                                                                      \
    break;                                                                      \
  }

#undef CTX_WARP_SIZE
#undef CTX_MAX
#undef CTX_MIN
#undef CTX_DIVIDE_ROUND_UP
