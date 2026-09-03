#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
using gdn_fp8_e4m3 = __nv_fp8_e4m3;
#else
struct alignas(1) gdn_fp8_e4m3 {
  unsigned char value;

  __device__ explicit gdn_fp8_e4m3(float) : value(0) {}
};
#endif

constexpr int GDN_CHANNEL_BLOCK_SIZE = 256;
constexpr int GDN_DECODE_VALUE_TILE = 64;
constexpr int GDN_DECODE_STATE_LOAD_UNROLL = 128;
constexpr int GDN_DECODE_STATE_UPDATE_TILE_ROWS = 32;
constexpr int GDN_DECODE_COOPERATIVE_K = 128;
constexpr int GDN_DECODE_COOPERATIVE_V = 16;
constexpr int GDN_DECODE_COOPERATIVE_V_PADDED = 20;
constexpr int GDN_DECODE_COOPERATIVE_THREADS = 128;
constexpr int GDN_DECODE_COOPERATIVE_VALUES_PER_WARP = 4;
constexpr int GDN_DECODE_PIPELINED_K = 128;
constexpr int GDN_DECODE_PIPELINED_V = 32;
constexpr int GDN_DECODE_PIPELINED_V_PADDED = 36;
constexpr int GDN_DECODE_PIPELINED_STAGES = 2;
constexpr int GDN_DECODE_PIPELINED_THREADS = 256;
constexpr int GDN_DECODE_PIPELINED_WARPS = GDN_DECODE_PIPELINED_THREADS / 32;
constexpr int GDN_DECODE_PIPELINED_VALUES_PER_WARP = 4;
constexpr int GDN_DECODE_KERNEL_COOPERATIVE = 1;
constexpr int GDN_DECODE_KERNEL_PIPELINED = 2;
constexpr int GDN_DECODE_KERNEL_VALUE_MAJOR_4 = 3;
constexpr int GDN_DECODE_KERNEL_VALUE_MAJOR_32 = 4;
constexpr int GDN_DECODE_VALUE_MAJOR_K = 128;
constexpr int GDN_DECODE_VALUE_MAJOR_V = 128;
constexpr int GDN_PACKED_CONV_WIDTH = 4;
constexpr int GDN_PREFILL_CONV_THREADS = 128;
constexpr int GDN_PREFILL_CONV_TOKEN_TILE = 64;
constexpr int GDN_RMSNORM_FAST_HIDDEN = 128;
constexpr int GDN_RMSNORM_ROWS_PER_BLOCK = 8;
constexpr int GDN_RMSNORM_TILED_LANES_PER_ROW = 16;
constexpr int GDN_RMSNORM_TILED_ROW_PAIR_OFFSET = 2;
constexpr int GDN_RMSNORM_TILED_ROWS_PER_BLOCK = 4;
constexpr int GDN_RMSNORM_TILED_MIN_ROWS = 1024;
constexpr int GDN_RMSNORM_TILED_VALUES_PER_LANE = 8;
constexpr int GDN_STATE_DTYPE_F16 = 0;
constexpr int GDN_STATE_DTYPE_BF16 = 1;
constexpr int GDN_STATE_DTYPE_F32 = 2;
constexpr uint32_t GDN_PENDING_KEY_BANK_MASK = 1u;

__device__ __forceinline__ float4 gdn_load_state_x4(const float *source) {
  return *reinterpret_cast<const float4 *>(source);
}

__device__ __forceinline__ float4
gdn_load_state_x4(const __half *source) {
  const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(source));
  const float2 hi =
      __half22float2(*reinterpret_cast<const __half2 *>(source + 2));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ float4
gdn_load_state_x4(const __nv_bfloat16 *source) {
  const float2 lo =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(source));
  const float2 hi = __bfloat1622float2(
      *reinterpret_cast<const __nv_bfloat162 *>(source + 2));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void gdn_store_state_x4(float *destination,
                                                   float4 value) {
  *reinterpret_cast<float4 *>(destination) = value;
}

__device__ __forceinline__ void gdn_store_state_x4(__half *destination,
                                                   float4 value) {
  *reinterpret_cast<__half2 *>(destination) =
      __floats2half2_rn(value.x, value.y);
  *reinterpret_cast<__half2 *>(destination + 2) =
      __floats2half2_rn(value.z, value.w);
}

__device__ __forceinline__ void
gdn_store_state_x4(__nv_bfloat16 *destination, float4 value) {
  *reinterpret_cast<__nv_bfloat162 *>(destination) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162 *>(destination + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

__device__ __forceinline__ bool
gdn_is_padding_row(const int32_t *__restrict__ slot_indices, int bidx) {
  return slot_indices && slot_indices[bidx] < 0;
}

template <typename T>
__device__ __forceinline__ T gdn_ragged_from_float(float value);

template <>
__device__ __forceinline__ float gdn_ragged_from_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16
gdn_ragged_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void gdn_packed_to_padded_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, size_t total_elements,
    int padded_len, int width, int64_t source_token_stride,
    float padding_value) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const size_t token = index / width;
    const int feature = index % width;
    const int row = token / padded_len;
    const int position = token % padded_len;
    const uint32_t start = cu_seqlens[row];
    const uint32_t row_len = cu_seqlens[row + 1] - start;
    output[index] = position < row_len
                        ? source[(size_t)(start + position) *
                                     source_token_stride +
                                 feature]
                        : gdn_ragged_from_float<T>(padding_value);
  }
}

template <typename T>
__global__ void gdn_padded_to_packed_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, int width,
    int64_t source_batch_stride, int64_t source_token_stride,
    int64_t source_feature_stride, int feature_inner_width) {
  const int row = blockIdx.x;
  const uint32_t start = cu_seqlens[row];
  const uint32_t row_len = cu_seqlens[row + 1] - start;
  const size_t row_elements = (size_t)row_len * width;
  for (size_t index = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
       index < row_elements;
       index += (size_t)gridDim.y * blockDim.x) {
    const int position = index / width;
    const int feature = index % width;
    const int feature_outer = feature / feature_inner_width;
    const int feature_inner = feature % feature_inner_width;
    output[(size_t)(start + position) * width + feature] =
        source[(size_t)row * source_batch_stride +
               (size_t)position * source_token_stride +
               (size_t)feature_outer * source_feature_stride + feature_inner];
  }
}

template <typename T>
__global__ void gdn_extract_ragged_conv_state_kernel(
    const T *__restrict__ padded_input, const T *__restrict__ initial_state,
    T *__restrict__ output, const uint32_t *__restrict__ cu_seqlens,
    size_t total_elements, int padded_len, int channels, int state_width,
    int64_t input_batch_stride, int64_t input_token_stride) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const int state_position = index % state_width;
    const size_t channel_row = index / state_width;
    const int channel = channel_row % channels;
    const int row = channel_row / channels;
    const int row_len = cu_seqlens[row + 1] - cu_seqlens[row];
    if (row_len >= state_width) {
      const int input_position = row_len - state_width + state_position;
      output[index] =
          padded_input[(size_t)row * input_batch_stride +
                       (size_t)input_position * input_token_stride +
                       channel];
    } else {
      const int retained = state_width - row_len;
      output[index] = state_position < retained
                          ? initial_state[((size_t)row * channels + channel) *
                                              state_width +
                                          state_position + row_len]
                          : padded_input[(size_t)row * input_batch_stride +
                                         (size_t)(state_position - retained) *
                                             input_token_stride +
                                         channel];
    }
  }
}

extern "C" void gdn_packed_to_padded(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_token_stride,
    float padding_value, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * padded_len * width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        total_elements, padded_len, width, source_token_stride, padding_value);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, total_elements,
        padded_len, width, source_token_stride, padding_value);
  }
}

extern "C" void gdn_padded_to_packed(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_batch_stride,
    int64_t source_token_stride, int64_t source_feature_stride,
    int feature_inner_width, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t max_row_elements = (size_t)padded_len * width;
  const int row_blocks =
      min((size_t)65535, (max_row_elements + THREADS - 1) / THREADS);
  const dim3 grid(batch_size, row_blocks);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        width, source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, width,
        source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  }
}

extern "C" void gdn_extract_ragged_conv_state(
    const void *padded_input, const void *initial_state, void *output,
    const uint32_t *cu_seqlens, int batch_size, int padded_len, int channels,
    int state_width, int64_t input_batch_stride, int64_t input_token_stride,
    int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * channels * state_width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)padded_input,
        (const __nv_bfloat16 *)initial_state, (__nv_bfloat16 *)output,
        cu_seqlens, total_elements, padded_len, channels, state_width,
        input_batch_stride, input_token_stride);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)padded_input, (const float *)initial_state,
        (float *)output, cu_seqlens, total_elements, padded_len, channels,
        state_width, input_batch_stride, input_token_stride);
  }
}

// (batch, head) -> row of the state buffer: identity on a gathered [B*H, ...] copy, or through the
// per-batch slot table when a kernel updates the recurrent state pool in place
__device__ __forceinline__ size_t gdn_state_row(const int32_t *__restrict__ slot_indices,
                                                int bidx, int h, int num_heads) {
  const size_t slot = slot_indices ? (size_t)slot_indices[bidx] : (size_t)bidx;
  return slot * num_heads + h;
}

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence (optimized)
//
// V-tiled recurrence with compile-time K dimension for register residency.
// Grid: (ceil(V/BV), B*H), Block: (BV,). Each thread owns BK registers of
// state. Shared memory holds k_buf and q_buf (2*BK floats).
//
// Optimizations over naive version:
//   - Template BK -> float s[BK] lives in true registers (1 cycle vs ~30)
//   - #pragma unroll on all k-loops -> full ILP
//   - Fused decay+kv_mem pass and fused state_update+output pass
//   - __fmaf_rn intrinsics for guaranteed fused multiply-add
//   - BV=64 threads -> 2 warps, 6 blocks/SM on Ampere
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

// Optimized kernel: BK known at compile time -> registers + full unrolling
template <typename StateT, int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const float *__restrict__ q,    // [BH, S, K]
    const float *__restrict__ k,    // [BH, S, K]
    const float *__restrict__ v,    // [BH, S, V]
    const float *__restrict__ g,    // [BH, S]
    const float *__restrict__ beta, // [BH, S]
    StateT *__restrict__ state,     // [BH, K, V] or the pool with slot_indices
    float *__restrict__ output,     // [BH, S, V]
    int seq_len, int v_dim, const int32_t *__restrict__ slot_indices,
    int num_heads) {

  const int v_tile = blockIdx.x;       // which V-tile
  const int bh = blockIdx.y;           // batch*head index
  const int tid = threadIdx.x;         // thread within tile [0, BV)
  const int v_idx = v_tile * BV + tid; // global V index

  if (v_idx >= v_dim)
    return;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Shared memory: k_buf[BK] + q_buf[BK]
  __shared__ float k_buf[BK];
  __shared__ float q_buf[BK];

  // Load state column into registers — BK is compile-time, so this is
  // a true register array (not spilled to local memory)
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
// Collaboratively load k_t into shared memory
// BK / BV loads per thread (e.g. 128/64 = 2)
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      k_buf[j] = k_bh[t * BK + j];
    }
    __syncthreads();

    // Load scalars for this timestep
    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    // Fused pass 1: decay state + compute kv_mem
    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    // Delta rule
    float delta = (v_t - kv_mem) * beta_t;

// Collaboratively load q_t into shared memory
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      q_buf[j] = q_bh[t * BK + j];
    }
    __syncthreads();

    // Fused pass 2: update state + compute output
    float y_t = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

// Write state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Fallback kernel: runtime k_dim, still V-tiled for occupancy
template <typename StateT, int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len, int k_dim, int v_dim,
    const int32_t *__restrict__ slot_indices, int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  if (v_idx >= v_dim)
    return;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * k_dim;
  const float *k_bh = k + (size_t)bh * seq_len * k_dim;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * k_dim * v_dim;

  extern __shared__ float shared[];
  float *k_buf = shared;
  float *q_buf = shared + k_dim;

  float s[MAX_K];
  for (int j = 0; j < k_dim; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
    for (int j = tid; j < k_dim; j += BV) {
      k_buf[j] = k_bh[t * k_dim + j];
    }
    __syncthreads();

    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    float delta = (v_t - kv_mem) * beta_t;

    for (int j = tid; j < k_dim; j += BV) {
      q_buf[j] = q_bh[t * k_dim + j];
    }
    __syncthreads();

    float y_t = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

  for (int j = 0; j < k_dim; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

template <typename StateT>
void launch_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    // Fast path for Qwen3-Next (k_dim=128)
    constexpr int BK = 128;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    // Fast path for models with k_dim=64
    constexpr int BK = 64;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    // Fallback for other k_dim values (runtime loop, still V-tiled)
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    size_t smem = 2 * k_dim * sizeof(float);
    gated_delta_rule_recurrence_kernel_fallback<StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                        seq_len, k_dim, v_dim, slot_indices,
                                        num_heads);
  }
}

extern "C" void gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <int WARP_SIZE>
__device__ __forceinline__ float gdn_warp_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return __shfl_sync(0xffffffff, x, 0, WARP_SIZE);
}

template <typename StateT, int BK, int NUM_WARPS, bool VALUE_MAJOR = false>
__global__ __launch_bounds__(
    32 * NUM_WARPS,
    2) void gated_delta_rule_recurrence_kernel_warp(const float *__restrict__ q,
                                                    const float *__restrict__ k,
                                                    const float *__restrict__ v,
                                                    const float *__restrict__ g,
                                                    const float
                                                        *__restrict__ beta,
                                                    StateT *__restrict__ state,
                                                    float *__restrict__ output,
                                                    int seq_len, int v_dim,
                                                    const int32_t *__restrict__ slot_indices,
                                                    int num_heads) {

  constexpr int WARP_SIZE = 32;
  static_assert(BK % WARP_SIZE == 0, "BK must be a multiple of warp size");
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int v_idx = blockIdx.x * NUM_WARPS + warp;
  const int bh = blockIdx.y;

  if (v_idx >= v_dim) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
      for (int t = 0; t < seq_len; t++) {
        out_bh[t * v_dim + v_idx] = 0.0f;
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  float s[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      s[r] = state_bh[v_idx * BK + row];
    } else {
      s[r] = state_bh[row * v_dim + v_idx];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;

    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      const int row = r * WARP_SIZE + lane;
      const float k_val = k_t[row];
      k_reg[r] = k_val;
      q_reg[r] = q_t[row];
      kv_partial = __fmaf_rn(s[r], k_val, kv_partial);
    }

    const float decay = expf(g_bh[t]);
    const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial);
    const float delta = (v_bh[t * v_dim + v_idx] - decay * kv_col) * beta_bh[t];

    float y_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      s[r] = __fmaf_rn(k_reg[r], delta, decay * s[r]);
      y_partial = __fmaf_rn(s[r], q_reg[r], y_partial);
    }

    const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
    if (lane == 0) {
      out_bh[t * v_dim + v_idx] = y_col;
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      state_bh[v_idx * BK + row] = s[r];
    } else {
      state_bh[row * v_dim + v_idx] = s[r];
    }
  }
}

template <typename StateT>
void launch_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);

  if (k_dim == 128) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 64, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    launch_gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh,
                                       seq_len, k_dim, v_dim, slot_indices,
                                       num_heads, stream);
  }
}

extern "C" void warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
void launch_vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return;
  }

  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS, true>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                   v_dim, slot_indices, num_heads);
}

extern "C" void vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT, int VALUES_PER_WARP, int NUM_WARPS>
__global__ __launch_bounds__(32 * NUM_WARPS, 2) void
gated_delta_rule_recurrence_kernel_vmajor_grouped(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len,
    const int32_t *__restrict__ slot_indices, int num_heads) {
  constexpr int WARP_SIZE = 32;
  constexpr int BK = 128;
  constexpr int V_DIM = 128;
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_group = blockIdx.x * NUM_WARPS + warp;
  const int first_value = value_group * VALUES_PER_WARP;
  const int bh = blockIdx.y;

  if (first_value >= V_DIM) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * V_DIM;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        for (int t = 0; t < seq_len; t++) {
          out_bh[t * V_DIM + first_value + value] = 0.0f;
        }
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * V_DIM;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads,
                            num_heads) *
                  BK * V_DIM;

  float s[VALUES_PER_WARP][ROWS_PER_LANE];
#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      s[value][row] = state_bh[(first_value + value) * BK + key];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;
    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial[VALUES_PER_WARP] = {};

#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      const float k_value = k_t[key];
      k_reg[row] = k_value;
      q_reg[row] = q_t[key];
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        kv_partial[value] =
            __fmaf_rn(s[value][row], k_value, kv_partial[value]);
      }
    }

    const float decay = expf(g_bh[t]);
#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      const int value_idx = first_value + value;
      const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial[value]);
      const float delta =
          (v_bh[t * V_DIM + value_idx] - decay * kv_col) * beta_bh[t];
      float y_partial = 0.0f;
#pragma unroll
      for (int row = 0; row < ROWS_PER_LANE; row++) {
        s[value][row] =
            __fmaf_rn(k_reg[row], delta, decay * s[value][row]);
        y_partial = __fmaf_rn(s[value][row], q_reg[row], y_partial);
      }
      const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
      if (lane == 0) {
        out_bh[t * V_DIM + value_idx] = y_col;
      }
    }
  }

#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      state_bh[(first_value + value) * BK + key] = s[value][row];
    }
  }
}

template <typename StateT, int VALUES_PER_WARP>
cudaError_t launch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return cudaErrorInvalidValue;
  }

  constexpr int NUM_WARPS = 4;
  constexpr int VALUES_PER_BLOCK = NUM_WARPS * VALUES_PER_WARP;
  const dim3 grid((v_dim + VALUES_PER_BLOCK - 1) / VALUES_PER_BLOCK, bh);
  const dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_vmajor_grouped<StateT, VALUES_PER_WARP,
                                                     NUM_WARPS>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                  slot_indices, num_heads);
  return cudaGetLastError();
}

template <typename StateT>
cudaError_t dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, cudaStream_t stream) {
  switch (values_per_warp) {
  case 2:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 2>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 4:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 4>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 8:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 8>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

extern "C" int vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len,
            k_dim, v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  }
}


// ============================================================================
// Kernel 1b: chunked_gated_delta_rule_recurrence (prefill optimization)
//
// Processes prefill tokens in BT-token chunks instead of one at a time.
// Within each chunk: parallel prefix sum of g, cooperative kk_dot computation,
// forward substitution (triangular solve), output computation, and state
// update.
//
// Same thread model as Kernel 1: one block per (v_tile, batch*head),
// one thread per V-column. Each thread owns BK registers of state.
//
// Shared memory holds:
//   k_chunk[BT * BK]  -- key vectors for current chunk
//   kk_dot[BT * BT]   -- dot(k[i], k[j]) lower-triangular matrix
//   gcum[BT]           -- cumulative sum of g within chunk
//   beta_s[BT]         -- beta values for chunk
//   q_buf[BK]          -- q vector (loaded one row at a time)
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] or [BH, V, K] (in/out)  output: [BH, S, V]
// ============================================================================

template <typename StateT, int BT, int BK, int BV, bool VALUE_MAJOR = false>
__global__ void
chunked_gated_delta_rule_kernel(const float *__restrict__ q,    // [BH, S, K]
                                const float *__restrict__ k,    // [BH, S, K]
                                const float *__restrict__ v,    // [BH, S, V]
                                const float *__restrict__ g,    // [BH, S]
                                const float *__restrict__ beta, // [BH, S]
                                StateT *__restrict__ state,     // key-major/value-major or pool
                                float *__restrict__ output,     // [BH, S, V]
                                int seq_len, int v_dim,
                                const int32_t *__restrict__ slot_indices,
                                int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  if (v_idx >= v_dim)
    return;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  const int num_chunks = (seq_len + BT - 1) / BT;

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Dynamic shared memory layout
  extern __shared__ float smem[];
  float *k_chunk = smem;                  // [BT * BK]
  float *kk_dot = smem + BT * BK;         // [BT * BT]
  float *gcum = smem + BT * BK + BT * BT; // [BT]
  float *beta_s = gcum + BT;              // [BT]
  float *q_buf = beta_s + BT;             // [BK]

  // Load state column into registers
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      s[j] = state_bh[v_idx * BK + j];
    } else {
      s[j] = state_bh[j * v_dim + v_idx];
    }
  }

  // Per-thread register array for corrected deltas
  float delta[BT];

  for (int c = 0; c < num_chunks; c++) {
    const int chunk_start = c * BT;
    const int chunk_len = min(BT, seq_len - chunk_start);

    // === Phase 1: Cooperative load of k, beta, g into shared memory ===
    for (int t = 0; t < chunk_len; t++) {
      for (int j = tid; j < BK; j += BV) {
        k_chunk[t * BK + j] = k_bh[(chunk_start + t) * BK + j];
      }
    }
    if (tid < chunk_len) {
      beta_s[tid] = beta_bh[chunk_start + tid];
      gcum[tid] = g_bh[chunk_start + tid];
    }
    __syncthreads();

    // === Phase 1b: Parallel prefix sum of g (Hillis-Steele) ===
    for (int stride = 1; stride < BT; stride <<= 1) {
      float prev = 0.0f;
      if (tid < chunk_len && (int)tid >= stride)
        prev = gcum[tid - stride];
      __syncthreads();
      if (tid < chunk_len && (int)tid >= stride)
        gcum[tid] += prev;
      __syncthreads();
    }

    // === Phase 2: Compute kk_dot[i][j] = dot(k[i], k[j]) for j < i ===
    // Only lower-triangular entries needed (strictly lower)
    for (int idx = tid; idx < chunk_len * chunk_len; idx += BV) {
      int i = idx / chunk_len;
      int j = idx % chunk_len;
      if (j < i) {
        float dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          dot = __fmaf_rn(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
        }
        kk_dot[i * BT + j] = dot;
      }
    }
    __syncthreads();

    // === Phase 3: Forward substitution (per V-column, in registers) ===
    // Computes corrected delta values via triangular solve
    for (int i = 0; i < chunk_len; i++) {
      float v_i = v_bh[(chunk_start + i) * v_dim + v_idx];
      float decay_i = expf(gcum[i]);
      float beta_i = beta_s[i];

      // Inter-chunk contribution: state @ k[i] with decay
      float kv_mem = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        kv_mem = __fmaf_rn(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
      }

      float rhs = beta_i * (v_i - kv_mem);

      // Subtract lower-triangular contributions (intra-chunk)
      for (int j = 0; j < i; j++) {
        float a_ij = beta_i * kk_dot[i * BT + j] * expf(gcum[i] - gcum[j]);
        rhs -= a_ij * delta[j];
      }
      delta[i] = rhs;
    }

    // === Phase 4: Output computation (per V-column) ===
    for (int i = 0; i < chunk_len; i++) {
      // Cooperatively load q[i] into shared
      for (int j = tid; j < BK; j += BV) {
        q_buf[j] = q_bh[(chunk_start + i) * BK + j];
      }
      __syncthreads();

      float decay_i = expf(gcum[i]);

      // Inter-chunk contribution: q[i] @ (state * decay)
      float o_val = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        o_val = __fmaf_rn(q_buf[d], s[d] * decay_i, o_val);
      }

      // Intra-chunk contribution: sum_{j<=i} dot(q[i], k[j]) * delta[j] *
      // exp(gcum[i] - gcum[j])
      for (int j = 0; j <= i; j++) {
        float qk_dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          qk_dot = __fmaf_rn(q_buf[d], k_chunk[j * BK + d], qk_dot);
        }
        o_val += qk_dot * delta[j] * expf(gcum[i] - gcum[j]);
      }

      out_bh[(chunk_start + i) * v_dim + v_idx] = o_val;
      __syncthreads();
    }

    // === Phase 5: State update for next chunk ===
    float g_total = gcum[chunk_len - 1];
#pragma unroll
    for (int d = 0; d < BK; d++) {
      float s_new = s[d] * expf(g_total);
      for (int t = 0; t < chunk_len; t++) {
        s_new += k_chunk[t * BK + d] * delta[t] * expf(g_total - gcum[t]);
      }
      s[d] = s_new;
    }

    __syncthreads();
  }

  // Write final state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      state_bh[v_idx * BK + j] = s[j];
    } else {
      state_bh[j * v_dim + v_idx] = s[j];
    }
  }
}

template <typename StateT, bool VALUE_MAJOR = false>
void launch_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    constexpr int BT = 64;
    constexpr int BK = 128;
    constexpr int BV = 64;
    // Shared memory: BT*BK + BT*BT + BT + BT + BK floats
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    // Request extended shared memory
    auto kernel =
        chunked_gated_delta_rule_kernel<StateT, BT, BK, BV, VALUE_MAJOR>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                          seq_len, v_dim, slot_indices,
                                          num_heads);
  } else if (k_dim == 64) {
    constexpr int BT = 64;
    constexpr int BK = 64;
    constexpr int BV = 64;
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    auto kernel =
        chunked_gated_delta_rule_kernel<StateT, BT, BK, BV, VALUE_MAJOR>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                          seq_len, v_dim, slot_indices,
                                          num_heads);
  } else if constexpr (!VALUE_MAJOR) {
    launch_gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh,
                                       seq_len, k_dim, v_dim, slot_indices,
                                       num_heads, stream);
  }
}

extern "C" void chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
cudaError_t launch_vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return cudaErrorInvalidValue;
  }

  constexpr int BT = 64;
  constexpr int BK = 128;
  constexpr int BV = 64;
  const size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);
  auto kernel = chunked_gated_delta_rule_kernel<StateT, BT, BK, BV, true>;
  const cudaError_t attribute_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attribute_status != cudaSuccess) {
    return attribute_status;
  }

  const dim3 grid((v_dim + BV - 1) / BV, bh);
  const dim3 block(BV);
  kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                        seq_len, v_dim, slot_indices,
                                        num_heads);
  return cudaGetLastError();
}

extern "C" int vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream));
  } else {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  }
}

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
//
// Each thread handles one channel: shift conv_state left by 1,
// insert new value, dot product with weight, apply SiLU.
//
// x: [B, 1, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state: [B, conv_dim, kernel_size] (in/out)
// output: [B, 1, conv_dim]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_update_kernel(
    const T *__restrict__ x,      // [B, 1, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    T *__restrict__ conv_state,   // [B, conv_dim, kernel_size]
    T *__restrict__ output,       // [B, 1, conv_dim]
    int batch_size, int conv_dim, int kernel_size, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    output[(size_t)b * conv_dim + ch] = (T)0.0f;
    return;
  }

  // Pointer to this batch/channel's conv state
  T *cs = conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;
  const T *w = weight + ch * kernel_size;

  // Shift state left by 1
  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  // Insert new value
  cs[kernel_size - 1] =
      x[(size_t)b * x_stride_b + (size_t)ch * x_stride_c];

  // Dot product with weight
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    acc += (float)cs[i] * (float)w[i];
  }

  // SiLU activation: x * sigmoid(x)
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[b * conv_dim + ch] = (T)result;
}

template <typename T> struct alignas(8) GdnConvWidth4 {
  T values[GDN_PACKED_CONV_WIDTH];
};

template <typename T>
__device__ __forceinline__ T gdn_conv_width4_update(
    T input, const GdnConvWidth4<T> &weights, GdnConvWidth4<T> *state) {
  GdnConvWidth4<T> values = *state;

#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
    values.values[i] = values.values[i + 1];
  }
  values.values[GDN_PACKED_CONV_WIDTH - 1] = input;
  *state = values;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
    acc += (float)values.values[i] * (float)weights.values[i];
  }
  const float sig = 1.0f / (1.0f + expf(-acc));
  return (T)(acc * sig);
}

template <typename T>
__global__ void causal_conv1d_update_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const size_t input_idx = (size_t)b * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[input_idx] = (T)0.0f;
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const size_t state_idx = state_row * conv_dim + ch;
  const size_t x_idx = (size_t)b * x_stride_b + (size_t)ch * x_stride_c;
  auto *state = reinterpret_cast<GdnConvWidth4<T> *>(conv_state);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  output[input_idx] = gdn_conv_width4_update(
      x[x_idx], weights[ch], &state[state_idx]);
}

extern "C" void causal_conv1d_update(const void *x, const void *weight,
                                     void *conv_state, void *output,
                                     int batch_size, int conv_dim,
                                     int kernel_size, int64_t x_stride_b,
                                     int64_t x_stride_s, int64_t x_stride_c,
                                     const int32_t *slot_indices, int dtype,
                                     int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE,
            batch_size);

  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    if (dtype == 0) {
      causal_conv1d_update_width4_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight, (__half *)conv_state,
          (__half *)output, batch_size, conv_dim, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    } else {
      causal_conv1d_update_width4_kernel<__nv_bfloat16>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
              conv_dim, x_stride_b, x_stride_s, x_stride_c, slot_indices);
    }
  } else if (dtype == 0) {
    causal_conv1d_update_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight, (__half *)conv_state,
        (__half *)output, batch_size, conv_dim, kernel_size, x_stride_b,
        x_stride_s, x_stride_c, slot_indices);
  } else {
    causal_conv1d_update_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
        conv_dim, kernel_size, x_stride_b, x_stride_s, x_stride_c,
        slot_indices);
  }
}

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
//
// Each thread handles one (channel, position), seeded from the prior state.
// A second pass retains the last kernel_size positions.
//
// x: [B, S, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state_out: [B, conv_dim, kernel_size]  output: [B, S, conv_dim]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_full_kernel(
    const T *__restrict__ x,      // [B, S, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    const T *__restrict__ conv_state,
    T *__restrict__ output, // [B, S, conv_dim]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  const size_t plane = (size_t)conv_dim * seq_len;

  if (idx >= plane || b >= batch_size)
    return;

  const int pos = (int)(idx / conv_dim);
  const int ch = (int)(idx % conv_dim);

  const size_t output_idx = ((size_t)b * seq_len + pos) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[output_idx] = (T)0.0f;
    return;
  }

  const T *w = weight + (size_t)ch * kernel_size;
  const T *cs =
      conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;

  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    int src_pos = pos - (kernel_size - 1) + i;
    float x_val = src_pos >= 0
                      ? (float)x[(size_t)b * x_stride_b +
                                 (size_t)src_pos * x_stride_s +
                                 (size_t)ch * x_stride_c]
                      : (float)cs[kernel_size + src_pos];
    acc += x_val * (float)w[i];
  }

  // SiLU
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[output_idx] = (T)result;
}

template <typename T>
__device__ __forceinline__ float causal_conv1d_width4_load(
    const T *__restrict__ x, const T *__restrict__ state, int pos,
    size_t x_batch_offset, int64_t x_stride_s, int64_t x_stride_c, int ch) {
  if (pos >= 0) {
    return (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                    (size_t)ch * x_stride_c];
  }
  return (float)state[GDN_PACKED_CONV_WIDTH + pos];
}

template <typename T, int TOKEN_TILE>
__global__ void causal_conv1d_full_width4_tiled_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    const T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int seq_len, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int start = blockIdx.y * TOKEN_TILE;
  const int b = blockIdx.z;
  if (ch >= conv_dim || start >= seq_len || b >= batch_size) {
    return;
  }

  const int end = min(start + TOKEN_TILE, seq_len);
  T *out = output + ((size_t)b * seq_len + start) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    for (int pos = start; pos < end; ++pos) {
      *out = (T)0.0f;
      out += conv_dim;
    }
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const T *state = conv_state +
                   (state_row * conv_dim + ch) * GDN_PACKED_CONV_WIDTH;
  const T *w = weight + (size_t)ch * GDN_PACKED_CONV_WIDTH;
  const size_t x_batch_offset = (size_t)b * x_stride_b;
  float x0 = causal_conv1d_width4_load(
      x, state, start - 3, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x1 = causal_conv1d_width4_load(
      x, state, start - 2, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x2 = causal_conv1d_width4_load(
      x, state, start - 1, x_batch_offset, x_stride_s, x_stride_c, ch);
  const float w0 = (float)w[0];
  const float w1 = (float)w[1];
  const float w2 = (float)w[2];
  const float w3 = (float)w[3];

  for (int pos = start; pos < end; ++pos) {
    const float x3 = (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                              (size_t)ch * x_stride_c];
    const float acc = __fmaf_rn(x0, w0, __fmaf_rn(x1, w1, __fmaf_rn(x2, w2, x3 * w3)));
    const float result = acc / (1.0f + expf(-acc));
    *out = (T)result;
    out += conv_dim;
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
}

template <typename T>
__global__ void save_conv_state_kernel(
    const T *__restrict__ x, // [B, S, conv_dim]
    // May alias conv_state_out (pooled in-place update): every read is ahead of the write position
    const T *conv_state_in,
    T *conv_state_out, // [B, conv_dim, kernel_size]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    return;
  }

  const size_t row = gdn_state_row(slot_indices, b, 0, 1);
  const T *prior = conv_state_in + (row * conv_dim + ch) * kernel_size;
  T *cs = conv_state_out + (row * conv_dim + ch) * kernel_size;

  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = prior[i + seq_len];
    } else {
      const int pos = seq_len - kernel_size + i;
      cs[i] = x[(size_t)b * x_stride_b + (size_t)pos * x_stride_s +
                (size_t)ch * x_stride_c];
    }
  }
}

extern "C" void causal_conv1d_full(const void *x, const void *weight,
                                   const void *conv_state_in,
                                   void *conv_state_out, void *output,
                                   int batch_size, int conv_dim, int seq_len,
                                   int kernel_size, int64_t x_stride_b,
                                   int64_t x_stride_s, int64_t x_stride_c,
                                   const int32_t *slot_indices, int dtype,
                                   int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  const dim3 state_block(256);
  dim3 block = state_block;
  const size_t plane = (size_t)conv_dim * seq_len;
  dim3 grid((unsigned int)((plane + 255) / 256), batch_size);

  const bool use_width4_tiled =
      kernel_size == GDN_PACKED_CONV_WIDTH && x_stride_c == 1;
  if (use_width4_tiled) {
    block = dim3(GDN_PREFILL_CONV_THREADS);
    grid = dim3((conv_dim + GDN_PREFILL_CONV_THREADS - 1) /
                    GDN_PREFILL_CONV_THREADS,
                (seq_len + GDN_PREFILL_CONV_TOKEN_TILE - 1) /
                    GDN_PREFILL_CONV_TOKEN_TILE,
                batch_size);
  }

  if (dtype == 0) {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<__half,
                                             GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __half *)x, (const __half *)weight,
              (const __half *)conv_state_in, (__half *)output, batch_size,
              conv_dim, seq_len, x_stride_b, x_stride_s, x_stride_c,
              slot_indices);
    } else {
      causal_conv1d_full_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight,
          (const __half *)conv_state_in, (__half *)output, batch_size,
          conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s, x_stride_c,
          slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__half><<<grid2, state_block, 0, custream>>>(
        (const __half *)x, (const __half *)conv_state_in,
        (__half *)conv_state_out, batch_size, conv_dim, seq_len, kernel_size,
        x_stride_b, x_stride_s, x_stride_c, slot_indices);
  } else {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<
          __nv_bfloat16, GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (const __nv_bfloat16 *)conv_state_in,
              (__nv_bfloat16 *)output, batch_size, conv_dim, seq_len,
              x_stride_b, x_stride_s, x_stride_c, slot_indices);
    } else {
      causal_conv1d_full_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
          (const __nv_bfloat16 *)conv_state_in, (__nv_bfloat16 *)output,
          batch_size, conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__nv_bfloat16>
        <<<grid2, state_block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)conv_state_in,
        (__nv_bfloat16 *)conv_state_out, batch_size, conv_dim, seq_len,
        kernel_size, x_stride_b, x_stride_s, x_stride_c, slot_indices);
  }
}

template <typename T, typename OutT>
__global__ void gdn_prepare_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, OutT *__restrict__ q_out,
    OutT *__restrict__ k_out, OutT *__restrict__ v_out,
    float *__restrict__ g_out, float *__restrict__ beta_out, int batch_size,
    int seq_len, int token_stride, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int tiled_v_heads) {
  const int token_head = blockIdx.x;
  const int hv = token_head % num_v_heads;
  const int token = token_head / num_v_heads;
  const int t = token % seq_len;
  const int bidx = token / seq_len;
  const int tid = threadIdx.x;

  if (bidx >= batch_size)
    return;

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const int bh = bidx * num_v_heads + hv;

  const size_t token_idx = (size_t)bidx * seq_len + t;
  const T *row = mixed_qkv + token_idx * conv_dim;
  const T *b_row = b + token_idx * num_v_heads;
  const T *a_row = a + token_idx * num_v_heads;

  __shared__ float red_q[256];
  __shared__ float red_k[256];
  __shared__ float q_mul;
  __shared__ float k_mul;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)head_k_dim);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);

    float b_val = (float)b_row[hv];
    float a_val = (float)a_row[hv] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_out[(size_t)bh * token_stride + t] = 1.0f / (1.0f + expf(-b_val));
    g_out[(size_t)bh * token_stride + t] = -expf(a_log[hv]) * softplus_val;
  }
  __syncthreads();

  OutT *q_dst = q_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *k_dst = k_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *v_dst = v_out + ((size_t)bh * token_stride + t) * head_v_dim;

  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_dst[d] = (OutT)(q_val * q_mul);
    k_dst[d] = (OutT)(k_val * k_mul);
  }

  for (int d = tid; d < head_v_dim; d += blockDim.x) {
    v_dst[d] = (OutT)(float)row[2 * key_dim + hv * head_v_dim + d];
  }
}

template <typename T, typename OutT>
static void launch_gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(batch_size * seq_len * num_v_heads);
  gdn_prepare_recurrence_kernel<T, OutT><<<grid, block, 0, stream>>>(
      (const T *)mixed_qkv, (const T *)b, (const T *)a, a_log, dt_bias,
      (OutT *)q_out, (OutT *)k_out, (OutT *)v_out, g_out, beta_out, batch_size,
      seq_len, token_stride, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
      tiled_v_heads);
}

extern "C" void gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int dtype, int out_bf16, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0 && out_bf16) {
    launch_gdn_prepare_recurrence<__half, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (dtype == 0) {
    launch_gdn_prepare_recurrence<__half, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (out_bf16) {
    launch_gdn_prepare_recurrence<__nv_bfloat16, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else {
    launch_gdn_prepare_recurrence<__nv_bfloat16, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  }
}

__device__ __forceinline__ float gdn_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T, typename StateT, int BV>
__global__ __launch_bounds__(32) void gdn_decode_recurrence_kernel_value_major(
    const T *__restrict__ mixed_qkv,
    const T *__restrict__ b,
    const T *__restrict__ a,
    const float *__restrict__ a_log,
    const float *__restrict__ dt_bias,
    StateT *__restrict__ state,
    T *__restrict__ output,
    int batch_size,
    int num_k_heads,
    int num_v_heads,
    int tiled_v_heads,
    int64_t b_batch_stride,
    int64_t b_head_stride,
    int64_t a_batch_stride,
    int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  static_assert(GDN_DECODE_VALUE_MAJOR_V % BV == 0,
                "V must divide into value tiles");
  constexpr int NUM_V_TILES = GDN_DECODE_VALUE_MAJOR_V / BV;

  const int lane = threadIdx.x;
  const int linear_tile = blockIdx.x;
  const int total_tiles = batch_size * num_v_heads * NUM_V_TILES;
  if (linear_tile >= total_tiles) {
    return;
  }
  const int v_tile = linear_tile % NUM_V_TILES;
  const int bh = linear_tile / NUM_V_TILES;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  T *out_bh = output + (size_t)bh * GDN_DECODE_VALUE_MAJOR_V;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (lane < BV) {
      out_bh[v_tile * BV + lane] = (T)0.0f;
    }
    return;
  }
  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * GDN_DECODE_VALUE_MAJOR_K;
  const int value_dim = num_v_heads * GDN_DECODE_VALUE_MAJOR_V;
  const int conv_dim = 2 * key_dim + value_dim;
  const int v_base = v_tile * BV;
  const T *row = mixed_qkv + (size_t)bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) *
                  GDN_DECODE_VALUE_MAJOR_V * GDN_DECODE_VALUE_MAJOR_K;

  float4 qv = gdn_load_state_x4(
      row + hk * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
  float4 kv = gdn_load_state_x4(
      row + key_dim + hk * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
  float q_norm = qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w;
  float k_norm = kv.x * kv.x + kv.y * kv.y + kv.z * kv.z + kv.w * kv.w;
  q_norm = gdn_warp_sum<32>(q_norm);
  k_norm = gdn_warp_sum<32>(k_norm);
  const float q_mul = rsqrtf(q_norm + 1.0e-6f) *
                      rsqrtf((float)GDN_DECODE_VALUE_MAJOR_K);
  const float k_mul = rsqrtf(k_norm + 1.0e-6f);
  qv = make_float4(qv.x * q_mul, qv.y * q_mul, qv.z * q_mul, qv.w * q_mul);
  kv = make_float4(kv.x * k_mul, kv.y * k_mul, kv.z * k_mul, kv.w * k_mul);

  float beta_t = 0.0f;
  float decay_t = 0.0f;
  if (lane == 0) {
    const float b_value =
        (float)b[(size_t)bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[(size_t)bidx * a_batch_stride + hv * a_head_stride] +
        dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  beta_t = __shfl_sync(0xffffffff, beta_t, 0);
  decay_t = __shfl_sync(0xffffffff, decay_t, 0);

  float4 h[BV];
#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    const float4 raw = gdn_load_state_x4(
        state_bh + (v_base + vi) * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
    h[vi] = make_float4(raw.x * decay_t, raw.y * decay_t,
                        raw.z * decay_t, raw.w * decay_t);
  }

  const float v_owned =
      lane < BV
          ? (float)row[2 * key_dim + hv * GDN_DECODE_VALUE_MAJOR_V + v_base + lane]
          : 0.0f;
  float out_owned = 0.0f;
#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    float state_dot_k = h[vi].x * kv.x;
    state_dot_k = __fmaf_rn(h[vi].y, kv.y, state_dot_k);
    state_dot_k = __fmaf_rn(h[vi].z, kv.z, state_dot_k);
    state_dot_k = __fmaf_rn(h[vi].w, kv.w, state_dot_k);
    state_dot_k = gdn_warp_sum<32>(state_dot_k);
    const float delta =
        (__shfl_sync(0xffffffff, v_owned, vi) - state_dot_k) * beta_t;

    h[vi].x = __fmaf_rn(kv.x, delta, h[vi].x);
    h[vi].y = __fmaf_rn(kv.y, delta, h[vi].y);
    h[vi].z = __fmaf_rn(kv.z, delta, h[vi].z);
    h[vi].w = __fmaf_rn(kv.w, delta, h[vi].w);

    float state_dot_q = h[vi].x * qv.x;
    state_dot_q = __fmaf_rn(h[vi].y, qv.y, state_dot_q);
    state_dot_q = __fmaf_rn(h[vi].z, qv.z, state_dot_q);
    state_dot_q = __fmaf_rn(h[vi].w, qv.w, state_dot_q);
    state_dot_q = gdn_warp_sum<32>(state_dot_q);
    if (lane == vi) {
      out_owned = state_dot_q;
    }
  }

#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    gdn_store_state_x4(
        state_bh + (v_base + vi) * GDN_DECODE_VALUE_MAJOR_K + lane * 4,
        h[vi]);
  }
  if (lane < BV) {
    out_bh[v_base + lane] = (T)out_owned;
  }
}

template <int VALUES_PER_WARP>
__device__ __forceinline__ float gdn_grouped_k_sum(float value) {
#pragma unroll
  for (int offset = 16; offset >= VALUES_PER_WARP; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ void gdn_cp_async_cg_16(void *dst,
                                                   const void *src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(dst_smem), "l"(src));
#else
  *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
#endif
}

__device__ __forceinline__ void gdn_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void gdn_cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

// Adapted from FlashInfer's Apache-2.0 nontranspose GDN kernel, Copyright (c) 2025 FlashInfer team.
// Exact source revision and license notices are in third_party/flashinfer_gdn.
template <typename T, typename StateT>
__global__ void gdn_decode_recurrence_kernel_cooperative(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  constexpr int BK = GDN_DECODE_COOPERATIVE_K;
  constexpr int BV = GDN_DECODE_COOPERATIVE_V;
  constexpr int VECTORS_PER_ROW = BV / 4;
  constexpr int STATE_VECTORS = BK * VECTORS_PER_ROW;
  constexpr int K_LANES_PER_VALUE = 32 / GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  constexpr int K_ITERATIONS = BK / K_LANES_PER_VALUE;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  const int v_local = lane % GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  const int k_lane = lane / GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  const int v_in_tile = warp * GDN_DECODE_COOPERATIVE_VALUES_PER_WARP + v_local;
  const int v_idx = v_tile * BV + v_in_tile;

  if (bidx >= batch_size) {
    return;
  }

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (tid < BV) {
      out_bh[v_tile * BV + tid] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ __align__(16) float
      state_buf[BK * GDN_DECODE_COOPERATIVE_V_PADDED];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float q_warp_sums[4];
  __shared__ float k_warp_sums[4];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

#pragma unroll
  for (int vector_idx = tid; vector_idx < STATE_VECTORS;
       vector_idx += GDN_DECODE_COOPERATIVE_THREADS) {
    const int k_idx = vector_idx / VECTORS_PER_ROW;
    const int v_vector = vector_idx % VECTORS_PER_ROW;
    const StateT *src =
        state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    float *dst = state_buf + k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_vector * 4;
    if constexpr (sizeof(StateT) == sizeof(float)) {
      __pipeline_memcpy_async(dst, src, sizeof(float4));
    } else {
      *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
    }
  }
  __pipeline_commit();

  const float q_value = (float)row[hk * BK + tid];
  const float k_value = (float)row[key_dim + hk * BK + tid];
  q_buf[tid] = q_value;
  k_buf[tid] = k_value;
  const float q_sum = gdn_warp_sum(q_value * q_value);
  const float k_sum = gdn_warp_sum(k_value * k_value);
  if (lane == 0) {
    q_warp_sums[warp] = q_sum;
    k_warp_sums[warp] = k_sum;
  }

  if (tid == 0) {
    const float b_value = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  __syncthreads();

  if (tid == 0) {
    const float q_norm = q_warp_sums[0] + q_warp_sums[1] + q_warp_sums[2] +
                         q_warp_sums[3];
    const float k_norm = k_warp_sums[0] + k_warp_sums[1] + k_warp_sums[2] +
                         k_warp_sums[3];
    q_mul = rsqrtf(q_norm + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(k_norm + 1.0e-6f);
  }
  __syncthreads();

  q_buf[tid] *= q_mul;
  k_buf[tid] *= k_mul;
  __pipeline_wait_prior(0);
  __syncthreads();

  float state_dot_k = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
    const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
    const float old_state =
        state_buf[k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_in_tile] * decay_t;
    state_dot_k = __fmaf_rn(old_state, k_buf[k_idx], state_dot_k);
  }
  state_dot_k =
      gdn_grouped_k_sum<GDN_DECODE_COOPERATIVE_VALUES_PER_WARP>(state_dot_k);

  float delta = 0.0f;
  if (k_lane == 0) {
    const float v_value = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
    delta = (v_value - state_dot_k) * beta_t;
  }
  delta = __shfl_sync(0xffffffff, delta, v_local);

  float state_dot_q = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
    const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
    const int state_idx = k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_in_tile;
    const float old_state = state_buf[state_idx] * decay_t;
    const float new_state = __fmaf_rn(k_buf[k_idx], delta, old_state);
    state_buf[state_idx] = new_state;
    state_dot_q = __fmaf_rn(new_state, q_buf[k_idx], state_dot_q);
  }
  state_dot_q =
      gdn_grouped_k_sum<GDN_DECODE_COOPERATIVE_VALUES_PER_WARP>(state_dot_q);
  if (k_lane == 0) {
    out_bh[v_idx] = (T)state_dot_q;
  }
  __syncthreads();

#pragma unroll
  for (int vector_idx = tid; vector_idx < STATE_VECTORS;
       vector_idx += GDN_DECODE_COOPERATIVE_THREADS) {
    const int k_idx = vector_idx / VECTORS_PER_ROW;
    const int v_vector = vector_idx % VECTORS_PER_ROW;
    const float *src =
        state_buf + k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_vector * 4;
    StateT *dst = state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    gdn_store_state_x4(dst, *reinterpret_cast<const float4 *>(src));
  }
}

// Adapted from FlashInfer's Apache-2.0 large-batch nontranspose GDN kernel.
// Exact source revision and license notices are in third_party/flashinfer_gdn.
template <typename T, typename StateT>
__global__ void gdn_decode_recurrence_kernel_pipelined(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  constexpr int BK = GDN_DECODE_PIPELINED_K;
  constexpr int BV = GDN_DECODE_PIPELINED_V;
  constexpr int VECTORS_PER_ROW = BV / 4;
  constexpr int STATE_VECTORS = BK * VECTORS_PER_ROW;
  constexpr int K_LANES_PER_VALUE =
      32 / GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  constexpr int K_ITERATIONS = BK / K_LANES_PER_VALUE;
  static_assert(BV == GDN_DECODE_PIPELINED_WARPS *
                          GDN_DECODE_PIPELINED_VALUES_PER_WARP,
                "each warp must own one group of values");
  static_assert(BK % K_LANES_PER_VALUE == 0,
                "K must divide across the grouped lanes");
  static_assert(GDN_DECODE_PIPELINED_V_PADDED % 4 == 0,
                "padded rows must preserve vector alignment");
  static_assert(GDN_DECODE_PIPELINED_STAGES == 2,
                "the decode pipeline expects two stages");

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bh = blockIdx.x;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  const int v_local = lane % GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  const int k_lane = lane / GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  const int v_in_tile = warp * GDN_DECODE_PIPELINED_VALUES_PER_WARP + v_local;

  if (bidx >= batch_size) {
    return;
  }

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    for (int v_idx = tid; v_idx < head_v_dim; v_idx += blockDim.x) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const int num_v_tiles = head_v_dim / BV;
  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ __align__(16)
      float state_buf[GDN_DECODE_PIPELINED_STAGES]
                     [BK * GDN_DECODE_PIPELINED_V_PADDED];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float q_warp_sums[GDN_DECODE_PIPELINED_WARPS];
  __shared__ float k_warp_sums[GDN_DECODE_PIPELINED_WARPS];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

  if (tid < BK) {
    const float q_value = (float)row[hk * BK + tid];
    const float k_value = (float)row[key_dim + hk * BK + tid];
    q_buf[tid] = q_value;
    k_buf[tid] = k_value;
  }

  const float q_value = tid < BK ? q_buf[tid] : 0.0f;
  const float k_value = tid < BK ? k_buf[tid] : 0.0f;
  const float q_sum = gdn_warp_sum(q_value * q_value);
  const float k_sum = gdn_warp_sum(k_value * k_value);
  if (lane == 0) {
    q_warp_sums[warp] = q_sum;
    k_warp_sums[warp] = k_sum;
  }

  if (tid == 0) {
    const float b_value = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  __syncthreads();

  if (tid == 0) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
#pragma unroll
    for (int warp_idx = 0; warp_idx < GDN_DECODE_PIPELINED_WARPS;
         warp_idx++) {
      q_norm += q_warp_sums[warp_idx];
      k_norm += k_warp_sums[warp_idx];
    }
    q_mul = rsqrtf(q_norm + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(k_norm + 1.0e-6f);
  }
  __syncthreads();

  if (tid < BK) {
    q_buf[tid] *= q_mul;
    k_buf[tid] *= k_mul;
  }

  if (num_v_tiles > 0) {
#pragma unroll
    for (int vector_idx = tid; vector_idx < STATE_VECTORS;
         vector_idx += GDN_DECODE_PIPELINED_THREADS) {
      const int k_idx = vector_idx / VECTORS_PER_ROW;
      const int v_vector = vector_idx % VECTORS_PER_ROW;
      const StateT *src = state_bh + k_idx * head_v_dim + v_vector * 4;
      float *dst = state_buf[0] +
                   k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
      if constexpr (sizeof(StateT) == sizeof(float)) {
        gdn_cp_async_cg_16(dst, src);
      } else {
        *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
      }
    }
    gdn_cp_async_commit();
  }
  __syncthreads();

  for (int v_tile = 0; v_tile < num_v_tiles; v_tile++) {
    const int stage = v_tile % GDN_DECODE_PIPELINED_STAGES;
    const int next_v_tile = v_tile + 1;
    gdn_cp_async_wait();
    __syncthreads();

    if (next_v_tile < num_v_tiles) {
      const int next_stage = next_v_tile % GDN_DECODE_PIPELINED_STAGES;
#pragma unroll
      for (int vector_idx = tid; vector_idx < STATE_VECTORS;
           vector_idx += GDN_DECODE_PIPELINED_THREADS) {
        const int k_idx = vector_idx / VECTORS_PER_ROW;
        const int v_vector = vector_idx % VECTORS_PER_ROW;
        const StateT *src = state_bh + k_idx * head_v_dim +
                            next_v_tile * BV + v_vector * 4;
        float *dst = state_buf[next_stage] +
                     k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
        if constexpr (sizeof(StateT) == sizeof(float)) {
          gdn_cp_async_cg_16(dst, src);
        } else {
          *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
        }
      }
      gdn_cp_async_commit();
    }

    float state_dot_k = 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
      const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
      const float old_state =
          state_buf[stage][k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_in_tile] *
          decay_t;
      state_dot_k = __fmaf_rn(old_state, k_buf[k_idx], state_dot_k);
    }
    state_dot_k =
        gdn_grouped_k_sum<GDN_DECODE_PIPELINED_VALUES_PER_WARP>(state_dot_k);

    const int v_idx = v_tile * BV + v_in_tile;
    float delta = 0.0f;
    if (k_lane == 0) {
      const float v_value = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
      delta = (v_value - state_dot_k) * beta_t;
    }
    delta = __shfl_sync(0xffffffff, delta, v_local);

    float state_dot_q = 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
      const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
      const int state_idx =
          k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_in_tile;
      const float old_state = state_buf[stage][state_idx] * decay_t;
      const float new_state = __fmaf_rn(k_buf[k_idx], delta, old_state);
      state_buf[stage][state_idx] = new_state;
      state_dot_q = __fmaf_rn(new_state, q_buf[k_idx], state_dot_q);
    }
    state_dot_q =
        gdn_grouped_k_sum<GDN_DECODE_PIPELINED_VALUES_PER_WARP>(state_dot_q);
    if (k_lane == 0) {
      out_bh[v_idx] = (T)state_dot_q;
    }
    __syncthreads();

#pragma unroll
    for (int vector_idx = tid; vector_idx < STATE_VECTORS;
         vector_idx += GDN_DECODE_PIPELINED_THREADS) {
      const int k_idx = vector_idx / VECTORS_PER_ROW;
      const int v_vector = vector_idx % VECTORS_PER_ROW;
      const float *src = state_buf[stage] +
                         k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
      StateT *dst =
          state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
      gdn_store_state_x4(dst, *reinterpret_cast<const float4 *>(src));
    }
    __syncthreads();
  }
}

template <typename T, typename StateT, int BK, int BV>
__global__ void gdn_decode_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;

  if (bidx >= batch_size)
    return;

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (v_idx < head_v_dim) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ float red_q[BV];
  __shared__ float red_k[BV];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;
  __shared__ float state_buf[BK * BV];

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < BK; d += BV) {
    float q_val = (float)row[hk * BK + d];
    float k_val = (float)row[key_dim + hk * BK + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = BV >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);
    float b_val = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    float a_val =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_t = 1.0f / (1.0f + expf(-b_val));
    decay_t = expf(-expf(a_log[hv]) * softplus_val);
  }
  __syncthreads();

  for (int d = tid; d < BK; d += BV) {
    q_buf[d] = (float)row[hk * BK + d] * q_mul;
    k_buf[d] = (float)row[key_dim + hk * BK + d] * k_mul;
  }
  __syncthreads();

  if (v_idx >= head_v_dim)
    return;

  float v_t = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
  float kv_mem = 0.0f;
#pragma unroll GDN_DECODE_STATE_LOAD_UNROLL
  for (int j = 0; j < BK; j++) {
      const float s = (float)state_bh[j * head_v_dim + v_idx] * decay_t;
    state_buf[j * BV + tid] = s;
    kv_mem = __fmaf_rn(s, k_buf[j], kv_mem);
  }

  float delta = (v_t - kv_mem) * beta_t;
  float y_t = 0.0f;
  static_assert(BK % GDN_DECODE_STATE_UPDATE_TILE_ROWS == 0,
                "BK must be divisible by the update tile");
#pragma unroll 1
  for (int base = 0; base < BK; base += GDN_DECODE_STATE_UPDATE_TILE_ROWS) {
#pragma unroll GDN_DECODE_STATE_UPDATE_TILE_ROWS
    for (int offset = 0; offset < GDN_DECODE_STATE_UPDATE_TILE_ROWS; offset++) {
      const int j = base + offset;
      const float s = __fmaf_rn(k_buf[j], delta, state_buf[j * BV + tid]);
      state_bh[j * head_v_dim + v_idx] = s;
      y_t = __fmaf_rn(s, q_buf[j], y_t);
    }
  }
  out_bh[v_idx] = (T)y_t;
}

template <typename T, typename StateT, int BV, int MAX_K>
__global__ void gdn_decode_recurrence_kernel_fallback(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;

  if (bidx >= batch_size)
    return;

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (v_idx < head_v_dim) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * head_k_dim * head_v_dim;

  extern __shared__ float shared[];
  float *red_q = shared;
  float *red_k = red_q + BV;
  float *q_buf = red_k + BV;
  float *k_buf = q_buf + head_k_dim;

  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < head_k_dim; d += BV) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = BV >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)head_k_dim);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);
    float b_val = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    float a_val =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_t = 1.0f / (1.0f + expf(-b_val));
    decay_t = expf(-expf(a_log[hv]) * softplus_val);
  }
  __syncthreads();

  for (int d = tid; d < head_k_dim; d += BV) {
    q_buf[d] = (float)row[hk * head_k_dim + d] * q_mul;
    k_buf[d] = (float)row[key_dim + hk * head_k_dim + d] * k_mul;
  }
  __syncthreads();

  if (v_idx >= head_v_dim)
    return;

  float s[MAX_K];
  for (int j = 0; j < head_k_dim; j++) {
    s[j] = (float)state_bh[j * head_v_dim + v_idx] * decay_t;
  }

  float v_t = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
  float kv_mem = 0.0f;
  for (int j = 0; j < head_k_dim; j++) {
    kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
  }

  float delta = (v_t - kv_mem) * beta_t;
  float y_t = 0.0f;
  for (int j = 0; j < head_k_dim; j++) {
    s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
    y_t = __fmaf_rn(s[j], q_buf[j], y_t);
  }

  for (int j = 0; j < head_k_dim; j++) {
    state_bh[j * head_v_dim + v_idx] = s[j];
  }
  out_bh[v_idx] = (T)y_t;
}

template <typename T, typename StateT>
void launch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, StateT *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, cudaStream_t stream) {
  if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4 ||
      kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_32) {
    if (head_k_dim == GDN_DECODE_VALUE_MAJOR_K &&
        head_v_dim == GDN_DECODE_VALUE_MAJOR_V) {
      const int bh = batch_size * num_v_heads;
      if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4) {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 4>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 4), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      } else {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 32>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 32), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      }
    }
    return;
  }
  constexpr int BV = GDN_DECODE_VALUE_TILE;
  dim3 grid((head_v_dim + BV - 1) / BV, batch_size * num_v_heads);
  dim3 block(BV);

  if (kernel_kind == GDN_DECODE_KERNEL_PIPELINED &&
      head_k_dim == GDN_DECODE_PIPELINED_K &&
      head_v_dim % GDN_DECODE_PIPELINED_V == 0) {
    dim3 pipelined_grid(batch_size * num_v_heads);
    dim3 pipelined_block(GDN_DECODE_PIPELINED_THREADS);
    gdn_decode_recurrence_kernel_pipelined<T, StateT>
        <<<pipelined_grid, pipelined_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (kernel_kind == GDN_DECODE_KERNEL_COOPERATIVE &&
             head_k_dim == GDN_DECODE_COOPERATIVE_K &&
             head_v_dim % GDN_DECODE_COOPERATIVE_V == 0) {
    constexpr int COOPERATIVE_BV = GDN_DECODE_COOPERATIVE_V;
    dim3 cooperative_grid(head_v_dim / COOPERATIVE_BV,
                          batch_size * num_v_heads);
    dim3 cooperative_block(GDN_DECODE_COOPERATIVE_THREADS);
    gdn_decode_recurrence_kernel_cooperative<T, StateT>
        <<<cooperative_grid, cooperative_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 128) {
    gdn_decode_recurrence_kernel<T, StateT, 128, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 64) {
    gdn_decode_recurrence_kernel<T, StateT, 64, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else {
    constexpr int MAX_K = 256;
    size_t smem = (2 * BV + 2 * head_k_dim) * sizeof(float);
    gdn_decode_recurrence_kernel_fallback<T, StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  }
}

template <typename T>
void dispatch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, void *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int state_dtype,
    cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__half *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__nv_bfloat16 *)state, output,
        batch_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        tiled_v_heads, b_batch_stride, b_head_stride, a_batch_stride,
        a_head_stride, slot_indices, kernel_kind, stream);
  } else {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (float *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  }
}

extern "C" void gdn_decode_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *state, void *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int dtype, int state_dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_decode_recurrence(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
        a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride,
        b_head_stride, a_batch_stride, a_head_stride, slot_indices, kernel_kind,
        state_dtype, custream);
  } else {
    dispatch_gdn_decode_recurrence(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, state,
        (__nv_bfloat16 *)output, batch_size, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride, b_head_stride,
        a_batch_stride, a_head_stride, slot_indices, kernel_kind, state_dtype,
        custream);
  }
}

__device__ __forceinline__ float gdn_silu(float x) {
  if (isnan(x)) {
    return x;
  }
  if (isinf(x)) {
    return x > 0.0f ? x : 0.0f;
  }
  if (x >= 0.0f) {
    return x / (1.0f + expf(-x));
  }
  const float ex = expf(x);
  return x * ex / (1.0f + ex);
}

__device__ __forceinline__ float gdn_warp_max(float value, int width = 32) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset, width));
  }
  return value;
}

__device__ __forceinline__ size_t gdn_fp8_scale_offset(
    int normalized_row, int groups, int scale_stride_m) {
  const int projection_row = normalized_row / groups;
  const int group = normalized_row - projection_row * groups;
  return (size_t)group * scale_stride_m + projection_row;
}

__device__ __forceinline__ void gdn_fp8_quant_params(
    float maximum, int scale_layout, float &quant_scale,
    float &inverse_quant_scale) {
  if (scale_layout == 0) {
    inverse_quant_scale = fmaxf(maximum, 1.0e-10f) / 448.0f;
    quant_scale = 1.0f / inverse_quant_scale;
    return;
  }
  quant_scale = maximum == 0.0f ? 1.0f : 448.0f / maximum;
  inverse_quant_scale = 1.0f / quant_scale;
}

__device__ __forceinline__ gdn_fp8_e4m3 gdn_fp8_quantize(
    float value, float quant_scale, float inverse_quant_scale,
    int scale_layout) {
  const float scaled = scale_layout == 0
                           ? value / inverse_quant_scale
                           : value * quant_scale;
  return gdn_fp8_e4m3(fminf(fmaxf(scaled, -448.0f), 448.0f));
}

template <typename T>
__global__ void
gdn_rmsnorm_gated_kernel(const T *__restrict__ x, const T *__restrict__ gate,
                         const T *__restrict__ weight, T *__restrict__ output,
                         int rows, int hidden_dim, int outer_dim_1,
                         int outer_dim_2, int64_t x_stride_0,
                         int64_t x_stride_1, int64_t x_stride_2,
                         int64_t x_stride_3, int64_t gate_stride_0,
                         int64_t gate_stride_1, int64_t gate_stride_2,
                         int64_t gate_stride_3, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const size_t x_row_offset = (size_t)outer_0 * x_stride_0 +
                              (size_t)outer_1 * x_stride_1 +
                              (size_t)outer_2 * x_stride_2;
  const size_t gate_row_offset = (size_t)outer_0 * gate_stride_0 +
                                 (size_t)outer_1 * gate_stride_1 +
                                 (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * hidden_dim;

  float sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float x_val = (float)x[x_row_offset + (size_t)i * x_stride_3];
    sum = __fmaf_rn(x_val, x_val, sum);
  }

  __shared__ float smem[256];
  smem[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(smem[0] / (float)hidden_dim + eps);
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    const float gate_val =
        (float)gate[gate_row_offset + (size_t)i * gate_stride_3];
    const float out =
        (float)x[x_row_offset + (size_t)i * x_stride_3] * inv_rms *
        (float)weight[i] * gdn_silu(gate_val);
    out_row[i] = (T)out;
  }
}

template <typename T, int HIDDEN_DIM, int ROWS_PER_BLOCK>
__global__ void gdn_rmsnorm_gated_warp_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, float eps) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const T *x_row = x + (size_t)outer_0 * x_stride_0 +
                   (size_t)outer_1 * x_stride_1 +
                   (size_t)outer_2 * x_stride_2;
  const T *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                      (size_t)outer_1 * gate_stride_1 +
                      (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * HIDDEN_DIM;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[i];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = (T)(values[i / 32] * inv_rms * (float)weight[i] *
                     gdn_silu((float)gate_row[i]));
  }
}

template <int ROWS_PER_BLOCK>
__global__ void gdn_rmsnorm_gated_quantized_warp_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, int outer_dim_1,
    int outer_dim_2, int64_t x_stride_0, int64_t x_stride_1,
    int64_t x_stride_2, int64_t x_stride_3, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, int64_t gate_stride_3,
    float eps) {
  constexpr int HIDDEN_DIM = GDN_RMSNORM_FAST_HIDDEN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const __nv_bfloat16 *x_row = x + (size_t)outer_0 * x_stride_0 +
                               (size_t)outer_1 * x_stride_1 +
                               (size_t)outer_2 * x_stride_2;
  const __nv_bfloat16 *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                                  (size_t)outer_1 * gate_stride_1 +
                                  (size_t)outer_2 * gate_stride_2;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[(size_t)i * x_stride_3];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

  __nv_bfloat16 rounded[HIDDEN_DIM / 32];
  float maximum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    rounded[i / 32] = __float2bfloat16_rn(
        values[i / 32] * inv_rms * (float)weight[i] *
        gdn_silu((float)gate_row[(size_t)i * gate_stride_3]));
    maximum = fmaxf(maximum, fabsf((float)rounded[i / 32]));
  }
  maximum = gdn_warp_max(maximum);
  float quant_scale;
  float inverse_quant_scale;
  gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                       inverse_quant_scale);
  if (lane == 0) {
    scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
        inverse_quant_scale;
  }
  gdn_fp8_e4m3 *out_row = output + (size_t)row * HIDDEN_DIM;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = gdn_fp8_quantize((float)rounded[i / 32], quant_scale,
                                  inverse_quant_scale, scale_layout);
  }
}

template <typename T> struct alignas(16) gdn_rmsnorm_vec8 {
  T data[GDN_RMSNORM_TILED_VALUES_PER_LANE];
};

template <typename T>
__global__ void gdn_rmsnorm_gated_rows4_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    float eps) {
  using Vec = gdn_rmsnorm_vec8<T>;
  constexpr int vecs_per_row =
      GDN_RMSNORM_FAST_HIDDEN / GDN_RMSNORM_TILED_VALUES_PER_LANE;
  constexpr unsigned warp_mask = 0xffffffffu;

  const int lane = threadIdx.x;
  const int half_warp = lane / GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int lane_in_half = lane % GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int first_row =
      blockIdx.x * GDN_RMSNORM_TILED_ROWS_PER_BLOCK + half_warp;
  const int second_row = first_row + GDN_RMSNORM_TILED_ROW_PAIR_OFFSET;
  const Vec weight_value =
      reinterpret_cast<const Vec *>(weight)[lane_in_half];

  Vec first_value{};
  Vec second_value{};
  if (first_row < rows) {
    first_value = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                   lane_in_half];
  }
  if (second_row < rows) {
    second_value = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                    lane_in_half];
  }

  float first_sum = 0.0f;
  float second_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)first_value.data[i];
    const float second = (float)second_value.data[i];
    first_sum = __fmaf_rn(first, first, first_sum);
    second_sum = __fmaf_rn(second, second, second_sum);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    first_sum += __shfl_down_sync(warp_mask, first_sum, offset,
                                  GDN_RMSNORM_TILED_LANES_PER_ROW);
    second_sum += __shfl_down_sync(warp_mask, second_sum, offset,
                                   GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float first_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, first_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);
  const float second_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, second_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);

  if (first_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[first_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)first_value.data[i] * first_inv_rms *
              (float)weight_value.data[i] *
              gdn_silu((float)gate_value.data[i]));
    }
    reinterpret_cast<Vec *>(output)[first_row * vecs_per_row + lane_in_half] =
        result;
  }
  if (second_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[second_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)second_value.data[i] * second_inv_rms *
              (float)weight_value.data[i] *
              gdn_silu((float)gate_value.data[i]));
    }
    reinterpret_cast<Vec *>(
        output)[second_row * vecs_per_row + lane_in_half] = result;
  }
}

__global__ void gdn_rmsnorm_gated_quantized_rows4_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, float eps) {
  using Vec = gdn_rmsnorm_vec8<__nv_bfloat16>;
  constexpr int vecs_per_row =
      GDN_RMSNORM_FAST_HIDDEN / GDN_RMSNORM_TILED_VALUES_PER_LANE;
  constexpr unsigned warp_mask = 0xffffffffu;

  const int lane = threadIdx.x;
  const int half_warp = lane / GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int lane_in_half = lane % GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int first_row =
      blockIdx.x * GDN_RMSNORM_TILED_ROWS_PER_BLOCK + half_warp;
  const int second_row = first_row + GDN_RMSNORM_TILED_ROW_PAIR_OFFSET;
  const Vec weight_value =
      reinterpret_cast<const Vec *>(weight)[lane_in_half];

  Vec values[2]{};
  if (first_row < rows) {
    values[0] = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                  lane_in_half];
  }
  if (second_row < rows) {
    values[1] = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                  lane_in_half];
  }

  float sums[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)values[0].data[i];
    const float second = (float)values[1].data[i];
    sums[0] = __fmaf_rn(first, first, sums[0]);
    sums[1] = __fmaf_rn(second, second, sums[1]);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    sums[0] += __shfl_down_sync(warp_mask, sums[0], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
    sums[1] += __shfl_down_sync(warp_mask, sums[1], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float inv_rms[2] = {
      rsqrtf(__shfl_sync(warp_mask, sums[0], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps),
      rsqrtf(__shfl_sync(warp_mask, sums[1], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps)};

#pragma unroll
  for (int row_index = 0; row_index < 2; ++row_index) {
    const int row = row_index == 0 ? first_row : second_row;
    if (row >= rows) {
      continue;
    }
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[row * vecs_per_row + lane_in_half];
    __nv_bfloat16 rounded[GDN_RMSNORM_TILED_VALUES_PER_LANE];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      rounded[i] = __float2bfloat16_rn(
          (float)values[row_index].data[i] * inv_rms[row_index] *
          (float)weight_value.data[i] * gdn_silu((float)gate_value.data[i]));
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
    }
    maximum = gdn_warp_max(maximum, GDN_RMSNORM_TILED_LANES_PER_ROW);
    float quant_scale;
    float inverse_quant_scale;
    gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                         inverse_quant_scale);
    if (lane_in_half == 0) {
      scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
          inverse_quant_scale;
    }
    gdn_fp8_e4m3 *out = output + (size_t)row * GDN_RMSNORM_FAST_HIDDEN +
                          lane_in_half * GDN_RMSNORM_TILED_VALUES_PER_LANE;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      out[i] = gdn_fp8_quantize((float)rounded[i], quant_scale,
                                inverse_quant_scale, scale_layout);
    }
  }
}

template <typename T>
__host__ __forceinline__ bool gdn_rmsnorm_vec8_aligned(const void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(gdn_rmsnorm_vec8<T>) == 0;
}

__host__ __forceinline__ bool gdn_rmsnorm_dense_rows(
    int rows, int outer_dim_1, int outer_dim_2, int64_t stride_0,
    int64_t stride_1, int64_t stride_2) {
  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_dim_0 = rows / outer_plane;
  return (outer_dim_2 <= 1 || stride_2 == GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_1 <= 1 ||
          stride_1 == (int64_t)outer_dim_2 * GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_0 <= 1 ||
          stride_0 == (int64_t)outer_plane * GDN_RMSNORM_FAST_HIDDEN);
}

extern "C" void gdn_rmsnorm_gated(const void *x, const void *gate,
                                  const void *weight, void *output, int rows,
                                  int hidden_dim, int outer_dim_1,
                                  int outer_dim_2, int64_t x_stride_0,
                                  int64_t x_stride_1, int64_t x_stride_2,
                                  int64_t x_stride_3, int64_t gate_stride_0,
                                  int64_t gate_stride_1, int64_t gate_stride_2,
                                  int64_t gate_stride_3, float eps, int dtype,
                                  int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_warp_kernel =
      hidden_dim == GDN_RMSNORM_FAST_HIDDEN && x_stride_3 == 1 &&
      gate_stride_3 == 1;
  if (use_warp_kernel) {
    const bool use_rows4_kernel =
        rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                               x_stride_1, x_stride_2) &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                               gate_stride_1, gate_stride_2);
    if (dtype == 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__half>(x) &&
        gdn_rmsnorm_vec8_aligned<__half>(gate) &&
        gdn_rmsnorm_vec8_aligned<__half>(weight) &&
        gdn_rmsnorm_vec8_aligned<__half>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __half *)x, (const __half *)gate, (const __half *)weight,
          (__half *)output, rows, eps);
      return;
    }
    if (dtype != 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows, eps);
      return;
    }

    const dim3 warp_block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
    const dim3 warp_grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                         GDN_RMSNORM_ROWS_PER_BLOCK);
    if (dtype == 0) {
      gdn_rmsnorm_gated_warp_kernel<__half, GDN_RMSNORM_FAST_HIDDEN,
                                     GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __half *)x, (const __half *)gate,
              (const __half *)weight, (__half *)output, rows, outer_dim_1,
              outer_dim_2, x_stride_0, x_stride_1, x_stride_2, gate_stride_0,
              gate_stride_1, gate_stride_2, eps);
    } else {
      gdn_rmsnorm_gated_warp_kernel<
          __nv_bfloat16, GDN_RMSNORM_FAST_HIDDEN,
          GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
              (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
              outer_dim_1, outer_dim_2, x_stride_0, x_stride_1, x_stride_2,
              gate_stride_0, gate_stride_1, gate_stride_2, eps);
    }
    return;
  }

  dim3 block(128);
  dim3 grid(rows);

  if (dtype == 0) {
    gdn_rmsnorm_gated_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)gate, (const __half *)weight,
        (__half *)output, rows, hidden_dim, outer_dim_1, outer_dim_2,
        x_stride_0, x_stride_1, x_stride_2, x_stride_3, gate_stride_0,
        gate_stride_1, gate_stride_2, gate_stride_3, eps);
  } else {
    gdn_rmsnorm_gated_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
        hidden_dim, outer_dim_1, outer_dim_2, x_stride_0, x_stride_1,
        x_stride_2, x_stride_3, gate_stride_0, gate_stride_1, gate_stride_2,
        gate_stride_3, eps);
  }
}

extern "C" void gdn_rmsnorm_gated_quantized_bf16(
    const void *x, const void *gate, const void *weight, void *output,
    float *scales, int rows, int groups, int scale_stride_m,
    int scale_layout, int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t x_stride_3,
    int64_t gate_stride_0, int64_t gate_stride_1, int64_t gate_stride_2,
    int64_t gate_stride_3, float eps, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_rows4_kernel =
      rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
      x_stride_3 == 1 && gate_stride_3 == 1 &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                             x_stride_1, x_stride_2) &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                             gate_stride_1, gate_stride_2) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight);
  if (use_rows4_kernel) {
    const dim3 block(32);
    const dim3 grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                    GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
    gdn_rmsnorm_gated_quantized_rows4_kernel<<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales, rows,
        groups, scale_stride_m, scale_layout, eps);
    return;
  }

  const dim3 block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
  const dim3 grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                  GDN_RMSNORM_ROWS_PER_BLOCK);
  gdn_rmsnorm_gated_quantized_warp_kernel<GDN_RMSNORM_ROWS_PER_BLOCK>
      <<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales,
          rows, groups, scale_stride_m, scale_layout, outer_dim_1,
          outer_dim_2, x_stride_0, x_stride_1, x_stride_2, x_stride_3,
          gate_stride_0, gate_stride_1, gate_stride_2, gate_stride_3, eps);
}

// ============================================================================
// Kernel 3: fused_gdn_gating
//
// Fuses: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// a_log and dt_bias are per-head (broadcast over batch*seq).
//
// b, a: [total]  a_log, dt_bias: [num_heads]
// beta_out, g_out: [total]
// ============================================================================

template <typename T>
__global__ void
fused_gdn_gating_kernel(const T *__restrict__ b,           // [total]
                        const T *__restrict__ a,           // [total]
                        const float *__restrict__ a_log,   // [num_heads]
                        const float *__restrict__ dt_bias, // [num_heads]
                        T *__restrict__ beta_out,          // [total]
                        T *__restrict__ g_out,             // [total]
                        int total_elements, int num_heads) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  // Head index: elements are laid out as [..., num_heads]
  int head_idx = idx % num_heads;

  // beta = sigmoid(b)
  float b_val = (float)b[idx];
  float beta = 1.0f / (1.0f + expf(-b_val));

  // g = -exp(a_log) * softplus(a + dt_bias)
  float a_val = (float)a[idx];
  float a_log_val = a_log[head_idx];
  float dt_bias_val = dt_bias[head_idx];

  float sp_input = a_val + dt_bias_val;
  float softplus_val = logf(1.0f + expf(sp_input));
  float g_val = -expf(a_log_val) * softplus_val;

  beta_out[idx] = (T)beta;
  g_out[idx] = (T)g_val;
}

extern "C" void fused_gdn_gating(const void *b, const void *a,
                                 const float *a_log, const float *dt_bias,
                                 void *beta_out, void *g_out,
                                 int total_elements, int num_heads, int dtype,
                                 int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(256);
  dim3 grid((total_elements + 255) / 256);

  if (dtype == 0) {
    fused_gdn_gating_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)b, (const __half *)a, a_log, dt_bias,
        (__half *)beta_out, (__half *)g_out, total_elements, num_heads);
  } else {
    fused_gdn_gating_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)b, (const __nv_bfloat16 *)a, a_log, dt_bias,
        (__nv_bfloat16 *)beta_out, (__nv_bfloat16 *)g_out, total_elements,
        num_heads);
  }
}

constexpr int GDN_SPEC_COMMIT_WARPS = 4;
constexpr int GDN_SPEC_COMMIT_MAX_K = 256;

template <typename T>
__global__ void gdn_speculative_conv_state_commit_kernel(
    const T *__restrict__ x, const T *__restrict__ initial_state,
    T *__restrict__ state_pool, const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int conv_dim, int kernel_size) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (channel >= conv_dim || batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows == 0) {
    return;
  }

  const T *prior = initial_state +
                   ((size_t)batch_idx * conv_dim + channel) * kernel_size;
  T *destination =
      state_pool +
      ((size_t)slot_indices[batch_idx] * conv_dim + channel) * kernel_size;
  const int pad = kernel_size - rows;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      destination[i] = prior[i + rows];
    } else {
      const int position = rows - kernel_size + i;
      destination[i] =
          x[((size_t)batch_idx * seq_len + position) * conv_dim + channel];
    }
  }
}

template <typename T, typename StateT, bool VALUE_MAJOR>
__global__ void gdn_speculative_recurrent_state_commit_kernel(
    const T *__restrict__ convolved_qkv, const T *__restrict__ b,
    const T *__restrict__ a,
    const StateT *__restrict__ initial_recurrent_state,
    const float *__restrict__ a_log, const float *__restrict__ dt_bias,
    StateT *__restrict__ recurrent_state_pool,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads) {
  constexpr int WARP_SIZE = 32;
  constexpr int K_PER_LANE = GDN_SPEC_COMMIT_MAX_K / WARP_SIZE;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_idx = blockIdx.x * GDN_SPEC_COMMIT_WARPS + warp;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows == 0) {
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head =
      tiled_v_heads ? value_head % num_k_heads : value_head / values_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t initial_head =
      ((size_t)batch_idx * num_v_heads + value_head) * head_k_dim *
      head_v_dim;
  const size_t destination_head =
      ((size_t)slot_indices[batch_idx] * num_v_heads + value_head) *
      head_k_dim * head_v_dim;
  const int thread_idx = warp * WARP_SIZE + lane;

  __shared__ float key_buffer[GDN_SPEC_COMMIT_MAX_K];
  __shared__ float norm_buffer[WARP_SIZE * GDN_SPEC_COMMIT_WARPS];
  __shared__ float key_multiplier;
  __shared__ float beta;
  __shared__ float decay;
  __shared__ float values[GDN_SPEC_COMMIT_WARPS];

  float state[K_PER_LANE];
#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      state[r] = initial_recurrent_state[initial_head + offset];
    }
  }

  for (int position = 0; position < rows; position++) {
    float key_norm_partial = 0.0f;
    for (int key_idx = thread_idx; key_idx < head_k_dim;
         key_idx += WARP_SIZE * GDN_SPEC_COMMIT_WARPS) {
      const int channel = key_dim + key_head * head_k_dim + key_idx;
      const float value =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
      key_buffer[key_idx] = value;
      key_norm_partial = __fmaf_rn(value, value, key_norm_partial);
    }
    norm_buffer[thread_idx] = key_norm_partial;
    __syncthreads();
    for (int stride = (WARP_SIZE * GDN_SPEC_COMMIT_WARPS) / 2; stride > 0;
         stride >>= 1) {
      if (thread_idx < stride) {
        norm_buffer[thread_idx] += norm_buffer[thread_idx + stride];
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      key_multiplier = rsqrtf(norm_buffer[0] + 1.0e-6f);
      const size_t gate_offset =
          ((size_t)batch_idx * seq_len + position) * num_v_heads +
          value_head;
      const float b_value = (float)b[gate_offset];
      const float a_value = (float)a[gate_offset] + dt_bias[value_head];
      const float softplus =
          a_value > 20.0f
              ? a_value
              : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                                : log1pf(expf(a_value)));
      beta = 1.0f / (1.0f + expf(-b_value));
      decay = expf(-expf(a_log[value_head]) * softplus);
    }
    if (lane == 0) {
      const int channel = 2 * key_dim + value_head * head_v_dim + value_idx;
      values[warp] =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
    }
    __syncthreads();

    float state_dot_key = 0.0f;
#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state_dot_key = __fmaf_rn(state[r], key, state_dot_key);
      }
    }
    state_dot_key = gdn_warp_sum<WARP_SIZE>(state_dot_key);
    const float delta = (values[warp] - decay * state_dot_key) * beta;

#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state[r] = __fmaf_rn(key, delta, decay * state[r]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      recurrent_state_pool[destination_head + offset] = state[r];
    }
  }
}

template <typename T, typename StateT>
void launch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const StateT *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    StateT *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, cudaStream_t stream) {
  dim3 conv_block(GDN_CHANNEL_BLOCK_SIZE);
  const int conv_dim =
      2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
  dim3 conv_grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                     GDN_CHANNEL_BLOCK_SIZE,
                 batch_size);
  gdn_speculative_conv_state_commit_kernel<T>
      <<<conv_grid, conv_block, 0, stream>>>(
          mixed_qkv, initial_conv_state, conv_state_pool, keep_rows,
          slot_indices, batch_size, seq_len, conv_dim, kernel_size);

  dim3 recurrence_block(32, GDN_SPEC_COMMIT_WARPS);
  dim3 recurrence_grid(
      (head_v_dim + GDN_SPEC_COMMIT_WARPS - 1) / GDN_SPEC_COMMIT_WARPS,
      batch_size * num_v_heads);
  if (value_major) {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, true>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads);
  } else {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, false>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads);
  }
}

template <typename T>
void dispatch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const void *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    void *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, int state_dtype, cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __half *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__half *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __nv_bfloat16 *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__nv_bfloat16 *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const float *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (float *)recurrent_state_pool, keep_rows, slot_indices,
        batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        kernel_size, tiled_v_heads, value_major, stream);
  }
}

extern "C" void gdn_speculative_state_commit(
    const void *mixed_qkv, const void *convolved_qkv, const void *b,
    const void *a, const void *initial_conv_state,
    const void *initial_recurrent_state, const float *a_log,
    const float *dt_bias, void *conv_state_pool, void *recurrent_state_pool,
    const uint32_t *keep_rows, const uint32_t *slot_indices, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int kernel_size, int tiled_v_heads, int value_major,
    int dtype, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_speculative_state_commit(
        (const __half *)mixed_qkv, (const __half *)convolved_qkv,
        (const __half *)b, (const __half *)a,
        (const __half *)initial_conv_state, initial_recurrent_state, a_log,
        dt_bias, (__half *)conv_state_pool, recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        state_dtype, custream);
  } else {
    dispatch_gdn_speculative_state_commit(
        (const __nv_bfloat16 *)mixed_qkv,
        (const __nv_bfloat16 *)convolved_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a,
        (const __nv_bfloat16 *)initial_conv_state, initial_recurrent_state,
        a_log, dt_bias, (__nv_bfloat16 *)conv_state_pool,
        recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, kernel_size,
        tiled_v_heads, value_major, state_dtype, custream);
  }
}

constexpr int GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH = 16;
constexpr int GDN_SPEC_CHECKPOINT_MAX_K = 256;
constexpr int GDN_SPEC_CHECKPOINT_VALUE_TILE = 32;
constexpr int GDN_SPEC_CHECKPOINT_WARPS = 4;
constexpr int GDN_SPEC_CHECKPOINT_VALUES_PER_WARP =
    GDN_SPEC_CHECKPOINT_VALUE_TILE / GDN_SPEC_CHECKPOINT_WARPS;
constexpr int GDN_SPEC_FUSED_MAX_TOKENS = 8;
constexpr int GDN_SPEC_FUSED_THREADS = 256;
constexpr int GDN_SPEC_FUSED_WARPS = GDN_SPEC_FUSED_THREADS / 32;
constexpr int GDN_SPEC_FUSED_VALUE_CHUNK = 32;
constexpr int GDN_SPEC_FUSED_VALUE_CHUNKS =
    GDN_DECODE_VALUE_MAJOR_V / GDN_SPEC_FUSED_VALUE_CHUNK;
constexpr int GDN_SPEC_FUSED_VALUES_PER_WARP =
    GDN_SPEC_FUSED_VALUE_CHUNK / GDN_SPEC_FUSED_WARPS;
// Paired reductions favor 16-bit states and underfilled or saturated F32 grids; serial sustains mid-grid bandwidth.
constexpr int GDN_SPEC_FUSED_PAIR_LOW_GRID_MAX = 144;
constexpr int GDN_SPEC_FUSED_PAIR_HIGH_GRID_MIN = 768;
constexpr uint32_t GDN_SPEC_CHECKPOINT_PAD_SLOT = 0xffffffffu;
constexpr int GDN_DEFERRED_STATE_DEPTH = 4;

__device__ __forceinline__ size_t
gdn_spec_checkpoint_base(uint32_t active_slot, int checkpoint_lanes) {
  return ((size_t)active_slot / checkpoint_lanes) * checkpoint_lanes;
}

template <typename T>
__global__ void gdn_speculative_conv_checkpoints_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int kernel_size, int checkpoint_lanes,
    bool write_checkpoints, T *pending_conv_input,
    const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  T state[GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH];
  T *source = channel_valid
                  ? state_pool +
                        ((size_t)active_slot * conv_dim + channel) * kernel_size
                  : nullptr;
  if (channel_valid) {
    for (int i = 0; i < kernel_size; i++) {
      state[i] = source[i];
    }
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
      for (int i = 0; i < kernel_size - 1; i++) {
        state[i] = state[i + 1];
      }
      state[kernel_size - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    for (int i = 0; i < kernel_size; i++) {
      source[i] = state[i];
    }
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const T *channel_weight = weight + (size_t)channel * kernel_size;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
    for (int i = 0; i < kernel_size - 1; i++) {
      state[i] = state[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state[kernel_size - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      acc = __fmaf_rn((float)state[i], (float)channel_weight[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;

    if (write_checkpoints) {
      T *destination =
          state_pool +
          (((base_slot + position) * conv_dim + channel) * kernel_size);
      for (int i = 0; i < kernel_size; i++) {
        destination[i] = state[i];
      }
    }
  }
}

template <typename T>
__global__ void gdn_speculative_conv_checkpoints_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  auto *states = reinterpret_cast<GdnConvWidth4<T> *>(state_pool);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  GdnConvWidth4<T> state;
  if (channel_valid) {
    state = states[(size_t)active_slot * conv_dim + channel];
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
#pragma unroll
      for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
        state.values[i] = state.values[i + 1];
      }
      state.values[GDN_PACKED_CONV_WIDTH - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    states[(size_t)active_slot * conv_dim + channel] = state;
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const GdnConvWidth4<T> channel_weight = weights[channel];
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
      state.values[i] = state.values[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state.values[GDN_PACKED_CONV_WIDTH - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
      acc = __fmaf_rn((float)state.values[i],
                      (float)channel_weight.values[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;
    if (write_checkpoints) {
      states[(base_slot + position) * conv_dim + channel] = state;
    }
  }
}

template <typename T>
void launch_gdn_speculative_conv_checkpoints(
    const T *x, const T *weight, T *state_pool, T *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    cudaStream_t stream) {
  if (pending_conv_input != nullptr &&
      (write_checkpoints || max_pending_rows < seq_len ||
       max_pending_rows > GDN_SPEC_FUSED_MAX_TOKENS ||
       pending_capacity <= 0 || conv_blocks <= 0 ||
       pending_keep_rows == nullptr || pending_epochs == nullptr ||
       conv_applied_epochs == nullptr)) {
    return;
  }
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                GDN_CHANNEL_BLOCK_SIZE,
            batch_size);
  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    gdn_speculative_conv_checkpoints_width4_kernel<T>
        <<<grid, block, 0, stream>>>(
            x, weight, state_pool, output, active_slots, batch_size, seq_len,
            conv_dim, checkpoint_lanes, write_checkpoints, pending_conv_input,
            pending_keep_rows, pending_epochs, conv_applied_epochs,
            max_pending_rows, pending_capacity, conv_blocks, x_stride_b,
            x_stride_s, x_stride_c);
  } else {
    gdn_speculative_conv_checkpoints_kernel<T><<<grid, block, 0, stream>>>(
        x, weight, state_pool, output, active_slots, batch_size, seq_len,
        conv_dim, kernel_size, checkpoint_lanes, write_checkpoints,
        pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c);
  }
}

extern "C" void gdn_speculative_conv_checkpoints(
    const void *x, const void *weight, void *state_pool, void *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, int write_checkpoints,
    void *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c, int dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    launch_gdn_speculative_conv_checkpoints(
        (const __half *)x, (const __half *)weight, (__half *)state_pool,
        (__half *)output, active_slots, batch_size, seq_len, conv_dim,
        kernel_size, checkpoint_lanes, write_checkpoints != 0,
        (__half *)pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c, custream);
  } else {
    launch_gdn_speculative_conv_checkpoints(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)state_pool, (__nv_bfloat16 *)output, active_slots,
        batch_size, seq_len, conv_dim, kernel_size, checkpoint_lanes,
        write_checkpoints != 0,
        (__nv_bfloat16 *)pending_conv_input, pending_keep_rows,
        pending_epochs, conv_applied_epochs, max_pending_rows,
        pending_capacity, conv_blocks, x_stride_b, x_stride_s, x_stride_c,
        custream);
  }
}

__device__ __forceinline__ float gdn_spec_softplus(float value) {
  return value > 20.0f
             ? value
             : (value > 0.0f ? value + log1pf(expf(-value))
                             : log1pf(expf(value)));
}

__device__ __forceinline__ float2 gdn_spec_warp_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset);
    y += __shfl_xor_sync(0xffffffff, y, offset);
  }
  return make_float2(x, y);
}

template <typename StateT>
__device__ __forceinline__ void gdn_spec_copy_state_chunk(
    StateT *shared_state, const StateT *state, int chunk, int thread) {
  constexpr int ELEMENTS_PER_COPY = 16 / sizeof(StateT);
  constexpr int ELEMENTS_PER_CHUNK =
      GDN_SPEC_FUSED_VALUE_CHUNK * GDN_DECODE_VALUE_MAJOR_K;
  constexpr int COPIES_PER_CHUNK = ELEMENTS_PER_CHUNK / ELEMENTS_PER_COPY;
  for (int copy = thread; copy < COPIES_PER_CHUNK;
       copy += GDN_SPEC_FUSED_THREADS) {
    const int element = copy * ELEMENTS_PER_COPY;
    gdn_cp_async_cg_16(shared_state + element,
                       state + chunk * ELEMENTS_PER_CHUNK + element);
  }
  gdn_cp_async_commit();
}

// Adapted from vLLM revision c8438a3d40168ce1d9eade0dc15ccbe5d27adb68.
// Copyright contributors to the vLLM project; Apache-2.0. See third_party/flashinfer_gdn.
template <typename T, typename StateT, bool PAIRED_REDUCTIONS,
          bool DIRECT_TRANSITIONS>
__global__ __launch_bounds__(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel(
        const T *__restrict__ mixed_qkv, const T *__restrict__ b,
        const T *__restrict__ a, const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
        T *__restrict__ output, gdn_fp8_e4m3 *__restrict__ quantized_output,
        float *__restrict__ output_scales, int scale_stride_m,
        int scale_layout, const uint32_t *__restrict__ active_slots,
        const T *__restrict__ gate, const T *__restrict__ norm_weight,
        float *__restrict__ transition_key,
        float *__restrict__ transition_delta,
        float *__restrict__ transition_decay,
        float *pending_key_banks,
        const uint32_t *__restrict__ pending_key_bank, float *pending_delta,
        float *pending_decay,
        const uint32_t *__restrict__ pending_keep_rows,
        const uint32_t *__restrict__ pending_epochs,
        uint32_t *__restrict__ recurrent_applied_epochs,
        int max_pending_rows, int pending_capacity,
        int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
        int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
        int64_t gate_stride_b, int64_t gate_stride_s,
        int64_t gate_stride_h, int64_t gate_stride_v, int batch_size,
        int seq_len, int num_k_heads, int num_v_heads,
        int checkpoint_lanes, int tiled_v_heads, float norm_eps) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int linear = tid; linear < seq_len * V;
         linear += GDN_SPEC_FUSED_THREADS) {
      const int position = linear / V;
      const int value = linear - position * V;
      const size_t offset =
          (((size_t)batch_idx * seq_len + position) * num_v_heads +
           value_head) *
              V +
          value;
      if (quantized_output != nullptr) {
        quantized_output[offset] = gdn_fp8_e4m3(0.0f);
      } else {
        output[offset] = (T)0.0f;
      }
    }
    if (quantized_output != nullptr) {
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(0.0f, scale_layout, quant_scale,
                           inverse_quant_scale);
      (void)quant_scale;
      for (int position = tid; position < seq_len;
           position += GDN_SPEC_FUSED_THREADS) {
        const int normalized_row =
            ((batch_idx * seq_len + position) * num_v_heads) + value_head;
        output_scales[gdn_fp8_scale_offset(normalized_row, num_v_heads,
                                           scale_stride_m)] =
            inverse_quant_scale;
      }
    }
    if constexpr (!DIRECT_TRANSITIONS) {
      if (transition_delta == nullptr) {
        return;
      }
      for (int linear = tid; linear < seq_len * V;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / V;
        const int value = linear - position * V;
        transition_delta[(((size_t)batch_idx * seq_len + position) *
                              num_v_heads +
                          value_head) *
                             V +
                         value] = 0.0f;
      }
      for (int position = tid; position < seq_len;
           position += GDN_SPEC_FUSED_THREADS) {
        transition_decay[((size_t)batch_idx * seq_len + position) *
                             num_v_heads +
                         value_head] = 0.0f;
      }
      const bool key_leader =
          tiled_v_heads ? value_head < num_k_heads
                        : value_head % (num_v_heads / num_k_heads) == 0;
      if (key_leader) {
        const int key_head = tiled_v_heads
                                 ? value_head % num_k_heads
                                 : value_head / (num_v_heads / num_k_heads);
        for (int linear = tid; linear < seq_len * K;
             linear += GDN_SPEC_FUSED_THREADS) {
          const int position = linear / K;
          const int key = linear - position * K;
          transition_key[(((size_t)batch_idx * seq_len + position) *
                              num_k_heads +
                          key_head) *
                             K +
                         key] = 0.0f;
        }
      }
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)V * K;
  StateT *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) *
                       state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);
  const size_t transition_row_base = (size_t)batch_idx * seq_len;

  __shared__ __align__(16) StateT shared_state[GDN_SPEC_FUSED_VALUE_CHUNK][K];
  __shared__ float shared_query[GDN_SPEC_FUSED_MAX_TOKENS][K];
  __shared__ float shared_key[GDN_SPEC_FUSED_MAX_TOKENS][K];
  __shared__ T shared_value[GDN_SPEC_FUSED_MAX_TOKENS][V];
  __shared__ T shared_output[GDN_SPEC_FUSED_MAX_TOKENS][V];
  __shared__ float shared_decay[GDN_SPEC_FUSED_MAX_TOKENS];
  __shared__ float shared_beta[GDN_SPEC_FUSED_MAX_TOKENS];
  __shared__ float shared_pending_key
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1]
      [DIRECT_TRANSITIONS ? K : 1];
  __shared__ float shared_pending_delta
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1]
      [DIRECT_TRANSITIONS ? V : 1];
  __shared__ float shared_pending_decay
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1];

  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  uint32_t published_key_bank = 0;
  float *published_key = nullptr;
  float *candidate_key = nullptr;
  if constexpr (DIRECT_TRANSITIONS) {
    if (active_slot < pending_capacity) {
      pending_epoch = pending_epochs[active_slot];
      pending_rows = (int)pending_keep_rows[active_slot];
      apply_pending =
          pending_epoch != 0 && pending_rows > 0 &&
          pending_rows <= max_pending_rows &&
          recurrent_applied_epochs[(size_t)active_slot * num_v_heads +
                                   value_head] != pending_epoch;
      published_key_bank =
          pending_key_bank[active_slot] & GDN_PENDING_KEY_BANK_MASK;
      const size_t key_bank_stride =
          (size_t)pending_capacity * max_pending_rows * num_k_heads * K;
      published_key = pending_key_banks + published_key_bank * key_bank_stride;
      candidate_key =
          pending_key_banks + (published_key_bank ^ 1u) * key_bank_stride;
    }
  }

  gdn_spec_copy_state_chunk(&shared_state[0][0], source, 0, tid);

  if constexpr (DIRECT_TRANSITIONS) {
    if (apply_pending) {
      for (int linear = tid; linear < pending_rows * K;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / K;
        const int key = linear - position * K;
        shared_pending_key[position][key] =
            published_key[(((size_t)active_slot * max_pending_rows + position) *
                               num_k_heads +
                           key_head) *
                              K +
                          key];
      }
      for (int linear = tid; linear < pending_rows * V;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / V;
        const int value = linear - position * V;
        shared_pending_delta[position][value] =
            pending_delta[(((size_t)active_slot * max_pending_rows + position) *
                               num_v_heads +
                           value_head) *
                              V +
                          value];
      }
      for (int position = tid; position < pending_rows;
           position += GDN_SPEC_FUSED_THREADS) {
        shared_pending_decay[position] =
            pending_decay[((size_t)active_slot * max_pending_rows + position) *
                              num_v_heads +
                          value_head];
      }
    }
  }

  if (warp < seq_len) {
    const int position = warp;
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float4 query =
        gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm = key.x * key.x + key.y * key.y + key.z * key.z +
                     key.w * key.w;
    if constexpr (PAIRED_REDUCTIONS) {
      const float2 qk_norm = gdn_spec_warp_sum_pair(query_norm, key_norm);
      query_norm = qk_norm.x;
      key_norm = qk_norm.y;
    } else {
      query_norm = gdn_warp_sum<32>(query_norm);
      key_norm = gdn_warp_sum<32>(key_norm);
    }
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const float query_values[4] = {query.x, query.y, query.z, query.w};
    const float key_values[4] = {key.x, key.y, key.z, key.w};

#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = lane * 4 + i;
      const int value = lane + i * 32;
      shared_query[position][key] = query_values[i] * query_multiplier;
      shared_key[position][key] = key_values[i] * key_multiplier;
      shared_value[position][value] =
          row[2 * key_dim + value_head * V + value];
    }
    if (lane == 0) {
      const size_t b_offset = (size_t)batch_idx * b_stride_b +
                              (size_t)position * b_stride_s +
                              (size_t)value_head * b_stride_h;
      const size_t a_offset = (size_t)batch_idx * a_stride_b +
                              (size_t)position * a_stride_s +
                              (size_t)value_head * a_stride_h;
      shared_beta[position] =
          1.0f / (1.0f + expf(-(float)b[b_offset]));
      const float biased_a = (float)a[a_offset] + dt_bias[value_head];
      shared_decay[position] =
          expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));
    }
  }
  __syncthreads();

  if constexpr (DIRECT_TRANSITIONS) {
    const bool key_leader =
        tiled_v_heads ? value_head < num_k_heads
                      : value_head % values_per_group == 0;
    if (key_leader && warp < seq_len) {
      const int position = warp;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int key = lane * 4 + i;
        candidate_key[(((size_t)active_slot * max_pending_rows + position) *
                           num_k_heads +
                       key_head) *
                          K +
                      key] = shared_key[position][key];
      }
    }
    if (warp < seq_len && lane == 0) {
      const int position = warp;
      pending_decay[((size_t)active_slot * max_pending_rows + position) *
                        num_v_heads +
                    value_head] = shared_decay[position];
    }
  } else if (transition_key != nullptr) {
    const bool key_leader =
        tiled_v_heads ? value_head < num_k_heads
                      : value_head % values_per_group == 0;
    if (key_leader && warp < seq_len) {
      const int position = warp;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int key = lane * 4 + i;
        transition_key[((transition_row_base + position) * num_k_heads +
                        key_head) *
                           K +
                       key] = shared_key[position][key];
      }
    }
    if (warp < seq_len && lane == 0) {
      const int position = warp;
      transition_decay[(transition_row_base + position) * num_v_heads +
                       value_head] = shared_decay[position];
    }
  }

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row = 0; row < VALUES_PER_WARP; row++) {
    value_rows[row] = warp + row * GDN_SPEC_FUSED_WARPS;
  }

#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    gdn_cp_async_wait();
    __syncthreads();

    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row = 0; row < VALUES_PER_WARP; row++) {
      state[row] = gdn_load_state_x4(
          &shared_state[value_rows[row]][key_base]);
    }
    __syncthreads();
    if (chunk + 1 < GDN_SPEC_FUSED_VALUE_CHUNKS) {
      gdn_spec_copy_state_chunk(&shared_state[0][0], source, chunk + 1, tid);
    }

    if constexpr (DIRECT_TRANSITIONS) {
      if (apply_pending) {
        for (int position = 0; position < pending_rows; position++) {
          const float4 key = *reinterpret_cast<const float4 *>(
              &shared_pending_key[position][key_base]);
          const float decay = shared_pending_decay[position];
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                              value_rows[row];
            const float delta = shared_pending_delta[position][value];
            state[row].x = __fmaf_rn(key.x, delta, state[row].x * decay);
            state[row].y = __fmaf_rn(key.y, delta, state[row].y * decay);
            state[row].z = __fmaf_rn(key.z, delta, state[row].z * decay);
            state[row].w = __fmaf_rn(key.w, delta, state[row].w * decay);
          }
        }
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          const int value =
              chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
          gdn_store_state_x4(source + (size_t)value * K + key_base,
                             state[row]);
          if constexpr (sizeof(StateT) < sizeof(float)) {
            state[row].x = (float)(StateT)state[row].x;
            state[row].y = (float)(StateT)state[row].y;
            state[row].z = (float)(StateT)state[row].z;
            state[row].w = (float)(StateT)state[row].w;
          }
        }
      }
    }

    for (int position = 0; position < seq_len; position++) {
      const float4 query = *reinterpret_cast<const float4 *>(
          &shared_query[position][key_base]);
      const float4 key = *reinterpret_cast<const float4 *>(
          &shared_key[position][key_base]);
      float state_dot_key[VALUES_PER_WARP];
#pragma unroll
      for (int row = 0; row < VALUES_PER_WARP; row++) {
        state[row].x *= shared_decay[position];
        state[row].y *= shared_decay[position];
        state[row].z *= shared_decay[position];
        state[row].w *= shared_decay[position];
        float dot = state[row].x * key.x;
        dot = __fmaf_rn(state[row].y, key.y, dot);
        dot = __fmaf_rn(state[row].z, key.z, dot);
        state_dot_key[row] = __fmaf_rn(state[row].w, key.w, dot);
      }
      if constexpr (PAIRED_REDUCTIONS) {
        const float2 state_dot_key01 =
            gdn_spec_warp_sum_pair(state_dot_key[0], state_dot_key[1]);
        const float2 state_dot_key23 =
            gdn_spec_warp_sum_pair(state_dot_key[2], state_dot_key[3]);
        state_dot_key[0] = state_dot_key01.x;
        state_dot_key[1] = state_dot_key01.y;
        state_dot_key[2] = state_dot_key23.x;
        state_dot_key[3] = state_dot_key23.y;

        float state_dot_query[VALUES_PER_WARP];
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          const int value =
              chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
          const float delta =
              ((float)shared_value[position][value] - state_dot_key[row]) *
              shared_beta[position];
          state[row].x = __fmaf_rn(key.x, delta, state[row].x);
          state[row].y = __fmaf_rn(key.y, delta, state[row].y);
          state[row].z = __fmaf_rn(key.z, delta, state[row].z);
          state[row].w = __fmaf_rn(key.w, delta, state[row].w);
          if (lane == 0) {
            if constexpr (DIRECT_TRANSITIONS) {
              pending_delta[(((size_t)active_slot * max_pending_rows +
                              position) *
                                 num_v_heads +
                             value_head) *
                                V +
                            value] = delta;
            } else if (transition_delta != nullptr) {
              transition_delta[((transition_row_base + position) *
                                    num_v_heads +
                                value_head) *
                                   V +
                               value] = delta;
            }
          }
          float dot = state[row].x * query.x;
          dot = __fmaf_rn(state[row].y, query.y, dot);
          dot = __fmaf_rn(state[row].z, query.z, dot);
          state_dot_query[row] = __fmaf_rn(state[row].w, query.w, dot);
        }
        const float2 state_dot_query01 =
            gdn_spec_warp_sum_pair(state_dot_query[0], state_dot_query[1]);
        const float2 state_dot_query23 =
            gdn_spec_warp_sum_pair(state_dot_query[2], state_dot_query[3]);
        state_dot_query[0] = state_dot_query01.x;
        state_dot_query[1] = state_dot_query01.y;
        state_dot_query[2] = state_dot_query23.x;
        state_dot_query[3] = state_dot_query23.y;
        if (lane == 0) {
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value =
                chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
            shared_output[position][value] = (T)state_dot_query[row];
          }
        }
      } else {
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          state_dot_key[row] = gdn_warp_sum<32>(state_dot_key[row]);
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row];
          const float delta =
              ((float)shared_value[position][value] - state_dot_key[row]) *
              shared_beta[position];
          state[row].x = __fmaf_rn(key.x, delta, state[row].x);
          state[row].y = __fmaf_rn(key.y, delta, state[row].y);
          state[row].z = __fmaf_rn(key.z, delta, state[row].z);
          state[row].w = __fmaf_rn(key.w, delta, state[row].w);
          if (lane == 0) {
            if constexpr (DIRECT_TRANSITIONS) {
              pending_delta[(((size_t)active_slot * max_pending_rows +
                              position) *
                                 num_v_heads +
                             value_head) *
                                V +
                            value] = delta;
            } else if (transition_delta != nullptr) {
              transition_delta[((transition_row_base + position) *
                                    num_v_heads +
                                value_head) *
                                   V +
                               value] = delta;
            }
          }
          float dot = state[row].x * query.x;
          dot = __fmaf_rn(state[row].y, query.y, dot);
          dot = __fmaf_rn(state[row].z, query.z, dot);
          const float state_dot_query =
              gdn_warp_sum<32>(__fmaf_rn(state[row].w, query.w, dot));
          if (lane == 0) {
            shared_output[position][value] = (T)state_dot_query;
          }
        }
      }

      if constexpr (!DIRECT_TRANSITIONS) {
        if (transition_delta == nullptr) {
          StateT *destination =
              state_pool +
              (((base_slot + position) * num_v_heads + value_head) *
               state_head_elements);
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                              value_rows[row];
            gdn_store_state_x4(destination + (size_t)value * K + key_base,
                               state[row]);
          }
        }
      }
    }
  }
  __syncthreads();

  if constexpr (DIRECT_TRANSITIONS) {
    if (tid == 0 && apply_pending) {
      recurrent_applied_epochs[(size_t)active_slot * num_v_heads +
                               value_head] = pending_epoch;
    }
  }

  if (warp < seq_len) {
    const int position = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      output_values[i] = (float)shared_output[position][value];
      sum_square =
          __fmaf_rn(output_values[i], output_values[i], sum_square);
    }
    sum_square = gdn_warp_sum<32>(sum_square);
    const float rstd = rsqrtf(sum_square / (float)V + norm_eps);
    T rounded[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      const size_t gate_offset =
          (size_t)batch_idx * gate_stride_b +
          (size_t)position * gate_stride_s +
          (size_t)value_head * gate_stride_h +
          (size_t)value * gate_stride_v;
      const float gate_value = (float)gate[gate_offset];
      const float silu_gate = gdn_silu(gate_value);
      const size_t output_offset =
          (((size_t)batch_idx * seq_len + position) * num_v_heads +
           value_head) *
              V +
          value;
      rounded[i] = (T)(output_values[i] * rstd * (float)norm_weight[value] *
                       silu_gate);
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
      if (quantized_output == nullptr) {
        output[output_offset] = rounded[i];
      }
    }
    if (quantized_output != nullptr) {
      maximum = gdn_warp_max(maximum);
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                           inverse_quant_scale);
      if (lane == 0) {
        const int normalized_row =
            ((batch_idx * seq_len + position) * num_v_heads) + value_head;
        output_scales[gdn_fp8_scale_offset(normalized_row, num_v_heads,
                                           scale_stride_m)] =
            inverse_quant_scale;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        const size_t output_offset =
            (((size_t)batch_idx * seq_len + position) * num_v_heads +
             value_head) *
                V +
            value;
        quantized_output[output_offset] =
            gdn_fp8_quantize((float)rounded[i], quant_scale,
                             inverse_quant_scale, scale_layout);
      }
    }
  }
}

__global__ __launch_bounds__(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_deferred_recurrence_rmsnorm_gate_value_major_128_kernel(
        const __nv_bfloat16 *__restrict__ mixed_qkv,
        const __nv_bfloat16 *__restrict__ b,
        const __nv_bfloat16 *__restrict__ a,
        const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, float *__restrict__ state_pool,
        __nv_bfloat16 *__restrict__ output,
        gdn_fp8_e4m3 *__restrict__ quantized_output,
        float *__restrict__ output_scales, int scale_stride_m,
        int scale_layout,
        const uint32_t *__restrict__ active_slots,
        float *__restrict__ deferred_key,
        float *__restrict__ deferred_delta,
        float *__restrict__ deferred_decay,
        const uint32_t *__restrict__ deferred_cursor,
        const __nv_bfloat16 *__restrict__ gate,
        const __nv_bfloat16 *__restrict__ norm_weight,
        int64_t b_stride_b, int64_t b_stride_h, int64_t a_stride_b,
        int64_t a_stride_h, int64_t gate_stride_b,
        int64_t gate_stride_h, int64_t gate_stride_v, int batch_size,
        int capacity, int num_k_heads, int num_v_heads,
        int tiled_v_heads, float norm_eps) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || active_slot >= capacity) {
    for (int value = tid; value < V; value += GDN_SPEC_FUSED_THREADS) {
      const size_t offset =
          ((size_t)batch_idx * num_v_heads + value_head) * V + value;
      if (quantized_output != nullptr) {
        quantized_output[offset] = gdn_fp8_e4m3(0.0f);
      } else {
        output[offset] = (__nv_bfloat16)0.0f;
      }
    }
    if (tid == 0 && quantized_output != nullptr) {
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(0.0f, scale_layout, quant_scale,
                           inverse_quant_scale);
      (void)quant_scale;
      output_scales[(size_t)value_head * scale_stride_m + batch_idx] =
          inverse_quant_scale;
    }
    return;
  }

  const uint32_t deferred_rows = deferred_cursor[active_slot];
  if (deferred_rows >= GDN_DEFERRED_STATE_DEPTH) {
    return;
  }
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const bool key_leader = tiled_v_heads
                              ? value_head < num_k_heads
                              : value_head % values_per_group == 0;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const __nv_bfloat16 *row = mixed_qkv + (size_t)batch_idx * conv_dim;
  float *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) * V * K;

  __shared__ float shared_query[K];
  __shared__ float shared_key[K];
  __shared__ float shared_pending_key[GDN_DEFERRED_STATE_DEPTH][K];
  __shared__ float shared_pending_decay[GDN_DEFERRED_STATE_DEPTH];
  __shared__ __nv_bfloat16 shared_output[V];
  __shared__ float shared_beta;
  __shared__ float shared_decay;

  for (int linear = tid; linear < (int)deferred_rows * K;
       linear += GDN_SPEC_FUSED_THREADS) {
    const int position = linear / K;
    const int key = linear - position * K;
    shared_pending_key[position][key] =
        deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                       position) *
                          num_k_heads +
                      key_head) *
                         K +
                     key];
  }
  for (int position = tid; position < (int)deferred_rows;
       position += GDN_SPEC_FUSED_THREADS) {
    shared_pending_decay[position] =
        deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                        position) *
                           num_v_heads +
                       value_head];
  }

  if (warp == 0) {
    float4 query = gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm =
        key.x * key.x + key.y * key.y + key.z * key.z + key.w * key.w;
    query_norm = gdn_warp_sum<32>(query_norm);
    key_norm = gdn_warp_sum<32>(key_norm);
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const float query_values[4] = {query.x, query.y, query.z, query.w};
    const float key_values[4] = {key.x, key.y, key.z, key.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key_idx = lane * 4 + i;
      shared_query[key_idx] = query_values[i] * query_multiplier;
      shared_key[key_idx] = key_values[i] * key_multiplier;
    }
    if (lane == 0) {
      const float b_value =
          (float)b[(size_t)batch_idx * b_stride_b +
                   value_head * b_stride_h];
      const float a_value =
          (float)a[(size_t)batch_idx * a_stride_b +
                   value_head * a_stride_h] +
          dt_bias[value_head];
      shared_beta = 1.0f / (1.0f + expf(-b_value));
      shared_decay =
          expf(-expf(a_log[value_head]) * gdn_spec_softplus(a_value));
    }
  }
  __syncthreads();

  if (key_leader) {
    for (int key = tid; key < K; key += GDN_SPEC_FUSED_THREADS) {
      deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                     deferred_rows) *
                        num_k_heads +
                    key_head) *
                       K +
                   key] = shared_key[key];
    }
  }
  if (tid == 0) {
    deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                    deferred_rows) *
                       num_v_heads +
                   value_head] = shared_decay;
  }

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
    value_rows[row_idx] = warp + row_idx * GDN_SPEC_FUSED_WARPS;
  }

#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      state[row_idx] =
          gdn_load_state_x4(source + (size_t)value * K + key_base);
    }

#pragma unroll
    for (int position = 0; position < GDN_DEFERRED_STATE_DEPTH - 1;
         position++) {
      if (position < deferred_rows) {
        const float4 key = *reinterpret_cast<const float4 *>(
            &shared_pending_key[position][key_base]);
        const float decay = shared_pending_decay[position];
#pragma unroll
        for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row_idx];
          float delta = lane == 0
                            ? deferred_delta
                                  [(((size_t)active_slot *
                                         GDN_DEFERRED_STATE_DEPTH +
                                     position) *
                                        num_v_heads +
                                    value_head) *
                                       V +
                                   value]
                            : 0.0f;
          delta = __shfl_sync(0xffffffff, delta, 0);
          state[row_idx].x =
              __fmaf_rn(key.x, delta, state[row_idx].x * decay);
          state[row_idx].y =
              __fmaf_rn(key.y, delta, state[row_idx].y * decay);
          state[row_idx].z =
              __fmaf_rn(key.z, delta, state[row_idx].z * decay);
          state[row_idx].w =
              __fmaf_rn(key.w, delta, state[row_idx].w * decay);
        }
      }
    }

    const float4 query =
        *reinterpret_cast<const float4 *>(&shared_query[key_base]);
    const float4 key =
        *reinterpret_cast<const float4 *>(&shared_key[key_base]);
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      state[row_idx].x *= shared_decay;
      state[row_idx].y *= shared_decay;
      state[row_idx].z *= shared_decay;
      state[row_idx].w *= shared_decay;
      float state_dot_key = state[row_idx].x * key.x;
      state_dot_key =
          __fmaf_rn(state[row_idx].y, key.y, state_dot_key);
      state_dot_key =
          __fmaf_rn(state[row_idx].z, key.z, state_dot_key);
      state_dot_key = gdn_warp_sum<32>(
          __fmaf_rn(state[row_idx].w, key.w, state_dot_key));
      float value_input =
          lane == 0
              ? (float)row[2 * key_dim + value_head * V + value]
              : 0.0f;
      value_input = __shfl_sync(0xffffffff, value_input, 0);
      const float delta = (value_input - state_dot_key) * shared_beta;
      state[row_idx].x = __fmaf_rn(key.x, delta, state[row_idx].x);
      state[row_idx].y = __fmaf_rn(key.y, delta, state[row_idx].y);
      state[row_idx].z = __fmaf_rn(key.z, delta, state[row_idx].z);
      state[row_idx].w = __fmaf_rn(key.w, delta, state[row_idx].w);
      if (lane == 0) {
        deferred_delta[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                         deferred_rows) *
                            num_v_heads +
                        value_head) *
                           V +
                       value] = delta;
      }
      float state_dot_query = state[row_idx].x * query.x;
      state_dot_query =
          __fmaf_rn(state[row_idx].y, query.y, state_dot_query);
      state_dot_query =
          __fmaf_rn(state[row_idx].z, query.z, state_dot_query);
      state_dot_query = gdn_warp_sum<32>(
          __fmaf_rn(state[row_idx].w, query.w, state_dot_query));
      if (lane == 0) {
        shared_output[value] = (__nv_bfloat16)state_dot_query;
      }
    }

    if (deferred_rows == GDN_DEFERRED_STATE_DEPTH - 1) {
#pragma unroll
      for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
        const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                          value_rows[row_idx];
        gdn_store_state_x4(source + (size_t)value * K + key_base,
                           state[row_idx]);
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      output_values[i] = (float)shared_output[value];
      sum_square =
          __fmaf_rn(output_values[i], output_values[i], sum_square);
    }
    sum_square = gdn_warp_sum<32>(sum_square);
    const float rstd = rsqrtf(sum_square / (float)V + norm_eps);
    __nv_bfloat16 rounded[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      const size_t gate_offset =
          (size_t)batch_idx * gate_stride_b +
          (size_t)value_head * gate_stride_h +
          (size_t)value * gate_stride_v;
      rounded[i] = __float2bfloat16_rn(
          output_values[i] * rstd * (float)norm_weight[value] *
          gdn_silu((float)gate[gate_offset]));
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
    }
    if (quantized_output != nullptr) {
      maximum = gdn_warp_max(maximum);
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                           inverse_quant_scale);
      if (lane == 0) {
        output_scales[(size_t)value_head * scale_stride_m + batch_idx] =
            inverse_quant_scale;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        quantized_output[((size_t)batch_idx * num_v_heads + value_head) * V +
                         value] =
            gdn_fp8_quantize((float)rounded[i], quant_scale,
                             inverse_quant_scale, scale_layout);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        output[((size_t)batch_idx * num_v_heads + value_head) * V + value] =
            rounded[i];
      }
    }
  }
}

__global__ void gdn_deferred_cursor_advance_kernel(
    uint32_t *__restrict__ deferred_cursor,
    const uint32_t *__restrict__ active_slots, int batch_size,
    int capacity) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || active_slot >= capacity) {
    return;
  }
  const uint32_t cursor = deferred_cursor[active_slot];
  deferred_cursor[active_slot] =
      cursor == GDN_DEFERRED_STATE_DEPTH - 1 ? 0 : cursor + 1;
}

__global__ __launch_bounds__(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_flush_deferred_state_value_major_128_kernel(
        float *__restrict__ state_pool,
        const uint32_t *__restrict__ active_slots,
        const float *__restrict__ deferred_key,
        const float *__restrict__ deferred_delta,
        const float *__restrict__ deferred_decay,
        const uint32_t *__restrict__ deferred_cursor, int batch_size,
        int capacity, int num_k_heads, int num_v_heads,
        int tiled_v_heads) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || active_slot >= capacity) {
    return;
  }
  const uint32_t deferred_rows = deferred_cursor[active_slot];
  if (deferred_rows == 0 || deferred_rows > GDN_DEFERRED_STATE_DEPTH) {
    return;
  }
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  float *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) * V * K;
  __shared__ __align__(16) float shared_state[GDN_SPEC_FUSED_VALUE_CHUNK][K];
  __shared__ float shared_pending_key[GDN_DEFERRED_STATE_DEPTH][K];
  __shared__ float shared_pending_decay[GDN_DEFERRED_STATE_DEPTH];

  gdn_spec_copy_state_chunk(&shared_state[0][0], source, 0, tid);
  for (int linear = tid; linear < (int)deferred_rows * K;
       linear += GDN_SPEC_FUSED_THREADS) {
    const int position = linear / K;
    const int key = linear - position * K;
    shared_pending_key[position][key] =
        deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                       position) *
                          num_k_heads +
                      key_head) *
                         K +
                     key];
  }
  for (int position = tid; position < (int)deferred_rows;
       position += GDN_SPEC_FUSED_THREADS) {
    shared_pending_decay[position] =
        deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                        position) *
                           num_v_heads +
                       value_head];
  }
  __syncthreads();

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
    value_rows[row_idx] = warp + row_idx * GDN_SPEC_FUSED_WARPS;
  }
#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    gdn_cp_async_wait();
    __syncthreads();
    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      state[row_idx] =
          gdn_load_state_x4(&shared_state[value_rows[row_idx]][key_base]);
    }
    __syncthreads();
    if (chunk + 1 < GDN_SPEC_FUSED_VALUE_CHUNKS) {
      gdn_spec_copy_state_chunk(&shared_state[0][0], source, chunk + 1, tid);
    }
#pragma unroll
    for (int position = 0; position < GDN_DEFERRED_STATE_DEPTH; position++) {
      if (position < deferred_rows) {
        const float4 key = *reinterpret_cast<const float4 *>(
            &shared_pending_key[position][key_base]);
        const float decay = shared_pending_decay[position];
#pragma unroll
        for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row_idx];
          float delta = lane == 0
                            ? deferred_delta
                                  [(((size_t)active_slot *
                                         GDN_DEFERRED_STATE_DEPTH +
                                     position) *
                                        num_v_heads +
                                    value_head) *
                                       V +
                                   value]
                            : 0.0f;
          delta = __shfl_sync(0xffffffff, delta, 0);
          state[row_idx].x =
              __fmaf_rn(key.x, delta, state[row_idx].x * decay);
          state[row_idx].y =
              __fmaf_rn(key.y, delta, state[row_idx].y * decay);
          state[row_idx].z =
              __fmaf_rn(key.z, delta, state[row_idx].z * decay);
          state[row_idx].w =
              __fmaf_rn(key.w, delta, state[row_idx].w * decay);
        }
      }
    }
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      gdn_store_state_x4(source + (size_t)value * K + key_base,
                         state[row_idx]);
    }
  }
}

__global__ void gdn_deferred_cursor_clear_kernel(
    uint32_t *__restrict__ deferred_cursor,
    const uint32_t *__restrict__ active_slots, int batch_size,
    int capacity) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot != GDN_SPEC_CHECKPOINT_PAD_SLOT && active_slot < capacity) {
    deferred_cursor[active_slot] = 0;
  }
}

extern "C" void gdn_deferred_recurrence_rmsnorm_gate_value_major_128(
    const void *mixed_qkv, const void *b, const void *a,
    const float *a_log, const float *dt_bias, float *state_pool, void *output,
    void *quantized_output, float *output_scales, int scale_stride_m,
    int scale_layout,
    const uint32_t *active_slots, float *deferred_key,
    float *deferred_delta, float *deferred_decay, uint32_t *deferred_cursor,
    const void *gate, const void *norm_weight, int64_t b_stride_b,
    int64_t b_stride_h, int64_t a_stride_b, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_h, int64_t gate_stride_v,
    int batch_size, int capacity, int num_k_heads, int num_v_heads,
    int tiled_v_heads, float norm_eps, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 recurrence_grid(batch_size, num_v_heads);
  gdn_deferred_recurrence_rmsnorm_gate_value_major_128_kernel
      <<<recurrence_grid, GDN_SPEC_FUSED_THREADS, 0, custream>>>(
          (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
          (const __nv_bfloat16 *)a, a_log, dt_bias, state_pool,
          (__nv_bfloat16 *)output, (gdn_fp8_e4m3 *)quantized_output,
          output_scales, scale_stride_m, scale_layout, active_slots,
          deferred_key, deferred_delta, deferred_decay, deferred_cursor,
          (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)norm_weight, b_stride_b, b_stride_h,
          a_stride_b, a_stride_h, gate_stride_b, gate_stride_h,
          gate_stride_v, batch_size, capacity, num_k_heads, num_v_heads,
          tiled_v_heads, norm_eps);
  constexpr int FINALIZE_THREADS = 256;
  gdn_deferred_cursor_advance_kernel
      <<<(batch_size + FINALIZE_THREADS - 1) / FINALIZE_THREADS,
         FINALIZE_THREADS, 0, custream>>>(deferred_cursor, active_slots,
                                          batch_size, capacity);
}

extern "C" void gdn_flush_deferred_state_value_major_128(
    float *state_pool, const uint32_t *active_slots,
    const float *deferred_key, const float *deferred_delta,
    const float *deferred_decay, uint32_t *deferred_cursor, int batch_size,
    int capacity, int num_k_heads, int num_v_heads, int tiled_v_heads,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 flush_grid(batch_size, num_v_heads);
  gdn_flush_deferred_state_value_major_128_kernel
      <<<flush_grid, GDN_SPEC_FUSED_THREADS, 0, custream>>>(
          state_pool, active_slots, deferred_key, deferred_delta,
          deferred_decay, deferred_cursor, batch_size, capacity, num_k_heads,
          num_v_heads, tiled_v_heads);
  constexpr int FINALIZE_THREADS = 256;
  gdn_deferred_cursor_clear_kernel
      <<<(batch_size + FINALIZE_THREADS - 1) / FINALIZE_THREADS,
         FINALIZE_THREADS, 0, custream>>>(deferred_cursor, active_slots,
                                          batch_size, capacity);
}

template <typename T, typename StateT>
__global__ __launch_bounds__(32 * GDN_SPEC_CHECKPOINT_WARPS)
    void gdn_speculative_recurrence_checkpoints_value_major_128_kernel(
        const T *__restrict__ mixed_qkv, const T *__restrict__ b,
        const T *__restrict__ a, const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
        T *__restrict__ output, const uint32_t *__restrict__ active_slots,
        int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
        int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
        int batch_size, int seq_len, int num_k_heads, int num_v_heads,
        int checkpoint_lanes, int tiled_v_heads) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_CHECKPOINT_VALUES_PER_WARP;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_base = blockIdx.x * GDN_SPEC_CHECKPOINT_VALUE_TILE +
                         warp * VALUES_PER_WARP;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    if (lane < VALUES_PER_WARP) {
      const int value_idx = value_base + lane;
      for (int position = 0; position < seq_len; position++) {
        output[((size_t)batch_head * seq_len + position) * V + value_idx] =
            (T)0.0f;
      }
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)V * K;
  const StateT *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) *
                       state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  float4 state[VALUES_PER_WARP];
#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
    state[value] = gdn_load_state_x4(
        source + (size_t)(value_base + value) * K + lane * 4);
  }

  for (int position = 0; position < seq_len; position++) {
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float4 query =
        gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm =
        key.x * key.x + key.y * key.y + key.z * key.z + key.w * key.w;
    query_norm = gdn_warp_sum<32>(query_norm);
    key_norm = gdn_warp_sum<32>(key_norm);
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    query = make_float4(query.x * query_multiplier,
                        query.y * query_multiplier,
                        query.z * query_multiplier,
                        query.w * query_multiplier);
    key = make_float4(key.x * key_multiplier, key.y * key_multiplier,
                      key.z * key_multiplier, key.w * key_multiplier);

    float beta = 0.0f;
    float decay = 0.0f;
    if (lane == 0) {
      const size_t b_offset = (size_t)batch_idx * b_stride_b +
                              (size_t)position * b_stride_s +
                              (size_t)value_head * b_stride_h;
      const size_t a_offset = (size_t)batch_idx * a_stride_b +
                              (size_t)position * a_stride_s +
                              (size_t)value_head * a_stride_h;
      beta = 1.0f / (1.0f + expf(-(float)b[b_offset]));
      const float biased_a = (float)a[a_offset] + dt_bias[value_head];
      decay = expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));
    }
    beta = __shfl_sync(0xffffffff, beta, 0);
    decay = __shfl_sync(0xffffffff, decay, 0);

#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      float4 next = make_float4(state[value].x * decay,
                                state[value].y * decay,
                                state[value].z * decay,
                                state[value].w * decay);
      float state_dot_key = next.x * key.x;
      state_dot_key = __fmaf_rn(next.y, key.y, state_dot_key);
      state_dot_key = __fmaf_rn(next.z, key.z, state_dot_key);
      state_dot_key = __fmaf_rn(next.w, key.w, state_dot_key);
      state_dot_key = gdn_warp_sum<32>(state_dot_key);
      float value_input =
          lane == 0
              ? (float)row[2 * key_dim + value_head * V + value_base + value]
              : 0.0f;
      value_input = __shfl_sync(0xffffffff, value_input, 0);
      const float delta = (value_input - state_dot_key) * beta;
      next.x = __fmaf_rn(key.x, delta, next.x);
      next.y = __fmaf_rn(key.y, delta, next.y);
      next.z = __fmaf_rn(key.z, delta, next.z);
      next.w = __fmaf_rn(key.w, delta, next.w);
      state[value] = next;

      float state_dot_query = next.x * query.x;
      state_dot_query = __fmaf_rn(next.y, query.y, state_dot_query);
      state_dot_query = __fmaf_rn(next.z, query.z, state_dot_query);
      state_dot_query = __fmaf_rn(next.w, query.w, state_dot_query);
      state_dot_query = gdn_warp_sum<32>(state_dot_query);
      if (lane == 0) {
        output[((size_t)batch_head * seq_len + position) * V + value_base +
               value] = (T)state_dot_query;
      }
    }

    StateT *destination =
        state_pool +
        (((base_slot + position) * num_v_heads + value_head) *
         state_head_elements);
#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      gdn_store_state_x4(
          destination + (size_t)(value_base + value) * K + lane * 4,
          state[value]);
    }
  }
}

template <typename T, typename StateT, bool VALUE_MAJOR>
__global__ void gdn_speculative_recurrence_checkpoints_fallback_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
    T *__restrict__ output, const uint32_t *__restrict__ active_slots,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int batch_size, int seq_len, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads) {
  const int value_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size || value_idx >= head_v_dim) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int position = 0; position < seq_len; position++) {
      output[((size_t)batch_head * seq_len + position) * head_v_dim +
             value_idx] = (T)0.0f;
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)head_k_dim * head_v_dim;
  const size_t source_head =
      ((size_t)active_slot * num_v_heads + value_head) * state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);
  float state[GDN_SPEC_CHECKPOINT_MAX_K];
  for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
    const size_t offset = VALUE_MAJOR
                              ? (size_t)value_idx * head_k_dim + key_idx
                              : (size_t)key_idx * head_v_dim + value_idx;
    state[key_idx] = state_pool[source_head + offset];
  }

  for (int position = 0; position < seq_len; position++) {
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float query_norm = 0.0f;
    float key_norm = 0.0f;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      const float query = (float)row[key_head * head_k_dim + key_idx];
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx];
      query_norm = __fmaf_rn(query, query, query_norm);
      key_norm = __fmaf_rn(key, key, key_norm);
    }
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)head_k_dim);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const size_t b_offset = (size_t)batch_idx * b_stride_b +
                            (size_t)position * b_stride_s +
                            (size_t)value_head * b_stride_h;
    const size_t a_offset = (size_t)batch_idx * a_stride_b +
                            (size_t)position * a_stride_s +
                            (size_t)value_head * a_stride_h;
    const float beta = 1.0f / (1.0f + expf(-(float)b[b_offset]));
    const float biased_a = (float)a[a_offset] + dt_bias[value_head];
    const float decay =
        expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));

    float state_dot_key = 0.0f;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      state[key_idx] *= decay;
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx] *
          key_multiplier;
      state_dot_key = __fmaf_rn(state[key_idx], key, state_dot_key);
    }
    const float value_input =
        (float)row[2 * key_dim + value_head * head_v_dim + value_idx];
    const float delta = (value_input - state_dot_key) * beta;
    float state_dot_query = 0.0f;
    const size_t destination_head =
        ((base_slot + position) * num_v_heads + value_head) *
        state_head_elements;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx] *
          key_multiplier;
      state[key_idx] = __fmaf_rn(key, delta, state[key_idx]);
      const float query =
          (float)row[key_head * head_k_dim + key_idx] * query_multiplier;
      state_dot_query = __fmaf_rn(state[key_idx], query, state_dot_query);
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      state_pool[destination_head + offset] = state[key_idx];
    }
    output[((size_t)batch_head * seq_len + position) * head_v_dim +
           value_idx] = (T)state_dot_query;
  }
}

template <typename T, typename StateT>
void launch_gdn_speculative_recurrence_checkpoints(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, StateT *state_pool, T *output,
    gdn_fp8_e4m3 *quantized_output, float *output_scales,
    int scale_stride_m, int scale_layout,
    const uint32_t *active_slots, const T *gate, const T *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps,
    cudaStream_t stream) {
  const bool batch_transitions = transition_delta != nullptr;
  const bool direct_transitions = slot_indexed_transitions != 0;
  if ((transition_key != nullptr) != batch_transitions ||
      (transition_decay != nullptr) != batch_transitions) {
    return;
  }
  if ((quantized_output != nullptr) != (output_scales != nullptr)) {
    return;
  }
  const bool has_pending =
      pending_key_banks != nullptr || pending_key_bank != nullptr ||
      pending_delta != nullptr || pending_decay != nullptr ||
      pending_keep_rows != nullptr || pending_epochs != nullptr ||
      recurrent_applied_epochs != nullptr;
  if (direct_transitions &&
      (batch_transitions || !has_pending || pending_key_banks == nullptr ||
       pending_key_bank == nullptr || pending_delta == nullptr ||
       pending_decay == nullptr || pending_keep_rows == nullptr ||
       pending_epochs == nullptr || recurrent_applied_epochs == nullptr ||
       max_pending_rows <= 0 ||
       max_pending_rows > GDN_SPEC_FUSED_MAX_TOKENS ||
       pending_capacity <= 0)) {
    return;
  }
  if (!direct_transitions && has_pending) {
    return;
  }
  if (gate != nullptr && norm_weight != nullptr) {
    dim3 fused_grid(batch_size, num_v_heads);
    const int grid_blocks = batch_size * num_v_heads;
    const bool paired_reductions =
        sizeof(StateT) < sizeof(float) ||
        grid_blocks <= GDN_SPEC_FUSED_PAIR_LOW_GRID_MAX ||
        grid_blocks >= GDN_SPEC_FUSED_PAIR_HIGH_GRID_MIN;
#define GDN_LAUNCH_SPEC_RECURRENCE(PAIRED, DIRECT)                           \
  gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel<           \
      T, StateT, PAIRED, DIRECT><<<fused_grid, GDN_SPEC_FUSED_THREADS, 0,    \
                                   stream>>>(                               \
      mixed_qkv, b, a, a_log, dt_bias, state_pool, output,                 \
      quantized_output, output_scales, scale_stride_m, scale_layout,        \
      active_slots,                                                        \
      gate, norm_weight, transition_key, transition_delta, transition_decay, \
      pending_key_banks, pending_key_bank, pending_delta, pending_decay,     \
      pending_keep_rows, pending_epochs, recurrent_applied_epochs,           \
      max_pending_rows, pending_capacity, b_stride_b, b_stride_s,            \
      b_stride_h, a_stride_b, a_stride_s, a_stride_h, gate_stride_b,         \
      gate_stride_s, gate_stride_h, gate_stride_v, batch_size, seq_len,      \
      num_k_heads, num_v_heads, checkpoint_lanes, tiled_v_heads, norm_eps)
    if (direct_transitions) {
      if (paired_reductions) {
        GDN_LAUNCH_SPEC_RECURRENCE(true, true);
      } else {
        GDN_LAUNCH_SPEC_RECURRENCE(false, true);
      }
    } else if (paired_reductions) {
      GDN_LAUNCH_SPEC_RECURRENCE(true, false);
    } else {
      GDN_LAUNCH_SPEC_RECURRENCE(false, false);
    }
#undef GDN_LAUNCH_SPEC_RECURRENCE
    return;
  }

  dim3 grid((head_v_dim + GDN_SPEC_CHECKPOINT_VALUE_TILE - 1) /
                GDN_SPEC_CHECKPOINT_VALUE_TILE,
            batch_size * num_v_heads);
  if (value_major && head_k_dim == GDN_DECODE_VALUE_MAJOR_K &&
      head_v_dim == GDN_DECODE_VALUE_MAJOR_V) {
    dim3 block(32, GDN_SPEC_CHECKPOINT_WARPS);
    gdn_speculative_recurrence_checkpoints_value_major_128_kernel<T, StateT>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, checkpoint_lanes,
            tiled_v_heads);
    return;
  }

  dim3 fallback_block(GDN_SPEC_CHECKPOINT_VALUE_TILE);
  if (value_major) {
    gdn_speculative_recurrence_checkpoints_fallback_kernel<T, StateT, true>
        <<<grid, fallback_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim,
            head_v_dim, checkpoint_lanes, tiled_v_heads);
  } else {
    gdn_speculative_recurrence_checkpoints_fallback_kernel<T, StateT, false>
        <<<grid, fallback_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state_pool, output, active_slots,
            b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
            a_stride_h,
            batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim,
            head_v_dim, checkpoint_lanes, tiled_v_heads);
  }
}

template <typename T>
void dispatch_gdn_speculative_recurrence_checkpoints(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, void *state_pool, T *output,
    gdn_fp8_e4m3 *quantized_output, float *output_scales,
    int scale_stride_m, int scale_layout,
    const uint32_t *active_slots, const T *gate, const T *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps, int state_dtype,
    cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (__half *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (__nv_bfloat16 *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, stream);
  } else {
    launch_gdn_speculative_recurrence_checkpoints(
        mixed_qkv, b, a, a_log, dt_bias, (float *)state_pool, output,
        quantized_output, output_scales, scale_stride_m, scale_layout,
        active_slots, gate, norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, stream);
  }
}

extern "C" void gdn_speculative_recurrence_checkpoints(
    const void *mixed_qkv, const void *b, const void *a,
    const float *a_log, const float *dt_bias, void *state_pool, void *output,
    void *quantized_output, float *output_scales, int scale_stride_m,
    int scale_layout,
    const uint32_t *active_slots, const void *gate, const void *norm_weight,
    float *transition_key, float *transition_delta, float *transition_decay,
    float *pending_key_banks, const uint32_t *pending_key_bank,
    float *pending_delta, float *pending_decay,
    const uint32_t *pending_keep_rows, const uint32_t *pending_epochs,
    uint32_t *recurrent_applied_epochs,
    int max_pending_rows, int pending_capacity,
    int slot_indexed_transitions,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int64_t gate_stride_b, int64_t gate_stride_s, int64_t gate_stride_h,
    int64_t gate_stride_v, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int value_major, float norm_eps, int dtype,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_speculative_recurrence_checkpoints(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
        a_log, dt_bias, state_pool, (__half *)output,
        (gdn_fp8_e4m3 *)quantized_output, output_scales, scale_stride_m,
        scale_layout, active_slots,
        (const __half *)gate, (const __half *)norm_weight, transition_key,
        transition_delta, transition_decay, pending_key_banks,
        pending_key_bank, pending_delta, pending_decay, pending_keep_rows,
        pending_epochs, recurrent_applied_epochs, max_pending_rows,
        pending_capacity, slot_indexed_transitions, b_stride_b, b_stride_s,
        b_stride_h, a_stride_b, a_stride_s, a_stride_h, gate_stride_b,
        gate_stride_s, gate_stride_h, gate_stride_v, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, checkpoint_lanes,
        tiled_v_heads, value_major, norm_eps, state_dtype, custream);
  } else {
    dispatch_gdn_speculative_recurrence_checkpoints(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, state_pool,
        (__nv_bfloat16 *)output, (gdn_fp8_e4m3 *)quantized_output,
        output_scales, scale_stride_m, scale_layout, active_slots,
        (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)norm_weight, transition_key, transition_delta,
        transition_decay, pending_key_banks, pending_key_bank, pending_delta,
        pending_decay, pending_keep_rows, pending_epochs,
        recurrent_applied_epochs, max_pending_rows, pending_capacity,
        slot_indexed_transitions,
        b_stride_b, b_stride_s, b_stride_h, a_stride_b, a_stride_s,
        a_stride_h, gate_stride_b, gate_stride_s, gate_stride_h,
        gate_stride_v, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, checkpoint_lanes, tiled_v_heads, value_major,
        norm_eps, state_dtype, custream);
  }
}

constexpr int GDN_TRANSITION_CONV_INPUT = 0;
constexpr int GDN_TRANSITION_KEY = 1;
constexpr int GDN_TRANSITION_DELTA = 2;
constexpr int GDN_TRANSITION_DECAY = 3;
constexpr int GDN_TRANSITION_CONV_STATE = 4;
constexpr int GDN_TRANSITION_RECURRENT_STATE = 5;

template <typename T, typename StateT>
__global__ void gdn_speculative_transition_commit_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ active_slots, int layer_count, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int conv_dim, int conv_width, int tiled_v_heads,
    int value_major) {
  const int conv_blocks = (conv_dim + blockDim.x - 1) / blockDim.x;
  const int state_elements = head_k_dim * head_v_dim;
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  const int batch_block = blockIdx.x;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  const uint32_t slot = active_slots[batch_idx];
  if (rows == 0 || slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    return;
  }

  if (batch_block < conv_blocks) {
    const int channel = batch_block * blockDim.x + threadIdx.x;
    if (channel >= conv_dim) {
      return;
    }
    const auto *input = reinterpret_cast<const T *>(
        pointer_table[GDN_TRANSITION_CONV_INPUT * layer_count + layer]);
    auto *conv_state = reinterpret_cast<T *>(
        pointer_table[GDN_TRANSITION_CONV_STATE * layer_count + layer]);
    T state[GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH];
    T *destination =
        conv_state + ((size_t)slot * conv_dim + channel) * conv_width;
    for (int i = 0; i < conv_width; i++) {
      state[i] = destination[i];
    }
    for (int position = 0; position < rows; position++) {
      for (int i = 0; i < conv_width - 1; i++) {
        state[i] = state[i + 1];
      }
      state[conv_width - 1] =
          input[((size_t)batch_idx * seq_len + position) * conv_dim +
                channel];
    }
    for (int i = 0; i < conv_width; i++) {
      destination[i] = state[i];
    }
    return;
  }

  const int recurrence_block = batch_block - conv_blocks;
  const int value_head = recurrence_block;
  if (value_head >= num_v_heads) {
    return;
  }
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads ? value_head % num_k_heads
                                     : value_head / values_per_group;
  const auto *transition_key = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_KEY * layer_count + layer]);
  const auto *transition_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_DELTA * layer_count + layer]);
  const auto *transition_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_DECAY * layer_count + layer]);
  auto *recurrent_state = reinterpret_cast<StateT *>(
      pointer_table[GDN_TRANSITION_RECURRENT_STATE * layer_count + layer]);
  const size_t state_head =
      ((size_t)slot * num_v_heads + value_head) * state_elements;
  __shared__ float shared_key[GDN_SPEC_FUSED_MAX_TOKENS]
                             [GDN_DECODE_VALUE_MAJOR_K];
  __shared__ float shared_delta[GDN_SPEC_FUSED_MAX_TOKENS]
                               [GDN_DECODE_VALUE_MAJOR_V];
  __shared__ float shared_decay[GDN_SPEC_FUSED_MAX_TOKENS];
  for (int linear = threadIdx.x; linear < rows * head_k_dim;
       linear += blockDim.x) {
    const int position = linear / head_k_dim;
    const int key = linear - position * head_k_dim;
    shared_key[position][key] =
        transition_key[(((size_t)batch_idx * seq_len + position) *
                            num_k_heads +
                        key_head) *
                           head_k_dim +
                       key];
  }
  for (int linear = threadIdx.x; linear < rows * head_v_dim;
       linear += blockDim.x) {
    const int position = linear / head_v_dim;
    const int value = linear - position * head_v_dim;
    shared_delta[position][value] =
        transition_delta[(((size_t)batch_idx * seq_len + position) *
                              num_v_heads +
                          value_head) *
                             head_v_dim +
                         value];
  }
  for (int position = threadIdx.x; position < rows;
       position += blockDim.x) {
    shared_decay[position] =
        transition_decay[((size_t)batch_idx * seq_len + position) *
                             num_v_heads +
                         value_head];
  }
  __syncthreads();
  for (int element = threadIdx.x; element < state_elements;
       element += blockDim.x) {
    const int value = element / head_k_dim;
    const int key = element - value * head_k_dim;
    const size_t state_offset = value_major
                                    ? (size_t)value * head_k_dim + key
                                    : (size_t)key * head_v_dim + value;
    float state = (float)recurrent_state[state_head + state_offset];
    for (int position = 0; position < rows; position++) {
      state = __fmaf_rn(shared_key[position][key],
                        shared_delta[position][value],
                        __fmul_rn(shared_decay[position], state));
    }
    recurrent_state[state_head + state_offset] = (StateT)state;
  }
}

template <typename T>
void dispatch_gdn_speculative_transition_commit_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *active_slots, int layer_count, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int value_major,
    int state_dtype, cudaStream_t stream) {
  const int conv_blocks =
      (conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE;
  dim3 grid(conv_blocks + num_v_heads, batch_size, layer_count);
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    gdn_speculative_transition_commit_batched_kernel<T, __half>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    gdn_speculative_transition_commit_batched_kernel<T, __nv_bfloat16>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major);
  } else {
    gdn_speculative_transition_commit_batched_kernel<T, float>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, keep_rows, active_slots, layer_count, batch_size,
            seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, value_major);
  }
}

extern "C" void gdn_speculative_transition_commit_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *active_slots, int layer_count, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int value_major,
    int activation_dtype, int state_dtype, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || seq_len <= 0 ||
      seq_len > GDN_SPEC_FUSED_MAX_TOKENS ||
      head_k_dim != GDN_DECODE_VALUE_MAJOR_K ||
      head_v_dim != GDN_DECODE_VALUE_MAJOR_V) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_speculative_transition_commit_batched<__half>(
        pointer_table, keep_rows, active_slots, layer_count, batch_size,
        seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, value_major, state_dtype, custream);
  } else {
    dispatch_gdn_speculative_transition_commit_batched<__nv_bfloat16>(
        pointer_table, keep_rows, active_slots, layer_count, batch_size,
        seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, value_major, state_dtype, custream);
  }
}

constexpr int GDN_TRANSITION_STAGE_SRC_CONV = 0;
constexpr int GDN_TRANSITION_STAGE_SRC_KEY = 1;
constexpr int GDN_TRANSITION_STAGE_SRC_DELTA = 2;
constexpr int GDN_TRANSITION_STAGE_SRC_DECAY = 3;
constexpr int GDN_TRANSITION_STAGE_DST_CONV = 4;
constexpr int GDN_TRANSITION_STAGE_DST_KEY = 5;
constexpr int GDN_TRANSITION_STAGE_DST_DELTA = 6;
constexpr int GDN_TRANSITION_STAGE_DST_DECAY = 7;
constexpr int GDN_TRANSITION_STAGE_DST_KEEP = 8;
constexpr int GDN_TRANSITION_STAGE_DST_EPOCH = 9;
constexpr int GDN_TRANSITION_STAGE_COPY_BLOCKS = 4;

template <typename T>
__global__ void gdn_speculative_transition_stage_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ destination_slots, int layer_count,
    int batch_size,
    int seq_len, int max_rows, int destination_capacity, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int conv_dim) {
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const uint32_t slot = destination_slots[batch_idx];
  if (slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || slot >= destination_capacity) {
    return;
  }
  const int rows = (int)keep_rows[batch_idx];
  auto *pending_keep = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_KEEP * layer_count + layer]);
  auto *pending_epoch = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_EPOCH * layer_count + layer]);
  uint32_t next_epoch = pending_epoch[slot] + 1;
  if (next_epoch == 0) {
    next_epoch = 1;
  }
  if (rows <= 0 || rows > seq_len || rows > max_rows) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      pending_keep[slot] = 0;
      pending_epoch[slot] = next_epoch;
    }
    return;
  }

  const size_t conv_elements = (size_t)rows * conv_dim;
  const size_t key_row_elements = (size_t)num_k_heads * head_k_dim;
  const size_t key_elements = (size_t)rows * key_row_elements;
  const size_t delta_row_elements = (size_t)num_v_heads * head_v_dim;
  const size_t delta_elements = (size_t)rows * delta_row_elements;
  const size_t decay_row_elements = num_v_heads;
  const size_t decay_elements = (size_t)rows * decay_row_elements;
  const size_t total_elements =
      conv_elements + key_elements + delta_elements + decay_elements;
  const auto *source_conv = reinterpret_cast<const T *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_CONV * layer_count + layer]);
  const auto *source_key = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_KEY * layer_count + layer]);
  const auto *source_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_DELTA * layer_count + layer]);
  const auto *source_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_DECAY * layer_count + layer]);
  auto *destination_conv = reinterpret_cast<T *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_CONV * layer_count + layer]);
  auto *destination_key = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_KEY * layer_count + layer]);
  auto *destination_delta = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_DELTA * layer_count + layer]);
  auto *destination_decay = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_DECAY * layer_count + layer]);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements; linear += stride) {
    if (linear < conv_elements) {
      destination_conv[((size_t)slot * max_rows * conv_dim) + linear] =
          source_conv[((size_t)batch_idx * seq_len * conv_dim) + linear];
    } else if (linear < conv_elements + key_elements) {
      const size_t offset = linear - conv_elements;
      destination_key[((size_t)slot * max_rows * key_row_elements) + offset] =
          source_key[((size_t)batch_idx * seq_len * key_row_elements) + offset];
    } else if (linear < conv_elements + key_elements + delta_elements) {
      const size_t offset = linear - conv_elements - key_elements;
      destination_delta[((size_t)slot * max_rows * delta_row_elements) +
                        offset] =
          source_delta[((size_t)batch_idx * seq_len * delta_row_elements) +
                       offset];
    } else {
      const size_t offset =
          linear - conv_elements - key_elements - delta_elements;
      destination_decay[((size_t)slot * max_rows * decay_row_elements) +
                        offset] =
          source_decay[((size_t)batch_idx * seq_len * decay_row_elements) +
                       offset];
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    pending_keep[slot] = (uint32_t)rows;
    pending_epoch[slot] = next_epoch;
  }
}

template <typename T>
void dispatch_gdn_speculative_transition_stage_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int seq_len, int max_rows,
    int destination_capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, cudaStream_t stream) {
  const size_t row_elements =
      (size_t)conv_dim + (size_t)num_k_heads * head_k_dim +
      (size_t)num_v_heads * head_v_dim + num_v_heads;
  const size_t required_blocks =
      ((size_t)seq_len * row_elements + GDN_CHANNEL_BLOCK_SIZE - 1) /
      GDN_CHANNEL_BLOCK_SIZE;
  const unsigned int copy_blocks = (unsigned int)(
      required_blocks < GDN_TRANSITION_STAGE_COPY_BLOCKS
          ? required_blocks
          : GDN_TRANSITION_STAGE_COPY_BLOCKS);
  dim3 grid(copy_blocks, batch_size, layer_count);
  gdn_speculative_transition_stage_batched_kernel<T>
      <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
          pointer_table, keep_rows, destination_slots, layer_count, batch_size,
          seq_len, max_rows, destination_capacity, num_k_heads,
          num_v_heads, head_k_dim, head_v_dim, conv_dim);
}

extern "C" void gdn_speculative_transition_stage_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int seq_len, int max_rows,
    int destination_capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, int activation_dtype,
    int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || seq_len <= 0 ||
      seq_len > GDN_SPEC_FUSED_MAX_TOKENS || max_rows < seq_len ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || destination_capacity <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
      head_v_dim <= 0 || conv_dim <= 0) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_speculative_transition_stage_batched<__half>(
        pointer_table, keep_rows, destination_slots, layer_count, batch_size,
        seq_len, max_rows, destination_capacity, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, conv_dim, custream);
  } else {
    dispatch_gdn_speculative_transition_stage_batched<__nv_bfloat16>(
        pointer_table, keep_rows, destination_slots, layer_count, batch_size,
        seq_len, max_rows, destination_capacity, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, conv_dim, custream);
  }
}

constexpr int GDN_TRANSITION_PUBLISH_KEEP = 0;
constexpr int GDN_TRANSITION_PUBLISH_EPOCH = 1;
constexpr int GDN_TRANSITION_PUBLISH_KEY_BANK = 2;

__global__ void gdn_pending_transition_publish_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ destination_slots, int layer_count,
    int batch_size, int max_rows, int destination_capacity) {
  const int batch_idx = blockIdx.x;
  const int layer = blockIdx.y;
  if (batch_idx >= batch_size || layer >= layer_count || threadIdx.x != 0) {
    return;
  }
  const uint32_t slot = destination_slots[batch_idx];
  if (slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || slot >= destination_capacity) {
    return;
  }
  auto *pending_keep = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_KEEP * layer_count + layer]);
  auto *pending_epoch = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_EPOCH * layer_count + layer]);
  auto *pending_key_bank = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_KEY_BANK * layer_count + layer]);
  const uint32_t rows = keep_rows[batch_idx];
  uint32_t next_epoch = pending_epoch[slot] + 1;
  if (next_epoch == 0) {
    next_epoch = 1;
  }
  pending_keep[slot] = rows <= (uint32_t)max_rows ? rows : 0;
  pending_key_bank[slot] =
      (pending_key_bank[slot] ^ GDN_PENDING_KEY_BANK_MASK) &
      GDN_PENDING_KEY_BANK_MASK;
  pending_epoch[slot] = next_epoch;
}

extern "C" void gdn_pending_transition_publish_batched(
    const uint64_t *pointer_table, const uint32_t *keep_rows,
    const uint32_t *destination_slots, int layer_count, int batch_size,
    int max_rows, int destination_capacity, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || max_rows <= 0 ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || destination_capacity <= 0) {
    return;
  }
  dim3 grid(batch_size, layer_count);
  gdn_pending_transition_publish_batched_kernel<<<
      grid, 1, 0, (cudaStream_t)stream>>>(
      pointer_table, keep_rows, destination_slots, layer_count, batch_size,
      max_rows, destination_capacity);
}

constexpr int GDN_TRANSITION_APPLY_PENDING_CONV = 0;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANKS = 1;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANK = 2;
constexpr int GDN_TRANSITION_APPLY_PENDING_DELTA = 3;
constexpr int GDN_TRANSITION_APPLY_PENDING_DECAY = 4;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEEP = 5;
constexpr int GDN_TRANSITION_APPLY_PENDING_EPOCH = 6;
constexpr int GDN_TRANSITION_APPLY_CONV_EPOCH = 7;
constexpr int GDN_TRANSITION_APPLY_RECURRENT_EPOCH = 8;
constexpr int GDN_TRANSITION_APPLY_CONV_STATE = 9;
constexpr int GDN_TRANSITION_APPLY_RECURRENT_STATE = 10;

template <typename T, typename StateT>
__global__ void gdn_pending_transition_apply_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ active_slots, int layer_count, int batch_size,
    int max_rows, int capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, int conv_width,
    int tiled_v_heads, int conv_blocks) {
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  const int batch_block = blockIdx.x;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const uint32_t slot = active_slots[batch_idx];
  if (slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || slot >= capacity) {
    return;
  }
  const auto *pending_keep = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEEP * layer_count + layer]);
  const auto *pending_epoch = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_EPOCH * layer_count + layer]);
  const int rows = (int)pending_keep[slot];
  const uint32_t epoch = pending_epoch[slot];
  if (epoch == 0 || rows <= 0 || rows > max_rows) {
    return;
  }

  if (batch_block < conv_blocks) {
    auto *applied_epochs = reinterpret_cast<uint32_t *>(
        pointer_table[GDN_TRANSITION_APPLY_CONV_EPOCH * layer_count + layer]);
    const size_t applied_offset = (size_t)slot * conv_blocks + batch_block;
    if (applied_epochs[applied_offset] == epoch) {
      return;
    }
    const int channel = batch_block * blockDim.x + threadIdx.x;
    if (channel < conv_dim) {
      const auto *pending_conv = reinterpret_cast<const T *>(
          pointer_table[GDN_TRANSITION_APPLY_PENDING_CONV * layer_count +
                        layer]);
      auto *conv_state = reinterpret_cast<T *>(
          pointer_table[GDN_TRANSITION_APPLY_CONV_STATE * layer_count + layer]);
      T state[GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH];
      T *destination =
          conv_state + ((size_t)slot * conv_dim + channel) * conv_width;
      for (int i = 0; i < conv_width; i++) {
        state[i] = destination[i];
      }
      for (int position = 0; position < rows; position++) {
        for (int i = 0; i < conv_width - 1; i++) {
          state[i] = state[i + 1];
        }
        state[conv_width - 1] =
            pending_conv[((size_t)slot * max_rows + position) * conv_dim +
                         channel];
      }
      for (int i = 0; i < conv_width; i++) {
        destination[i] = state[i];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      applied_epochs[applied_offset] = epoch;
    }
    return;
  }

  const int value_head = batch_block - conv_blocks;
  if (value_head >= num_v_heads) {
    return;
  }
  auto *applied_epochs = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_RECURRENT_EPOCH * layer_count +
                    layer]);
  const size_t applied_offset = (size_t)slot * num_v_heads + value_head;
  if (applied_epochs[applied_offset] == epoch) {
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads ? value_head % num_k_heads
                                     : value_head / values_per_group;
  const auto *pending_key_banks = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEY_BANKS * layer_count +
                    layer]);
  const auto *pending_key_bank = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEY_BANK * layer_count +
                    layer]);
  const size_t key_bank_stride =
      (size_t)capacity * max_rows * num_k_heads * head_k_dim;
  const auto *pending_key =
      pending_key_banks +
      (pending_key_bank[slot] & GDN_PENDING_KEY_BANK_MASK) * key_bank_stride;
  const auto *pending_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_DELTA * layer_count + layer]);
  const auto *pending_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_DECAY * layer_count + layer]);
  auto *recurrent_state = reinterpret_cast<StateT *>(
      pointer_table[GDN_TRANSITION_APPLY_RECURRENT_STATE * layer_count +
                    layer]);
  __shared__ float shared_key[GDN_SPEC_FUSED_MAX_TOKENS]
                             [GDN_DECODE_VALUE_MAJOR_K];
  __shared__ float shared_delta[GDN_SPEC_FUSED_MAX_TOKENS]
                               [GDN_DECODE_VALUE_MAJOR_V];
  __shared__ float shared_decay[GDN_SPEC_FUSED_MAX_TOKENS];
  for (int linear = threadIdx.x; linear < rows * head_k_dim;
       linear += blockDim.x) {
    const int position = linear / head_k_dim;
    const int key = linear - position * head_k_dim;
    shared_key[position][key] =
        pending_key[(((size_t)slot * max_rows + position) * num_k_heads +
                     key_head) *
                        head_k_dim +
                    key];
  }
  for (int linear = threadIdx.x; linear < rows * head_v_dim;
       linear += blockDim.x) {
    const int position = linear / head_v_dim;
    const int value = linear - position * head_v_dim;
    shared_delta[position][value] =
        pending_delta[(((size_t)slot * max_rows + position) * num_v_heads +
                       value_head) *
                          head_v_dim +
                      value];
  }
  for (int position = threadIdx.x; position < rows;
       position += blockDim.x) {
    shared_decay[position] =
        pending_decay[((size_t)slot * max_rows + position) * num_v_heads +
                      value_head];
  }
  __syncthreads();

  const int state_elements = head_k_dim * head_v_dim;
  const size_t state_head =
      ((size_t)slot * num_v_heads + value_head) * state_elements;
  for (int element = threadIdx.x; element < state_elements;
       element += blockDim.x) {
    const int value = element / head_k_dim;
    const int key = element - value * head_k_dim;
    float state = (float)recurrent_state[state_head + element];
    for (int position = 0; position < rows; position++) {
      state = __fmaf_rn(shared_key[position][key],
                        shared_delta[position][value],
                        __fmul_rn(shared_decay[position], state));
    }
    recurrent_state[state_head + element] = (StateT)state;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    applied_epochs[applied_offset] = epoch;
  }
}

template <typename T>
void dispatch_gdn_pending_transition_apply_batched(
    const uint64_t *pointer_table, const uint32_t *active_slots,
    int layer_count, int batch_size, int max_rows, int capacity,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int state_dtype,
    cudaStream_t stream) {
  const int conv_blocks =
      (conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE;
  dim3 grid(conv_blocks + num_v_heads, batch_size, layer_count);
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    gdn_pending_transition_apply_batched_kernel<T, __half>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    gdn_pending_transition_apply_batched_kernel<T, __nv_bfloat16>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  } else {
    gdn_pending_transition_apply_batched_kernel<T, float>
        <<<grid, GDN_CHANNEL_BLOCK_SIZE, 0, stream>>>(
            pointer_table, active_slots, layer_count, batch_size, max_rows,
            capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            conv_dim, conv_width, tiled_v_heads, conv_blocks);
  }
}

extern "C" void gdn_pending_transition_apply_batched(
    const uint64_t *pointer_table, const uint32_t *active_slots,
    int layer_count, int batch_size, int max_rows, int capacity,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int conv_dim, int conv_width, int tiled_v_heads, int activation_dtype,
    int state_dtype, int64_t stream) {
  if (layer_count <= 0 || batch_size <= 0 || max_rows <= 0 ||
      max_rows > GDN_SPEC_FUSED_MAX_TOKENS || capacity <= 0 ||
      num_k_heads <= 0 || num_v_heads <= 0 ||
      num_v_heads % num_k_heads != 0 ||
      head_k_dim != GDN_DECODE_VALUE_MAJOR_K ||
      head_v_dim != GDN_DECODE_VALUE_MAJOR_V || conv_dim <= 0 ||
      conv_width <= 0 || conv_width > GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  if (activation_dtype == 0) {
    dispatch_gdn_pending_transition_apply_batched<__half>(
        pointer_table, active_slots, layer_count, batch_size, max_rows,
        capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, state_dtype, custream);
  } else {
    dispatch_gdn_pending_transition_apply_batched<__nv_bfloat16>(
        pointer_table, active_slots, layer_count, batch_size, max_rows,
        capacity, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_dim,
        conv_width, tiled_v_heads, state_dtype, custream);
  }
}
