#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

constexpr int GDN_CHANNEL_BLOCK_SIZE = 256;
constexpr int GDN_DECODE_VALUE_TILE = 64;
constexpr int GDN_DECODE_STATE_LOAD_UNROLL = 128;
constexpr int GDN_DECODE_STATE_UPDATE_TILE_ROWS = 32;
constexpr int GDN_DECODE_COOPERATIVE_K = 128;
constexpr int GDN_DECODE_COOPERATIVE_V = 16;
constexpr int GDN_DECODE_COOPERATIVE_V_PADDED = 20;
constexpr int GDN_DECODE_COOPERATIVE_THREADS = 128;
constexpr int GDN_DECODE_COOPERATIVE_VALUES_PER_WARP = 4;
constexpr int GDN_PACKED_CONV_WIDTH = 4;

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
template <int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const float *__restrict__ q,    // [BH, S, K]
    const float *__restrict__ k,    // [BH, S, K]
    const float *__restrict__ v,    // [BH, S, V]
    const float *__restrict__ g,    // [BH, S]
    const float *__restrict__ beta, // [BH, S]
    float *__restrict__ state,      // [BH, K, V] or the pool with slot_indices
    float *__restrict__ output,     // [BH, S, V]
    int seq_len, int v_dim, const int32_t *__restrict__ slot_indices,
    int num_heads) {

  const int v_tile = blockIdx.x;       // which V-tile
  const int bh = blockIdx.y;           // batch*head index
  const int tid = threadIdx.x;         // thread within tile [0, BV)
  const int v_idx = v_tile * BV + tid; // global V index

  if (v_idx >= v_dim)
    return;

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  float *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;
  float *out_bh = output + (size_t)bh * seq_len * v_dim;

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
template <int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, float *__restrict__ state,
    float *__restrict__ output, int seq_len, int k_dim, int v_dim,
    const int32_t *__restrict__ slot_indices, int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  if (v_idx >= v_dim)
    return;

  const float *q_bh = q + (size_t)bh * seq_len * k_dim;
  const float *k_bh = k + (size_t)bh * seq_len * k_dim;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  float *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * k_dim * v_dim;
  float *out_bh = output + (size_t)bh * seq_len * v_dim;

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

extern "C" void gated_delta_rule_recurrence(const float *q, const float *k,
                                            const float *v, const float *g,
                                            const float *beta, float *state,
                                            float *output, int bh, int seq_len,
                                            int k_dim, int v_dim,
                                            const int32_t *slot_indices,
                                            int num_heads, int64_t stream) {

  const cudaStream_t custream = (cudaStream_t)stream;

  if (k_dim == 128) {
    // Fast path for Qwen3-Next (k_dim=128)
    constexpr int BK = 128;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<BK, BV>
        <<<grid, block, 0, custream>>>(q, k, v, g, beta, state, output, seq_len,
                                       v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    // Fast path for models with k_dim=64
    constexpr int BK = 64;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<BK, BV>
        <<<grid, block, 0, custream>>>(q, k, v, g, beta, state, output, seq_len,
                                       v_dim, slot_indices, num_heads);
  } else {
    // Fallback for other k_dim values (runtime loop, still V-tiled)
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    size_t smem = 2 * k_dim * sizeof(float);
    gated_delta_rule_recurrence_kernel_fallback<BV, MAX_K>
        <<<grid, block, smem, custream>>>(q, k, v, g, beta, state, output,
                                          seq_len, k_dim, v_dim, slot_indices,
                                          num_heads);
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

template <int BK, int NUM_WARPS>
__global__ __launch_bounds__(
    32 * NUM_WARPS,
    2) void gated_delta_rule_recurrence_kernel_warp(const float *__restrict__ q,
                                                    const float *__restrict__ k,
                                                    const float *__restrict__ v,
                                                    const float *__restrict__ g,
                                                    const float
                                                        *__restrict__ beta,
                                                    float *__restrict__ state,
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

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  float *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;
  float *out_bh = output + (size_t)bh * seq_len * v_dim;

  float s[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    s[r] = state_bh[row * v_dim + v_idx];
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
    state_bh[row * v_dim + v_idx] = s[r];
  }
}

extern "C" void warp_gated_delta_rule_recurrence(const float *q, const float *k,
                                                 const float *v, const float *g,
                                                 const float *beta,
                                                 float *state, float *output,
                                                 int bh, int seq_len, int k_dim,
                                                 int v_dim,
                                                 const int32_t *slot_indices,
                                                 int num_heads, int64_t stream) {

  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);

  if (k_dim == 128) {
    gated_delta_rule_recurrence_kernel_warp<128, NUM_WARPS>
        <<<grid, block, 0, custream>>>(q, k, v, g, beta, state, output, seq_len,
                                       v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    gated_delta_rule_recurrence_kernel_warp<64, NUM_WARPS>
        <<<grid, block, 0, custream>>>(q, k, v, g, beta, state, output, seq_len,
                                       v_dim, slot_indices, num_heads);
  } else {
    gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh, seq_len,
                                k_dim, v_dim, slot_indices, num_heads, stream);
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
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

template <int BT, int BK, int BV>
__global__ void
chunked_gated_delta_rule_kernel(const float *__restrict__ q,    // [BH, S, K]
                                const float *__restrict__ k,    // [BH, S, K]
                                const float *__restrict__ v,    // [BH, S, V]
                                const float *__restrict__ g,    // [BH, S]
                                const float *__restrict__ beta, // [BH, S]
                                float *__restrict__ state,      // [BH, K, V] or pool
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

  const int num_chunks = (seq_len + BT - 1) / BT;

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  float *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;
  float *out_bh = output + (size_t)bh * seq_len * v_dim;

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
    s[j] = state_bh[j * v_dim + v_idx];
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
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

extern "C" void chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, float *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int64_t stream) {

  const cudaStream_t custream = (cudaStream_t)stream;

  if (k_dim == 128) {
    constexpr int BT = 64;
    constexpr int BK = 128;
    constexpr int BV = 64;
    // Shared memory: BT*BK + BT*BT + BT + BT + BK floats
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    // Request extended shared memory
    auto kernel = chunked_gated_delta_rule_kernel<BT, BK, BV>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, custream>>>(q, k, v, g, beta, state, output,
                                            seq_len, v_dim, slot_indices,
                                            num_heads);
  } else if (k_dim == 64) {
    constexpr int BT = 64;
    constexpr int BK = 64;
    constexpr int BV = 64;
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    auto kernel = chunked_gated_delta_rule_kernel<BT, BK, BV>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, custream>>>(q, k, v, g, beta, state, output,
                                            seq_len, v_dim, slot_indices,
                                            num_heads);
  } else {
    // Fallback: use the sequential kernel for unsupported k_dim
    gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh, seq_len,
                                k_dim, v_dim, slot_indices, num_heads, stream);
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
__global__ void causal_conv1d_update_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const size_t state_idx = state_row * conv_dim + ch;
  const size_t input_idx = (size_t)b * conv_dim + ch;
  const size_t x_idx = (size_t)b * x_stride_b + (size_t)ch * x_stride_c;
  auto *state = reinterpret_cast<GdnConvWidth4<T> *>(conv_state);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  GdnConvWidth4<T> values = state[state_idx];
  const GdnConvWidth4<T> channel_weights = weights[ch];

#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
    values.values[i] = values.values[i + 1];
  }
  values.values[GDN_PACKED_CONV_WIDTH - 1] = x[x_idx];
  state[state_idx] = values;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
    acc += (float)values.values[i] * (float)channel_weights.values[i];
  }
  const float sig = 1.0f / (1.0f + expf(-acc));
  output[input_idx] = (T)(acc * sig);
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

  output[((size_t)b * seq_len + pos) * conv_dim + ch] = (T)result;
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

  // Main convolution kernel
  dim3 block(256);
  const size_t plane = (size_t)conv_dim * seq_len;
  dim3 grid((unsigned int)((plane + 255) / 256), batch_size);

  if (dtype == 0) {
    causal_conv1d_full_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight,
        (const __half *)conv_state_in, (__half *)output, batch_size, conv_dim,
        seq_len, kernel_size, x_stride_b, x_stride_s, x_stride_c,
        slot_indices);
    dim3 grid2((conv_dim + 255) / 256, batch_size);
    save_conv_state_kernel<__half><<<grid2, block, 0, custream>>>(
        (const __half *)x, (const __half *)conv_state_in,
        (__half *)conv_state_out, batch_size, conv_dim, seq_len, kernel_size,
        x_stride_b, x_stride_s, x_stride_c, slot_indices);
  } else {
    causal_conv1d_full_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (const __nv_bfloat16 *)conv_state_in, (__nv_bfloat16 *)output,
        batch_size, conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s,
        x_stride_c, slot_indices);
    dim3 grid2((conv_dim + 255) / 256, batch_size);
    save_conv_state_kernel<__nv_bfloat16><<<grid2, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)conv_state_in,
        (__nv_bfloat16 *)conv_state_out, batch_size, conv_dim, seq_len,
        kernel_size, x_stride_b, x_stride_s, x_stride_c, slot_indices);
  }
}

template <typename T>
__global__ void gdn_prepare_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, float *__restrict__ q_out,
    float *__restrict__ k_out, float *__restrict__ v_out,
    float *__restrict__ g_out, float *__restrict__ beta_out, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int tiled_v_heads) {
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
    beta_out[(size_t)bh * seq_len + t] = 1.0f / (1.0f + expf(-b_val));
    g_out[(size_t)bh * seq_len + t] = -expf(a_log[hv]) * softplus_val;
  }
  __syncthreads();

  float *q_dst = q_out + ((size_t)bh * seq_len + t) * head_k_dim;
  float *k_dst = k_out + ((size_t)bh * seq_len + t) * head_k_dim;
  float *v_dst = v_out + ((size_t)bh * seq_len + t) * head_v_dim;

  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_dst[d] = q_val * q_mul;
    k_dst[d] = k_val * k_mul;
  }

  for (int d = tid; d < head_v_dim; d += blockDim.x) {
    v_dst[d] = (float)row[2 * key_dim + hv * head_v_dim + d];
  }
}

extern "C" void gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, float *q_out, float *k_out, float *v_out,
    float *g_out, float *beta_out, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int tiled_v_heads,
    int dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(256);
  dim3 grid(batch_size * seq_len * num_v_heads);

  if (dtype == 0) {
    gdn_prepare_recurrence_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a, a_log,
        dt_bias, q_out, k_out, v_out, g_out, beta_out, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads);
  } else {
    gdn_prepare_recurrence_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, q_out, k_out, v_out, g_out,
        beta_out, batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim,
        head_v_dim, tiled_v_heads);
  }
}

__device__ __forceinline__ float gdn_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float gdn_cooperative_k_sum(float value) {
#pragma unroll
  for (int offset = 16; offset >= GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
       offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

// Adapted from FlashInfer's Apache-2.0 nontranspose GDN kernel, Copyright (c) 2025 FlashInfer team.
// Exact source revision and license notices are in third_party/flashinfer_gdn.
template <typename T>
__global__ void gdn_decode_recurrence_kernel_cooperative(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, float *__restrict__ state,
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

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const T *row = mixed_qkv + bidx * conv_dim;
  float *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;
  T *out_bh = output + bh * head_v_dim;

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
    const float *src =
        state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    float *dst = state_buf + k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_vector * 4;
    __pipeline_memcpy_async(dst, src, sizeof(float4));
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
  state_dot_k = gdn_cooperative_k_sum(state_dot_k);

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
  state_dot_q = gdn_cooperative_k_sum(state_dot_q);
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
    float *dst = state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
  }
}

template <typename T, int BK, int BV>
__global__ void gdn_decode_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, float *__restrict__ state,
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

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  float *state_bh = state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;
  T *out_bh = output + bh * head_v_dim;

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
    const float s = state_bh[j * head_v_dim + v_idx] * decay_t;
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

template <typename T, int BV, int MAX_K>
__global__ void gdn_decode_recurrence_kernel_fallback(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, float *__restrict__ state,
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

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  float *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * head_k_dim * head_v_dim;
  T *out_bh = output + bh * head_v_dim;

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
    s[j] = state_bh[j * head_v_dim + v_idx] * decay_t;
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

extern "C" void
gdn_decode_recurrence(const void *mixed_qkv, const void *b, const void *a,
                      const float *a_log, const float *dt_bias, float *state,
                      void *output, int batch_size, int num_k_heads,
                      int num_v_heads, int head_k_dim, int head_v_dim,
                      int tiled_v_heads, int64_t b_batch_stride,
                      int64_t b_head_stride, int64_t a_batch_stride,
                      int64_t a_head_stride, const int32_t *slot_indices,
                      int use_cooperative, int dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int BV = GDN_DECODE_VALUE_TILE;
  dim3 grid((head_v_dim + BV - 1) / BV, batch_size * num_v_heads);
  dim3 block(BV);

  if (use_cooperative) {
    constexpr int COOPERATIVE_BV = GDN_DECODE_COOPERATIVE_V;
    dim3 cooperative_grid(head_v_dim / COOPERATIVE_BV,
                          batch_size * num_v_heads);
    dim3 cooperative_block(GDN_DECODE_COOPERATIVE_THREADS);
    if (dtype == 0) {
      gdn_decode_recurrence_kernel_cooperative<__half>
          <<<cooperative_grid, cooperative_block, 0, custream>>>(
              (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
              a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
              num_v_heads, head_v_dim, tiled_v_heads, b_batch_stride,
              b_head_stride, a_batch_stride, a_head_stride, slot_indices);
    } else {
      gdn_decode_recurrence_kernel_cooperative<__nv_bfloat16>
          <<<cooperative_grid, cooperative_block, 0, custream>>>(
              (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
              (const __nv_bfloat16 *)a, a_log, dt_bias, state,
              (__nv_bfloat16 *)output, batch_size, num_k_heads, num_v_heads,
              head_v_dim, tiled_v_heads, b_batch_stride, b_head_stride,
              a_batch_stride, a_head_stride, slot_indices);
    }
  } else if (head_k_dim == 128) {
    if (dtype == 0) {
      gdn_decode_recurrence_kernel<__half, 128, BV>
          <<<grid, block, 0, custream>>>(
              (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
              a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
              num_v_heads, head_v_dim, tiled_v_heads, b_batch_stride,
              b_head_stride, a_batch_stride, a_head_stride, slot_indices);
    } else {
      gdn_decode_recurrence_kernel<__nv_bfloat16, 128, BV>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
              (const __nv_bfloat16 *)a, a_log, dt_bias, state, (__nv_bfloat16 *)output,
              batch_size, num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
              b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
              slot_indices);
    }
  } else if (head_k_dim == 64) {
    if (dtype == 0) {
      gdn_decode_recurrence_kernel<__half, 64, BV>
          <<<grid, block, 0, custream>>>(
              (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
              a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
              num_v_heads, head_v_dim, tiled_v_heads, b_batch_stride,
              b_head_stride, a_batch_stride, a_head_stride, slot_indices);
    } else {
      gdn_decode_recurrence_kernel<__nv_bfloat16, 64, BV>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
              (const __nv_bfloat16 *)a, a_log, dt_bias, state, (__nv_bfloat16 *)output,
              batch_size, num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
              b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
              slot_indices);
    }
  } else {
    constexpr int MAX_K = 256;
    size_t smem = (2 * BV + 2 * head_k_dim) * sizeof(float);
    if (dtype == 0) {
      gdn_decode_recurrence_kernel_fallback<__half, BV, MAX_K>
          <<<grid, block, smem, custream>>>(
              (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
              a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
              num_v_heads, head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride,
              b_head_stride, a_batch_stride, a_head_stride, slot_indices);
    } else {
      gdn_decode_recurrence_kernel_fallback<__nv_bfloat16, BV, MAX_K>
          <<<grid, block, smem, custream>>>(
              (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
              (const __nv_bfloat16 *)a, a_log, dt_bias, state, (__nv_bfloat16 *)output,
              batch_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
              tiled_v_heads, b_batch_stride, b_head_stride, a_batch_stride,
              a_head_stride, slot_indices);
    }
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
