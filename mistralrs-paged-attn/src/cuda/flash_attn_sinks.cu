// flash_attn_sinks.cu
//
// Tiled flash attention with per-head attention sinks for prefill.
// Uses FlashAttention-2 online softmax with shared-memory K/V tiling:
//   - Never materializes the N x N attention matrix
//   - BR warps per block, each handling one Q row
//   - K/V tiles loaded cooperatively into shared memory (float32)
//   - Two-pass per tile: compute scores, then one rescale + V accumulation
//   - Sinks integrated once after all tiles
//
// Layout: Q [B, num_heads, S, D], K [B, num_kv_heads, S, D], V same as K
// GQA: kv_head = q_head / (num_heads / num_kv_heads)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

#define FA_CUDA_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T val);

template <> __device__ __forceinline__ float to_float<__half>(__half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <> __device__ __forceinline__ float to_float<float>(float val) {
  return val;
}

template <typename T> __device__ __forceinline__ T from_float(float val);

template <> __device__ __forceinline__ __half from_float<__half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <> __device__ __forceinline__ float from_float<float>(float val) {
  return val;
}

// ---------------------------------------------------------------------------
// Warp-level reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// Tiled flash attention kernel with sinks
//
// Template params:
//   scalar_t : __half, __nv_bfloat16, or float
//   HEAD_DIM : actual head dimension (64, 80, 96, 112, 128, 192, 256)
//   BR       : number of warps per block (= number of Q rows per block)
//   BC       : KV tile size (number of KV positions per shared-memory tile)
//
// Grid : (num_heads, batch_size, cdiv(q_len, BR))
// Block: (BR * WARP_SIZE)
//
// Shared memory (float32 to avoid bank conflicts):
//   k_smem[BC][D_PAD]  — K tile
//   v_smem[BC][D_PAD]  — V tile
//
// Each warp handles one Q row. Within each tile:
//   Pass 1: compute BC dot-product scores using k_smem
//   Pass 2: one rescale of accumulators, then accumulate V from v_smem
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM, int BR, int BC>
__launch_bounds__(BR *WARP_SIZE) __global__ void flash_attn_sinks_kernel(
    const scalar_t *__restrict__ Q,  // [B, num_heads, q_len, D]
    const scalar_t *__restrict__ K,  // [B, num_kv_heads, kv_len, D]
    const scalar_t *__restrict__ V,  // [B, num_kv_heads, kv_len, D]
    scalar_t *__restrict__ O,        // [B, num_heads, q_len, D]
    const float *__restrict__ sinks, // [num_heads] or nullptr
    const float scale, const int q_len, const int kv_len, const int num_heads,
    const int num_kv_heads,
    const int window_size // 0 = no window (full)
) {
  // Padded head dim for clean warp division (round up to multiple of 32)
  constexpr int D_PAD = ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE; // elements per thread
  constexpr int BLOCK_SIZE = BR * WARP_SIZE;

  // Shared memory for K and V tiles (float32 avoids bank conflicts)
  __shared__ float k_smem[BC * D_PAD];
  __shared__ float v_smem[BC * D_PAD];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Which query row this warp handles
  const int q_row = q_tile_idx * BR + warp_id;

  // Offset between Q local row index and absolute KV position
  // (bottom-right aligned causal masking, matching Metal kernel)
  const int kv_offset = kv_len - q_len;

  // Offsets into contiguous [B, H, S, D] layout
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * q_len + q_row) * HEAD_DIM;
  const int kv_base =
      (batch_idx * num_kv_heads + kv_head_idx) * kv_len * HEAD_DIM;

  // Load Q row into registers (pre-scaled)
  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    q_reg[i] = (q_row < q_len && d < HEAD_DIM)
                   ? to_float(Q[q_offset + d]) * scale
                   : 0.0f;
  }

  // Output accumulator in registers
  float o_acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -FLT_MAX; // running max
  float l_i = 0.0f;     // running sum of exp

  // Block-level KV bounds (union of all warps' causal windows)
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end =
      min(kv_len, block_q_end + kv_offset); // max causal bound across warps

  // Per-warp causal bounds
  const int my_kv_start = (window_size > 0 && q_row < q_len)
                              ? max(0, q_row + kv_offset - window_size + 1)
                              : 0;
  const int my_kv_end = (q_row < q_len) ? (q_row + kv_offset + 1) : 0;

  // Score buffer for current tile (in registers)
  float scores[BC];

  // Tile loop over K/V
  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile into shared memory ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] =
          (kj < tile_len && kd < HEAD_DIM)
              ? to_float(__ldg(&K[kv_base + (tile_start + kj) * HEAD_DIM + kd]))
              : 0.0f;
    }

    // --- Cooperatively load V tile into shared memory ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] =
          (vj < tile_len && vd < HEAD_DIM)
              ? to_float(__ldg(&V[kv_base + (tile_start + vj) * HEAD_DIM + vd]))
              : 0.0f;
    }
    __syncthreads();

    if (q_row < q_len) {
      // --- Pass 1: Compute scores for this tile ---
      float tile_max = -FLT_MAX;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        // Causal: no valid positions beyond q_row
        if (kv_pos >= my_kv_end) {
          // All remaining positions in tile are also beyond causal bound
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -FLT_MAX;
          break;
        }

        // Sliding window check
        if (kv_pos < my_kv_start) {
          scores[j] = -FLT_MAX;
          continue;
        }

        // Dot product: q_reg . K[j] from shared memory
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * WARP_SIZE + lane_id];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = fmaxf(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -FLT_MAX) {
        const float m_new = fmaxf(m_i, tile_max);
        const float rescale = expf(m_i - m_new);

        // Rescale old accumulators (ONCE per tile, not per position)
#pragma unroll
        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        // Accumulate V contributions
        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -FLT_MAX)
            continue; // masked position

          const float p = expf(scores[j] - m_i);
          l_i += p;

#pragma unroll
          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * WARP_SIZE + lane_id];
          }
        }
      }
    }
    __syncthreads();
  }

  if (q_row >= q_len)
    return;

  // Integrate per-head sinks into the softmax denominator.
  // The sink adds exp(sink_val) to the denominator but does NOT
  // contribute any value to the output (virtual token).
  if (sinks != nullptr) {
    const float sink_val = sinks[head_idx];
    const float m_new = fmaxf(m_i, sink_val);
    const float rescale = expf(m_i - m_new);
#pragma unroll
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + expf(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    if (d < HEAD_DIM) {
      O[q_offset + d] = from_float<scalar_t>(o_acc[i] * inv_l);
    }
  }
}

// ---------------------------------------------------------------------------
// Varlen tiled flash attention kernel with sinks
//
// Q is padded: [B, num_heads, max_q_len, D]
// K/V are packed: [total_kv, num_kv_heads, D]
// O is padded: [B, num_heads, max_q_len, D]
//
// cu_seqlens_q[B+1]: cumulative Q lengths
// cu_seqlens_k[B+1]: cumulative KV lengths
//
// Grid : (num_heads, batch_size, cdiv(max_q_len, BR))
// Block: (BR * WARP_SIZE)
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM, int BR, int BC>
__launch_bounds__(BR *WARP_SIZE) __global__ void flash_attn_sinks_varlen_kernel(
    const scalar_t *__restrict__ Q,  // [B, num_heads, max_q_len, D]
    const scalar_t *__restrict__ K,  // [total_kv, num_kv_heads, D]
    const scalar_t *__restrict__ V,  // [total_kv, num_kv_heads, D]
    scalar_t *__restrict__ O,        // [B, num_heads, max_q_len, D]
    const float *__restrict__ sinks, // [num_heads] or nullptr
    const unsigned int *__restrict__ cu_seqlens_q, // [B+1]
    const unsigned int *__restrict__ cu_seqlens_k, // [B+1]
    const float scale, const int max_q_len, const int num_heads,
    const int num_kv_heads,
    const int window_size // 0 = no window (full)
) {
  constexpr int D_PAD = ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE;
  constexpr int BLOCK_SIZE = BR * WARP_SIZE;

  __shared__ float k_smem[BC * D_PAD];
  __shared__ float v_smem[BC * D_PAD];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Per-batch-item lengths (derived from cumulative arrays)
  const int my_q_len =
      (int)(cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]);
  const int kv_start = (int)cu_seqlens_k[batch_idx];
  const int my_kv_len =
      (int)(cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]);

  const int q_row = q_tile_idx * BR + warp_id;

  // Out-of-bounds warps must NOT return early: they participate in cooperative
  // K/V loads and __syncthreads barriers inside the tile loop. Returning early
  // causes barrier divergence (undefined behavior / hangs on partially-filled
  // tiles). Instead, we guard per-warp score computation with
  // `if (q_row < my_q_len)` inside the loop, matching the non-varlen kernel.
  const bool q_row_valid = (q_row < my_q_len);

  // Bottom-right aligned causal offset (safe even for out-of-bounds q_row;
  // only used inside the q_row_valid guard)
  const int kv_offset = my_kv_len - my_q_len;

  // Q offset: padded [B, H, max_q_len, D]
  // For out-of-bounds q_row this may index into padding but q_reg will be
  // zero-masked below, so no correctness issue.
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * max_q_len + q_row) * HEAD_DIM;

  // Load Q row into registers (pre-scaled); out-of-bounds rows get zeros
  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    q_reg[i] = (q_row_valid && d < HEAD_DIM) ? to_float(Q[q_offset + d]) * scale
                                             : 0.0f;
  }

  float o_acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  // Block-level KV bounds (shared across all warps in the block)
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, my_q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end = min(my_kv_len, block_q_end + kv_offset);

  // Per-warp causal bounds (only meaningful when q_row_valid)
  const int my_kv_start_w =
      (window_size > 0) ? max(0, q_row + kv_offset - window_size + 1) : 0;
  const int my_kv_end = q_row + kv_offset + 1;

  float scores[BC];

  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile from packed layout ---
    // K: [total_kv, num_kv_heads, D]
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] =
          (kj < tile_len && kd < HEAD_DIM)
              ? to_float(__ldg(&K[((kv_start + tile_start + kj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  kd]))
              : 0.0f;
    }

    // --- Cooperatively load V tile from packed layout ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] =
          (vj < tile_len && vd < HEAD_DIM)
              ? to_float(__ldg(&V[((kv_start + tile_start + vj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  vd]))
              : 0.0f;
    }
    __syncthreads();

    if (q_row_valid) {
      // --- Pass 1: Compute scores ---
      float tile_max = -FLT_MAX;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        if (kv_pos >= my_kv_end) {
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -FLT_MAX;
          break;
        }

        if (kv_pos < my_kv_start_w) {
          scores[j] = -FLT_MAX;
          continue;
        }

        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * WARP_SIZE + lane_id];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = fmaxf(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -FLT_MAX) {
        const float m_new = fmaxf(m_i, tile_max);
        const float rescale = expf(m_i - m_new);

#pragma unroll
        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -FLT_MAX)
            continue;

          const float p = expf(scores[j] - m_i);
          l_i += p;

#pragma unroll
          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * WARP_SIZE + lane_id];
          }
        }
      }
    }
    __syncthreads();
  }

  // Out-of-bounds warps exit after participating in all barriers
  if (!q_row_valid)
    return;

  // Integrate sinks
  if (sinks != nullptr) {
    const float sink_val = sinks[head_idx];
    const float m_new = fmaxf(m_i, sink_val);
    const float rescale = expf(m_i - m_new);
#pragma unroll
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + expf(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    if (d < HEAD_DIM) {
      O[q_offset + d] = from_float<scalar_t>(o_acc[i] * inv_l);
    }
  }
}

// ---------------------------------------------------------------------------
// Dispatch on HEAD_DIM (compile-time) then launch
// ---------------------------------------------------------------------------

template <typename scalar_t>
void flash_attn_sinks_launch(const void *Q, const void *K, const void *V,
                             void *O, const float *sinks, float scale,
                             int batch_size, int q_len, int kv_len,
                             int num_heads, int num_kv_heads, int head_dim,
                             int window_size, cudaStream_t stream) {
  constexpr int BR = 8; // warps per block (= Q rows per block)
  const dim3 grid(num_heads, batch_size, (q_len + BR - 1) / BR);
  const dim3 block(WARP_SIZE * BR);

#define LAUNCH_KERNEL(D, BC_VAL)                                               \
  flash_attn_sinks_kernel<scalar_t, D, BR, BC_VAL>                             \
      <<<grid, block, 0, stream>>>(reinterpret_cast<const scalar_t *>(Q),      \
                                   reinterpret_cast<const scalar_t *>(K),      \
                                   reinterpret_cast<const scalar_t *>(V),      \
                                   reinterpret_cast<scalar_t *>(O), sinks,     \
                                   scale, q_len, kv_len, num_heads,            \
                                   num_kv_heads, window_size);                 \
  FA_CUDA_CHECK(cudaGetLastError())

  switch (head_dim) {
  case 64:
    LAUNCH_KERNEL(64, 64);
    break;
  case 80:
    LAUNCH_KERNEL(80, 32);
    break;
  case 96:
    LAUNCH_KERNEL(96, 32);
    break;
  case 112:
    LAUNCH_KERNEL(112, 32);
    break;
  case 128:
    LAUNCH_KERNEL(128, 32);
    break;
  case 192:
    LAUNCH_KERNEL(192, 16);
    break;
  case 256:
    LAUNCH_KERNEL(256, 16);
    break;
  default:
    fprintf(stderr,
            "flash_attn_sinks: unsupported head_dim=%d. "
            "Supported: 64, 80, 96, 112, 128, 192, 256\n",
            head_dim);
    break;
  }
#undef LAUNCH_KERNEL
}

// ---------------------------------------------------------------------------
// Extern "C" entry points (one per dtype, matching FFI pattern)
// ---------------------------------------------------------------------------

extern "C" void flash_attn_sinks_f16(const void *Q, const void *K,
                                     const void *V, void *O, const float *sinks,
                                     float scale, int batch_size, int q_len,
                                     int kv_len, int num_heads,
                                     int num_kv_heads, int head_dim,
                                     int window_size, cudaStream_t stream) {
  flash_attn_sinks_launch<__half>(Q, K, V, O, sinks, scale, batch_size, q_len,
                                  kv_len, num_heads, num_kv_heads, head_dim,
                                  window_size, stream);
}

extern "C" void
flash_attn_sinks_bf16(const void *Q, const void *K, const void *V, void *O,
                      const float *sinks, float scale, int batch_size,
                      int q_len, int kv_len, int num_heads, int num_kv_heads,
                      int head_dim, int window_size, cudaStream_t stream) {
  flash_attn_sinks_launch<__nv_bfloat16>(Q, K, V, O, sinks, scale, batch_size,
                                         q_len, kv_len, num_heads, num_kv_heads,
                                         head_dim, window_size, stream);
}

extern "C" void flash_attn_sinks_f32(const void *Q, const void *K,
                                     const void *V, void *O, const float *sinks,
                                     float scale, int batch_size, int q_len,
                                     int kv_len, int num_heads,
                                     int num_kv_heads, int head_dim,
                                     int window_size, cudaStream_t stream) {
  flash_attn_sinks_launch<float>(Q, K, V, O, sinks, scale, batch_size, q_len,
                                 kv_len, num_heads, num_kv_heads, head_dim,
                                 window_size, stream);
}

// ---------------------------------------------------------------------------
// Varlen dispatch on HEAD_DIM (compile-time) then launch
// ---------------------------------------------------------------------------

template <typename scalar_t>
void flash_attn_sinks_varlen_launch(
    const void *Q, const void *K, const void *V, void *O, const float *sinks,
    const unsigned int *cu_seqlens_q, const unsigned int *cu_seqlens_k,
    float scale, int batch_size, int max_q_len, int num_heads, int num_kv_heads,
    int head_dim, int window_size, cudaStream_t stream) {
  constexpr int BR = 8;
  const dim3 grid(num_heads, batch_size, (max_q_len + BR - 1) / BR);
  const dim3 block(WARP_SIZE * BR);

#define LAUNCH_VARLEN_KERNEL(D, BC_VAL)                                        \
  flash_attn_sinks_varlen_kernel<scalar_t, D, BR, BC_VAL>                      \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const scalar_t *>(Q),                               \
          reinterpret_cast<const scalar_t *>(K),                               \
          reinterpret_cast<const scalar_t *>(V),                               \
          reinterpret_cast<scalar_t *>(O), sinks, cu_seqlens_q, cu_seqlens_k,  \
          scale, max_q_len, num_heads, num_kv_heads, window_size);             \
  FA_CUDA_CHECK(cudaGetLastError())

  switch (head_dim) {
  case 64:
    LAUNCH_VARLEN_KERNEL(64, 64);
    break;
  case 80:
    LAUNCH_VARLEN_KERNEL(80, 32);
    break;
  case 96:
    LAUNCH_VARLEN_KERNEL(96, 32);
    break;
  case 112:
    LAUNCH_VARLEN_KERNEL(112, 32);
    break;
  case 128:
    LAUNCH_VARLEN_KERNEL(128, 32);
    break;
  case 192:
    LAUNCH_VARLEN_KERNEL(192, 16);
    break;
  case 256:
    LAUNCH_VARLEN_KERNEL(256, 16);
    break;
  default:
    fprintf(stderr,
            "flash_attn_sinks_varlen: unsupported head_dim=%d. "
            "Supported: 64, 80, 96, 112, 128, 192, 256\n",
            head_dim);
    break;
  }
#undef LAUNCH_VARLEN_KERNEL
}

// ---------------------------------------------------------------------------
// Varlen extern "C" entry points
// ---------------------------------------------------------------------------

extern "C" void flash_attn_sinks_varlen_f16(
    const void *Q, const void *K, const void *V, void *O, const float *sinks,
    const unsigned int *cu_seqlens_q, const unsigned int *cu_seqlens_k,
    float scale, int batch_size, int max_q_len, int num_heads, int num_kv_heads,
    int head_dim, int window_size, cudaStream_t stream) {
  flash_attn_sinks_varlen_launch<__half>(
      Q, K, V, O, sinks, cu_seqlens_q, cu_seqlens_k, scale, batch_size,
      max_q_len, num_heads, num_kv_heads, head_dim, window_size, stream);
}

extern "C" void flash_attn_sinks_varlen_bf16(
    const void *Q, const void *K, const void *V, void *O, const float *sinks,
    const unsigned int *cu_seqlens_q, const unsigned int *cu_seqlens_k,
    float scale, int batch_size, int max_q_len, int num_heads, int num_kv_heads,
    int head_dim, int window_size, cudaStream_t stream) {
  flash_attn_sinks_varlen_launch<__nv_bfloat16>(
      Q, K, V, O, sinks, cu_seqlens_q, cu_seqlens_k, scale, batch_size,
      max_q_len, num_heads, num_kv_heads, head_dim, window_size, stream);
}

extern "C" void flash_attn_sinks_varlen_f32(
    const void *Q, const void *K, const void *V, void *O, const float *sinks,
    const unsigned int *cu_seqlens_q, const unsigned int *cu_seqlens_k,
    float scale, int batch_size, int max_q_len, int num_heads, int num_kv_heads,
    int head_dim, int window_size, cudaStream_t stream) {
  flash_attn_sinks_varlen_launch<float>(
      Q, K, V, O, sinks, cu_seqlens_q, cu_seqlens_k, scale, batch_size,
      max_q_len, num_heads, num_kv_heads, head_dim, window_size, stream);
}
