// flash_attn_sinks.cu
//
// Fused flash attention with per-head attention sinks for prefill.
// Uses FlashAttention-2 online softmax algorithm:
//   - Never materializes the N x N attention matrix
//   - Processes Q one row at a time (one warp per row)
//   - Integrates per-head sinks into the softmax denominator
//
// Sinks semantics: a "virtual" KV pair that contributes exp(sink_h) to the
// softmax denominator but has no V contribution (probability mass absorption).
//
// Layout: Q [B, num_heads, S, D], K [B, num_kv_heads, S, D], V same as K
// GQA: kv_head = q_head / (num_heads / num_kv_heads)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

#define FA_CUDA_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<__half>(__half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float to_float<float>(float val) {
  return val;
}

template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ __half from_float<__half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16
from_float<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ float from_float<float>(float val) {
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

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, offset));
  }
  return val;
}

// ---------------------------------------------------------------------------
// Flash attention kernel with sinks
//
// One warp per query row. NUM_WARPS warps per block (= NUM_WARPS Q rows/block)
// Grid: (num_heads, batch_size, cdiv(seq_len, NUM_WARPS))
// Block: (WARP_SIZE * NUM_WARPS)
//
// Each thread in a warp handles HEAD_DIM_PAD / WARP_SIZE elements of the
// head dimension (coalesced: lane i reads d = i, i+32, i+64, ...).
//
// Template params:
//   scalar_t: __half, __nv_bfloat16, or float
//   HEAD_DIM: actual head dimension (64, 80, 96, 112, 128, 256)
//   NUM_WARPS: number of warps (= Q rows) per block
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
__launch_bounds__(WARP_SIZE *NUM_WARPS) __global__
    void flash_attn_sinks_kernel(
        const scalar_t *__restrict__ Q, // [B, num_heads, S, D]
        const scalar_t *__restrict__ K, // [B, num_kv_heads, S, D]
        const scalar_t *__restrict__ V, // [B, num_kv_heads, S, D]
        scalar_t *__restrict__ O,       // [B, num_heads, S, D]
        const float *__restrict__ sinks, // [num_heads] or nullptr
        const float scale, const int seq_len, const int num_heads,
        const int num_kv_heads, const int window_size // 0 = no window (full)
    ) {
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Which query row this warp handles
  const int q_row = q_tile_idx * NUM_WARPS + warp_id;
  if (q_row >= seq_len)
    return;

  // Padded head dim for clean warp division
  constexpr int D_PAD =
      ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE; // elements per thread

  // Offsets into contiguous [B, H, S, D] layout
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * seq_len + q_row) * HEAD_DIM;
  const int kv_head_offset =
      (batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_DIM;

  // Load Q row into registers (pre-scaled)
  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    q_reg[i] = (d < HEAD_DIM) ? to_float(Q[q_offset + d]) * scale : 0.0f;
  }

  // Output accumulator in registers
  float o_acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -FLT_MAX; // running max
  float l_i = 0.0f;     // running sum of exp

  // Causal attention: attend to positions [kv_start, q_row] inclusive
  const int kv_start =
      (window_size > 0) ? max(0, q_row - window_size + 1) : 0;
  const int kv_end = q_row + 1;

  // Process K/V positions one at a time with online softmax
  for (int j = kv_start; j < kv_end; j++) {
    const int kv_offset = kv_head_offset + j * HEAD_DIM;

    // Dot product: q_reg . K[j]
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; i++) {
      const int d = i * WARP_SIZE + lane_id;
      const float k_val =
          (d < HEAD_DIM) ? to_float(__ldg(&K[kv_offset + d])) : 0.0f;
      dot += q_reg[i] * k_val;
    }
    dot = warp_reduce_sum(dot); // broadcast to all lanes

    // Online softmax update
    const float m_new = fmaxf(m_i, dot);
    const float rescale = expf(m_i - m_new);
    const float p = expf(dot - m_new);

// Rescale old accumulators and update running stats
#pragma unroll
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + p;
    m_i = m_new;

// Accumulate: o_acc += p * V[j]
#pragma unroll
    for (int i = 0; i < EPT; i++) {
      const int d = i * WARP_SIZE + lane_id;
      const float v_val =
          (d < HEAD_DIM) ? to_float(__ldg(&V[kv_offset + d])) : 0.0f;
      o_acc[i] += p * v_val;
    }
  }

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
// Dispatch on HEAD_DIM (compile-time) then launch
// ---------------------------------------------------------------------------

template <typename scalar_t>
void flash_attn_sinks_launch(const void *Q, const void *K, const void *V,
                             void *O, const float *sinks, float scale,
                             int batch_size, int seq_len, int num_heads,
                             int num_kv_heads, int head_dim, int window_size,
                             cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;
  const dim3 grid(num_heads, batch_size,
                  (seq_len + NUM_WARPS - 1) / NUM_WARPS);
  const dim3 block(WARP_SIZE * NUM_WARPS);

#define LAUNCH_KERNEL(D)                                                       \
  flash_attn_sinks_kernel<scalar_t, D, NUM_WARPS><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                   \
      reinterpret_cast<const scalar_t *>(K),                                   \
      reinterpret_cast<const scalar_t *>(V),                                   \
      reinterpret_cast<scalar_t *>(O), sinks, scale, seq_len, num_heads,       \
      num_kv_heads, window_size);                                              \
  FA_CUDA_CHECK(cudaGetLastError())

  switch (head_dim) {
  case 64:
    LAUNCH_KERNEL(64);
    break;
  case 80:
    LAUNCH_KERNEL(80);
    break;
  case 96:
    LAUNCH_KERNEL(96);
    break;
  case 112:
    LAUNCH_KERNEL(112);
    break;
  case 128:
    LAUNCH_KERNEL(128);
    break;
  case 192:
    LAUNCH_KERNEL(192);
    break;
  case 256:
    LAUNCH_KERNEL(256);
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
                                     const void *V, void *O,
                                     const float *sinks, float scale,
                                     int batch_size, int seq_len,
                                     int num_heads, int num_kv_heads,
                                     int head_dim, int window_size,
                                     cudaStream_t stream) {
  flash_attn_sinks_launch<__half>(Q, K, V, O, sinks, scale, batch_size,
                                  seq_len, num_heads, num_kv_heads, head_dim,
                                  window_size, stream);
}

extern "C" void flash_attn_sinks_bf16(const void *Q, const void *K,
                                      const void *V, void *O,
                                      const float *sinks, float scale,
                                      int batch_size, int seq_len,
                                      int num_heads, int num_kv_heads,
                                      int head_dim, int window_size,
                                      cudaStream_t stream) {
  flash_attn_sinks_launch<__nv_bfloat16>(Q, K, V, O, sinks, scale, batch_size,
                                         seq_len, num_heads, num_kv_heads,
                                         head_dim, window_size, stream);
}

extern "C" void flash_attn_sinks_f32(const void *Q, const void *K,
                                     const void *V, void *O,
                                     const float *sinks, float scale,
                                     int batch_size, int seq_len,
                                     int num_heads, int num_kv_heads,
                                     int head_dim, int window_size,
                                     cudaStream_t stream) {
  flash_attn_sinks_launch<float>(Q, K, V, O, sinks, scale, batch_size, seq_len,
                                 num_heads, num_kv_heads, head_dim, window_size,
                                 stream);
}
