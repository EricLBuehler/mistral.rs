// Adapted from ggml gated-delta-net-mma.cu at 7aca0d2ea; license:
// ../../third_party/ggml_gdn/LICENSE.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int D = 128;
constexpr int CHUNK = 16;
constexpr int THREADS = 256;
constexpr float NORM_EPS = 1.0e-6f;
constexpr float SOFTPLUS_THRESHOLD = 20.0f;
using bf16 = __nv_bfloat16;

struct GdnMmaParams {
  const float *q;
  const float *k;
  const float *v;
  const float *g;
  const float *beta;
  float *state;
  bf16 *output;
  const int32_t *slots;
  int batch_size;
  int num_heads;
  int seq_len;
  int64_t stream;
};

struct GdnMmaFusedParams {
  const bf16 *mixed_qkv;
  const bf16 *b;
  const bf16 *a;
  const float *a_log;
  const float *dt_bias;
  float *state;
  bf16 *output;
  const int32_t *slots;
  int batch_size;
  int num_heads;
  int seq_len;
  int num_k_heads;
  int tiled_v_heads;
  int64_t stream;
};

__device__ __forceinline__ void read_qk(const GdnMmaParams &a, int seq,
                                        int head, int t, int lane, float4 &q,
                                        float4 &k) {
  const int64_t off =
      (((int64_t)seq * a.num_heads + head) * a.seq_len + t) * D + 4 * lane;
  q = *reinterpret_cast<const float4 *>(a.q + off);
  k = *reinterpret_cast<const float4 *>(a.k + off);
}

__device__ __forceinline__ void read_qk(const GdnMmaFusedParams &a, int seq,
                                        int head, int t, int lane, float4 &q,
                                        float4 &k) {
  const int key_head = a.tiled_v_heads ? head % a.num_k_heads
                                       : head / (a.num_heads / a.num_k_heads);
  const int64_t off =
      ((int64_t)seq * a.seq_len + t) * ((2 * a.num_k_heads + a.num_heads) * D) +
      key_head * D + 4 * lane;
  const auto q01 = *reinterpret_cast<const __nv_bfloat162 *>(a.mixed_qkv + off);
  const auto q23 =
      *reinterpret_cast<const __nv_bfloat162 *>(a.mixed_qkv + off + 2);
  const auto k01 = *reinterpret_cast<const __nv_bfloat162 *>(a.mixed_qkv + off +
                                                             a.num_k_heads * D);
  const auto k23 = *reinterpret_cast<const __nv_bfloat162 *>(
      a.mixed_qkv + off + a.num_k_heads * D + 2);
  const float2 q0 = __bfloat1622float2(q01), q1 = __bfloat1622float2(q23);
  const float2 k0 = __bfloat1622float2(k01), k1 = __bfloat1622float2(k23);
  float qsum = q0.x * q0.x + q0.y * q0.y + q1.x * q1.x + q1.y * q1.y;
  float ksum = k0.x * k0.x + k0.y * k0.y + k1.x * k1.x + k1.y * k1.y;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    qsum += __shfl_xor_sync(0xffffffff, qsum, offset);
    ksum += __shfl_xor_sync(0xffffffff, ksum, offset);
  }
  const float qmul = rsqrtf(qsum + NORM_EPS) * rsqrtf((float)D);
  const float kmul = rsqrtf(ksum + NORM_EPS);
  q = make_float4(q0.x * qmul, q0.y * qmul, q1.x * qmul, q1.y * qmul);
  k = make_float4(k0.x * kmul, k0.y * kmul, k1.x * kmul, k1.y * kmul);
}

__device__ __forceinline__ void read_gates(const GdnMmaParams &a, int seq,
                                           int head, int t, float &g,
                                           float &beta) {
  const int64_t off = ((int64_t)seq * a.num_heads + head) * a.seq_len + t;
  g = a.g[off];
  beta = a.beta[off];
}

__device__ __forceinline__ void read_gates(const GdnMmaFusedParams &a, int seq,
                                           int head, int t, float &g,
                                           float &beta) {
  const int64_t off = ((int64_t)seq * a.seq_len + t) * a.num_heads + head;
  const float gate = __bfloat162float(a.a[off]) + a.dt_bias[head];
  const float softplus =
      gate > SOFTPLUS_THRESHOLD
          ? gate
          : (gate > 0.0f ? gate + log1pf(expf(-gate)) : log1pf(expf(gate)));
  g = -expf(a.a_log[head]) * softplus;
  beta = 1.0f / (1.0f + expf(-__bfloat162float(a.b[off])));
}

__device__ __forceinline__ float read_value(const GdnMmaParams &a, int seq,
                                            int head, int t, int value) {
  return a.v[(((int64_t)seq * a.num_heads + head) * a.seq_len + t) * D + value];
}

__device__ __forceinline__ float read_value(const GdnMmaFusedParams &a, int seq,
                                            int head, int t, int value) {
  const int64_t off =
      ((int64_t)seq * a.seq_len + t) * ((2 * a.num_k_heads + a.num_heads) * D) +
      (2 * a.num_k_heads + head) * D + value;
  return __bfloat162float(a.mixed_qkv[off]);
}

struct acc {
  static constexpr int ne = 4;
  float x[ne] = {};
  static __device__ __forceinline__ int get_i(int i) {
    return threadIdx.x / 4 + (i / 2) * 8;
  }
  static __device__ __forceinline__ int get_j(int i) {
    return 2 * (threadIdx.x % 4) + i % 2;
  }
};

template <int I> struct operand {
  static constexpr int rows = I;
  uint32_t x[I / 4];
};
using a_tile = operand<16>;
using b_tile = operand<8>;

template <int STRIDE>
__device__ __forceinline__ int swizzle(int offset, int row) {
  if constexpr ((STRIDE * 4) % 32 != 0)
    return offset;
  if constexpr ((STRIDE * 4) % 64 != 0)
    return offset ^ (((row / 4) % 2) << 2);
  if constexpr ((STRIDE * 4) % 128 != 0)
    return offset ^ (((row / 2) % 4) << 2);
  return offset ^ ((row % 8) << 2);
}

template <int R, int C> struct matrix {
  bf16 hi[R * C], lo[R * C];
  static __device__ __forceinline__ int index(int r, int c) {
    return 2 * swizzle<C / 2>(r * (C / 2) + c / 2, r) + c % 2;
  }
  __device__ __forceinline__ void store2(int r, int c, float2 x) {
    const int i = index(r, c);
    const auto h = __float22bfloat162_rn(x);
    const float2 rounded = __bfloat1622float2(h);
    *reinterpret_cast<__nv_bfloat162 *>(hi + i) = h;
    *reinterpret_cast<__nv_bfloat162 *>(lo + i) =
        __float22bfloat162_rn(make_float2(x.x - rounded.x, x.y - rounded.y));
  }
  __device__ __forceinline__ void store(int r, int c, float x) {
    const int i = index(r, c);
    const bf16 h = __float2bfloat16(x);
    hi[i] = h;
    lo[i] = __float2bfloat16(x - __bfloat162float(h));
  }
};

template <int C, int V> struct alignas(128) shared {
  matrix<C, D> q, k;
  matrix<C, C> p, inverse;
  matrix<C, V> delta;
  union {
    matrix<D, V> state;
    matrix<C, V> solved;
  } scratch;
  float lower[C * C], prefix[C], beta[C];
  static __device__ __forceinline__ int lower_index(int r, int c) {
    return r * C + c;
  }
};

template <class T, bool TRANS, int R, int C>
__device__ __forceinline__ T load(const matrix<R, C> &m, int row, int col,
                                  bool low) {
  T t;
#if __CUDA_ARCH__ >= 800
  constexpr int I = T::rows;
  const int r = TRANS ? threadIdx.x % 16 : threadIdx.x % I;
  const int c = TRANS ? (I == 16 ? (threadIdx.x / 16) * 4 : 0)
                      : ((threadIdx.x / I) * 4) % 8;
  const int start = TRANS ? col * (C / 2) + row / 2 : row * (C / 2) + col / 2;
  const int offset = swizzle<C / 2>(start + r * (C / 2) + c, r);
  const auto *packed = reinterpret_cast<const uint32_t *>(low ? m.lo : m.hi);
  const uint32_t address = __cvta_generic_to_shared(packed + offset);
  if constexpr (TRANS && I == 16) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(t.x[0]), "=r"(t.x[2]), "=r"(t.x[1]), "=r"(t.x[3])
        : "r"(address));
  } else if constexpr (TRANS) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(t.x[0]), "=r"(t.x[1])
                 : "r"(address));
  } else if constexpr (I == 16) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(t.x[0]), "=r"(t.x[1]), "=r"(t.x[2]), "=r"(t.x[3])
                 : "r"(address));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(t.x[0]), "=r"(t.x[1])
                 : "r"(address));
  }
#endif
  return t;
}

__device__ __forceinline__ void mma(acc &d, const a_tile &a, const b_tile &b) {
#if __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]));
#endif
}

// Retain both first-order BF16 residual products; the recurrent state remains
// FP32.
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR,
          int BC>
__device__ __forceinline__ void product(acc &c, const matrix<AR, AC> &a,
                                        const matrix<BR, BC> &b, int row,
                                        int col) {
#pragma unroll
  for (int k = 0; k < K; k += 16) {
    const auto ah = load<a_tile, TA>(a, row, k, false);
    const auto al = load<a_tile, TA>(a, row, k, true);
    const auto bh = load<b_tile, TB>(b, col, k, false);
    const auto bl = load<b_tile, TB>(b, col, k, true);
    mma(c, al, bh);
    mma(c, ah, bl);
    mma(c, ah, bh);
  }
}

template <int R, int C>
__device__ __forceinline__ void store(matrix<R, C> &m, const acc &x, int r,
                                      int c) {
#pragma unroll
  for (int i = 0; i < acc::ne; i += 2) {
    m.store2(r + acc::get_i(i), c + acc::get_j(i),
             make_float2(x.x[i], x.x[i + 1]));
  }
}

template <int C, int V, int BLOCKS = 1, typename Params>
static __global__ __launch_bounds__(THREADS, BLOCKS) void gdn_single(Params a) {
#if __CUDA_ARCH__ >= 800
  constexpr int WARPS = 8;
  constexpr int N = 8;
  static_assert(C >= 16 && (C & (C - 1)) == 0 && V % N == 0 && D % V == 0,
                "invalid GDN tile");
  extern __shared__ __align__(128) unsigned char bytes[];
  auto &s = *reinterpret_cast<shared<C, V> *>(bytes);
  const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
  const int slice = blockIdx.x % (D / V);
  const int h = (blockIdx.x / (D / V)) % a.num_heads;
  const int seq = blockIdx.x / ((D / V) * a.num_heads);
  const int slot = a.slots ? a.slots[seq] : seq;
  const int v0 = slice * V;
  bf16 *out = a.output + ((int64_t)seq * a.seq_len * a.num_heads + h) * D + v0;
  if (slot < 0) {
    for (int i = tid; i < a.seq_len * V; i += THREADS) {
      out[(int64_t)(i / V) * a.num_heads * D + i % V] = __float2bfloat16(0.0f);
    }
    return;
  }
  const int64_t soff = ((int64_t)slot * a.num_heads + h) * D * D;
  constexpr int STATE_TILES = (D / 16) * (V / N);
  constexpr int PER_WARP = STATE_TILES / WARPS;
  static_assert(STATE_TILES % WARPS == 0, "state must divide across warps");
  acc state[PER_WARP];
#pragma unroll
  for (int j = 0; j < PER_WARP; ++j) {
    const int tile_index = warp + j * WARPS, r = (tile_index / (V / N)) * 16,
              c = (tile_index % (V / N)) * N;
#pragma unroll
    for (int i = 0; i < acc::ne; ++i) {
      state[j].x[i] =
          a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)];
    }
  }
  auto load_qk = [&](int64_t t, float4 &q, float4 &k) {
    t += warp;
    q = k = make_float4(0, 0, 0, 0);
    if (t < a.seq_len) {
      read_qk(a, seq, h, t, lane, q, k);
    }
  };
  // Carry the first eight input rows across chunks to hide global-load latency.
  constexpr bool PREFETCH = C == 16 && V == 32;
  float4 next_q, next_k;
  if constexpr (PREFETCH) {
    load_qk(0, next_q, next_k);
  }
  for (int64_t t0 = 0; t0 < a.seq_len; t0 += C) {
    const int valid = min((int64_t)C, a.seq_len - t0);
#pragma unroll
    for (int j = 0; j < C / 8; ++j) {
      const int t = warp + 8 * j, d = 4 * lane;
      float4 q, k;
      if (PREFETCH && j == 0) {
        q = next_q;
        k = next_k;
      } else {
        load_qk(t0 + 8 * j, q, k);
      }
      s.q.store2(t, d, make_float2(q.x, q.y));
      s.q.store2(t, d + 2, make_float2(q.z, q.w));
      s.k.store2(t, d, make_float2(k.x, k.y));
      s.k.store2(t, d + 2, make_float2(k.z, k.w));
    }
    // Load gates in parallel before the FP32 prefix scan.
    if (warp == 0) {
      float carry = 0.f;
      for (int base = 0; base < C; base += 32) {
        const int t = base + lane;
        float g = 0.f, beta = 0.f;
        if (t < valid) {
          read_gates(a, seq, h, t0 + t, g, beta);
        }
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
          const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
          if (lane >= offset) {
            g += previous;
          }
        }
        g += carry;
        if (t < C) {
          s.prefix[t] = g;
          s.beta[t] = beta;
        }
        carry = __shfl_sync(0xffffffff, g, 31, 32);
      }
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
      const int ti = warp + j * WARPS;
      store(s.scratch.state, state[j], (ti / (V / N)) * 16, (ti % (V / N)) * N);
    }
    __syncthreads();
    if constexpr (PREFETCH) {
      if (t0 + C < a.seq_len) {
        load_qk(t0 + C, next_q, next_k);
      }
    }
    // V32 overlaps Gram work with one output tile per projection warp.
    constexpr bool SPLIT_PROJECTION = C == 16 && V == 32;
    acc base_output[((C / 16) * (V / N) + WARPS - 1) / WARPS];
    // delta = beta * (v - decay * K S), base output = decay * Q S
    auto project = [&]() {
      if constexpr (V == 32) {
        if (warp < 4) {
          return;
        }
      }
      for (int ti = (V == 32 ? warp - 4 : warp); ti < (C / 16) * (V / N);
           ti += (V == 32 ? 4 : WARPS)) {
        const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
        acc qs, ks;
        product<D, false, true>(qs, s.q, s.scratch.state, r, c);
        product<D, false, true>(ks, s.k, s.scratch.state, r, c);
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
          const int t = r + acc::get_i(i), v = c + acc::get_j(i);
          const float decay = expf(s.prefix[t]);
          const float vv =
              t < valid ? read_value(a, seq, h, t0 + t, v0 + v) : 0.f;
          s.delta.store(t, v, s.beta[t] * (vv - decay * ks.x[i]));
          base_output[ti / WARPS].x[i] = decay * qs.x[i];
        }
      }
    };
    // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
    for (int ti = warp; ti < 2 * (C / 16) * (C / N) && (V != 32 || warp < 4);
         ti += (V == 32 ? 4 : WARPS)) {
      const bool gram = ti < (C / 16) * (C / N);
      const int t = ti % ((C / 16) * (C / N)), r = (t / (C / N)) * 16,
                c = (t % (C / N)) * N;
      acc x;
      if (gram) {
        product<D>(x, s.k, s.k, r, c);
      } else {
        product<D>(x, s.q, s.k, r, c);
      }
#pragma unroll
      for (int i = 0; i < acc::ne; ++i) {
        const int rr = r + acc::get_i(i), cc = c + acc::get_j(i);
        const float v = rr < valid && cc <= rr
                            ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc])
                            : 0.f;
        if (gram) {
          if (cc < rr) {
            s.lower[s.lower_index(rr, cc)] = v * s.beta[rr];
          }
        } else {
          s.p.store(rr, cc, v);
        }
      }
    }
    if constexpr (V == 32) {
      project();
    }
    __syncthreads();
    // inverse = (I + A)^-1 by forward substitution
    for (int row = warp; row < C; row += WARPS) {
      float x[(C + 31) / 32];
#pragma unroll
      for (int j = 0; j < (C + 31) / 32; ++j) {
        const int col = lane + 32 * j;
        x[j] = col == row
                   ? 1.f
                   : (col < row ? -s.lower[s.lower_index(row, col)] : 0.f);
      }
      for (int pivot = row - 1; pivot > 0; --pivot) {
        const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#pragma unroll
        for (int j = 0; j < (C + 31) / 32; ++j) {
          const int col = lane + 32 * j;
          if (col < pivot) {
            x[j] -= xp * s.lower[s.lower_index(pivot, col)];
          }
        }
      }
#pragma unroll
      for (int j = 0; j < (C + 31) / 32; ++j) {
        if (lane + 32 * j < C) {
          s.inverse.store(row, lane + 32 * j, x[j]);
        }
      }
    }
    if constexpr (V != 32) {
      project();
    }
    __syncthreads();
    // solved = inverse * delta
    for (int ti = warp; ti < (C / 16) * (V / N); ti += WARPS) {
      const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
      acc x;
      product<C, false, true>(x, s.inverse, s.delta, r, c);
      store(s.scratch.solved, x, r, c);
    }
    __syncthreads();
    // out = base output + P * solved
    for (int ti = (SPLIT_PROJECTION ? warp - 4 : warp);
         ti < (C / 16) * (V / N) && (!SPLIT_PROJECTION || warp >= 4);
         ti += WARPS) {
      const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
      acc x;
      product<C, false, true>(x, s.p, s.scratch.solved, r, c);
#pragma unroll
      for (int i = 0; i < acc::ne; ++i) {
        const int t = r + acc::get_i(i), v = c + acc::get_j(i);
        if (t < valid) {
          out[(t0 + t) * a.num_heads * D + v] =
              __float2bfloat16(base_output[ti / WARPS].x[i] + x.x[i]);
        }
      }
    }
    // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
    for (int i = tid; i < C * V; i += THREADS) {
      const int t = i / V, v = i % V, j = s.scratch.solved.index(t, v);
      const float x = __bfloat162float(s.scratch.solved.hi[j]) +
                      __bfloat162float(s.scratch.solved.lo[j]);
      s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
    }
    __syncthreads();
    const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
      const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16,
                c = (ti % (V / N)) * N;
#pragma unroll
      for (int i = 0; i < acc::ne; ++i) {
        state[j].x[i] *= decay;
      }
      product<C, true, true>(state[j], s.k, s.delta, r, c);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < PER_WARP; ++j) {
    const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16,
              c = (ti % (V / N)) * N;
#pragma unroll
    for (int i = 0; i < acc::ne; ++i) {
      a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)] =
          state[j].x[i];
    }
  }
#else
  (void)a;
#endif
}

} // namespace

extern "C" int gdn_mma_prefill_init() {
  int device, available;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    return -(int)status;
  status = cudaDeviceGetAttribute(
      &available, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (status != cudaSuccess)
    return -(int)status;
  if (available < (int)sizeof(shared<CHUNK, D>))
    return 0;
  status = cudaFuncSetAttribute(gdn_single<CHUNK, D, 1, GdnMmaParams>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(shared<CHUNK, D>));
  if (status != cudaSuccess)
    return -(int)status;
  status = cudaFuncSetAttribute(gdn_single<CHUNK, D, 1, GdnMmaFusedParams>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                sizeof(shared<CHUNK, D>));
  return status == cudaSuccess ? 1 : -(int)status;
}

extern "C" int gdn_mma_fused_prefill(const GdnMmaFusedParams *params) {
  const auto stream = reinterpret_cast<cudaStream_t>(params->stream);
  gdn_single<CHUNK, D><<<params->batch_size * params->num_heads, dim3(32, 8),
                         sizeof(shared<CHUNK, D>), stream>>>(*params);
  return (int)cudaGetLastError();
}

extern "C" int gdn_mma_prefill(const GdnMmaParams *params) {
  const auto stream = reinterpret_cast<cudaStream_t>(params->stream);
  gdn_single<CHUNK, D><<<params->batch_size * params->num_heads, dim3(32, 8),
                         sizeof(shared<CHUNK, D>), stream>>>(*params);
  return (int)cudaGetLastError();
}
