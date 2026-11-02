// GDN (Gated Delta Net) Metal kernels for mistral.rs
// Ported from CUDA kernels in cuda/gdn.cu

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#endif

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence
//
// V-tiled recurrence with compile-time K dimension for register residency.
// Grid: (ceil(V/BV), B*H), Block: (BV,). Each thread owns BK registers of
// state. Threadgroup memory holds k_buf and q_buf (2*BK floats).
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

template <int BK, int BV>
[[kernel]] void gated_delta_rule_kernel(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &seq_len [[buffer(7)]],
    constant int &v_dim [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int v_tile = tgpig.x;
  const int bh = tgpig.y;
  const int v_idx = v_tile * BV + (int)tid;

  if (v_idx >= v_dim)
    return;

  const device float *q_bh = q + bh * seq_len * BK;
  const device float *k_bh = k + bh * seq_len * BK;
  const device float *v_bh = v + bh * seq_len * v_dim;
  const device float *g_bh = g + bh * seq_len;
  const device float *beta_bh = beta + bh * seq_len;
  device float *state_bh = state + bh * BK * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  threadgroup float k_buf[BK];
  threadgroup float q_buf[BK];

  // Load state column into registers
  float s[BK];
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
    // Collaboratively load k_t into threadgroup memory
    for (int j = (int)tid; j < BK; j += BV) {
      k_buf[j] = k_bh[t * BK + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float decay = exp(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    // Fused pass 1: decay state + compute kv_mem
    float kv_mem = 0.0f;
    for (int j = 0; j < BK; j++) {
      s[j] *= decay;
      kv_mem = fma(s[j], k_buf[j], kv_mem);
    }

    float delta = (v_t - kv_mem) * beta_t;

    // Collaboratively load q_t into threadgroup memory
    for (int j = (int)tid; j < BK; j += BV) {
      q_buf[j] = q_bh[t * BK + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fused pass 2: update state + compute output
    float y_t = 0.0f;
    for (int j = 0; j < BK; j++) {
      s[j] = fma(k_buf[j], delta, s[j]);
      y_t = fma(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write state back
  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Fallback kernel: runtime k_dim, still V-tiled
template <int BV, int MAX_K>
[[kernel]] void gated_delta_rule_kernel_fallback(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &seq_len [[buffer(7)]],
    constant int &k_dim [[buffer(8)]], constant int &v_dim [[buffer(9)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int v_tile = tgpig.x;
  const int bh = tgpig.y;
  const int v_idx = v_tile * BV + (int)tid;

  if (v_idx >= v_dim)
    return;

  const device float *q_bh = q + bh * seq_len * k_dim;
  const device float *k_bh = k + bh * seq_len * k_dim;
  const device float *v_bh = v + bh * seq_len * v_dim;
  const device float *g_bh = g + bh * seq_len;
  const device float *beta_bh = beta + bh * seq_len;
  device float *state_bh = state + bh * k_dim * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  threadgroup float *k_buf = shared_mem;
  threadgroup float *q_buf = shared_mem + k_dim;

  float s[MAX_K];
  for (int j = 0; j < k_dim; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
    for (int j = (int)tid; j < k_dim; j += BV) {
      k_buf[j] = k_bh[t * k_dim + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float decay = exp(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] *= decay;
      kv_mem = fma(s[j], k_buf[j], kv_mem);
    }

    float delta = (v_t - kv_mem) * beta_t;

    for (int j = (int)tid; j < k_dim; j += BV) {
      q_buf[j] = q_bh[t * k_dim + j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float y_t = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] = fma(k_buf[j], delta, s[j]);
      y_t = fma(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (int j = 0; j < k_dim; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Explicit instantiations
template [[host_name("gated_delta_rule_128_64")]] [[kernel]]
void gated_delta_rule_kernel<128, 64>(
    const device float *, const device float *, const device float *,
    const device float *, const device float *, device float *, device float *,
    constant int &, constant int &, uint2, uint);

template [[host_name("gated_delta_rule_64_64")]] [[kernel]]
void gated_delta_rule_kernel<64, 64>(const device float *, const device float *,
                                     const device float *, const device float *,
                                     const device float *, device float *,
                                     device float *, constant int &,
                                     constant int &, uint2, uint);

template [[host_name("gated_delta_rule_fallback")]] [[kernel]]
void gated_delta_rule_kernel_fallback<64, 256>(
    const device float *, const device float *, const device float *,
    const device float *, const device float *, device float *, device float *,
    constant int &, constant int &, constant int &, threadgroup float *, uint2,
    uint);

// ============================================================================
// Kernel 1a: pooled single-token decode
//
// Same V-tiled math as Kernel 1 for seq_len == 1, but the state lives in the recurrent pool and row
// slots[bh / num_heads] is updated in place, so decode needs no gather/scatter of the state.
// q,k: [BH, K]  v: [BH, V]  g,beta: [BH]  state pool: [P, H, K, V]  output: [BH, V]
// ============================================================================

template <int BK, int BV>
[[kernel]] void gated_delta_rule_pooled_decode_kernel(
    const device float *q [[buffer(0)]], const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]], const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]], device float *state [[buffer(5)]],
    device float *output [[buffer(6)]], constant int &v_dim [[buffer(7)]],
    const device uint *slots [[buffer(8)]], constant int &num_heads [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int bh = tgpig.y;
  const int v_idx = tgpig.x * BV + (int)tid;

  threadgroup float k_buf[BK];
  threadgroup float q_buf[BK];
  for (int j = (int)tid; j < BK; j += BV) {
    k_buf[j] = k[bh * BK + j];
    q_buf[j] = q[bh * BK + j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (v_idx >= v_dim)
    return;

  const int b = bh / num_heads;
  const int h = bh % num_heads;
  device float *state_bh =
      state + ((long)slots[b] * num_heads + h) * BK * v_dim;

  const float decay = exp(g[bh]);
  float s[BK];
  float kv_mem = 0.0f;
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx] * decay;
    kv_mem = fma(s[j], k_buf[j], kv_mem);
  }
  const float delta = (v[bh * v_dim + v_idx] - kv_mem) * beta[bh];
  float y = 0.0f;
  for (int j = 0; j < BK; j++) {
    s[j] = fma(k_buf[j], delta, s[j]);
    y = fma(s[j], q_buf[j], y);
    state_bh[j * v_dim + v_idx] = s[j];
  }
  output[bh * v_dim + v_idx] = y;
}

#define instantiate_gdn_pooled_decode(bk, bv)                                  \
  template [[host_name("gated_delta_rule_pooled_decode_" #bk "_" #bv)]]       \
  [[kernel]] void gated_delta_rule_pooled_decode_kernel<bk, bv>(               \
      const device float *, const device float *, const device float *,        \
      const device float *, const device float *, device float *,              \
      device float *, constant int &, const device uint *, constant int &,     \
      uint2, uint);

instantiate_gdn_pooled_decode(128, 64)
instantiate_gdn_pooled_decode(64, 64)

// ============================================================================
// Gated RMSNorm on the recurrence output: out = x * rsqrt(mean(x^2) + eps) * w * silu(gate), per row of dim.
// One simdgroup per row; x, gate, out: [rows, dim]  weight: [dim]
// ============================================================================

template <typename T>
[[kernel]] void gdn_rmsnorm_gated_kernel(
    const device T *x [[buffer(0)]], const device T *gate [[buffer(1)]],
    const device T *weight [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const device T *xr = x + (long)row * dim;
  const device T *gr = gate + (long)row * dim;
  device T *orow = out + (long)row * dim;
  float sum_sq = 0.0f;
  for (int i = (int)lane; i < dim; i += 32) {
    const float v = static_cast<float>(xr[i]);
    sum_sq = fma(v, v, sum_sq);
  }
  const float inv_rms = rsqrt(simd_sum(sum_sq) / float(dim) + eps);
  for (int i = (int)lane; i < dim; i += 32) {
    const float g = static_cast<float>(gr[i]);
    const float silu = g / (1.0f + exp(-g));
    orow[i] = static_cast<T>(static_cast<float>(xr[i]) * inv_rms *
                             static_cast<float>(weight[i]) * silu);
  }
}

#define instantiate_gdn_rmsnorm_gated(type)                                    \
  template [[host_name("gdn_rmsnorm_gated_" #type)]] [[kernel]] void           \
  gdn_rmsnorm_gated_kernel<type>(const device type *, const device type *,     \
                                 const device type *, device type *,           \
                                 constant int &, constant float &, uint, uint);

instantiate_gdn_rmsnorm_gated(float)
instantiate_gdn_rmsnorm_gated(half)
instantiate_gdn_rmsnorm_gated(bfloat)

// ============================================================================
// Kernel 1c: GDN recurrence straight from the convolved projection (any token count)
//
// MLX-style layout: one simdgroup per (batch, v head, v column); each lane holds DK / 32 state entries and the key
// dot products reduce with simd_sum, so the grid has B * Hv * V * 32 threads. Per token it computes the q/k L2 norms,
// q scale, beta = sigmoid(b) and decay = exp(-exp(a_log) * softplus(a + dt_bias)) in-kernel. mixed: [B, T, 2 * key_dim + value_dim]
// b, a: [B, T, Hv] (strided)  state: [rows, Hv, K, V] f32 at row slots[b] (or b)  y: [B, T, Hv, V]
// ============================================================================

struct GdnPrefillParams {
  int seq_len;
  int num_k_heads;
  int num_v_heads;
  int head_v_dim;
  int tiled_v_heads;
  int use_slots;
  int b_stride[3];
  int a_stride[3];
};

template <typename T, int DK>
[[kernel]] void gdn_prefill_from_convolved_kernel(
    const device T *mixed [[buffer(0)]], const device T *b [[buffer(1)]],
    const device T *a [[buffer(2)]], const device float *a_log [[buffer(3)]],
    const device float *dt_bias [[buffer(4)]], device float *state [[buffer(5)]],
    device T *y [[buffer(6)]], const device uint *slots [[buffer(7)]],
    constant GdnPrefillParams &p [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int N = DK / 32;
  const int dv = gid.y;
  const int n = gid.z;
  if (dv >= p.head_v_dim)
    return;
  const int bidx = n / p.num_v_heads;
  const int hv = n - bidx * p.num_v_heads;
  const int hk = p.tiled_v_heads ? hv % p.num_k_heads
                                 : hv / (p.num_v_heads / p.num_k_heads);
  const int key_dim = p.num_k_heads * DK;
  const int conv_dim = 2 * key_dim + p.num_v_heads * p.head_v_dim;
  const long row = p.use_slots ? (long)slots[bidx] : (long)bidx;
  device float *state_col =
      state + (row * p.num_v_heads + hv) * DK * p.head_v_dim + dv;
  const float neg_a = -exp(a_log[hv]);
  const float bias = dt_bias[hv];
  const float q_scale = rsqrt(float(DK));

  float s[N];
  for (int i = 0; i < N; i++) {
    s[i] = state_col[(lane * N + i) * p.head_v_dim];
  }

  for (int t = 0; t < p.seq_len; t++) {
    const device T *tok = mixed + ((long)bidx * p.seq_len + t) * conv_dim;
    float q[N];
    float k[N];
    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (int i = 0; i < N; i++) {
      q[i] = float(tok[hk * DK + lane * N + i]);
      k[i] = float(tok[key_dim + hk * DK + lane * N + i]);
      q_ss = fma(q[i], q[i], q_ss);
      k_ss = fma(k[i], k[i], k_ss);
    }
    const float q_mul = rsqrt(simd_sum(q_ss) + 1.0e-6f) * q_scale;
    const float k_mul = rsqrt(simd_sum(k_ss) + 1.0e-6f);
    const float b_val = float(b[bidx * p.b_stride[0] + t * p.b_stride[1] + hv * p.b_stride[2]]);
    const float a_val =
        float(a[bidx * p.a_stride[0] + t * p.a_stride[1] + hv * p.a_stride[2]]) + bias;
    const float softplus_val =
        a_val > 20.0f ? a_val
                      : (a_val > 0.0f ? a_val + log(1.0f + exp(-a_val))
                                      : log(1.0f + exp(a_val)));
    const float beta_t = 1.0f / (1.0f + exp(-b_val));
    const float decay = exp(neg_a * softplus_val);

    float kv_mem = 0.0f;
    for (int i = 0; i < N; i++) {
      s[i] *= decay;
      kv_mem = fma(s[i], k[i] * k_mul, kv_mem);
    }
    kv_mem = simd_sum(kv_mem);
    const float v_t = float(tok[2 * key_dim + hv * p.head_v_dim + dv]);
    const float delta = (v_t - kv_mem) * beta_t;
    float out = 0.0f;
    for (int i = 0; i < N; i++) {
      s[i] = fma(k[i] * k_mul, delta, s[i]);
      out = fma(s[i], q[i] * q_mul, out);
    }
    out = simd_sum(out);
    if (lane == 0) {
      y[(((long)bidx * p.seq_len + t) * p.num_v_heads + hv) * p.head_v_dim + dv] = T(out);
    }
  }

  for (int i = 0; i < N; i++) {
    state_col[(lane * N + i) * p.head_v_dim] = s[i];
  }
}

#define instantiate_gdn_prefill_from_convolved(type, dk)                       \
  template [[host_name("gdn_prefill_from_convolved_" #type "_" #dk)]]         \
  [[kernel]] void gdn_prefill_from_convolved_kernel<type, dk>(                 \
      const device type *, const device type *, const device type *,           \
      const device float *, const device float *, device float *,              \
      device type *, const device uint *, constant GdnPrefillParams &, uint3,  \
      uint);

instantiate_gdn_prefill_from_convolved(
    float, 128) instantiate_gdn_prefill_from_convolved(float, 64)
    instantiate_gdn_prefill_from_convolved(half, 128)
        instantiate_gdn_prefill_from_convolved(half, 64)
            instantiate_gdn_prefill_from_convolved(bfloat, 128)
                instantiate_gdn_prefill_from_convolved(bfloat, 64)

                    template <typename T, int DK, int VC>
                    [[kernel]] void gdn_prefill_from_convolved_grouped_kernel(
                        const device T *mixed [[buffer(0)]],
                        const device T *b [[buffer(1)]],
                        const device T *a [[buffer(2)]],
                        const device float *a_log [[buffer(3)]],
                        const device float *dt_bias [[buffer(4)]],
                        device float *state [[buffer(5)]],
                        device T *y [[buffer(6)]],
                        const device uint *slots [[buffer(7)]],
                        constant GdnPrefillParams &p [[buffer(8)]],
                        uint3 gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constexpr int N = DK / 32;
  const int dv = gid.y * VC;
  const int n = gid.z;
  if (dv >= p.head_v_dim)
    return;
  const int bidx = n / p.num_v_heads;
  const int hv = n - bidx * p.num_v_heads;
  const int hk = p.tiled_v_heads ? hv % p.num_k_heads
                                 : hv / (p.num_v_heads / p.num_k_heads);
  const int key_dim = p.num_k_heads * DK;
  const int conv_dim = 2 * key_dim + p.num_v_heads * p.head_v_dim;
  const long row = p.use_slots ? (long)slots[bidx] : (long)bidx;
  device float *state_col =
      state + (row * p.num_v_heads + hv) * DK * p.head_v_dim + dv;
  const float neg_a = -exp(a_log[hv]);
  const float bias = dt_bias[hv];
  const float q_scale = rsqrt(float(DK));

  float s[VC][N];
#pragma unroll
  for (int v = 0; v < VC; v++) {
#pragma unroll
    for (int i = 0; i < N; i++) {
      s[v][i] = dv + v < p.head_v_dim
                    ? state_col[(lane * N + i) * p.head_v_dim + v]
                    : 0.0f;
    }
  }

  for (int t = 0; t < p.seq_len; t++) {
    const device T *tok = mixed + ((long)bidx * p.seq_len + t) * conv_dim;
    float q[N];
    float k[N];
    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (int i = 0; i < N; i++) {
      q[i] = float(tok[hk * DK + lane * N + i]);
      k[i] = float(tok[key_dim + hk * DK + lane * N + i]);
      q_ss = fma(q[i], q[i], q_ss);
      k_ss = fma(k[i], k[i], k_ss);
    }
    const float q_mul = rsqrt(simd_sum(q_ss) + 1.0e-6f) * q_scale;
    const float k_mul = rsqrt(simd_sum(k_ss) + 1.0e-6f);
    const float b_val =
        float(b[bidx * p.b_stride[0] + t * p.b_stride[1] + hv * p.b_stride[2]]);
    const float a_val =
        float(
            a[bidx * p.a_stride[0] + t * p.a_stride[1] + hv * p.a_stride[2]]) +
        bias;
    const float softplus_val =
        a_val > 20.0f ? a_val
                      : (a_val > 0.0f ? a_val + log(1.0f + exp(-a_val))
                                      : log(1.0f + exp(a_val)));
    const float beta_t = 1.0f / (1.0f + exp(-b_val));
    const float decay = exp(neg_a * softplus_val);

#pragma unroll
    for (int v = 0; v < VC; v++) {
      if (dv + v >= p.head_v_dim)
        continue;
      float kv_mem = 0.0f;
#pragma unroll
      for (int i = 0; i < N; i++) {
        s[v][i] *= decay;
        kv_mem = fma(s[v][i], k[i] * k_mul, kv_mem);
      }
      kv_mem = simd_sum(kv_mem);
      const float v_t = float(tok[2 * key_dim + hv * p.head_v_dim + dv + v]);
      const float delta = (v_t - kv_mem) * beta_t;
      float out = 0.0f;
#pragma unroll
      for (int i = 0; i < N; i++) {
        s[v][i] = fma(k[i] * k_mul, delta, s[v][i]);
        out = fma(s[v][i], q[i] * q_mul, out);
      }
      out = simd_sum(out);
      if (lane == 0) {
        y[(((long)bidx * p.seq_len + t) * p.num_v_heads + hv) * p.head_v_dim +
          dv + v] = T(out);
      }
    }
  }

#pragma unroll
  for (int v = 0; v < VC; v++) {
    if (dv + v >= p.head_v_dim)
      continue;
#pragma unroll
    for (int i = 0; i < N; i++) {
      state_col[(lane * N + i) * p.head_v_dim + v] = s[v][i];
    }
  }
}

#define instantiate_gdn_prefill_grouped(type, dk, vc)                          \
  template [[host_name("gdn_prefill_from_convolved_grouped_" #type "_" #dk     \
                       "_" #vc)]] [[kernel]] void                              \
  gdn_prefill_from_convolved_grouped_kernel<type, dk, vc>(                     \
      const device type *, const device type *, const device type *,           \
      const device float *, const device float *, device float *,              \
      device type *, const device uint *, constant GdnPrefillParams &, uint3,  \
      uint);

#define instantiate_gdn_prefill_grouped_columns(type, dk)                      \
  instantiate_gdn_prefill_grouped(type, dk, 4)

instantiate_gdn_prefill_grouped_columns(float, 128)
    instantiate_gdn_prefill_grouped_columns(float, 64)
        instantiate_gdn_prefill_grouped_columns(half, 128)
            instantiate_gdn_prefill_grouped_columns(half, 64)
                instantiate_gdn_prefill_grouped_columns(bfloat, 128)
                    instantiate_gdn_prefill_grouped_columns(bfloat, 64)

    // ============================================================================
    // Kernel 1b: chunked_gated_delta_rule_recurrence (prefill optimization)
    //
    // Processes prefill tokens in BT-token chunks instead of one at a time.
    // Same thread model as Kernel 1: one block per (v_tile, batch*head),
    // one thread per V-column. Each thread owns BK registers of state.
    //
    // Uses BT=32 on Metal to fit within 32KB threadgroup memory limit.
    //
    // q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
    // state: [BH, K, V] (in/out)  output: [BH, S, V]
    // ============================================================================

    template <int BT, int BK, int BV>
    [[kernel]] void chunked_gated_delta_rule_kernel(
        const device float *q [[buffer(0)]],
        const device float *k [[buffer(1)]],
        const device float *v [[buffer(2)]],
        const device float *g [[buffer(3)]],
        const device float *beta [[buffer(4)]],
        device float *state [[buffer(5)]], device float *output [[buffer(6)]],
        constant int &seq_len [[buffer(7)]], constant int &v_dim [[buffer(8)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
  const int v_tile = tgpig.x;
  const int bh = tgpig.y;
  const int v_idx = v_tile * BV + (int)tid;

  if (v_idx >= v_dim)
    return;

  const int num_chunks = (seq_len + BT - 1) / BT;

  const device float *q_bh = q + bh * seq_len * BK;
  const device float *k_bh = k + bh * seq_len * BK;
  const device float *v_bh = v + bh * seq_len * v_dim;
  const device float *g_bh = g + bh * seq_len;
  const device float *beta_bh = beta + bh * seq_len;
  device float *state_bh = state + bh * BK * v_dim;
  device float *out_bh = output + bh * seq_len * v_dim;

  // Threadgroup memory
  threadgroup float k_chunk[BT * BK];
  threadgroup float kk_dot[BT * BT];
  threadgroup float gcum[BT];
  threadgroup float beta_s[BT];
  threadgroup float q_buf[BK];

  // Load state column into registers
  float s[BK];
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  // Per-thread register array for corrected deltas
  float delta_arr[BT];

  for (int c = 0; c < num_chunks; c++) {
    const int chunk_start = c * BT;
    const int chunk_len = min(BT, seq_len - chunk_start);

    // === Phase 1: Cooperative load of k, beta, g into threadgroup memory ===
    for (int t = 0; t < chunk_len; t++) {
      for (int j = (int)tid; j < BK; j += BV) {
        k_chunk[t * BK + j] = k_bh[(chunk_start + t) * BK + j];
      }
    }
    if ((int)tid < chunk_len) {
      beta_s[tid] = beta_bh[chunk_start + tid];
      gcum[tid] = g_bh[chunk_start + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 1b: Parallel prefix sum of g (Hillis-Steele) ===
    for (int stride = 1; stride < BT; stride <<= 1) {
      float prev = 0.0f;
      if ((int)tid < chunk_len && (int)tid >= stride)
        prev = gcum[tid - stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if ((int)tid < chunk_len && (int)tid >= stride)
        gcum[tid] += prev;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Phase 2: Compute kk_dot[i][j] = dot(k[i], k[j]) for j < i ===
    for (int idx = (int)tid; idx < chunk_len * chunk_len; idx += BV) {
      int i = idx / chunk_len;
      int j = idx % chunk_len;
      if (j < i) {
        float dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          dot = fma(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
        }
        kk_dot[i * BT + j] = dot;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 3: Forward substitution (per V-column, in registers) ===
    for (int i = 0; i < chunk_len; i++) {
      float v_i = v_bh[(chunk_start + i) * v_dim + v_idx];
      float decay_i = exp(gcum[i]);
      float beta_i = beta_s[i];

      // Inter-chunk contribution: state @ k[i] with decay
      float kv_mem = 0.0f;
      for (int d = 0; d < BK; d++) {
        kv_mem = fma(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
      }

      float rhs = beta_i * (v_i - kv_mem);

      // Subtract lower-triangular contributions (intra-chunk)
      for (int j = 0; j < i; j++) {
        float a_ij = beta_i * kk_dot[i * BT + j] * exp(gcum[i] - gcum[j]);
        rhs -= a_ij * delta_arr[j];
      }
      delta_arr[i] = rhs;
    }

    // === Phase 4: Output computation (per V-column) ===
    for (int i = 0; i < chunk_len; i++) {
      // Cooperatively load q[i] into threadgroup memory
      for (int j = (int)tid; j < BK; j += BV) {
        q_buf[j] = q_bh[(chunk_start + i) * BK + j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float decay_i = exp(gcum[i]);

      // Inter-chunk contribution: q[i] @ (state * decay)
      float o_val = 0.0f;
      for (int d = 0; d < BK; d++) {
        o_val = fma(q_buf[d], s[d] * decay_i, o_val);
      }

      // Intra-chunk contribution
      for (int j = 0; j <= i; j++) {
        float qk_dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          qk_dot = fma(q_buf[d], k_chunk[j * BK + d], qk_dot);
        }
        o_val += qk_dot * delta_arr[j] * exp(gcum[i] - gcum[j]);
      }

      out_bh[(chunk_start + i) * v_dim + v_idx] = o_val;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Phase 5: State update for next chunk ===
    float g_total = gcum[chunk_len - 1];
    for (int d = 0; d < BK; d++) {
      float s_new = s[d] * exp(g_total);
      for (int t = 0; t < chunk_len; t++) {
        s_new += k_chunk[t * BK + d] * delta_arr[t] * exp(g_total - gcum[t]);
      }
      s[d] = s_new;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write final state back
  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Explicit instantiations for chunked kernel (BT=32 to fit 32KB threadgroup
// memory)
template [[host_name("chunked_gated_delta_rule_32_128_64")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 128, 64>(
    const device float *, const device float *, const device float *,
    const device float *, const device float *, device float *, device float *,
    constant int &, constant int &, uint2, uint);

template [[host_name("chunked_gated_delta_rule_32_64_64")]] [[kernel]]
void chunked_gated_delta_rule_kernel<32, 64, 64>(
    const device float *, const device float *, const device float *,
    const device float *, const device float *, device float *, device float *,
    constant int &, constant int &, uint2, uint);

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
// ============================================================================

template <typename T>
[[kernel]] void causal_conv1d_update_kernel(
    const device T *x [[buffer(0)]], const device T *weight [[buffer(1)]],
    device T *conv_state [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &batch_size [[buffer(4)]],
    constant int &conv_dim [[buffer(5)]],
    constant int &kernel_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const int b = gid.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  device T *cs = conv_state + (b * conv_dim + ch) * kernel_size;
  const device T *w = weight + ch * kernel_size;

  // Shift state left by 1
  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  cs[kernel_size - 1] = x[b * conv_dim + ch];

  // Dot product with weight
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    acc += (float)cs[i] * (float)w[i];
  }

  // SiLU activation
  float sig = 1.0f / (1.0f + exp(-acc));
  float result = acc * sig;

  output[b * conv_dim + ch] = (T)result;
}

#define instantiate_conv1d_update(type)                                        \
  template [[host_name("causal_conv1d_update_" #type)]] [[kernel]]             \
  void causal_conv1d_update_kernel<type>(                                      \
      const device type *, const device type *, device type *, device type *,  \
      constant int &, constant int &, constant int &, uint2);

instantiate_conv1d_update(half);
instantiate_conv1d_update(bfloat16_t);

// Pooled variant: sequence b's conv state is pool row slots[b], shifted in place.
template <typename T>
[[kernel]] void causal_conv1d_update_pooled_kernel(
    const device T *x [[buffer(0)]], const device T *weight [[buffer(1)]],
    device T *conv_state [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &batch_size [[buffer(4)]],
    constant int &conv_dim [[buffer(5)]],
    constant int &kernel_size [[buffer(6)]],
    const device uint *slots [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const int b = gid.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  device T *cs = conv_state + ((long)slots[b] * conv_dim + ch) * kernel_size;
  const device T *w = weight + ch * kernel_size;

  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  cs[kernel_size - 1] = x[b * conv_dim + ch];

  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    acc += (float)cs[i] * (float)w[i];
  }
  const float sig = 1.0f / (1.0f + exp(-acc));
  output[b * conv_dim + ch] = (T)(acc * sig);
}

#define instantiate_conv1d_update_pooled(type)                                 \
  template [[host_name("causal_conv1d_update_pooled_" #type)]] [[kernel]]      \
  void causal_conv1d_update_pooled_kernel<type>(                               \
      const device type *, const device type *, device type *, device type *,  \
      constant int &, constant int &, constant int &, const device uint *,     \
      uint2);

instantiate_conv1d_update_pooled(half);
instantiate_conv1d_update_pooled(bfloat16_t);

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
// ============================================================================

template <typename T, bool TOKEN_MAJOR>
[[kernel]] void causal_conv1d_full_kernel(
    const device T *x [[buffer(0)]], const device T *weight [[buffer(1)]],
    const device T *conv_state [[buffer(2)]], device T *output [[buffer(3)]],
    constant int &batch_size [[buffer(4)]],
    constant int &conv_dim [[buffer(5)]], constant int &seq_len [[buffer(6)]],
    constant int &kernel_size [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const int pos = gid.y;
  const int b = gid.z;

  if (ch >= conv_dim || pos >= seq_len || b >= batch_size)
    return;

  const long x_base = TOKEN_MAJOR ? (long)b * seq_len * conv_dim + ch
                                  : ((long)b * conv_dim + ch) * seq_len;
  const int x_stride = TOKEN_MAJOR ? conv_dim : 1;
  const device T *x_bch = x + x_base;
  const device T *w = weight + ch * kernel_size;
  const device T *cs = conv_state + (b * conv_dim + ch) * kernel_size;

  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    int src_pos = pos - (kernel_size - 1) + i;
    float x_val = src_pos >= 0 ? (float)x_bch[(long)src_pos * x_stride]
                               : (float)cs[kernel_size + src_pos];
    acc += x_val * (float)w[i];
  }

  // SiLU
  float sig = 1.0f / (1.0f + exp(-acc));
  float result = acc * sig;

  output[x_base + (long)pos * x_stride] = (T)result;
}

template <typename T, bool TOKEN_MAJOR>
[[kernel]] void save_conv_state_kernel(const device T *x [[buffer(0)]],
                                       const device T *conv_state_in
                                       [[buffer(1)]],
                                       device T *conv_state_out [[buffer(2)]],
                                       constant int &batch_size [[buffer(3)]],
                                       constant int &conv_dim [[buffer(4)]],
                                       constant int &seq_len [[buffer(5)]],
                                       constant int &kernel_size [[buffer(6)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  const int ch = gid.x;
  const int b = gid.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const long x_base = TOKEN_MAJOR ? (long)b * seq_len * conv_dim + ch
                                  : ((long)b * conv_dim + ch) * seq_len;
  const int x_stride = TOKEN_MAJOR ? conv_dim : 1;
  const device T *x_bch = x + x_base;
  const device T *prior = conv_state_in + (b * conv_dim + ch) * kernel_size;
  device T *cs = conv_state_out + (b * conv_dim + ch) * kernel_size;

  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = prior[i + seq_len];
    } else {
      cs[i] = x_bch[(long)(seq_len - kernel_size + i) * x_stride];
    }
  }
}

#define instantiate_conv1d_full(type, token_major, suffix)                     \
  template [[host_name("causal_conv1d_full" suffix "_" #type)]] [[kernel]]     \
  void causal_conv1d_full_kernel<type, token_major>(                           \
      const device type *, const device type *, const device type *,           \
      device type *, constant int &, constant int &, constant int &,           \
      constant int &, uint3);                                                  \
  template [[host_name("save_conv_state" suffix "_" #type)]] [[kernel]]        \
  void save_conv_state_kernel<type, token_major>(                              \
      const device type *, const device type *, device type *, constant int &, \
      constant int &, constant int &, constant int &, uint2);

instantiate_conv1d_full(half, false, "");
instantiate_conv1d_full(bfloat16_t, false, "");
instantiate_conv1d_full(half, true, "_token_major");
instantiate_conv1d_full(bfloat16_t, true, "_token_major");

// ============================================================================
// Kernel 3: fused_gdn_gating
//
// Fuses: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// ============================================================================

template <typename T>
[[kernel]] void fused_gdn_gating_kernel(
    const device T *b_in [[buffer(0)]], const device T *a_in [[buffer(1)]],
    const device float *a_log [[buffer(2)]],
    const device float *dt_bias [[buffer(3)]], device T *beta_out [[buffer(4)]],
    device T *g_out [[buffer(5)]], constant int &total_elements [[buffer(6)]],
    constant int &num_heads [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  const int idx = gid;
  if (idx >= total_elements)
    return;

  int head_idx = idx % num_heads;

  // beta = sigmoid(b)
  float b_val = (float)b_in[idx];
  float beta_val = 1.0f / (1.0f + exp(-b_val));

  // g = -exp(a_log) * softplus(a + dt_bias)
  float a_val = (float)a_in[idx];
  float a_log_val = a_log[head_idx];
  float dt_bias_val = dt_bias[head_idx];

  float sp_input = a_val + dt_bias_val;
  float softplus_val = log(1.0f + exp(sp_input));
  float g_val = -exp(a_log_val) * softplus_val;

  beta_out[idx] = (T)beta_val;
  g_out[idx] = (T)g_val;
}

#define instantiate_gdn_gating(type)                                           \
  template [[host_name("fused_gdn_gating_" #type)]] [[kernel]]                 \
  void fused_gdn_gating_kernel<type>(                                          \
      const device type *, const device type *, const device float *,          \
      const device float *, device type *, device type *, constant int &,      \
      constant int &, uint);

instantiate_gdn_gating(half);
instantiate_gdn_gating(bfloat16_t);
