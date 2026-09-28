#include <metal_stdlib>

using namespace metal;

// Fused: out = (rms_norm(x, weight, eps) + residual) [* scale if provided]
// One threadgroup per row; threads cooperate on the sum-of-squares reduction.
template <typename T>
void rmsnorm_residual_impl(device const T *x, device const T *residual,
                           device const T *weight, device const T *scale,
                           device T *out, constant uint &n_cols,
                           constant float &eps, constant uint &has_scale,
                           uint tid [[thread_index_in_threadgroup]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tg_size [[threads_per_threadgroup]],
                           threadgroup float *shared) {
  const uint row_off = row * n_cols;

  float local_sum = 0.0f;
  for (uint i = tid; i < n_cols; i += tg_size) {
    float v = float(x[row_off + i]);
    local_sum += v * v;
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_rms = rsqrt(shared[0] / float(n_cols) + eps);
  const float scale_v = has_scale ? float(scale[0]) : 1.0f;

  for (uint i = tid; i < n_cols; i += tg_size) {
    float normed = float(x[row_off + i]) * inv_rms * float(weight[i]);
    float v = (normed + float(residual[row_off + i])) * scale_v;
    out[row_off + i] = T(v);
  }
}

#define INSTANTIATE_RMSNORM_RESIDUAL(NAME, T)                                  \
  kernel void NAME(                                                            \
      device const T *x [[buffer(0)]], device const T *residual [[buffer(1)]], \
      device const T *weight [[buffer(2)]],                                    \
      device const T *scale [[buffer(3)]], device T *out [[buffer(4)]],        \
      constant uint &n_cols [[buffer(5)]], constant float &eps [[buffer(6)]],  \
      constant uint &has_scale [[buffer(7)]],                                  \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint row [[threadgroup_position_in_grid]],                               \
      uint tg_size [[threads_per_threadgroup]],                                \
      threadgroup float *shared [[threadgroup(0)]]) {                          \
    rmsnorm_residual_impl<T>(x, residual, weight, scale, out, n_cols, eps,     \
                             has_scale, tid, row, tg_size, shared);            \
  }

INSTANTIATE_RMSNORM_RESIDUAL(rmsnorm_residual_f32, float)
INSTANTIATE_RMSNORM_RESIDUAL(rmsnorm_residual_f16, half)
INSTANTIATE_RMSNORM_RESIDUAL(rmsnorm_residual_bf16, bfloat)

// Fused pre-norm residual: sum = x + residual (rounded to T), out = rms_norm(sum, weight, eps).
// Follows candle's rmsnorm arithmetic (rounded sum, reduction order, 1 / sqrt); fast-math lowering can still
// leave half-precision outputs one ULP apart from the runtime-compiled candle kernel.
template <typename T>
void add_rmsnorm_impl(device const T *x, device const T *residual,
                      device const T *weight, device T *sum_out,
                      device T *norm_out, constant uint &n_cols,
                      constant float &eps,
                      uint tid [[thread_index_in_threadgroup]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tg_size [[threads_per_threadgroup]],
                      threadgroup float *shared) {
  const uint row_off = row * n_cols;

  float local_sum = 0.0f;
  for (uint i = tid; i < n_cols; i += tg_size) {
    const T s = T(float(x[row_off + i]) + float(residual[row_off + i]));
    sum_out[row_off + i] = s;
    local_sum = local_sum + float(s) * float(s);
  }
  shared[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  for (uint s = tg_size / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv_norm = 1.0f / sqrt(shared[0] / float(n_cols) + eps);
  for (uint i = tid; i < n_cols; i += tg_size) {
    float val = float(sum_out[row_off + i]) * inv_norm;
    val *= float(weight[i]);
    norm_out[row_off + i] = T(val);
  }
}

#define INSTANTIATE_ADD_RMSNORM(NAME, T)                                       \
  kernel void NAME(                                                            \
      device const T *x [[buffer(0)]], device const T *residual [[buffer(1)]], \
      device const T *weight [[buffer(2)]], device T *sum_out [[buffer(3)]],   \
      device T *norm_out [[buffer(4)]], constant uint &n_cols [[buffer(5)]],   \
      constant float &eps [[buffer(6)]],                                       \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint row [[threadgroup_position_in_grid]],                               \
      uint tg_size [[threads_per_threadgroup]],                                \
      threadgroup float *shared [[threadgroup(0)]]) {                          \
    add_rmsnorm_impl<T>(x, residual, weight, sum_out, norm_out, n_cols, eps,   \
                        tid, row, tg_size, shared);                            \
  }

INSTANTIATE_ADD_RMSNORM(add_rmsnorm_f32, float)
INSTANTIATE_ADD_RMSNORM(add_rmsnorm_f16, half)
INSTANTIATE_ADD_RMSNORM(add_rmsnorm_bf16, bfloat)

// Fused per-head RMSNorm + RoPE over q and k, plus an optional norm-only v, reading strided [B, H, S, D] inputs.
// Rows are q rows, then k rows, then v rows; row order follows the output layout (token- or heads-major) so writes
// are contiguous. The rotary part covers the first 2 * rot_dim elements (NeoX split-half or GPT-J interleaved); the
// cos/sin row is positions[b * S + s], or (cos_batch_stride ? b * cos_batch_stride : 0) + s without positions.
struct QkNormRopeParams {
  long q_stride[4];
  long k_stride[4];
  long v_stride[4];
  int batch;
  int q_heads;
  int k_heads;
  int v_heads;
  int seq_len;
  int head_dim;
  int rot_dim;
  float q_eps;
  float k_eps;
  float v_eps;
  int token_major;
  int use_positions;
  int cos_batch_stride;
};

template <typename T, bool NEOX>
void qk_norm_rope_impl(device const T *q, device const T *k, device const T *v,
                       device const T *q_weight, device const T *k_weight,
                       device const T *v_weight, device const T *cos,
                       device const T *sin, device const uint *positions,
                       device T *q_out, device T *k_out, device T *v_out,
                       constant QkNormRopeParams &p, uint row, uint tid,
                       uint tg_size, uint simd_lid, uint simd_gid,
                       threadgroup float *shared) {
  const int q_rows = p.batch * p.q_heads * p.seq_len;
  const int k_rows = p.batch * p.k_heads * p.seq_len;
  int local_row = int(row);
  int kind = 0;
  if (local_row >= q_rows) {
    local_row -= q_rows;
    kind = 1;
    if (local_row >= k_rows) {
      local_row -= k_rows;
      kind = 2;
    }
  }
  const int heads = kind == 0 ? p.q_heads : (kind == 1 ? p.k_heads : p.v_heads);
  const bool token_major = p.token_major != 0 && kind != 2;
  const int inner = token_major ? heads : p.seq_len;
  const int outer = token_major ? p.seq_len : heads;
  const int tmp = local_row / inner;
  const int seq = token_major ? tmp % outer : local_row % inner;
  const int head = token_major ? local_row % inner : tmp % outer;
  const int b = tmp / outer;

  device const T *src = kind == 0 ? q : (kind == 1 ? k : v);
  device const T *weight = kind == 0 ? q_weight : (kind == 1 ? k_weight : v_weight);
  constant long *stride = kind == 0 ? p.q_stride : (kind == 1 ? p.k_stride : p.v_stride);
  device T *dst = (kind == 0 ? q_out : (kind == 1 ? k_out : v_out)) + (long)local_row * p.head_dim;
  const float eps = kind == 0 ? p.q_eps : (kind == 1 ? p.k_eps : p.v_eps);
  const long base = (long)b * stride[0] + (long)head * stride[1] + (long)seq * stride[2];
  const long sd = stride[3];

  float sum = 0.0f;
  for (int col = int(tid); col < p.head_dim; col += int(tg_size)) {
    const float value = float(src[base + col * sd]);
    sum += value * value;
  }
  sum = simd_sum(sum);
  if (simd_lid == 0) {
    shared[simd_gid] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < (tg_size + 31) / 32; i++) {
    total += shared[i];
  }
  const float inv_rms = rsqrt(total / float(p.head_dim) + eps);

  int rotated = 0;
  if (kind != 2) {
    rotated = 2 * p.rot_dim;
    const long cos_row = p.use_positions
                             ? (long)positions[b * p.seq_len + seq]
                             : (long)(p.cos_batch_stride ? b * p.cos_batch_stride : 0) + seq;
    device const T *cos_row_ptr = cos + cos_row * p.rot_dim;
    device const T *sin_row_ptr = sin + cos_row * p.rot_dim;
    for (int r = int(tid); r < p.rot_dim; r += int(tg_size)) {
      const int x_idx = NEOX ? r : 2 * r;
      const int y_idx = NEOX ? p.rot_dim + r : 2 * r + 1;
      const float x = float(src[base + x_idx * sd]) * inv_rms * float(weight[x_idx]);
      const float y = float(src[base + y_idx * sd]) * inv_rms * float(weight[y_idx]);
      const float c = float(cos_row_ptr[r]);
      const float s = float(sin_row_ptr[r]);
      dst[x_idx] = T(x * c - y * s);
      dst[y_idx] = T(y * c + x * s);
    }
  }
  for (int col = rotated + int(tid); col < p.head_dim; col += int(tg_size)) {
    dst[col] = T(float(src[base + col * sd]) * inv_rms * float(weight[col]));
  }
}

#define INSTANTIATE_QK_NORM_ROPE(NAME, T, NEOX)                                \
  kernel void NAME(                                                            \
      device const T *q [[buffer(0)]], device const T *k [[buffer(1)]],        \
      device const T *v [[buffer(2)]], device const T *q_weight [[buffer(3)]], \
      device const T *k_weight [[buffer(4)]],                                  \
      device const T *v_weight [[buffer(5)]], device const T *cos [[buffer(6)]], \
      device const T *sin [[buffer(7)]],                                       \
      device const uint *positions [[buffer(8)]], device T *q_out [[buffer(9)]], \
      device T *k_out [[buffer(10)]], device T *v_out [[buffer(11)]],          \
      constant QkNormRopeParams &p [[buffer(12)]],                             \
      uint row [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint tg_size [[threads_per_threadgroup]],                                \
      uint simd_lid [[thread_index_in_simdgroup]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      threadgroup float *shared [[threadgroup(0)]]) {                          \
    qk_norm_rope_impl<T, NEOX>(q, k, v, q_weight, k_weight, v_weight, cos, sin, \
                               positions, q_out, k_out, v_out, p, row, tid,    \
                               tg_size, simd_lid, simd_gid, shared);           \
  }

INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_neox_f32, float, true)
INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_neox_f16, half, true)
INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_neox_bf16, bfloat, true)
INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_gptj_f32, float, false)
INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_gptj_f16, half, false)
INSTANTIATE_QK_NORM_ROPE(qk_norm_rope_gptj_bf16, bfloat, false)
