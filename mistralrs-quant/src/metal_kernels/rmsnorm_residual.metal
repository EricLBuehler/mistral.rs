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
