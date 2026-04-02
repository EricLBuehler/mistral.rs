// Selective SSM (Mamba-2 style) Metal kernel for mistral.rs
// Ported from CUDA kernel in cuda/ssm.cu
//
// This kernel implements the selective scan operation:
//   state[t] = state[t-1] * exp(dt[t] * A) + dt[t] * B[t] * x[t]
//   y[t] = sum_d(state[t] * C[t]) + D * x[t]

#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32

static inline float simd_sum(float x) {
  // Use simd_shuffle to perform a warp-level reduction
  for (ushort offset = SIMD_SIZE / 2; offset > 0; offset >>= 1) {
    x += simd_shuffle_xor(x, offset);
  }
  return x;
}

// Mamba-2 SSM scan kernel (group-aware)
//
// Each simdgroup (32 threads) processes one (head, head_offset) pair across
// all timesteps. Lanes within the simdgroup process c_factor items each.
//
// Template parameter c_factor = items per lane = ceil(d_state / SIMD_SIZE)

template <int c_factor>
[[kernel]] void ssm_scan_kernel(
    const device float *x [[buffer(0)]], // (batch, seq_len, n_heads * head_dim)
    const device float *dt [[buffer(1)]], // (batch, seq_len, n_heads)
    const device float *A [[buffer(2)]],  // (n_heads,)
    const device float *B [[buffer(3)]],  // (batch, seq_len, n_heads * d_state)
    const device float *C [[buffer(4)]],  // (batch, seq_len, n_heads * d_state)
    const device float *D [[buffer(5)]],  // (n_heads,)
    const device float *dt_bias [[buffer(6)]], // (n_heads,)
    device float *state [[buffer(7)]], // (batch, n_heads, head_dim, d_state)
    device float *y [[buffer(8)]],     // (batch, seq_len, n_heads * head_dim)
    constant int &n_heads [[buffer(9)]], constant int &head_dim [[buffer(10)]],
    constant int &d_state [[buffer(11)]], constant int &seq_len [[buffer(12)]],
    constant float &dt_min [[buffer(13)]],
    constant float &dt_max [[buffer(14)]],
    uint2 tgpig [[threadgroup_position_in_grid]], // (warp_idx, batch_idx)
    uint lane [[thread_index_in_simdgroup]]) {
  const int batch_idx = tgpig.y;
  const int warp_idx = tgpig.x;

  const int head_idx = warp_idx / head_dim;
  const int head_off = warp_idx % head_dim;

  if (head_idx >= n_heads)
    return;

  const int d_inner = n_heads * head_dim;

  // Load A value for this head
  const float a_val = A[head_idx];

  // Load D value for this head
  const float d_val = D[head_idx];

  // Load dt_bias for this head
  const float dt_bias_val = dt_bias[head_idx];

  // Initialize state from input state tensor
  // state layout: (batch, n_heads, head_dim, d_state)
  const int state_base = batch_idx * n_heads * head_dim * d_state +
                         head_idx * head_dim * d_state + head_off * d_state;
  float reg_state[c_factor];
  for (int j = 0; j < c_factor; j++) {
    reg_state[j] = state[state_base + (int)lane * c_factor + j];
  }

  // Process each timestep
  for (int t = 0; t < seq_len; t++) {
    // Load dt and apply bias + softplus + clamp
    float dt_val = dt[batch_idx * seq_len * n_heads + t * n_heads + head_idx];
    dt_val += dt_bias_val;
    // softplus: log(1 + exp(x))
    dt_val = (dt_val > 20.0f) ? dt_val : log(1.0f + exp(dt_val));
    // clamp
    dt_val = min(max(dt_val, dt_min), dt_max);

    // dA = exp(dt * A)
    const float dA = exp(dt_val * a_val);

    // x value for this (head, head_offset) at timestep t
    const float x_val = x[batch_idx * seq_len * d_inner + t * d_inner +
                          head_idx * head_dim + head_off];
    const float x_dt = x_val * dt_val;

    // Compute state update and output dot product
    float state_sum = 0.0f;
    const int bc_base = batch_idx * seq_len * n_heads * d_state +
                        t * n_heads * d_state + head_idx * d_state;
    for (int j = 0; j < c_factor; j++) {
      const int state_off = (int)lane * c_factor + j;
      const float b_val = B[bc_base + state_off];
      const float c_val = C[bc_base + state_off];

      // state = state * dA + B * x_dt
      reg_state[j] = reg_state[j] * dA + b_val * x_dt;
      state_sum += reg_state[j] * c_val;
    }

    // Simd-reduce the output across lanes
    state_sum = simd_sum(state_sum);

    // Lane 0 writes the output (+ D skip connection)
    if (lane == 0) {
      y[batch_idx * seq_len * d_inner + t * d_inner + head_idx * head_dim +
        head_off] = state_sum + d_val * x_val;
    }
  }

  // Write final state back
  for (int j = 0; j < c_factor; j++) {
    state[state_base + (int)lane * c_factor + j] = reg_state[j];
  }
}

// Explicit instantiations for common d_state / SIMD_SIZE ratios
template [[host_name("ssm_scan_c1")]] [[kernel]]
void ssm_scan_kernel<1>(const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, device float *, device float *,
                        constant int &, constant int &, constant int &,
                        constant int &, constant float &, constant float &,
                        uint2, uint);

template [[host_name("ssm_scan_c2")]] [[kernel]]
void ssm_scan_kernel<2>(const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, device float *, device float *,
                        constant int &, constant int &, constant int &,
                        constant int &, constant float &, constant float &,
                        uint2, uint);

template [[host_name("ssm_scan_c4")]] [[kernel]]
void ssm_scan_kernel<4>(const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, device float *, device float *,
                        constant int &, constant int &, constant int &,
                        constant int &, constant float &, constant float &,
                        uint2, uint);

template [[host_name("ssm_scan_c8")]] [[kernel]]
void ssm_scan_kernel<8>(const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, const device float *,
                        const device float *, device float *, device float *,
                        constant int &, constant int &, constant int &,
                        constant int &, constant float &, constant float &,
                        uint2, uint);
