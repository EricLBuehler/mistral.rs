// Selective SSM (Mamba-2 style) CUDA kernel for mistral.rs
// Adapted from llama.cpp's ssm-scan implementation.
//
// This kernel implements the selective scan operation:
//   state[t] = state[t-1] * exp(dt[t] * A) + dt[t] * B[t] * x[t]
//   y[t] = sum_d(state[t] * C[t]) + D * x[t]
//
// Supports Mamba-2 style: multiple heads, head_dim, n_groups, d_state.

#include <cstdint>
#include <cmath>

#define WARP_SIZE 32

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
    }
    return x;
}

// Mamba-2 SSM scan kernel (group-aware)
//
// Each warp processes one (head, head_offset) pair across all timesteps.
// Lanes within the warp process d_state / WARP_SIZE items each.
//
// Inputs:
//   x:     (batch, seq_len, n_heads, head_dim) - input after conv1d
//   dt:    (batch, seq_len, n_heads)            - timestep (pre-softplus, pre-bias)
//   A:     (n_heads,)                           - state matrix (A_log, negated exp)
//   B:     (batch, seq_len, n_heads, d_state)   - input projection (expanded from groups)
//   C:     (batch, seq_len, n_heads, d_state)   - output projection (expanded from groups)
//   D:     (n_heads,)                           - skip connection weight
//   dt_bias: (n_heads,)                         - bias for dt
//   state: (batch, n_heads, head_dim, d_state)  - SSM state (read/write)
//
// Output:
//   y:     (batch, seq_len, n_heads, head_dim)  - output
//
// Template parameter c_factor = d_state / WARP_SIZE = items per lane

template <int c_factor>
__global__ void __launch_bounds__(WARP_SIZE, 1)
ssm_scan_kernel(
    const float * __restrict__ x,       // (batch, seq_len, n_heads * head_dim)
    const float * __restrict__ dt,      // (batch, seq_len, n_heads)
    const float * __restrict__ A,       // (n_heads,)
    const float * __restrict__ B,       // (batch, seq_len, n_heads * d_state)
    const float * __restrict__ C,       // (batch, seq_len, n_heads * d_state)
    const float * __restrict__ D,       // (n_heads,)
    const float * __restrict__ dt_bias, // (n_heads,)
    float * __restrict__ state,         // (batch, n_heads, head_dim, d_state)
    float * __restrict__ y,             // (batch, seq_len, n_heads * head_dim)
    int n_heads,
    int head_dim,
    int d_state,
    int seq_len,
    float dt_min,
    float dt_max
) {
    // Grid: (num_warps_per_batch, batch_size)
    // Each warp handles one (head, head_offset) pair
    const int batch_idx = blockIdx.y;
    const int warp_idx  = blockIdx.x;
    const int lane      = threadIdx.x;

    const int head_idx  = warp_idx / head_dim;
    const int head_off  = warp_idx % head_dim;

    if (head_idx >= n_heads) return;

    const int d_inner = n_heads * head_dim;

    // Load A value for this head
    const float a_val = A[head_idx];

    // Load D value for this head
    const float d_val = D[head_idx];

    // Load dt_bias for this head
    const float dt_bias_val = dt_bias[head_idx];

    // Initialize state from input state tensor
    // state layout: (batch, n_heads, head_dim, d_state)
    const int state_base = batch_idx * n_heads * head_dim * d_state
                         + head_idx * head_dim * d_state
                         + head_off * d_state;
    float reg_state[c_factor];
    #pragma unroll
    for (int j = 0; j < c_factor; j++) {
        reg_state[j] = state[state_base + lane * c_factor + j];
    }

    // Process each timestep
    for (int t = 0; t < seq_len; t++) {
        // Load dt and apply bias + softplus + clamp
        float dt_val = dt[batch_idx * seq_len * n_heads + t * n_heads + head_idx];
        dt_val += dt_bias_val;
        // softplus: log(1 + exp(x))
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        // clamp
        dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);

        // dA = exp(dt * A)
        const float dA = expf(dt_val * a_val);

        // x value for this (head, head_offset) at timestep t
        const float x_val = x[batch_idx * seq_len * d_inner + t * d_inner + head_idx * head_dim + head_off];
        const float x_dt = x_val * dt_val;

        // Compute state update and output dot product
        float state_sum = 0.0f;
        const int bc_base = batch_idx * seq_len * n_heads * d_state
                          + t * n_heads * d_state
                          + head_idx * d_state;
        #pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const int state_off = lane * c_factor + j;
            const float b_val = B[bc_base + state_off];
            const float c_val = C[bc_base + state_off];

            // state = state * dA + B * x_dt
            reg_state[j] = reg_state[j] * dA + b_val * x_dt;
            state_sum += reg_state[j] * c_val;
        }

        // Warp-reduce the output across lanes
        state_sum = warp_reduce_sum(state_sum);

        // Lane 0 writes the output (+ D skip connection)
        if (lane == 0) {
            y[batch_idx * seq_len * d_inner + t * d_inner + head_idx * head_dim + head_off]
                = state_sum + d_val * x_val;
        }
    }

    // Write final state back
    #pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[state_base + lane * c_factor + j] = reg_state[j];
    }
}

// Extern C wrapper
extern "C" void selective_scan_cuda(
    const float *x,
    const float *dt,
    const float *A,
    const float *B,
    const float *C,
    const float *D,
    const float *dt_bias,
    float *state,
    float *y,
    int batch_size,
    int n_heads,
    int head_dim,
    int d_state,
    int seq_len,
    float dt_min,
    float dt_max,
    int64_t stream
) {
    const cudaStream_t custream = (cudaStream_t)stream;
    const int n_warps = n_heads * head_dim;

    dim3 grid(n_warps, batch_size);
    dim3 block(WARP_SIZE);

    // Dispatch based on c_factor = d_state / WARP_SIZE
    // Common values: d_state=16 (c_factor=1 with padding), 64, 128, 256
    const int c_factor = (d_state + WARP_SIZE - 1) / WARP_SIZE;

    if (c_factor == 1) {
        ssm_scan_kernel<1><<<grid, block, 0, custream>>>(
            x, dt, A, B, C, D, dt_bias, state, y,
            n_heads, head_dim, d_state, seq_len, dt_min, dt_max);
    } else if (c_factor == 2) {
        ssm_scan_kernel<2><<<grid, block, 0, custream>>>(
            x, dt, A, B, C, D, dt_bias, state, y,
            n_heads, head_dim, d_state, seq_len, dt_min, dt_max);
    } else if (c_factor == 4) {
        ssm_scan_kernel<4><<<grid, block, 0, custream>>>(
            x, dt, A, B, C, D, dt_bias, state, y,
            n_heads, head_dim, d_state, seq_len, dt_min, dt_max);
    } else if (c_factor == 8) {
        ssm_scan_kernel<8><<<grid, block, 0, custream>>>(
            x, dt, A, B, C, D, dt_bias, state, y,
            n_heads, head_dim, d_state, seq_len, dt_min, dt_max);
    } else {
        // Fallback: use c_factor=8 as max
        // This covers d_state up to 256
        ssm_scan_kernel<8><<<grid, block, 0, custream>>>(
            x, dt, A, B, C, D, dt_bias, state, y,
            n_heads, head_dim, d_state, seq_len, dt_min, dt_max);
    }
}
