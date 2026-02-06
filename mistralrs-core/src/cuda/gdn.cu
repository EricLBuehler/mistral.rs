#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence
//
// Fuses the entire sequential recurrence loop into a single kernel.
// One block per (batch, head) pair, V_DIM threads per block.
// Each thread owns one column of the state matrix.
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

__global__ void gated_delta_rule_recurrence_kernel(
    const float *__restrict__ q,     // [BH, S, K]
    const float *__restrict__ k,     // [BH, S, K]
    const float *__restrict__ v,     // [BH, S, V]
    const float *__restrict__ g,     // [BH, S]
    const float *__restrict__ beta,  // [BH, S]
    float *__restrict__ state,       // [BH, K, V]
    float *__restrict__ output,      // [BH, S, V]
    int seq_len, int k_dim, int v_dim) {

  const int bh = blockIdx.x; // batch*head index
  const int tid = threadIdx.x; // v_dim index

  if (tid >= v_dim)
    return;

  // Pointers for this (batch, head)
  const float *q_bh = q + bh * seq_len * k_dim;
  const float *k_bh = k + bh * seq_len * k_dim;
  const float *v_bh = v + bh * seq_len * v_dim;
  const float *g_bh = g + bh * seq_len;
  const float *beta_bh = beta + bh * seq_len;
  float *state_bh = state + bh * k_dim * v_dim;
  float *out_bh = output + bh * seq_len * v_dim;

  // Shared memory for k_buf and q_buf (loaded collaboratively)
  extern __shared__ float shared[];
  float *k_buf = shared;            // [k_dim]
  float *q_buf = shared + k_dim;    // [k_dim]

  // Load state column into registers
  // Each thread handles state[j][tid] for all j in [0, k_dim)
  // We use register array -- k_dim is typically 128, fits in registers
  // For very large k_dim, we'd need a different approach, but 128 is fine
  float s[128]; // max supported k_dim
  for (int j = 0; j < k_dim; j++) {
    s[j] = state_bh[j * v_dim + tid];
  }

  for (int t = 0; t < seq_len; t++) {
    // Load g_t and compute decay
    float g_t = g_bh[t];
    float decay = expf(g_t);

    // Apply decay to state column
    for (int j = 0; j < k_dim; j++) {
      s[j] *= decay;
    }

    // Collaboratively load k_t into shared memory
    for (int j = tid; j < k_dim; j += v_dim) {
      k_buf[j] = k_bh[t * k_dim + j];
    }
    __syncthreads();

    // Load beta_t and v_t for this thread
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + tid];

    // Compute kv_mem = sum_j(state[j][tid] * k_buf[j])
    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      kv_mem += s[j] * k_buf[j];
    }

    // Compute delta = (v_t - kv_mem) * beta_t
    float delta = (v_t - kv_mem) * beta_t;

    // Update state: state[j][tid] += k_buf[j] * delta
    for (int j = 0; j < k_dim; j++) {
      s[j] += k_buf[j] * delta;
    }

    // Collaboratively load q_t into shared memory (with scale already applied)
    for (int j = tid; j < k_dim; j += v_dim) {
      q_buf[j] = q_bh[t * k_dim + j];
    }
    __syncthreads();

    // Compute y_t = sum_j(state[j][tid] * q_buf[j])
    float y_t = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      y_t += s[j] * q_buf[j];
    }

    // Write output
    out_bh[t * v_dim + tid] = y_t;

    __syncthreads(); // Ensure shared memory is free for next iteration
  }

  // Write state back
  for (int j = 0; j < k_dim; j++) {
    state_bh[j * v_dim + tid] = s[j];
  }
}

extern "C" void gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, float *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, int64_t stream) {

  const cudaStream_t custream = (cudaStream_t)stream;
  // One block per (batch, head), v_dim threads per block
  dim3 grid(bh);
  dim3 block(v_dim);
  // Shared memory: k_buf[k_dim] + q_buf[k_dim]
  size_t smem = 2 * k_dim * sizeof(float);

  gated_delta_rule_recurrence_kernel<<<grid, block, smem, custream>>>(
      q, k, v, g, beta, state, output, seq_len, k_dim, v_dim);
}

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
//
// Each thread handles one channel: shift conv_state left by 1,
// insert new value, dot product with weight, apply SiLU.
//
// x: [B, conv_dim, 1]  weight: [conv_dim, kernel_size]
// conv_state: [B, conv_dim, kernel_size] (in/out)
// output: [B, conv_dim, 1]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_update_kernel(
    const T *__restrict__ x,        // [B, conv_dim, 1]
    const T *__restrict__ weight,   // [conv_dim, kernel_size]
    T *__restrict__ conv_state,     // [B, conv_dim, kernel_size]
    T *__restrict__ output,         // [B, conv_dim, 1]
    int batch_size, int conv_dim, int kernel_size) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  // Pointer to this batch/channel's conv state
  T *cs = conv_state + (b * conv_dim + ch) * kernel_size;
  const T *w = weight + ch * kernel_size;

  // Shift state left by 1
  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  // Insert new value
  cs[kernel_size - 1] = x[b * conv_dim + ch];

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

extern "C" void causal_conv1d_update(const void *x, const void *weight,
                                     void *conv_state, void *output,
                                     int batch_size, int conv_dim,
                                     int kernel_size, int dtype,
                                     int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(256);
  dim3 grid((conv_dim + 255) / 256, batch_size);

  if (dtype == 0) {
    // f16
    causal_conv1d_update_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight, (__half *)conv_state,
        (__half *)output, batch_size, conv_dim, kernel_size);
  } else {
    // bf16
    causal_conv1d_update_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
        conv_dim, kernel_size);
  }
}

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
//
// Each thread handles one (channel, position): causal window with
// zero-padding, dot product with weight, SiLU.
// A second pass writes the conv_state from the last kernel_size positions.
//
// x: [B, conv_dim, S]  weight: [conv_dim, kernel_size]
// conv_state_out: [B, conv_dim, kernel_size]  output: [B, conv_dim, S]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_full_kernel(
    const T *__restrict__ x,       // [B, conv_dim, S]
    const T *__restrict__ weight,  // [conv_dim, kernel_size]
    T *__restrict__ output,        // [B, conv_dim, S]
    int batch_size, int conv_dim, int seq_len, int kernel_size) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int pos = blockIdx.y;
  const int b = blockIdx.z;

  if (ch >= conv_dim || pos >= seq_len || b >= batch_size)
    return;

  const T *x_bch = x + (b * conv_dim + ch) * seq_len;
  const T *w = weight + ch * kernel_size;

  // Causal convolution: sum over kernel_size window ending at pos
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    int src_pos = pos - (kernel_size - 1) + i;
    float x_val = (src_pos >= 0) ? (float)x_bch[src_pos] : 0.0f;
    acc += x_val * (float)w[i];
  }

  // SiLU
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[(b * conv_dim + ch) * seq_len + pos] = (T)result;
}

template <typename T>
__global__ void save_conv_state_kernel(
    const T *__restrict__ x,           // [B, conv_dim, S]
    T *__restrict__ conv_state_out,    // [B, conv_dim, kernel_size]
    int batch_size, int conv_dim, int seq_len, int kernel_size) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const T *x_bch = x + (b * conv_dim + ch) * seq_len;
  T *cs = conv_state_out + (b * conv_dim + ch) * kernel_size;

  // Save last kernel_size positions (zero-pad if seq_len < kernel_size)
  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = (T)0.0f;
    } else {
      cs[i] = x_bch[seq_len - kernel_size + i];
    }
  }
}

extern "C" void causal_conv1d_full(const void *x, const void *weight,
                                   void *conv_state_out, void *output,
                                   int batch_size, int conv_dim, int seq_len,
                                   int kernel_size, int dtype,
                                   int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  // Main convolution kernel
  dim3 block(256);
  dim3 grid((conv_dim + 255) / 256, seq_len, batch_size);

  if (dtype == 0) {
    causal_conv1d_full_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight, (__half *)output,
        batch_size, conv_dim, seq_len, kernel_size);
    // Save conv state
    dim3 grid2((conv_dim + 255) / 256, batch_size);
    save_conv_state_kernel<__half><<<grid2, block, 0, custream>>>(
        (const __half *)x, (__half *)conv_state_out, batch_size, conv_dim,
        seq_len, kernel_size);
  } else {
    causal_conv1d_full_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)output, batch_size, conv_dim, seq_len, kernel_size);
    dim3 grid2((conv_dim + 255) / 256, batch_size);
    save_conv_state_kernel<__nv_bfloat16><<<grid2, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (__nv_bfloat16 *)conv_state_out, batch_size,
        conv_dim, seq_len, kernel_size);
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
__global__ void fused_gdn_gating_kernel(
    const T *__restrict__ b,         // [total]
    const T *__restrict__ a,         // [total]
    const float *__restrict__ a_log, // [num_heads]
    const float *__restrict__ dt_bias, // [num_heads]
    T *__restrict__ beta_out,        // [total]
    T *__restrict__ g_out,           // [total]
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
