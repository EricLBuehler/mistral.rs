// GDN (Gated Delta Net) Metal kernels for mistral.rs
// Ported from CUDA kernels in cuda/gdn.cu

#include <metal_stdlib>
using namespace metal;

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
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant int &seq_len [[buffer(7)]],
    constant int &v_dim [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const int v_tile = tgpig.x;
    const int bh = tgpig.y;
    const int v_idx = v_tile * BV + (int)tid;

    if (v_idx >= v_dim) return;

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
    const device float *q [[buffer(0)]],
    const device float *k [[buffer(1)]],
    const device float *v [[buffer(2)]],
    const device float *g [[buffer(3)]],
    const device float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant int &seq_len [[buffer(7)]],
    constant int &k_dim [[buffer(8)]],
    constant int &v_dim [[buffer(9)]],
    threadgroup float *shared_mem [[threadgroup(0)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const int v_tile = tgpig.x;
    const int bh = tgpig.y;
    const int v_idx = v_tile * BV + (int)tid;

    if (v_idx >= v_dim) return;

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
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant int&, constant int&,
    uint2, uint);

template [[host_name("gated_delta_rule_64_64")]] [[kernel]]
void gated_delta_rule_kernel<64, 64>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant int&, constant int&,
    uint2, uint);

template [[host_name("gated_delta_rule_fallback")]] [[kernel]]
void gated_delta_rule_kernel_fallback<64, 256>(
    const device float*, const device float*, const device float*,
    const device float*, const device float*,
    device float*, device float*,
    constant int&, constant int&, constant int&,
    threadgroup float*,
    uint2, uint);

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
// ============================================================================

template <typename T>
[[kernel]] void causal_conv1d_update_kernel(
    const device T *x [[buffer(0)]],
    const device T *weight [[buffer(1)]],
    device T *conv_state [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant int &batch_size [[buffer(4)]],
    constant int &conv_dim [[buffer(5)]],
    constant int &kernel_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int ch = gid.x;
    const int b = gid.y;

    if (ch >= conv_dim || b >= batch_size) return;

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

#define instantiate_conv1d_update(type) \
    template [[host_name("causal_conv1d_update_" #type)]] [[kernel]] \
    void causal_conv1d_update_kernel<type>( \
        const device type*, const device type*, device type*, device type*, \
        constant int&, constant int&, constant int&, uint2);

instantiate_conv1d_update(half);
instantiate_conv1d_update(bfloat16_t);

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
// ============================================================================

template <typename T>
[[kernel]] void causal_conv1d_full_kernel(
    const device T *x [[buffer(0)]],
    const device T *weight [[buffer(1)]],
    device T *output [[buffer(2)]],
    constant int &batch_size [[buffer(3)]],
    constant int &conv_dim [[buffer(4)]],
    constant int &seq_len [[buffer(5)]],
    constant int &kernel_size [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    const int ch = gid.x;
    const int pos = gid.y;
    const int b = gid.z;

    if (ch >= conv_dim || pos >= seq_len || b >= batch_size) return;

    const device T *x_bch = x + (b * conv_dim + ch) * seq_len;
    const device T *w = weight + ch * kernel_size;

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int src_pos = pos - (kernel_size - 1) + i;
        float x_val = (src_pos >= 0) ? (float)x_bch[src_pos] : 0.0f;
        acc += x_val * (float)w[i];
    }

    // SiLU
    float sig = 1.0f / (1.0f + exp(-acc));
    float result = acc * sig;

    output[(b * conv_dim + ch) * seq_len + pos] = (T)result;
}

template <typename T>
[[kernel]] void save_conv_state_kernel(
    const device T *x [[buffer(0)]],
    device T *conv_state_out [[buffer(1)]],
    constant int &batch_size [[buffer(2)]],
    constant int &conv_dim [[buffer(3)]],
    constant int &seq_len [[buffer(4)]],
    constant int &kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int ch = gid.x;
    const int b = gid.y;

    if (ch >= conv_dim || b >= batch_size) return;

    const device T *x_bch = x + (b * conv_dim + ch) * seq_len;
    device T *cs = conv_state_out + (b * conv_dim + ch) * kernel_size;

    int pad = kernel_size - seq_len;
    for (int i = 0; i < kernel_size; i++) {
        if (i < pad) {
            cs[i] = (T)0.0f;
        } else {
            cs[i] = x_bch[seq_len - kernel_size + i];
        }
    }
}

#define instantiate_conv1d_full(type) \
    template [[host_name("causal_conv1d_full_" #type)]] [[kernel]] \
    void causal_conv1d_full_kernel<type>( \
        const device type*, const device type*, device type*, \
        constant int&, constant int&, constant int&, constant int&, uint3); \
    template [[host_name("save_conv_state_" #type)]] [[kernel]] \
    void save_conv_state_kernel<type>( \
        const device type*, device type*, \
        constant int&, constant int&, constant int&, constant int&, uint2);

instantiate_conv1d_full(half);
instantiate_conv1d_full(bfloat16_t);

// ============================================================================
// Kernel 3: fused_gdn_gating
//
// Fuses: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// ============================================================================

template <typename T>
[[kernel]] void fused_gdn_gating_kernel(
    const device T *b_in [[buffer(0)]],
    const device T *a_in [[buffer(1)]],
    const device float *a_log [[buffer(2)]],
    const device float *dt_bias [[buffer(3)]],
    device T *beta_out [[buffer(4)]],
    device T *g_out [[buffer(5)]],
    constant int &total_elements [[buffer(6)]],
    constant int &num_heads [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const int idx = gid;
    if (idx >= total_elements) return;

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

#define instantiate_gdn_gating(type) \
    template [[host_name("fused_gdn_gating_" #type)]] [[kernel]] \
    void fused_gdn_gating_kernel<type>( \
        const device type*, const device type*, \
        const device float*, const device float*, \
        device type*, device type*, \
        constant int&, constant int&, uint);

instantiate_gdn_gating(half);
instantiate_gdn_gating(bfloat16_t);
