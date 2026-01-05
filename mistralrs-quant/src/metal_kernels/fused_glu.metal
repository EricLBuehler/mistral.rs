#include <metal_stdlib>
#include "utils.metal"
using namespace metal;

// Activation types matching Rust GluActivationType enum
// Silu = 0, Gelu = 1, Relu = 2
constant int GLU_SILU = 0;
constant int GLU_GELU = 1;
constant int GLU_RELU = 2;

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
template <typename T>
inline T glu_silu(T x) {
    return x / (T(1.0f) + exp(-x));
}

// GELU activation (tanh approximation)
template <typename T>
inline T glu_gelu(T x) {
    const T SQRT_2_OVER_PI = T(0.7978845608f);
    const T COEFF = T(0.044715f);
    T x3 = x * x * x;
    T inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    return T(0.5f) * x * (T(1.0f) + tanh(inner));
}

// ReLU activation: max(0, x)
template <typename T>
inline T glu_relu(T x) {
    return max(x, T(0.0f));
}

// Apply activation based on type
template <typename T>
inline T apply_activation(T x, int activation) {
    switch (activation) {
        case GLU_SILU:
            return glu_silu(x);
        case GLU_GELU:
            return glu_gelu(x);
        case GLU_RELU:
            return glu_relu(x);
        default:
            return x;
    }
}

// Fused GLU kernel: output = activation(a) * b
template <typename T>
[[kernel]] void fused_glu(const device T *a [[buffer(0)]],
                          const device T *b [[buffer(1)]],
                          device T *output [[buffer(2)]],
                          constant uint &n_elements [[buffer(3)]],
                          constant int &activation [[buffer(4)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid < n_elements) {
        T a_val = a[tid];
        T b_val = b[tid];
        output[tid] = apply_activation(a_val, activation) * b_val;
    }
}

#define instantiate_fused_glu(type)                                            \
  template [[host_name("fused_glu_" #type)]] [[kernel]] void                   \
  fused_glu<type>(const device type *a [[buffer(0)]],                          \
                  const device type *b [[buffer(1)]],                          \
                  device type *output [[buffer(2)]],                           \
                  constant uint &n_elements [[buffer(3)]],                     \
                  constant int &activation [[buffer(4)]],                      \
                  uint tid [[thread_position_in_grid]]);

instantiate_fused_glu(float);
instantiate_fused_glu(half);
#if __METAL_VERSION__ >= 310
instantiate_fused_glu(bfloat);
#endif
