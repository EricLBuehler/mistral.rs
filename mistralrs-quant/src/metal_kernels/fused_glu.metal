#include "utils.metal"
#include <metal_stdlib>
using namespace metal;

// Activation types matching Rust GluActivationType enum
// Silu = 0, Gelu = 1, Relu = 2, GeluErf = 3
constant int GLU_SILU = 0;
constant int GLU_GELU = 1;
constant int GLU_RELU = 2;
constant int GLU_GELU_ERF = 3;

// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
inline float glu_silu(float x) { return x / (1.0f + exp(-x)); }

// GELU activation (tanh approximation), matching candle's unary.metal gelu<T>
// exactly
inline float glu_gelu(float x) {
  if (x > 5) {
    return x;
  }
  float x_sq = x * x;
  float x_cube = x_sq * x;
  float alpha = x + static_cast<float>(0.044715) * x_cube;
  float beta = (static_cast<float>(M_2_SQRTPI_F * M_SQRT1_2_F) * alpha);
  return static_cast<float>(0.5) * x *
         (static_cast<float>(1.0) + float(precise::tanh(beta)));
}

// ReLU activation: max(0, x)
inline float glu_relu(float x) { return max(x, 0.0f); }

// erf implementation matching candle's unary.metal erf<T> (Abramowitz &
// Stegun 7.1.26)
inline float glu_erf(float in) {
  float x = in;
  float a1 = 0.254829592;
  float a2 = -0.284496736;
  float a3 = 1.421413741;
  float a4 = -1.453152027;
  float a5 = 1.061405429;
  float p = 0.3275911;
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x);
  float t = 1.0 / (1.0 + p * x);
  float y =
      1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return float(sign * y);
}

// GELU (exact ERF version), matching candle's unary.metal gelu_erf<T> exactly
inline float glu_gelu_erf(float x) {
  return float(x * (1 + glu_erf(x * M_SQRT1_2_F)) / 2);
}

// Apply activation based on type
inline float apply_activation(float x, int activation) {
  switch (activation) {
  case GLU_SILU:
    return glu_silu(x);
  case GLU_GELU:
    return glu_gelu(x);
  case GLU_RELU:
    return glu_relu(x);
  case GLU_GELU_ERF:
    return glu_gelu_erf(x);
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
    float a_val = float(a[tid]);
    // Cast activation back to T before multiplying, matching candle's
    // two-step behavior: unary op in float32 -> cast to T -> binary mul in T
    T activated = T(apply_activation(a_val, activation));
    output[tid] = activated * b[tid];
  }
}

#define instantiate_fused_glu(type)                                            \
  template [[host_name("fused_glu_" #type)]] [[kernel]] void fused_glu<type>(  \
      const device type *a [[buffer(0)]], const device type *b [[buffer(1)]],  \
      device type *output [[buffer(2)]],                                       \
      constant uint &n_elements [[buffer(3)]],                                 \
      constant int &activation [[buffer(4)]],                                  \
      uint tid [[thread_position_in_grid]]);

instantiate_fused_glu(float);
instantiate_fused_glu(half);
#if __METAL_VERSION__ >= 310
instantiate_fused_glu(bfloat);
#endif
