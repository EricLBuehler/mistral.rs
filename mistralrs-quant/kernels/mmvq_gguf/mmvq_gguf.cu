// GGUF matvec kernels with Q8_1-quantized activations.
// Adapted from llama.cpp's CUDA mmvq path for the GGUF types used here.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

// Constants, types, and helpers shared with the indexed MoE kernels.

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2
#define QK_K 256
#define K_SCALE_SIZE 12
#define MMVQ_NWARPS_SINGLE_COL_PLAIN 8
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN 2
#define MMVQ_NWARPS_SINGLE_COL_FUSED_GLU 4
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU 2
#define MMVQ_NWARPS_SINGLE_COL_FUSED_QKV 8
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV 2
#define MMVQ_FUSED_QKV_PACKED_MIN_ROW_RATIO 2

// Matches candle's MATRIX_ROW_PADDING.
#define MATRIX_ROW_PADDING 512

typedef uint16_t ggml_fp16_t;

static __device__ __forceinline__ float warp_reduce_sum_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
  }
  return x;
}

static __device__ __forceinline__ float warp_reduce_max_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE));
  }
  return x;
}

enum GluActivation {
  GLU_SILU = 0,
  GLU_GELU = 1,
  GLU_RELU = 2,
  GLU_GELU_ERF = 3
};

static __device__ __forceinline__ float glu_silu(float x) {
  return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float glu_gelu(float x) {
  const float kSqrt2OverPi = 0.7978845608f;
  const float kCoeff = 0.044715f;
  const float x3 = x * x * x;
  const float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float glu_relu(float x) {
  return fmaxf(x, 0.0f);
}

static __device__ __forceinline__ float glu_gelu_erf(float x) {
  return x * normcdff(x);
}

static __device__ __forceinline__ float apply_glu_activation(float x, int act) {
  switch (act) {
  case GLU_SILU:
    return glu_silu(x);
  case GLU_GELU:
    return glu_gelu(x);
  case GLU_RELU:
    return glu_relu(x);
  case GLU_GELU_ERF:
    return glu_gelu_erf(x);
  default:
    return glu_silu(x);
  }
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t *x8,
                                                        const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8,
                                                         const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int
get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

static __device__ __forceinline__ int
get_int_from_uint8_aligned(const uint8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

#define MIN_CC_DP4A 610

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b,
                                                     int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
  return __dp4a(a, b, c);
#else
  const int8_t *a8 = (const int8_t *)&a;
  const int8_t *b8 = (const int8_t *)&b;
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

// ---------------------------------------------------------------------------
// Block structs for each supported quantization type
// ---------------------------------------------------------------------------

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
  half d;
  int8_t qs[QK8_0];
} block_q8_0;

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
  half2 ds;
  int8_t qs[QK8_0];
} block_q8_1;

#define QR2_K 4
#define QI2_K (QK_K / (4 * QR2_K))
typedef struct {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  half2 dm;
} block_q2_K;

#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))
typedef struct {
  uint8_t hmask[QK_K / 8];
  uint8_t qs[QK_K / 4];
  uint8_t scales[K_SCALE_SIZE];
  half d;
} block_q3_K;

#define QR4_K 2
#define QI4_K (QK_K / (4 * QR4_K))
typedef struct {
  half2 dm;
  uint8_t scales[3 * QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_q4_K;

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))
typedef struct {
  half2 dm;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
} block_q5_K;

#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))
typedef struct {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  half d;
} block_q6_K;

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
  half d;
  uint8_t qs[QK4_0 / 2];
} block_q4_0;

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
  half2 dm;
  uint8_t qs[QK4_1 / 2];
} block_q4_1;

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
  half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
} block_q5_0;

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
  half2 dm;
  uint8_t qh[4];
  uint8_t qs[QK5_1 / 2];
} block_q5_1;

// VDR = vec-dot unroll factor per type.
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_1_Q8_1_MMVQ 2
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

// ---------------------------------------------------------------------------
// vec_dot impl helpers (per-quant-type)
// ---------------------------------------------------------------------------

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_0_q8_1_impl(const int *v, const int *u, const float &d4,
                       const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = __half22float2(ds8);
  return d4 * (sumi * ds8f.x - (8 * vdr / QI4_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_1_q8_1_impl(const int *v, const int *u, const half2 &dm4,
                       const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm4f = __half22float2(dm4);
  const float2 ds8f = __half22float2(ds8);
  const float d4d8 = dm4f.x * ds8f.x;
  const float m4s8 = dm4f.y * ds8f.y;
  return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_0_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const float &d5, const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = __half22float2(ds8);
  return d5 * (sumi * ds8f.x - (16 * vdr / QI5_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const half2 &dm5, const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm5f = __half22float2(dm5);
  const float2 ds8f = __half22float2(ds8);
  const float d5d8 = dm5f.x * ds8f.x;
  const float m5s8 = dm5f.y * ds8f.y;
  return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q8_0_q8_1_impl(const int *v, const int *u, const half &d8_0,
                       const half &d8_1) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return sumi * __half2float(d8_0) * __half2float(d8_1);
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1_impl_mmvq(const int &v, const int *__restrict__ u,
                            const uint8_t *__restrict__ scales,
                            const half2 &dm2, const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    const int sc = scales[2 * i];
    const int vi = (v >> (2 * i)) & 0x03030303;
    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
    int m = sc >> 4;
    m |= m << 8;
    m |= m << 16;
    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
  }
  const float2 dm2f = __half22float2(dm2);
  return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset,
    const float &d3, const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % (QK_K / 32);
    const int sc_shift_low = 4 * (isc / (QK_K / 32));
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
    const int isc_high = isc % (QK_K / 64);
    const int sc_shift_high = 2 * (isc / (QK_K / 64));
    const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3)
                        << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = __vsubss4(vil, vih);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d3 * sumf;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const half2 &dm4, const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 =
        ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm5,
    const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 =
        ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm5f = __half22float2(dm5);
  return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

// vec_dot wrappers for each quant type.

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &kbx, const int &iqs);

static __device__ __forceinline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_0 *bq4_0 = (const block_q4_0 *)vbq + kbx;
  int v[VDR_Q4_0_Q8_1_MMVQ];
  int u[2 * VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8(bq4_0->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
  }
  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_1 *bq4_1 = (const block_q4_1 *)vbq + kbx;
  int v[VDR_Q4_1_Q8_1_MMVQ];
  int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
  }
  return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_0 *bq5_0 = (const block_q5_0 *)vbq + kbx;
  int vl[VDR_Q5_0_Q8_1_MMVQ];
  int vh[VDR_Q5_0_Q8_1_MMVQ];
  int u[2 * VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8(bq5_0->qs, iqs + i);
    vh[i] = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
  }
  return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d,
                                                    bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_1 *bq5_1 = (const block_q5_1 *)vbq + kbx;
  int vl[VDR_Q5_1_Q8_1_MMVQ];
  int vh[VDR_Q5_1_Q8_1_MMVQ];
  int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
    vh[i] = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
  }
  return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm,
                                                    bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq + kbx;
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
  }
  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                    __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q2_K *bq2_K = (const block_q2_K *)vbq + kbx;
  const int bq8_offset = QR2_K * (iqs / QI8_1);
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const uint8_t *scales = bq2_K->scales + scale_offset;
  const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
  int u[QR2_K];
  float d8[QR2_K];
#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q3_K *bq3_K = (const block_q3_K *)vbq + kbx;
  const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const float d = bq3_K->d;
  const int vl = get_int_from_uint8(bq3_K->qs, iqs);
  const int vh =
      ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
  int u[QR3_K];
  float d8[QR3_K];
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d,
                                     d8);
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_K *bq4_K = (const block_q4_K *)vbq + kbx;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 =
      (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];
  const uint16_t *scales = (const uint16_t *)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_K *bq5_K = (const block_q5_K *)vbq + kbx;
  int vl[2];
  int vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];
  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int *ql =
      (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));
  vl[0] = ql[0];
  vl[1] = ql[4];
  vh[0] = qh[0] >> bq8_offset;
  vh[1] = qh[4] >> bq8_offset;
  const uint16_t *scales = (const uint16_t *)bq5_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q6_K *bq6_K = (const block_q6_K *)vbq + kbx;
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_from_uint8(bq6_K->ql, iqs);
  const int vh =
      get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                                        iqs % (QI6_K / 4)) >>
      vh_shift;
  const int8_t *scales = bq6_K->scales + scale_offset;
  int u[QR6_K];
  float d8[QR6_K];
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
  }
  return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// Core mat-vec-q template.

static constexpr __device__ int mmvq_nwarps_plain_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_PLAIN;
  }
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int mmvq_rows_per_cuda_block_plain_for(int ncols_dst) {
  return (ncols_dst == 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN : 2;
}

static constexpr __device__ int mmvq_nwarps_fused_glu_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_FUSED_GLU;
  }
  return 2;
}

static constexpr __device__ int mmvq_rows_per_cuda_block_fused_glu_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU;
  }
  return (ncols_dst <= 4) ? 1 : 2;
}

static constexpr __device__ int mmvq_nwarps_fused_qkv_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_FUSED_QKV;
  }
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int mmvq_rows_per_cuda_block_fused_qkv_for(int ncols_dst) {
  return (ncols_dst == 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV : 2;
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void
mmvq_core_impl(const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
               dst_t *__restrict__ dst, const int ncols_x, const int nrows_x,
               const int stride_col_y, const int stride_col_dst) {

  constexpr int nwarps = mmvq_nwarps_plain_for(ncols_dst);
  constexpr int rows_per_cuda_block = mmvq_rows_per_cuda_block_plain_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  // Partial sums.
  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        const int weight_kbx = row * blocks_per_row_x + kbx;
        tmp[j][i] +=
            vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], weight_kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum_f32(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      dst[j * stride_col_dst + row0 + threadIdx.x] = (dst_t)tmp[j][threadIdx.x];
    }
  }
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void mmvq_core_fused_glu_impl(
    const void *__restrict__ vx_gate, const void *__restrict__ vx_up,
    const block_q8_1 *__restrict__ y, dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_y,
    const int stride_col_dst, const int activation) {

  constexpr int nwarps = mmvq_nwarps_fused_glu_for(ncols_dst);
  constexpr int rows_per_cuda_block = mmvq_rows_per_cuda_block_fused_glu_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};
  float tmp_up[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        const int weight_kbx = row * blocks_per_row_x + kbx;
        const block_q8_1 *yj = &y[j * stride_col_y + kby];
        tmp_gate[j][i] += vec_dot_q_cuda(vx_gate, yj, weight_kbx, kqs);
        tmp_up[j][i] += vec_dot_q_cuda(vx_up, yj, weight_kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared_gate[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                                  [rows_per_cuda_block][WARP_SIZE];
  __shared__ float tmp_shared_up[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                                [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared_gate[threadIdx.y - 1][j][i][threadIdx.x] = tmp_gate[j][i];
        tmp_shared_up[threadIdx.y - 1][j][i][threadIdx.x] = tmp_up[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
        tmp_up[j][i] += tmp_shared_up[l][j][i][threadIdx.x];
      }
      tmp_gate[j][i] = warp_reduce_sum_f32(tmp_gate[j][i]);
      tmp_up[j][i] = warp_reduce_sum_f32(tmp_up[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      const dst_t gate_val = (dst_t)tmp_gate[j][threadIdx.x];
      const dst_t up_val = (dst_t)tmp_up[j][threadIdx.x];
      const dst_t activated =
          (dst_t)apply_glu_activation((float)gate_val, activation);
      dst[j * stride_col_dst + row0 + threadIdx.x] = activated * up_val;
    }
  }
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void mmvq_core_fused_qkv_impl(
    const void *__restrict__ vx_q, const void *__restrict__ vx_k,
    const void *__restrict__ vx_v, const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ q_dst, dst_t *__restrict__ k_dst,
    dst_t *__restrict__ v_dst, const int ncols_x, const int nrows_q,
    const int nrows_k, const int nrows_v, const int stride_col_y) {

  constexpr int nwarps = mmvq_nwarps_fused_qkv_for(ncols_dst);
  constexpr int rows_per_cuda_block = mmvq_rows_per_cuda_block_fused_qkv_for(ncols_dst);

  const void *vx;
  dst_t *dst;
  int nrows_x;
  int row_block;
  if (gridDim.y == 1) {
    const int q_blocks = (nrows_q + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const int k_blocks = (nrows_k + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const int block = blockIdx.x;
    if (block < q_blocks) {
      row_block = block;
      vx = vx_q;
      dst = q_dst;
      nrows_x = nrows_q;
    } else if (block < q_blocks + k_blocks) {
      row_block = block - q_blocks;
      vx = vx_k;
      dst = k_dst;
      nrows_x = nrows_k;
    } else {
      row_block = block - q_blocks - k_blocks;
      vx = vx_v;
      dst = v_dst;
      nrows_x = nrows_v;
    }
  } else {
    row_block = blockIdx.x;
    if (blockIdx.y == 0) {
      vx = vx_q;
      dst = q_dst;
      nrows_x = nrows_q;
    } else if (blockIdx.y == 1) {
      vx = vx_k;
      dst = k_dst;
      nrows_x = nrows_k;
    } else {
      vx = vx_v;
      dst = v_dst;
      nrows_x = nrows_v;
    }
  }

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * row_block;
  if (row0 >= nrows_x) {
    return;
  }

  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        if (row >= nrows_x) {
          continue;
        }
        const int weight_kbx = row * blocks_per_row_x + kbx;
        tmp[j][i] +=
            vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], weight_kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum_f32(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      dst[j * nrows_x + row0 + threadIdx.x] = (dst_t)tmp[j][threadIdx.x];
    }
  }
}

// ---------------------------------------------------------------------------
// Extern-C kernel entry points
//
// Macro expands `MMVQ_PLAIN_ENTRY(tag, block_q_t, qk, qi, vdr, vec_dot,
// dst_tag, dst_c_type, ncols)` into one `__global__` function for batch
// size `ncols`. The Rust launcher switches on batch size 1..=8.
// ---------------------------------------------------------------------------

#define MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                         dst_tag, dst_c_type, ncols)                           \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_plain_for(ncols) * WARP_SIZE, 1)                             \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda##ncols(                         \
          const void *__restrict__ vx, const void *__restrict__ vy,            \
          dst_c_type *__restrict__ dst, const int ncols_x, const int nrows_x,  \
          const int stride_col_y, const int stride_col_dst) {                  \
    mmvq_core_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot,    \
                   ncols>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,   \
                          stride_col_y, stride_col_dst);                       \
  }

#define MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, \
                             dst_tag, dst_c_type, ncols)                       \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_fused_glu_for(ncols) * WARP_SIZE, 1)                         \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda##ncols(                     \
          const void *__restrict__ vx_gate, const void *__restrict__ vx_up,    \
          const void *__restrict__ vy, dst_c_type *__restrict__ dst,           \
          const int ncols_x, const int nrows_x, const int stride_col_y,        \
          const int stride_col_dst, const int activation) {                    \
    mmvq_core_fused_glu_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val,   \
                             vec_dot, ncols>(                                  \
        vx_gate, vx_up, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,         \
        stride_col_y, stride_col_dst, activation);                             \
  }

#define MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, \
                             dst_tag, dst_c_type, ncols)                       \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_fused_qkv_for(ncols) * WARP_SIZE, 1)                         \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda##ncols(                     \
          const void *__restrict__ vx_q, const void *__restrict__ vx_k,        \
          const void *__restrict__ vx_v, const void *__restrict__ vy,          \
          dst_c_type *__restrict__ q_dst, dst_c_type *__restrict__ k_dst,      \
          dst_c_type *__restrict__ v_dst, const int ncols_x,                   \
          const int nrows_q, const int nrows_k, const int nrows_v,             \
          const int stride_col_y) {                                            \
    mmvq_core_fused_qkv_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val,   \
                             vec_dot, ncols>(                                  \
        vx_q, vx_k, vx_v, (const block_q8_1 *)vy, q_dst, k_dst, v_dst,         \
        ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y);                     \
  }

// -- plain entries for all 10 supported quant types, batch sizes 1..8, bf16 +
// f16 + f32 --
#define MMVQ_PLAIN_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot) \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 1)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 2)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 3)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 4)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 5)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 6)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 7)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 8)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 1)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 2)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 3)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 4)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 5)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 6)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 7)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 8)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 1)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 2)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 3)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 4)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 5)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 6)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 7)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 8)

#define MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val,      \
                                 vec_dot, dst_tag, dst_c_type)                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 1)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 2)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 3)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 4)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 5)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 6)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 7)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 8)

#define MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val,      \
                                 vec_dot, dst_tag, dst_c_type)                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 1)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 2)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 3)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 4)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 5)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 6)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 7)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 8)

MMVQ_PLAIN_BATCH_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                     vec_dot_q5_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1)
MMVQ_PLAIN_BATCH_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                     vec_dot_q3_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                     vec_dot_q5_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)

#define MMVQ_FUSED_GLU_TYPE_SET(tag, block_q_t, qk_val, qi_val, vdr_val,       \
                                vec_dot)                                       \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           bf16, __nv_bfloat16)                                \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f16, half)                                          \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f32, float)

MMVQ_FUSED_GLU_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                        vec_dot_q5_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                        vec_dot_q3_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                        vec_dot_q5_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)

#define MMVQ_FUSED_QKV_TYPE_SET(tag, block_q_t, qk_val, qi_val, vdr_val,       \
                                vec_dot)                                       \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           bf16, __nv_bfloat16)                                \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f16, half)                                          \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f32, float)

MMVQ_FUSED_QKV_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                        vec_dot_q5_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                        vec_dot_q3_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                        vec_dot_q5_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)

// Padding-aware BF16/F16/F32 -> Q8_1 quantize kernels.

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_bf16(const __nv_bfloat16 *__restrict__ x,
                             void *__restrict__ vy, const int kx,
                             const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __bfloat162float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f16(const half *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f32(const float *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

// Host-side launchers used by `mistralrs-quant/src/gguf/ffi.rs`.

#define MMVQ_LAUNCHER_PLAIN(tag, dst_tag, dst_c_type)                          \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_plain(                  \
      const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,     \
      int stride_col_y, int stride_col_dst, int b_size, void *stream) {        \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN : 2;             \
    const unsigned int nblocks =                                               \
        (unsigned int)((nrows_x + rows_per_block - 1) / rows_per_block);       \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_PLAIN;                                   \
    } else if (b_size <= 4) {                                                  \
      nwarps = 4;                                                              \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, 1, 1);                                                  \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda1<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda2<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda3<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda4<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda5<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda6<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda7<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda8<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

#define MMVQ_LAUNCHER_FUSED_GLU(tag, dst_tag, dst_c_type)                      \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_fused_glu(              \
      const void *vx_gate, const void *vx_up, const void *vy, void *dst,       \
      int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst,          \
      int b_size, int activation, void *stream) {                              \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU              \
                      : ((b_size <= 4) ? 1 : 2);                              \
    const unsigned int nblocks =                                               \
        (unsigned int)((nrows_x + rows_per_block - 1) / rows_per_block);       \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_FUSED_GLU;                               \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, 1, 1);                                                  \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda1<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda2<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda3<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda4<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda5<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda6<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda7<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda8<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

#define MMVQ_LAUNCHER_FUSED_QKV(tag, dst_tag, dst_c_type)                      \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_fused_qkv(              \
      const void *vx_q, const void *vx_k, const void *vx_v, const void *vy,    \
      void *q_dst, void *k_dst, void *v_dst, int ncols_x, int nrows_q,         \
      int nrows_k, int nrows_v, int stride_col_y, int b_size, void *stream) {  \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV : 2;         \
    int max_nrows = nrows_q > nrows_k ? nrows_q : nrows_k;                     \
    max_nrows = max_nrows > nrows_v ? max_nrows : nrows_v;                     \
    int min_nrows = nrows_q < nrows_k ? nrows_q : nrows_k;                     \
    min_nrows = min_nrows < nrows_v ? min_nrows : nrows_v;                     \
    const bool use_packed =                                                    \
        max_nrows >= MMVQ_FUSED_QKV_PACKED_MIN_ROW_RATIO * min_nrows;          \
    const unsigned int q_blocks =                                               \
        (unsigned int)((nrows_q + rows_per_block - 1) / rows_per_block);       \
    const unsigned int k_blocks =                                               \
        (unsigned int)((nrows_k + rows_per_block - 1) / rows_per_block);       \
    const unsigned int v_blocks =                                               \
        (unsigned int)((nrows_v + rows_per_block - 1) / rows_per_block);       \
    const unsigned int nblocks = use_packed                                    \
        ? q_blocks + k_blocks + v_blocks                                       \
        : (unsigned int)((max_nrows + rows_per_block - 1) / rows_per_block);   \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_FUSED_QKV;                               \
    } else if (b_size <= 4) {                                                  \
      nwarps = 4;                                                              \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, use_packed ? 1 : 3, 1);                                 \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda1<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda2<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda3<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda4<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda5<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda6<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda7<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda8<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

MMVQ_LAUNCHER_PLAIN(q4_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q8_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q2_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q3_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q6_k, bf16, __nv_bfloat16)

MMVQ_LAUNCHER_PLAIN(q4_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q8_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q2_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q3_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q6_k, f16, half)

MMVQ_LAUNCHER_PLAIN(q4_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q4_1, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_1, f32, float)
MMVQ_LAUNCHER_PLAIN(q8_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q2_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q3_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q4_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q6_k, f32, float)

#define MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(tag)                                  \
  MMVQ_LAUNCHER_FUSED_GLU(tag, bf16, __nv_bfloat16)                            \
  MMVQ_LAUNCHER_FUSED_GLU(tag, f16, half)                                      \
  MMVQ_LAUNCHER_FUSED_GLU(tag, f32, float)

MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_1)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_1)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q2_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q3_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q6_k)

#define MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(tag)                                  \
  MMVQ_LAUNCHER_FUSED_QKV(tag, bf16, __nv_bfloat16)                            \
  MMVQ_LAUNCHER_FUSED_QKV(tag, f16, half)                                      \
  MMVQ_LAUNCHER_FUSED_QKV(tag, f32, float)

MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q2_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q3_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q6_k)

// Quantize launchers.

extern "C" void launch_mmvq_gguf_quantize_q8_1_bf16(const void *x, void *vy,
                                                    int kx, int kx_padded,
                                                    int num_rows,
                                                    void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_bf16<<<grid, block, 0, s>>>((const __nv_bfloat16 *)x,
                                                      vy, kx, kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f16(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f16<<<grid, block, 0, s>>>((const half *)x, vy, kx,
                                                     kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f32(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f32<<<grid, block, 0, s>>>((const float *)x, vy, kx,
                                                     kx_padded);
}
