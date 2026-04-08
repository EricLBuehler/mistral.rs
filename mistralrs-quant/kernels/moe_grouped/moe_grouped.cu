// Grouped MoE GEMM kernel for prefill optimization.
//
// Instead of launching one block per (output_feature, token, topk_slot) like
// the indexed_moe kernel, this groups tokens by expert and iterates over them
// within each block. This reduces block count by ~500x and enables weight
// data reuse across tokens via L1 cache.
//
// Grid: (N, num_experts) vs indexed_moe's (N, batch, topk)
// For Gemma4 8K tokens: 90K blocks vs 44.8M blocks.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256

// ============== Helper functions ==============

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
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

// ============== Block type definitions ==============

#define QK_K 256
#define K_SCALE_SIZE 12

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

// VDR constants (same as indexed_moe)
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

// ============== vec_dot implementations ==============

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
  return sumi * dm4f.x * ds8f.x + dm4f.y * ds8f.y / (QI8_1 / (vdr * QR4_1));
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
  return sumi * dm5f.x * ds8f.x + dm5f.y * ds8f.y / (QI5_1 / vdr);
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

// ============== vec_dot wrapper functions ==============

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &iqs);

static __device__ __forceinline__ float
vd_q8_0_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq;
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
vd_q4_0_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_0 *bq4_0 = (const block_q4_0 *)vbq;
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
vd_q4_1_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_1 *bq4_1 = (const block_q4_1 *)vbq;
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
vd_q5_0_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_0 *bq5_0 = (const block_q5_0 *)vbq;
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
vd_q5_1_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_1 *bq5_1 = (const block_q5_1 *)vbq;
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
vd_q8_1_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1_v, const int &iqs) {
  const block_q8_1 *bq8_1_w = (const block_q8_1 *)vbq;
  int v[VDR_Q8_1_Q8_1_MMVQ];
  int u[VDR_Q8_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8_aligned(bq8_1_w->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1_v->qs, iqs + i);
  }
  const float2 dmw = __half22float2(bq8_1_w->ds);
  const float2 dmv = __half22float2(bq8_1_v->ds);
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < VDR_Q8_1_Q8_1_MMVQ; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return dmw.x * dmv.x * sumi + dmw.y * dmv.y;
}

static __device__ __forceinline__ float
vd_q2_K_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q2_K *bq2_K = (const block_q2_K *)vbq;
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
vd_q3_K_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q3_K *bq3_K = (const block_q3_K *)vbq;
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
vd_q4_K_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_K *bq4_K = (const block_q4_K *)vbq;
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
vd_q5_K_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_K *bq5_K = (const block_q5_K *)vbq;
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
vd_q6_K_q8_1(const void *__restrict__ vbq,
             const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q6_K *bq6_K = (const block_q6_K *)vbq;
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

// ============== Dispatch kernels ==============

// Phase 1: Count tokens per expert using atomics
static __global__ void moe_dispatch_count_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t *__restrict__ expert_counts,   // [num_experts] zero-initialized
    const int total_assignments) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    atomicAdd(&expert_counts[expert], 1);
  }
}

// Phase 2: Exclusive prefix sum for expert_bounds (single block, <=1024 experts)
static __global__ void moe_dispatch_prefix_sum_kernel(
    const int32_t *__restrict__ expert_counts, // [num_experts]
    int32_t *__restrict__ expert_bounds,        // [num_experts + 1]
    const int num_experts) {
  // Simple sequential prefix sum - fine for num_experts <= 1024
  // Only one thread does this work
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    expert_bounds[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      expert_bounds[i + 1] = expert_bounds[i] + expert_counts[i];
    }
  }
}

// Phase 3: Scatter token indices into sorted positions
static __global__ void moe_dispatch_scatter_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t *__restrict__ expert_cursors,   // [num_experts] init to expert_bounds[i]
    int32_t *__restrict__ sorted_token_ids, // [total_assignments] output
    const int total_assignments) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    const int pos = atomicAdd(&expert_cursors[expert], 1);
    sorted_token_ids[pos] = idx; // flat index into topk_ids: token = idx/topk
  }
}

// ============== Grouped MoE GEMM kernel ==============
//
// Grid: (N, num_experts), Block: (32, 4) = 128 threads
// Each block handles one output row (N dimension) for one expert,
// iterating over all tokens assigned to that expert.
// Weight data stays in L1 cache across token iterations.

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ void moe_grouped_gemm_impl(
    const void *__restrict__ all_weights, // [num_experts, N, K] in quantized blocks
    const void *__restrict__ all_inputs,  // [num_input_rows, K_padded] in Q8_1 blocks
    const int32_t *__restrict__ expert_bounds,    // [num_experts + 1]
    const int32_t *__restrict__ sorted_token_ids, // [total_assignments]
    const float *__restrict__ topk_weights,       // [total_assignments] or NULL
    float *__restrict__ all_outputs,              // [total_assignments, N]
    const int N, const int K, const int K_padded,
    const int num_experts, const int topk, const int input_dim1) {

  const int n_row = blockIdx.x;
  const int expert = blockIdx.y;

  if (n_row >= N || expert >= num_experts)
    return;

  const int t_start = expert_bounds[expert];
  const int t_end = expert_bounds[expert + 1];

  if (t_start >= t_end)
    return;

  // Weight pointer for this (expert, n_row)
  const int blocks_per_row_w = K / qk;
  const int blocks_per_row_x = K_padded / QK8_1;

  const block_q_t *w_base = (const block_q_t *)all_weights;
  const block_q8_1 *x_base = (const block_q8_1 *)all_inputs;

  const block_q_t *w_row = w_base + ((size_t)expert * N + n_row) * blocks_per_row_w;

  constexpr int nwarps = 4;
  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];

  // Process each token assigned to this expert
  for (int ti = t_start; ti < t_end; ti++) {
    const int flat_idx = sorted_token_ids[ti];
    // flat_idx is index into flattened topk_ids: token = flat_idx / topk
    const int input_row = (input_dim1 == 1) ? (flat_idx / topk) : flat_idx;

    const block_q8_1 *x = x_base + (size_t)input_row * blocks_per_row_x;

    float tmp = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_w;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (tid % (qi / vdr));
      tmp += vec_dot_q_cuda(&w_row[kbx], &x[kby], kqs);
    }

    // Reduce across warps
    if (threadIdx.y > 0) {
      tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
      }
      tmp = warp_reduce_sum(tmp);
      if (threadIdx.x == 0) {
        float result = tmp;
        if (topk_weights) {
          result *= topk_weights[flat_idx];
        }
        all_outputs[(size_t)ti * N + n_row] = result;
      }
    }
    __syncthreads();
  }
}

// ============== Kernel instantiations ==============

extern "C" __global__ void moe_grouped_gemm_q8_0(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                          vd_q8_0_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q4_0(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                          vd_q4_0_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q4_1(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                          vd_q4_1_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q5_0(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                          vd_q5_0_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q5_1(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                          vd_q5_1_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q8_1(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                          vd_q8_1_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q2k(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                          vd_q2_K_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q3k(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                          vd_q3_K_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q4k(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                          vd_q4_K_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q5k(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                          vd_q5_K_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

extern "C" __global__ void moe_grouped_gemm_q6k(
    const void *all_weights, const void *all_inputs,
    const int32_t *expert_bounds, const int32_t *sorted_token_ids,
    const float *topk_weights, float *all_outputs,
    int N, int K, int K_padded, int num_experts, int topk, int input_dim1) {
  moe_grouped_gemm_impl<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                          vd_q6_K_q8_1>(
      all_weights, all_inputs, expert_bounds, sorted_token_ids,
      topk_weights, all_outputs, N, K, K_padded, num_experts, topk, input_dim1);
}

// ============== C wrapper functions for FFI ==============

extern "C" void launch_moe_dispatch(
    const int32_t *topk_ids,        // [total_assignments] flattened, device
    int32_t *expert_bounds,          // [num_experts + 1] output, device
    int32_t *sorted_token_ids,       // [total_assignments] output, device
    int total_assignments,
    int num_experts,
    void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  // Allocate temporary buffers
  int32_t *expert_counts;
  int32_t *expert_cursors;
  cudaMallocAsync(&expert_counts, num_experts * sizeof(int32_t), s);
  cudaMallocAsync(&expert_cursors, num_experts * sizeof(int32_t), s);

  // Phase 1: Count tokens per expert
  cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), s);
  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_count_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_counts, total_assignments);
  }

  // Phase 2: Prefix sum -> expert_bounds
  moe_dispatch_prefix_sum_kernel<<<1, 1, 0, s>>>(
      expert_counts, expert_bounds, num_experts);

  // Phase 3: Copy expert_bounds to expert_cursors for scatter
  cudaMemcpyAsync(expert_cursors, expert_bounds,
                  num_experts * sizeof(int32_t), cudaMemcpyDeviceToDevice, s);

  // Phase 4: Scatter token indices
  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_scatter_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_cursors, sorted_token_ids, total_assignments);
  }

  cudaFreeAsync(expert_counts, s);
  cudaFreeAsync(expert_cursors, s);
}

// Macro for generating launch wrappers
#define LAUNCH_MOE_GROUPED_GEMM(suffix)                                        \
  extern "C" void launch_moe_grouped_gemm_##suffix(                            \
      const void *all_weights, const void *all_inputs,                         \
      const int32_t *expert_bounds, const int32_t *sorted_token_ids,           \
      const float *topk_weights, float *all_outputs, int N, int K,            \
      int K_padded, int num_experts, int topk, int input_dim1,                \
      void *stream) {                                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    dim3 grid(N, num_experts);                                                 \
    dim3 block(WARP_SIZE, 4, 1);                                               \
    moe_grouped_gemm_##suffix<<<grid, block, 0, s>>>(                          \
        all_weights, all_inputs, expert_bounds, sorted_token_ids,              \
        topk_weights, all_outputs, N, K, K_padded, num_experts, topk,         \
        input_dim1);                                                           \
  }

LAUNCH_MOE_GROUPED_GEMM(q8_0)
LAUNCH_MOE_GROUPED_GEMM(q4_0)
LAUNCH_MOE_GROUPED_GEMM(q4_1)
LAUNCH_MOE_GROUPED_GEMM(q5_0)
LAUNCH_MOE_GROUPED_GEMM(q5_1)
LAUNCH_MOE_GROUPED_GEMM(q8_1)
LAUNCH_MOE_GROUPED_GEMM(q2k)
LAUNCH_MOE_GROUPED_GEMM(q3k)
LAUNCH_MOE_GROUPED_GEMM(q4k)
LAUNCH_MOE_GROUPED_GEMM(q5k)
LAUNCH_MOE_GROUPED_GEMM(q6k)
