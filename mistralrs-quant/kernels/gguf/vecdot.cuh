#pragma once

// Shared GGUF dot products adapted from llama.cpp and candle-kernels.
// Source revision and license: third_party/ggml_mmvq/README.md.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#define QK_K 256
#define K_SCALE_SIZE 12

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
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

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &iqs);

static __device__ __forceinline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
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
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
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
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
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
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
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
vec_dot_q8_1_q8_1(const void *__restrict__ vbq,
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
  return dmw.x * dmv.x * sumi;
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
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
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
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
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
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
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
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
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
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
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
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

#define GGUF_DENSE_PREFETCH_FORMATS(M)                                         \
  M(q4_0, Q4_0, 2, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,               \
    vec_dot_q4_0_q8_1)                                                         \
  M(q4_1, Q4_1, 3, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,               \
    vec_dot_q4_1_q8_1)                                                         \
  M(q5_0, Q5_0, 6, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,               \
    vec_dot_q5_0_q8_1)                                                         \
  M(q5_1, Q5_1, 7, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,               \
    vec_dot_q5_1_q8_1)                                                         \
  M(q8_0, Q8_0, 8, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,               \
    vec_dot_q8_0_q8_1)                                                         \
  M(q4k, Q4K, 12, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,                 \
    vec_dot_q4_K_q8_1)                                                         \
  M(q5k, Q5K, 13, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,                 \
    vec_dot_q5_K_q8_1)                                                         \
  M(q6k, Q6K, 14, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,                 \
    vec_dot_q6_K_q8_1)

#define GGUF_DENSE_VECTOR_FORMATS(M)                                           \
  GGUF_DENSE_PREFETCH_FORMATS(M)                                               \
  M(q2k, Q2K, 10, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,                 \
    vec_dot_q2_K_q8_1)                                                         \
  M(q3k, Q3K, 11, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,                 \
    vec_dot_q3_K_q8_1)

#define GGUF_VECTOR_FORMATS(M)                                                 \
  GGUF_DENSE_VECTOR_FORMATS(M)                                                 \
  M(q8_1, Q8_1, 9, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,               \
    vec_dot_q8_1_q8_1)
