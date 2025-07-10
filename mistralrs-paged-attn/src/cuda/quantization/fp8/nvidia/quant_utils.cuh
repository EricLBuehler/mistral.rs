#pragma once

#include "../../../attention/attention_dtypes.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>
#include "cuda_fp8.h"

namespace vllm {
#ifndef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return float_to_half(half_to_float(tmp.x) * scale);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
  tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U),
                                                         scale, fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion<uint4, uint2>(const uint2& a, const float scale,
                                    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale,
                                                        fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                        scale, fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale,
                                                          fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  uint16_t tmp = res.x;

  // half -> float
  return half_to_float(tmp) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale,
                                                  fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}
  #endif  // ENABLE_FP8

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

}  // namespace fp8
#endif  // not USE_ROCM
}  // namespace vllm