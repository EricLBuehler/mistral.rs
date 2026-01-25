/*
 * AFQ (Affine Fast Quantization) CUDA Utilities
 *
 * Common definitions and helper functions for AFQ CUDA kernels.
 */

#ifndef AFQ_UTILS_CUH
#define AFQ_UTILS_CUH

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Block sizes for various operations
#define AFQ_BLOCK_SIZE 256
#define AFQ_WARP_SIZE 32

// Tile sizes for matrix operations
#define AFQ_TILE_M 32
#define AFQ_TILE_N 32
#define AFQ_TILE_K 32

// Utility: ceiling division
__host__ __device__ __forceinline__ unsigned int cdiv(unsigned int a,
                                                       unsigned int b) {
  return (a + b - 1) / b;
}

// ============================================================================
// Bit extraction helpers
// ============================================================================

// Extract N-bit value from packed u32 at given position
template <int bits>
__device__ __forceinline__ uint32_t extract_bits(uint32_t packed, int pos) {
  constexpr uint32_t mask = (1u << bits) - 1;
  return (packed >> (pos * bits)) & mask;
}

// Specialized extraction for 3-bit (spans bytes)
__device__ __forceinline__ uint32_t extract_3bit(const uint8_t *data,
                                                  int idx) {
  // 8 values packed into 3 bytes (24 bits)
  int byte_offset = (idx * 3) / 8;
  int bit_offset = (idx * 3) % 8;

  uint32_t val;
  if (bit_offset <= 5) {
    val = (data[byte_offset] >> bit_offset) & 0x7;
  } else {
    val = ((data[byte_offset] >> bit_offset) |
           (data[byte_offset + 1] << (8 - bit_offset))) &
          0x7;
  }
  return val;
}

// Specialized extraction for 6-bit (spans bytes)
__device__ __forceinline__ uint32_t extract_6bit(const uint8_t *data,
                                                  int idx) {
  // 4 values packed into 3 bytes (24 bits)
  int byte_offset = (idx * 6) / 8;
  int bit_offset = (idx * 6) % 8;

  uint32_t val;
  if (bit_offset <= 2) {
    val = (data[byte_offset] >> bit_offset) & 0x3f;
  } else {
    val = ((data[byte_offset] >> bit_offset) |
           (data[byte_offset + 1] << (8 - bit_offset))) &
          0x3f;
  }
  return val;
}

// ============================================================================
// Dequantization helpers (q * scale + bias)
// ============================================================================

template <typename T>
__device__ __forceinline__ T dequant_value(uint32_t q, T scale, T bias);

template <>
__device__ __forceinline__ float dequant_value<float>(uint32_t q, float scale,
                                                       float bias) {
  return fmaf((float)q, scale, bias);
}

template <>
__device__ __forceinline__ __half dequant_value<__half>(uint32_t q, __half scale,
                                                         __half bias) {
  return __hfma(__uint2half_rn(q), scale, bias);
}

#if __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ __nv_bfloat16
dequant_value<__nv_bfloat16>(uint32_t q, __nv_bfloat16 scale,
                              __nv_bfloat16 bias) {
  return __hfma(__uint2bfloat16_rn(q), scale, bias);
}
#endif

// ============================================================================
// Quantization helpers (round((w - bias) / scale))
// ============================================================================

template <typename T, int bits>
__device__ __forceinline__ uint32_t quant_value(T w, T scale, T bias);

template <>
__device__ __forceinline__ uint32_t quant_value<float, 2>(float w, float scale,
                                                           float bias) {
  float q = roundf((w - bias) / scale);
  return min(max((int)q, 0), 3);
}

template <>
__device__ __forceinline__ uint32_t quant_value<float, 3>(float w, float scale,
                                                           float bias) {
  float q = roundf((w - bias) / scale);
  return min(max((int)q, 0), 7);
}

template <>
__device__ __forceinline__ uint32_t quant_value<float, 4>(float w, float scale,
                                                           float bias) {
  float q = roundf((w - bias) / scale);
  return min(max((int)q, 0), 15);
}

template <>
__device__ __forceinline__ uint32_t quant_value<float, 6>(float w, float scale,
                                                           float bias) {
  float q = roundf((w - bias) / scale);
  return min(max((int)q, 0), 63);
}

template <>
__device__ __forceinline__ uint32_t quant_value<float, 8>(float w, float scale,
                                                           float bias) {
  float q = roundf((w - bias) / scale);
  return min(max((int)q, 0), 255);
}

// Half precision versions
template <>
__device__ __forceinline__ uint32_t quant_value<__half, 2>(__half w,
                                                            __half scale,
                                                            __half bias) {
  float wf = __half2float(w);
  float sf = __half2float(scale);
  float bf = __half2float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 3);
}

template <>
__device__ __forceinline__ uint32_t quant_value<__half, 3>(__half w,
                                                            __half scale,
                                                            __half bias) {
  float wf = __half2float(w);
  float sf = __half2float(scale);
  float bf = __half2float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 7);
}

template <>
__device__ __forceinline__ uint32_t quant_value<__half, 4>(__half w,
                                                            __half scale,
                                                            __half bias) {
  float wf = __half2float(w);
  float sf = __half2float(scale);
  float bf = __half2float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 15);
}

template <>
__device__ __forceinline__ uint32_t quant_value<__half, 6>(__half w,
                                                            __half scale,
                                                            __half bias) {
  float wf = __half2float(w);
  float sf = __half2float(scale);
  float bf = __half2float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 63);
}

template <>
__device__ __forceinline__ uint32_t quant_value<__half, 8>(__half w,
                                                            __half scale,
                                                            __half bias) {
  float wf = __half2float(w);
  float sf = __half2float(scale);
  float bf = __half2float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 255);
}

#if __CUDA_ARCH__ >= 800
// BFloat16 versions
template <>
__device__ __forceinline__ uint32_t
quant_value<__nv_bfloat16, 2>(__nv_bfloat16 w, __nv_bfloat16 scale,
                               __nv_bfloat16 bias) {
  float wf = __bfloat162float(w);
  float sf = __bfloat162float(scale);
  float bf = __bfloat162float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 3);
}

template <>
__device__ __forceinline__ uint32_t
quant_value<__nv_bfloat16, 3>(__nv_bfloat16 w, __nv_bfloat16 scale,
                               __nv_bfloat16 bias) {
  float wf = __bfloat162float(w);
  float sf = __bfloat162float(scale);
  float bf = __bfloat162float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 7);
}

template <>
__device__ __forceinline__ uint32_t
quant_value<__nv_bfloat16, 4>(__nv_bfloat16 w, __nv_bfloat16 scale,
                               __nv_bfloat16 bias) {
  float wf = __bfloat162float(w);
  float sf = __bfloat162float(scale);
  float bf = __bfloat162float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 15);
}

template <>
__device__ __forceinline__ uint32_t
quant_value<__nv_bfloat16, 6>(__nv_bfloat16 w, __nv_bfloat16 scale,
                               __nv_bfloat16 bias) {
  float wf = __bfloat162float(w);
  float sf = __bfloat162float(scale);
  float bf = __bfloat162float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 63);
}

template <>
__device__ __forceinline__ uint32_t
quant_value<__nv_bfloat16, 8>(__nv_bfloat16 w, __nv_bfloat16 scale,
                               __nv_bfloat16 bias) {
  float wf = __bfloat162float(w);
  float sf = __bfloat162float(scale);
  float bf = __bfloat162float(bias);
  float q = roundf((wf - bf) / sf);
  return min(max((int)q, 0), 255);
}
#endif

// ============================================================================
// Warp-level primitives
// ============================================================================

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = AFQ_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = AFQ_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
#pragma unroll
  for (int offset = AFQ_WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Broadcast value from lane 0 to all lanes
__device__ __forceinline__ float warp_broadcast(float val) {
  return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// Group index calculation
// ============================================================================

// Given a column index, compute the group index
__device__ __forceinline__ int get_group_idx(int col, int group_size) {
  return col / group_size;
}

#endif
