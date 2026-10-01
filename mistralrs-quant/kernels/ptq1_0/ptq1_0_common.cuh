// Device helpers shared by the PTQ1_0 CUDA kernels.
#pragma once

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "ptq1_0_math.cuh"

#define WARP_SIZE 32
#define PREPARE_THREADS 256
#define FWHT_SCALE 0.03125f // 1 / sqrt(FWHT_BLOCK)

static __device__ __forceinline__ float to_float(float v) { return v; }
static __device__ __forceinline__ float to_float(__half v) {
  return __half2float(v);
}
static __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

static __device__ __forceinline__ void store_float(float *p, float v) {
  *p = v;
}
static __device__ __forceinline__ void store_float(__half *p, float v) {
  *p = __float2half(v);
}
static __device__ __forceinline__ void store_float(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

static __device__ __forceinline__ float block_scale(const uint8_t *blk) {
  const unsigned short bits = *reinterpret_cast<const unsigned short *>(
      blk + ptq1_0::BLOCK_BYTES - 2);
  return __half2float(__ushort_as_half(bits));
}

// Unnormalized FWHT of the FWHT_BLOCK floats in `s`; every thread of the CTA
// must call it.
static __device__ __forceinline__ void fwht_shared(float *s, int tid) {
  for (int h = 1; h < ptq1_0::FWHT_BLOCK; h <<= 1) {
    for (int pair = tid; pair < ptq1_0::FWHT_BLOCK / 2;
         pair += PREPARE_THREADS) {
      const int i0 = ptq1_0::fwht_low_index(pair, h);
      const float a = s[i0];
      const float b = s[i0 + h];
      s[i0] = a + b;
      s[i0 + h] = a - b;
    }
    __syncthreads();
  }
}
