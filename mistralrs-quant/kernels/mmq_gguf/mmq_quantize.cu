// MMQ quantize kernel: f32/f16/bf16 -> block_q8_1_mmq
// Adapted from llama.cpp quantize.cu

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING % (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0,
              "Risk of out-of-bounds access.");

template <typename input_t>
static __device__ __forceinline__ float mmq_to_float(const input_t v) {
  return (float)v;
}

template <>
__device__ __forceinline__ float mmq_to_float<half>(const half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float
mmq_to_float<__nv_bfloat16>(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename input_t>
static __device__ __forceinline__ float4 load_mmq4_scalar(
    const input_t *__restrict__ x, const int64_t base, const int64_t i0,
    const int64_t ne00) {
  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (i0 + 0 < ne00) {
    xi.x = mmq_to_float(x[base + 0]);
  }
  if (i0 + 1 < ne00) {
    xi.y = mmq_to_float(x[base + 1]);
  }
  if (i0 + 2 < ne00) {
    xi.z = mmq_to_float(x[base + 2]);
  }
  if (i0 + 3 < ne00) {
    xi.w = mmq_to_float(x[base + 3]);
  }

  return xi;
}

template <typename input_t>
static __device__ __forceinline__ float4
load_mmq4(const input_t *__restrict__ x, const int64_t base, const int64_t i0,
          const int64_t ne00) {
  return load_mmq4_scalar(x, base, i0, ne00);
}

template <>
__device__ __forceinline__ float4
load_mmq4<float>(const float *__restrict__ x, const int64_t base,
                 const int64_t i0, const int64_t ne00) {
  if (i0 + 3 < ne00) {
    const float4 *x4 = (const float4 *)x;
    return x4[base / 4];
  }

  return load_mmq4_scalar(x, base, i0, ne00);
}

template <>
__device__ __forceinline__ float4
load_mmq4<half>(const half *__restrict__ x, const int64_t base,
                const int64_t i0, const int64_t ne00) {
  if (i0 + 3 < ne00) {
    const half2 *x2 = (const half2 *)x;
    const float2 x01 = __half22float2(x2[base / 2]);
    const float2 x23 = __half22float2(x2[base / 2 + 1]);
    return make_float4(x01.x, x01.y, x23.x, x23.y);
  }

  return load_mmq4_scalar(x, base, i0, ne00);
}

template <>
__device__ __forceinline__ float4
load_mmq4<__nv_bfloat16>(const __nv_bfloat16 *__restrict__ x,
                         const int64_t base, const int64_t i0,
                         const int64_t ne00) {
  if (i0 + 3 < ne00) {
    const __nv_bfloat162 *x2 = (const __nv_bfloat162 *)x;
    const float2 x01 = __bfloat1622float2(x2[base / 2]);
    const float2 x23 = __bfloat1622float2(x2[base / 2 + 1]);
    return make_float4(x01.x, x01.y, x23.x, x23.y);
  }

  return load_mmq4_scalar(x, base, i0, ne00);
}

template <typename input_t, mmq_q8_1_ds_layout ds_layout>
static __global__ void
quantize_mmq_q8_1(const input_t *__restrict__ x,
                  const int32_t *__restrict__ ids,
                  void *__restrict__ vy, const int64_t ne00, const int64_t s01,
                  const int64_t s02, const int64_t s03, const int64_t ne0,
                  const int ne1, const int ne2) {

  constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= ne0) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z % ne2;
  const int64_t i3 = blockIdx.z / ne2;

  const int64_t i00 = i0;
  const int64_t i01 = ids ? ids[i1] : i1;
  const int64_t i02 = i2;
  const int64_t i03 = i3;

  block_q8_1_mmq *y = (block_q8_1_mmq *)vy;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib =
      ib0 + (i0 / (4 * QK8_1)) * ne1 + blockIdx.x; // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1);            // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const int64_t base = i03 * s03 + i02 * s02 + i01 * s01 + i00;
  const float4 xi = load_mmq4(x, base, i0, ne00);
  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));

  // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }

  float sum;
  if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    sum = xi.x + xi.y + xi.z + xi.w;

    // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
  }

  const float d_inv = 127.0f / amax;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);

  // Write back 4 int8 values as a single 32 bit value for better memory
  // bandwidth:
  char4 *yqs4 = (char4 *)y[ib].qs;
  yqs4[iqs / 4] = q;

  if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
    if (iqs % 16 != 0 || iqs >= 96) {
      return;
    }

    y[ib].d2s6[2 + iqs / 16] = sum;

    if (iqs % 64 != 0) {
      return;
    }

    const float d = 1.0f / d_inv;

    y[ib].d2s6[iqs / 64] = d;

    return;
  }

  if (iqs % 32 != 0) {
    return;
  }

  const float d = 1.0f / d_inv;

  if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
    y[ib].ds4[iqs / 32] = make_half2(d, sum);
  } else {
    y[ib].d4[iqs / 32] = d;
  }
}

template <mmq_q8_1_ds_layout ds_layout>
static void launch_mmq_quantize_q8_1_typed(
    const void *x, const int32_t *ids, void *vy, int type_x, int64_t ne00,
    int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, void *stream) {
  const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                              (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
  const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  const cudaStream_t s = (cudaStream_t)stream;

  switch ((ggml_type)type_x) {
  case GGML_TYPE_F32:
    quantize_mmq_q8_1<float, ds_layout><<<num_blocks, block_size, 0, s>>>(
        (const float *)x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    break;
  case GGML_TYPE_F16:
    quantize_mmq_q8_1<half, ds_layout><<<num_blocks, block_size, 0, s>>>(
        (const half *)x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    break;
  case GGML_TYPE_BF16:
    quantize_mmq_q8_1<__nv_bfloat16, ds_layout>
        <<<num_blocks, block_size, 0, s>>>((const __nv_bfloat16 *)x, ids, vy,
                                           ne00, s01, s02, s03, ne0, ne1,
                                           ne2);
    break;
  default:
    break;
  }
}

// C-linkage quantize launchers

extern "C" void
launch_mmq_quantize_q8_1_D4(const void *x, const int32_t *ids, void *vy,
                            int type_x, int64_t ne00, int64_t s01, int64_t s02,
                            int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2,
                            int64_t ne3, void *stream) {
  launch_mmq_quantize_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_D4>(
      x, ids, vy, type_x, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}

extern "C" void
launch_mmq_quantize_q8_1_DS4(const void *x, const int32_t *ids, void *vy,
                             int type_x, int64_t ne00, int64_t s01, int64_t s02,
                             int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2,
                             int64_t ne3, void *stream) {
  launch_mmq_quantize_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_DS4>(
      x, ids, vy, type_x, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}

extern "C" void
launch_mmq_quantize_q8_1_D2S6(const void *x, const int32_t *ids, void *vy,
                              int type_x, int64_t ne00,
    int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1,
    int64_t ne2, int64_t ne3, void *stream) {
  launch_mmq_quantize_q8_1_typed<MMQ_Q8_1_DS_LAYOUT_D2S6>(
      x, ids, vy, type_x, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}
