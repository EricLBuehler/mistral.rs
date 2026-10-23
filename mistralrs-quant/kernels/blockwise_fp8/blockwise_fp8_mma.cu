#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kGroup = 128;
constexpr int kTileRows = 16;
constexpr int kSplitK = 8;
constexpr int kMaxRows = 32;
constexpr int kQuantizeWarps = 4;
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0e-12f;

enum Status : int { kOk = 0, kInvalid = 1, kLaunchFailed = 2 };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define MISTRALRS_FP8_MMA_AVAILABLE 1
#endif

__device__ __forceinline__ void mma_e4m3(float (&c)[4], uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#ifdef MISTRALRS_FP8_MMA_AVAILABLE
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// One block per 16-row weight tile, kSplitK warps striding the 128-wide k groups, smem reduce at the end.
// Scales: ws[n / 128][kg] with row stride ws_stride; xs(row, kg) at row * xs_stride_m + kg * xs_stride_g.
template <int NT>
__global__ void __launch_bounds__(kSplitK * 32) fp8_mma_gemv_kernel(
    const uint8_t *__restrict__ w, const uint8_t *__restrict__ xq,
    const float *__restrict__ xs, const float *__restrict__ ws,
    __nv_bfloat16 *__restrict__ y, int m, int n, int k, int ws_stride,
    int xs_stride_m, int xs_stride_g) {
#ifdef MISTRALRS_FP8_MMA_AVAILABLE
  __shared__ float red[kSplitK][32][NT * 4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * kTileRows;
  const int r0 = n0 + g, r1 = r0 + 8;
  const uint8_t *w0 = w + static_cast<size_t>(r0) * k + t * 16;
  const uint8_t *w1 = w + static_cast<size_t>(r1) * k + t * 16;
  const uint8_t *xb[NT];
  bool xv[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int tok = j * 8 + g;
    xv[j] = tok < m;
    xb[j] = xq + static_cast<size_t>(xv[j] ? tok : 0) * k + t * 16;
  }
  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  const int groups = k / kGroup;
  const float *ws_row = ws + static_cast<size_t>(n0 / kGroup) * ws_stride;
#pragma unroll 2
  for (int kg = warp; kg < groups; kg += kSplitK) {
    float part[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) part[j][0] = part[j][1] = part[j][2] = part[j][3] = 0.f;
    const int kb = kg * kGroup;
    uint4 wa[2], wb[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      wa[s] = __ldg(reinterpret_cast<const uint4 *>(w0 + kb + s * 64));
      wb[s] = __ldg(reinterpret_cast<const uint4 *>(w1 + kb + s * 64));
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        uint4 x = make_uint4(0u, 0u, 0u, 0u);
        if (xv[j]) x = __ldg(reinterpret_cast<const uint4 *>(xb[j] + kb + s * 64));
        mma_e4m3(part[j], wa[s].x, wb[s].x, wa[s].y, wb[s].y, x.x, x.y);
        mma_e4m3(part[j], wa[s].z, wb[s].z, wa[s].w, wb[s].w, x.z, x.w);
      }
    }
    const float wsc = __ldg(ws_row + kg);
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int tok0 = j * 8 + t * 2, tok1 = tok0 + 1;
      const float s0 = tok0 < m ? __ldg(xs + static_cast<size_t>(tok0) * xs_stride_m + static_cast<size_t>(kg) * xs_stride_g) * wsc : 0.f;
      const float s1 = tok1 < m ? __ldg(xs + static_cast<size_t>(tok1) * xs_stride_m + static_cast<size_t>(kg) * xs_stride_g) * wsc : 0.f;
      acc[j][0] += part[j][0] * s0;
      acc[j][1] += part[j][1] * s1;
      acc[j][2] += part[j][2] * s0;
      acc[j][3] += part[j][3] * s1;
    }
  }
  if (warp > 0) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      red[warp][lane][j * 4 + 0] = acc[j][0];
      red[warp][lane][j * 4 + 1] = acc[j][1];
      red[warp][lane][j * 4 + 2] = acc[j][2];
      red[warp][lane][j * 4 + 3] = acc[j][3];
    }
  }
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int src = 1; src < kSplitK; ++src) {
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      acc[j][0] += red[src][lane][j * 4 + 0];
      acc[j][1] += red[src][lane][j * 4 + 1];
      acc[j][2] += red[src][lane][j * 4 + 2];
      acc[j][3] += red[src][lane][j * 4 + 3];
    }
  }
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int tok0 = j * 8 + t * 2, tok1 = tok0 + 1;
    if (tok0 < m) {
      y[static_cast<size_t>(tok0) * n + r0] = __float2bfloat16(acc[j][0]);
      y[static_cast<size_t>(tok0) * n + r1] = __float2bfloat16(acc[j][2]);
    }
    if (tok1 < m) {
      y[static_cast<size_t>(tok1) * n + r0] = __float2bfloat16(acc[j][1]);
      y[static_cast<size_t>(tok1) * n + r1] = __float2bfloat16(acc[j][3]);
    }
  }
#endif
}

// Dynamic per-row 1x128 E4M3 quantization; one warp per (row, group), 4 values per lane.
__global__ void __launch_bounds__(kQuantizeWarps * 32) quantize_rows_bf16_kernel(
    const __nv_bfloat16 *__restrict__ x, uint8_t *__restrict__ xq,
    float *__restrict__ xs, int rows, int cols, int xs_stride_m, int xs_stride_g) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int groups = cols / kGroup;
  const int item = blockIdx.x * kQuantizeWarps + warp;
  if (item >= rows * groups) return;
  const int row = item / groups, group = item % groups;
  const size_t offset = static_cast<size_t>(row) * cols + static_cast<size_t>(group) * kGroup + lane * 4;
  const uint2 raw = *reinterpret_cast<const uint2 *>(x + offset);
  const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.x));
  const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&raw.y));
  float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
  const float scale = fmaxf(amax / kFp8Max, kMinScale);
  const float inv = 1.0f / scale;
  const __nv_fp8x2_storage_t q0 = __nv_cvt_float2_to_fp8x2(make_float2(f0.x * inv, f0.y * inv), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t q1 = __nv_cvt_float2_to_fp8x2(make_float2(f1.x * inv, f1.y * inv), __NV_SATFINITE, __NV_E4M3);
  *reinterpret_cast<uint32_t *>(xq + offset) = static_cast<uint32_t>(q0) | (static_cast<uint32_t>(q1) << 16);
  if (lane == 0) xs[static_cast<size_t>(row) * xs_stride_m + static_cast<size_t>(group) * xs_stride_g] = scale;
}

template <int NT>
void launch_gemv(const void *w, const void *xq, const float *xs, const float *ws, void *y, int m,
                 int n, int k, int ws_stride, int xs_stride_m, int xs_stride_g, cudaStream_t stream) {
  fp8_mma_gemv_kernel<NT><<<n / kTileRows, kSplitK * 32, 0, stream>>>(
      static_cast<const uint8_t *>(w), static_cast<const uint8_t *>(xq), xs, ws,
      static_cast<__nv_bfloat16 *>(y), m, n, k, ws_stride, xs_stride_m, xs_stride_g);
}

}  // namespace

extern "C" const char *mistralrs_fp8_mma_error_string(int status) {
  switch (status) {
    case kOk: return "ok";
    case kInvalid: return "invalid arguments";
    case kLaunchFailed: return "kernel launch failed";
    default: return "unknown status";
  }
}

extern "C" int mistralrs_fp8_mma_gemv(const void *w, const void *xq, const float *xs,
                                      const float *ws, void *y, int m, int n, int k,
                                      int ws_stride, int xs_stride_m, int xs_stride_g,
                                      cudaStream_t stream) {
  if (m < 1 || m > kMaxRows || n <= 0 || k <= 0 || n % kTileRows != 0 || k % kGroup != 0) return kInvalid;
  if ((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xq)) & 15) return kInvalid;
  switch ((m + 7) / 8) {
    case 1: launch_gemv<1>(w, xq, xs, ws, y, m, n, k, ws_stride, xs_stride_m, xs_stride_g, stream); break;
    case 2: launch_gemv<2>(w, xq, xs, ws, y, m, n, k, ws_stride, xs_stride_m, xs_stride_g, stream); break;
    case 3: launch_gemv<3>(w, xq, xs, ws, y, m, n, k, ws_stride, xs_stride_m, xs_stride_g, stream); break;
    default: launch_gemv<4>(w, xq, xs, ws, y, m, n, k, ws_stride, xs_stride_m, xs_stride_g, stream); break;
  }
  return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailed;
}

extern "C" int mistralrs_fp8_mma_quantize_bf16(const void *x, void *xq, float *xs, int rows,
                                               int cols, int xs_stride_m, int xs_stride_g,
                                               cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || cols % kGroup != 0) return kInvalid;
  if ((reinterpret_cast<uintptr_t>(x) & 7) || (reinterpret_cast<uintptr_t>(xq) & 3)) return kInvalid;
  const int items = rows * (cols / kGroup);
  quantize_rows_bf16_kernel<<<(items + kQuantizeWarps - 1) / kQuantizeWarps, kQuantizeWarps * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x), static_cast<uint8_t *>(xq), xs, rows, cols, xs_stride_m, xs_stride_g);
  return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailed;
}
