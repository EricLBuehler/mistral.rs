#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#define HC_THREADS 128
#define HC_WARP_SIZE 32

constexpr int HC_COMBINE_VECTOR = 4;
constexpr int HC_COMBINE_RETAINED_VALUES = 32;
constexpr uintptr_t HC_COMBINE_ALIGNMENT = 16;
constexpr int HC_UP_VECTOR = 8;
constexpr int HC_UP_LANES = 8;
constexpr int HC_UP_THREADS = 256;

template <typename T, int V> struct alignas(sizeof(T) * V) HcPack {
  T x[V];
};

template <typename T> __device__ __forceinline__ float hc_f(T v) {
  return (float)v;
}
template <typename T> __device__ __forceinline__ T hc_t(float v);
template <> __device__ __forceinline__ float hc_t<float>(float v) { return v; }
template <> __device__ __forceinline__ __half hc_t<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 hc_t<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float hc_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float hc_silu(float x) { return x * hc_sigmoid(x); }

__device__ __forceinline__ float hc_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, o);
  }
  return v;
}

// All threads must call; blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float hc_block_sum(float v, float *smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  v = hc_warp_sum(v);
  __syncthreads();
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  v = threadIdx.x < warps ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = hc_warp_sum(v);
    if (lane == 0) {
      smem[0] = v;
    }
  }
  __syncthreads();
  return smem[0];
}

template <int FORMAT>
__device__ __forceinline__ float hc_weight(const void *weight, size_t i) {
  if constexpr (FORMAT == 1) {
    return __bfloat162float(((const __nv_bfloat16 *)weight)[i]);
  } else {
    return __half2float(((const __half *)weight)[i]);
  }
}

template <typename T, int FORMAT>
__global__ void hc_up_mix_kernel(const T *__restrict__ lo, const void *up,
                                 const T *__restrict__ xn, T *__restrict__ out,
                                 int hidden, int rank, int hc) {
  const int warp = threadIdx.x / HC_WARP_SIZE;
  const int lane = threadIdx.x % HC_WARP_SIZE;
  const int h = blockIdx.x * (HC_THREADS / HC_WARP_SIZE) + warp;
  if (h >= hidden) {
    return;
  }
  const int token = blockIdx.y;
  const T *input = lo + (size_t)token * rank;
  float mixed = 0.0f;
  for (int stream = 0; stream < hc; ++stream) {
    const int row = stream * hidden + h;
    float sum = 0.0f;
    for (int k = lane; k < rank; k += HC_WARP_SIZE) {
      sum = fmaf(hc_f(input[k]), hc_weight<FORMAT>(up, (size_t)row * rank + k),
                 sum);
    }
    sum = hc_warp_sum(sum);
    if (lane == 0) {
      mixed += hc_f(xn[(size_t)token * hc * hidden + row]) *
               hc_sigmoid(hc_f(hc_t<T>(sum)));
    }
  }
  if (lane == 0) {
    out[(size_t)token * hidden + h] = hc_t<T>(mixed / hc);
  }
}

template <typename T, bool STRIDED>
__global__ void hc_silu_kernel(const T *__restrict__ input,
                               T *__restrict__ output, size_t elements,
                               size_t columns, size_t row_stride, int hc) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) {
    const size_t source =
        STRIDED ? (i / columns) * row_stride + i % columns : i;
    const float scaled = hc_f(hc_t<T>(hc_f(input[source]) / hc));
    output[i] = hc_t<T>(hc_silu(scaled));
  }
}

template <typename T, int FORMAT>
__global__ void
hc_up_mix_vector_kernel(const T *__restrict__ lo, const void *__restrict__ up,
                        const T *__restrict__ xn, T *__restrict__ out,
                        int hidden, int rank, int hc) {
  using Weight = std::conditional_t<FORMAT == 1, __nv_bfloat16, __half>;
  const int lane = threadIdx.x % HC_UP_LANES;
  const int h =
      blockIdx.x * (HC_UP_THREADS / HC_UP_LANES) + threadIdx.x / HC_UP_LANES;
  if (h >= hidden)
    return;
  const int token = blockIdx.y;
  const auto *input = reinterpret_cast<const HcPack<T, HC_UP_VECTOR> *>(
      lo + (size_t)token * rank);
  float mixed = 0.0f;
  for (int stream = 0; stream < hc; ++stream) {
    const int row = stream * hidden + h;
    const auto *weight = reinterpret_cast<const HcPack<Weight, HC_UP_VECTOR> *>(
        static_cast<const Weight *>(up) + (size_t)row * rank);
    float sum = 0.0f;
    for (int k = lane; k < rank / HC_UP_VECTOR; k += HC_UP_LANES) {
      const auto x = input[k];
      const auto w = weight[k];
#pragma unroll
      for (int component = 0; component < HC_UP_VECTOR; ++component)
        sum = fmaf(hc_f(x.x[component]), hc_f(w.x[component]), sum);
    }
#pragma unroll
    for (int offset = HC_UP_LANES / 2; offset > 0; offset /= 2)
      sum += __shfl_xor_sync(__activemask(), sum, offset, HC_UP_LANES);
    if (lane == 0)
      mixed += hc_f(xn[(size_t)token * hc * hidden + row]) *
               hc_sigmoid(hc_f(hc_t<T>(sum)));
  }
  if (lane == 0)
    out[(size_t)token * hidden + h] = hc_t<T>(mixed / hc);
}

template <typename T>
__global__ void
hc_norm_kernel(const T *__restrict__ x, const float *__restrict__ weight,
               T *__restrict__ out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const int stream = row % hc;
  const T *xr = x + row * hidden;
  const float *wr = weight + (size_t)stream * hidden;
  T *orow = out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = hc_f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = hc_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = hc_t<T>(hc_f(xr[i]) * inv * wr[i]);
  }
}

template <typename T>
__global__ void hc_mix_kernel(const T *__restrict__ xn,
                              const T *__restrict__ gate, T *__restrict__ out,
                              size_t total, int hc, int hidden) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const size_t t = idx / hidden;
  const int i = idx - t * hidden;
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    const size_t o = (t * hc + c) * hidden + i;
    acc = fmaf(hc_f(xn[o]), hc_sigmoid(hc_f(gate[o])), acc);
  }
  out[idx] = hc_t<T>(acc / (float)hc);
}

// next_weight enables grouped RMSNorm of the rounded residual for the next
// mixer.
template <typename T, bool STRIDED>
__global__ void
hc_combine_kernel(const T *__restrict__ res, const T *__restrict__ block_out,
                  const T *__restrict__ inject, T *__restrict__ res_out,
                  const float *__restrict__ next_weight, T *__restrict__ xn_out,
                  size_t inject_stride, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const size_t t = row / hc;
  const int stream = row - t * hc;
  const size_t inject_index = STRIDED ? t * inject_stride + stream : row;
  const float w = 2.0f * hc_sigmoid(hc_f(inject[inject_index]) / (float)hc);
  const T *rr = res + row * hidden;
  const T *br = block_out + t * hidden;
  T *orow = res_out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const T v = hc_t<T>(fmaf(hc_f(br[i]), w, hc_f(rr[i])));
    orow[i] = v;
    const float vf = hc_f(v);
    ss = fmaf(vf, vf, ss);
  }
  if (next_weight == nullptr) {
    return;
  }
  ss = hc_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  const float *wr = next_weight + (size_t)stream * hidden;
  T *xr = xn_out + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    xr[i] = hc_t<T>(hc_f(orow[i]) * inv * wr[i]);
  }
}

__device__ __forceinline__ float hc_combine_sum(float value, float *shared) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = hc_warp_sum(value);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < blockDim.x / 32 ? shared[lane] : 0.0f;
    value = hc_warp_sum(value);
    if (lane == 0)
      shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

template <typename T, int V, bool STRIDED>
__global__ void
hc_combine_vector_kernel(const T *__restrict__ res, const T *__restrict__ block,
                         const T *__restrict__ inject, T *__restrict__ output,
                         const float *__restrict__ weight,
                         T *__restrict__ normed, size_t stride, int streams,
                         int hidden, float epsilon) {
  __shared__ float shared[32];
  constexpr int ITEMS = HC_COMBINE_RETAINED_VALUES / V;
  const size_t row = blockIdx.x, token = row / streams;
  const int stream = row - token * streams;
  const float gate =
      2.0f * hc_sigmoid(hc_f(inject[STRIDED ? token * stride + stream : row]) /
                        streams);
  float retained[ITEMS][V];
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int index = (threadIdx.x + item * blockDim.x) * V;
    if (index < hidden) {
      const auto residual =
          reinterpret_cast<const HcPack<T, V> *>(res + row * hidden)[index / V];
      const auto value = reinterpret_cast<const HcPack<T, V> *>(
          block + token * hidden)[index / V];
      HcPack<T, V> combined;
#pragma unroll
      for (int component = 0; component < V; ++component) {
        combined.x[component] = hc_t<T>(
            fmaf(hc_f(value.x[component]), gate, hc_f(residual.x[component])));
        retained[item][component] = hc_f(combined.x[component]);
        sum = fmaf(retained[item][component], retained[item][component], sum);
      }
      reinterpret_cast<HcPack<T, V> *>(output + row * hidden)[index / V] =
          combined;
    }
  }
  if (weight == nullptr)
    return;
  const float inv = rsqrtf(hc_combine_sum(sum, shared) / hidden + epsilon);
#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int index = (threadIdx.x + item * blockDim.x) * V;
    if (index < hidden) {
      const auto scale = reinterpret_cast<const HcPack<float, V> *>(
          weight + (size_t)stream * hidden)[index / V];
      HcPack<T, V> normalized;
#pragma unroll
      for (int component = 0; component < V; ++component)
        normalized.x[component] =
            hc_t<T>(retained[item][component] * inv * scale.x[component]);
      reinterpret_cast<HcPack<T, V> *>(normed + row * hidden)[index / V] =
          normalized;
    }
  }
}

#define HC_DISPATCH(dtype, ...)                                                \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else if ((dtype) == 1) {                                                 \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = float;                                                  \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

#define HC_DISPATCH_HALF(dtype, ...)                                           \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

static inline int hc_row_threads(int hidden) {
  int threads = 32;
  while (threads < 512 && threads * 4 < hidden) {
    threads <<= 1;
  }
  return threads;
}

extern "C" void mistral_hc_up_mix(const void *lo, const void *up,
                                  const void *xn, void *out, int tokens,
                                  int hidden, int rank, int hc, int format,
                                  int dtype, int multiprocessors,
                                  int64_t stream) {
  const cudaStream_t s = (cudaStream_t)stream;
  const bool vector =
      rank >= HC_UP_VECTOR * HC_UP_LANES && rank % HC_UP_VECTOR == 0 &&
      hidden >= multiprocessors * (HC_UP_THREADS / HC_UP_LANES) &&
      (reinterpret_cast<uintptr_t>(lo) % (HC_UP_VECTOR * sizeof(__half))) ==
          0 &&
      (reinterpret_cast<uintptr_t>(up) % (HC_UP_VECTOR * sizeof(__half))) == 0;
  if (vector) {
    const dim3 vector_grid((hidden + HC_UP_THREADS / HC_UP_LANES - 1) /
                               (HC_UP_THREADS / HC_UP_LANES),
                           tokens);
#define HC_UP_VECTOR_LAUNCH(FORMAT)                                            \
  hc_up_mix_vector_kernel<scalar_t, FORMAT>                                    \
      <<<vector_grid, HC_UP_THREADS, 0, s>>>(                                  \
          (const scalar_t *)lo, up, (const scalar_t *)xn, (scalar_t *)out,     \
          hidden, rank, hc)
    HC_DISPATCH_HALF(dtype, {
      if (format == 1) {
        HC_UP_VECTOR_LAUNCH(1);
      } else {
        HC_UP_VECTOR_LAUNCH(0);
      }
    });
#undef HC_UP_VECTOR_LAUNCH
    return;
  }
  const dim3 grid((hidden + HC_THREADS / HC_WARP_SIZE - 1) /
                      (HC_THREADS / HC_WARP_SIZE),
                  tokens);
#define HC_UP(FORMAT)                                                          \
  hc_up_mix_kernel<scalar_t, FORMAT><<<grid, HC_THREADS, 0, s>>>(              \
      (const scalar_t *)lo, up, (const scalar_t *)xn, (scalar_t *)out, hidden, \
      rank, hc)
  HC_DISPATCH_HALF(dtype, {
    if (format == 1) {
      HC_UP(1);
    } else {
      HC_UP(0);
    }
  });
#undef HC_UP
}

extern "C" void mistral_hc_silu(const void *input, void *output,
                                size_t elements, size_t columns,
                                size_t row_stride, int hc, int dtype,
                                int64_t stream) {
  const int threads = 256;
#define HC_SILU(STRIDED)                                                       \
  hc_silu_kernel<scalar_t, STRIDED>                                            \
      <<<(elements + threads - 1) / threads, threads, 0,                       \
         (cudaStream_t)stream>>>((const scalar_t *)input, (scalar_t *)output,  \
                                 elements, columns, row_stride, hc)
  HC_DISPATCH(dtype, {
    if (columns == row_stride) {
      HC_SILU(false);
    } else {
      HC_SILU(true);
    }
  });
#undef HC_SILU
}

extern "C" void mistral_hc_norm(const void *x, const float *weight, void *out,
                                int rows, int hc, int hidden, float eps,
                                int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  HC_DISPATCH(dtype, hc_norm_kernel<scalar_t>
              <<<rows, hc_row_threads(hidden), 0, s>>>((const scalar_t *)x,
                                                       weight, (scalar_t *)out,
                                                       hc, hidden, eps));
}

extern "C" void mistral_hc_mix(const void *xn, const void *gate, void *out,
                               int tokens, int hc, int hidden, int dtype,
                               int64_t stream) {
  const size_t total = (size_t)tokens * hidden;
  if (total == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const unsigned int blocks = (total + threads - 1) / threads;
  HC_DISPATCH(dtype, hc_mix_kernel<scalar_t><<<blocks, threads, 0, s>>>(
                         (const scalar_t *)xn, (const scalar_t *)gate,
                         (scalar_t *)out, total, hc, hidden));
}

extern "C" void mistral_hc_combine(const void *res, const void *block_out,
                                   const void *inject, void *res_out,
                                   const float *next_weight, void *xn_out,
                                   size_t inject_stride, int rows, int hc,
                                   int hidden, float eps, int dtype,
                                   int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = hc_row_threads(hidden);
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(res) |
                             reinterpret_cast<uintptr_t>(block_out) |
                             reinterpret_cast<uintptr_t>(res_out) |
                             reinterpret_cast<uintptr_t>(next_weight) |
                             reinterpret_cast<uintptr_t>(xn_out);
  const bool vectorized = next_weight != nullptr &&
                          hidden >= HC_THREADS * HC_COMBINE_VECTOR &&
                          hidden % HC_COMBINE_VECTOR == 0 &&
                          hidden <= threads * HC_COMBINE_RETAINED_VALUES &&
                          pointers % HC_COMBINE_ALIGNMENT == 0;
#define HC_COMBINE(STRIDED)                                                    \
  do {                                                                         \
    if (vectorized) {                                                          \
      hc_combine_vector_kernel<scalar_t, HC_COMBINE_VECTOR, STRIDED>           \
          <<<rows, threads, 0, s>>>(                                           \
              (const scalar_t *)res, (const scalar_t *)block_out,              \
              (const scalar_t *)inject, (scalar_t *)res_out, next_weight,      \
              (scalar_t *)xn_out, inject_stride, hc, hidden, eps);             \
    } else {                                                                   \
      hc_combine_kernel<scalar_t, STRIDED><<<rows, threads, 0, s>>>(           \
          (const scalar_t *)res, (const scalar_t *)block_out,                  \
          (const scalar_t *)inject, (scalar_t *)res_out, next_weight,          \
          (scalar_t *)xn_out, inject_stride, hc, hidden, eps);                 \
    }                                                                          \
  } while (0)
  HC_DISPATCH(dtype, {
    if (inject_stride == (size_t)hc) {
      HC_COMBINE(false);
    } else {
      HC_COMBINE(true);
    }
  });
#undef HC_COMBINE
}
