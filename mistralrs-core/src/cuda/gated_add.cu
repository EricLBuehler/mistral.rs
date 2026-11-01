#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int GATED_ADD_THREADS = 256;

extern "C" __device__ float __nv_expf(float);

// Preserve Candle's dtype rounding under fast math, including the F32 sigmoid's double reciprocal.
__device__ __forceinline__ float exact_add(float a, float b) {
  float out;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float exact_mul(float a, float b) {
  float out;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float exact_div(float a, float b) {
  float out;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ float exact_float(double value) {
  float out;
  asm("cvt.rn.f32.f64 %0, %1;" : "=f"(out) : "d"(value));
  return out;
}

template <typename T> __device__ __forceinline__ T rounded_sigmoid(T value);

template <>
__device__ __forceinline__ __nv_bfloat16 rounded_sigmoid(__nv_bfloat16 value) {
  return __float2bfloat16_rn(
      exact_div(1.0f, exact_add(1.0f, __nv_expf(-__bfloat162float(value)))));
}

template <> __device__ __forceinline__ __half rounded_sigmoid(__half value) {
  return __hdiv(__float2half_rn(1.0f),
                __hadd_rn(__float2half_rn(1.0f), hexp(__hneg(value))));
}

template <> __device__ __forceinline__ float rounded_sigmoid(float value) {
  return exact_float(
      __ddiv_rn(1.0, static_cast<double>(exact_add(1.0f, __nv_expf(-value)))));
}

template <typename T> __device__ __forceinline__ T rounded_mul(T a, T b) {
  return __hmul_rn(a, b);
}
template <typename T> __device__ __forceinline__ T rounded_add(T a, T b) {
  return __hadd_rn(a, b);
}
template <> __device__ __forceinline__ float rounded_mul(float a, float b) {
  return exact_mul(a, b);
}
template <> __device__ __forceinline__ float rounded_add(float a, float b) {
  return exact_add(a, b);
}

template <typename T>
__global__ void sigmoid_gated_add(const T *residual, const T *value,
                                  const T *gate, T *output, size_t elements,
                                  size_t columns) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const T raw = gate[index / columns];
    const T factor = rounded_sigmoid(raw);
    output[index] =
        rounded_add(residual[index], rounded_mul(value[index], factor));
  }
}

template <typename T>
__global__ void sigmoid_gated_add_rows(const T *residual, const T *value,
                                       const T *gate, T *output,
                                       size_t columns) {
  const size_t tiles = (columns + GATED_ADD_THREADS - 1) / GATED_ADD_THREADS;
  const size_t row = blockIdx.x / tiles;
  const size_t column = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
  __shared__ T factor;
  if (threadIdx.x == 0) {
    factor = rounded_sigmoid(gate[row]);
  }
  __syncthreads();
  if (column < columns) {
    const size_t index = row * columns + column;
    output[index] =
        rounded_add(residual[index], rounded_mul(value[index], factor));
  }
}

extern "C" void launch_shared_gated_add(const void *residual, const void *value,
                                        const void *gate, void *output,
                                        size_t elements, size_t columns,
                                        int dtype, void *stream) {
  const int blocks = (elements + GATED_ADD_THREADS - 1) / GATED_ADD_THREADS;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (dtype == 0 && columns >= GATED_ADD_THREADS) {
    const int row_blocks =
        (elements / columns) *
        ((columns + GATED_ADD_THREADS - 1) / GATED_ADD_THREADS);
    sigmoid_gated_add_rows<float><<<row_blocks, GATED_ADD_THREADS, 0, s>>>(
        static_cast<const float *>(residual), static_cast<const float *>(value),
        static_cast<const float *>(gate), static_cast<float *>(output),
        columns);
    return;
  }
#define RUN(T)                                                                 \
  sigmoid_gated_add<T><<<blocks, GATED_ADD_THREADS, 0, s>>>(                   \
      static_cast<const T *>(residual), static_cast<const T *>(value),         \
      static_cast<const T *>(gate), static_cast<T *>(output), elements,        \
      columns)
  if (dtype == 0) {
    RUN(float);
  } else if (dtype == 1) {
    RUN(__half);
  } else {
    RUN(__nv_bfloat16);
  }
#undef RUN
}
