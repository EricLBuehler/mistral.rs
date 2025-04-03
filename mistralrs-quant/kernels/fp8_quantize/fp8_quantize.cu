#include <assert.h>
#include <cmath>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename T>
__global__ void
apply_scalar_quantize(const T *d_in, __nv_fp8_e4m3 *d_out,
                      const uint32_t elem_count, const float *abs_max,
                      const float fp8_max_val, const float fp8_min_val) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elem_count) {
    const float scale =
        MIN(MAX(fp8_max_val / abs_max[0], fp8_min_val), fp8_max_val);
    d_out[idx] = static_cast<__nv_fp8_e4m3>((float)scale * (float)d_in[idx]);
  }
}

// Generic AbsFunctor (will be specialized for each type)
template <typename T> struct AbsFunctor {
  __host__ __device__ float operator()(const T &x) const {
    return fabsf(static_cast<float>(x));
  }
};

// Specialization for float
template <> struct AbsFunctor<float> {
  __host__ __device__ float operator()(const float &x) const {
    return fabsf(x);
  }
};

// Specialization for __half
template <> struct AbsFunctor<__half> {
  __host__ __device__ float operator()(const __half &x) const {
    return fabsf(__half2float(x));
  }
};

// Specialization for __nv_bfloat16
template <> struct AbsFunctor<__nv_bfloat16> {
  __host__ __device__ float operator()(const __nv_bfloat16 &x) const {
    return fabsf(__bfloat162float(x));
  }
};

template <typename T>
void quantize_scalar_fp8(const T *d_in, __nv_fp8_e4m3 *d_out, float *s_out,
                         const uint32_t elem_count, cudaStream_t stream) {
  // Allocate device memory for the maximum output
  float *abs_max = nullptr;
  CUDA_CHECK(cudaMallocAsync((void **)&abs_max, sizeof(float), stream));

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Create a transform iterator using our specialized functors
  using TransformIter =
      cub::TransformInputIterator<float, AbsFunctor<T>, const T *>;
  TransformIter d_in_abs(d_in, AbsFunctor<T>());

  // First call: get temporary storage requirements
  cub::DeviceReduce::Max(nullptr, temp_storage_bytes, d_in_abs, abs_max,
                         elem_count, stream);
  CUDA_CHECK(cudaGetLastError());

  // Allocate temporary storage on device
  CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  // Second call: perform the max reduction
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in_abs, abs_max,
                         elem_count, stream);
  CUDA_CHECK(cudaGetLastError());

  const float fp8_max_val =
      static_cast<float>(std::numeric_limits<__nv_fp8_e4m3>::max());
  const float fp8_min_val =
      static_cast<float>(std::numeric_limits<__nv_fp8_e4m3>::min());
  const uint32_t nthreads = 1024;
  const int nblocks = (elem_count + nthreads - 1) / nthreads;
  apply_scalar_quantize<T><<<nblocks, nthreads, 0, stream>>>(
      d_in, d_out, elem_count, abs_max, fp8_max_val, fp8_min_val);
  CUDA_CHECK(cudaGetLastError());

  // CUDA_CHECK(cudaMemcpyAsync((void *)s_out, (void *)abs_max, sizeof(float),
  //                            cudaMemcpyDeviceToDevice, stream));
  
  float test_value = 2;
  CUDA_CHECK(cudaMemcpyAsync((void *)s_out, (void *)&test_value, sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  // Clean up
  CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
  CUDA_CHECK(cudaFreeAsync(abs_max, stream));
  CUDA_CHECK(cudaGetLastError());
}

#define QUANTIZE_SCALAR(TYPENAME, RUST_NAME)                                   \
  extern "C" void quantize_scalar_fp8_##RUST_NAME(                             \
      const TYPENAME *d_in, __nv_fp8_e4m3 *d_out, float *s_out,                \
      const uint32_t elem_count, cudaStream_t stream) {                        \
    quantize_scalar_fp8<TYPENAME>(d_in, d_out, s_out, elem_count, stream);     \
  }

QUANTIZE_SCALAR(float, f32)
QUANTIZE_SCALAR(__nv_bfloat16, bf16)
QUANTIZE_SCALAR(__half, f16)