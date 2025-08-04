#include <cstdint>
#include <cuda.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

#define VECTOR_SIZE 128

// Custom atomicMax for float
__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

template <typename T>
__global__ void
dequant_fp8_vector_kernel(const __nv_fp8_e4m3 *__restrict__ weight,
                          const float *__restrict__ scale,
                          T *__restrict__ output, size_t num_elements) {
  // Each thread block handles one vector (128 elements)
  size_t vector_idx = blockIdx.x;
  size_t thread_idx = threadIdx.x;
  size_t vectors_per_row = gridDim.x;

  // Calculate starting position for this vector
  size_t vector_start = vector_idx * VECTOR_SIZE;

  // Load the scale for this vector
  float vector_scale = scale[vector_idx];

  // Each thread handles multiple elements within the vector
  for (size_t i = thread_idx;
       i < VECTOR_SIZE && (vector_start + i) < num_elements; i += blockDim.x) {
    size_t global_idx = vector_start + i;
    float w_val = __half2float(
        __nv_cvt_fp8_to_halfraw(weight[global_idx].__x, __NV_E4M3));
    output[global_idx] = static_cast<T>(w_val * vector_scale);
  }
}

template <typename T>
__global__ void quant_fp8_vector_kernel(const T *__restrict__ input,
                                        __nv_fp8_e4m3 *__restrict__ weight,
                                        float *__restrict__ scale,
                                        size_t num_elements) {
  // Each thread block handles one vector (128 elements)
  size_t vector_idx = blockIdx.x;
  size_t thread_idx = threadIdx.x;

  // Calculate starting position for this vector
  size_t vector_start = vector_idx * VECTOR_SIZE;

  // Shared memory for finding max in the vector
  __shared__ float vector_absmax;

  if (thread_idx == 0) {
    vector_absmax = 0.0f;
  }
  __syncthreads();

  // First pass: find maximum absolute value in the vector
  for (size_t i = thread_idx;
       i < VECTOR_SIZE && (vector_start + i) < num_elements; i += blockDim.x) {
    size_t global_idx = vector_start + i;
    float val = static_cast<float>(input[global_idx]);
    float absval = fabsf(val);
    atomicMaxFloat(&vector_absmax, absval);
  }
  __syncthreads();

  // Calculate scale factor
  __shared__ float vector_scale;
  if (thread_idx == 0) {
    vector_scale = vector_absmax / 448.0f;
    if (vector_scale < 1e-12f)
      vector_scale = 1e-12f; // Avoid division by zero
    scale[vector_idx] = vector_scale;
  }
  __syncthreads();

  // Second pass: quantize values
  for (size_t i = thread_idx;
       i < VECTOR_SIZE && (vector_start + i) < num_elements; i += blockDim.x) {
    size_t global_idx = vector_start + i;
    float val = static_cast<float>(input[global_idx]);
    float scaled_val = val / vector_scale;
    // Clamp to FP8 E4M3 range
    if (scaled_val > 448.0f)
      scaled_val = 448.0f;
    if (scaled_val < -448.0f)
      scaled_val = -448.0f;
    __half h_val = __float2half(scaled_val);
    weight[global_idx].__x =
        __nv_cvt_halfraw_to_fp8(h_val, __NV_SATFINITE, __NV_E4M3);
  }
}

// Dequantization kernels
extern "C" void
launch_dequant_fp8_vector_kernel_f32(const __nv_fp8_e4m3 *d_weight,
                                     const float *d_scale, float *d_output,
                                     size_t num_elements, cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  dequant_fp8_vector_kernel<float><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_dequant_fp8_vector_kernel_f16(const __nv_fp8_e4m3 *d_weight,
                                     const float *d_scale, __half *d_output,
                                     size_t num_elements, cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  dequant_fp8_vector_kernel<__half><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_dequant_fp8_vector_kernel_bf16(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale,
    __nv_bfloat16 *d_output, size_t num_elements, cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  dequant_fp8_vector_kernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

// Quantization kernels
extern "C" void launch_quant_fp8_vector_kernel_f32(const float *d_input,
                                                   __nv_fp8_e4m3 *d_weight,
                                                   float *d_scale,
                                                   size_t num_elements,
                                                   cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  quant_fp8_vector_kernel<float><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_quant_fp8_vector_kernel_f16(const __half *d_input,
                                                   __nv_fp8_e4m3 *d_weight,
                                                   float *d_scale,
                                                   size_t num_elements,
                                                   cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  quant_fp8_vector_kernel<__half><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void
launch_quant_fp8_vector_kernel_bf16(const __nv_bfloat16 *d_input,
                                    __nv_fp8_e4m3 *d_weight, float *d_scale,
                                    size_t num_elements, cudaStream_t stream) {
  size_t num_vectors = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
  dim3 blockDim(256);
  dim3 gridDim(num_vectors);

  quant_fp8_vector_kernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, num_elements);
  CUDA_CHECK(cudaGetLastError());
}