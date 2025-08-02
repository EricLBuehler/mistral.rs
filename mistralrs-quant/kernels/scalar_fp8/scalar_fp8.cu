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

template <typename T>
__global__ void fp8_to_dtype_kernel(const __nv_fp8_e4m3 *__restrict__ input,
                                    T *__restrict__ output,
                                    size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float val =
        __half2float(__nv_cvt_fp8_to_halfraw(input[idx].__x, __NV_E4M3));
    output[idx] = static_cast<T>(val);
  }
}

template <typename T>
__global__ void dtype_to_fp8_kernel(const T *__restrict__ input,
                                    __nv_fp8_e4m3 *__restrict__ output,
                                    size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    float val = static_cast<float>(input[idx]);
    // Clamp to FP8 E4M3 range
    if (val > 448.0f)
      val = 448.0f;
    if (val < -448.0f)
      val = -448.0f;
    __half h_val = __float2half(val);
    output[idx].__x = __nv_cvt_halfraw_to_fp8(h_val, __NV_SATFINITE, __NV_E4M3);
  }
}

extern "C" void launch_fp8_to_f32_kernel(const __nv_fp8_e4m3 *d_input,
                                         float *d_output, size_t num_elements,
                                         cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  fp8_to_dtype_kernel<float>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_to_f16_kernel(const __nv_fp8_e4m3 *d_input,
                                         __half *d_output, size_t num_elements,
                                         cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  fp8_to_dtype_kernel<__half>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_fp8_to_bf16_kernel(const __nv_fp8_e4m3 *d_input,
                                          __nv_bfloat16 *d_output,
                                          size_t num_elements,
                                          cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  fp8_to_dtype_kernel<__nv_bfloat16>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_f32_to_fp8_kernel(const float *d_input,
                                         __nv_fp8_e4m3 *d_output,
                                         size_t num_elements,
                                         cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  dtype_to_fp8_kernel<float>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_f16_to_fp8_kernel(const __half *d_input,
                                         __nv_fp8_e4m3 *d_output,
                                         size_t num_elements,
                                         cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  dtype_to_fp8_kernel<__half>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_bf16_to_fp8_kernel(const __nv_bfloat16 *d_input,
                                          __nv_fp8_e4m3 *d_output,
                                          size_t num_elements,
                                          cudaStream_t stream) {
  const int block_size = 256;
  const int num_blocks = (num_elements + block_size - 1) / block_size;

  dtype_to_fp8_kernel<__nv_bfloat16>
      <<<num_blocks, block_size, 0, stream>>>(d_input, d_output, num_elements);
  CUDA_CHECK(cudaGetLastError());
}