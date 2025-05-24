#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(err); \
    } \
  } while (0)

namespace {

template <typename T> __device__ inline float to_float(T v);
template <> __device__ inline float to_float(__half v) { return __half2float(v); }
template <> __device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ inline float to_float(float v) { return v; }

template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline __half from_float(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
template <> __device__ inline float from_float(float v) { return v; }

template <typename T>
__global__ void gemm_fp8_blockwise_kernel(
    const __nv_fp8_e4m3* __restrict__ weight,
    const float* __restrict__ scale,
    const T* __restrict__ input,
    T* __restrict__ output,
    int batch,
    int in_dim,
    int out_dim,
    int weight_row_stride,
    int scale_stride,
    int block_y,
    int block_x) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= batch || col >= out_dim) return;

  float acc = 0.f;
  for (int k = 0; k < in_dim; ++k) {
    float x_val = to_float(input[row * in_dim + k]);
    float w_val = __half2float(__nv_cvt_fp8_to_halfraw(weight[col * weight_row_stride + k].__x, __NV_E4M3));
    float s = scale[(col / block_y) * scale_stride + (k / block_x)];
    acc += x_val * (w_val * s);
  }
  output[row * out_dim + col] = from_float<T>(acc);
}

} // namespace

extern "C" void launch_gemm_fp8_blockwise_kernel_f16(
    const __nv_fp8_e4m3* weight,
    const float* scale,
    const __half* input,
    __half* output,
    int batch,
    int in_dim,
    int out_dim,
    int weight_row_stride,
    int scale_stride,
    int block_y,
    int block_x,
    cudaStream_t stream) {
  dim3 blockDim(16,16);
  dim3 gridDim((out_dim + blockDim.x - 1)/blockDim.x,
               (batch + blockDim.y - 1)/blockDim.y);
  gemm_fp8_blockwise_kernel<<<gridDim, blockDim, 0, stream>>>(
      weight, scale, input, output, batch, in_dim, out_dim,
      weight_row_stride, scale_stride, block_y, block_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_gemm_fp8_blockwise_kernel_bf16(
    const __nv_fp8_e4m3* weight,
    const float* scale,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch,
    int in_dim,
    int out_dim,
    int weight_row_stride,
    int scale_stride,
    int block_y,
    int block_x,
    cudaStream_t stream) {
  dim3 blockDim(16,16);
  dim3 gridDim((out_dim + blockDim.x - 1)/blockDim.x,
               (batch + blockDim.y - 1)/blockDim.y);
  gemm_fp8_blockwise_kernel<<<gridDim, blockDim, 0, stream>>>(
      weight, scale, input, output, batch, in_dim, out_dim,
      weight_row_stride, scale_stride, block_y, block_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_gemm_fp8_blockwise_kernel_f32(
    const __nv_fp8_e4m3* weight,
    const float* scale,
    const float* input,
    float* output,
    int batch,
    int in_dim,
    int out_dim,
    int weight_row_stride,
    int scale_stride,
    int block_y,
    int block_x,
    cudaStream_t stream) {
  dim3 blockDim(16,16);
  dim3 gridDim((out_dim + blockDim.x - 1)/blockDim.x,
               (batch + blockDim.y - 1)/blockDim.y);
  gemm_fp8_blockwise_kernel<<<gridDim, blockDim, 0, stream>>>(
      weight, scale, input, output, batch, in_dim, out_dim,
      weight_row_stride, scale_stride, block_y, block_x);
  CUDA_CHECK(cudaGetLastError());
}


