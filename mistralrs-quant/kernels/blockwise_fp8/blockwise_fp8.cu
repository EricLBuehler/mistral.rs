#include <cstdint>
#include <cuda.h>
#include <stdio.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

constexpr int FP8_ROWWISE_THREADS = 256;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_MIN_SCALE = 1e-12f;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

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
__global__ void dequant_fp8_blockwise_kernel(
    const __nv_fp8_e4m3 *__restrict__ weight, const float *__restrict__ scale,
    T *__restrict__ output, int weight_height, int weight_width,
    int weight_row_stride, int scale_stride, int weight_block_size_y,
    int weight_block_size_x) {
  // Each block corresponds to a tile.
  int grid_y = blockIdx.y; // tile row index
  int grid_x = blockIdx.x; // tile column index

  // Compute the starting indices for this tile.
  int start_y = grid_y * weight_block_size_y;
  int start_x = grid_x * weight_block_size_x;

  // Load the block's scale factor into shared memory.
  __shared__ float block_scale;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_scale = scale[grid_y * scale_stride + grid_x];
  }
  __syncthreads(); // Ensure all threads see the loaded value.

  // Loop over the tile using a fixed blockDim, covering the whole tile.
  for (int local_y = threadIdx.y; local_y < weight_block_size_y;
       local_y += blockDim.y) {
    for (int local_x = threadIdx.x; local_x < weight_block_size_x;
         local_x += blockDim.x) {
      int weight_y = start_y + local_y;
      int weight_x = start_x + local_x;
      if (weight_y < weight_height && weight_x < weight_width) {
        int pos = weight_y * weight_row_stride + weight_x;
        float w_val =
            __half2float(__nv_cvt_fp8_to_halfraw(weight[pos].__x, __NV_E4M3));
        output[pos] = static_cast<T>(w_val * block_scale);
      }
    }
  }
}

template <typename T>
__global__ void quant_fp8_blockwise_kernel(
    const T *__restrict__ input, __nv_fp8_e4m3 *__restrict__ weight,
    float *__restrict__ scale, int weight_height, int weight_width,
    int weight_row_stride, int scale_stride, int weight_block_size_y,
    int weight_block_size_x) {
  // Each block corresponds to a tile.
  int grid_y = blockIdx.y; // tile row index
  int grid_x = blockIdx.x; // tile column index

  // Compute the starting indices for this tile.
  int start_y = grid_y * weight_block_size_y;
  int start_x = grid_x * weight_block_size_x;

  // Find max absolute value in the block
  __shared__ float block_absmax;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_absmax = 0.0f;
  }
  __syncthreads();

  // First pass: find maximum absolute value in the block
  for (int local_y = threadIdx.y; local_y < weight_block_size_y;
       local_y += blockDim.y) {
    for (int local_x = threadIdx.x; local_x < weight_block_size_x;
         local_x += blockDim.x) {
      int weight_y = start_y + local_y;
      int weight_x = start_x + local_x;
      if (weight_y < weight_height && weight_x < weight_width) {
        int pos = weight_y * weight_row_stride + weight_x;
        float val = static_cast<float>(input[pos]);
        float absval = fabsf(val);
        // Use custom atomicMax for float
        atomicMaxFloat(&block_absmax, absval);
      }
    }
  }
  __syncthreads();

  // Calculate scale factor (FP8 E4M3 max value is 448.0)
  __shared__ float block_scale;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_scale = block_absmax / 448.0f;
    if (block_scale < 1e-12f)
      block_scale = 1e-12f; // Avoid division by zero
    scale[grid_y * scale_stride + grid_x] = block_scale;
  }
  __syncthreads();

  // Second pass: quantize values
  for (int local_y = threadIdx.y; local_y < weight_block_size_y;
       local_y += blockDim.y) {
    for (int local_x = threadIdx.x; local_x < weight_block_size_x;
         local_x += blockDim.x) {
      int weight_y = start_y + local_y;
      int weight_x = start_x + local_x;
      if (weight_y < weight_height && weight_x < weight_width) {
        int pos = weight_y * weight_row_stride + weight_x;
        float val = static_cast<float>(input[pos]);
        float scaled_val = val / block_scale;
        // Clamp to FP8 E4M3 range
        if (scaled_val > 448.0f)
          scaled_val = 448.0f;
        if (scaled_val < -448.0f)
          scaled_val = -448.0f;
        __half h_val = __float2half(scaled_val);
        weight[pos].__x =
            __nv_cvt_halfraw_to_fp8(h_val, __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
}

template <typename T>
__global__ void quant_fp8_rowwise_kernel(const T *__restrict__ input,
                                         __nv_fp8_e4m3 *__restrict__ output,
                                         float *__restrict__ scales, int rows,
                                         int columns, int row_stride) {
  int row = blockIdx.x;
  if (row >= rows)
    return;

  float local_max = 0.0f;
  size_t input_row = static_cast<size_t>(row) * row_stride;
  size_t output_row = static_cast<size_t>(row) * columns;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    local_max =
        fmaxf(local_max, fabsf(static_cast<float>(input[input_row + column])));
  }

  __shared__ float maxima[FP8_ROWWISE_THREADS];
  maxima[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = FP8_ROWWISE_THREADS / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    __syncthreads();
  }

  float scale = fmaxf(maxima[0] / FP8_E4M3_MAX, FP8_MIN_SCALE);
  if (threadIdx.x == 0)
    scales[row] = scale;
  for (int column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = static_cast<float>(input[input_row + column]) / scale;
    value = fminf(fmaxf(value, -FP8_E4M3_MAX), FP8_E4M3_MAX);
    output[output_row + column].__x =
        __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

template <typename T>
__global__ void quant_fp8_static_kernel(const T *__restrict__ input,
                                        const float *__restrict__ scale,
                                        __nv_fp8_e4m3 *__restrict__ output,
                                        size_t elements) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  float value = static_cast<float>(input[index]) / scale[0];
  value = fminf(fmaxf(value, -FP8_E4M3_MAX), FP8_E4M3_MAX);
  output[index].__x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f32(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale, float *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<float><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f16(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale, __half *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<__half><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_dequant_fp8_blockwise_kernel_bf16(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale,
    __nv_bfloat16 *d_output, int weight_height, int weight_width,
    int weight_row_stride, int scale_stride, int weight_block_size_y,
    int weight_block_size_x, cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      d_weight, d_scale, d_output, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_quant_fp8_blockwise_kernel_f32(
    const float *d_input, __nv_fp8_e4m3 *d_weight, float *d_scale,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  quant_fp8_blockwise_kernel<float><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_quant_fp8_blockwise_kernel_f16(
    const __half *d_input, __nv_fp8_e4m3 *d_weight, float *d_scale,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  quant_fp8_blockwise_kernel<__half><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void launch_quant_fp8_blockwise_kernel_bf16(
    const __nv_bfloat16 *d_input, __nv_fp8_e4m3 *d_weight, float *d_scale,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x,
    cudaStream_t stream) {
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;
  dim3 blockDim(32, 32);
  dim3 gridDim(grid_x, grid_y);

  quant_fp8_blockwise_kernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(
      d_input, d_weight, d_scale, weight_height, weight_width,
      weight_row_stride, scale_stride, weight_block_size_y,
      weight_block_size_x);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" int launch_quant_fp8_rowwise_kernel_f32(const float *input,
                                                   __nv_fp8_e4m3 *output,
                                                   float *scales, int rows,
                                                   int columns, int row_stride,
                                                   cudaStream_t stream) {
  quant_fp8_rowwise_kernel<float><<<rows, FP8_ROWWISE_THREADS, 0, stream>>>(
      input, output, scales, rows, columns, row_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_quant_fp8_rowwise_kernel_f16(const __half *input,
                                                   __nv_fp8_e4m3 *output,
                                                   float *scales, int rows,
                                                   int columns, int row_stride,
                                                   cudaStream_t stream) {
  quant_fp8_rowwise_kernel<__half><<<rows, FP8_ROWWISE_THREADS, 0, stream>>>(
      input, output, scales, rows, columns, row_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_quant_fp8_rowwise_kernel_bf16(const __nv_bfloat16 *input,
                                                    __nv_fp8_e4m3 *output,
                                                    float *scales, int rows,
                                                    int columns, int row_stride,
                                                    cudaStream_t stream) {
  quant_fp8_rowwise_kernel<__nv_bfloat16>
      <<<rows, FP8_ROWWISE_THREADS, 0, stream>>>(input, output, scales, rows,
                                                 columns, row_stride);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_quant_fp8_static_kernel_f32(const float *input,
                                                  const float *scale,
                                                  __nv_fp8_e4m3 *output,
                                                  size_t elements,
                                                  cudaStream_t stream) {
  int blocks = (elements + FP8_ROWWISE_THREADS - 1) / FP8_ROWWISE_THREADS;
  quant_fp8_static_kernel<float><<<blocks, FP8_ROWWISE_THREADS, 0, stream>>>(
      input, scale, output, elements);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_quant_fp8_static_kernel_f16(const __half *input,
                                                  const float *scale,
                                                  __nv_fp8_e4m3 *output,
                                                  size_t elements,
                                                  cudaStream_t stream) {
  int blocks = (elements + FP8_ROWWISE_THREADS - 1) / FP8_ROWWISE_THREADS;
  quant_fp8_static_kernel<__half><<<blocks, FP8_ROWWISE_THREADS, 0, stream>>>(
      input, scale, output, elements);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_quant_fp8_static_kernel_bf16(const __nv_bfloat16 *input,
                                                   const float *scale,
                                                   __nv_fp8_e4m3 *output,
                                                   size_t elements,
                                                   cudaStream_t stream) {
  int blocks = (elements + FP8_ROWWISE_THREADS - 1) / FP8_ROWWISE_THREADS;
  quant_fp8_static_kernel<__nv_bfloat16>
      <<<blocks, FP8_ROWWISE_THREADS, 0, stream>>>(input, scale, output,
                                                   elements);
  return static_cast<int>(cudaGetLastError());
}
