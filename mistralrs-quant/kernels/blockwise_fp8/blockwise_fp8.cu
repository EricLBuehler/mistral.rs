#include <cstdint>
#include <cuda.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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

  // Use threadIdx to cover elements in the tile.
  int local_y = threadIdx.y;
  int local_x = threadIdx.x;

  // Compute global indices.
  int weight_y = start_y + local_y;
  int weight_x = start_x + local_x;

  // Load the block's scale factor into shared memory.
  __shared__ float block_scale;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_scale = scale[grid_y * scale_stride + grid_x];
  }
  __syncthreads(); // Ensure all threads see the loaded value.

  // Bounds check: if within the dimensions of the weight matrix.
  if (weight_y < weight_height && weight_x < weight_width) {
    int pos = weight_y * weight_row_stride + weight_x;
    float w_val =
        __half2float(__nv_cvt_fp8_to_halfraw(weight[pos].__x, __NV_E4M3));
    // Use the shared scale factor.
    output[pos] = static_cast<T>(w_val * block_scale);
  }
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f32(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale, float *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x) {
  // Calculate grid dimensions.
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;

  // Set block dimensions to match the block size.
  dim3 blockDim(weight_block_size_x, weight_block_size_y);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<float>
      <<<gridDim, blockDim>>>(d_weight, d_scale, d_output, weight_height,
                              weight_width, weight_row_stride, scale_stride,
                              weight_block_size_y, weight_block_size_x);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f16(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale, __half *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x) {
  // Calculate grid dimensions.
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;

  // Set block dimensions to match the block size.
  dim3 blockDim(weight_block_size_x, weight_block_size_y);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<__half>
      <<<gridDim, blockDim>>>(d_weight, d_scale, d_output, weight_height,
                              weight_width, weight_row_stride, scale_stride,
                              weight_block_size_y, weight_block_size_x);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_bf16(
    const __nv_fp8_e4m3 *d_weight, const float *d_scale,
    __nv_bfloat16 *d_output, int weight_height, int weight_width,
    int weight_row_stride, int scale_stride, int weight_block_size_y,
    int weight_block_size_x) {
  // Calculate grid dimensions.
  int grid_y = (weight_height + weight_block_size_y - 1) / weight_block_size_y;
  int grid_x = (weight_width + weight_block_size_x - 1) / weight_block_size_x;

  // Set block dimensions to match the block size.
  dim3 blockDim(weight_block_size_x, weight_block_size_y);
  dim3 gridDim(grid_x, grid_y);

  dequant_fp8_blockwise_kernel<__nv_bfloat16>
      <<<gridDim, blockDim>>>(d_weight, d_scale, d_output, weight_height,
                              weight_width, weight_row_stride, scale_stride,
                              weight_block_size_y, weight_block_size_x);
}