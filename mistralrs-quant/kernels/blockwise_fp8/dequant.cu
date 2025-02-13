#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "cuda_fp8.h"

// CUDA kernel for blockwise dequantization.
// - weight: pointer to the input F8E4rows3 matrix (size rows x cols)
// - scale: pointer to the scale matrix (size grid_y x grid_x)
// - out: pointer to the output array (size rows x cols)
// - rows, cols: dimensions of the weight matrix
// - block_size_rows, block_size_cols: size of each block in the weight matrix
template <typename T>
__global__ void fp8_blockwise_dequantize_kernel(
    const __nv_fp8_e4m3* __restrict__ weight, 
    const float* __restrict__ scale,
    T* __restrict__ out,
    int rows, int cols, 
    int block_size_rows, int block_size_cols
) {
    // Each CUDA block corresponds to a block of the weight matrix.
    // threadIdx.x indexes the column within the block,
    // threadIdx.y indexes the row.
    int local_y = threadIdx.y;
    int local_x = threadIdx.x;
    // Block indices in the grid correspond to the block’s position.
    int block_y = blockIdx.y;
    int block_x = blockIdx.x;
    // Compute the global coordinates in the weight matrix.
    int global_y = block_y * block_size_rows + local_y;
    int global_x = block_x * block_size_cols + local_x;
    
    // Check bounds for matrices whose dimensions might not be an exact multiple of the block size.
    if (global_y < rows && global_x < cols) {
        // Compute the linear index for the weight element.
        int weight_pos = global_y * cols + global_x;
        // Each block uses a single scale factor; scale is assumed to be stored with a known stride.
        float s = scale[block_y * block_size_cols + block_x];
        // Convert the FP8 value to float and multiply by the scale.
        float weight_val = float(weight[weight_pos]);
        T result = static_cast<T>(weight_val * s);
        out[weight_pos] = result;
    }
}

template <typename T>
void launch_fp8_blockwise_dequantize(
    const __nv_fp8_e4m3* d_weight,
    const float* d_scale,
    T* d_out,
    int rows, int cols,
    int block_size_rows, int block_size_cols
) {
    // Each CUDA block covers one block from the original computation.
    // Block dimensions: (block_size_cols, block_size_rows)
    dim3 blockDim(block_size_cols, block_size_rows);
    // Grid dimensions: number of blocks needed along each axis.
    int grid_x = (rows + block_size_cols - 1) / block_size_cols;
    int grid_y = (cols + block_size_rows - 1) / block_size_rows;
    dim3 gridDim(grid_x, grid_y);

    // Launch the kernel.
    fp8_blockwise_dequantize_kernel<T><<<gridDim, blockDim>>>(
        d_weight, d_scale, d_out,
        rows, cols, block_size_rows, block_size_cols
    );

    return;
}

#define FP8_BLOCKWISE_DEQUANT(TYPENAME, RUST_NAME) \
extern "C" void launch_fp8_blockwise_dequantize_##RUST_NAME( \
    const void* d_weight, \
    const float* d_scale, \
    TYPENAME* d_out, \
    int rows, int cols, \
    int block_size_rows, int block_size_cols \
) { \
    const __nv_fp8_e4m3* weight = reinterpret_cast<const __nv_fp8_e4m3*>(d_weight);\
    launch_fp8_blockwise_dequantize<TYPENAME>(weight, d_scale, d_out, rows, cols, block_size_rows, block_size_cols); \
}

FP8_BLOCKWISE_DEQUANT(float, f32)
FP8_BLOCKWISE_DEQUANT(__nv_bfloat16, bf16)
FP8_BLOCKWISE_DEQUANT(__half, f16)

