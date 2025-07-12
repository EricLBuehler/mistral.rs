#include <cassert>
#include <cstdint>
#include <cuda.h>

extern "C" void launch_gemm_fp8_blockwise_kernel_f16(
    const uint8_t* weight, const float* scale, const uint16_t* input,
    uint16_t* output, int batch, int in_dim, int out_dim,
    int weight_row_stride, int scale_stride, int block_y, int block_x) {
  assert(false);
}

extern "C" void launch_gemm_fp8_blockwise_kernel_bf16(
    const uint8_t* weight, const float* scale, const uint16_t* input,
    uint16_t* output, int batch, int in_dim, int out_dim,
    int weight_row_stride, int scale_stride, int block_y, int block_x) {
  assert(false);
}

extern "C" void launch_gemm_fp8_blockwise_kernel_f32(
    const uint8_t* weight, const float* scale, const float* input,
    float* output, int batch, int in_dim, int out_dim,
    int weight_row_stride, int scale_stride, int block_y, int block_x) {
  assert(false);
}

