#include <cassert>
#include <cstdint>
#include <cuda.h>

extern "C" void launch_dequant_fp8_blockwise_kernel_f32(
    const uint8_t *d_weight, const float *d_scale, float *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x) {
  assert(false);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_f16(
    const uint8_t *d_weight, const float *d_scale, uint16_t *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x) {
  assert(false);
}

extern "C" void launch_dequant_fp8_blockwise_kernel_bf16(
    const uint8_t *d_weight, const float *d_scale, uint16_t *d_output,
    int weight_height, int weight_width, int weight_row_stride,
    int scale_stride, int weight_block_size_y, int weight_block_size_x) {
  assert(false);
}