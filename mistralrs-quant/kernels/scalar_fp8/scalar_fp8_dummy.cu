#include <cassert>
#include <cstdint>
#include <cuda.h>

extern "C" void launch_fp8_to_f32_kernel(const uint8_t *d_input,
                                         float *d_output, size_t num_elements,
                                         cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_fp8_to_f16_kernel(const uint8_t *d_input,
                                         uint16_t *d_output,
                                         size_t num_elements,
                                         cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_fp8_to_bf16_kernel(const uint8_t *d_input,
                                          uint16_t *d_output,
                                          size_t num_elements,
                                          cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_f32_to_fp8_kernel(const float *d_input,
                                         uint8_t *d_output, size_t num_elements,
                                         cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_f16_to_fp8_kernel(const uint16_t *d_input,
                                         uint8_t *d_output, size_t num_elements,
                                         cudaStream_t stream) {
  assert(false);
}

extern "C" void launch_bf16_to_fp8_kernel(const uint16_t *d_input,
                                          uint8_t *d_output,
                                          size_t num_elements,
                                          cudaStream_t stream) {
  assert(false);
}