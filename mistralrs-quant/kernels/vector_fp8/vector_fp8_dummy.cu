#include <cstdint>
#include <stdio.h>

// Dummy implementations when CUDA is not available or FP8 is not supported

extern "C" void launch_dequant_fp8_vector_kernel_f32(const void *d_weight,
                                                     const float *d_scale,
                                                     float *d_output,
                                                     size_t num_elements,
                                                     void *stream) {
  fprintf(
      stderr,
      "FP8 vector dequantization kernels are not available in this build.\n");
}

extern "C" void launch_dequant_fp8_vector_kernel_f16(const void *d_weight,
                                                     const float *d_scale,
                                                     void *d_output,
                                                     size_t num_elements,
                                                     void *stream) {
  fprintf(
      stderr,
      "FP8 vector dequantization kernels are not available in this build.\n");
}

extern "C" void launch_dequant_fp8_vector_kernel_bf16(const void *d_weight,
                                                      const float *d_scale,
                                                      void *d_output,
                                                      size_t num_elements,
                                                      void *stream) {
  fprintf(
      stderr,
      "FP8 vector dequantization kernels are not available in this build.\n");
}

extern "C" void launch_quant_fp8_vector_kernel_f32(const float *d_input,
                                                   void *d_weight,
                                                   float *d_scale,
                                                   size_t num_elements,
                                                   void *stream) {
  fprintf(stderr,
          "FP8 vector quantization kernels are not available in this build.\n");
}

extern "C" void launch_quant_fp8_vector_kernel_f16(const void *d_input,
                                                   void *d_weight,
                                                   float *d_scale,
                                                   size_t num_elements,
                                                   void *stream) {
  fprintf(stderr,
          "FP8 vector quantization kernels are not available in this build.\n");
}

extern "C" void launch_quant_fp8_vector_kernel_bf16(const void *d_input,
                                                    void *d_weight,
                                                    float *d_scale,
                                                    size_t num_elements,
                                                    void *stream) {
  fprintf(stderr,
          "FP8 vector quantization kernels are not available in this build.\n");
}