#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mistralrs_cutlass_fp8_output_dtype {
  MISTRALRS_CUTLASS_FP8_OUTPUT_F16 = 0,
  MISTRALRS_CUTLASS_FP8_OUTPUT_BF16 = 1,
};

const char *mistralrs_cutlass_fp8_error_string(int status);
int mistralrs_cutlass_fp8_blockwise_prepare(void);
int mistralrs_cutlass_fp8_blockwise_workspace_size(
    int m, int n, int k, int output_dtype, int sm_count, size_t *bytes);
int mistralrs_cutlass_fp8_blockwise_gemm(
    const void *activation, const void *weight,
    const float *activation_scales, const float *weight_scales, void *output,
    int m, int n, int k, int output_dtype, void *workspace,
    size_t workspace_bytes, int sm_count, void *stream);
int mistralrs_fp8_quantize_activation_f16(const void *input, void *output,
                                          float *scales, int rows, int cols,
                                          void *stream);
int mistralrs_fp8_quantize_activation_bf16(const void *input, void *output,
                                           float *scales, int rows, int cols,
                                           void *stream);

#ifdef __cplusplus
}
#endif
