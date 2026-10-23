#ifndef MISTRALRS_GGUF_AFFINE_PACKED_H
#define MISTRALRS_GGUF_AFFINE_PACKED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  MRS_GGUF_AFFINE_SUCCESS = 0,
  MRS_GGUF_AFFINE_INVALID_ARGUMENT = -1,
};

enum {
  MRS_GGUF_AFFINE_Q4_0 = 2,
  MRS_GGUF_AFFINE_Q4_1 = 3,
  MRS_GGUF_AFFINE_Q5_0 = 6,
  MRS_GGUF_AFFINE_Q5_1 = 7,
  MRS_GGUF_AFFINE_Q8_0 = 8,
  MRS_GGUF_AFFINE_Q8_1 = 9,
  MRS_GGUF_AFFINE_Q2_K = 10,
  MRS_GGUF_AFFINE_Q3_K = 11,
  MRS_GGUF_AFFINE_Q4_K = 12,
  MRS_GGUF_AFFINE_Q5_K = 13,
  MRS_GGUF_AFFINE_Q6_K = 14,
  MRS_GGUF_AFFINE_Q8_K = 15,
};

int mrs_gguf_affine_repack_f16(int format, const void *src, void *payload,
                               void *scales, void *offsets, int k, int n,
                               int padded_n, uintptr_t stream);

int mrs_gguf_affine_repack_bf16(int format, const void *src, void *payload,
                                void *scales, void *offsets, int k, int n,
                                int padded_n, uintptr_t stream);

#ifdef __cplusplus
}
#endif

#endif
