#ifndef MISTRALRS_NVFP4_CUTLASS_H
#define MISTRALRS_NVFP4_CUTLASS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { MISTRALRS_NVFP4_BF16 = 0, MISTRALRS_NVFP4_F16 = 1 };
enum { MISTRALRS_NVFP4_PREFILL = 0, MISTRALRS_NVFP4_DECODE_DP32 = 1 };

typedef struct mistralrs_nvfp4_context {
  int32_t device;
  int32_t sm_count;
  int32_t dtype;
  int32_t kernel;
} mistralrs_nvfp4_context;

typedef struct mistralrs_nvfp4_shape {
  int32_t m;
  int32_t n;
  int32_t k;
} mistralrs_nvfp4_shape;

typedef struct mistralrs_nvfp4_launch {
  mistralrs_nvfp4_shape shape;
  mistralrs_nvfp4_context context;
  const void *a_packed;
  const void *w_packed;
  const void *a_scale_swizzled;
  const void *w_scale_swizzled;
  const float *weight_global;
  const float *activation_global;
  void *output;
  void *workspace;
  size_t workspace_bytes;
  void *stream;
} mistralrs_nvfp4_launch;

typedef struct mistralrs_nvfp4_resources {
  mistralrs_nvfp4_context context;
  int32_t major;
  int32_t minor;
  int32_t threads;
  int32_t registers_per_thread;
  size_t shared_bytes;
  size_t local_bytes;
} mistralrs_nvfp4_resources;

/* Success is zero, CUDA errors are negative, CUTLASS errors are positive. */
const char *mistralrs_nvfp4_error_string(int status);

/* Prepare outside capture; the requested device must already be current. */
int mistralrs_nvfp4_prepare(int32_t device, int32_t dtype, int32_t kernel, mistralrs_nvfp4_resources *resources);
int mistralrs_nvfp4_workspace_size(const mistralrs_nvfp4_context *context, const mistralrs_nvfp4_shape *shape,
                                   size_t *bytes);

/* The current device, stream, and all buffers must match the prepared context. */
/* Caller-owned operand/workspace storage must outlive eager work and graph replay. */
/* A/W/scales/output/workspace require 16-byte alignment; globals require 4-byte alignment. */
/* Output is RN_dtype((FP32_acc * weight_global[n]) * activation_global[0]); bias is external. */
int mistralrs_nvfp4_gemm(const mistralrs_nvfp4_launch *launch);

/* The swizzle preserves bytes and zeroes padding; source and destination must not overlap. */
int mistralrs_nvfp4_scale_bytes(int32_t rows, int32_t k, size_t *bytes);
int mistralrs_nvfp4_swizzle_host(const void *source, void *dest, int32_t rows, int32_t k, size_t dest_bytes);
int mistralrs_nvfp4_swizzle_cuda(const void *source, void *dest, int32_t rows, int32_t k, size_t dest_bytes,
                                 void *stream);

#ifdef __cplusplus
}
#endif
#endif
