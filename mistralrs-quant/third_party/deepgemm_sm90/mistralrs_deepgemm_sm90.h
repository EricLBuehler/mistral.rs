// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MistralrsDeepGemmStatus {
  MISTRALRS_DEEPGEMM_SUCCESS = 0,
  MISTRALRS_DEEPGEMM_UNAVAILABLE = 1,
  MISTRALRS_DEEPGEMM_INVALID_ARGUMENT = 2,
  MISTRALRS_DEEPGEMM_NOT_PREPARED = 3,
  MISTRALRS_DEEPGEMM_WORKSPACE_TOO_SMALL = 4,
  MISTRALRS_DEEPGEMM_CAPTURE_ACTIVE = 5,
  MISTRALRS_DEEPGEMM_COMPILE_ERROR = 6,
  MISTRALRS_DEEPGEMM_CUDA_ERROR = 7,
  MISTRALRS_DEEPGEMM_INTERNAL_ERROR = 8,
} MistralrsDeepGemmStatus;

typedef enum MistralrsDeepGemmPlanFlags {
  MISTRALRS_DEEPGEMM_PLAN_SWAP_AB = 1U,
} MistralrsDeepGemmPlanFlags;

typedef struct MistralrsDeepGemmPlan {
  uint32_t abi_version;
  uint32_t flags;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t block_k;
  uint32_t num_stages;
  uint32_t num_tma_multicast;
  uint32_t sm_count;
  uint32_t smem_bytes;
  uint32_t device_ordinal;
  uint32_t reserved;
  size_t workspace_bytes;
  uint64_t cache_key;
} MistralrsDeepGemmPlan;

typedef struct MistralrsDeepGemmPrepared {
  MistralrsDeepGemmPlan plan;
  uintptr_t function;
} MistralrsDeepGemmPrepared;

const char* mistralrs_deepgemm_sm90_error_string(int32_t status);

const char* mistralrs_deepgemm_sm90_last_error();

int32_t mistralrs_deepgemm_sm90_plan(uint32_t m, uint32_t n, uint32_t k,
                                     MistralrsDeepGemmPlan* plan);

int32_t mistralrs_deepgemm_sm90_prepare(const MistralrsDeepGemmPlan* plan,
                                        const char* include_dir,
                                        cudaStream_t stream,
                                        MistralrsDeepGemmPrepared* prepared);

int32_t mistralrs_deepgemm_sm90_gemm(const MistralrsDeepGemmPrepared* prepared,
                                     const void* activation_bf16,
                                     const void* weight_e4m3,
                                     const float* weight_scales,
                                     void* output_bf16, void* workspace,
                                     size_t workspace_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
