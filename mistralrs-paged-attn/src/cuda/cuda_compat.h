#pragma once

#include <cstdint>
#include <mutex>

#ifndef USE_ROCM
#define VLLM_LDG(arg) __ldg(arg)
#else
#define VLLM_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
#define VLLM_SHFL_XOR_SYNC(var, lane_mask)                                     \
  __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#else
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
#endif

#ifndef USE_ROCM
#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
#define VLLM_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
inline cudaError_t VLLM_EnsureMaxDynamicSharedMemorySize(const void *func,
                                                       int val) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  cudaFuncAttributes attributes;
  const auto result = cudaFuncGetAttributes(&attributes, func);
  if (result != cudaSuccess || attributes.maxDynamicSharedSizeBytes >= val) {
    return result;
  }
  // Lowering the limit can invalidate a later launch or an existing graph.
  return cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              val);
}

#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  VLLM_EnsureMaxDynamicSharedMemorySize(FUNC, VAL)
#else
#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif
