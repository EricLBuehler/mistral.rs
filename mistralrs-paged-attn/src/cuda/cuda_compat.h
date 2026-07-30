#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_set>

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
inline cudaError_t VLLM_SetMaxDynamicSharedMemorySizeOnce(const void *func,
                                                          int val) {
  static std::mutex mutex;
  static std::unordered_set<uint64_t> seen;
  const auto key =
      (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(func)) >> 4) ^
      (static_cast<uint64_t>(static_cast<uint32_t>(val)) << 32) ^
      static_cast<uint64_t>(static_cast<uint32_t>(val));
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (seen.find(key) != seen.end()) {
      return cudaSuccess;
    }
  }
  const auto result =
      cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           val);
  if (result == cudaSuccess) {
    std::lock_guard<std::mutex> lock(mutex);
    seen.insert(key);
  }
  return result;
}

#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  VLLM_SetMaxDynamicSharedMemorySizeOnce(FUNC, VAL)
#else
#define VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL)        \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif
