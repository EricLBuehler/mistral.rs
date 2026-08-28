// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mistralrs_deepgemm_sm90.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <deep_gemm/fp8_gemm.cuh>
#include <deep_gemm/official_sm90.cuh>

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kBlockSize = MISTRALRS_DEEPGEMM_BLOCK_SIZE;
constexpr uint32_t kActivationScaleMAlignment =
    MISTRALRS_DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT;
constexpr uint32_t kWaveBalancedBlockM = 64;
constexpr uint32_t kWaveBalancedBlockN = 8;
constexpr uint32_t kWaveBalancedStages = 12;
constexpr uint32_t kWaveBalancedMinWaveDivisor = 2;
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
constexpr uint32_t kSwapAbThreshold = 32;
#endif
constexpr size_t kWorkspaceAlignment = 128;
constexpr uintptr_t kTensorAlignment = 16;

enum class KernelFamily {
  Production,
  Official1D2D,
  SmallMSwapAB,
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  Legacy,
#endif
};

static_assert(sizeof(void*) == 8);
static_assert(std::is_standard_layout_v<MistralrsDeepGemmPlan>);
static_assert(offsetof(MistralrsDeepGemmPlan, workspace_bytes) == 56);
static_assert(offsetof(MistralrsDeepGemmPlan, cache_key) == 64);
static_assert(sizeof(MistralrsDeepGemmPlan) == 72);
static_assert(std::is_standard_layout_v<MistralrsDeepGemmPrepared>);
static_assert(offsetof(MistralrsDeepGemmPrepared, function) == 72);
static_assert(sizeof(MistralrsDeepGemmPrepared) == 80);

thread_local std::string last_error;

std::mutex compiler_mutex;

struct SmallMSwapABTuning {
  uint32_t n;
  uint32_t k;
  uint32_t max_m;
};

constexpr std::array<SmallMSwapABTuning, 4> kSmallMSwapABTunings = {{
    {16384, 5120, 16},
    {14336, 5120, 16},
    {5120, 6144, 8},
    {5120, 17408, 8},
}};

struct OfficialSmallMTuning {
  uint32_t n;
  uint32_t k;
  uint32_t max_m;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t cluster_m;
  uint32_t cluster_n;
};

constexpr std::array<OfficialSmallMTuning, 2> kOfficialSmallMTunings = {{
    {34816, 5120, 2, 32, 144, 1, 1},
    {34816, 5120, 16, 64, 144, 2, 1},
}};

bool useTunedSmallMSwapAB(uint32_t m, uint32_t n, uint32_t k) {
  return std::any_of(kSmallMSwapABTunings.begin(), kSmallMSwapABTunings.end(),
                     [&](const SmallMSwapABTuning& tuning) {
                       return m <= tuning.max_m && n == tuning.n && k == tuning.k;
                     });
}

bool useWaveBalancedSmallMSwapAB(uint32_t m, uint32_t n, uint32_t k,
                                 uint32_t sm_count) {
  // A second logical M tile would reread every weight block.
  if (m > kWaveBalancedBlockN || k > std::numeric_limits<int32_t>::max() ||
      sm_count == 0) {
    return false;
  }
  const uint64_t blocks =
      (static_cast<uint64_t>(n) + kWaveBalancedBlockM - 1) /
      kWaveBalancedBlockM;
  const int smem_bytes = deep_gemm::jit::get_smem_size(
      kWaveBalancedStages, static_cast<int>(k), kWaveBalancedBlockM,
      kWaveBalancedBlockN, kBlockSize, true);
  return blocks <= sm_count && blocks * kWaveBalancedMinWaveDivisor >= sm_count &&
         smem_bytes > 0 &&
         smem_bytes <= static_cast<int>(mistralrs::deepgemm_official::kSmemCapacity);
}

bool useSmallMSwapAB(uint32_t m, uint32_t n, uint32_t k, uint32_t sm_count) {
  return useTunedSmallMSwapAB(m, n, k) ||
         useWaveBalancedSmallMSwapAB(m, n, k, sm_count);
}

KernelFamily productionKernelFamily(uint32_t m, uint32_t n, uint32_t k,
                                    uint32_t sm_count) {
  return useSmallMSwapAB(m, n, k, sm_count) ? KernelFamily::SmallMSwapAB
                                            : KernelFamily::Official1D2D;
}

mistralrs::deepgemm_official::GemmConfig productionOfficialConfig(
    uint32_t m, uint32_t n, uint32_t k, uint32_t sm_count) {
  const auto heuristic =
      mistralrs::deepgemm_official::getBestGemmConfig(m, n, k, sm_count);
  for (const auto& tuning : kOfficialSmallMTunings) {
    if (m > tuning.max_m || n != tuning.n || k != tuning.k) {
      continue;
    }
    const auto tuned = mistralrs::deepgemm_official::getGemmConfig(
        m, n, k, sm_count, tuning.block_m, tuning.block_n, tuning.cluster_m,
        tuning.cluster_n);
    return tuned.valid() ? tuned : heuristic;
  }
  return heuristic;
}

KernelFamily planKernelFamily(const MistralrsDeepGemmPlan& plan) {
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D) != 0) {
    return KernelFamily::Official1D2D;
  }
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0 &&
      useSmallMSwapAB(plan.m, plan.n, plan.k, plan.sm_count)) {
    return KernelFamily::SmallMSwapAB;
  }
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  return KernelFamily::Legacy;
#else
  return KernelFamily::SmallMSwapAB;
#endif
}

bool pdlRequested() {
  const char* value = std::getenv("MISTRALRS_DEEPGEMM_PDL");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

size_t alignUp(size_t value, size_t alignment) {
  if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("DeepGEMM workspace size overflow");
  }
  return (value + alignment - 1) / alignment * alignment;
}

bool pointerAligned(const void* pointer, uintptr_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

uint64_t hashValue(uint64_t hash, uint64_t value) {
  constexpr uint64_t prime = 0x100000001b3ULL;
  for (uint32_t byte = 0; byte < sizeof(value); ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= prime;
  }
  return hash;
}

uint64_t planCacheKey(const MistralrsDeepGemmPlan& plan) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  hash = hashValue(hash, plan.abi_version);
  hash = hashValue(hash, plan.flags);
  hash = hashValue(hash, plan.m);
  hash = hashValue(hash, plan.n);
  hash = hashValue(hash, plan.k);
  hash = hashValue(hash, plan.block_m);
  hash = hashValue(hash, plan.block_n);
  hash = hashValue(hash, plan.block_k);
  hash = hashValue(hash, plan.num_stages);
  hash = hashValue(hash, plan.num_tma_multicast);
  hash = hashValue(hash, plan.sm_count);
  hash = hashValue(hash, plan.smem_bytes);
  hash = hashValue(hash, plan.device_ordinal);
  hash = hashValue(hash, plan.reserved);
  hash = hashValue(hash, plan.workspace_bytes);
  return hash;
}

bool samePlan(const MistralrsDeepGemmPlan& lhs, const MistralrsDeepGemmPlan& rhs) {
  return lhs.abi_version == rhs.abi_version && lhs.flags == rhs.flags && lhs.m == rhs.m &&
         lhs.n == rhs.n && lhs.k == rhs.k && lhs.block_m == rhs.block_m &&
         lhs.block_n == rhs.block_n && lhs.block_k == rhs.block_k &&
         lhs.num_stages == rhs.num_stages &&
         lhs.num_tma_multicast == rhs.num_tma_multicast && lhs.sm_count == rhs.sm_count &&
         lhs.smem_bytes == rhs.smem_bytes && lhs.device_ordinal == rhs.device_ordinal &&
         lhs.reserved == rhs.reserved && lhs.workspace_bytes == rhs.workspace_bytes &&
         lhs.cache_key == rhs.cache_key;
}

int32_t cudaFailure(const char* operation, cudaError_t status) {
  last_error = std::string(operation) + ": " + cudaGetErrorString(status);
  return MISTRALRS_DEEPGEMM_CUDA_ERROR;
}

int32_t driverFailure(const char* operation, CUresult status) {
  const char* message = nullptr;
  cuGetErrorString(status, &message);
  last_error = std::string(operation) + ": " + (message == nullptr ? "unknown" : message);
  return MISTRALRS_DEEPGEMM_CUDA_ERROR;
}

int32_t currentContext(CUcontext* context) {
  CUresult status = cuCtxGetCurrent(context);
  if (status != CUDA_SUCCESS) {
    return driverFailure("cuCtxGetCurrent", status);
  }
  if (*context == nullptr) {
    last_error = "DeepGEMM requires a current CUDA context";
    return MISTRALRS_DEEPGEMM_UNAVAILABLE;
  }
  return MISTRALRS_DEEPGEMM_SUCCESS;
}

int32_t createPlan(uint32_t m, uint32_t n, uint32_t k, KernelFamily family,
                   MistralrsDeepGemmPlan* plan) {
  if (plan == nullptr || m == 0 || n == 0 || k == 0 || n % kBlockSize != 0 ||
      k % kBlockSize != 0 || m > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      n > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      k > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    last_error =
        "DeepGEMM requires nonzero int32 shapes with N and K divisible by the 128x128 scale block";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }

  CUcontext context = nullptr;
  int32_t status = currentContext(&context);
  if (status != MISTRALRS_DEEPGEMM_SUCCESS) {
    return status;
  }
  CUdevice device = 0;
  auto driver_status = cuCtxGetDevice(&device);
  if (driver_status != CUDA_SUCCESS) {
    return driverFailure("cuCtxGetDevice", driver_status);
  }
  int major = 0;
  int minor = 0;
  int sm_count = 0;
  driver_status = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (driver_status == CUDA_SUCCESS) {
    driver_status =
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  }
  if (driver_status == CUDA_SUCCESS) {
    driver_status = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  }
  if (driver_status != CUDA_SUCCESS) {
    return driverFailure("cuDeviceGetAttribute", driver_status);
  }
  if (major != 9 || minor != 0) {
    last_error = "DeepGEMM decode provider requires an SM90 CUDA device";
    return MISTRALRS_DEEPGEMM_UNAVAILABLE;
  }
  if (family == KernelFamily::Production) {
    family = productionKernelFamily(m, n, k, static_cast<uint32_t>(sm_count));
  }

  uint32_t flags = 0;
  uint32_t block_m = 0;
  uint32_t block_n = 0;
  uint32_t num_stages = 0;
  uint32_t num_tma_multicast = 0;
  uint32_t smem_bytes = 0;
  if (family == KernelFamily::Official1D2D) {
    const auto config = productionOfficialConfig(m, n, k, static_cast<uint32_t>(sm_count));
    if (config.valid()) {
      flags |= MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D;
      if (config.multicast_on_a()) {
        flags |= MISTRALRS_DEEPGEMM_PLAN_MULTICAST_ON_A;
      }
      if (pdlRequested()) {
        flags |= MISTRALRS_DEEPGEMM_PLAN_PDL;
      }
      block_m = config.block_m;
      block_n = config.block_n;
      num_stages = config.num_stages;
      num_tma_multicast = config.cluster_size();
      smem_bytes = config.smem_bytes;
    }
  } else if (family == KernelFamily::SmallMSwapAB) {
    if (!useSmallMSwapAB(m, n, k, static_cast<uint32_t>(sm_count))) {
      last_error = "DeepGEMM small-M swap-AB does not support the requested shape";
      return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
    }
    const bool wave_balanced = useWaveBalancedSmallMSwapAB(
        m, n, k, static_cast<uint32_t>(sm_count));
    const auto config = wave_balanced
                            ? deep_gemm::jit::GemmConfig{
                                  kWaveBalancedBlockM, kWaveBalancedBlockN,
                                  kWaveBalancedStages, 1,
                                  deep_gemm::jit::get_smem_size(
                                      kWaveBalancedStages, static_cast<int>(k),
                                      kWaveBalancedBlockM, kWaveBalancedBlockN,
                                      kBlockSize, true)}
                            : deep_gemm::jit::get_best_gemm_config(
                                  n, m, k, 1, sm_count, false, true);
    auto [small_m_block_m, small_m_block_n, small_m_num_stages,
          small_m_num_tma_multicast, small_m_smem_bytes] = config;
    if (small_m_block_m <= 0 || small_m_block_n <= 0 || small_m_num_stages <= 0 ||
        small_m_num_tma_multicast <= 0 || small_m_smem_bytes <= 0) {
      last_error = "DeepGEMM could not plan the requested small-M shape";
      return MISTRALRS_DEEPGEMM_UNAVAILABLE;
    }
    flags |= MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB;
    block_m = static_cast<uint32_t>(small_m_block_m);
    block_n = static_cast<uint32_t>(small_m_block_n);
    num_stages = static_cast<uint32_t>(small_m_num_stages);
    num_tma_multicast = static_cast<uint32_t>(small_m_num_tma_multicast);
    smem_bytes = static_cast<uint32_t>(small_m_smem_bytes);
  }
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  else {
    const bool swap_ab = m < kSwapAbThreshold;
    auto config = swap_ab
                      ? deep_gemm::jit::get_best_gemm_config(n, m, k, 1,
                                                             sm_count, false, true)
                      : deep_gemm::jit::get_best_gemm_config(m, n, k, 1,
                                                             sm_count);
    auto [legacy_block_m, legacy_block_n, legacy_num_stages,
          legacy_num_tma_multicast, legacy_smem_bytes] = config;
    if (legacy_block_m <= 0 || legacy_block_n <= 0 || legacy_num_stages <= 0 ||
        legacy_num_tma_multicast <= 0 || legacy_smem_bytes <= 0) {
      last_error = "DeepGEMM could not plan the requested shape";
      return MISTRALRS_DEEPGEMM_UNAVAILABLE;
    }
    if (swap_ab) {
      flags |= MISTRALRS_DEEPGEMM_PLAN_SWAP_AB;
    }
    block_m = static_cast<uint32_t>(legacy_block_m);
    block_n = static_cast<uint32_t>(legacy_block_n);
    num_stages = static_cast<uint32_t>(legacy_num_stages);
    num_tma_multicast = static_cast<uint32_t>(legacy_num_tma_multicast);
    smem_bytes = static_cast<uint32_t>(legacy_smem_bytes);
  }
#endif
  if (block_m == 0 || block_n == 0 || num_stages == 0 || num_tma_multicast == 0 ||
      smem_bytes == 0) {
    last_error = "DeepGEMM could not plan the requested shape";
    return MISTRALRS_DEEPGEMM_UNAVAILABLE;
  }

  size_t activation_elements = static_cast<size_t>(m) * static_cast<size_t>(k);
  if (activation_elements / k != m) {
    last_error = "DeepGEMM activation workspace size overflow";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  size_t activation_bytes = alignUp(activation_elements, kWorkspaceAlignment);
  size_t scale_rows = alignUp(m, 4);
  size_t scale_elements = scale_rows * (k / kBlockSize);
  if (scale_elements / (k / kBlockSize) != scale_rows ||
      scale_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    last_error = "DeepGEMM activation scale workspace size overflow";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  size_t workspace_bytes =
      activation_bytes + alignUp(scale_elements * sizeof(float), kWorkspaceAlignment);
  if (workspace_bytes < activation_bytes) {
    last_error = "DeepGEMM workspace size overflow";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }

  *plan = {};
  plan->abi_version = kAbiVersion;
  plan->flags = flags;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->block_m = block_m;
  plan->block_n = block_n;
  plan->block_k = kBlockSize;
  plan->num_stages = num_stages;
  plan->num_tma_multicast = num_tma_multicast;
  plan->sm_count = static_cast<uint32_t>(sm_count);
  plan->smem_bytes = smem_bytes;
  plan->device_ordinal = static_cast<uint32_t>(device);
  plan->workspace_bytes = workspace_bytes;
  plan->cache_key = planCacheKey(*plan);
  return MISTRALRS_DEEPGEMM_SUCCESS;
}

int32_t validatePlan(const MistralrsDeepGemmPlan* plan) {
  if (plan == nullptr || plan->abi_version != kAbiVersion) {
    last_error = "DeepGEMM plan has an incompatible ABI version";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  MistralrsDeepGemmPlan expected{};
  const KernelFamily family = planKernelFamily(*plan);
  int32_t status = createPlan(plan->m, plan->n, plan->k, family, &expected);
  if (status != MISTRALRS_DEEPGEMM_SUCCESS) {
    return status;
  }
  if (!samePlan(*plan, expected)) {
    last_error = "DeepGEMM plan does not match the current device and shape";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  return MISTRALRS_DEEPGEMM_SUCCESS;
}

std::filesystem::path resolveIncludeDir(const char* include_dir) {
  if (include_dir != nullptr && include_dir[0] != '\0') {
    return include_dir;
  }
  return {};
}

std::filesystem::path resolveFamilyIncludeDir(const std::filesystem::path& include_dir,
                                              KernelFamily family) {
  if (include_dir.empty()) {
    return {};
  }
  const char* family_dir = nullptr;
  const char* marker = nullptr;
  if (family == KernelFamily::Official1D2D) {
    family_dir = "official";
    marker = "deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh";
  } else if (family == KernelFamily::SmallMSwapAB) {
    family_dir = "skinny";
    marker = "deep_gemm/nvrtc_cutlass.cuh";
  }
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  else {
    family_dir = "legacy";
    marker = "deep_gemm/nvrtc_cutlass.cuh";
  }
#endif
  std::error_code error;
  auto nested = include_dir / family_dir;
  if (std::filesystem::is_regular_file(nested / marker, error)) {
    return nested;
  }
  error.clear();
  if (std::filesystem::is_regular_file(include_dir / marker, error)) {
    return include_dir;
  }
  return {};
}

__device__ float warpMax(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, offset));
  }
  return value;
}

__global__ void quantizeBf16ToFp8E4m3(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                                      float* scales, uint32_t m, uint32_t k,
                                      uint32_t scale_stride_m) {
  uint32_t k_blocks = k / kBlockSize;
  uint64_t scale_count = static_cast<uint64_t>(m) * k_blocks;
  uint64_t warp = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warpSize;
  uint64_t warp_stride = static_cast<uint64_t>(gridDim.x) * blockDim.x / warpSize;
  uint32_t lane = threadIdx.x % warpSize;
  for (; warp < scale_count; warp += warp_stride) {
    uint32_t row = static_cast<uint32_t>(warp / k_blocks);
    uint32_t k_block = static_cast<uint32_t>(warp % k_blocks);
    const __nv_bfloat16* input_block =
        input + static_cast<uint64_t>(row) * k + k_block * kBlockSize;
    __nv_fp8_e4m3* output_block =
        output + static_cast<uint64_t>(row) * k + k_block * kBlockSize;

    __nv_bfloat162 values[2];
    values[0] = reinterpret_cast<const __nv_bfloat162*>(input_block)[lane];
    values[1] = reinterpret_cast<const __nv_bfloat162*>(input_block)[lane + 32];
    float first_x = __bfloat162float(values[0].x);
    float first_y = __bfloat162float(values[0].y);
    float second_x = __bfloat162float(values[1].x);
    float second_y = __bfloat162float(values[1].y);
    float amax = warpMax(fmaxf(fmaxf(fabsf(first_x), fabsf(first_y)),
                               fmaxf(fabsf(second_x), fabsf(second_y))));
    float quant_scale = amax == 0.0f ? 1.0f : 448.0f / amax;
    if (lane == 0) {
      scales[static_cast<uint64_t>(k_block) * scale_stride_m + row] = 1.0f / quant_scale;
    }
    output_block[lane * 2] = __nv_fp8_e4m3(first_x * quant_scale);
    output_block[lane * 2 + 1] = __nv_fp8_e4m3(first_y * quant_scale);
    output_block[64 + lane * 2] = __nv_fp8_e4m3(second_x * quant_scale);
    output_block[64 + lane * 2 + 1] = __nv_fp8_e4m3(second_y * quant_scale);
  }
}

#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
void launchNormalGemm(CUfunction function, void* mat_a, int ld_a, void* mat_b, int ld_b,
                      void* mat_d, int ld_d, float* scales_a, float* scales_b, uint32_t shape_m,
                      uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
                      uint32_t block_k, uint32_t num_tma_multicast, CUstream stream,
                      uint32_t num_sms, uint32_t smem_size) {
  auto tma_a_desc = deep_gemm::make_2d_tma_a_desc(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, 1,
      deep_gemm::GemmType::Normal, ld_a);
  auto tma_b_desc = deep_gemm::make_2d_tma_b_desc(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, 1,
      deep_gemm::GemmType::Normal, ld_b);
  auto tma_scales_a_desc = deep_gemm::make_2d_tma_scales_a_desc(
      scales_a, shape_m, shape_k, block_m, block_k, 1, deep_gemm::GemmType::Normal);
  auto tma_d_desc = deep_gemm::make_2d_tma_d_desc(
      reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, 1,
      deep_gemm::GemmType::Normal, ld_d * 2);

  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attribute.value.clusterDim = {num_tma_multicast, 1, 1};
  CUlaunchConfig config{};
  config.gridDimX = num_sms;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = deep_gemm::get_num_threads_per_sm<128, 128>(static_cast<int32_t>(block_m));
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  auto* output = reinterpret_cast<__nv_bfloat16*>(mat_d);
  deep_gemm::NormalSchedulerInput input{};
  input.shape_m = shape_m;
  void* parameters[] = {&output, &scales_b, &input, &tma_a_desc, &tma_b_desc,
                        &tma_scales_a_desc, &tma_d_desc};
  CHECK_CUDA(cuLaunchKernelEx(&config, function, parameters, nullptr));
}
#endif

void launchSmallMSwapABGemm(CUfunction function, void* mat_a, int ld_a, void* mat_b,
                            int ld_b, void* mat_d, int ld_d, float* scales_a,
                            float* scales_b, uint32_t shape_m, uint32_t shape_n,
                            uint32_t shape_k, uint32_t block_m, uint32_t block_n,
                            uint32_t block_k, uint32_t num_tma_multicast, CUstream stream,
                            uint32_t num_sms, uint32_t smem_size) {
  auto tma_a_desc = deep_gemm::make_2d_tma_a_desc_swapAB(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, 1,
      deep_gemm::GemmType::Normal, ld_a);
  auto tma_b_desc = deep_gemm::make_2d_tma_b_desc_swapAB(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, 1,
      deep_gemm::GemmType::Normal, ld_b);
  auto tma_scales_b_desc = deep_gemm::make_2d_tma_scales_b_desc_swapAB(
      scales_b, shape_n, shape_k, block_n, block_k, 1, deep_gemm::GemmType::Normal);
  auto tma_d_desc = deep_gemm::make_2d_tma_d_desc_swapAB(
      reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, 1,
      deep_gemm::GemmType::Normal, ld_d * 2);

  CUlaunchAttribute attribute{};
  attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attribute.value.clusterDim = {num_tma_multicast, 1, 1};
  CUlaunchConfig config{};
  config.gridDimX = num_sms;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = deep_gemm::get_num_threads_per_sm<128, 128>(static_cast<int32_t>(block_m));
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  auto* output = reinterpret_cast<__nv_bfloat16*>(mat_d);
  deep_gemm::NormalSchedulerInputSwapAB input{};
  input.shape_n = shape_n;
  void* parameters[] = {&output, &scales_a, &input, &tma_a_desc, &tma_b_desc,
                        &tma_scales_b_desc, &tma_d_desc};
  CHECK_CUDA(cuLaunchKernelEx(&config, function, parameters, nullptr));
}

CUtensorMapSwizzle officialOutputSwizzle(uint32_t block_n) {
  switch (deep_gemm::jit::getOfficialOutputSwizzle(block_n)) {
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    default:
      throw std::runtime_error("DeepGEMM official kernel has an invalid output swizzle");
  }
}

void launchOfficialGemm(CUfunction function, void* mat_a, int ld_a, void* mat_b, int ld_b,
                        void* mat_d, int ld_d, float* scales_a, float* scales_b,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        uint32_t block_m, uint32_t block_n, uint32_t block_k,
                        uint32_t num_tma_multicast, bool multicast_on_a, bool enable_pdl,
                        CUstream stream, uint32_t num_sms, uint32_t smem_size) {
  auto tma_a_desc = deep_gemm::make_2d_tma_a_desc(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, 1,
      deep_gemm::GemmType::Normal, ld_a);
  auto tma_b_desc = deep_gemm::make_2d_tma_b_desc(
      reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, 1,
      deep_gemm::GemmType::Normal, ld_b);
  auto tma_scales_a_desc = deep_gemm::make_2d_tma_scales_a_desc(
      scales_a, shape_m, shape_k, block_m, block_k, 1, deep_gemm::GemmType::Normal);

  const auto output_swizzle = officialOutputSwizzle(block_n);
  const uint32_t output_smem_columns =
      deep_gemm::jit::getOfficialOutputSwizzle(block_n) / sizeof(__nv_bfloat16);
  uint64_t output_global_dims[2] = {shape_n, shape_m};
  uint32_t output_smem_dims[2] = {output_smem_columns, block_m};
  auto tma_d_desc = deep_gemm::make_2d_tma_copy_desc(
      reinterpret_cast<__nv_bfloat16*>(mat_d), output_global_dims,
      static_cast<uint64_t>(ld_d) * sizeof(__nv_bfloat16), output_smem_dims,
      output_swizzle);

  CUlaunchAttribute attributes[2]{};
  uint32_t attribute_count = 0;
  if (num_tma_multicast > 1) {
    auto& cluster = attributes[attribute_count++];
    cluster.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster.value.clusterDim = {num_tma_multicast, 1, 1};
  }
  if (enable_pdl) {
    auto& pdl = attributes[attribute_count++];
    pdl.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    pdl.value.programmaticStreamSerializationAllowed = 1;
  }

  CUlaunchConfig config{};
  config.gridDimX = num_sms;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 128 + (block_m <= 64 ? 128 : 256);
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;
  config.attrs = attributes;
  config.numAttrs = attribute_count;

  int* grouped_layout = nullptr;
  static_cast<void>(multicast_on_a);
  void* parameters[] = {&scales_b, &grouped_layout, &shape_m, &shape_n, &shape_k,
                        &tma_a_desc, &tma_b_desc, &tma_d_desc, &tma_scales_a_desc};
  CHECK_CUDA(cuLaunchKernelEx(&config, function, parameters, nullptr));
}

int32_t validatePreparedLaunch(const MistralrsDeepGemmPrepared* prepared, uint32_t m) {
  if (prepared == nullptr || prepared->function == 0 ||
      prepared->plan.abi_version != kAbiVersion) {
    last_error = "DeepGEMM launch received an invalid prepared plan";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  const auto* plan = &prepared->plan;
  if (m == 0 || m > plan->m) {
    last_error = "DeepGEMM launch M exceeds the prepared row capacity";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  if ((plan->flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0 && m != plan->m) {
    last_error = "DeepGEMM small-M swap-AB launches require the prepared row count";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  return MISTRALRS_DEEPGEMM_SUCCESS;
}

int32_t launchPrequantizedGemm(const MistralrsDeepGemmPrepared* prepared, uint32_t m,
                               const void* activation_e4m3, const float* activation_scales,
                               uint32_t activation_scale_stride_m, const void* weight_e4m3,
                               const float* weight_scales, void* output_bf16,
                               cudaStream_t stream) {
  int32_t status = validatePreparedLaunch(prepared, m);
  if (status != MISTRALRS_DEEPGEMM_SUCCESS) {
    return status;
  }
  uint32_t expected_scale_stride_m =
      static_cast<uint32_t>(alignUp(m, kActivationScaleMAlignment));
  if (activation_scale_stride_m != expected_scale_stride_m) {
    last_error = "DeepGEMM activation scales require an M stride aligned to 4 rows";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
  if (activation_e4m3 == nullptr || activation_scales == nullptr || weight_e4m3 == nullptr ||
      weight_scales == nullptr || output_bf16 == nullptr ||
      !pointerAligned(activation_e4m3, kTensorAlignment) ||
      !pointerAligned(activation_scales, kTensorAlignment) ||
      !pointerAligned(weight_e4m3, kTensorAlignment) ||
      !pointerAligned(weight_scales, kTensorAlignment) ||
      !pointerAligned(output_bf16, kTensorAlignment)) {
    last_error = "DeepGEMM prequantized tensors require nonnull 16-byte aligned pointers";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }

  const auto* plan = &prepared->plan;
  auto function = reinterpret_cast<CUfunction>(prepared->function);
  const bool official = (plan->flags & MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D) != 0;
  const bool swap_ab = (plan->flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0;
  if (official) {
    launchOfficialGemm(
        function, const_cast<void*>(activation_e4m3), plan->k,
        const_cast<void*>(weight_e4m3), plan->k, output_bf16, plan->n,
        const_cast<float*>(activation_scales), const_cast<float*>(weight_scales), m,
        plan->n, plan->k, plan->block_m, plan->block_n, plan->block_k,
        plan->num_tma_multicast,
        (plan->flags & MISTRALRS_DEEPGEMM_PLAN_MULTICAST_ON_A) != 0,
        (plan->flags & MISTRALRS_DEEPGEMM_PLAN_PDL) != 0,
        reinterpret_cast<CUstream>(stream), plan->sm_count, plan->smem_bytes);
  }
  else if (swap_ab) {
    launchSmallMSwapABGemm(
        function, const_cast<void*>(weight_e4m3), plan->k, const_cast<void*>(activation_e4m3),
        plan->k, output_bf16, plan->n, const_cast<float*>(weight_scales),
        const_cast<float*>(activation_scales), plan->n, m, plan->k, plan->block_m, plan->block_n,
        plan->block_k, plan->num_tma_multicast, reinterpret_cast<CUstream>(stream), plan->sm_count,
        plan->smem_bytes);
  }
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
  else {
    launchNormalGemm(
        function, const_cast<void*>(activation_e4m3), plan->k, const_cast<void*>(weight_e4m3),
        plan->k, output_bf16, plan->n, const_cast<float*>(activation_scales),
        const_cast<float*>(weight_scales), m, plan->n, plan->k, plan->block_m, plan->block_n,
        plan->block_k, plan->num_tma_multicast, reinterpret_cast<CUstream>(stream), plan->sm_count,
        plan->smem_bytes);
  }
#else
  else {
    last_error = "DeepGEMM production plan has an unknown kernel family";
    return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
  }
#endif
  return MISTRALRS_DEEPGEMM_SUCCESS;
}

int32_t classifyException(const std::exception& error) {
  last_error = error.what();
  if (last_error.find("runtime nvcc") != std::string::npos ||
      last_error.find("include bundle") != std::string::npos ||
      last_error.find("requires an SM90") != std::string::npos) {
    return MISTRALRS_DEEPGEMM_UNAVAILABLE;
  }
  if (last_error.find("nvcc") != std::string::npos) {
    return MISTRALRS_DEEPGEMM_COMPILE_ERROR;
  }
  if (last_error.find("CUDA") != std::string::npos ||
      last_error.find("cuda") != std::string::npos) {
    return MISTRALRS_DEEPGEMM_CUDA_ERROR;
  }
  return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
}

}  // namespace

extern "C" const char* mistralrs_deepgemm_sm90_error_string(int32_t status) {
  switch (status) {
    case MISTRALRS_DEEPGEMM_SUCCESS:
      return "success";
    case MISTRALRS_DEEPGEMM_UNAVAILABLE:
      return "provider unavailable";
    case MISTRALRS_DEEPGEMM_INVALID_ARGUMENT:
      return "invalid argument";
    case MISTRALRS_DEEPGEMM_NOT_PREPARED:
      return "plan not prepared";
    case MISTRALRS_DEEPGEMM_WORKSPACE_TOO_SMALL:
      return "workspace too small";
    case MISTRALRS_DEEPGEMM_CAPTURE_ACTIVE:
      return "preparation attempted during CUDA graph capture";
    case MISTRALRS_DEEPGEMM_COMPILE_ERROR:
      return "runtime compilation failed";
    case MISTRALRS_DEEPGEMM_CUDA_ERROR:
      return "CUDA operation failed";
    case MISTRALRS_DEEPGEMM_INTERNAL_ERROR:
      return "internal provider error";
    default:
      return "unknown provider status";
  }
}

extern "C" const char* mistralrs_deepgemm_sm90_last_error() {
  return last_error.c_str();
}

extern "C" int32_t mistralrs_deepgemm_sm90_plan(uint32_t m, uint32_t n, uint32_t k,
                                                 MistralrsDeepGemmPlan* plan) {
  last_error.clear();
  try {
    return createPlan(m, n, k, KernelFamily::Production, plan);
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM planning error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}

#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
extern "C" int32_t mistralrs_deepgemm_sm90_plan_legacy_for_test(
    uint32_t m, uint32_t n, uint32_t k, MistralrsDeepGemmPlan* plan) {
  last_error.clear();
  try {
    return createPlan(m, n, k, KernelFamily::Legacy, plan);
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM legacy diagnostic planning error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}
#endif

extern "C" int32_t mistralrs_deepgemm_sm90_prepare(const MistralrsDeepGemmPlan* plan,
                                                    const char* include_dir,
                                                    cudaStream_t stream,
                                                    MistralrsDeepGemmPrepared* prepared) {
  last_error.clear();
  try {
    if (plan == nullptr || prepared == nullptr) {
      last_error = "DeepGEMM prepare received a null plan or output";
      return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
    }
    CUstreamCaptureStatus capture_status;
    auto driver_status =
        cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
    if (driver_status != CUDA_SUCCESS) {
      return driverFailure("cuStreamIsCapturing", driver_status);
    }
    if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
      last_error = "DeepGEMM plans must be prepared before CUDA graph capture";
      return MISTRALRS_DEEPGEMM_CAPTURE_ACTIVE;
    }

    int32_t status = validatePlan(plan);
    if (status != MISTRALRS_DEEPGEMM_SUCCESS) {
      return status;
    }

    std::lock_guard<std::mutex> compile_lock(compiler_mutex);
    auto headers = resolveIncludeDir(include_dir);
    const KernelFamily family = planKernelFamily(*plan);
    auto family_headers = resolveFamilyIncludeDir(headers, family);
    deep_gemm::jit::Compiler::setIncludeDirs(
        family_headers.empty() ? std::vector<std::filesystem::path>{}
                               : std::vector<std::filesystem::path>{family_headers});
    const bool swap_ab = (plan->flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0;
    deep_gemm::jit::KernelFamily jit_family = deep_gemm::jit::KernelFamily::Official1D2D;
    if (family == KernelFamily::SmallMSwapAB) {
      jit_family = deep_gemm::jit::KernelFamily::SmallMSwapAB;
    }
#ifdef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
    if (family == KernelFamily::Legacy) {
      jit_family = deep_gemm::jit::KernelFamily::Legacy;
    }
#endif
    auto* runtime = deep_gemm::jit::getGlobalCompiler().build(
        plan->n, plan->k, plan->block_m, plan->block_n, plan->block_k, 1, plan->num_stages,
        plan->num_tma_multicast, deep_gemm::GemmType::Normal, swap_ab, jit_family,
        (plan->flags & MISTRALRS_DEEPGEMM_PLAN_MULTICAST_ON_A) != 0,
        plan->sm_count);
    auto kernel = runtime->getKernel();
    CHECK_CUDA(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                    static_cast<int>(plan->smem_bytes), kernel,
                                    static_cast<CUdevice>(plan->device_ordinal)));
    CUfunction function = nullptr;
    CHECK_CUDA(cuKernelGetFunction(&function, kernel));
    CHECK_CUDA(cuFuncLoad(function));
    *prepared = {*plan, reinterpret_cast<uintptr_t>(function)};
    return MISTRALRS_DEEPGEMM_SUCCESS;
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM preparation error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}

extern "C" int32_t mistralrs_deepgemm_sm90_gemm(
    const MistralrsDeepGemmPrepared* prepared, uint32_t m, const void* activation_bf16,
    const void* weight_e4m3, const float* weight_scales, void* output_bf16,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  last_error.clear();
  try {
    int32_t status = validatePreparedLaunch(prepared, m);
    if (status != MISTRALRS_DEEPGEMM_SUCCESS) {
      return status;
    }
    const auto* plan = &prepared->plan;
    if (activation_bf16 == nullptr || weight_e4m3 == nullptr || weight_scales == nullptr ||
        output_bf16 == nullptr || workspace == nullptr ||
        !pointerAligned(activation_bf16, kTensorAlignment) ||
        !pointerAligned(weight_e4m3, kTensorAlignment) ||
        !pointerAligned(weight_scales, kTensorAlignment) ||
        !pointerAligned(output_bf16, kTensorAlignment) ||
        !pointerAligned(workspace, kWorkspaceAlignment)) {
      last_error = "DeepGEMM tensors require nonnull 16-byte aligned pointers and 128-byte workspace";
      return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
    }
    if (workspace_bytes < plan->workspace_bytes) {
      last_error = "DeepGEMM caller workspace is smaller than the prepared plan requires";
      return MISTRALRS_DEEPGEMM_WORKSPACE_TOO_SMALL;
    }

    auto* workspace_bytes_ptr = static_cast<std::byte*>(workspace);
    size_t activation_elements = static_cast<size_t>(m) * plan->k;
    size_t activation_bytes = alignUp(activation_elements, kWorkspaceAlignment);
    auto* activation_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(workspace_bytes_ptr);
    auto* activation_scales = reinterpret_cast<float*>(workspace_bytes_ptr + activation_bytes);
    uint64_t scale_count = static_cast<uint64_t>(m) * (plan->k / kBlockSize);
    uint32_t blocks = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(plan->sm_count) * 8,
                           std::max<uint64_t>(1, (scale_count + 7) / 8)));
    quantizeBf16ToFp8E4m3<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation_bf16), activation_fp8, activation_scales,
        m, plan->k, static_cast<uint32_t>(alignUp(m, kActivationScaleMAlignment)));
    auto cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      return cudaFailure("DeepGEMM activation quantization launch", cuda_status);
    }

    return launchPrequantizedGemm(
        prepared, m, activation_fp8, activation_scales,
        static_cast<uint32_t>(alignUp(m, kActivationScaleMAlignment)), weight_e4m3, weight_scales,
        output_bf16, stream);
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM launch error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}

extern "C" int32_t mistralrs_deepgemm_sm90_gemm_prequantized(
    const MistralrsDeepGemmPrepared* prepared, uint32_t m, const void* activation_e4m3,
    const float* activation_scales, uint32_t activation_scale_stride_m,
    const void* weight_e4m3, const float* weight_scales, void* output_bf16,
    cudaStream_t stream) {
  last_error.clear();
  try {
    return launchPrequantizedGemm(prepared, m, activation_e4m3, activation_scales,
                                  activation_scale_stride_m, weight_e4m3, weight_scales,
                                  output_bf16, stream);
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM prequantized launch error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}
