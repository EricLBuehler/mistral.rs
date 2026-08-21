// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mistralrs_deepgemm_sm90.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kSwapAbThreshold = 32;
constexpr size_t kWorkspaceAlignment = 128;
constexpr uintptr_t kTensorAlignment = 16;

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

int32_t createPlan(uint32_t m, uint32_t n, uint32_t k, MistralrsDeepGemmPlan* plan) {
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

  bool swap_ab = m < kSwapAbThreshold;
  auto config = swap_ab
                    ? deep_gemm::jit::get_best_gemm_config(n, m, k, 1,
                                                           sm_count, false, true)
                    : deep_gemm::jit::get_best_gemm_config(m, n, k, 1,
                                                           sm_count);
  auto [block_m, block_n, num_stages, num_tma_multicast, smem_bytes] = config;
  if (block_m <= 0 || block_n <= 0 || num_stages <= 0 || num_tma_multicast <= 0 ||
      smem_bytes <= 0) {
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
  plan->flags = swap_ab ? MISTRALRS_DEEPGEMM_PLAN_SWAP_AB : 0;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  plan->block_m = static_cast<uint32_t>(block_m);
  plan->block_n = static_cast<uint32_t>(block_n);
  plan->block_k = kBlockSize;
  plan->num_stages = static_cast<uint32_t>(num_stages);
  plan->num_tma_multicast = static_cast<uint32_t>(num_tma_multicast);
  plan->sm_count = static_cast<uint32_t>(sm_count);
  plan->smem_bytes = static_cast<uint32_t>(smem_bytes);
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
  int32_t status = createPlan(plan->m, plan->n, plan->k, &expected);
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

bool includeDirUsable(const std::filesystem::path& include_dir) {
  if (include_dir.empty()) {
    return false;
  }
  std::error_code error;
  return std::filesystem::is_regular_file(
      include_dir / "deep_gemm" / "nvrtc_cutlass.cuh", error);
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

void launchNormalGemmSwapAB(CUfunction function, void* mat_a, int ld_a, void* mat_b, int ld_b,
                            void* mat_d, int ld_d, float* scales_a, float* scales_b,
                            uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                            uint32_t block_m, uint32_t block_n, uint32_t block_k,
                            uint32_t num_tma_multicast, CUstream stream, uint32_t num_sms,
                            uint32_t smem_size) {
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
    return createPlan(m, n, k, plan);
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM planning error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}

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
    deep_gemm::jit::Compiler::setIncludeDirs(includeDirUsable(headers)
                                                  ? std::vector<std::filesystem::path>{headers}
                                                  : std::vector<std::filesystem::path>{});
    bool swap_ab = (plan->flags & MISTRALRS_DEEPGEMM_PLAN_SWAP_AB) != 0;
    auto* runtime = deep_gemm::jit::getGlobalCompiler().build(
        plan->n, plan->k, plan->block_m, plan->block_n, plan->block_k, 1, plan->num_stages,
        plan->num_tma_multicast, deep_gemm::GemmType::Normal, swap_ab);
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
    const MistralrsDeepGemmPrepared* prepared, const void* activation_bf16,
    const void* weight_e4m3, const float* weight_scales, void* output_bf16,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  last_error.clear();
  try {
    if (prepared == nullptr || prepared->function == 0 ||
        prepared->plan.abi_version != kAbiVersion) {
      last_error = "DeepGEMM launch received an invalid prepared plan";
      return MISTRALRS_DEEPGEMM_INVALID_ARGUMENT;
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

    auto function = reinterpret_cast<CUfunction>(prepared->function);

    auto* workspace_bytes_ptr = static_cast<std::byte*>(workspace);
    size_t activation_elements = static_cast<size_t>(plan->m) * plan->k;
    size_t activation_bytes = alignUp(activation_elements, kWorkspaceAlignment);
    auto* activation_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(workspace_bytes_ptr);
    auto* activation_scales = reinterpret_cast<float*>(workspace_bytes_ptr + activation_bytes);
    uint64_t scale_count = static_cast<uint64_t>(plan->m) * (plan->k / kBlockSize);
    uint32_t blocks = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(plan->sm_count) * 8,
                           std::max<uint64_t>(1, (scale_count + 7) / 8)));
    quantizeBf16ToFp8E4m3<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation_bf16), activation_fp8, activation_scales,
        plan->m, plan->k, static_cast<uint32_t>(alignUp(plan->m, 4)));
    auto cuda_status = cudaPeekAtLastError();
    if (cuda_status != cudaSuccess) {
      return cudaFailure("DeepGEMM activation quantization launch", cuda_status);
    }

    bool swap_ab = (plan->flags & MISTRALRS_DEEPGEMM_PLAN_SWAP_AB) != 0;
    if (swap_ab) {
      launchNormalGemmSwapAB(
          function, const_cast<void*>(weight_e4m3), plan->k, activation_fp8, plan->k, output_bf16,
          plan->n, const_cast<float*>(weight_scales), activation_scales, plan->n, plan->m, plan->k,
          plan->block_m, plan->block_n, plan->block_k, plan->num_tma_multicast,
          reinterpret_cast<CUstream>(stream), plan->sm_count, plan->smem_bytes);
    } else {
      launchNormalGemm(
          function, activation_fp8, plan->k, const_cast<void*>(weight_e4m3), plan->k, output_bf16,
          plan->n, activation_scales, const_cast<float*>(weight_scales), plan->m, plan->n, plan->k,
          plan->block_m, plan->block_n, plan->block_k, plan->num_tma_multicast,
          reinterpret_cast<CUstream>(stream), plan->sm_count, plan->smem_bytes);
    }
    return MISTRALRS_DEEPGEMM_SUCCESS;
  } catch (const std::exception& error) {
    return classifyException(error);
  } catch (...) {
    last_error = "Unknown DeepGEMM launch error";
    return MISTRALRS_DEEPGEMM_INTERNAL_ERROR;
  }
}
