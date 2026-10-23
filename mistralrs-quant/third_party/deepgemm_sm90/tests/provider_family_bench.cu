// SPDX-License-Identifier: Apache-2.0

#include "../mistralrs_deepgemm_sm90.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef MISTRALRS_DEEPGEMM_ENABLE_LEGACY_DIAGNOSTICS
#error "provider_family_bench requires the test-only legacy diagnostics build"
#endif

namespace {

constexpr uint32_t kM = 512;
constexpr int kWarmupIterations = 10;
constexpr int kBenchmarkIterations = 100;

struct Shape {
  const char* name;
  uint32_t n;
  uint32_t k;
};

struct Allocation {
  void* activation = nullptr;
  void* activation_scales = nullptr;
  void* weight = nullptr;
  void* weight_scales = nullptr;
  void* output = nullptr;

  ~Allocation() {
    cudaFree(output);
    cudaFree(weight_scales);
    cudaFree(weight);
    cudaFree(activation_scales);
    cudaFree(activation);
  }
};

bool checkCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool checkProvider(int32_t status, const char* operation) {
  if (status == MISTRALRS_DEEPGEMM_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s: %s: %s\n", operation,
               mistralrs_deepgemm_sm90_error_string(status),
               mistralrs_deepgemm_sm90_last_error());
  return false;
}

bool allocate(const Shape& shape, cudaStream_t stream, Allocation* allocation) {
  const size_t activation_bytes = static_cast<size_t>(kM) * shape.k;
  const size_t activation_scale_bytes =
      static_cast<size_t>(shape.k / 128) * kM * sizeof(float);
  const size_t weight_bytes = static_cast<size_t>(shape.n) * shape.k;
  const size_t weight_scale_bytes =
      static_cast<size_t>(shape.n / 128) * (shape.k / 128) * sizeof(float);
  const size_t output_bytes = static_cast<size_t>(kM) * shape.n * 2;
  return checkCuda(cudaMalloc(&allocation->activation, activation_bytes),
                   "allocate activation") &&
         checkCuda(cudaMalloc(&allocation->activation_scales, activation_scale_bytes),
                   "allocate activation scales") &&
         checkCuda(cudaMalloc(&allocation->weight, weight_bytes), "allocate weight") &&
         checkCuda(cudaMalloc(&allocation->weight_scales, weight_scale_bytes),
                   "allocate weight scales") &&
         checkCuda(cudaMalloc(&allocation->output, output_bytes), "allocate output") &&
         checkCuda(cudaMemsetAsync(allocation->activation, 0, activation_bytes, stream),
                   "clear activation") &&
         checkCuda(cudaMemsetAsync(allocation->activation_scales, 0,
                                   activation_scale_bytes, stream),
                   "clear activation scales") &&
         checkCuda(cudaMemsetAsync(allocation->weight, 0, weight_bytes, stream),
                   "clear weight") &&
         checkCuda(cudaMemsetAsync(allocation->weight_scales, 0,
                                   weight_scale_bytes, stream),
                   "clear weight scales");
}

bool benchmarkFamily(const Shape& shape, const char* family, bool legacy,
                     const char* include_dir, cudaStream_t stream,
                     const Allocation& allocation, float* microseconds) {
  MistralrsDeepGemmPlan plan{};
  MistralrsDeepGemmPrepared prepared{};
  const int32_t plan_status =
      legacy ? mistralrs_deepgemm_sm90_plan_legacy_for_test(kM, shape.n, shape.k, &plan)
             : mistralrs_deepgemm_sm90_plan(kM, shape.n, shape.k, &plan);
  if (!checkProvider(plan_status, "plan") ||
      !checkProvider(mistralrs_deepgemm_sm90_prepare(&plan, include_dir, stream, &prepared),
                     "prepare")) {
    return false;
  }

  auto launch = [&]() {
    return checkProvider(mistralrs_deepgemm_sm90_gemm_prequantized(
                             &prepared, kM, allocation.activation,
                             static_cast<const float*>(allocation.activation_scales), kM,
                             allocation.weight,
                             static_cast<const float*>(allocation.weight_scales),
                             allocation.output, stream),
                         "gemm");
  };
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    if (!launch()) {
      return false;
    }
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!checkCuda(cudaEventCreate(&start), "create start event") ||
      !checkCuda(cudaEventCreate(&stop), "create stop event") ||
      !checkCuda(cudaEventRecord(start, stream), "record start")) {
    return false;
  }
  for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
    if (!launch()) {
      return false;
    }
  }
  float milliseconds = 0;
  const bool success = checkCuda(cudaEventRecord(stop, stream), "record stop") &&
                       checkCuda(cudaEventSynchronize(stop), "synchronize stop") &&
                       checkCuda(cudaEventElapsedTime(&milliseconds, start, stop),
                                 "elapsed time");
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  if (!success) {
    return false;
  }

  *microseconds = milliseconds * 1000.0f / kBenchmarkIterations;
  std::printf("%-8s %-8s M=%u N=%5u K=%5u %8.3f us BM=%3u BN=%3u stages=%2u cluster=%u\n",
              shape.name, family, kM, shape.n, shape.k, *microseconds, plan.block_m,
              plan.block_n, plan.num_stages, plan.num_tma_multicast);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* include_dir = argc > 1 ? argv[1] : nullptr;
  cudaStream_t stream = nullptr;
  if (!checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream")) {
    return 1;
  }
  const std::vector<Shape> shapes = {
      {"gdn", 16384, 5120},
      {"attn", 14336, 5120},
      {"output", 5120, 6144},
      {"gate_up", 34816, 5120},
      {"down", 5120, 17408},
  };
  for (const auto& shape : shapes) {
    Allocation allocation;
    if (!allocate(shape, stream, &allocation)) {
      return 1;
    }
    float legacy = 0;
    float official = 0;
    if (!benchmarkFamily(shape, "legacy", true, include_dir, stream, allocation, &legacy) ||
        !benchmarkFamily(shape, "official", false, include_dir, stream, allocation,
                         &official)) {
      return 1;
    }
    std::printf("%-8s speedup %.3fx (%+.1f%%)\n", shape.name, legacy / official,
                (legacy / official - 1.0f) * 100.0f);
  }
  cudaStreamDestroy(stream);
  return 0;
}
