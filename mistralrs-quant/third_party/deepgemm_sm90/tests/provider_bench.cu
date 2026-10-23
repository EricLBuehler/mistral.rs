// SPDX-License-Identifier: Apache-2.0

#include "../mistralrs_deepgemm_sm90.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct Shape {
  const char* name;
  uint32_t n;
  uint32_t k;
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

bool benchmark(const Shape& shape, uint32_t m, const char* include_dir, cudaStream_t stream) {
  MistralrsDeepGemmPlan plan{};
  MistralrsDeepGemmPrepared prepared{};
  if (!checkProvider(mistralrs_deepgemm_sm90_plan(m, shape.n, shape.k, &plan), "plan") ||
      !checkProvider(mistralrs_deepgemm_sm90_prepare(&plan, include_dir, stream, &prepared),
                     "prepare")) {
    return false;
  }

  void* activation = nullptr;
  void* weight = nullptr;
  void* weight_scales = nullptr;
  void* output = nullptr;
  void* workspace = nullptr;
  size_t activation_bytes = static_cast<size_t>(m) * shape.k * 2;
  size_t weight_bytes = static_cast<size_t>(shape.n) * shape.k;
  size_t scale_bytes =
      static_cast<size_t>(shape.n / 128) * (shape.k / 128) * sizeof(float);
  size_t output_bytes = static_cast<size_t>(m) * shape.n * 2;
  if (!checkCuda(cudaMalloc(&activation, activation_bytes), "allocate activation") ||
      !checkCuda(cudaMalloc(&weight, weight_bytes), "allocate weight") ||
      !checkCuda(cudaMalloc(&weight_scales, scale_bytes), "allocate scales") ||
      !checkCuda(cudaMalloc(&output, output_bytes), "allocate output") ||
      !checkCuda(cudaMalloc(&workspace, plan.workspace_bytes), "allocate workspace") ||
      !checkCuda(cudaMemsetAsync(activation, 0, activation_bytes, stream), "clear activation") ||
      !checkCuda(cudaMemsetAsync(weight, 0, weight_bytes, stream), "clear weight") ||
      !checkCuda(cudaMemsetAsync(weight_scales, 0, scale_bytes, stream), "clear scales")) {
    return false;
  }

  for (int iteration = 0; iteration < 10; ++iteration) {
    if (!checkProvider(mistralrs_deepgemm_sm90_gemm(
                           &prepared, m, activation, weight,
                           static_cast<float*>(weight_scales), output, workspace,
                           plan.workspace_bytes, stream),
                       "warmup")) {
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
  constexpr int iterations = 100;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (!checkProvider(mistralrs_deepgemm_sm90_gemm(
                           &prepared, m, activation, weight,
                           static_cast<float*>(weight_scales), output, workspace,
                           plan.workspace_bytes, stream),
                       "benchmark")) {
      return false;
    }
  }
  if (!checkCuda(cudaEventRecord(stop, stream), "record stop") ||
      !checkCuda(cudaEventSynchronize(stop), "synchronize stop")) {
    return false;
  }
  float milliseconds = 0.0f;
  if (!checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time")) {
    return false;
  }
  std::printf("%-8s M=%2u N=%5u K=%5u %.3f us\n", shape.name, m, shape.n, shape.k,
              milliseconds * 1000.0f / iterations);

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(workspace);
  cudaFree(output);
  cudaFree(weight_scales);
  cudaFree(weight);
  cudaFree(activation);
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
  for (uint32_t m : {1U, 8U, 16U, 21U, 24U, 128U, 192U, 256U, 512U, 2304U, 3840U, 4096U}) {
    for (const auto& shape : shapes) {
      if (!benchmark(shape, m, include_dir, stream)) {
        return 1;
      }
    }
  }
  cudaStreamDestroy(stream);
  return 0;
}
