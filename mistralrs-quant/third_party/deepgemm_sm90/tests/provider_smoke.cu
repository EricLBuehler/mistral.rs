// SPDX-License-Identifier: Apache-2.0

#include "../mistralrs_deepgemm_sm90.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool cudaOk(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
  return false;
}

bool providerOk(int32_t status, const char* operation) {
  if (status == MISTRALRS_DEEPGEMM_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s: %s: %s\n", operation,
               mistralrs_deepgemm_sm90_error_string(status),
               mistralrs_deepgemm_sm90_last_error());
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const char* include_dir = argc > 1 ? argv[1] : nullptr;
  cudaStream_t stream = nullptr;
  if (!cudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
    return 1;
  }

  MistralrsDeepGemmPlan plan{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(1, 128, 128, &plan), "plan")) {
    return 1;
  }
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_SWAP_AB) == 0 || plan.workspace_bytes == 0) {
    std::fprintf(stderr, "unexpected small-M plan\n");
    return 1;
  }

  MistralrsDeepGemmPlan captured_plan{};
  MistralrsDeepGemmPrepared captured_prepared{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(2, 128, 128, &captured_plan), "captured plan")) {
    return 1;
  }
  if (!cudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture")) {
    return 1;
  }
  int32_t capture_status =
      mistralrs_deepgemm_sm90_prepare(&captured_plan, include_dir, stream,
                                      &captured_prepared);
  if (capture_status != MISTRALRS_DEEPGEMM_CAPTURE_ACTIVE) {
    std::fprintf(stderr, "prepare during capture returned %d\n", capture_status);
    return 1;
  }
  cudaGraph_t empty_graph = nullptr;
  if (!cudaOk(cudaStreamEndCapture(stream, &empty_graph), "cudaStreamEndCapture")) {
    return 1;
  }
  if (empty_graph != nullptr) {
    cudaGraphDestroy(empty_graph);
  }

  MistralrsDeepGemmPrepared prepared{};
  if (!providerOk(mistralrs_deepgemm_sm90_prepare(&plan, include_dir, stream, &prepared),
                  "prepare")) {
    return 1;
  }

  std::vector<__nv_bfloat16> host_activation(128, __float2bfloat16(1.0f));
  std::vector<__nv_fp8_e4m3> host_weight(128 * 128, __nv_fp8_e4m3(1.0f));
  float host_scale = 1.0f;
  std::vector<__nv_bfloat16> host_output(128);

  __nv_bfloat16* activation = nullptr;
  __nv_fp8_e4m3* weight = nullptr;
  float* weight_scale = nullptr;
  __nv_bfloat16* output = nullptr;
  void* workspace = nullptr;
  if (!cudaOk(cudaMalloc(&activation, host_activation.size() * sizeof(*activation)),
              "cudaMalloc activation") ||
      !cudaOk(cudaMalloc(&weight, host_weight.size() * sizeof(*weight)), "cudaMalloc weight") ||
      !cudaOk(cudaMalloc(&weight_scale, sizeof(*weight_scale)), "cudaMalloc scale") ||
      !cudaOk(cudaMalloc(&output, host_output.size() * sizeof(*output)), "cudaMalloc output") ||
      !cudaOk(cudaMalloc(&workspace, plan.workspace_bytes), "cudaMalloc workspace")) {
    return 1;
  }
  if (!cudaOk(cudaMemcpyAsync(activation, host_activation.data(),
                              host_activation.size() * sizeof(*activation),
                              cudaMemcpyHostToDevice, stream),
              "copy activation") ||
      !cudaOk(cudaMemcpyAsync(weight, host_weight.data(), host_weight.size() * sizeof(*weight),
                              cudaMemcpyHostToDevice, stream),
              "copy weight") ||
      !cudaOk(cudaMemcpyAsync(weight_scale, &host_scale, sizeof(host_scale),
                              cudaMemcpyHostToDevice, stream),
              "copy scale")) {
    return 1;
  }
  if (!providerOk(mistralrs_deepgemm_sm90_gemm(
                      &prepared, plan.m, activation, weight, weight_scale, output, workspace,
                      plan.workspace_bytes, stream),
                  "gemm") ||
      !cudaOk(cudaMemcpyAsync(host_output.data(), output,
                              host_output.size() * sizeof(*output), cudaMemcpyDeviceToHost,
                              stream),
              "copy output") ||
      !cudaOk(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
    return 1;
  }
  for (auto value : host_output) {
    if (std::fabs(__bfloat162float(value) - 128.0f) > 1.0f) {
      std::fprintf(stderr, "incorrect output: %f\n", __bfloat162float(value));
      return 1;
    }
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (!cudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "graph begin") ||
      !providerOk(mistralrs_deepgemm_sm90_gemm(
                      &prepared, plan.m, activation, weight, weight_scale, output, workspace,
                      plan.workspace_bytes, stream),
                  "captured gemm") ||
      !cudaOk(cudaStreamEndCapture(stream, &graph), "graph end") ||
      !cudaOk(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "cudaGraphInstantiate")) {
    return 1;
  }
  std::fill(host_activation.begin(), host_activation.end(), __float2bfloat16(2.0f));
  std::fill(host_output.begin(), host_output.end(), __float2bfloat16(0.0f));
  if (!cudaOk(cudaMemcpyAsync(activation, host_activation.data(),
                              host_activation.size() * sizeof(*activation),
                              cudaMemcpyHostToDevice, stream),
              "update activation") ||
      !cudaOk(cudaMemsetAsync(output, 0, host_output.size() * sizeof(*output), stream),
              "clear output") ||
      !cudaOk(cudaGraphLaunch(executable, stream), "cudaGraphLaunch") ||
      !cudaOk(cudaMemcpyAsync(host_output.data(), output,
                              host_output.size() * sizeof(*output), cudaMemcpyDeviceToHost,
                              stream),
              "copy replay output") ||
      !cudaOk(cudaStreamSynchronize(stream), "captured synchronize")) {
    return 1;
  }
  for (auto value : host_output) {
    if (std::fabs(__bfloat162float(value) - 256.0f) > 2.0f) {
      std::fprintf(stderr, "incorrect replay output: %f\n", __bfloat162float(value));
      return 1;
    }
  }

  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaFree(workspace);
  cudaFree(output);
  cudaFree(weight_scale);
  cudaFree(weight);
  cudaFree(activation);
  cudaStreamDestroy(stream);
  return 0;
}
