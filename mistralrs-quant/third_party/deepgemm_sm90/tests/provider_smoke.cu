// SPDX-License-Identifier: Apache-2.0

#include "../mistralrs_deepgemm_sm90.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace {

constexpr uint32_t kM = 1;
constexpr uint32_t kN = 5120;
constexpr uint32_t kK = 6144;
constexpr uint32_t kGateN = 34816;
constexpr uint32_t kGateK = 5120;
constexpr uint32_t kAttentionN = 14336;
constexpr uint32_t kGdnN = 16384;
constexpr uint32_t kDownK = 17408;
constexpr uint32_t kGenericN = 7168;
constexpr uint32_t kGenericK = 4096;
constexpr uint32_t kOutsideWaveN = 12288;
constexpr uint32_t kWaveBalancedBlockM = 64;
constexpr uint32_t kWaveBalancedBlockN = 8;
constexpr uint32_t kWaveBalancedStages = 12;
constexpr uint32_t kScaleBlock = MISTRALRS_DEEPGEMM_BLOCK_SIZE;
constexpr uint32_t kScaleStrideM = MISTRALRS_DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT;

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

bool selectionOk(uint32_t m, uint32_t n, uint32_t k, bool expect_small_m) {
  MistralrsDeepGemmPlan plan{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(m, n, k, &plan), "selection plan")) {
    return false;
  }
  const bool small_m = (plan.flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0;
  const bool official = (plan.flags & MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D) != 0;
  if (small_m != expect_small_m || official == expect_small_m) {
    std::fprintf(stderr,
                 "unexpected M=%u N=%u K=%u selection: flags=%u BM=%u BN=%u stages=%u\n",
                 m, n, k, plan.flags, plan.block_m, plan.block_n, plan.num_stages);
    return false;
  }
  return true;
}

bool waveBalancedSelectionOk() {
  MistralrsDeepGemmPlan plan{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(8, kGenericN, kGenericK, &plan),
                  "wave-balanced plan")) {
    return false;
  }
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) == 0 ||
      plan.block_m != kWaveBalancedBlockM || plan.block_n != kWaveBalancedBlockN ||
      plan.num_stages != kWaveBalancedStages) {
    std::fprintf(stderr, "unexpected wave-balanced plan\n");
    return false;
  }
  return selectionOk(9, kGenericN, kGenericK, false) &&
         selectionOk(8, kOutsideWaveN, kGenericK, false);
}

bool officialTileOk(uint32_t m, uint32_t block_m, uint32_t cluster_size) {
  MistralrsDeepGemmPlan plan{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(m, kGateN, kGateK, &plan),
                  "official tile plan")) {
    return false;
  }
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D) == 0 ||
      (plan.flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) != 0 ||
      plan.block_m != block_m || plan.block_n != 144 ||
      plan.num_tma_multicast != cluster_size ||
      (plan.flags & MISTRALRS_DEEPGEMM_PLAN_MULTICAST_ON_A) != 0) {
    std::fprintf(stderr,
                 "unexpected gate M=%u tile: flags=%u BM=%u BN=%u cluster=%u\n",
                 m, plan.flags, plan.block_m, plan.block_n,
                 plan.num_tma_multicast);
    return false;
  }
  return true;
}

bool cpuReferenceOk(const std::vector<__nv_fp8_e4m3>& activation,
                    const std::vector<float>& activation_scales,
                    const std::vector<__nv_fp8_e4m3>& weight,
                    const std::vector<float>& weight_scales,
                    const std::vector<__nv_bfloat16>& output) {
  const uint32_t k_blocks = kK / kScaleBlock;
  for (uint32_t column = 0; column < kN; ++column) {
    float expected = 0.0f;
    for (uint32_t inner = 0; inner < kK; ++inner) {
      const uint32_t k_block = inner / kScaleBlock;
      const float activation_scale = activation_scales[k_block * kScaleStrideM];
      const float weight_scale = weight_scales[(column / kScaleBlock) * k_blocks + k_block];
      expected += static_cast<float>(activation[inner]) * activation_scale *
                  static_cast<float>(weight[static_cast<size_t>(column) * kK + inner]) *
                  weight_scale;
    }
    const __nv_bfloat16 expected_bf16 = __float2bfloat16_rn(expected);
    if (std::memcmp(&output[column], &expected_bf16, sizeof(expected_bf16)) != 0) {
      std::fprintf(stderr, "CPU reference mismatch at N=%u: expected=%f actual=%f\n",
                   column, __bfloat162float(expected_bf16),
                   __bfloat162float(output[column]));
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* include_dir = argc > 1 ? argv[1] : nullptr;
  cudaStream_t stream = nullptr;
  if (!cudaOk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
    return 1;
  }

  if (!officialTileOk(2, 32, 1) || !officialTileOk(3, 64, 2) ||
      !officialTileOk(8, 64, 2) || !officialTileOk(16, 64, 2) ||
      !selectionOk(17, kGateN, kGateK, false) ||
      !selectionOk(8, kN, kK, true) || !selectionOk(9, kN, kK, false) ||
      !selectionOk(8, kN, kDownK, true) ||
      !selectionOk(9, kN, kDownK, false) ||
      !selectionOk(16, kAttentionN, kGateK, true) ||
      !selectionOk(17, kAttentionN, kGateK, false) ||
      !selectionOk(16, kGdnN, kGateK, true) ||
      !selectionOk(17, kGdnN, kGateK, false) || !waveBalancedSelectionOk()) {
    return 1;
  }

  MistralrsDeepGemmPlan plan{};
  if (!providerOk(mistralrs_deepgemm_sm90_plan(kM, kN, kK, &plan), "plan")) {
    return 1;
  }
  if ((plan.flags & MISTRALRS_DEEPGEMM_PLAN_SMALL_M_SWAP_AB) == 0 ||
      (plan.flags & MISTRALRS_DEEPGEMM_PLAN_OFFICIAL_1D2D) != 0 || plan.block_n != 8 ||
      plan.workspace_bytes == 0) {
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

  std::vector<__nv_bfloat16> host_activation(kK, __float2bfloat16(1.0f));
  std::vector<__nv_fp8_e4m3> host_activation_fp8(kK, __nv_fp8_e4m3(1.0f));
  std::vector<float> host_activation_scales((kK / kScaleBlock) * kScaleStrideM, 1.0f);
  std::vector<__nv_fp8_e4m3> host_weight(static_cast<size_t>(kN) * kK,
                                         __nv_fp8_e4m3(1.0f));
  std::vector<float> host_weight_scales(
      static_cast<size_t>(kN / kScaleBlock) * (kK / kScaleBlock), 1.0f);
  std::vector<__nv_bfloat16> host_output(kN);
  std::vector<__nv_bfloat16> host_prequantized_output(kN);

  __nv_bfloat16* activation = nullptr;
  __nv_fp8_e4m3* activation_fp8 = nullptr;
  float* activation_scales = nullptr;
  __nv_fp8_e4m3* weight = nullptr;
  float* weight_scale = nullptr;
  __nv_bfloat16* output = nullptr;
  __nv_bfloat16* prequantized_output = nullptr;
  void* workspace = nullptr;
  if (!cudaOk(cudaMalloc(&activation, host_activation.size() * sizeof(*activation)),
              "cudaMalloc activation") ||
      !cudaOk(cudaMalloc(&activation_fp8,
                         host_activation_fp8.size() * sizeof(*activation_fp8)),
              "cudaMalloc FP8 activation") ||
      !cudaOk(cudaMalloc(&activation_scales,
                         host_activation_scales.size() * sizeof(*activation_scales)),
              "cudaMalloc activation scales") ||
      !cudaOk(cudaMalloc(&weight, host_weight.size() * sizeof(*weight)), "cudaMalloc weight") ||
      !cudaOk(cudaMalloc(&weight_scale,
                         host_weight_scales.size() * sizeof(*weight_scale)),
              "cudaMalloc scale") ||
      !cudaOk(cudaMalloc(&output, host_output.size() * sizeof(*output)), "cudaMalloc output") ||
      !cudaOk(cudaMalloc(&prequantized_output,
                         host_prequantized_output.size() * sizeof(*prequantized_output)),
              "cudaMalloc prequantized output") ||
      !cudaOk(cudaMalloc(&workspace, plan.workspace_bytes), "cudaMalloc workspace")) {
    return 1;
  }
  if (!cudaOk(cudaMemcpyAsync(activation, host_activation.data(),
                              host_activation.size() * sizeof(*activation),
                              cudaMemcpyHostToDevice, stream),
              "copy activation") ||
      !cudaOk(cudaMemcpyAsync(activation_fp8, host_activation_fp8.data(),
                              host_activation_fp8.size() * sizeof(*activation_fp8),
                              cudaMemcpyHostToDevice, stream),
              "copy FP8 activation") ||
      !cudaOk(cudaMemcpyAsync(activation_scales, host_activation_scales.data(),
                              host_activation_scales.size() * sizeof(*activation_scales),
                              cudaMemcpyHostToDevice, stream),
              "copy activation scales") ||
      !cudaOk(cudaMemcpyAsync(weight, host_weight.data(), host_weight.size() * sizeof(*weight),
                              cudaMemcpyHostToDevice, stream),
              "copy weight") ||
      !cudaOk(cudaMemcpyAsync(weight_scale, host_weight_scales.data(),
                              host_weight_scales.size() * sizeof(*weight_scale),
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
    if (std::fabs(__bfloat162float(value) - static_cast<float>(kK)) > 2.0f) {
      std::fprintf(stderr, "incorrect output: %f\n", __bfloat162float(value));
      return 1;
    }
  }

  int32_t invalid_stride_status = mistralrs_deepgemm_sm90_gemm_prequantized(
      &prepared, plan.m, activation_fp8, activation_scales, plan.m, weight, weight_scale,
      output, stream);
  if (invalid_stride_status != MISTRALRS_DEEPGEMM_INVALID_ARGUMENT) {
    std::fprintf(stderr, "invalid activation scale stride returned %d\n", invalid_stride_status);
    return 1;
  }
  if (!providerOk(mistralrs_deepgemm_sm90_gemm_prequantized(
                      &prepared, plan.m, activation_fp8, activation_scales,
                      kScaleStrideM, weight, weight_scale, output, stream),
                  "prequantized gemm") ||
      !cudaOk(cudaMemcpyAsync(host_output.data(), output,
                              host_output.size() * sizeof(*output), cudaMemcpyDeviceToHost,
                              stream),
              "copy prequantized output") ||
      !cudaOk(cudaStreamSynchronize(stream), "prequantized synchronize")) {
    return 1;
  }
  for (auto value : host_output) {
    if (std::fabs(__bfloat162float(value) - static_cast<float>(kK)) > 2.0f) {
      std::fprintf(stderr, "incorrect prequantized output: %f\n", __bfloat162float(value));
      return 1;
    }
  }

  for (uint32_t inner = 0; inner < kK; ++inner) {
    host_activation_fp8[inner] = __nv_fp8_e4m3(static_cast<float>(inner % 8 + 1) / 8.0f);
  }
  for (uint32_t column = 0; column < kN; ++column) {
    for (uint32_t inner = 0; inner < kK; ++inner) {
      host_weight[static_cast<size_t>(column) * kK + inner] =
          __nv_fp8_e4m3(static_cast<float>((column * 3 + inner) % 8 + 1) / 8.0f);
    }
  }
  std::fill(host_activation_scales.begin(), host_activation_scales.end(), 8.0f);
  const uint32_t k_blocks = kK / kScaleBlock;
  const uint32_t n_blocks = kN / kScaleBlock;
  for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
    host_activation_scales[k_block * kScaleStrideM] =
        0.5f / static_cast<float>(1U << (k_block % 4));
  }
  for (uint32_t n_block = 0; n_block < n_blocks; ++n_block) {
    for (uint32_t k_block = 0; k_block < k_blocks; ++k_block) {
      host_weight_scales[n_block * k_blocks + k_block] =
          0.5f / static_cast<float>(1U << ((n_block + k_block) % 4));
    }
  }
  if (!cudaOk(cudaMemcpyAsync(activation_fp8, host_activation_fp8.data(),
                              host_activation_fp8.size() * sizeof(*activation_fp8),
                              cudaMemcpyHostToDevice, stream),
              "copy reference activation") ||
      !cudaOk(cudaMemcpyAsync(activation_scales, host_activation_scales.data(),
                              host_activation_scales.size() * sizeof(*activation_scales),
                              cudaMemcpyHostToDevice, stream),
              "copy reference activation scales") ||
      !cudaOk(cudaMemcpyAsync(weight, host_weight.data(), host_weight.size() * sizeof(*weight),
                              cudaMemcpyHostToDevice, stream),
              "copy reference weight") ||
      !cudaOk(cudaMemcpyAsync(weight_scale, host_weight_scales.data(),
                              host_weight_scales.size() * sizeof(*weight_scale),
                              cudaMemcpyHostToDevice, stream),
              "copy reference weight scales") ||
      !providerOk(mistralrs_deepgemm_sm90_gemm_prequantized(
                      &prepared, plan.m, activation_fp8, activation_scales, kScaleStrideM,
                      weight, weight_scale, output, stream),
                  "CPU reference gemm") ||
      !cudaOk(cudaMemcpyAsync(host_output.data(), output,
                              host_output.size() * sizeof(*output), cudaMemcpyDeviceToHost,
                              stream),
              "copy CPU reference output") ||
      !cudaOk(cudaStreamSynchronize(stream), "CPU reference synchronize") ||
      !cpuReferenceOk(host_activation_fp8, host_activation_scales, host_weight,
                      host_weight_scales, host_output)) {
    return 1;
  }

  std::fill(host_activation_fp8.begin(), host_activation_fp8.end(), __nv_fp8_e4m3(1.0f));
  std::fill(host_activation_scales.begin(), host_activation_scales.end(), 1.0f);
  std::fill(host_weight.begin(), host_weight.end(), __nv_fp8_e4m3(1.0f));
  std::fill(host_weight_scales.begin(), host_weight_scales.end(), 1.0f);
  if (!cudaOk(cudaMemcpyAsync(activation_fp8, host_activation_fp8.data(),
                              host_activation_fp8.size() * sizeof(*activation_fp8),
                              cudaMemcpyHostToDevice, stream),
              "restore FP8 activation") ||
      !cudaOk(cudaMemcpyAsync(activation_scales, host_activation_scales.data(),
                              host_activation_scales.size() * sizeof(*activation_scales),
                              cudaMemcpyHostToDevice, stream),
              "restore activation scales") ||
      !cudaOk(cudaMemcpyAsync(weight, host_weight.data(), host_weight.size() * sizeof(*weight),
                              cudaMemcpyHostToDevice, stream),
              "restore weight") ||
      !cudaOk(cudaMemcpyAsync(weight_scale, host_weight_scales.data(),
                              host_weight_scales.size() * sizeof(*weight_scale),
                              cudaMemcpyHostToDevice, stream),
              "restore weight scales")) {
    return 1;
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  if (!cudaOk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
              "graph begin") ||
      !providerOk(mistralrs_deepgemm_sm90_gemm(
                      &prepared, plan.m, activation, weight, weight_scale, output, workspace,
                      plan.workspace_bytes, stream),
                  "captured gemm") ||
      !providerOk(mistralrs_deepgemm_sm90_gemm_prequantized(
                      &prepared, plan.m, activation_fp8, activation_scales,
                      kScaleStrideM, weight, weight_scale, prequantized_output, stream),
                  "captured prequantized gemm") ||
      !cudaOk(cudaStreamEndCapture(stream, &graph), "graph end") ||
      !cudaOk(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
              "cudaGraphInstantiate")) {
    return 1;
  }
  std::fill(host_activation.begin(), host_activation.end(), __float2bfloat16(2.0f));
  std::fill(host_activation_fp8.begin(), host_activation_fp8.end(), __nv_fp8_e4m3(2.0f));
  std::fill(host_output.begin(), host_output.end(), __float2bfloat16(0.0f));
  std::fill(host_prequantized_output.begin(), host_prequantized_output.end(),
            __float2bfloat16(0.0f));
  if (!cudaOk(cudaMemcpyAsync(activation, host_activation.data(),
                              host_activation.size() * sizeof(*activation),
                              cudaMemcpyHostToDevice, stream),
              "update activation") ||
      !cudaOk(cudaMemcpyAsync(activation_fp8, host_activation_fp8.data(),
                              host_activation_fp8.size() * sizeof(*activation_fp8),
                              cudaMemcpyHostToDevice, stream),
              "update FP8 activation") ||
      !cudaOk(cudaMemsetAsync(output, 0, host_output.size() * sizeof(*output), stream),
              "clear output") ||
      !cudaOk(cudaMemsetAsync(prequantized_output, 0,
                              host_prequantized_output.size() * sizeof(*prequantized_output),
                              stream),
              "clear prequantized output") ||
      !cudaOk(cudaGraphLaunch(executable, stream), "cudaGraphLaunch") ||
      !cudaOk(cudaMemcpyAsync(host_output.data(), output,
                              host_output.size() * sizeof(*output), cudaMemcpyDeviceToHost,
                              stream),
              "copy replay output") ||
      !cudaOk(cudaMemcpyAsync(host_prequantized_output.data(), prequantized_output,
                              host_prequantized_output.size() * sizeof(*prequantized_output),
                              cudaMemcpyDeviceToHost, stream),
              "copy prequantized replay output") ||
      !cudaOk(cudaStreamSynchronize(stream), "captured synchronize")) {
    return 1;
  }
  for (auto value : host_output) {
    if (std::fabs(__bfloat162float(value) - 2.0f * static_cast<float>(kK)) > 4.0f) {
      std::fprintf(stderr, "incorrect replay output: %f\n", __bfloat162float(value));
      return 1;
    }
  }
  for (auto value : host_prequantized_output) {
    if (std::fabs(__bfloat162float(value) - 2.0f * static_cast<float>(kK)) > 4.0f) {
      std::fprintf(stderr, "incorrect prequantized replay output: %f\n",
                   __bfloat162float(value));
      return 1;
    }
  }

  cudaGraphExecDestroy(executable);
  cudaGraphDestroy(graph);
  cudaFree(workspace);
  cudaFree(prequantized_output);
  cudaFree(output);
  cudaFree(weight_scale);
  cudaFree(weight);
  cudaFree(activation_scales);
  cudaFree(activation_fp8);
  cudaFree(activation);
  cudaStreamDestroy(stream);
  return 0;
}
