/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>

#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"

struct GdnPrefillSm90Params {
  const void *mixed_qkv;
  const void *b;
  const void *a;
  const float *a_log;
  const float *dt_bias;
  float *state;
  const uint32_t *slots;
  void *output;
  void *workspace;
  uint64_t workspace_bytes;
  int32_t batch_size;
  int32_t seq_len;
  int32_t num_k_heads;
  int32_t num_v_heads;
  int32_t sm_count;
  int64_t stream;
};

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr float kNormEpsilon = 1.0e-6f;
constexpr size_t kWorkspaceAlignment = 256;
constexpr size_t kTmaDescriptorBytes = 128;
constexpr uint32_t kPaddingSlot = UINT32_MAX;

static_assert(sizeof(cute::TmaDescriptor) == kTmaDescriptorBytes);

struct WorkspaceLayout {
  size_t q;
  size_t k;
  size_t v;
  size_t alpha;
  size_t beta;
  size_t packed_state;
  size_t cu_seqlens;
  size_t scheduler;
  size_t bytes;
};

size_t align_up(size_t value) {
  return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

size_t append(size_t &cursor, size_t bytes) {
  cursor = align_up(cursor);
  const size_t offset = cursor;
  cursor += bytes;
  return offset;
}

WorkspaceLayout make_workspace_layout(int batch_size, int seq_len,
                                      int num_k_heads, int num_v_heads,
                                      int sm_count, bool pack_state) {
  const size_t tokens = static_cast<size_t>(batch_size) * seq_len;
  const size_t key_elements = tokens * num_k_heads * kHeadDim;
  const size_t value_elements = tokens * num_v_heads * kHeadDim;
  const size_t gate_elements = tokens * num_v_heads;
  const size_t state_elements =
      static_cast<size_t>(batch_size) * num_v_heads * kHeadDim * kHeadDim;
  size_t cursor = 0;
  WorkspaceLayout layout{};
  layout.q = append(cursor, key_elements * sizeof(__nv_bfloat16));
  layout.k = append(cursor, key_elements * sizeof(__nv_bfloat16));
  layout.v = append(cursor, value_elements * sizeof(__nv_bfloat16));
  layout.alpha = append(cursor, gate_elements * sizeof(float));
  layout.beta = append(cursor, gate_elements * sizeof(float));
  if (pack_state) {
    layout.packed_state = append(cursor, state_elements * sizeof(float));
  }
  layout.cu_seqlens =
      append(cursor, static_cast<size_t>(batch_size + 1) * sizeof(int64_t));
  layout.scheduler =
      append(cursor, static_cast<size_t>(sm_count) * kTmaDescriptorBytes);
  layout.bytes = align_up(cursor);
  return layout;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ bool is_padding(const uint32_t *slots,
                                           int batch) {
  return slots != nullptr && slots[batch] == kPaddingSlot;
}

__global__ void prepare_inputs_kernel(
    const __nv_bfloat16 *__restrict__ mixed_qkv,
    const __nv_bfloat16 *__restrict__ b,
    const __nv_bfloat16 *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, const uint32_t *__restrict__ slots,
    __nv_bfloat16 *__restrict__ q, __nv_bfloat16 *__restrict__ k,
    __nv_bfloat16 *__restrict__ v, float *__restrict__ alpha,
    float *__restrict__ beta, int64_t *__restrict__ cu_seqlens,
    int total_tokens, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads) {
  const int token = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread / kWarpSize;
  const int lane = thread % kWarpSize;
  if (token >= total_tokens) {
    return;
  }
  const bool padding = is_padding(slots, token / seq_len);
  const int key_dim = num_k_heads * kHeadDim;
  const int conv_dim = 2 * key_dim + num_v_heads * kHeadDim;
  const __nv_bfloat16 *row =
      mixed_qkv + static_cast<int64_t>(token) * conv_dim;

  for (int head = warp; head < num_k_heads; head += kWarpsPerBlock) {
    const int64_t output_base =
        (static_cast<int64_t>(token) * num_k_heads + head) * kHeadDim;
    if (padding) {
      for (int feature = lane; feature < kHeadDim; feature += kWarpSize) {
        q[output_base + feature] = __float2bfloat16_rn(0.0f);
        k[output_base + feature] = __float2bfloat16_rn(0.0f);
      }
      continue;
    }
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int feature = lane; feature < kHeadDim; feature += kWarpSize) {
      const float q_value =
          static_cast<float>(row[head * kHeadDim + feature]);
      const float k_value =
          static_cast<float>(row[key_dim + head * kHeadDim + feature]);
      q_norm = fmaf(q_value, q_value, q_norm);
      k_norm = fmaf(k_value, k_value, k_norm);
    }
    q_norm = __shfl_sync(0xffffffff, warp_sum(q_norm), 0);
    k_norm = __shfl_sync(0xffffffff, warp_sum(k_norm), 0);
    const float q_scale = rsqrtf(q_norm + kNormEpsilon);
    const float k_scale = rsqrtf(k_norm + kNormEpsilon);
    for (int feature = lane; feature < kHeadDim; feature += kWarpSize) {
      q[output_base + feature] = __float2bfloat16_rn(
          static_cast<float>(row[head * kHeadDim + feature]) * q_scale);
      k[output_base + feature] = __float2bfloat16_rn(
          static_cast<float>(row[key_dim + head * kHeadDim + feature]) *
          k_scale);
    }
  }

  const int value_elements = num_v_heads * kHeadDim;
  const int64_t value_output_base =
      static_cast<int64_t>(token) * value_elements;
  const int64_t value_input_base = 2 * key_dim;
  for (int index = thread; index < value_elements; index += blockDim.x) {
    v[value_output_base + index] =
        padding ? __float2bfloat16_rn(0.0f) : row[value_input_base + index];
  }

  for (int head = thread; head < num_v_heads; head += blockDim.x) {
    const int64_t gate_index =
        static_cast<int64_t>(token) * num_v_heads + head;
    if (padding) {
      alpha[gate_index] = 1.0f;
      beta[gate_index] = 0.0f;
      continue;
    }
    const float b_value = static_cast<float>(b[gate_index]);
    const float a_value = static_cast<float>(a[gate_index]) + dt_bias[head];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    alpha[gate_index] = expf(-expf(a_log[head]) * softplus);
    beta[gate_index] = 1.0f / (1.0f + expf(-b_value));
  }

  if (token == 0) {
    for (int batch = thread; batch <= batch_size; batch += blockDim.x) {
      cu_seqlens[batch] = static_cast<int64_t>(batch) * seq_len;
    }
  }
}

__global__ void gather_state_kernel(const float *__restrict__ state,
                                    const uint32_t *__restrict__ slots,
                                    float *__restrict__ packed_state,
                                    int64_t row_elements,
                                    int64_t total_elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int batch = static_cast<int>(index / row_elements);
    const int64_t row_offset = index % row_elements;
    const uint32_t slot = slots == nullptr ? static_cast<uint32_t>(batch)
                                           : slots[batch];
    packed_state[index] =
        slot == kPaddingSlot ? 0.0f : state[slot * row_elements + row_offset];
  }
}

__global__ void scatter_state_kernel(const float *__restrict__ packed_state,
                                     const uint32_t *__restrict__ slots,
                                     float *__restrict__ state,
                                     int64_t row_elements,
                                     int64_t total_elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total_elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int batch = static_cast<int>(index / row_elements);
    const int64_t row_offset = index % row_elements;
    const uint32_t slot = slots == nullptr ? static_cast<uint32_t>(batch)
                                           : slots[batch];
    if (slot != kPaddingSlot) {
      state[slot * row_elements + row_offset] = packed_state[index];
    }
  }
}

int launch_blocks(int64_t elements, int threads) {
  return static_cast<int>((elements + threads - 1) / threads);
}

}  // namespace

extern "C" uint64_t mistralrs_flashinfer_gdn_sm90_workspace_size(
    int32_t batch_size, int32_t seq_len, int32_t num_k_heads,
    int32_t num_v_heads, int32_t sm_count, int32_t has_slots) {
  if (batch_size <= 0 || seq_len <= 0 || num_k_heads <= 0 ||
      num_v_heads < num_k_heads || num_v_heads % num_k_heads != 0 ||
      sm_count <= 0 || (has_slots != 0 && has_slots != 1)) {
    return 0;
  }
  return make_workspace_layout(batch_size, seq_len, num_k_heads, num_v_heads,
                               sm_count, has_slots != 0)
      .bytes;
}

extern "C" int mistralrs_flashinfer_gdn_sm90_launch(
    const GdnPrefillSm90Params *params) {
  if (params == nullptr || params->mixed_qkv == nullptr || params->b == nullptr ||
      params->a == nullptr || params->a_log == nullptr ||
      params->dt_bias == nullptr || params->state == nullptr ||
      params->output == nullptr || params->workspace == nullptr ||
      params->batch_size <= 0 || params->seq_len <= 0 ||
      params->num_k_heads <= 0 ||
      params->num_v_heads < params->num_k_heads ||
      params->num_v_heads % params->num_k_heads != 0 ||
      params->sm_count <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const bool pack_state = params->slots != nullptr;
  const WorkspaceLayout layout = make_workspace_layout(
      params->batch_size, params->seq_len, params->num_k_heads,
      params->num_v_heads, params->sm_count, pack_state);
  if (params->workspace_bytes < layout.bytes) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto *workspace = static_cast<uint8_t *>(params->workspace);
  auto *q = reinterpret_cast<__nv_bfloat16 *>(workspace + layout.q);
  auto *k = reinterpret_cast<__nv_bfloat16 *>(workspace + layout.k);
  auto *v = reinterpret_cast<__nv_bfloat16 *>(workspace + layout.v);
  auto *alpha = reinterpret_cast<float *>(workspace + layout.alpha);
  auto *beta = reinterpret_cast<float *>(workspace + layout.beta);
  // Gathered value-major state already matches FlashInfer's K-contiguous layout.
  float *kernel_state = params->state;
  if (pack_state) {
    kernel_state = reinterpret_cast<float *>(workspace + layout.packed_state);
  }
  auto *cu_seqlens =
      reinterpret_cast<int64_t *>(workspace + layout.cu_seqlens);
  auto *scheduler = workspace + layout.scheduler;
  const int total_tokens = params->batch_size * params->seq_len;
  const int64_t state_row_elements = static_cast<int64_t>(params->num_v_heads) *
                                     kHeadDim * kHeadDim;
  const int64_t state_elements =
      static_cast<int64_t>(params->batch_size) * state_row_elements;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(params->stream);

  prepare_inputs_kernel<<<total_tokens, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(params->mixed_qkv),
      static_cast<const __nv_bfloat16 *>(params->b),
      static_cast<const __nv_bfloat16 *>(params->a), params->a_log,
      params->dt_bias, params->slots, q, k, v, alpha, beta, cu_seqlens,
      total_tokens, params->batch_size, params->seq_len, params->num_k_heads,
      params->num_v_heads);
  if (pack_state) {
    gather_state_kernel<<<launch_blocks(state_elements, kThreads), kThreads, 0,
                          stream>>>(
        params->state, params->slots, kernel_state, state_row_elements,
        state_elements);
  }
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  try {
    flat::launch_delta_rule_prefill_kernel_gbai<
        true, true, true, true, false, cutlass::arch::Sm90, nv_bfloat16,
        nv_bfloat16, float>(
        stream, static_cast<__nv_bfloat16 *>(params->output), kernel_state, q,
        k, v, kernel_state, alpha, beta, cu_seqlens, scheduler,
        params->batch_size, params->num_k_heads, params->num_k_heads,
        params->num_v_heads, params->num_v_heads, kHeadDim, total_tokens,
        1.0f / sqrtf(static_cast<float>(kHeadDim)), params->sm_count, nullptr,
        nullptr, 0);
  } catch (const std::exception &) {
    return -1;
  }
  status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  if (pack_state) {
    scatter_state_kernel<<<launch_blocks(state_elements, kThreads), kThreads,
                           0, stream>>>(
        kernel_state, params->slots, params->state, state_row_elements,
        state_elements);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
