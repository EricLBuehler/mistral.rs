/*
 * Copyright (c) 2024 by FlashInfer team.
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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace mistralrs_radix_topk {

constexpr uint32_t kBlockThreads = 1024;
constexpr uint32_t kRadix = 256;
constexpr uint32_t kHistogramBuffers = 3;
constexpr uint32_t kMaxCtasPerGroup = 256;
constexpr uint32_t kMinTopK = 8;
constexpr uint32_t kMaxTopK = 32;
constexpr size_t kLaunchSmemHeadroom = 2048;

template <typename T> struct Traits;

template <> struct Traits<float> {
  using Ordered = uint32_t;

  static __device__ __forceinline__ Ordered to_ordered(float value) {
    if (value != value) {
      return neg_inf_ordered();
    }
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  }

  static __device__ __forceinline__ float from_ordered(Ordered ordered) {
    const uint32_t bits = (ordered & 0x80000000u)
                              ? (ordered ^ 0x80000000u)
                              : ~ordered;
    return __uint_as_float(bits);
  }

  static __host__ __device__ constexpr Ordered neg_inf_ordered() {
    return 0x007fffffu;
  }
};

template <> struct Traits<__half> {
  using Ordered = uint16_t;

  static __device__ __forceinline__ Ordered to_ordered(__half value) {
    if (__hisnan(value)) {
      return neg_inf_ordered();
    }
    const uint16_t bits = __half_as_ushort(value);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
  }

  static __device__ __forceinline__ float from_ordered(Ordered ordered) {
    const uint16_t bits = (ordered & 0x8000u)
                              ? static_cast<uint16_t>(ordered ^ 0x8000u)
                              : static_cast<uint16_t>(~ordered);
    return __half2float(__ushort_as_half(bits));
  }

  static __host__ __device__ constexpr Ordered neg_inf_ordered() {
    return 0x03ffu;
  }
};

template <> struct Traits<__nv_bfloat16> {
  using Ordered = uint16_t;

  static __device__ __forceinline__ Ordered to_ordered(__nv_bfloat16 value) {
    if (__hisnan(value)) {
      return neg_inf_ordered();
    }
    const uint16_t bits = __bfloat16_as_ushort(value);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
  }

  static __device__ __forceinline__ float from_ordered(Ordered ordered) {
    const uint16_t bits = (ordered & 0x8000u)
                              ? static_cast<uint16_t>(ordered ^ 0x8000u)
                              : static_cast<uint16_t>(~ordered);
    return __bfloat162float(__ushort_as_bfloat16(bits));
  }

  static __host__ __device__ constexpr Ordered neg_inf_ordered() {
    return 0x007fu;
  }
};

struct alignas(16) RowState {
  uint32_t histogram[kHistogramBuffers][kRadix];
  int arrival_counter;
  uint32_t gt_count[kMaxCtasPerGroup];
  uint32_t eq_count[kMaxCtasPerGroup];
};

static_assert(sizeof(RowState) % sizeof(uint32_t) == 0);

inline constexpr size_t workspace_words_per_row() {
  return sizeof(RowState) / sizeof(uint32_t);
}

__device__ __forceinline__ int load_acquire(int *ptr) {
  int value;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(value)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(value) : "l"(ptr));
#endif
  return value;
}

__device__ __forceinline__ void add_release(int *ptr, int value) {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(value)
               : "memory");
#else
  __threadfence();
  atomicAdd(ptr, value);
#endif
}

__device__ __forceinline__ int atomic_add_release(int *ptr, int value) {
  int old;
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(old)
               : "l"(ptr), "r"(value)
               : "memory");
#else
  __threadfence();
  old = atomicAdd(ptr, value);
#endif
  return old;
}

__device__ __forceinline__ void store_release(int *ptr, int value) {
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile("st.release.gpu.b32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(value)
               : "memory");
#else
  __threadfence();
  atomicExch(ptr, value);
#endif
}

__device__ __forceinline__ void group_barrier(RowState *state, int &phase,
                                               uint32_t ctas_per_group,
                                               uint32_t thread_idx) {
  __syncthreads();
  if (thread_idx == 0) {
    add_release(&state->arrival_counter, 1);
    const int target = (phase + 1) * static_cast<int>(ctas_per_group);
    while (load_acquire(&state->arrival_counter) < target) {
    }
  }
  __syncthreads();
  ++phase;
}

template <typename T, uint32_t Width>
struct alignas(sizeof(T) * Width) AlignedVector {
  T values[Width];
};

template <typename T, uint32_t VecSize>
__device__ __forceinline__ void
load_ordered(const T *input, typename Traits<T>::Ordered *ordered,
             uint32_t length, uint32_t thread_idx) {
  using Vector = AlignedVector<T, VecSize>;
  const uint32_t aligned_length = length / VecSize * VecSize;
  for (uint32_t offset = thread_idx * VecSize; offset < aligned_length;
       offset += kBlockThreads * VecSize) {
    const Vector vector = *reinterpret_cast<const Vector *>(input + offset);
#pragma unroll
    for (uint32_t item = 0; item < VecSize; ++item) {
      ordered[offset + item] = Traits<T>::to_ordered(vector.values[item]);
    }
  }
  for (uint32_t offset = aligned_length + thread_idx; offset < length;
       offset += kBlockThreads) {
    ordered[offset] = Traits<T>::to_ordered(input[offset]);
  }
  __syncthreads();
}

__device__ __forceinline__ uint32_t warp_min(uint32_t value) {
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <typename T, uint32_t VecSize>
__global__ void __launch_bounds__(kBlockThreads) ranked_radix_select(
    const T *__restrict__ input, float *__restrict__ packed_output, int nrows,
    int ncols, int top_k, RowState *__restrict__ row_states,
    uint32_t chunk_size, uint32_t ctas_per_group) {
  using TypeTraits = Traits<T>;
  using Ordered = typename TypeTraits::Ordered;
  constexpr uint32_t kOrderedBits = sizeof(Ordered) * 8;
  constexpr uint32_t kRounds = kOrderedBits / 8;
  constexpr uint32_t kFixedWords = kRadix + 32 + 8;
  constexpr uint32_t kFixedBytes = (kFixedWords * sizeof(uint32_t) + 15) & ~15;

  const uint32_t global_cta = blockIdx.x;
  const uint32_t group = global_cta / ctas_per_group;
  const uint32_t cta_in_group = global_cta % ctas_per_group;
  const uint32_t thread_idx = threadIdx.x;
  const uint32_t lane = thread_idx & 31;
  const uint32_t warp = thread_idx >> 5;
  const uint32_t num_groups = gridDim.x / ctas_per_group;

  extern __shared__ __align__(16) unsigned char shared_bytes[];
  uint32_t *local_histogram = reinterpret_cast<uint32_t *>(shared_bytes);
  uint32_t *warp_scratch = local_histogram + kRadix;
  uint32_t *scalars = warp_scratch + 32;
  Ordered *shared_ordered =
      reinterpret_cast<Ordered *>(shared_bytes + kFixedBytes);

  RowState *state = row_states + group;
  int barrier_phase = 0;
  const uint32_t iterations =
      (static_cast<uint32_t>(nrows) + num_groups - 1) / num_groups;

  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    const uint32_t row = group + iteration * num_groups;
    if (row >= static_cast<uint32_t>(nrows)) {
      break;
    }

    float *row_output = packed_output + static_cast<size_t>(row) * 2 * top_k;
    if (cta_in_group == 0) {
      for (uint32_t slot = thread_idx; slot < static_cast<uint32_t>(top_k);
           slot += kBlockThreads) {
        row_output[slot] = -__int_as_float(0x7f800000);
        row_output[top_k + slot] = 0.0f;
      }
    }

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end =
        min(chunk_start + chunk_size, static_cast<uint32_t>(ncols));
    const uint32_t actual_chunk_size =
        chunk_start < static_cast<uint32_t>(ncols)
            ? chunk_end - chunk_start
            : 0;
    const T *row_input = input + static_cast<size_t>(row) * ncols;
    load_ordered<T, VecSize>(row_input + chunk_start, shared_ordered,
                             actual_chunk_size, thread_idx);

    if (thread_idx == 0) {
      scalars[0] = 0;
      scalars[1] = static_cast<uint32_t>(top_k);
    }
    __syncthreads();

#pragma unroll
    for (uint32_t round = 0; round < kRounds; ++round) {
      const uint32_t global_round = iteration * kRounds + round;
      uint32_t *current_histogram =
          state->histogram[global_round % kHistogramBuffers];
      uint32_t *next_histogram =
          state->histogram[(global_round + 1) % kHistogramBuffers];

      for (uint32_t bin = thread_idx; bin < kRadix;
           bin += kBlockThreads) {
        local_histogram[bin] = 0;
        if (cta_in_group == 0) {
          next_histogram[bin] = 0;
        }
      }
      __syncthreads();

      const uint32_t shift = kOrderedBits - 8 * (round + 1);
      const uint32_t prefix_mask =
          shift + 8 == kOrderedBits ? 0u : (~0u << (shift + 8));
      const uint32_t prefix = scalars[0];
      for (uint32_t index = thread_idx; index < actual_chunk_size;
           index += kBlockThreads) {
        const uint32_t ordered = static_cast<uint32_t>(shared_ordered[index]);
        if ((ordered & prefix_mask) == prefix) {
          atomicAdd(&local_histogram[(ordered >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();

      for (uint32_t bin = thread_idx; bin < kRadix;
           bin += kBlockThreads) {
        const uint32_t count = local_histogram[bin];
        if (count != 0) {
          atomicAdd(&current_histogram[bin], count);
        }
      }
      group_barrier(state, barrier_phase, ctas_per_group, thread_idx);

      for (uint32_t bin = thread_idx; bin < kRadix;
           bin += kBlockThreads) {
        local_histogram[bin] = current_histogram[bin];
      }
      __syncthreads();
      if (thread_idx == 0) {
        uint32_t remaining = scalars[1];
        for (int bin = kRadix - 1; bin >= 0; --bin) {
          const uint32_t count = local_histogram[bin];
          if (remaining > count) {
            remaining -= count;
          } else {
            scalars[0] = prefix | (static_cast<uint32_t>(bin) << shift);
            scalars[1] = remaining;
            break;
          }
        }
      }
      __syncthreads();
    }

    const Ordered pivot = static_cast<Ordered>(scalars[0]);
    uint32_t thread_gt = 0;
    uint32_t thread_eq = 0;
    for (uint32_t index = thread_idx; index < actual_chunk_size;
         index += kBlockThreads) {
      const Ordered ordered = shared_ordered[index];
      const bool valid = ordered > TypeTraits::neg_inf_ordered();
      thread_gt += static_cast<uint32_t>(valid && ordered > pivot);
      thread_eq += static_cast<uint32_t>(valid && ordered == pivot);
    }

    if (thread_idx < 2) {
      scalars[2 + thread_idx] = 0;
    }
    __syncthreads();
    if (thread_gt != 0) {
      atomicAdd(&scalars[2], thread_gt);
    }
    if (thread_eq != 0) {
      atomicAdd(&scalars[3], thread_eq);
    }
    __syncthreads();
    if (thread_idx == 0) {
      state->gt_count[cta_in_group] = scalars[2];
      state->eq_count[cta_in_group] = scalars[3];
    }
    group_barrier(state, barrier_phase, ctas_per_group, thread_idx);

    if (thread_idx == 0) {
      uint32_t gt_prefix = 0;
      uint32_t eq_prefix = 0;
      uint32_t total_gt = 0;
      for (uint32_t cta = 0; cta < ctas_per_group; ++cta) {
        total_gt += state->gt_count[cta];
        if (cta < cta_in_group) {
          gt_prefix += state->gt_count[cta];
          eq_prefix += state->eq_count[cta];
        }
      }
      const uint32_t eq_needed =
          static_cast<uint32_t>(top_k) > total_gt
              ? static_cast<uint32_t>(top_k) - total_gt
              : 0;
      scalars[2] = gt_prefix;
      scalars[3] = total_gt;
      scalars[4] = eq_prefix;
      scalars[5] = eq_prefix < eq_needed
                       ? min(state->eq_count[cta_in_group], eq_needed - eq_prefix)
                       : 0;
      scalars[6] = 0;
    }
    __syncthreads();

    for (uint32_t index = thread_idx; index < actual_chunk_size;
         index += kBlockThreads) {
      const Ordered ordered = shared_ordered[index];
      if (ordered > pivot && ordered > TypeTraits::neg_inf_ordered()) {
        const uint32_t local_slot = atomicAdd(&scalars[6], 1u);
        const uint32_t slot = scalars[2] + local_slot;
        row_output[slot] = TypeTraits::from_ordered(ordered);
        row_output[top_k + slot] = static_cast<float>(chunk_start + index);
      }
    }
    __syncthreads();

    uint32_t previous = 0;
    const uint32_t eq_take = scalars[5];
    for (uint32_t selected = 0; selected < eq_take; ++selected) {
      uint32_t local_min = UINT32_MAX;
      for (uint32_t index = thread_idx; index < actual_chunk_size;
           index += kBlockThreads) {
        const uint32_t global_index = chunk_start + index;
        if (shared_ordered[index] == pivot &&
            shared_ordered[index] > TypeTraits::neg_inf_ordered() &&
            (selected == 0 || global_index > previous)) {
          local_min = min(local_min, global_index);
        }
      }
      local_min = warp_min(local_min);
      if (lane == 0) {
        warp_scratch[warp] = local_min;
      }
      __syncthreads();
      if (warp == 0) {
        uint32_t block_min = lane < 32 ? warp_scratch[lane] : UINT32_MAX;
        block_min = warp_min(block_min);
        if (lane == 0) {
          scalars[7] = block_min;
        }
      }
      __syncthreads();
      previous = scalars[7];
      if (thread_idx == 0) {
        const uint32_t slot = scalars[3] + scalars[4] + selected;
        row_output[slot] = TypeTraits::from_ordered(pivot);
        row_output[top_k + slot] = static_cast<float>(previous);
      }
      __syncthreads();
    }

    group_barrier(state, barrier_phase, ctas_per_group, thread_idx);
  }

  __syncthreads();
  if (thread_idx == 0) {
    const int exit_target =
        (barrier_phase + 1) * static_cast<int>(ctas_per_group);
    scalars[0] = static_cast<uint32_t>(
        atomic_add_release(&state->arrival_counter, 1) + 1 == exit_target);
  }
  __syncthreads();
  if (scalars[0] != 0) {
    for (uint32_t word = thread_idx;
         word < kHistogramBuffers * kRadix; word += kBlockThreads) {
      reinterpret_cast<uint32_t *>(state->histogram)[word] = 0;
    }
    __syncthreads();
    if (thread_idx == 0) {
      store_release(&state->arrival_counter, 0);
    }
  }
}

__device__ __forceinline__ uint32_t float_to_ordered(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_to_float(uint32_t ordered) {
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__global__ void ranked_radix_sort_pack(float *__restrict__ packed_output,
                                       int top_k) {
  const uint32_t lane = threadIdx.x;
  float *row_output = packed_output + static_cast<size_t>(blockIdx.x) * 2 * top_k;

  unsigned long long key = ~0ull;
  if (lane < static_cast<uint32_t>(top_k)) {
    const uint32_t ordered = float_to_ordered(row_output[lane]);
    const uint32_t index = static_cast<uint32_t>(row_output[top_k + lane]);
    key = (static_cast<unsigned long long>(~ordered) << 32) | index;
  }

#pragma unroll
  for (uint32_t size = 2; size <= 32; size <<= 1) {
#pragma unroll
    for (uint32_t stride = size >> 1; stride > 0; stride >>= 1) {
      const unsigned long long other =
          __shfl_xor_sync(0xffffffffu, key, stride);
      const bool ascending = (lane & size) == 0;
      const bool lower_lane = (lane & stride) == 0;
      const bool take_min = ascending == lower_lane;
      key = take_min ? min(key, other) : max(key, other);
    }
  }

  if (lane < static_cast<uint32_t>(top_k)) {
    const uint32_t ordered = ~static_cast<uint32_t>(key >> 32);
    row_output[lane] = ordered_to_float(ordered);
    row_output[top_k + lane] = static_cast<float>(static_cast<uint32_t>(key));
  }
}

template <typename T, uint32_t VecSize>
inline cudaError_t launch_vec(const T *input, float *packed_output, int nrows,
                              int ncols, int top_k, RowState *states,
                              uint32_t chunk_size, uint32_t ctas_per_group,
                              int num_sms, size_t smem_size,
                              cudaStream_t stream, bool *launched) {
  *launched = false;
  auto kernel = ranked_radix_select<T, VecSize>;
  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_size));
  if (status != cudaSuccess) {
    return status;
  }

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, kBlockThreads, smem_size);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_blocks_per_sm <= 0) {
    return cudaSuccess;
  }
  const uint64_t resident_ctas =
      static_cast<uint64_t>(active_blocks_per_sm) * num_sms;
  const uint32_t num_groups = static_cast<uint32_t>(std::min<uint64_t>(
      static_cast<uint64_t>(nrows), resident_ctas / ctas_per_group));
  if (num_groups == 0) {
    return cudaSuccess;
  }
  const uint32_t total_ctas = num_groups * ctas_per_group;

  // The software group barrier is safe only when the complete grid has cooperative residency.
  void *args[] = {&input,          &packed_output, &nrows,
                  &ncols,          &top_k,          &states,
                  &chunk_size,     &ctas_per_group};
  status = cudaLaunchCooperativeKernel(kernel, dim3(total_ctas),
                                       dim3(kBlockThreads), args, smem_size,
                                       stream);
  if (status != cudaSuccess) {
    return status;
  }
  ranked_radix_sort_pack<<<nrows, 32, 0, stream>>>(packed_output, top_k);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    *launched = true;
  }
  return status;
}

template <typename T>
inline cudaError_t launch(const T *input, float *packed_output, void *workspace,
                          int nrows, int ncols, int top_k,
                          cudaStream_t stream, bool *launched) {
  using Ordered = typename Traits<T>::Ordered;
  constexpr size_t kFixedWords = kRadix + 32 + 8;
  constexpr size_t kFixedBytes =
      (kFixedWords * sizeof(uint32_t) + 15) & ~size_t(15);

  *launched = false;
  if (top_k < static_cast<int>(kMinTopK) ||
      top_k > static_cast<int>(kMaxTopK) || ncols <= top_k || nrows <= 0) {
    return cudaSuccess;
  }

  int device;
  int num_sms;
  int max_smem;
  int cooperative_launch;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                                  device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaDeviceGetAttribute(&max_smem,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaDeviceGetAttribute(&cooperative_launch,
                                  cudaDevAttrCooperativeLaunch, device);
  if (status != cudaSuccess) {
    return status;
  }
  if (cooperative_launch == 0 || num_sms <= 0 ||
      max_smem <= static_cast<int>(kFixedBytes + kLaunchSmemHeadroom)) {
    return cudaSuccess;
  }

  const uint32_t max_vec = 16 / sizeof(T);
  const uint32_t vec_size = std::gcd(max_vec, static_cast<uint32_t>(ncols));
  uint32_t max_chunk = static_cast<uint32_t>(
      (max_smem - kFixedBytes - kLaunchSmemHeadroom) / sizeof(Ordered));
  max_chunk = max_chunk / vec_size * vec_size;
  if (max_chunk < kBlockThreads * vec_size) {
    return cudaSuccess;
  }

  const uint32_t ctas_per_group =
      (static_cast<uint32_t>(ncols) + max_chunk - 1) / max_chunk;
  if (ctas_per_group == 0 || ctas_per_group > kMaxCtasPerGroup) {
    return cudaSuccess;
  }
  uint32_t chunk_size =
      (static_cast<uint32_t>(ncols) + ctas_per_group - 1) / ctas_per_group;
  chunk_size = (chunk_size + vec_size - 1) / vec_size * vec_size;
  if (chunk_size > max_chunk) {
    return cudaSuccess;
  }
  const size_t smem_size = kFixedBytes + chunk_size * sizeof(Ordered);
  auto *states = reinterpret_cast<RowState *>(workspace);

  switch (vec_size) {
  case 1:
    return launch_vec<T, 1>(input, packed_output, nrows, ncols, top_k,
                            states, chunk_size, ctas_per_group, num_sms,
                            smem_size, stream, launched);
  case 2:
    return launch_vec<T, 2>(input, packed_output, nrows, ncols, top_k,
                            states, chunk_size, ctas_per_group, num_sms,
                            smem_size, stream, launched);
  case 4:
    return launch_vec<T, 4>(input, packed_output, nrows, ncols, top_k,
                            states, chunk_size, ctas_per_group, num_sms,
                            smem_size, stream, launched);
  case 8:
    if constexpr (sizeof(T) == 2) {
      return launch_vec<T, 8>(input, packed_output, nrows, ncols, top_k,
                              states, chunk_size, ctas_per_group, num_sms,
                              smem_size, stream, launched);
    }
    return cudaSuccess;
  default:
    return cudaSuccess;
  }
}

} // namespace mistralrs_radix_topk
