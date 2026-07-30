#include <cstddef>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace mistralrs_routed_lora {

constexpr uint32_t kInvalidId = UINT32_MAX;
constexpr int kWarpSize = 32;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kGroupedRouteTile = 2;
constexpr int kGroupedRankTile = 4;
constexpr int kGroupedOutputTile = 64;
constexpr int kMaxRank = 512;
constexpr int kWmmaTile = 16;
constexpr int kWmmaRankCap = 128;

static_assert(kWmmaRankCap == kWarps * kWmmaTile);

struct alignas(8) AdapterWeight {
  uint64_t a;
  uint64_t b;
  uint64_t scales;
  uint32_t rank;
  uint32_t rank_stride;
  float scale;
  uint32_t flags;
};

static_assert(sizeof(AdapterWeight) == 40);
static_assert(offsetof(AdapterWeight, rank) == 24);
static_assert(offsetof(AdapterWeight, scale) == 32);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ int rank_subgroup_width(int rank) {
  return rank <= 8 ? 8 : (rank <= 16 ? 16 : 32);
}

__device__ __forceinline__ float subgroup_sum(float value, int width,
                                              uint32_t mask) {
  for (int offset = width / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset, width);
  }
  return value;
}

template <typename T> struct ElementOps;

template <> struct ElementOps<float> {
  using Pair = float2;

  __device__ __forceinline__ static float to_float(float value) {
    return value;
  }

  __device__ __forceinline__ static float from_float(float value) {
    return value;
  }

  __device__ __forceinline__ static float pair_dot(Pair lhs, Pair rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y;
  }
};

template <> struct ElementOps<__half> {
  using Pair = half2;

  __device__ __forceinline__ static float to_float(__half value) {
    return __half2float(value);
  }

  __device__ __forceinline__ static __half from_float(float value) {
    return __float2half_rn(value);
  }

  __device__ __forceinline__ static float pair_dot(Pair lhs, Pair rhs) {
    const float2 lhs_f = __half22float2(lhs);
    const float2 rhs_f = __half22float2(rhs);
    return lhs_f.x * rhs_f.x + lhs_f.y * rhs_f.y;
  }
};

template <> struct ElementOps<__nv_bfloat16> {
  using Pair = __nv_bfloat162;

  __device__ __forceinline__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }

  __device__ __forceinline__ static float pair_dot(Pair lhs, Pair rhs) {
    return __bfloat162float(lhs.x) * __bfloat162float(rhs.x) +
           __bfloat162float(lhs.y) * __bfloat162float(rhs.y);
  }
};

template <typename T> struct WmmaElement;

template <> struct WmmaElement<__half> {
  using Type = __half;
};

template <> struct WmmaElement<__nv_bfloat16> {
  using Type = __nv_bfloat16;
};

template <typename T>
__device__ __forceinline__ float warp_dot(const T *lhs, const T *rhs,
                                          int length) {
  const int lane = threadIdx.x % kWarpSize;
  float sum = 0.0f;
  using Pair = typename ElementOps<T>::Pair;
  const uintptr_t alignment =
      reinterpret_cast<uintptr_t>(lhs) | reinterpret_cast<uintptr_t>(rhs);
  if ((length & 1) == 0 && (alignment & (alignof(Pair) - 1)) == 0) {
    const Pair *lhs_pairs = reinterpret_cast<const Pair *>(lhs);
    const Pair *rhs_pairs = reinterpret_cast<const Pair *>(rhs);
    const int pair_count = length / 2;
    for (int index = lane; index < pair_count; index += kWarpSize) {
      sum += ElementOps<T>::pair_dot(lhs_pairs[index], rhs_pairs[index]);
    }
  } else {
    for (int index = lane; index < length; index += kWarpSize) {
      sum += ElementOps<T>::to_float(lhs[index]) *
             ElementOps<T>::to_float(rhs[index]);
    }
  }
  return warp_sum(sum);
}

struct WeightView {
  const AdapterWeight *descriptor;
  uint32_t expert;
};

__device__ __forceinline__ WeightView weight_view(const AdapterWeight *weights,
                                                  uint32_t pair_id, int slice,
                                                  int num_experts,
                                                  int num_adapter_slots) {
  if (pair_id == kInvalidId || num_experts <= 0) {
    return {nullptr, 0};
  }
  const uint32_t slot = pair_id / static_cast<uint32_t>(num_experts);
  const uint32_t expert = pair_id - slot * static_cast<uint32_t>(num_experts);
  if (slot >= static_cast<uint32_t>(num_adapter_slots) ||
      expert >= static_cast<uint32_t>(num_experts)) {
    return {nullptr, 0};
  }
  const AdapterWeight *descriptor =
      weights + static_cast<size_t>(slice) * num_adapter_slots + slot;
  if (descriptor->a == 0 || descriptor->b == 0 || descriptor->rank == 0 ||
      descriptor->rank > descriptor->rank_stride ||
      descriptor->rank > kMaxRank) {
    return {nullptr, 0};
  }
  return {descriptor, expert};
}

__device__ __forceinline__ uint32_t pair_for_route(
    const uint32_t *token_adapter_slots, const uint32_t *topk_expert_ids,
    uint32_t route, int top_k, int num_experts, int num_adapter_slots) {
  const uint32_t slot =
      token_adapter_slots == nullptr ? 0 : token_adapter_slots[route / top_k];
  const uint32_t expert = topk_expert_ids[route];
  if (slot == kInvalidId || slot >= static_cast<uint32_t>(num_adapter_slots) ||
      expert >= static_cast<uint32_t>(num_experts)) {
    return kInvalidId;
  }
  return slot * static_cast<uint32_t>(num_experts) + expert;
}

__global__ void count_routes(const uint32_t *__restrict__ token_adapter_slots,
                             const uint32_t *__restrict__ topk_expert_ids,
                             uint32_t *__restrict__ route_pair_ids,
                             uint32_t *__restrict__ pair_counts,
                             uint32_t *__restrict__ num_active_routes,
                             int num_routes, int top_k, int num_experts,
                             int num_adapter_slots) {
  uint32_t local_active = 0;
  for (int route = blockIdx.x * blockDim.x + threadIdx.x; route < num_routes;
       route += blockDim.x * gridDim.x) {
    const uint32_t pair =
        pair_for_route(token_adapter_slots, topk_expert_ids, route, top_k,
                       num_experts, num_adapter_slots);
    route_pair_ids[route] = pair;
    if (pair != kInvalidId) {
      atomicAdd(pair_counts + pair, 1U);
      ++local_active;
    }
  }

  __shared__ uint32_t active_per_thread[kThreads];
  active_per_thread[threadIdx.x] = local_active;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      active_per_thread[threadIdx.x] += active_per_thread[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && active_per_thread[0] != 0) {
    atomicAdd(num_active_routes, active_per_thread[0]);
  }
}

__global__ void count_pair_blocks(const uint32_t *__restrict__ pair_counts,
                                  uint32_t *__restrict__ pair_blocks,
                                  int num_pairs, int block_size) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < num_pairs;
       pair += blockDim.x * gridDim.x) {
    const uint32_t count = pair_counts[pair];
    pair_blocks[pair] = (count + static_cast<uint32_t>(block_size) - 1) /
                        static_cast<uint32_t>(block_size);
  }
}

__global__ void finalize_pairs(const uint32_t *__restrict__ pair_counts,
                               const uint32_t *__restrict__ pair_blocks,
                               uint32_t *__restrict__ pair_offsets,
                               uint32_t *__restrict__ block_pair_ids,
                               uint32_t *__restrict__ num_padded_routes,
                               int num_pairs, int block_size) {
  for (int pair = blockIdx.x * blockDim.x + threadIdx.x; pair < num_pairs;
       pair += blockDim.x * gridDim.x) {
    const uint32_t block_offset = pair_offsets[pair];
    const uint32_t blocks = pair_blocks[pair];
    for (uint32_t block = 0; block < blocks; ++block) {
      block_pair_ids[block_offset + block] = static_cast<uint32_t>(pair);
    }
    pair_offsets[pair] = block_offset * static_cast<uint32_t>(block_size);
    if (pair == num_pairs - 1) {
      const uint32_t total_blocks = block_offset + blocks;
      pair_offsets[num_pairs] =
          total_blocks * static_cast<uint32_t>(block_size);
      *num_padded_routes = total_blocks * static_cast<uint32_t>(block_size);
    }
  }
}

__global__ void scatter_routes(const uint32_t *__restrict__ route_pair_ids,
                               const uint32_t *__restrict__ pair_offsets,
                               uint32_t *__restrict__ pair_cursors,
                               uint32_t *__restrict__ sorted_route_ids,
                               int num_routes) {
  for (int route = blockIdx.x * blockDim.x + threadIdx.x; route < num_routes;
       route += blockDim.x * gridDim.x) {
    const uint32_t pair = route_pair_ids[route];
    if (pair == kInvalidId) {
      continue;
    }
    const uint32_t index =
        pair_offsets[pair] + atomicAdd(pair_cursors + pair, 1U);
    sorted_route_ids[index] = static_cast<uint32_t>(route);
  }
}

__device__ __forceinline__ uint32_t
input_row_for_route(uint32_t route, const uint32_t *route_input_rows,
                    int input_mode, int top_k) {
  if (route_input_rows != nullptr) {
    return route_input_rows[route];
  }
  return input_mode == 1 ? route / static_cast<uint32_t>(top_k) : route;
}

__device__ __forceinline__ uint32_t
output_row_for_route(uint32_t route, const uint32_t *route_output_rows) {
  return route_output_rows == nullptr ? route : route_output_rows[route];
}

template <typename T>
__global__ void
fused_direct(const T *__restrict__ input, T *__restrict__ output,
             const AdapterWeight *__restrict__ weights,
             const uint32_t *__restrict__ token_adapter_slots,
             const uint32_t *__restrict__ topk_expert_ids,
             const uint32_t *__restrict__ route_input_rows,
             const uint32_t *__restrict__ route_output_rows,
             const float *__restrict__ route_output_scales, int num_routes,
             int top_k, int num_experts, int num_adapter_slots, int num_slices,
             int input_features, int output_features, int output_row_stride,
             int output_slice_stride, int input_mode, int output_splits) {
  const uint32_t route = static_cast<uint32_t>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.y);
  const int output_split = static_cast<int>(blockIdx.z);
  if (route >= static_cast<uint32_t>(num_routes) || slice >= num_slices ||
      output_split >= output_splits) {
    return;
  }
  const uint32_t pair =
      pair_for_route(token_adapter_slots, topk_expert_ids, route, top_k,
                     num_experts, num_adapter_slots);
  const WeightView view =
      weight_view(weights, pair, slice, num_experts, num_adapter_slots);
  if (view.descriptor == nullptr) {
    return;
  }

  const AdapterWeight descriptor = *view.descriptor;
  const T *a = reinterpret_cast<const T *>(descriptor.a) +
               static_cast<size_t>(view.expert) * descriptor.rank_stride *
                   input_features;
  const T *b = reinterpret_cast<const T *>(descriptor.b) +
               static_cast<size_t>(view.expert) * output_features *
                   descriptor.rank_stride;
  const float scale =
      descriptor.scales == 0
          ? descriptor.scale
          : reinterpret_cast<const float *>(descriptor.scales)[view.expert];
  if (scale == 0.0f) {
    return;
  }
  const uint32_t input_row_index =
      input_row_for_route(route, route_input_rows, input_mode, top_k);
  const T *input_row =
      input + static_cast<size_t>(input_row_index) * input_features;

  extern __shared__ float hidden[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  for (int rank_row = warp; rank_row < static_cast<int>(descriptor.rank);
       rank_row += kWarps) {
    const float value =
        warp_dot(input_row, a + static_cast<size_t>(rank_row) * input_features,
                 input_features);
    if (lane == 0) {
      hidden[rank_row] = value;
    }
  }
  __syncthreads();

  float routed_scale = scale;
  if (route_output_scales != nullptr) {
    routed_scale *= route_output_scales[route];
  }
  const uint32_t output_row = output_row_for_route(route, route_output_rows);
  T *output_row_ptr = output +
                      static_cast<size_t>(output_row) * output_row_stride +
                      static_cast<size_t>(slice) * output_slice_stride;
  const int columns_per_split =
      (output_features + output_splits - 1) / output_splits;
  const int output_begin = output_split * columns_per_split;
  const int output_end = min(output_features, output_begin + columns_per_split);
  const int subgroup_width = rank_subgroup_width(descriptor.rank);
  const int subgroup_lane = threadIdx.x % subgroup_width;
  const int subgroup = threadIdx.x / subgroup_width;
  const int subgroups_per_block = blockDim.x / subgroup_width;
  const int subgroup_in_warp = (threadIdx.x % kWarpSize) / subgroup_width;
  const uint32_t subgroup_mask =
      subgroup_width == kWarpSize ? 0xffffffffU
                                  : ((1U << subgroup_width) - 1U)
                                        << (subgroup_in_warp * subgroup_width);
  for (int output_col = output_begin + subgroup; output_col < output_end;
       output_col += subgroups_per_block) {
    const T *b_row =
        b + static_cast<size_t>(output_col) * descriptor.rank_stride;
    float sum = 0.0f;
    for (uint32_t rank_col = subgroup_lane; rank_col < descriptor.rank;
         rank_col += subgroup_width) {
      sum += ElementOps<T>::to_float(b_row[rank_col]) * hidden[rank_col];
    }
    sum = subgroup_sum(sum, subgroup_width, subgroup_mask);
    if (subgroup_lane == 0) {
      const float base = ElementOps<T>::to_float(output_row_ptr[output_col]);
      output_row_ptr[output_col] =
          ElementOps<T>::from_float(base + routed_scale * sum);
    }
  }
}

template <typename T>
__global__ void grouped_shrink(const T *__restrict__ input,
                               const AdapterWeight *__restrict__ weights,
                               const uint32_t *__restrict__ sorted_route_ids,
                               const uint32_t *__restrict__ block_pair_ids,
                               const uint32_t *__restrict__ route_input_rows,
                               float *__restrict__ hidden, int num_routes,
                               int max_blocks, int block_size, int top_k,
                               int num_experts, int num_adapter_slots,
                               int num_slices, int input_features, int max_rank,
                               int input_mode, int rank_tiles) {
  const int block = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.z);
  if (block >= max_blocks || slice >= num_slices) {
    return;
  }
  const uint32_t pair = block_pair_ids[block];
  const WeightView view =
      weight_view(weights, pair, slice, num_experts, num_adapter_slots);
  if (view.descriptor == nullptr) {
    return;
  }
  const AdapterWeight descriptor = *view.descriptor;
  if (descriptor.rank > static_cast<uint32_t>(max_rank)) {
    return;
  }
  const float adapter_scale =
      descriptor.scales == 0
          ? descriptor.scale
          : reinterpret_cast<const float *>(descriptor.scales)[view.expert];
  if (adapter_scale == 0.0f) {
    return;
  }

  const int tile = static_cast<int>(blockIdx.y);
  const int route_tile = tile / rank_tiles;
  const int rank_tile = tile - route_tile * rank_tiles;
  const int warp = threadIdx.x / kWarpSize;
  const int route_in_tile = warp / kGroupedRankTile;
  const int rank_in_tile = warp - route_in_tile * kGroupedRankTile;
  const int sorted_index =
      block * block_size + route_tile * kGroupedRouteTile + route_in_tile;
  const uint32_t route = sorted_route_ids[sorted_index];
  const int rank_row = rank_tile * kGroupedRankTile + rank_in_tile;
  if (route == kInvalidId || route >= static_cast<uint32_t>(num_routes) ||
      rank_row >= static_cast<int>(descriptor.rank)) {
    return;
  }

  const uint32_t input_row_index =
      input_row_for_route(route, route_input_rows, input_mode, top_k);
  const T *input_row =
      input + static_cast<size_t>(input_row_index) * input_features;
  const T *a = reinterpret_cast<const T *>(descriptor.a) +
               static_cast<size_t>(view.expert) * descriptor.rank_stride *
                   input_features;
  const float value =
      warp_dot(input_row, a + static_cast<size_t>(rank_row) * input_features,
               input_features);
  if (threadIdx.x % kWarpSize == 0) {
    const size_t hidden_index =
        (static_cast<size_t>(slice) * num_routes + route) * max_rank + rank_row;
    hidden[hidden_index] = value;
  }
}

template <typename T>
__global__ void
grouped_expand_add(const float *__restrict__ hidden, T *__restrict__ output,
                   const AdapterWeight *__restrict__ weights,
                   const uint32_t *__restrict__ sorted_route_ids,
                   const uint32_t *__restrict__ block_pair_ids,
                   const uint32_t *__restrict__ route_output_rows,
                   const float *__restrict__ route_output_scales,
                   int num_routes, int max_blocks, int block_size,
                   int num_experts, int num_adapter_slots, int num_slices,
                   int output_features, int output_row_stride,
                   int output_slice_stride, int max_rank) {
  const int block = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.z);
  if (block >= max_blocks || slice >= num_slices) {
    return;
  }
  const uint32_t pair = block_pair_ids[block];
  const WeightView view =
      weight_view(weights, pair, slice, num_experts, num_adapter_slots);
  if (view.descriptor == nullptr) {
    return;
  }
  const AdapterWeight descriptor = *view.descriptor;
  if (descriptor.rank > static_cast<uint32_t>(max_rank)) {
    return;
  }

  const int output_base = static_cast<int>(blockIdx.y) * kGroupedOutputTile;
  const int cells = block_size * kGroupedOutputTile;
  const T *b = reinterpret_cast<const T *>(descriptor.b) +
               static_cast<size_t>(view.expert) * output_features *
                   descriptor.rank_stride;
  float routed_scale =
      descriptor.scales == 0
          ? descriptor.scale
          : reinterpret_cast<const float *>(descriptor.scales)[view.expert];
  if (routed_scale == 0.0f) {
    return;
  }

  const int subgroup_width = rank_subgroup_width(descriptor.rank);
  const int subgroup_lane = threadIdx.x % subgroup_width;
  const int subgroup = threadIdx.x / subgroup_width;
  const int subgroups_per_block = blockDim.x / subgroup_width;
  const int subgroup_in_warp = (threadIdx.x % kWarpSize) / subgroup_width;
  const uint32_t subgroup_mask =
      subgroup_width == kWarpSize ? 0xffffffffU
                                  : ((1U << subgroup_width) - 1U)
                                        << (subgroup_in_warp * subgroup_width);
  for (int cell = subgroup; cell < cells; cell += subgroups_per_block) {
    const int route_in_block = cell / kGroupedOutputTile;
    const int output_col =
        output_base + cell - route_in_block * kGroupedOutputTile;
    if (output_col >= output_features) {
      continue;
    }
    const uint32_t route =
        sorted_route_ids[block * block_size + route_in_block];
    if (route == kInvalidId || route >= static_cast<uint32_t>(num_routes)) {
      continue;
    }
    const float *hidden_row =
        hidden + (static_cast<size_t>(slice) * num_routes + route) * max_rank;
    const T *b_row =
        b + static_cast<size_t>(output_col) * descriptor.rank_stride;
    float scale = routed_scale;
    if (route_output_scales != nullptr) {
      scale *= route_output_scales[route];
    }
    float sum = 0.0f;
    for (uint32_t rank_col = subgroup_lane; rank_col < descriptor.rank;
         rank_col += subgroup_width) {
      sum += hidden_row[rank_col] * ElementOps<T>::to_float(b_row[rank_col]);
    }
    sum = subgroup_sum(sum, subgroup_width, subgroup_mask);
    if (subgroup_lane != 0) {
      continue;
    }
    const uint32_t output_row = output_row_for_route(route, route_output_rows);
    T *output_ptr =
        output + static_cast<size_t>(output_row) * output_row_stride +
        static_cast<size_t>(slice) * output_slice_stride + output_col;
    const float base = ElementOps<T>::to_float(*output_ptr);
    *output_ptr = ElementOps<T>::from_float(base + scale * sum);
  }
}

template <typename T>
__global__ void
grouped_fused_wmma(const T *__restrict__ input, T *__restrict__ output,
                   const AdapterWeight *__restrict__ weights,
                   const uint32_t *__restrict__ sorted_route_ids,
                   const uint32_t *__restrict__ block_pair_ids,
                   const uint32_t *__restrict__ route_input_rows,
                   const uint32_t *__restrict__ route_output_rows,
                   const float *__restrict__ route_output_scales,
                   int num_routes, int max_blocks, int top_k, int num_experts,
                   int num_adapter_slots, int num_slices, int input_features,
                   int output_features, int output_row_stride,
                   int output_slice_stride, int input_mode, int output_splits) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  using WmmaT = typename WmmaElement<T>::Type;

  const int block = static_cast<int>(blockIdx.x);
  const int slice = static_cast<int>(blockIdx.y);
  const int output_split = static_cast<int>(blockIdx.z);
  if (block >= max_blocks || slice >= num_slices ||
      output_split >= output_splits) {
    return;
  }
  const uint32_t pair = block_pair_ids[block];
  const WeightView view =
      weight_view(weights, pair, slice, num_experts, num_adapter_slots);
  if (view.descriptor == nullptr || view.descriptor->rank > kWmmaRankCap) {
    return;
  }
  const AdapterWeight descriptor = *view.descriptor;
  const float adapter_scale =
      descriptor.scales == 0
          ? descriptor.scale
          : reinterpret_cast<const float *>(descriptor.scales)[view.expert];
  if (adapter_scale == 0.0f) {
    return;
  }
  const int rank_padded = (static_cast<int>(descriptor.rank) + kWmmaTile - 1) /
                          kWmmaTile * kWmmaTile;
  const int rank_tiles = rank_padded / kWmmaTile;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const T *a = reinterpret_cast<const T *>(descriptor.a) +
               static_cast<size_t>(view.expert) * descriptor.rank_stride *
                   input_features;
  const T *b = reinterpret_cast<const T *>(descriptor.b) +
               static_cast<size_t>(view.expert) * output_features *
                   descriptor.rank_stride;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  T *x_tile = reinterpret_cast<T *>(shared_raw);
  T *weight_tiles = x_tile + kWmmaTile * kWmmaTile;
  float *hidden_float =
      reinterpret_cast<float *>(weight_tiles + kWarps * kWmmaTile * kWmmaTile);

  wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, WmmaT,
                 wmma::row_major>
      x_fragment;
  wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, WmmaT,
                 wmma::col_major>
      weight_fragment;
  wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
      hidden_fragment;
  wmma::fill_fragment(hidden_fragment, 0.0f);

  const int input_tiles = (input_features + kWmmaTile - 1) / kWmmaTile;
  for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
    const int input_base = input_tile * kWmmaTile;
    for (int index = threadIdx.x; index < kWmmaTile * kWmmaTile;
         index += blockDim.x) {
      const int route_in_block = index / kWmmaTile;
      const int input_col = input_base + index % kWmmaTile;
      const uint32_t route =
          sorted_route_ids[block * kWmmaTile + route_in_block];
      T value = ElementOps<T>::from_float(0.0f);
      if (route != kInvalidId && route < static_cast<uint32_t>(num_routes) &&
          input_col < input_features) {
        const uint32_t input_row =
            input_row_for_route(route, route_input_rows, input_mode, top_k);
        value =
            input[static_cast<size_t>(input_row) * input_features + input_col];
      }
      x_tile[index] = value;
    }
    T *warp_weight_tile = weight_tiles + warp * kWmmaTile * kWmmaTile;
    for (int index = lane; index < kWmmaTile * kWmmaTile; index += kWarpSize) {
      const int rank_col = warp * kWmmaTile + index / kWmmaTile;
      const int input_col = input_base + index % kWmmaTile;
      T value = ElementOps<T>::from_float(0.0f);
      if (warp < rank_tiles && rank_col < static_cast<int>(descriptor.rank) &&
          input_col < input_features) {
        value = a[static_cast<size_t>(rank_col) * input_features + input_col];
      }
      warp_weight_tile[index] = value;
    }
    __syncthreads();
    if (warp < rank_tiles) {
      wmma::load_matrix_sync(
          x_fragment, reinterpret_cast<const WmmaT *>(x_tile), kWmmaTile);
      wmma::load_matrix_sync(weight_fragment,
                             reinterpret_cast<const WmmaT *>(warp_weight_tile),
                             kWmmaTile);
      wmma::mma_sync(hidden_fragment, x_fragment, weight_fragment,
                     hidden_fragment);
    }
    __syncthreads();
  }

  if (warp < rank_tiles) {
    wmma::store_matrix_sync(hidden_float + warp * kWmmaTile, hidden_fragment,
                            rank_padded, wmma::mem_row_major);
  }
  __syncthreads();

  T *hidden = reinterpret_cast<T *>(shared_raw);
  for (int index = threadIdx.x; index < kWmmaTile * rank_padded;
       index += blockDim.x) {
    hidden[index] = ElementOps<T>::from_float(hidden_float[index]);
  }
  __syncthreads();

  T *expand_weight_tiles = hidden + kWmmaTile * kWmmaRankCap;
  float *output_tiles = reinterpret_cast<float *>(
      expand_weight_tiles + kWarps * kWmmaTile * kWmmaTile);
  T *warp_weight_tile = expand_weight_tiles + warp * kWmmaTile * kWmmaTile;
  float *warp_output_tile = output_tiles + warp * kWmmaTile * kWmmaTile;
  for (int output_base = (output_split * kWarps + warp) * kWmmaTile;
       output_base < output_features;
       output_base += output_splits * kWarps * kWmmaTile) {
    wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
        output_fragment;
    wmma::fill_fragment(output_fragment, 0.0f);
    for (int rank_tile = 0; rank_tile < rank_tiles; ++rank_tile) {
      const int rank_base = rank_tile * kWmmaTile;
      for (int index = lane; index < kWmmaTile * kWmmaTile;
           index += kWarpSize) {
        const int output_col = output_base + index / kWmmaTile;
        const int rank_col = rank_base + index % kWmmaTile;
        T value = ElementOps<T>::from_float(0.0f);
        if (output_col < output_features &&
            rank_col < static_cast<int>(descriptor.rank)) {
          value = b[static_cast<size_t>(output_col) * descriptor.rank_stride +
                    rank_col];
        }
        warp_weight_tile[index] = value;
      }
      __syncwarp();
      wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, WmmaT,
                     wmma::row_major>
          hidden_fragment_t;
      wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, WmmaT,
                     wmma::col_major>
          expand_weight_fragment;
      wmma::load_matrix_sync(
          hidden_fragment_t,
          reinterpret_cast<const WmmaT *>(hidden + rank_base), rank_padded);
      wmma::load_matrix_sync(expand_weight_fragment,
                             reinterpret_cast<const WmmaT *>(warp_weight_tile),
                             kWmmaTile);
      wmma::mma_sync(output_fragment, hidden_fragment_t, expand_weight_fragment,
                     output_fragment);
    }
    wmma::store_matrix_sync(warp_output_tile, output_fragment, kWmmaTile,
                            wmma::mem_row_major);
    __syncwarp();

    for (int index = lane; index < kWmmaTile * kWmmaTile; index += kWarpSize) {
      const int route_in_block = index / kWmmaTile;
      const int output_col = output_base + index % kWmmaTile;
      const uint32_t route =
          sorted_route_ids[block * kWmmaTile + route_in_block];
      if (route == kInvalidId || route >= static_cast<uint32_t>(num_routes) ||
          output_col >= output_features) {
        continue;
      }
      float scale = adapter_scale;
      if (route_output_scales != nullptr) {
        scale *= route_output_scales[route];
      }
      const uint32_t output_row =
          output_row_for_route(route, route_output_rows);
      T *output_ptr =
          output + static_cast<size_t>(output_row) * output_row_stride +
          static_cast<size_t>(slice) * output_slice_stride + output_col;
      const float base = ElementOps<T>::to_float(*output_ptr);
      *output_ptr =
          ElementOps<T>::from_float(base + scale * warp_output_tile[index]);
    }
    __syncwarp();
  }
#endif
}

inline int status_code(cudaError_t status) { return static_cast<int>(status); }

int build_metadata(const uint32_t *token_adapter_slots,
                   const uint32_t *topk_expert_ids, uint32_t *route_pair_ids,
                   uint32_t *pair_counts, uint32_t *pair_offsets,
                   uint32_t *pair_cursors, uint32_t *sorted_route_ids,
                   uint32_t *block_pair_ids, uint32_t *num_active_routes,
                   uint32_t *num_padded_routes, int num_tokens, int top_k,
                   int num_experts, int num_adapter_slots, int block_size,
                   int max_padded_routes, int max_blocks, void *scan_workspace,
                   size_t scan_workspace_bytes, cudaStream_t stream) {
  if (token_adapter_slots == nullptr || topk_expert_ids == nullptr ||
      route_pair_ids == nullptr || pair_counts == nullptr ||
      pair_offsets == nullptr || pair_cursors == nullptr ||
      sorted_route_ids == nullptr || block_pair_ids == nullptr ||
      num_active_routes == nullptr || num_padded_routes == nullptr ||
      scan_workspace == nullptr || scan_workspace_bytes == 0 ||
      num_tokens <= 0 || top_k <= 0 || num_experts <= 0 ||
      num_adapter_slots <= 0 || block_size <= 0 || max_padded_routes <= 0 ||
      max_blocks <= 0) {
    return status_code(cudaErrorInvalidValue);
  }
  const int64_t num_routes_i64 = static_cast<int64_t>(num_tokens) * top_k;
  const int64_t num_pairs_i64 =
      static_cast<int64_t>(num_experts) * num_adapter_slots;
  if (num_routes_i64 > INT32_MAX || num_pairs_i64 > INT32_MAX ||
      static_cast<int64_t>(max_blocks) * block_size < max_padded_routes) {
    return status_code(cudaErrorInvalidValue);
  }
  const int num_routes = static_cast<int>(num_routes_i64);
  const int num_pairs = static_cast<int>(num_pairs_i64);

  cudaError_t status = cudaMemsetAsync(
      pair_counts, 0, static_cast<size_t>(num_pairs) * sizeof(uint32_t),
      stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(pair_cursors, 0,
                           static_cast<size_t>(num_pairs) * sizeof(uint32_t),
                           stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(
      sorted_route_ids, 0xff,
      static_cast<size_t>(max_padded_routes) * sizeof(uint32_t), stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(block_pair_ids, 0xff,
                           static_cast<size_t>(max_blocks) * sizeof(uint32_t),
                           stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(num_active_routes, 0, sizeof(uint32_t), stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(num_padded_routes, 0, sizeof(uint32_t), stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }

  const int blocks = (num_routes + kThreads - 1) / kThreads;
  count_routes<<<blocks, kThreads, 0, stream>>>(
      token_adapter_slots, topk_expert_ids, route_pair_ids, pair_counts,
      num_active_routes, num_routes, top_k, num_experts, num_adapter_slots);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status_code(status);
  }
  const int pair_grid = (num_pairs + kThreads - 1) / kThreads;
  count_pair_blocks<<<pair_grid, kThreads, 0, stream>>>(
      pair_counts, pair_cursors, num_pairs, block_size);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cub::DeviceScan::ExclusiveSum(scan_workspace, scan_workspace_bytes,
                                         pair_cursors, pair_offsets, num_pairs,
                                         stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  finalize_pairs<<<pair_grid, kThreads, 0, stream>>>(
      pair_counts, pair_cursors, pair_offsets, block_pair_ids,
      num_padded_routes, num_pairs, block_size);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status_code(status);
  }
  status = cudaMemsetAsync(pair_cursors, 0,
                           static_cast<size_t>(num_pairs) * sizeof(uint32_t),
                           stream);
  if (status != cudaSuccess) {
    return status_code(status);
  }
  scatter_routes<<<blocks, kThreads, 0, stream>>>(
      route_pair_ids, pair_offsets, pair_cursors, sorted_route_ids, num_routes);
  return status_code(cudaGetLastError());
}

size_t metadata_workspace_size(int num_pairs) {
  if (num_pairs <= 0) {
    return 0;
  }
  size_t bytes = 0;
  const cudaError_t status = cub::DeviceScan::ExclusiveSum(
      nullptr, bytes, static_cast<const uint32_t *>(nullptr),
      static_cast<uint32_t *>(nullptr), num_pairs);
  return status == cudaSuccess ? bytes : 0;
}

template <typename T>
int launch_direct(const T *input, T *output, const AdapterWeight *weights,
                  const uint32_t *token_adapter_slots,
                  const uint32_t *topk_expert_ids,
                  const uint32_t *route_input_rows,
                  const uint32_t *route_output_rows,
                  const float *route_output_scales, int num_tokens, int top_k,
                  int num_experts, int num_adapter_slots, int num_slices,
                  int input_features, int output_features,
                  int output_row_stride, int output_slice_stride, int max_rank,
                  int input_mode, int output_splits, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || weights == nullptr ||
      topk_expert_ids == nullptr ||
      (token_adapter_slots == nullptr && num_adapter_slots != 1) ||
      num_tokens <= 0 || top_k <= 0 || num_experts <= 0 ||
      num_adapter_slots <= 0 || num_slices <= 0 || input_features <= 0 ||
      output_features <= 0 || output_row_stride < output_features ||
      max_rank <= 0 || max_rank > kMaxRank || input_mode < 0 ||
      input_mode > 1 || output_splits <= 0 || output_splits > 65535) {
    return status_code(cudaErrorInvalidValue);
  }
  const int64_t num_routes_i64 = static_cast<int64_t>(num_tokens) * top_k;
  if (num_routes_i64 > INT32_MAX || num_slices > 65535) {
    return status_code(cudaErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(num_routes_i64), num_slices,
                  output_splits);
  fused_direct<T><<<grid, kThreads, max_rank * sizeof(float), stream>>>(
      input, output, weights, token_adapter_slots, topk_expert_ids,
      route_input_rows, route_output_rows, route_output_scales,
      static_cast<int>(num_routes_i64), top_k, num_experts, num_adapter_slots,
      num_slices, input_features, output_features, output_row_stride,
      output_slice_stride, input_mode, output_splits);
  return status_code(cudaGetLastError());
}

template <typename T>
int launch_grouped_shrink(const T *input, const AdapterWeight *weights,
                          const uint32_t *sorted_route_ids,
                          const uint32_t *block_pair_ids,
                          const uint32_t *route_input_rows, float *hidden,
                          int num_routes, int max_blocks, int block_size,
                          int top_k, int num_experts, int num_adapter_slots,
                          int num_slices, int input_features, int max_rank,
                          int input_mode, cudaStream_t stream) {
  if (input == nullptr || weights == nullptr || sorted_route_ids == nullptr ||
      block_pair_ids == nullptr || hidden == nullptr || num_routes <= 0 ||
      max_blocks <= 0 || block_size <= 0 || top_k <= 0 || num_experts <= 0 ||
      num_adapter_slots <= 0 || num_slices <= 0 || input_features <= 0 ||
      max_rank <= 0 || max_rank > kMaxRank || input_mode < 0 ||
      input_mode > 1 || block_size % kGroupedRouteTile != 0) {
    return status_code(cudaErrorInvalidValue);
  }
  const int rank_tiles = (max_rank + kGroupedRankTile - 1) / kGroupedRankTile;
  const int route_tiles = block_size / kGroupedRouteTile;
  const int64_t grid_y = static_cast<int64_t>(route_tiles) * rank_tiles;
  if (grid_y > 65535 || num_slices > 65535) {
    return status_code(cudaErrorInvalidValue);
  }
  const dim3 grid(max_blocks, static_cast<unsigned int>(grid_y), num_slices);
  grouped_shrink<T><<<grid, kThreads, 0, stream>>>(
      input, weights, sorted_route_ids, block_pair_ids, route_input_rows,
      hidden, num_routes, max_blocks, block_size, top_k, num_experts,
      num_adapter_slots, num_slices, input_features, max_rank, input_mode,
      rank_tiles);
  return status_code(cudaGetLastError());
}

template <typename T>
int launch_grouped_expand(
    const float *hidden, T *output, const AdapterWeight *weights,
    const uint32_t *sorted_route_ids, const uint32_t *block_pair_ids,
    const uint32_t *route_output_rows, const float *route_output_scales,
    int num_routes, int max_blocks, int block_size, int num_experts,
    int num_adapter_slots, int num_slices, int output_features,
    int output_row_stride, int output_slice_stride, int max_rank,
    cudaStream_t stream) {
  if (hidden == nullptr || output == nullptr || weights == nullptr ||
      sorted_route_ids == nullptr || block_pair_ids == nullptr ||
      num_routes <= 0 || max_blocks <= 0 || block_size <= 0 ||
      num_experts <= 0 || num_adapter_slots <= 0 || num_slices <= 0 ||
      output_features <= 0 || output_row_stride < output_features ||
      max_rank <= 0 || max_rank > kMaxRank) {
    return status_code(cudaErrorInvalidValue);
  }
  const int output_tiles =
      (output_features + kGroupedOutputTile - 1) / kGroupedOutputTile;
  if (output_tiles > 65535 || num_slices > 65535) {
    return status_code(cudaErrorInvalidValue);
  }
  const dim3 grid(max_blocks, output_tiles, num_slices);
  grouped_expand_add<T><<<grid, kThreads, 0, stream>>>(
      hidden, output, weights, sorted_route_ids, block_pair_ids,
      route_output_rows, route_output_scales, num_routes, max_blocks,
      block_size, num_experts, num_adapter_slots, num_slices, output_features,
      output_row_stride, output_slice_stride, max_rank);
  return status_code(cudaGetLastError());
}

template <typename T>
int launch_grouped_wmma(
    const T *input, T *output, const AdapterWeight *weights,
    const uint32_t *sorted_route_ids, const uint32_t *block_pair_ids,
    const uint32_t *route_input_rows, const uint32_t *route_output_rows,
    const float *route_output_scales, int num_routes, int max_blocks, int top_k,
    int num_experts, int num_adapter_slots, int num_slices, int input_features,
    int output_features, int output_row_stride, int output_slice_stride,
    int input_mode, int output_splits, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || weights == nullptr ||
      sorted_route_ids == nullptr || block_pair_ids == nullptr ||
      num_routes <= 0 || max_blocks <= 0 || top_k <= 0 || num_experts <= 0 ||
      num_adapter_slots <= 0 || num_slices <= 0 || input_features <= 0 ||
      output_features <= 0 || output_row_stride < output_features ||
      input_mode < 0 || input_mode > 1 || num_slices > 65535 ||
      output_splits <= 0 || output_splits > 65535) {
    return status_code(cudaErrorInvalidValue);
  }
  const size_t shrink_bytes =
      (kWmmaTile * kWmmaTile + kWarps * kWmmaTile * kWmmaTile) * sizeof(T) +
      kWmmaTile * kWmmaRankCap * sizeof(float);
  const size_t expand_bytes =
      (kWmmaTile * kWmmaRankCap + kWarps * kWmmaTile * kWmmaTile) * sizeof(T) +
      kWarps * kWmmaTile * kWmmaTile * sizeof(float);
  const size_t shared_bytes =
      shrink_bytes > expand_bytes ? shrink_bytes : expand_bytes;
  const dim3 grid(max_blocks, num_slices, output_splits);
  grouped_fused_wmma<T><<<grid, kThreads, shared_bytes, stream>>>(
      input, output, weights, sorted_route_ids, block_pair_ids,
      route_input_rows, route_output_rows, route_output_scales, num_routes,
      max_blocks, top_k, num_experts, num_adapter_slots, num_slices,
      input_features, output_features, output_row_stride, output_slice_stride,
      input_mode, output_splits);
  return status_code(cudaGetLastError());
}

} // namespace mistralrs_routed_lora

extern "C" int launch_routed_lora_build_metadata(
    const uint32_t *token_adapter_slots, const uint32_t *topk_expert_ids,
    uint32_t *route_pair_ids, uint32_t *pair_counts, uint32_t *pair_offsets,
    uint32_t *pair_cursors, uint32_t *sorted_route_ids,
    uint32_t *block_pair_ids, uint32_t *num_active_routes,
    uint32_t *num_padded_routes, int num_tokens, int top_k, int num_experts,
    int num_adapter_slots, int block_size, int max_padded_routes,
    int max_blocks, void *scan_workspace, size_t scan_workspace_bytes,
    cudaStream_t stream) {
  return mistralrs_routed_lora::build_metadata(
      token_adapter_slots, topk_expert_ids, route_pair_ids, pair_counts,
      pair_offsets, pair_cursors, sorted_route_ids, block_pair_ids,
      num_active_routes, num_padded_routes, num_tokens, top_k, num_experts,
      num_adapter_slots, block_size, max_padded_routes, max_blocks,
      scan_workspace, scan_workspace_bytes, stream);
}

extern "C" size_t routed_lora_metadata_workspace_size(int num_experts,
                                                      int num_adapter_slots) {
  const int64_t pairs = static_cast<int64_t>(num_experts) * num_adapter_slots;
  if (pairs <= 0 || pairs > INT32_MAX) {
    return 0;
  }
  return mistralrs_routed_lora::metadata_workspace_size(
      static_cast<int>(pairs));
}

#define DEFINE_ROUTED_LORA_LAUNCHERS(SUFFIX, TYPE)                             \
  extern "C" int launch_routed_lora_direct_##SUFFIX(                           \
      const TYPE *input, TYPE *output,                                         \
      const mistralrs_routed_lora::AdapterWeight *weights,                     \
      const uint32_t *token_adapter_slots, const uint32_t *topk_expert_ids,    \
      const uint32_t *route_input_rows, const uint32_t *route_output_rows,     \
      const float *route_output_scales, int num_tokens, int top_k,             \
      int num_experts, int num_adapter_slots, int num_slices,                  \
      int input_features, int output_features, int output_row_stride,          \
      int output_slice_stride, int max_rank, int input_mode,                   \
      int output_splits, cudaStream_t stream) {                                \
    return mistralrs_routed_lora::launch_direct(                               \
        input, output, weights, token_adapter_slots, topk_expert_ids,          \
        route_input_rows, route_output_rows, route_output_scales, num_tokens,  \
        top_k, num_experts, num_adapter_slots, num_slices, input_features,     \
        output_features, output_row_stride, output_slice_stride, max_rank,     \
        input_mode, output_splits, stream);                                    \
  }                                                                            \
  extern "C" int launch_routed_lora_grouped_shrink_##SUFFIX(                   \
      const TYPE *input, const mistralrs_routed_lora::AdapterWeight *weights,  \
      const uint32_t *sorted_route_ids, const uint32_t *block_pair_ids,        \
      const uint32_t *route_input_rows, float *hidden, int num_routes,         \
      int max_blocks, int block_size, int top_k, int num_experts,              \
      int num_adapter_slots, int num_slices, int input_features, int max_rank, \
      int input_mode, cudaStream_t stream) {                                   \
    return mistralrs_routed_lora::launch_grouped_shrink(                       \
        input, weights, sorted_route_ids, block_pair_ids, route_input_rows,    \
        hidden, num_routes, max_blocks, block_size, top_k, num_experts,        \
        num_adapter_slots, num_slices, input_features, max_rank, input_mode,   \
        stream);                                                               \
  }                                                                            \
  extern "C" int launch_routed_lora_grouped_expand_##SUFFIX(                   \
      const float *hidden, TYPE *output,                                       \
      const mistralrs_routed_lora::AdapterWeight *weights,                     \
      const uint32_t *sorted_route_ids, const uint32_t *block_pair_ids,        \
      const uint32_t *route_output_rows, const float *route_output_scales,     \
      int num_routes, int max_blocks, int block_size, int num_experts,         \
      int num_adapter_slots, int num_slices, int output_features,              \
      int output_row_stride, int output_slice_stride, int max_rank,            \
      cudaStream_t stream) {                                                   \
    return mistralrs_routed_lora::launch_grouped_expand(                       \
        hidden, output, weights, sorted_route_ids, block_pair_ids,             \
        route_output_rows, route_output_scales, num_routes, max_blocks,        \
        block_size, num_experts, num_adapter_slots, num_slices,                \
        output_features, output_row_stride, output_slice_stride, max_rank,     \
        stream);                                                               \
  }

DEFINE_ROUTED_LORA_LAUNCHERS(f32, float)
DEFINE_ROUTED_LORA_LAUNCHERS(f16, __half)
DEFINE_ROUTED_LORA_LAUNCHERS(bf16, __nv_bfloat16)

#define DEFINE_ROUTED_LORA_WMMA_LAUNCHER(SUFFIX, TYPE)                         \
  extern "C" int launch_routed_lora_grouped_wmma_##SUFFIX(                     \
      const TYPE *input, TYPE *output,                                         \
      const mistralrs_routed_lora::AdapterWeight *weights,                     \
      const uint32_t *sorted_route_ids, const uint32_t *block_pair_ids,        \
      const uint32_t *route_input_rows, const uint32_t *route_output_rows,     \
      const float *route_output_scales, int num_routes, int max_blocks,        \
      int top_k, int num_experts, int num_adapter_slots, int num_slices,       \
      int input_features, int output_features, int output_row_stride,          \
      int output_slice_stride, int input_mode, int output_splits,              \
      cudaStream_t stream) {                                                   \
    return mistralrs_routed_lora::launch_grouped_wmma(                         \
        input, output, weights, sorted_route_ids, block_pair_ids,              \
        route_input_rows, route_output_rows, route_output_scales, num_routes,  \
        max_blocks, top_k, num_experts, num_adapter_slots, num_slices,         \
        input_features, output_features, output_row_stride,                    \
        output_slice_stride, input_mode, output_splits, stream);               \
  }

DEFINE_ROUTED_LORA_WMMA_LAUNCHER(f16, __half)
DEFINE_ROUTED_LORA_WMMA_LAUNCHER(bf16, __nv_bfloat16)
