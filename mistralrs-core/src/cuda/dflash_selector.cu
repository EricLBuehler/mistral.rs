#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int DFLASH_SELECTOR_MAX_K = 128;
constexpr int DFLASH_SELECTOR_BLOCK_SIZE = 256;
constexpr int DFLASH_SELECTOR_WARP_SIZE = 32;

enum DFlashSelectorDType : int {
  DFLASH_SELECTOR_F32 = 0,
  DFLASH_SELECTOR_BF16 = 1,
};

__device__ __forceinline__ float selector_load(const void *data,
                                               const size_t index,
                                               const int dtype) {
  if (dtype == DFLASH_SELECTOR_BF16) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(data)[index]);
  }
  return reinterpret_cast<const float *>(data)[index];
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = DFLASH_SELECTOR_WARP_SIZE / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void dflash_greedy_select_kernel(
    const float *__restrict__ packed_topk,
    const void *__restrict__ projected_hidden,
    const void *__restrict__ predecessor_codebook,
    const void *__restrict__ successor_codebook,
    const uint32_t *__restrict__ anchors, uint32_t *__restrict__ selected,
    const int batch, const int positions, const int rank, const int vocab,
    const int k, const int hidden_dtype, const int predecessor_dtype,
    const int successor_dtype) {
  const int sequence = blockIdx.x;
  if (sequence >= batch) {
    return;
  }

  __shared__ float candidate_scores[DFLASH_SELECTOR_MAX_K];
  __shared__ uint32_t predecessor;

  if (threadIdx.x == 0) {
    predecessor = anchors[sequence];
  }
  __syncthreads();

  const int lane = threadIdx.x % DFLASH_SELECTOR_WARP_SIZE;
  const int warp = threadIdx.x / DFLASH_SELECTOR_WARP_SIZE;
  const int warps = blockDim.x / DFLASH_SELECTOR_WARP_SIZE;
  const int packed_width = 2 * k + 2;

  for (int position = 0; position < positions; ++position) {
    const int row = sequence * positions + position;
    const size_t hidden_row = static_cast<size_t>(row) * rank;
    const size_t predecessor_row = static_cast<size_t>(predecessor) * rank;
    const float *packed_row = packed_topk + static_cast<size_t>(row) * packed_width;

    for (int candidate_slot = warp; candidate_slot < k;
         candidate_slot += warps) {
      const uint32_t candidate =
          static_cast<uint32_t>(packed_row[k + candidate_slot]);
      float score = -INFINITY;
      if (predecessor < static_cast<uint32_t>(vocab) &&
          candidate < static_cast<uint32_t>(vocab)) {
        float dot = 0.0f;
        const size_t successor_row = static_cast<size_t>(candidate) * rank;
        for (int column = lane; column < rank;
             column += DFLASH_SELECTOR_WARP_SIZE) {
          dot += selector_load(predecessor_codebook,
                               predecessor_row + column,
                               predecessor_dtype) *
                 selector_load(projected_hidden, hidden_row + column,
                               hidden_dtype) *
                 selector_load(successor_codebook, successor_row + column,
                               successor_dtype);
        }
        dot = warp_sum(dot);
        if (lane == 0) {
          score = packed_row[candidate_slot] + dot;
        }
      }
      if (lane == 0) {
        candidate_scores[candidate_slot] = score;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float best_score = -INFINITY;
      int best_slot = 0;
      for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
        const float score = candidate_scores[candidate_slot];
        if (score > best_score) {
          best_score = score;
          best_slot = candidate_slot;
        }
      }
      predecessor = static_cast<uint32_t>(packed_row[k + best_slot]);
      selected[row] = predecessor;
    }
    __syncthreads();
  }
}

} // namespace

extern "C" void dflash_greedy_select(
    const float *packed_topk, const void *projected_hidden,
    const void *predecessor_codebook, const void *successor_codebook,
    const uint32_t *anchors, uint32_t *selected, int batch, int positions,
    int rank, int vocab, int k, int hidden_dtype, int predecessor_dtype,
    int successor_dtype, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dflash_greedy_select_kernel<<<batch, DFLASH_SELECTOR_BLOCK_SIZE, 0,
                                cuda_stream>>>(
      packed_topk, projected_hidden, predecessor_codebook, successor_codebook,
      anchors, selected, batch, positions, rank, vocab, k, hidden_dtype,
      predecessor_dtype, successor_dtype);
}
