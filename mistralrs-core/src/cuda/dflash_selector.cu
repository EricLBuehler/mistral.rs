#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int DFLASH_SELECTOR_MAX_K = 128;
constexpr int DFLASH_SELECTOR_BLOCK_SIZE = 256;
constexpr int DFLASH_SELECTOR_WARP_SIZE = 32;
constexpr uint32_t DFLASH_SELECTOR_INVALID_TOKEN = UINT32_MAX;

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
    const int k, const int packed_width, const int hidden_dtype,
    const int predecessor_dtype, const int successor_dtype) {
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
  for (int position = 0; position < positions; ++position) {
    const int row = sequence * positions + position;
    const size_t hidden_row = static_cast<size_t>(row) * rank;
    const size_t predecessor_row = static_cast<size_t>(predecessor) * rank;
    const float *packed_row =
        packed_topk + static_cast<size_t>(row) * packed_width;

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

__global__ void dflash_sample_select_kernel(
    const float *__restrict__ packed_topk,
    const void *__restrict__ projected_hidden,
    const void *__restrict__ predecessor_codebook,
    const void *__restrict__ successor_codebook,
    const uint32_t *__restrict__ anchors,
    const float *__restrict__ inverse_temperatures,
    const float *__restrict__ uniforms, uint32_t *__restrict__ selected,
    uint32_t *__restrict__ candidate_ids,
    float *__restrict__ candidate_probs, const int batch, const int positions,
    const int rank, const int vocab, const int k, const int packed_width,
    const int hidden_dtype, const int predecessor_dtype,
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
  const float inverse_temperature = inverse_temperatures[sequence];

  for (int position = 0; position < positions; ++position) {
    const int row = sequence * positions + position;
    const size_t hidden_row = static_cast<size_t>(row) * rank;
    const size_t predecessor_row = static_cast<size_t>(predecessor) * rank;
    const float *packed_row =
        packed_topk + static_cast<size_t>(row) * packed_width;
    uint32_t *row_candidate_ids =
        candidate_ids + static_cast<size_t>(row) * k;
    float *row_candidate_probs =
        candidate_probs + static_cast<size_t>(row) * k;

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
        row_candidate_ids[candidate_slot] = candidate;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int selected_slot = -1;
      if (inverse_temperature <= 0.0f) {
        float best_score = -INFINITY;
        selected_slot = 0;
        for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
          const float score = candidate_scores[candidate_slot];
          if (score > best_score) {
            best_score = score;
            selected_slot = candidate_slot;
          }
        }
        for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
          row_candidate_probs[candidate_slot] =
              candidate_slot == selected_slot ? 1.0f : 0.0f;
        }
      } else if (isfinite(inverse_temperature)) {
        float max_score = -INFINITY;
        bool has_nan = false;
        for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
          const float score = candidate_scores[candidate_slot];
          has_nan |= isnan(score);
          max_score = fmaxf(max_score, score);
        }

        float denominator = 0.0f;
        if (!has_nan && isfinite(max_score)) {
          for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
            const float weight =
                expf((candidate_scores[candidate_slot] - max_score) *
                     inverse_temperature);
            row_candidate_probs[candidate_slot] = weight;
            denominator += weight;
          }
        }

        const float uniform = uniforms[row];
        if (!has_nan && isfinite(max_score) && denominator > 0.0f &&
            isfinite(denominator) && uniform >= 0.0f && uniform < 1.0f &&
            isfinite(uniform)) {
          const float target = fminf(uniform * denominator,
                                     nextafterf(denominator, -INFINITY));
          float cumulative = 0.0f;
          for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
            const float weight = row_candidate_probs[candidate_slot];
            row_candidate_probs[candidate_slot] = weight / denominator;
            cumulative += weight;
            if (selected_slot < 0 && target < cumulative) {
              selected_slot = candidate_slot;
            }
          }
          if (selected_slot < 0) {
            for (int candidate_slot = k - 1; candidate_slot >= 0;
                 --candidate_slot) {
              if (row_candidate_probs[candidate_slot] > 0.0f) {
                selected_slot = candidate_slot;
                break;
              }
            }
          }
        }
      }

      if (selected_slot < 0) {
        predecessor = DFLASH_SELECTOR_INVALID_TOKEN;
        selected[row] = DFLASH_SELECTOR_INVALID_TOKEN;
        for (int candidate_slot = 0; candidate_slot < k; ++candidate_slot) {
          row_candidate_probs[candidate_slot] = NAN;
        }
      } else {
        predecessor = row_candidate_ids[selected_slot];
        selected[row] = predecessor;
      }
    }
    __syncthreads();
  }
}

} // namespace

extern "C" void dflash_greedy_select(
    const float *packed_topk, const void *projected_hidden,
    const void *predecessor_codebook, const void *successor_codebook,
    const uint32_t *anchors, uint32_t *selected, int batch, int positions,
    int rank, int vocab, int k, int packed_width, int hidden_dtype,
    int predecessor_dtype, int successor_dtype, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dflash_greedy_select_kernel<<<batch, DFLASH_SELECTOR_BLOCK_SIZE, 0,
                                cuda_stream>>>(
      packed_topk, projected_hidden, predecessor_codebook, successor_codebook,
      anchors, selected, batch, positions, rank, vocab, k, packed_width,
      hidden_dtype, predecessor_dtype, successor_dtype);
}

extern "C" void dflash_sample_select(
    const float *packed_topk, const void *projected_hidden,
    const void *predecessor_codebook, const void *successor_codebook,
    const uint32_t *anchors, const float *inverse_temperatures,
    const float *uniforms, uint32_t *selected, uint32_t *candidate_ids,
    float *candidate_probs, int batch, int positions, int rank, int vocab,
    int k, int packed_width, int hidden_dtype, int predecessor_dtype,
    int successor_dtype, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dflash_sample_select_kernel<<<batch, DFLASH_SELECTOR_BLOCK_SIZE, 0,
                                cuda_stream>>>(
      packed_topk, projected_hidden, predecessor_codebook, successor_codebook,
      anchors, inverse_temperatures, uniforms, selected, candidate_ids,
      candidate_probs, batch, positions, rank, vocab, k, packed_width,
      hidden_dtype, predecessor_dtype, successor_dtype);
}
