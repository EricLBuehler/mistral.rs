#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "cuda_compat.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {
namespace moe {

namespace {
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
}  // namespace

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids,
                                            int32_t* sorted_token_ids,
                                            int32_t* expert_ids,
                                            int32_t* total_tokens_post_pad,
                                            int32_t num_experts,
                                            int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts +
                      1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                          block_size) *
                      block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

// TODO(simon): this is temporarily adapted from
// https://github.com/sgl-project/sglang/commit/31548116a8dc8c6df7e146e0587335a59fc5b9d7
// we did this to unblock Deepseek V3 but there should be a better
// implementation to manage shared memory.
template <typename scalar_t>
__global__ void moe_align_block_size_global_mem_kernel(
    scalar_t* __restrict__ topk_ids,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* tokens_cnts,
    int32_t* cumsum
  ) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] +=
          tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                          block_size) *
                      block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += VLLM_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

}  // namespace moe
}  // namespace vllm

#define CALL_MOE_ALIGN_BLOCK_SIZE_KERNEL(T)                                                       \
  vllm::moe::moe_align_block_size_kernel<T, int32_t><<<1, num_thread, shared_mem_i32, stream>>>(  \
    reinterpret_cast<T*>(topk_ids),                                                               \
    reinterpret_cast<int32_t*>(sorted_token_ids),                                                 \
    reinterpret_cast<int32_t*>(experts_ids),                                                      \
    reinterpret_cast<int32_t*>(num_tokens_post_pad),                                              \
    num_experts,                                                                                  \
    block_size,                                                                                   \
    numel                                                                                         \
  );

extern "C" void moe_align_block_size(
  void *topk_ids, 
  int64_t num_experts,
  int64_t block_size,
  int64_t numel,
  void *sorted_token_ids,
  void *experts_ids,
  void *num_tokens_post_pad,
  uint32_t dtype
) {
  const int32_t num_thread = max((int32_t)num_experts, WARP_SIZE);
  const int32_t shared_mem_i32 =
      ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  const cudaStream_t stream = 0;

  if (dtype == 0) {
    CALL_MOE_ALIGN_BLOCK_SIZE_KERNEL(half);
  } else if (dtype == 1) {
    CALL_MOE_ALIGN_BLOCK_SIZE_KERNEL(__nv_bfloat16);
  } else {
    CALL_MOE_ALIGN_BLOCK_SIZE_KERNEL(float);
  }
}

#define CALL_MOE_SUM_KERNEL(T, TOPK)                               \
  vllm::moe::moe_sum_kernel<T, TOPK><<<grid, block, 0, stream>>>(  \
    reinterpret_cast<T*>(output),                                  \
    reinterpret_cast<T*>(input),                                   \
    hidden_size                                                    \
  );

extern "C" void moe_sum(
  void *input,              // [num_tokens, topk, hidden_size]
  void *output,             // [num_tokens, hidden_size]

  uint64_t hidden_size,
  uint64_t num_tokens,
  uint64_t topk,
  uint32_t dtype            // 0 => f16; 1 => bf16; 2 => f32
) {
  dim3 grid(num_tokens);
  dim3 block(MIN(hidden_size, 1024));
  const cudaStream_t stream = 0;

  switch (topk) {
    case 2:
      if (dtype == 0) {
        CALL_MOE_SUM_KERNEL(half, 2);
      } else if (dtype == 1) {
        CALL_MOE_SUM_KERNEL(__nv_bfloat16, 2);
      } else {
        CALL_MOE_SUM_KERNEL(float, 2);
      }
      break;
    case 3:
      if (dtype == 0) {
        CALL_MOE_SUM_KERNEL(half, 3);
      } else if (dtype == 1) {
        CALL_MOE_SUM_KERNEL(__nv_bfloat16, 3);
      } else {
        CALL_MOE_SUM_KERNEL(float, 3);
      }
      break;
    case 4:
      if (dtype == 0) {
        CALL_MOE_SUM_KERNEL(half, 4);
      } else if (dtype == 1) {
        CALL_MOE_SUM_KERNEL(__nv_bfloat16, 4);
      } else {
        CALL_MOE_SUM_KERNEL(float, 4);
      }
      break;
    default:
      if (dtype == 0) {
        CALL_MOE_SUM_KERNEL(half, 1);
      } else if (dtype == 1) {
        CALL_MOE_SUM_KERNEL(__nv_bfloat16, 1);
      } else {
        CALL_MOE_SUM_KERNEL(float, 1);
      }
      break;
  }
}
