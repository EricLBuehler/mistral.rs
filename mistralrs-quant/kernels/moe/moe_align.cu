// Derived from vLLM (Apache-2.0): https://github.com/vllm-project/vllm

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace {

__device__ void _moe_align_block_size(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t padded_num_experts, int32_t experts_per_warp,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded, int32_t max_num_m_blocks) {
  extern __shared__ int32_t shared_counts[];

  // blockIdx.x == 1 fills sorted_token_ids with the sentinel `numel`.
  if (blockIdx.x % 2) {
    for (size_t it = threadIdx.x; it < (size_t)max_num_tokens_padded; it += blockDim.x) {
      sorted_token_ids[it] = (int32_t)numel;
    }
    return;
  }

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }
  __syncthreads();

  using BlockScan = cub::BlockScan<int32_t, 1024>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int expert_count = 0;
  int expert_id = threadIdx.x;
  if (expert_id < num_experts) {
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
    expert_count = CEILDIV(expert_count, block_size) * block_size;
  }

  int cumsum_val;
  BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
  if (expert_id <= num_experts) {
    cumsum[expert_id] = cumsum_val;
  }
  if (expert_id == num_experts) {
    total_tokens_post_pad[0] = cumsum_val;
  }
  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  // Fill remaining blocks with -1.
  const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
  for (size_t i = fill_start_idx; i < (size_t)max_num_m_blocks; i += blockDim.x) {
    expert_ids[i] = -1;
  }
}

__global__ void moe_align_block_size_kernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts, int32_t padded_num_experts, int32_t experts_per_warp,
    int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded) {
  _moe_align_block_size(topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad,
                        num_experts, padded_num_experts, experts_per_warp, block_size,
                        numel, cumsum, max_num_tokens_padded,
                        CEILDIV(max_num_tokens_padded, block_size));
}

__global__ void count_and_sort_expert_tokens_kernel(
    const int32_t* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer, size_t numel, int32_t num_experts) {
  const size_t tid = blockIdx.y * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.y;
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    if (expert_id >= num_experts) {
      continue;
    }
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = (int32_t)i;
  }
}

}  // namespace

extern "C" void launch_moe_align(const int32_t* topk_ids, int32_t* sorted_token_ids,
                                 int32_t* expert_ids, int32_t* num_tokens_post_pad,
                                 int32_t* cumsum, int32_t num_experts,
                                 int32_t block_size, int32_t numel,
                                 int32_t max_num_tokens_padded, cudaStream_t stream) {
  int32_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  int experts_per_warp = WARP_SIZE;
  int threads = 1024;

  size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
  size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

  // blockIdx.x == 0: count + align (BlockScan); blockIdx.x == 1: fill sentinels.
  moe_align_block_size_kernel<<<2, threads, shared_mem_size, stream>>>(
      topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts,
      padded_num_experts, experts_per_warp, block_size, (size_t)numel, cumsum,
      max_num_tokens_padded);

  const int block_threads = 256 < threads ? 256 : threads;
  const int n_blocks = (numel + block_threads - 1) / block_threads;
  const int actual_blocks = n_blocks < 65535 ? n_blocks : 65535;
  dim3 grid_dims(1, actual_blocks);
  count_and_sort_expert_tokens_kernel<<<grid_dims, block_threads, 0, stream>>>(
      topk_ids, sorted_token_ids, cumsum, (size_t)numel, num_experts);
}
