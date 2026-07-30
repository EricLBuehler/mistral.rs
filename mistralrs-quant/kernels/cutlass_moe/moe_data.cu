// Data-preparation kernels for the CUTLASS grouped-GEMM MoE path.
// Ported from vLLM csrc/.../cutlass/moe/moe_data.cu (Apache-2.0), bf16 variant without fp8 scales.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int THREADS_PER_EXPERT = 512;
constexpr int GATHER_THREADS = 256;

__global__ void compute_problem_sizes_kernel(const int32_t* __restrict__ topk_ids,
                                             int32_t* problem_sizes1,
                                             int32_t* problem_sizes2,
                                             int32_t* atomic_buffer,
                                             const int topk_length, const int n,
                                             const int k, const bool is_gated) {
  int expert_id = blockIdx.x;
  int const n1 = is_gated ? 2 * n : n;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = n1;
    problem_sizes1[expert_id * 3 + 2] = k;
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = k;
    problem_sizes2[expert_id * 3 + 2] = n;
  }
}

__global__ void compute_expert_offsets_kernel(
    const int32_t* __restrict__ problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, const int num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_arg_sorts_kernel(const int32_t* __restrict__ topk_ids,
                                         int32_t* input_permutation,
                                         int32_t* output_permutation,
                                         int32_t* atomic_buffer,
                                         const int topk_length, const int topk) {
  int const blk_expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == blk_expert_id) {
      int start = atomicAdd(&atomic_buffer[blk_expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

// dst[r, :] = src[map[r], :], rows of width k in bf16.
__global__ void gather_rows_bf16_kernel(__nv_bfloat16* __restrict__ dst,
                                        const __nv_bfloat16* __restrict__ src,
                                        const int32_t* __restrict__ map,
                                        const int num_rows, const int k) {
  int row = blockIdx.x;
  if (row >= num_rows) return;
  const __nv_bfloat16* src_row = src + static_cast<int64_t>(map[row]) * k;
  __nv_bfloat16* dst_row = dst + static_cast<int64_t>(row) * k;
  for (int i = threadIdx.x; i < k; i += GATHER_THREADS) {
    dst_row[i] = src_row[i];
  }
}

// out[slot, :] = weight[slot] * in[out_perm[slot], :] for token-order slots; bf16 rows, f32 weights.
__global__ void gather_weighted_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                            const __nv_bfloat16* __restrict__ in,
                                            const int32_t* __restrict__ out_perm,
                                            const float* __restrict__ weights,
                                            const int num_rows, const int k) {
  int row = blockIdx.x;
  if (row >= num_rows) return;
  const float w = weights[row];
  const __nv_bfloat16* in_row = in + static_cast<int64_t>(out_perm[row]) * k;
  __nv_bfloat16* out_row = out + static_cast<int64_t>(row) * k;
  for (int i = threadIdx.x; i < k; i += GATHER_THREADS) {
    out_row[i] = __float2bfloat16(w * __bfloat162float(in_row[i]));
  }
}

// Per-expert base pointers and leading dims for a grouped GEMM over expert-sorted rows.
// a: [total_rows, k] row-major, b: [num_experts, n, k] (row-major within expert -> column-major [k, n] view),
// d: [total_rows, n] row-major.
__global__ void get_group_starts_bf16_kernel(
    const int32_t* __restrict__ expert_offsets, const __nv_bfloat16* a_base,
    const __nv_bfloat16* b_base, __nv_bfloat16* d_base,
    __nv_bfloat16 const** a_ptrs, __nv_bfloat16 const** b_ptrs,
    __nv_bfloat16** d_ptrs, int64_t* lda, int64_t* ldb, int64_t* ldd,
    const int64_t n, const int64_t k) {
  int expert_id = threadIdx.x;
  int64_t expert_offset = expert_offsets[expert_id];

  a_ptrs[expert_id] = a_base + expert_offset * k;
  b_ptrs[expert_id] = b_base + static_cast<int64_t>(expert_id) * n * k;
  d_ptrs[expert_id] = d_base + expert_offset * n;
  lda[expert_id] = k;
  ldb[expert_id] = k;
  ldd[expert_id] = n;
}

extern "C" void launch_cutlass_moe_problem_sizes(
    const int32_t* topk_ids, int32_t* problem_sizes1, int32_t* problem_sizes2,
    int32_t* atomic_buffer, int num_experts, int topk_length, int n, int k,
    bool is_gated, cudaStream_t stream) {
  cudaMemsetAsync(atomic_buffer, 0, num_experts * sizeof(int32_t), stream);
  compute_problem_sizes_kernel<<<num_experts, THREADS_PER_EXPERT, 0, stream>>>(
      topk_ids, problem_sizes1, problem_sizes2, atomic_buffer, topk_length, n,
      k, is_gated);
}

extern "C" void launch_cutlass_moe_expert_offsets(
    const int32_t* problem_sizes1, int32_t* expert_offsets,
    int32_t* atomic_buffer, int num_experts, cudaStream_t stream) {
  compute_expert_offsets_kernel<<<1, 1, 0, stream>>>(
      problem_sizes1, expert_offsets, atomic_buffer, num_experts);
}

extern "C" void launch_cutlass_moe_arg_sorts(
    const int32_t* topk_ids, int32_t* input_permutation,
    int32_t* output_permutation, int32_t* atomic_buffer, int num_experts,
    int topk_length, int topk, cudaStream_t stream) {
  compute_arg_sorts_kernel<<<num_experts, THREADS_PER_EXPERT, 0, stream>>>(
      topk_ids, input_permutation, output_permutation, atomic_buffer,
      topk_length, topk);
}

extern "C" void launch_cutlass_moe_gather_rows_bf16(
    void* dst, const void* src, const int32_t* map, int num_rows, int k,
    cudaStream_t stream) {
  gather_rows_bf16_kernel<<<num_rows, GATHER_THREADS, 0, stream>>>(
      static_cast<__nv_bfloat16*>(dst), static_cast<const __nv_bfloat16*>(src),
      map, num_rows, k);
}

extern "C" void launch_cutlass_moe_gather_weighted_bf16(
    void* out, const void* in, const int32_t* out_perm, const float* weights,
    int num_rows, int k, cudaStream_t stream) {
  gather_weighted_bf16_kernel<<<num_rows, GATHER_THREADS, 0, stream>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(in),
      out_perm, weights, num_rows, k);
}

extern "C" void launch_cutlass_moe_group_starts_bf16(
    const int32_t* expert_offsets, const void* a_base, const void* b_base,
    void* d_base, const void** a_ptrs, const void** b_ptrs, void** d_ptrs,
    int64_t* lda, int64_t* ldb, int64_t* ldd, int num_experts, int64_t n,
    int64_t k, cudaStream_t stream) {
  get_group_starts_bf16_kernel<<<1, num_experts, 0, stream>>>(
      expert_offsets, static_cast<const __nv_bfloat16*>(a_base),
      static_cast<const __nv_bfloat16*>(b_base),
      static_cast<__nv_bfloat16*>(d_base),
      reinterpret_cast<__nv_bfloat16 const**>(a_ptrs),
      reinterpret_cast<__nv_bfloat16 const**>(b_ptrs),
      reinterpret_cast<__nv_bfloat16**>(d_ptrs), lda, ldb, ldd, n, k);
}
