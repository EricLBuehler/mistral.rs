#include "../gguf/vecdot.cuh"

extern "C" __global__ void quantize_q8_1(const float *__restrict__ x,
                                         void *__restrict__ vy, const int kx,
                                         const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

extern "C" __global__ void
quantize_q8_1_bf16(const __nv_bfloat16 *__restrict__ x, void *__restrict__ vy,
                   const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? __bfloat162float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

extern "C" __global__ void quantize_q8_1_f16(const half *__restrict__ x,
                                             void *__restrict__ vy,
                                             const int kx,
                                             const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;

  if (ix >= kx_padded) {
    return;
  }

  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;
  block_q8_1 *y = (block_q8_1 *)vy;

  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = ix < kx ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

extern "C" void launch_quantize_q8_1_bf16(const void *x, void *vy, int kx,
                                          int kx_padded, int num_rows,
                                          void *stream) {
  int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  quantize_q8_1_bf16<<<grid, block, 0, s>>>((const __nv_bfloat16 *)x, vy, kx,
                                            kx_padded);
}

extern "C" void launch_quantize_q8_1_f16(const void *x, void *vy, int kx,
                                         int kx_padded, int num_rows,
                                         void *stream) {
  int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  quantize_q8_1_f16<<<grid, block, 0, s>>>((const half *)x, vy, kx, kx_padded);
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void indexed_moe_forward(const void *__restrict__ all_weights,
                                    const void *__restrict__ all_inputs,
                                    const unsigned int *__restrict__ indices,
                                    float *__restrict__ all_outputs,
                                    const int n, const int k, const int batch,
                                    const int topk, const int k_padded,
                                    const int input_dim1) {

  const int current_batch = blockIdx.y;
  const int current_topk = blockIdx.z;
  const int task_id = current_batch * gridDim.z + current_topk;

  if (task_id >= gridDim.y * gridDim.z) {
    return;
  }

  const int input_idx = (input_dim1 == 1) ? current_batch : task_id;
  const unsigned int expert_id = indices[task_id];

  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t weight_blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t weight_expert_stride_bytes =
      (size_t)n * weight_blocks_per_row * weight_block_size;
  const size_t input_task_stride_bytes =
      (size_t)k_padded / QK8_1 * input_block_size;
  const size_t output_task_stride_elems = n;

  const void *current_input_ptr =
      (const char *)all_inputs + input_idx * input_task_stride_bytes;
  const void *current_weight_ptr =
      (const char *)all_weights + expert_id * weight_expert_stride_bytes;
  float *current_output_ptr = all_outputs + task_id * output_task_stride_elems;

  constexpr int ncols_y = 1;
  constexpr int nwarps = 4;
  constexpr int rows_per_cuda_block = 1;

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;

  if (row0 >= n) {
    return;
  }

  const int blocks_per_row_x = (k + qk - 1) / qk;
  const int blocks_per_col_y = k_padded / QK8_1;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp = 0.0f;

  const block_q_t *w = (const block_q_t *)current_weight_ptr;
  const block_q8_1 *x = (const block_q8_1 *)current_input_ptr;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));
    tmp += vec_dot_q_cuda(&w[kbx + row0 * blocks_per_row_x], &x[kby], kqs);
  }

  __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];
  if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
  }
  __syncthreads();

  if (threadIdx.y == 0) {
    for (int l = 0; l < nwarps - 1; ++l) {
      tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) {
      current_output_ptr[row0] = tmp;
    }
  }
}

extern "C" __global__ void indexed_moe_forward_q4_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                      vec_dot_q4_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q4_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                      vec_dot_q4_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                      vec_dot_q5_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                      vec_dot_q5_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q8_1_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                      vec_dot_q8_1_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q2k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                      vec_dot_q2_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q3k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                      vec_dot_q3_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q4k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                      vec_dot_q4_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                      vec_dot_q5_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q6k_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                      vec_dot_q6_K_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q8_0_q8_1(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded, const int input_dim1) {
  indexed_moe_forward<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                      vec_dot_q8_0_q8_1>(all_weights, all_inputs, indices,
                                         all_outputs, n, k, batch, topk,
                                         k_padded, input_dim1);
}

extern "C" void launch_quantize_q8_1(const float *x, void *vy, int kx,
                                     int kx_padded, int num_blocks_x,
                                     int num_rows, void *stream) {
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  quantize_q8_1<<<grid, block, 0, cuda_stream>>>(x, vy, kx, kx_padded);
}

extern "C" void launch_indexed_moe_forward_q2k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q2k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q3k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q3k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q6k_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q6k_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q8_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q8_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q4_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q4_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5_0_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5_0_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q5_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q5_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

extern "C" void launch_indexed_moe_forward_q8_1_q8_1(
    const void *all_weights, const void *all_inputs,
    const unsigned int *indices, float *all_outputs, int n, int k, int batch,
    int topk, int k_padded, int input_dim1, void *stream) {
  dim3 grid(n, batch, topk);
  dim3 block(WARP_SIZE, 4, 1);
  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  indexed_moe_forward_q8_1_q8_1<<<grid, block, 0, cuda_stream>>>(
      all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,
      k_padded, input_dim1);
}

static __device__ __forceinline__ float gelu_pytorch_tanh(float x) {
  float x3 = x * x * x;
  float inner = 0.7978845608028654f * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

enum MoeOutputType : int {
  MOE_OUTPUT_F32 = 0,
  MOE_OUTPUT_F16 = 1,
  MOE_OUTPUT_BF16 = 2,
};

static __device__ __forceinline__ void
moe_store_output(void *outputs, size_t index, int output_type, float value) {
  if (output_type == MOE_OUTPUT_F32) {
    static_cast<float *>(outputs)[index] = value;
  } else if (output_type == MOE_OUTPUT_F16) {
    static_cast<half *>(outputs)[index] = __float2half_rn(value);
  } else {
    static_cast<__nv_bfloat16 *>(outputs)[index] = __float2bfloat16_rn(value);
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_gate_up_pair_impl(
    const void *__restrict__ gate_weights, const void *__restrict__ up_weights,
    const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, void *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int num_experts, const int k_padded, const int output_type) {
  constexpr int ROWS_PER_WARP = 2;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const size_t output_base = (size_t)task_id * 2 * n;
  if (expert_id >= (unsigned int)num_experts) {
    for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
      if (threadIdx.x == 0) {
        moe_store_output(all_outputs, output_base + row0 + r, output_type,
                         0.0f);
        moe_store_output(all_outputs, output_base + n + row0 + r, output_type,
                         0.0f);
      }
    }
    return;
  }
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * sizeof(block_q_t);
  const size_t input_stride_bytes =
      (size_t)k_padded / QK8_1 * sizeof(block_q8_1);

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)current_batch * input_stride_bytes);
  const block_q_t *gate_w =
      (const block_q_t *)((const char *)gate_weights +
                          (size_t)expert_id * expert_stride_bytes);
  const block_q_t *up_w =
      (const block_q_t *)((const char *)up_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;
  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      gate_sum += vec_dot_q_cuda(&gate_w[kbx + (size_t)row * blocks_per_row],
                                 &x[kby], kqs);
      up_sum += vec_dot_q_cuda(&up_w[kbx + (size_t)row * blocks_per_row],
                               &x[kby], kqs);
    }

    gate_sum = warp_reduce_sum(gate_sum);
    up_sum = warp_reduce_sum(up_sum);
    if (threadIdx.x == 0) {
      moe_store_output(all_outputs, output_base + row, output_type, gate_sum);
      moe_store_output(all_outputs, output_base + n + row, output_type, up_sum);
    }
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_lora_down_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, void *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int num_experts, const int k_padded, const int output_type) {
  constexpr int ROWS_PER_WARP = 4;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const size_t output_base = (size_t)task_id * n;
  if (expert_id >= (unsigned int)num_experts) {
    for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
      if (threadIdx.x == 0) {
        moe_store_output(all_outputs, output_base + row0 + r, output_type,
                         0.0f);
      }
    }
    return;
  }
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * sizeof(block_q_t);
  const size_t input_stride_bytes =
      (size_t)k_padded / QK8_1 * sizeof(block_q8_1);

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)task_id * input_stride_bytes);
  const block_q_t *w =
      (const block_q_t *)((const char *)all_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;
  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      sum +=
          vec_dot_q_cuda(&w[kbx + (size_t)row * blocks_per_row], &x[kby], kqs);
    }

    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
      moe_store_output(all_outputs, output_base + row, output_type, sum);
    }
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_fused_gate_up_impl(
    const void *__restrict__ gate_weights, const void *__restrict__ up_weights,
    const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded,
    const int act_type) { // 0=gelu_pytorch_tanh, 1=silu

  constexpr int ROWS_PER_WARP = 2;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];

  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * weight_block_size;
  const size_t input_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;

  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)current_batch * input_stride_bytes);
  const block_q_t *gate_w =
      (const block_q_t *)((const char *)gate_weights +
                          (size_t)expert_id * expert_stride_bytes);
  const block_q_t *up_w =
      (const block_q_t *)((const char *)up_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;

  float *out = all_outputs + (size_t)task_id * n;

  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float g_sum = 0.0f;
    float u_sum = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      g_sum += vec_dot_q_cuda(&gate_w[kbx + (size_t)row * blocks_per_row],
                              &x[kby], kqs);
      u_sum += vec_dot_q_cuda(&up_w[kbx + (size_t)row * blocks_per_row],
                              &x[kby], kqs);
    }

    g_sum = warp_reduce_sum(g_sum);
    u_sum = warp_reduce_sum(u_sum);

    if (threadIdx.x == 0) {
      float activated =
          (act_type == 0) ? gelu_pytorch_tanh(g_sum) : silu(g_sum);
      out[row] = u_sum * activated;
    }
  }
}

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void moe_gemv_down_aggregate_impl(
    const void *__restrict__ all_weights, const void *__restrict__ all_inputs,
    const unsigned int *__restrict__ indices,
    const float *__restrict__ topk_weights_ptr, float *__restrict__ all_outputs,
    const int n, const int k, const int batch, const int topk,
    const int k_padded) {

  constexpr int ROWS_PER_WARP = 4;
  constexpr int NWARPS = 4;
  constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;

  const int warp_id = threadIdx.y;
  const int row0 = ROWS_PER_BLOCK * blockIdx.x + warp_id * ROWS_PER_WARP;
  const int current_topk = blockIdx.y;
  const int current_batch = blockIdx.z;

  if (row0 >= n)
    return;

  const int task_id = current_batch * topk + current_topk;
  const unsigned int expert_id = indices[task_id];
  const float tw = topk_weights_ptr[task_id];

  // Weight layout: [num_experts, n, k_blocks]
  const size_t weight_block_size = sizeof(block_q_t);
  const size_t input_block_size = sizeof(block_q8_1);
  const size_t blocks_per_row = ((size_t)k + qk - 1) / qk;
  const size_t expert_stride_bytes =
      (size_t)n * blocks_per_row * weight_block_size;
  const size_t input_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;

  // Each topk slot has its own input row (input_dim1=topk for down proj)
  const block_q8_1 *x =
      (const block_q8_1 *)((const char *)all_inputs +
                           (size_t)task_id * input_stride_bytes);

  const block_q_t *w =
      (const block_q_t *)((const char *)all_weights +
                          (size_t)expert_id * expert_stride_bytes);

  constexpr int blocks_per_iter = vdr * WARP_SIZE / qi;
  const int blocks_per_row_x = (int)blocks_per_row;

  float *out = all_outputs + (size_t)current_batch * n;

  for (int r = 0; r < ROWS_PER_WARP && row0 + r < n; ++r) {
    const int row = row0 + r;
    float tmp = 0.0f;

    for (int kbx = threadIdx.x / (qi / vdr); kbx < blocks_per_row_x;
         kbx += blocks_per_iter) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (threadIdx.x % (qi / vdr));
      tmp +=
          vec_dot_q_cuda(&w[kbx + (size_t)row * blocks_per_row], &x[kby], kqs);
    }

    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
      atomicAdd(&out[row], tmp * tw);
    }
  }
}

#define FUSED_GATE_UP_KERNEL(suffix, qk_val, qi_val, block_type, vdr_val,      \
                             vd_fn)                                            \
  extern "C" __global__ void moe_gemv_fused_gate_up_##suffix(                  \
      const void *__restrict__ gate_weights,                                   \
      const void *__restrict__ up_weights,                                     \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      float *__restrict__ all_outputs, const int n, const int k,               \
      const int batch, const int topk, const int k_padded,                     \
      const int act_type) {                                                    \
    moe_gemv_fused_gate_up_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(   \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, k_padded, act_type);                                      \
  }

FUSED_GATE_UP_KERNEL(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
FUSED_GATE_UP_KERNEL(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1)
FUSED_GATE_UP_KERNEL(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1)
FUSED_GATE_UP_KERNEL(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                     vec_dot_q5_0_q8_1)
FUSED_GATE_UP_KERNEL(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1)
FUSED_GATE_UP_KERNEL(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                     vec_dot_q8_1_q8_1)
FUSED_GATE_UP_KERNEL(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1)
FUSED_GATE_UP_KERNEL(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                     vec_dot_q3_K_q8_1)
FUSED_GATE_UP_KERNEL(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1)
FUSED_GATE_UP_KERNEL(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                     vec_dot_q5_K_q8_1)
FUSED_GATE_UP_KERNEL(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)

#define LORA_DECODE_KERNELS(suffix, qk_val, qi_val, block_type, vdr_val,       \
                            vd_fn)                                             \
  extern "C" __global__ void moe_gemv_gate_up_pair_##suffix(                   \
      const void *__restrict__ gate_weights,                                   \
      const void *__restrict__ up_weights,                                     \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      void *__restrict__ all_outputs, const int n, const int k,                \
      const int batch, const int topk, const int num_experts,                  \
      const int k_padded, const int output_type) {                             \
    moe_gemv_gate_up_pair_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(    \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, num_experts, k_padded, output_type);                      \
  }                                                                            \
  extern "C" __global__ void moe_gemv_lora_down_##suffix(                      \
      const void *__restrict__ all_weights,                                    \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      void *__restrict__ all_outputs, const int n, const int k,                \
      const int batch, const int topk, const int num_experts,                  \
      const int k_padded, const int output_type) {                             \
    moe_gemv_lora_down_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(       \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,      \
        num_experts, k_padded, output_type);                                   \
  }

LORA_DECODE_KERNELS(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                    vec_dot_q8_0_q8_1)
LORA_DECODE_KERNELS(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                    vec_dot_q4_0_q8_1)
LORA_DECODE_KERNELS(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                    vec_dot_q4_1_q8_1)
LORA_DECODE_KERNELS(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                    vec_dot_q5_0_q8_1)
LORA_DECODE_KERNELS(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                    vec_dot_q5_1_q8_1)
LORA_DECODE_KERNELS(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                    vec_dot_q8_1_q8_1)
LORA_DECODE_KERNELS(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                    vec_dot_q2_K_q8_1)
LORA_DECODE_KERNELS(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                    vec_dot_q3_K_q8_1)
LORA_DECODE_KERNELS(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                    vec_dot_q4_K_q8_1)
LORA_DECODE_KERNELS(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                    vec_dot_q5_K_q8_1)
LORA_DECODE_KERNELS(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                    vec_dot_q6_K_q8_1)

#define DOWN_AGGREGATE_KERNEL(suffix, qk_val, qi_val, block_type, vdr_val,     \
                              vd_fn)                                           \
  extern "C" __global__ void moe_gemv_down_aggregate_##suffix(                 \
      const void *__restrict__ all_weights,                                    \
      const void *__restrict__ all_inputs,                                     \
      const unsigned int *__restrict__ indices,                                \
      const float *__restrict__ topk_weights_ptr,                              \
      float *__restrict__ all_outputs, const int n, const int k,               \
      const int batch, const int topk, const int k_padded) {                   \
    moe_gemv_down_aggregate_impl<qk_val, qi_val, block_type, vdr_val, vd_fn>(  \
        all_weights, all_inputs, indices, topk_weights_ptr, all_outputs, n, k, \
        batch, topk, k_padded);                                                \
  }

DOWN_AGGREGATE_KERNEL(q8_0_q8_1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                      vec_dot_q8_0_q8_1)
DOWN_AGGREGATE_KERNEL(q4_0_q8_1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                      vec_dot_q4_0_q8_1)
DOWN_AGGREGATE_KERNEL(q4_1_q8_1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                      vec_dot_q4_1_q8_1)
DOWN_AGGREGATE_KERNEL(q5_0_q8_1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                      vec_dot_q5_0_q8_1)
DOWN_AGGREGATE_KERNEL(q5_1_q8_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                      vec_dot_q5_1_q8_1)
DOWN_AGGREGATE_KERNEL(q8_1_q8_1, QK8_1, QI8_1, block_q8_1, VDR_Q8_1_Q8_1_MMVQ,
                      vec_dot_q8_1_q8_1)
DOWN_AGGREGATE_KERNEL(q2k_q8_1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                      vec_dot_q2_K_q8_1)
DOWN_AGGREGATE_KERNEL(q3k_q8_1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                      vec_dot_q3_K_q8_1)
DOWN_AGGREGATE_KERNEL(q4k_q8_1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                      vec_dot_q4_K_q8_1)
DOWN_AGGREGATE_KERNEL(q5k_q8_1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                      vec_dot_q5_K_q8_1)
DOWN_AGGREGATE_KERNEL(q6k_q8_1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                      vec_dot_q6_K_q8_1)

#define LAUNCH_FUSED_GATE_UP(suffix)                                           \
  extern "C" void launch_moe_gemv_fused_gate_up_##suffix(                      \
      const void *gate_weights, const void *up_weights,                        \
      const void *all_inputs, const unsigned int *indices, float *all_outputs, \
      int n, int k, int batch, int topk, int k_padded, int act_type,           \
      void *stream) {                                                          \
    const int NWARPS = 4;                                                      \
    const int ROWS_PER_WARP = 2;                                               \
    const int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                         \
    dim3 grid(1 + (n - 1) / ROWS_PER_BLOCK, topk, batch);                      \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_fused_gate_up_##suffix<<<grid, block, 0, s>>>(                    \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, k_padded, act_type);                                      \
  }

LAUNCH_FUSED_GATE_UP(q8_0_q8_1)
LAUNCH_FUSED_GATE_UP(q4_0_q8_1)
LAUNCH_FUSED_GATE_UP(q4_1_q8_1)
LAUNCH_FUSED_GATE_UP(q5_0_q8_1)
LAUNCH_FUSED_GATE_UP(q5_1_q8_1)
LAUNCH_FUSED_GATE_UP(q8_1_q8_1)
LAUNCH_FUSED_GATE_UP(q2k_q8_1)
LAUNCH_FUSED_GATE_UP(q3k_q8_1)
LAUNCH_FUSED_GATE_UP(q4k_q8_1)
LAUNCH_FUSED_GATE_UP(q5k_q8_1)
LAUNCH_FUSED_GATE_UP(q6k_q8_1)

#define LAUNCH_LORA_DECODE(suffix)                                             \
  extern "C" int launch_moe_gemv_gate_up_pair_##suffix(                        \
      const void *gate_weights, const void *up_weights,                        \
      const void *all_inputs, const unsigned int *indices, void *all_outputs,  \
      int n, int k, int batch, int topk, int k_padded, int num_experts,        \
      int output_type, void *stream) {                                         \
    if (output_type < MOE_OUTPUT_F32 || output_type > MOE_OUTPUT_BF16)         \
      return static_cast<int>(cudaErrorInvalidValue);                          \
    constexpr int NWARPS = 4;                                                  \
    constexpr int ROWS_PER_WARP = 2;                                           \
    constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                     \
    dim3 grid(1 + (n - 1) / ROWS_PER_BLOCK, topk, batch);                      \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_gate_up_pair_##suffix<<<grid, block, 0, s>>>(                     \
        gate_weights, up_weights, all_inputs, indices, all_outputs, n, k,      \
        batch, topk, num_experts, k_padded, output_type);                      \
    return static_cast<int>(cudaGetLastError());                               \
  }                                                                            \
  extern "C" int launch_moe_gemv_lora_down_##suffix(                           \
      const void *all_weights, const void *all_inputs,                         \
      const unsigned int *indices, void *all_outputs, int n, int k, int batch, \
      int topk, int k_padded, int num_experts, int output_type,                \
      void *stream) {                                                          \
    if (output_type < MOE_OUTPUT_F32 || output_type > MOE_OUTPUT_BF16)         \
      return static_cast<int>(cudaErrorInvalidValue);                          \
    constexpr int NWARPS = 4;                                                  \
    constexpr int ROWS_PER_WARP = 4;                                           \
    constexpr int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                     \
    dim3 grid((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, topk, batch);         \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_lora_down_##suffix<<<grid, block, 0, s>>>(                        \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk,      \
        num_experts, k_padded, output_type);                                   \
    return static_cast<int>(cudaGetLastError());                               \
  }

LAUNCH_LORA_DECODE(q8_0_q8_1)
LAUNCH_LORA_DECODE(q4_0_q8_1)
LAUNCH_LORA_DECODE(q4_1_q8_1)
LAUNCH_LORA_DECODE(q5_0_q8_1)
LAUNCH_LORA_DECODE(q5_1_q8_1)
LAUNCH_LORA_DECODE(q8_1_q8_1)
LAUNCH_LORA_DECODE(q2k_q8_1)
LAUNCH_LORA_DECODE(q3k_q8_1)
LAUNCH_LORA_DECODE(q4k_q8_1)
LAUNCH_LORA_DECODE(q5k_q8_1)
LAUNCH_LORA_DECODE(q6k_q8_1)

#define LAUNCH_DOWN_AGGREGATE(suffix)                                          \
  extern "C" void launch_moe_gemv_down_aggregate_##suffix(                     \
      const void *all_weights, const void *all_inputs,                         \
      const unsigned int *indices, const float *topk_weights_ptr,              \
      float *all_outputs, int n, int k, int batch, int topk, int k_padded,     \
      void *stream) {                                                          \
    const int NWARPS = 4;                                                      \
    const int ROWS_PER_WARP = 4;                                               \
    const int ROWS_PER_BLOCK = NWARPS * ROWS_PER_WARP;                         \
    dim3 grid((n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, topk, batch);         \
    dim3 block(WARP_SIZE, NWARPS, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    moe_gemv_down_aggregate_##suffix<<<grid, block, 0, s>>>(                   \
        all_weights, all_inputs, indices, topk_weights_ptr, all_outputs, n, k, \
        batch, topk, k_padded);                                                \
  }

LAUNCH_DOWN_AGGREGATE(q8_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q4_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q4_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q5_0_q8_1)
LAUNCH_DOWN_AGGREGATE(q5_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q8_1_q8_1)
LAUNCH_DOWN_AGGREGATE(q2k_q8_1)
LAUNCH_DOWN_AGGREGATE(q3k_q8_1)
LAUNCH_DOWN_AGGREGATE(q4k_q8_1)
LAUNCH_DOWN_AGGREGATE(q5k_q8_1)
LAUNCH_DOWN_AGGREGATE(q6k_q8_1)
