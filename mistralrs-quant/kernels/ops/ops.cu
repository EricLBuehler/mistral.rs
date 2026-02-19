// Get inspiration from
// https://github.com/pytorch/pytorch/blob/65aa16f968af2cd18ff8c25cc657e7abda594bfc/aten/src/ATen/native/cuda/Nonzero.cu
#include <assert.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <stdio.h>

#if __CUDACC_VER_MAJOR__ >= 13
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#endif

#if __CUDACC_VER_MAJOR__ >= 13
#define USE_THRUST_INPUT_ITERS 1
#else
#define USE_THRUST_INPUT_ITERS 0
#endif

#if USE_THRUST_INPUT_ITERS
#define DECLARE_NONZERO_ITER(NAME, TYPE, PTR)                                  \
  auto NAME = thrust::make_transform_iterator(PTR, NonZeroOp<TYPE>());
#define DECLARE_COUNTING_ITER(NAME, START)                                     \
  auto NAME = thrust::make_counting_iterator<uint32_t>(START);
#else
#define DECLARE_NONZERO_ITER(NAME, TYPE, PTR)                                  \
  cub::TransformInputIterator<bool, NonZeroOp<TYPE>, const TYPE *> NAME(       \
      PTR, NonZeroOp<TYPE>());
#define DECLARE_COUNTING_ITER(NAME, START)                                     \
  cub::CountingInputIterator<uint32_t> NAME(START);
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int next_power_of_2(const uint32_t num_nonzero) {
  int result = 1;
  while (result < num_nonzero) {
    result <<= 1;
  }
  return result;
}

template <typename T> struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const T &a) const {
    return (a != T(0));
  }
};

// count the number of non-zero elements in an array, to better allocate memory
template <typename T>
void count_nonzero(const T *d_in, const uint32_t N, uint32_t *h_out,
                   cudaStream_t stream) {
  DECLARE_NONZERO_ITER(itr, T, d_in);
  size_t temp_storage_bytes = 0;
  uint32_t *d_num_nonzero;
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_num_nonzero, sizeof(uint32_t), stream));
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr,
                                    d_num_nonzero, N, stream));
  void **d_temp_storage;
  CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, itr,
                                    d_num_nonzero, N, stream));
  CUDA_CHECK(cudaMemcpyAsync(h_out, d_num_nonzero, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaFreeAsync(d_num_nonzero, stream));
  CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
}

#define COUNT_NONZERO_OP(TYPENAME, RUST_NAME)                                  \
  extern "C" uint32_t count_nonzero_##RUST_NAME(                               \
      const TYPENAME *d_in, uint32_t N, cudaStream_t stream) {                 \
    uint32_t result;                                                           \
    count_nonzero(d_in, N, &result, stream);                                   \
    return result;                                                             \
  }
#define COUNT_NONZERO_OP_DUMMY(RUST_NAME)                                      \
  extern "C" uint32_t count_nonzero_##RUST_NAME(                               \
      const uint16_t *d_in, uint32_t N, cudaStream_t stream) {                 \
    return 0;                                                                  \
  }

#if __CUDA_ARCH__ >= 800
COUNT_NONZERO_OP(__nv_bfloat16, bf16)
#else
COUNT_NONZERO_OP_DUMMY(bf16)
#endif

#if __CUDA_ARCH__ >= 530
COUNT_NONZERO_OP(__half, f16)
#else
COUNT_NONZERO_OP_DUMMY(f16)
#endif

COUNT_NONZERO_OP(float, f32)
COUNT_NONZERO_OP(double, f64)
COUNT_NONZERO_OP(uint8_t, u8)
COUNT_NONZERO_OP(uint32_t, u32)
COUNT_NONZERO_OP(int64_t, i64)
COUNT_NONZERO_OP(int32_t, i32)
COUNT_NONZERO_OP(int16_t, i16)

__global__ void transform_indices(const uint32_t *temp_indices,
                                  const uint32_t num_nonzero,
                                  const uint32_t *dims, const uint32_t num_dims,
                                  uint32_t *d_out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_nonzero) {
    int temp_index = temp_indices[idx];
    for (int i = num_dims - 1; i >= 0; i--) {
      d_out[idx * num_dims + i] = temp_index % dims[i];
      if (dims[i] == 0) {
        // Handle the error appropriately
      } else {
        temp_index /= dims[i];
      }
    }
  }
}

// get the indices of non-zero elements in an array
template <typename T>
void nonzero(const T *d_in, const uint32_t N, const uint32_t num_nonzero,
             const uint32_t *dims, const uint32_t num_dims, uint32_t *d_out,
             cudaStream_t stream) {
  DECLARE_NONZERO_ITER(itr, T, d_in);
  DECLARE_COUNTING_ITER(counting_itr, 0);
  uint32_t *out_temp;
  uint32_t *num_selected_out;
  CUDA_CHECK(cudaMallocAsync((void **)&out_temp, num_nonzero * sizeof(uint32_t),
                             stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&num_selected_out, sizeof(uint32_t), stream));
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes,
                                        counting_itr, itr, out_temp,
                                        num_selected_out, N, stream));
  void **d_temp_storage;
  CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
  CUDA_CHECK(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
                                        counting_itr, itr, out_temp,
                                        num_selected_out, (int)N, stream));
  int nthreads = next_power_of_2(num_nonzero);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (num_nonzero + nthreads - 1) / nthreads;
  transform_indices<<<nblocks, nthreads, 0, stream>>>(out_temp, num_nonzero,
                                                      dims, num_dims, d_out);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFreeAsync(out_temp, stream));
  CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
  CUDA_CHECK(cudaFreeAsync(num_selected_out, stream));
}

#define NONZERO_OP(TYPENAME, RUST_NAME)                                        \
  extern "C" void nonzero_##RUST_NAME(const TYPENAME *d_in, uint32_t N,        \
                                      uint32_t num_nonzero,                    \
                                      const uint32_t *dims, uint32_t num_dims, \
                                      uint32_t *d_out, cudaStream_t stream) {  \
    nonzero(d_in, N, num_nonzero, dims, num_dims, d_out, stream);              \
  }

#define NONZERO_OP_DUMMY(RUST_NAME)                                            \
  extern "C" void nonzero_##RUST_NAME(const uint16_t *d_in, uint32_t N,        \
                                      uint32_t num_nonzero,                    \
                                      const uint32_t *dims, uint32_t num_dims, \
                                      uint32_t *d_out, cudaStream_t stream) {  \
    assert(false);                                                             \
  }

#if __CUDA_ARCH__ >= 800
NONZERO_OP(__nv_bfloat16, bf16)
#else
NONZERO_OP_DUMMY(bf16)
#endif

#if __CUDA_ARCH__ >= 530
NONZERO_OP(__half, f16)
#else
NONZERO_OP_DUMMY(f16)
#endif

NONZERO_OP(float, f32)
NONZERO_OP(double, f64)
NONZERO_OP(uint8_t, u8)
NONZERO_OP(uint32_t, u32)
NONZERO_OP(int64_t, i64)
NONZERO_OP(int32_t, i32)
NONZERO_OP(int16_t, i16)

template <typename T>
__global__ void bitwise_and__kernel(const T *d_in1, const T *d_in2, T *d_out,
                                    const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] & d_in2[idx];
  }
}

template <typename T>
__global__ void bitwise_or__kernel(const T *d_in1, const T *d_in2, T *d_out,
                                   const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] | d_in2[idx];
  }
}

template <typename T>
__global__ void bitwise_xor__kernel(const T *d_in1, const T *d_in2, T *d_out,
                                    const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] ^ d_in2[idx];
  }
}

template <typename T>
void bitwise_and(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_and__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void bitwise_or(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_or__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void bitwise_xor(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_xor__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  CUDA_CHECK(cudaGetLastError());
}

#define BITWISE_OP(TYPENAME, RUST_NAME)                                        \
  extern "C" void bitwise_and_##RUST_NAME(const TYPENAME *d_in1,               \
                                          const TYPENAME *d_in2,               \
                                          TYPENAME *d_out, uint32_t N) {       \
    bitwise_and(d_in1, d_in2, d_out, N);                                       \
  }                                                                            \
  extern "C" void bitwise_or_##RUST_NAME(const TYPENAME *d_in1,                \
                                         const TYPENAME *d_in2,                \
                                         TYPENAME *d_out, uint32_t N) {        \
    bitwise_or(d_in1, d_in2, d_out, N);                                        \
  }                                                                            \
  extern "C" void bitwise_xor_##RUST_NAME(const TYPENAME *d_in1,               \
                                          const TYPENAME *d_in2,               \
                                          TYPENAME *d_out, uint32_t N) {       \
    bitwise_xor(d_in1, d_in2, d_out, N);                                       \
  }

BITWISE_OP(uint8_t, u8)
BITWISE_OP(uint32_t, u32)
BITWISE_OP(int64_t, i64)
BITWISE_OP(int32_t, i32)

template <typename T>
__global__ void leftshift_kernel(const T *d_in1, T *d_out, const uint32_t N,
                                 const int32_t k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] << k;
  }
}

template <typename T>
void leftshift(const T *d_in1, T *d_out, int N, const int32_t k) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  leftshift_kernel<<<nblocks, nthreads>>>(d_in1, d_out, N, k);
  CUDA_CHECK(cudaGetLastError());
}

#define LEFTSHIFT_OP(TYPENAME, RUST_NAME)                                      \
  extern "C" void leftshift_##RUST_NAME(                                       \
      const TYPENAME *d_in1, TYPENAME *d_out, uint32_t N, int32_t k) {         \
    leftshift(d_in1, d_out, N, k);                                             \
  }

LEFTSHIFT_OP(uint8_t, u8)
LEFTSHIFT_OP(int32_t, i32)
LEFTSHIFT_OP(uint32_t, u32)
LEFTSHIFT_OP(int64_t, i64)

// ============================================================================
// Fused GPT-OSS SwiGLU kernel
// Fuses 7 operations into 1: clamp, multiply, sigmoid, etc.
//
// Formula:
//   gate_clamped = min(gate, limit)
//   up_clamped = clamp(up, -limit, limit)
//   glu = gate_clamped * sigmoid(gate_clamped * alpha)
//   output = (up_clamped + 1) * glu
// ============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float fast_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void gptoss_swiglu_kernel(const T *__restrict__ gate,
                                     const T *__restrict__ up,
                                     T *__restrict__ output, const uint32_t N,
                                     const float alpha, const float limit) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // Load inputs
  float g = (float)gate[idx];
  float u = (float)up[idx];

  // Clamp gate (max only) and up (both min and max)
  float gate_clamped = fminf(g, limit);
  float up_clamped = fmaxf(fminf(u, limit), -limit);

  // glu = gate_clamped * sigmoid(gate_clamped * alpha)
  float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);

  // output = (up_clamped + 1) * glu
  float result = (up_clamped + 1.0f) * glu;

  output[idx] = (T)result;
}

// Vectorized version for better memory bandwidth (4 elements at a time)
template <typename T, typename T4>
__global__ void gptoss_swiglu_kernel_vec4(const T4 *__restrict__ gate,
                                          const T4 *__restrict__ up,
                                          T4 *__restrict__ output,
                                          const uint32_t N4, const float alpha,
                                          const float limit) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N4)
    return;

  T4 g4 = gate[idx];
  T4 u4 = up[idx];

  float g0 = (float)((T *)&g4)[0];
  float g1 = (float)((T *)&g4)[1];
  float g2 = (float)((T *)&g4)[2];
  float g3 = (float)((T *)&g4)[3];

  float u0 = (float)((T *)&u4)[0];
  float u1 = (float)((T *)&u4)[1];
  float u2 = (float)((T *)&u4)[2];
  float u3 = (float)((T *)&u4)[3];

// Process 4 elements
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float g = (i == 0) ? g0 : (i == 1) ? g1 : (i == 2) ? g2 : g3;
    float u = (i == 0) ? u0 : (i == 1) ? u1 : (i == 2) ? u2 : u3;

    float gate_clamped = fminf(g, limit);
    float up_clamped = fmaxf(fminf(u, limit), -limit);
    float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);
    float result = (up_clamped + 1.0f) * glu;

    if (i == 0)
      ((T *)&g4)[0] = (T)result;
    else if (i == 1)
      ((T *)&g4)[1] = (T)result;
    else if (i == 2)
      ((T *)&g4)[2] = (T)result;
    else
      ((T *)&g4)[3] = (T)result;
  }

  output[idx] = g4;
}

extern "C" void gptoss_swiglu_f16(const __half *gate, const __half *up,
                                  __half *output, uint32_t N, float alpha,
                                  float limit, cudaStream_t stream) {
  // Use vectorized kernel when N is divisible by 4
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel_vec4<__half, uint64_t>
        <<<nblocks, nthreads, 0, stream>>>(
            (const uint64_t *)gate, (const uint64_t *)up, (uint64_t *)output,
            N4, alpha, limit);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                           alpha, limit);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_bf16(const __nv_bfloat16 *gate,
                                   const __nv_bfloat16 *up,
                                   __nv_bfloat16 *output, uint32_t N,
                                   float alpha, float limit,
                                   cudaStream_t stream) {
  // Use vectorized kernel when N is divisible by 4
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel_vec4<__nv_bfloat16, uint64_t>
        <<<nblocks, nthreads, 0, stream>>>(
            (const uint64_t *)gate, (const uint64_t *)up, (uint64_t *)output,
            N4, alpha, limit);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                           alpha, limit);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_f32(const float *gate, const float *up,
                                  float *output, uint32_t N, float alpha,
                                  float limit, cudaStream_t stream) {
  // Use vectorized kernel when N is divisible by 4
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel_vec4<float, float4><<<nblocks, nthreads, 0, stream>>>(
        (const float4 *)gate, (const float4 *)up, (float4 *)output, N4, alpha,
        limit);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(gate, up, output, N,
                                                           alpha, limit);
  }
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused GPT-OSS SwiGLU kernel for INTERLEAVED gate/up data
//
// This kernel handles interleaved gate/up format: [..., intermediate_size, 2]
// where gate = data[..., :, 0] and up = data[..., :, 1]
//
// Avoids 2 tensor copies from narrow().squeeze().contiguous()
// ============================================================================

template <typename T>
__global__ void gptoss_swiglu_interleaved_kernel(
    const T *__restrict__ gate_up, // [N, intermediate_size, 2] interleaved
    T *__restrict__ output,        // [N, intermediate_size]
    const uint32_t N,              // num_tokens * topk
    const uint32_t intermediate_size, const float alpha, const float limit) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_elements = N * intermediate_size;
  if (idx >= total_elements)
    return;

  // Decode position
  const int n = idx / intermediate_size;
  const int i = idx % intermediate_size;

  // Read interleaved values: gate at offset 0, up at offset 1
  const int base_idx = (n * intermediate_size + i) * 2;
  float g = (float)gate_up[base_idx];     // gate
  float u = (float)gate_up[base_idx + 1]; // up

  // Clamp gate (max only) and up (both min and max)
  float gate_clamped = fminf(g, limit);
  float up_clamped = fmaxf(fminf(u, limit), -limit);

  // glu = gate_clamped * sigmoid(gate_clamped * alpha)
  float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);

  // output = (up_clamped + 1) * glu
  float result = (up_clamped + 1.0f) * glu;

  output[idx] = (T)result;
}

extern "C" void gptoss_swiglu_interleaved_f16(const __half *gate_up,
                                              __half *output, uint32_t N,
                                              uint32_t intermediate_size,
                                              float alpha, float limit,
                                              cudaStream_t stream) {
  const uint32_t total = N * intermediate_size;
  const int nthreads = 256;
  const int nblocks = (total + nthreads - 1) / nthreads;
  gptoss_swiglu_interleaved_kernel<<<nblocks, nthreads, 0, stream>>>(
      gate_up, output, N, intermediate_size, alpha, limit);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_interleaved_bf16(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *output, uint32_t N,
    uint32_t intermediate_size, float alpha, float limit, cudaStream_t stream) {
  const uint32_t total = N * intermediate_size;
  const int nthreads = 256;
  const int nblocks = (total + nthreads - 1) / nthreads;
  gptoss_swiglu_interleaved_kernel<<<nblocks, nthreads, 0, stream>>>(
      gate_up, output, N, intermediate_size, alpha, limit);
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_interleaved_f32(const float *gate_up,
                                              float *output, uint32_t N,
                                              uint32_t intermediate_size,
                                              float alpha, float limit,
                                              cudaStream_t stream) {
  const uint32_t total = N * intermediate_size;
  const int nthreads = 256;
  const int nblocks = (total + nthreads - 1) / nthreads;
  gptoss_swiglu_interleaved_kernel<<<nblocks, nthreads, 0, stream>>>(
      gate_up, output, N, intermediate_size, alpha, limit);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused Softmax with Sinks kernel for GPT-OSS attention
//
// This kernel computes softmax over attention logits while including a per-head
// "sink" value in the normalization, then drops the sink from the output.
//
// This avoids:
// 1. Tensor concatenation to add sinks
// 2. Broadcast operations for max subtraction
// 3. narrow().contiguous() copy to drop sinks
//
// Input:
//   logits: [batch, heads, q_len, k_len] - attention scores after q@k.T * scale
//   sinks: [heads] - per-head sink values
//   mask: [batch, 1, q_len, k_len] or nullptr - attention mask (0 = attend,
//   -inf = mask)
//
// Output:
//   scores: [batch, heads, q_len, k_len] - softmax probabilities (sink dropped)
// ============================================================================

// Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Fused softmax with sinks - one block per (batch, head, query) position
// Each block processes one row of the attention matrix
template <typename T>
__global__ void softmax_with_sinks_kernel(
    const T *__restrict__ logits, // [batch, heads, q_len, k_len]
    const T *__restrict__ sinks,  // [heads]
    const T *__restrict__ mask,   // [batch, 1, q_len, k_len] or nullptr
    T *__restrict__ output,       // [batch, heads, q_len, k_len]
    const int batch_size, const int num_heads, const int q_len, const int k_len,
    const float scale // softmax scale (usually 1.0, already applied)
) {
  // Each block handles one (batch, head, query) = one row of length k_len
  const int row_idx = blockIdx.x;
  const int total_rows = batch_size * num_heads * q_len;
  if (row_idx >= total_rows)
    return;

  // Decode indices
  const int b = row_idx / (num_heads * q_len);
  const int h = (row_idx / q_len) % num_heads;
  const int q = row_idx % q_len;

  // Pointers to this row
  const int logits_offset = ((b * num_heads + h) * q_len + q) * k_len;
  const T *row_logits = logits + logits_offset;
  T *row_output = output + logits_offset;

  // Get sink value for this head
  const float sink_val = (float)sinks[h];

  // Mask offset (mask has shape [batch, 1, q_len, k_len])
  const int mask_offset = (b * q_len + q) * k_len;
  const T *row_mask = mask ? (mask + mask_offset) : nullptr;

  // Shared memory for reductions
  __shared__ float s_max;
  __shared__ float s_sum;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Step 1: Find max (including sink)
  float local_max = -INFINITY;
  for (int k = tid; k < k_len; k += block_size) {
    float val = (float)row_logits[k];
    if (row_mask) {
      val += (float)row_mask[k]; // mask is 0 or -inf
    }
    local_max = fmaxf(local_max, val);
  }
  // Include sink in max computation
  if (tid == 0) {
    local_max = fmaxf(local_max, sink_val);
  }

  // Warp reduction for max
  local_max = warp_reduce_max(local_max);

  // Block reduction for max (if more than 1 warp)
  if (block_size > 32) {
    __shared__ float warp_maxes[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
      warp_maxes[warp_id] = local_max;
    }
    __syncthreads();
    if (tid < 32) {
      local_max = (tid < (block_size + 31) / 32) ? warp_maxes[tid] : -INFINITY;
      local_max = warp_reduce_max(local_max);
    }
  }
  if (tid == 0) {
    s_max = local_max;
  }
  __syncthreads();
  const float row_max = s_max;

  // Step 2: Compute exp(x - max) and sum (including sink)
  float local_sum = 0.0f;
  for (int k = tid; k < k_len; k += block_size) {
    float val = (float)row_logits[k];
    if (row_mask) {
      val += (float)row_mask[k];
    }
    local_sum += expf(val - row_max);
  }
  // Include sink in sum computation
  if (tid == 0) {
    local_sum += expf(sink_val - row_max);
  }

  // Warp reduction for sum
  local_sum = warp_reduce_sum(local_sum);

  // Block reduction for sum
  if (block_size > 32) {
    __shared__ float warp_sums[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
      warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    if (tid < 32) {
      local_sum = (tid < (block_size + 31) / 32) ? warp_sums[tid] : 0.0f;
      local_sum = warp_reduce_sum(local_sum);
    }
  }
  if (tid == 0) {
    s_sum = local_sum;
  }
  __syncthreads();
  const float row_sum = s_sum;

  // Step 3: Write normalized outputs (sink is NOT written - it's dropped)
  const float inv_sum = 1.0f / row_sum;
  for (int k = tid; k < k_len; k += block_size) {
    float val = (float)row_logits[k];
    if (row_mask) {
      val += (float)row_mask[k];
    }
    row_output[k] = (T)(expf(val - row_max) * inv_sum);
  }
}

// Launch wrapper for f16
extern "C" void softmax_with_sinks_f16(const __half *logits,
                                       const __half *sinks, const __half *mask,
                                       __half *output, int batch_size,
                                       int num_heads, int q_len, int k_len,
                                       float scale, cudaStream_t stream) {
  const int total_rows = batch_size * num_heads * q_len;
  // Choose block size based on k_len
  int block_size = 256;
  if (k_len <= 64)
    block_size = 64;
  else if (k_len <= 128)
    block_size = 128;
  else if (k_len <= 256)
    block_size = 256;
  else if (k_len <= 512)
    block_size = 512;
  else
    block_size = 1024;

  softmax_with_sinks_kernel<<<total_rows, block_size, 0, stream>>>(
      logits, sinks, mask, output, batch_size, num_heads, q_len, k_len, scale);
  CUDA_CHECK(cudaGetLastError());
}

// Launch wrapper for bf16
extern "C" void softmax_with_sinks_bf16(const __nv_bfloat16 *logits,
                                        const __nv_bfloat16 *sinks,
                                        const __nv_bfloat16 *mask,
                                        __nv_bfloat16 *output, int batch_size,
                                        int num_heads, int q_len, int k_len,
                                        float scale, cudaStream_t stream) {
  const int total_rows = batch_size * num_heads * q_len;
  int block_size = 256;
  if (k_len <= 64)
    block_size = 64;
  else if (k_len <= 128)
    block_size = 128;
  else if (k_len <= 256)
    block_size = 256;
  else if (k_len <= 512)
    block_size = 512;
  else
    block_size = 1024;

  softmax_with_sinks_kernel<<<total_rows, block_size, 0, stream>>>(
      logits, sinks, mask, output, batch_size, num_heads, q_len, k_len, scale);
  CUDA_CHECK(cudaGetLastError());
}

// Launch wrapper for f32
extern "C" void softmax_with_sinks_f32(const float *logits, const float *sinks,
                                       const float *mask, float *output,
                                       int batch_size, int num_heads, int q_len,
                                       int k_len, float scale,
                                       cudaStream_t stream) {
  const int total_rows = batch_size * num_heads * q_len;
  int block_size = 256;
  if (k_len <= 64)
    block_size = 64;
  else if (k_len <= 128)
    block_size = 128;
  else if (k_len <= 256)
    block_size = 256;
  else if (k_len <= 512)
    block_size = 512;
  else
    block_size = 1024;

  softmax_with_sinks_kernel<<<total_rows, block_size, 0, stream>>>(
      logits, sinks, mask, output, batch_size, num_heads, q_len, k_len, scale);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Fused GLU (Gated Linear Unit) kernel: output = activation(a) * b
// Supports SiLU, GELU (approximate), and ReLU activations
//
// This fuses the activation function and element-wise multiplication into
// a single kernel pass, eliminating intermediate tensor allocation.
// ============================================================================

// Activation type enum - must match Rust GluActivationType
enum GluActivation {
  GLU_SILU = 0,
  GLU_GELU = 1,
  GLU_RELU = 2,
  GLU_GELU_ERF = 3
};

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float glu_silu(float x) {
  return x / (1.0f + expf(-x));
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float glu_gelu(float x) {
  const float kSqrt2OverPi = 0.7978845608f;
  const float kCoeff = 0.044715f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// ReLU activation: max(0, x)
__device__ __forceinline__ float glu_relu(float x) { return fmaxf(x, 0.0f); }

// GELU (exact ERF version): x * normcdf(x), matching candle's CUDA impl
__device__ __forceinline__ float glu_gelu_erf(float x) {
  return x * normcdff(x);
}

__device__ __forceinline__ float apply_glu_activation(float x, int act) {
  switch (act) {
  case GLU_SILU:
    return glu_silu(x);
  case GLU_GELU:
    return glu_gelu(x);
  case GLU_RELU:
    return glu_relu(x);
  case GLU_GELU_ERF:
    return glu_gelu_erf(x);
  default:
    return glu_silu(x);
  }
}

// Scalar kernel for general case
template <typename T>
__global__ void fused_glu_kernel(const T *__restrict__ a, // input to activation
                                 const T *__restrict__ b, // multiplier
                                 T *__restrict__ output, const uint32_t N,
                                 const int activation) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float a_val = (float)a[idx];
  // Cast activation back to T before multiplying, matching candle's
  // two-step behavior: unary op in float32 -> cast to T -> binary mul in T
  T activated = (T)apply_glu_activation(a_val, activation);
  output[idx] = activated * b[idx];
}

// Vectorized version for 4 elements at a time
template <typename T, typename T4>
__global__ void fused_glu_kernel_vec4(const T4 *__restrict__ a,
                                      const T4 *__restrict__ b,
                                      T4 *__restrict__ output,
                                      const uint32_t N4, const int activation) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N4)
    return;

  T4 a4 = a[idx];
  T4 b4 = b[idx];
  T4 out4;

  float a0 = (float)((T *)&a4)[0];
  float a1 = (float)((T *)&a4)[1];
  float a2 = (float)((T *)&a4)[2];
  float a3 = (float)((T *)&a4)[3];

  // Cast activation back to T before multiplying, matching candle's
  // two-step behavior: unary op in float32 -> cast to T -> binary mul in T
  T act0 = (T)apply_glu_activation(a0, activation);
  T act1 = (T)apply_glu_activation(a1, activation);
  T act2 = (T)apply_glu_activation(a2, activation);
  T act3 = (T)apply_glu_activation(a3, activation);

  ((T *)&out4)[0] = act0 * ((T *)&b4)[0];
  ((T *)&out4)[1] = act1 * ((T *)&b4)[1];
  ((T *)&out4)[2] = act2 * ((T *)&b4)[2];
  ((T *)&out4)[3] = act3 * ((T *)&b4)[3];

  output[idx] = out4;
}

extern "C" void fused_glu_f16(const __half *a, const __half *b, __half *output,
                              uint32_t N, int activation, cudaStream_t stream) {
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    fused_glu_kernel_vec4<__half, uint64_t><<<nblocks, nthreads, 0, stream>>>(
        (const uint64_t *)a, (const uint64_t *)b, (uint64_t *)output, N4,
        activation);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    fused_glu_kernel<<<nblocks, nthreads, 0, stream>>>(a, b, output, N,
                                                       activation);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void fused_glu_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                               __nv_bfloat16 *output, uint32_t N,
                               int activation, cudaStream_t stream) {
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    fused_glu_kernel_vec4<__nv_bfloat16, uint64_t>
        <<<nblocks, nthreads, 0, stream>>>((const uint64_t *)a,
                                           (const uint64_t *)b,
                                           (uint64_t *)output, N4, activation);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    fused_glu_kernel<<<nblocks, nthreads, 0, stream>>>(a, b, output, N,
                                                       activation);
  }
  CUDA_CHECK(cudaGetLastError());
}

extern "C" void fused_glu_f32(const float *a, const float *b, float *output,
                              uint32_t N, int activation, cudaStream_t stream) {
  if (N % 4 == 0) {
    const int N4 = N / 4;
    const int nthreads = 256;
    const int nblocks = (N4 + nthreads - 1) / nthreads;
    fused_glu_kernel_vec4<float, float4><<<nblocks, nthreads, 0, stream>>>(
        (const float4 *)a, (const float4 *)b, (float4 *)output, N4, activation);
  } else {
    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;
    fused_glu_kernel<<<nblocks, nthreads, 0, stream>>>(a, b, output, N,
                                                       activation);
  }
  CUDA_CHECK(cudaGetLastError());
}
