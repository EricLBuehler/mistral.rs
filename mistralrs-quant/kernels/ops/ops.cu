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

#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
__global__ void gptoss_swiglu_kernel(
    const T *__restrict__ gate,
    const T *__restrict__ up,
    T *__restrict__ output,
    const uint32_t N,
    const float alpha,
    const float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

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
__global__ void gptoss_swiglu_kernel_vec4(
    const T4 *__restrict__ gate,
    const T4 *__restrict__ up,
    T4 *__restrict__ output,
    const uint32_t N4,
    const float alpha,
    const float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N4) return;

    T4 g4 = gate[idx];
    T4 u4 = up[idx];

    float g0 = (float)((T*)&g4)[0];
    float g1 = (float)((T*)&g4)[1];
    float g2 = (float)((T*)&g4)[2];
    float g3 = (float)((T*)&g4)[3];

    float u0 = (float)((T*)&u4)[0];
    float u1 = (float)((T*)&u4)[1];
    float u2 = (float)((T*)&u4)[2];
    float u3 = (float)((T*)&u4)[3];

    // Process 4 elements
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float g = (i == 0) ? g0 : (i == 1) ? g1 : (i == 2) ? g2 : g3;
        float u = (i == 0) ? u0 : (i == 1) ? u1 : (i == 2) ? u2 : u3;

        float gate_clamped = fminf(g, limit);
        float up_clamped = fmaxf(fminf(u, limit), -limit);
        float glu = gate_clamped * fast_sigmoid(gate_clamped * alpha);
        float result = (up_clamped + 1.0f) * glu;

        if (i == 0) ((T*)&g4)[0] = (T)result;
        else if (i == 1) ((T*)&g4)[1] = (T)result;
        else if (i == 2) ((T*)&g4)[2] = (T)result;
        else ((T*)&g4)[3] = (T)result;
    }

    output[idx] = g4;
}

extern "C" void gptoss_swiglu_f16(
    const __half *gate,
    const __half *up,
    __half *output,
    uint32_t N,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    // Use vectorized kernel when N is divisible by 4
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<__half, uint64_t><<<nblocks, nthreads, 0, stream>>>(
            (const uint64_t*)gate, (const uint64_t*)up, (uint64_t*)output,
            N4, alpha, limit
        );
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(
            gate, up, output, N, alpha, limit
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_bf16(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    __nv_bfloat16 *output,
    uint32_t N,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    // Use vectorized kernel when N is divisible by 4
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<__nv_bfloat16, uint64_t><<<nblocks, nthreads, 0, stream>>>(
            (const uint64_t*)gate, (const uint64_t*)up, (uint64_t*)output,
            N4, alpha, limit
        );
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(
            gate, up, output, N, alpha, limit
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gptoss_swiglu_f32(
    const float *gate,
    const float *up,
    float *output,
    uint32_t N,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    // Use vectorized kernel when N is divisible by 4
    if (N % 4 == 0) {
        const int N4 = N / 4;
        const int nthreads = 256;
        const int nblocks = (N4 + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel_vec4<float, float4><<<nblocks, nthreads, 0, stream>>>(
            (const float4*)gate, (const float4*)up, (float4*)output,
            N4, alpha, limit
        );
    } else {
        const int nthreads = 256;
        const int nblocks = (N + nthreads - 1) / nthreads;
        gptoss_swiglu_kernel<<<nblocks, nthreads, 0, stream>>>(
            gate, up, output, N, alpha, limit
        );
    }
    CUDA_CHECK(cudaGetLastError());
}
