// Get inspiration from
// https://github.com/pytorch/pytorch/blob/65aa16f968af2cd18ff8c25cc657e7abda594bfc/aten/src/ATen/native/cuda/Nonzero.cu
#include <assert.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <stdio.h>

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
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
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
  cub::TransformInputIterator<bool, NonZeroOp<T>, const T *> itr(
      d_in, NonZeroOp<T>());
  cub::CountingInputIterator<uint32_t> counting_itr(0);
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
