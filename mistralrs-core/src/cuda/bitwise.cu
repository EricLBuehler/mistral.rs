
#include <cstdio>
#include <cub/cub.cuh>
#include <stdint.h>

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

int next_power_of_2(const uint32_t num_nonzero) {
  int result = 1;
  while (result < num_nonzero) {
    result <<= 1;
  }
  return result;
}

template <typename T>
void bitwise_and(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_and__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  cudaDeviceSynchronize();
}

template <typename T>
void bitwise_or(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_or__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  cudaDeviceSynchronize();
}

template <typename T>
void bitwise_xor(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_xor__kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  cudaDeviceSynchronize();
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
