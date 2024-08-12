#include <stdint.h>

int next_power_of_2(const uint32_t num_nonzero) {
  int result = 1;
  while (result < num_nonzero) {
    result <<= 1;
  }
  return result;
}

template <typename T>
__global__ void bitwise_or_kernel(const T *d_in1, const T *d_in2, T *d_out,
                                   const uint32_t N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in1[idx] | d_in2[idx];
  }
}

template <typename T>
void bitwise_or(const T *d_in1, const T *d_in2, T *d_out, int N) {
  int nthreads = next_power_of_2(N);
  if (nthreads > 1024) {
    nthreads = 1024;
  }
  const int nblocks = (N + nthreads - 1) / nthreads;
  bitwise_or_kernel<<<nblocks, nthreads>>>(d_in1, d_in2, d_out, N);
  cudaDeviceSynchronize();
}

#define BITWISE_OR_OP(TYPENAME, RUST_NAME)                                     \
  extern "C" void bitwise_or_##RUST_NAME(const TYPENAME *d_in1,                \
                                         const TYPENAME *d_in2,                \
                                         TYPENAME *d_out, uint32_t N) {        \
    bitwise_or(d_in1, d_in2, d_out, N);                                        \
  }                             

BITWISE_OR_OP(uint8_t, u8)
BITWISE_OR_OP(int32_t, i32)


template <typename T>
__global__ void leftshift_kernel(const T *d_in1, T *d_out,
                                   const uint32_t N, const int32_t k) {
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
  cudaDeviceSynchronize();
}

#define LEFTSHIFT_OP(TYPENAME, RUST_NAME)                                                 \
  extern "C" void leftshift_##RUST_NAME(const TYPENAME *d_in1,                            \
                                         TYPENAME *d_out, uint32_t N, int32_t k) {        \
    leftshift(d_in1, d_out, N, k);                                                        \
  }

LEFTSHIFT_OP(uint8_t, u8)
LEFTSHIFT_OP(int32_t, i32)
