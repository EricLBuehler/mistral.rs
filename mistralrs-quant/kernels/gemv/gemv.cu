// Vectorized GEMV row reduction follows llama.cpp mmvf.cu.
#include "row.cuh"

template <typename T, typename Vec2, int BLOCK_SIZE, int BATCH_SIZE>
__global__ void
gemv_kernel_batched(const T *__restrict__ A, const T *__restrict__ X,
                    const T *__restrict__ bias, T *__restrict__ Y, int M, int K,
                    bool has_bias) {
  const mistralrs_gemv::RowArgs<T> args{A, X, bias, Y, M, K, has_bias};
  mistralrs_gemv::row<T, Vec2, BLOCK_SIZE, BATCH_SIZE>(args, blockIdx.x,
                                                       threadIdx.x);
}

__host__ int get_optimal_block_size(int K) {
  const int K2 = K / 2;
  if (K2 <= 32)
    return 32;
  if (K2 <= 64)
    return 64;
  if (K2 <= 128)
    return 128;
  return 256;
}

#define LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, BATCH_SIZE)                   \
  gemv_kernel_batched<T, Vec2, BLOCK_SIZE, BATCH_SIZE>                         \
      <<<grid, dim3(BLOCK_SIZE), 0, stream>>>(A, X, bias, Y, M, K, has_bias)

#define DISPATCH_BATCH_SIZE(T, Vec2, BLOCK_SIZE)                               \
  switch (batch_size) {                                                        \
  case 1:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 1);                               \
    break;                                                                     \
  case 2:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 2);                               \
    break;                                                                     \
  case 3:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 3);                               \
    break;                                                                     \
  case 4:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 4);                               \
    break;                                                                     \
  case 5:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 5);                               \
    break;                                                                     \
  case 6:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 6);                               \
    break;                                                                     \
  case 7:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 7);                               \
    break;                                                                     \
  case 8:                                                                      \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 8);                               \
    break;                                                                     \
  default:                                                                     \
    LAUNCH_GEMV_BATCHED(T, Vec2, BLOCK_SIZE, 1);                               \
    break;                                                                     \
  }

#define DISPATCH_BLOCK_SIZE(T, Vec2)                                           \
  switch (block_size) {                                                        \
  case 32:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 32);                                          \
    break;                                                                     \
  case 64:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 64);                                          \
    break;                                                                     \
  case 128:                                                                    \
    DISPATCH_BATCH_SIZE(T, Vec2, 128);                                         \
    break;                                                                     \
  case 256:                                                                    \
    DISPATCH_BATCH_SIZE(T, Vec2, 256);                                         \
    break;                                                                     \
  default:                                                                     \
    DISPATCH_BATCH_SIZE(T, Vec2, 256);                                         \
    break;                                                                     \
  }

extern "C" void launch_gemv_bf16(const __nv_bfloat16 *A, const __nv_bfloat16 *X,
                                 const __nv_bfloat16 *bias, __nv_bfloat16 *Y,
                                 int M, int K, int batch_size, bool has_bias,
                                 cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat162);
}

extern "C" void launch_gemv_f16(const __half *A, const __half *X,
                                const __half *bias, __half *Y, int M, int K,
                                int batch_size, bool has_bias,
                                cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(__half, half2);
}

extern "C" void launch_gemv_f32(const float *A, const float *X,
                                const float *bias, float *Y, int M, int K,
                                int batch_size, bool has_bias,
                                cudaStream_t stream) {
  int block_size = get_optimal_block_size(K);
  dim3 grid(M);

  DISPATCH_BLOCK_SIZE(float, float2);
}
