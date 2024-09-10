/*
Adapted from https://github.com/turboderp/exllamav2
*/
#include <torch/all.h>

#include <cuda_runtime.h>

#include "q_matrix.cuh"
#include "matrix_view.cuh"
#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"
#include "q_gemm_kernel.cuh"

#define MAX_Q_GEMM_ROWS 32
#define EXL2_BLOCK_KN_SIZE 64
#define EXL2_BLOCK_M_SIZE_MAX 8
#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)
#if defined(USE_ROCM)
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transA,
    hipblasOperation_t transB, int m, int n, int k, const half* alpha,
    const half* AP, int lda, const half* BP, int ldb, const half* beta,
    half* CP, int ldc) {
  return hipblasHgemm(handle, transA, transB, m, n, k,
                      reinterpret_cast<const hipblasHalf*>(alpha),
                      reinterpret_cast<const hipblasHalf*>(AP), lda,
                      reinterpret_cast<const hipblasHalf*>(BP), ldb,
                      reinterpret_cast<const hipblasHalf*>(beta),
                      reinterpret_cast<hipblasHalf*>(CP), ldc);
}
  #define hipblasHgemm __compat_hipblasHgemm
#endif
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

void gemm_half_q_half_cuda_part(const half* a, QMatrix* b, half* c, int size_m,
                                int size_n, int size_k, int m_count,
                                bool clear) {
  {
    dim3 blockDim, gridDim;
    blockDim.x = EXL2_BLOCK_KN_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(size_n, EXL2_BLOCK_KN_SIZE * 4);
    gridDim.y = DIVIDE(size_m, m_count);
    gridDim.z = DIVIDE(b->height, EXL2_BLOCK_KN_SIZE);

    fp_gemm_half_q_half_kernel kernel = pick_gemm_half_q_half_kernel(m_count);

    kernel<<<gridDim, blockDim, 0>>>(
        a, b->cuda_q_weight, b->cuda_q_scale, b->cuda_q_scale_max, c, size_m,
        size_n, size_k, b->height, b->groups, b->cuda_q_group_map,
        b->cuda_q_perm, b->rows_8, b->rows_6, b->rows_5, b->rows_4, b->rows_3,
        b->rows_2, clear);
  }
}

void gemm_half_q_half_cuda(cublasHandle_t cublas_handle, const half* a,
                           QMatrix* b, half* c, int size_m, int size_n,
                           int size_k, bool clear, half* temp_dq) {
  if (size_m > MAX_Q_GEMM_ROWS) {
    // Reconstruct FP16 matrix, then cuBLAS
    b->reconstruct(temp_dq);

    // cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    const half alpha = __float2half(1.0f);
    const half beta = clear ? __float2half(0.0f) : __float2half(1.0f);
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
                &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
  } else {
    // Quantized matmul

    int block_m_size_max = EXL2_BLOCK_M_SIZE_MAX;
    int max_chunks = size_m / block_m_size_max;
    int last_chunk = max_chunks * block_m_size_max;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks) {
      gemm_half_q_half_cuda_part(a, b, c, last_chunk, size_n, size_k,
                                 block_m_size_max, clear);
    }

    if (last_chunk_size) {
      gemm_half_q_half_cuda_part(a + last_chunk * size_k, b,
                                 c + last_chunk * size_n, last_chunk_size,
                                 size_n, size_k, last_chunk_size, clear);
    }
  }
}