/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MMA_CUH_
#define FLASHINFER_MMA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace flashinfer {

namespace mma {

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120400)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890))
#define FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED
#endif
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_STMATRIX_M8N8X4_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED
#endif
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 750))
#define FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED
#define FLASHINFER_LDMATRIX_M8N8X4_ENABLED
#endif
#endif

#if defined(__CUDA_ARCH__)
#define FLASHINFER_RUNTIME_ASSERT(x) __brkpt()
#else
#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)
#endif

enum class MMAMode {
  kInit = 0U,
  kInplaceUpdate = 1U,
};

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, _, %1, _}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 instruction, loads data from shared memory
 *   to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {_, %0, _, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {%0, %1, _, _}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX ldmatrix m8n8.x4 transposed instruction, loads data from
 *   shared memory to fragment and transposes the fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {_, _, %0, %1}, [%2];\n"
               : "=r"(R[0]), "=r"(R[1])
               : "r"(smem_int_ptr));
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for ldmatrix instruction");
#endif
}

/*!
 * \brief Wrapper of PTX stmatrix m8n8.x4 instruction, stores data from fragment
 *   to shared memory
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename T>
__device__ __forceinline__ void stmatrix_m8n8x4(uint32_t* R, T* smem_ptr) {
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(smem_int_ptr), "r"(R[0]), "r"(R[1]), "r"(R[2]), "r"(R[3]));
#else
  // Fallback implementation, slower than PTX instruction
  const uint32_t tx = threadIdx.x;
  uint4 word;
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
    word.x = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4);
    word.y = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 1);
    word.z = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 2);
    word.w = __shfl_sync(0xffffffff, R[reg_id], (tx % 8) * 4 + 3);
    if (tx / 8 == reg_id) {
      *(uint4*)smem_ptr = word;
    }
  }
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k32 instructions for row major and column major f8 matrix
 *   multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(float* C, uint32_t* A,
                                                                   uint32_t* B) {
  static_assert(sizeof(T) == 1, "DType must be 8bit floating data type");
#if defined(FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else {  // e5m2
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {  // e5m2
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
#else
  FLASHINFER_RUNTIME_ASSERT(
      "fp8 mma instruction is only available for sm89, PTX 8.4+ and CUDA 12.4+");
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k16 instructions for row major and column major f16 matrix
 *   multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(float* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(0.f), "f"(0.f),
            "f"(0.f), "f"(0.f));
    }
  } else {
    if constexpr (std::is_same_v<T, half>) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "f"(C[4]), "f"(C[5]),
            "f"(C[6]), "f"(C[7]));
    }
  }
#elif defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  if constexpr (std::is_same_v<T, half>) {
    if constexpr (mma_mode == MMAMode::kInit) {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
          : "r"(A[2]), "r"(A[3]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[2]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7, %8, %9, %10};\n"
          : "=f"(C[4]), "=f"(C[5]), "=f"(C[6]), "=f"(C[7])
          : "r"(A[2]), "r"(A[3]), "r"(B[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
template <typename DType>
__device__ __forceinline__ void m16k32_rowsum_f8f8f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 1, "DType must be 8bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
#if defined(FLASHINFER_MMA_F8F8F32_M16N8K32_ENABLED)
  if constexpr (std::is_same_v<DType, __nv_fp8_e4m3>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(943208504),
          "r"(943208504), "f"(d[0]), "f"(d[1]));
  } else {  // e5m2
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e5m2.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1010580540),
          "r"(1010580540), "f"(d[0]), "f"(d[1]));
  }
#else
  FLASHINFER_RUNTIME_ASSERT(
      "fp8 mma instruction is only available for sm89, PTX 8.4+ and CUDA 12.4+");
#endif
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
template <typename DType>
__device__ __forceinline__ void m16k16_rowsum_f16f16f32(float* d, DType* s) {
  static_assert(sizeof(DType) == 2, "DType must be 16bit floating data type");
  uint32_t* s_u32 = (uint32_t*)(s);
#if defined(FLASHINFER_MMA_F16F16F32_M16N8K16_ENABLED)
  if constexpr (std::is_same_v<DType, half>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320),
          "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  0.,  %9,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(s_u32[2]), "r"(s_u32[3]), "r"(1065369472),
          "r"(1065369472), "f"(d[0]), "f"(d[1]));
  }
#elif defined(FLASHINFER_MMA_F16F16F32_M16N8K8_ENABLED)
  if constexpr (std::is_same_v<DType, half>) {
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  0.,  %6,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[0]), "r"(s_u32[1]), "r"(1006648320), "f"(d[0]), "f"(d[1]));
    asm volatile(
        "{\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  _,  %1,  _},"
        "{%2,  %3},"
        "{%4},"
        "{%5,  0.,  %6,  0.};\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1])
        : "r"(s_u32[2]), "r"(s_u32[3]), "r"(1006648320), "f"(d[0]), "f"(d[1]));
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

/*!
 * \brief Wrapper of two mma m16n8k16 instructions for row major and column major f16 matrix
 *   multiplication, accumulated in f16.
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(uint32_t* C, uint32_t* A,
                                                                     uint32_t* B) {
#if defined(FLASHINFER_MMA_F16F16F16_M16N8K16_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        "{%6,  %7},"
        "{%8,  %9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#elif defined(FLASHINFER_MMA_F16F16F16_M16N8K8_ENABLED)
  if constexpr (mma_mode == MMAMode::kInit) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(0), "r"(0));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(0), "r"(0));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[2]), "r"(A[3]), "r"(B[1]), "r"(C[0]), "r"(C[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[2]), "r"(C[2]), "r"(C[3]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3},"
        "{%4},"
        "{%5, %6};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[2]), "r"(A[3]), "r"(B[3]), "r"(C[2]), "r"(C[3]));
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported CUDA architecture for mma instruction");
#endif
}

}  // namespace mma

}  // namespace flashinfer

#endif  // FLASHINFER_MMA_CUH_
