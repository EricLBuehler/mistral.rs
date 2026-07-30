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
#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cuda/pipeline>

#include "cp_async.cuh"
#include "mma.cuh"

namespace flashinfer {

enum class SwizzleMode {
  k64B,
  k128B,
};

// Use 128bit as the granularity to fetch/store data per thread to maximize memory bandwidth
using b128_t = uint4;

/*!
 * \brief Compute the number of elements that can be stored in a b128_t.
 * \tparam T The data type of the elements.
 */
template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t upcast_size() {
  return sizeof(b128_t) / sizeof(T);
}

template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
  if constexpr (swizzle_mode == SwizzleMode::k128B) {
    return i * stride + (j ^ (i % 8));
  } else {
    // swizzle_mode == SwizzleMode::k64B
    return i * stride + (j ^ ((i / 2) % 4));
  }
}

/*!
 * \brief The shared memory wrapper.
 */
template <SwizzleMode swizzle_mode>
struct smem_t {
  // The base pointer.
  b128_t* base;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {}

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \tparam stride The stride (in terms of b128_t's) in the permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return i * stride + (j ^ (i % 8));
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(stride == 4);
      return i * stride + (j ^ ((i / 2) % 4));
    }
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset,
                                                                      uint32_t step_idx) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(step_size == 2, "Unsupported step size");
      return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
    }
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    } else {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_right_half(R, smem_ptr);
  }

  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::stmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_right_half(R, smem_ptr);
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr) {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(smem_ptr,
                                                           reinterpret_cast<const b128_t*>(gptr));
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_64b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b_from_64b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
