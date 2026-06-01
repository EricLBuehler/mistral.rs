/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
#define FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_

#include <cuda_runtime.h>

#include <cstdint>

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x1);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x1) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x2);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x2) == 0) ? 0x5410 : 0x3276);
  return x;
}

__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_8b_trans(uint32_t x) {
  uint32_t tmp = __shfl_xor_sync(0xffffffff, x, 0x4);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x4) == 0) ? 0x6420 : 0x3175);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x8);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x8) == 0) ? 0x5410 : 0x3276);
  tmp = __shfl_xor_sync(0xffffffff, x, 0x10);
  x = __byte_perm(x, tmp, ((threadIdx.x & 0x10) == 0) ? 0x5410 : 0x3276);
  return x;
}

// Convert 16b fragment layout to 4b fragment layout.
__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_4b(uint32_t x) {
  // Broadcast from the thread 0 of each group of 4 (thread t gets value from thread t & ~3)
  uint32_t tmp0 = __shfl_sync(0xffffffff, x, threadIdx.x & ~0x3u);
  // Similarly, broadcast from the thread 1 of each group of 4
  uint32_t tmp1 = __shfl_sync(0xffffffff, x, (threadIdx.x & ~0x3u) + 1);
  // Select byte i = (threadIdx.x % 4) of each register and assemble them together.
  uint32_t byte_idx = threadIdx.x & 0x3u;
  x = __byte_perm(tmp0, tmp1, byte_idx * 0x0101u + 0x0400u);
  return x;
}

// Convert transposed 16b fragment layout to 4b (NVfp4) fragment layout.
// Counterpart to frag_layout_swizzle_16b_to_4b for the column-major (transposed) case.
__device__ __forceinline__ uint32_t frag_layout_swizzle_16b_to_4b_trans(uint32_t x) {
  // Shuffle the data across threads. We group threads in a stride of 4: {i, i+4, i+8, i+12, ...,
  // i+28} (i in {0,1,2,3}). Thread {i, i+4, i+8, i+12} receives data from thread i and i+8. Thread
  // {i+16, i+20, i+24, i+28} receives data from thread i+4 and i+12.
  unsigned src_thrd = (threadIdx.x & ~0x1cu) + ((threadIdx.x & 0x10u) >> 2);
  uint32_t tmp0 = __shfl_sync(0xffffffff, x, src_thrd);
  uint32_t tmp1 = __shfl_sync(0xffffffff, x, src_thrd + 8u);
  // Select byte. Thread ((i / 8) % 2 == 0) selects [6,4,2,0]
  // Thread ((i / 8) % 2 == 1) selects [7,5,3,1].
  uint32_t select_code = (threadIdx.x & 0x8u) ? 0x7531u : 0x6420u;
  uint32_t tmp = __byte_perm(tmp0, tmp1, select_code);
  // Right-shift by 4 bits to align 4b nibbles to the correct place.
  tmp = tmp >> (threadIdx.x & 0x4u);
  // At this point the 4b data are distributed in individual bytes.
  // Pack them into byte 0 and byte 2 for efficient data conversion.
  tmp = tmp & 0x0F0F0F0F;
  tmp = tmp | (tmp >> 4);
  return tmp;
}

#endif  // FLASHINFER_FRAG_LAYOUT_SWIZZLE_CUH_
