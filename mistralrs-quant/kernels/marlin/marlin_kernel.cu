/*
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MARLIN_CUDA_KERNEL_CUH
#define MARLIN_CUDA_KERNEL_CUH
#include "marlin/marlin_dtypes.cuh"
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace marlin;

// m16n8k16 tensor core mma instruction with fp16/bf16 inputs and fp32
// output/accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename ScalarType<scalar_t>::FragA &a_frag,
                           const typename ScalarType<scalar_t>::FragB &frag_b,
                           typename ScalarType<scalar_t>::FragC &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  float *c = reinterpret_cast<float *>(&frag_c);
  if constexpr (std::is_same<scalar_t, half>::value) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <typename scalar_t>
__device__ inline void ldsm4(typename ScalarType<scalar_t>::FragA &frag_a,
                             const void *smem_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem));
}

// Lookup-table based 3-input logical operation; explicitly used for
// dequantization as the compiler does not seem to automatically recognize it in
// all cases.
template <int lut> __device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(res)
               : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

template <typename scalar_t>
__device__ inline typename ScalarType<scalar_t>::FragB dequant(int q);

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16
// values. We mostly follow the strategy in the link below, with some small
// changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
template <>
__device__ inline typename ScalarType<half>::FragB dequant<half>(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point
  // directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  typename ScalarType<half>::FragB frag_b;
  frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo),
                      *reinterpret_cast<const half2 *>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2 *>(&hi),
                      *reinterpret_cast<const half2 *>(&MUL),
                      *reinterpret_cast<const half2 *>(&ADD));
  return frag_b;
}

template <>
__device__ inline typename ScalarType<nv_bfloat16>::FragB
dequant<nv_bfloat16>(int q) {
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;

  // Guarantee that the `(a & b) | c` operations are LOP3s.

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);

  typename ScalarType<nv_bfloat16>::FragB frag_b;
  static constexpr uint32_t MUL = 0x3F803F80;
  static constexpr uint32_t ADD = 0xC308C308;

  frag_b[0] = __hfma2(*reinterpret_cast<nv_bfloat162 *>(&lo),
                      *reinterpret_cast<const nv_bfloat162 *>(&MUL),
                      *reinterpret_cast<const nv_bfloat162 *>(&ADD));
  frag_b[1] = __hfma2(*reinterpret_cast<nv_bfloat162 *>(&hi),
                      *reinterpret_cast<const nv_bfloat162 *>(&MUL),
                      *reinterpret_cast<const nv_bfloat162 *>(&ADD));
  return frag_b;
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <typename scalar_t>
__device__ inline void scale(typename ScalarType<scalar_t>::FragB &frag_b,
                             typename ScalarType<scalar_t>::FragS &frag_s,
                             int i) {
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  scalar_t2 s =
      ScalarType<scalar_t>::num2num2(reinterpret_cast<scalar_t *>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <typename scalar_t,
          const int threads,         // number of threads in a threadblock
          const int thread_m_blocks, // number of 16x16 blocks in the m
                                     // dimension (batchsize) of the
                                     // threadblock
          const int thread_n_blocks, // same for n dimension (output)
          const int thread_k_blocks, // same for k dimension (reduction)
          const int stages, // number of stages for the async global->shared
                            // fetch pipeline
          const int group_blocks = -1 // number of consecutive 16x16 blocks
                                      // with a separate quantization scale
          >
__global__ void
Marlin(const int4 *__restrict__ A, // fp16 input matrix of shape mxk
       const int4 *__restrict__ B, // 4bit quantized weight matrix of shape kxn
       int4 *__restrict__ C,       // fp16 output buffer of shape mxn
       const int4 *__restrict__ s, // fp16 quantization scales of shape
                                   // (k/groupsize)xn
       int prob_m,                 // batch dimension m
       int prob_n,                 // output dimension n
       int prob_k,                 // reduction dimension k
       int *locks // extra global storage for barrier synchronization
) {
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  using FragA = typename ScalarType<scalar_t>::FragA;
  using FragB = typename ScalarType<scalar_t>::FragB;
  using FragC = typename ScalarType<scalar_t>::FragC;
  using FragS = typename ScalarType<scalar_t>::FragS;

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  // Ensure that the number of tiles in each stripe is a multiple of the
  // groupsize; this avoids an annoying special case where a stripe starts in
  // the middle of group.
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) *
            ceildiv(iters, (group_blocks / thread_k_blocks));

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters; // number of threadblock tiles in the current slice
  int slice_count =
      0;         // total number of active threadblocks in the current slice
  int slice_idx; // index of threadblock in current slice; numbered bottom to
                 // top

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Compute all information about the current slice which is required for
  // synchronization.
  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8; // stride of the A matrix in global memory
  // We typically use `constexpr` to indicate that this value is a compile-time
  // constant
  constexpr int a_sh_stride =
      16 * thread_k_blocks / 8; // stride of an A matrix tile in shared memory
  constexpr int a_gl_rd_delta_o =
      16 * thread_k_blocks /
      8; // delta between subsequent A tiles in global memory
  int a_gl_rd_delta_i =
      a_gl_stride *
      (threads / a_gl_rd_delta_o); // between subsequent accesses within a tile
  constexpr int a_sh_wr_delta =
      a_sh_stride * (threads / a_gl_rd_delta_o); // between shared memory writes
  constexpr int a_sh_rd_delta_o =
      2 * ((threads / 32) /
           (thread_n_blocks / 4)); // between shared memory tile reads
  constexpr int a_sh_rd_delta_i =
      a_sh_stride * 16; // within a shared memory tile
  constexpr int a_sh_stage =
      a_sh_stride * (16 * thread_m_blocks); // overall size of a tile
  constexpr int a_sh_wr_iters =
      ceildiv(a_sh_stage,
              a_sh_wr_delta); // number of shared write iterations for a tile

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd =
      b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd;
  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
              (threadIdx.x % 32) % 4;

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4 *B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  // Shared memory storage for global fetch pipelines.
  int4 *sh_a = sh;
  int4 *sh_b = sh_a + (stages * a_sh_stage);
  int4 *sh_s = sh_b + (stages * b_sh_stage);
  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float *>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }
      // Only fetch scales if this tile starts a new group
      if constexpr (group_blocks != -1) {
        // This assumes group_blocks >= thread_k_blocks
        // and would need to be modified to support smaller groups.
        static_assert(group_blocks >= thread_k_blocks);
        if (pipe % (group_blocks / thread_k_blocks) == 0) {
          int4 *sh_s_stage = sh_s + s_sh_stage * pipe;
          if (s_sh_wr_pred)
            cp_async4(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    // It may seem inefficient that we reload the groups for every sub-tile;
    // however, this does not seem to be a significant bottleneck, while some
    // theoretically better attempts have lead to bad instruction ordering by
    // the compiler and correspondingly a noticeable drop in performance.
    if constexpr (group_blocks != -1) {
      // This assumes group_blocks >= thread_k_blocks
      // and would need to be modified to support smaller groups.
      static_assert(group_blocks >= thread_k_blocks);
      int4 *sh_s_stage =
          sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) *
                               (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4 *>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    }
    int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4<scalar_t>(frag_a[k % 2][i],
                      &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4 *>(
        &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  // Execute the actual tensor core matmul of a sub-tile.
  auto matmul = [&](int k) {
// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant<scalar_t>(b_quant);
      // If there are no groups, we can just scale the final output once and can
      // avoid doing so for each weight.
      if (group_blocks != -1)
        scale<scalar_t>(frag_b0, frag_s[k % 2][j], 0);
      FragB frag_b1 = dequant<scalar_t>(b_quant_shift);
      if (group_blocks != -1)
        scale<scalar_t>(frag_b1, frag_s[k % 2][j], 1);
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<scalar_t>(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma<scalar_t>(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) +
                      (threadIdx.x % b_sh_stride);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float *c_rd = reinterpret_cast<float *>(
                    &sh[red_sh_delta * j + red_sh_rd]);
                float *c_wr = reinterpret_cast<float *>(&sh[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] =
                  reinterpret_cast<int4 *>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float *c_rd =
                reinterpret_cast<float *>(&sh[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) +
                    4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
// Interestingly, doing direct global accesses here really seems to mess up
// the compiler and lead to slowdowns, hence we also use async-copies even
// though these fetches are not actually asynchronous.
#pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(&sh[c_sh_wr + c_sh_wr_delta * i],
                         &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                            c_gl_wr_delta_i * (i % 2)],
                         i < (thread_m_blocks - 1) * 4 ||
                             8 * (i / 2) + row < prob_m);
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float *>(
                  &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] +=
                  Dtype::num2float(reinterpret_cast<scalar_t *>(&c_red)[j]);
            }
          }
          if (!last) {
            int4 c;
#pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<scalar_t *>(&c)[j] =
                  Dtype::float2num(reinterpret_cast<float *>(
                      &frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]);
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] =
                c;
          }
        }
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr =
        (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS &s) {
      scalar_t2 res =
          Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (group_blocks == -1) {
        res = __hmul2(res, s[0]);
      }

      ((scalar_t2 *)sh)[idx] = res;
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0;
         i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {
#pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  // Main loop.
  while (slice_iters) {
// We unroll over both the global fetch and the register load pipeline to
// ensure all shared memory accesses are static. Note that both pipelines have
// even length meaning that the next iteration will always start at index 0.
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if (group_blocks == -1 && last) {
        if (s_sh_wr_pred)
          cp_async4(&sh_s[s_sh_wr], &s[s_gl_rd]);
        cp_async_fence();
      }
      thread_block_reduce();
      if (group_blocks == -1 && last) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          reinterpret_cast<int4 *>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          reinterpret_cast<int4 *>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      }
      if (slice_count > 1) { // only globally reduce if there is more than one
                             // block in a slice
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last) // only the last block in a slice actually writes the result
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
const int USER_THREADS =
    256;              // Note: This is only used with user-provided thread_k/n
const int STAGES = 4; // 4 pipeline stages fit into shared memory
const int SHARED_MEM =
    96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)
static constexpr int pack_factor_4bit =
    8; // We have 8 4-bit vals inside a 32 bit

#define __CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS,           \
                  GROUP_BLOCKS, NUM_THREADS)                                   \
  else if (thread_m_blocks == THREAD_M_BLOCKS &&                               \
           thread_n_blocks == THREAD_N_BLOCKS &&                               \
           thread_k_blocks == THREAD_K_BLOCKS &&                               \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {       \
    cudaFuncSetAttribute(                                                      \
        Marlin<scalar_t, NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,        \
               THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>,                         \
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);              \
    Marlin<scalar_t, NUM_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS,            \
           THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>                              \
        <<<blocks, NUM_THREADS, SHARED_MEM, stream>>>(                         \
            A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks);        \
  }

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority
    // thread_k, thread_n, num_threads
    {128, 128, 256}, // Default
    {128, 64, 128},  // Reduce N 2X, same K
    {64, 256, 256},  // Reduce K 2X, increase N 2X
    {64, 128, 128},  // Reduce K 2X, same N
};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority
    // thread_k, thread_n, num_threads
    {64, 256, 256},  // Default
    {128, 128, 256}, // Reduce N 2X, increase K 2X
    {64, 128, 128},  // Reduce N 2X, same K
    {128, 64, 128},  // Reduce N 4X, increase K 2X
};

bool is_valid_config(thread_config_t const &th_config, int prob_m, int prob_n,
                     int prob_k) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // thread_k can be only 128 or 64 (because it must be less than groupsize
  // which is 128)
  if (th_config.thread_k != 128 && th_config.thread_k != 64) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  return true;
}

thread_config_t determine_thread_config(int prob_m, int prob_n, int prob_k) {
  if (prob_m <= 16) {
    for (auto th_config : small_batch_thread_configs) {
      if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
        return th_config;
      }
    }

  } else {
    for (auto th_config : large_batch_thread_configs) {
      if (is_valid_config(th_config, prob_m, prob_n, prob_k)) {
        return th_config;
      }
    }
  }

  return thread_config_t{-1, -1, -1};
}

#define CALL_IF(N_BLOCKS, K_BLOCKS, NUM_THREADS)                               \
  __CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                            \
  __CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                             \
  __CALL_IF(1, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                            \
  __CALL_IF(1, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                             \
  __CALL_IF(2, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                            \
  __CALL_IF(2, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                             \
  __CALL_IF(3, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                            \
  __CALL_IF(3, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)                             \
  __CALL_IF(4, N_BLOCKS, K_BLOCKS, -1, NUM_THREADS)                            \
  __CALL_IF(4, N_BLOCKS, K_BLOCKS, 8, NUM_THREADS)

template <typename scalar_t>
void marlin_matmul(const void *A, const void *B, void *s, void *C, int prob_m,
                   int prob_k, int prob_n, void *workspace, int groupsize,
                   cudaStream_t stream) {

  int dev = 0;
  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;
  int max_par = 16;

  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

  // Set thread config
  thread_config_t th_config;
  if (thread_k != -1 && thread_n != -1) {
    // User-defined config
    th_config = thread_config_t{thread_k, thread_n, USER_THREADS};
  } else {
    // Auto config
    th_config = determine_thread_config(prob_m, prob_n, prob_k);
  }

  if (!is_valid_config(th_config, prob_m, prob_n, prob_k)) {
    assert(false);
  }

  int num_threads = th_config.num_threads;
  thread_k = th_config.thread_k;
  thread_n = th_config.thread_n;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_m == 0 || prob_n == 0 || prob_k == 0) {
    return;
  }

  const int4 *A_ptr = (const int4 *)A;
  const int4 *B_ptr = (const int4 *)B;
  int4 *C_ptr = (int4 *)C;
  const int4 *s_ptr = (const int4 *)s;

  int *locks = (int *)workspace;

  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any
      // padding
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    // For compilation speed, we only define the kernel configurations that have
    // seemed useful (in terms of performance) in our testing, however many more
    // are, in principle, possible.
    if (false) {
    }
    CALL_IF(8, 8, 256)
    CALL_IF(16, 4, 256)
    CALL_IF(8, 4, 128)
    CALL_IF(4, 8, 128)
    else {
      assert(false);
    }

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }
}

extern "C" void marlin_4bit_f16(const void *A, const void *B, void *s, void *C,
                                int prob_m, int prob_k, int prob_n,
                                void *workspace, int groupsize,
                                cudaStream_t stream) {
  marlin_matmul<half>(A, B, s, C, prob_m, prob_k, prob_n, workspace, groupsize,
                      stream);
}

extern "C" void marlin_4bit_bf16(const void *A, const void *B, void *s, void *C,
                                 int prob_m, int prob_k, int prob_n,
                                 void *workspace, int groupsize,
                                 cudaStream_t stream) {
  marlin_matmul<nv_bfloat16>(A, B, s, C, prob_m, prob_k, prob_n, workspace,
                             groupsize, stream);
}

template <int const num_threads, int const num_bits, bool const has_perm>
__global__ void
gptq_marlin_repack_kernel(uint32_t const *__restrict__ b_q_weight_ptr,
                          uint32_t const *__restrict__ perm_ptr,
                          uint32_t *__restrict__ out_ptr, int size_k,
                          int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  int k_tiles = size_k / tile_k_size;
  int n_tiles = size_n / tile_n_size;
  int block_k_tiles = ceildiv(k_tiles, gridDim.x);

  int start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = tile_k_size / 4;

  int4 *sh_perm_ptr = sh;
  int4 *sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = tile_k_size / pack_factor;

  constexpr int stage_n_threads = tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * tile_k_size) / 4;

    int4 const *perm_int4_ptr = reinterpret_cast<int4 const *>(perm_ptr);

    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * tile_n_size;

    int4 *sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        int k_id = threadIdx.x / stage_n_threads;
        int n_id = threadIdx.x % stage_n_threads;

        uint32_t const *sh_perm_int_ptr =
            reinterpret_cast<uint32_t const *>(sh_perm_ptr);

        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const *>(&(
                b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
      }

    } else {
      if (threadIdx.x < stage_size) {
        int k_id = threadIdx.x / stage_n_threads;
        int n_id = threadIdx.x % stage_n_threads;

        int first_k = k_tile_id * tile_k_size;
        int first_k_packed = first_k / pack_factor;

        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<int4 const *>(
                      &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                       first_n + (n_id * 4)])));
      }
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    int warp_id = threadIdx.x / 32;
    int th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = 64;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4 *sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t *sh_stage_int_ptr = reinterpret_cast<uint32_t *>(sh_stage_ptr);

    uint32_t *sh_perm_int_ptr = reinterpret_cast<uint32_t *>(sh_perm_ptr);

    uint32_t vals[8];

    if constexpr (has_perm) {
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];

        uint32_t src_k = sh_perm_int_ptr[k_idx];
        uint32_t src_k_pos = src_k % pack_factor;

        uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
        uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;

        uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
        uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;

        vals[i] = b1_cur_val;
        vals[4 + i] = b2_cur_val;
      }

    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

#pragma unroll
      for (int i = 0; i < tile_ints; i++) {
        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
        b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
      }

#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cur_elem = tc_row + tc_offsets[i];
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;

        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      }
    }

    constexpr int tile_size = tile_k_size * tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (num_bits == 4) {
      constexpr int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

#define CALL_IF2(NUM_BITS, HAS_PERM)                                           \
  else if (num_bits == NUM_BITS && has_perm == HAS_PERM) {                     \
    cudaFuncSetAttribute(                                                      \
        gptq_marlin_repack_kernel<repack_threads, NUM_BITS, HAS_PERM>,         \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);          \
    gptq_marlin_repack_kernel<repack_threads, NUM_BITS, HAS_PERM>              \
        <<<blocks, repack_threads, max_shared_mem, stream>>>(                  \
            b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n);                \
  }

extern "C" void gptq_marlin_repack(void *weight, void *perm, void *out,
                                   int size_k, int size_n, int num_bits) {
  // Verify compatibility with marlin tile of 16x64
  assert(size_k % tile_k_size == 0);
  assert(size_n % tile_n_size == 0);
  assert(num_bits == 4 || num_bits == 8);
  const int pack_factor = 32 / num_bits;

  bool has_perm = true;
  int dev = 0;
  // Get ptrs
  uint32_t const *b_q_weight_ptr = reinterpret_cast<uint32_t const *>(weight);
  uint32_t const *perm_ptr = reinterpret_cast<uint32_t const *>(perm);
  uint32_t *out_ptr = reinterpret_cast<uint32_t *>(out);

  // Get dev info
  cudaStream_t stream = 0;
  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  assert(max_shared_mem > 0);

  if (false) {
  }
  CALL_IF2(4, false)
  CALL_IF2(4, true)
  CALL_IF2(8, false)
  CALL_IF2(8, true)
  else {
    assert(false);
  }
}

#endif
