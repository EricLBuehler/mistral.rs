// Prism ternary (PTQ1_0) matmul with the Hadamard weight fold applied to
// activations. Activations are quantized to int8 (one scale per weight block)
// so the dot product runs on dp4a.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#include "ptq1_0_common.cuh"

#define MATMUL_WARPS 8
#define MATMUL_TOKENS_PER_PASS 8
#define MATMUL_PREFETCH 2
#define BLOCKS_PER_WARP_STEP 4 // 4 blocks x 7 words fill 28 of 32 lanes
#define BPL_LANES_PER_ROW 8 // K / 128 is a multiple of 8 for folded weights
#define BPL_ROWS_PER_WARP (WARP_SIZE / BPL_LANES_PER_ROW)
#define BPL_MAX_TOKENS 15 // from 16 tokens the tensor-core GEMM takes over
#define BPL_X_PITCH 9 // int4 per staged block: 8 data + 1 pad, so 8 lanes on
                      // different blocks read distinct banks
#define BPL_X_SMEM_MAX (48 * 1024)
#define SCALE_WORD (ptq1_0::BLOCK_WORDS - 1) // scale is in its top half
#define INT8_MAX_F 127.0f

// One CTA per (FWHT_BLOCK columns, token): gather, signs and FWHT in shared
// memory, then int8 quantization with one scale per 128 columns. Without
// do_fwht it only quantizes.
template <typename T>
static __global__ void
ptq1_0_prepare_kernel(const T *__restrict__ x, const float *__restrict__ signs,
                      const uint32_t *__restrict__ gather,
                      int8_t *__restrict__ xq, float *__restrict__ xscale,
                      int k, int do_fwht) {
  __shared__ float s[ptq1_0::FWHT_BLOCK];
  const int tid = threadIdx.x;
  const int col0 = blockIdx.x * ptq1_0::FWHT_BLOCK;
  const size_t base = static_cast<size_t>(blockIdx.y) * k;

  for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += PREPARE_THREADS) {
    const int p = col0 + i;
    float v = 0.0f;
    if (p < k) {
      v = to_float(x[base + (gather ? gather[p] : p)]);
      if (signs) {
        v *= signs[p];
      }
    }
    s[i] = v;
  }
  __syncthreads();

  if (do_fwht) {
    fwht_shared(s, tid);
  }

  // Each warp quantizes one 128-column segment, four columns per lane.
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int seg = col0 + warp * ptq1_0::BLOCK_ELEMS;
  if (seg < k) {
    const float scale = do_fwht ? FWHT_SCALE : 1.0f;
    float v[4];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = s[warp * ptq1_0::BLOCK_ELEMS + lane * 4 + j] * scale;
      amax = fmaxf(amax, fabsf(v[j]));
    }
    amax = ptq1_0::warp_max(amax);
    const float inv = amax > 0.0f ? INT8_MAX_F / amax : 0.0f;
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      packed |= (static_cast<uint32_t>(__float2int_rn(v[j] * inv)) & 0xFFu)
                << (8 * j);
    }
    reinterpret_cast<uint32_t *>(xq + base + seg)[lane] = packed;
    if (lane == 0) {
      xscale[static_cast<size_t>(blockIdx.y) * (k / ptq1_0::BLOCK_ELEMS) +
             seg / ptq1_0::BLOCK_ELEMS] = amax / INT8_MAX_F;
    }
  }
}

// One warp per output row. Lane L reads word L % 7 of block (step * 4 + L / 7),
// so a warp step loads 112 contiguous bytes.
// One thread per weight block, BPL_LANES_PER_ROW lanes per row: straight-line
// decode of all 128 weights and vector loads of the activations.
template <typename OutT, int TT, bool XSMEM>
static __global__ void
ptq1_0_matmul_bpl_kernel(const uint32_t *__restrict__ w,
                         const int8_t *__restrict__ xq,
                         const float *__restrict__ xscale,
                         OutT *__restrict__ dst, int ncols_x, int nrows_x,
                         int b_size) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int sub = lane % BPL_LANES_PER_ROW;
  const int first_row =
      (blockIdx.x * MATMUL_WARPS + threadIdx.x / WARP_SIZE) * BPL_ROWS_PER_WARP;
  const int row = first_row + lane / BPL_LANES_PER_ROW;
  const int t0 = blockIdx.y * TT;
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;
  const size_t row_words = static_cast<size_t>(nblk) * ptq1_0::BLOCK_WORDS;
  const size_t row_at = min(row, nrows_x - 1);
  const uint32_t *wrow = w + row_at * row_words;

  const int4 *x16[TT];
  const int2 *x8[TT];
  const float *srow[TT];
  float acc[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    const int tok = min(t0 + t, b_size - 1);
    const int8_t *xbytes = xq + static_cast<size_t>(tok) * ncols_x;
    x16[t] = reinterpret_cast<const int4 *>(xbytes);
    x8[t] = reinterpret_cast<const int2 *>(xbytes);
    srow[t] = xscale + static_cast<size_t>(tok) * nblk;
    acc[t] = 0.0f;
  }

  extern __shared__ int4 x_stage[];
  if (XSMEM) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      int4 *dst = x_stage + static_cast<size_t>(t) * nblk * BPL_X_PITCH;
      for (int c = threadIdx.x; c < nblk * 8; c += blockDim.x) {
        dst[(c >> 3) * BPL_X_PITCH + (c & 7)] = __ldg(x16[t] + c);
      }
    }
    __syncthreads();
  }
  // Activation words for block b: 16 bytes at index n of 8, or 8 bytes at
  // index u of 16.
  auto load16 = [&](int t, int b, int n) -> int4 {
    if (XSMEM) {
      return x_stage[static_cast<size_t>(t) * nblk * BPL_X_PITCH +
                     b * BPL_X_PITCH + n];
    }
    return __ldg(x16[t] + b * 8 + n);
  };
  auto load8 = [&](int t, int b, int u) -> int2 {
    if (XSMEM) {
      const int4 v = x_stage[static_cast<size_t>(t) * nblk * BPL_X_PITCH +
                             b * BPL_X_PITCH + (u >> 1)];
      return (u & 1) ? make_int2(v.z, v.w) : make_int2(v.x, v.y);
    }
    return __ldg(x8[t] + b * 16 + u);
  };

  for (int b = sub; b < nblk; b += BPL_LANES_PER_ROW) {
    uint32_t wd[ptq1_0::BLOCK_WORDS];
#pragma unroll
    for (int i = 0; i < ptq1_0::BLOCK_WORDS; ++i) {
      wd[i] = __ldg(wrow + static_cast<size_t>(b) * ptq1_0::BLOCK_WORDS + i);
    }
    const float d = __half2float(
        __ushort_as_half(static_cast<unsigned short>(wd[6] >> 16)));
    int isum[TT] = {};

    uint32_t lo[4], hi[4];
#pragma unroll
    for (int g = 0; g < ptq1_0::WIDE_WORDS; ++g) {
      ptq1_0::init_state(wd[g], g, lo[g], hi[g]);
    }
#pragma unroll
    for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
      int q[4];
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        q[g] = ptq1_0::decode_step(lo[g], hi[g], 3u);
      }
#pragma unroll
      for (int t = 0; t < TT; ++t) {
        const int4 xv = load16(t, b, n);
        isum[t] = ptq1_0::dot4(q[0], xv.x, isum[t]);
        isum[t] = ptq1_0::dot4(q[1], xv.y, isum[t]);
        isum[t] = ptq1_0::dot4(q[2], xv.z, isum[t]);
        isum[t] = ptq1_0::dot4(q[3], xv.w, isum[t]);
      }
    }

#pragma unroll
    for (int g = 0; g < 2; ++g) {
      ptq1_0::init_state(wd[ptq1_0::WIDE_WORDS + g], ptq1_0::WIDE_WORDS + g,
                         lo[g], hi[g]);
    }
#pragma unroll
    for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
      int q[2];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        q[g] = ptq1_0::decode_step(lo[g], hi[g], 3u);
      }
#pragma unroll
      for (int t = 0; t < TT; ++t) {
        const int2 xv = load8(t, b, 10 + n);
        isum[t] = ptq1_0::dot4(q[0], xv.x, isum[t]);
        isum[t] = ptq1_0::dot4(q[1], xv.y, isum[t]);
      }
    }

    // qh holds 8 weights: two digits of two bytes per dp4a, so the second
    // step starts two digits further on.
    uint32_t q0_lo, q0_hi;
    ptq1_0::init_state(wd[6], 6, q0_lo, q0_hi);
    const uint32_t w0_lo = q0_lo * 3u;
    const uint32_t w0_hi = q0_hi * 3u;
    const uint32_t w1_lo = ((q0_lo * 9u) & ptq1_0::LANE_MASK) * 3u;
    const uint32_t w1_hi = ((q0_lo * 27u) & ptq1_0::LANE_MASK) * 3u;
    const int qa = static_cast<int>(ptq1_0::sub_bytes(
        ptq1_0::byte_perm(w0_lo, w0_hi, 0x7531), ptq1_0::ONES));
    const int qb = static_cast<int>(ptq1_0::sub_bytes(
        ptq1_0::byte_perm(w1_lo, w1_hi, 0x7531), ptq1_0::ONES));
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const int2 xv = load8(t, b, 15);
      isum[t] = ptq1_0::dot4(qa, xv.x, isum[t]);
      isum[t] = ptq1_0::dot4(qb, xv.y, isum[t]);
    }

#pragma unroll
    for (int t = 0; t < TT; ++t) {
      acc[t] += d * __ldg(srow[t] + b) * static_cast<float>(isum[t]);
    }
  }

#pragma unroll
  for (int t = 0; t < TT; ++t) {
    for (int off = BPL_LANES_PER_ROW / 2; off > 0; off >>= 1) {
      acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], off, BPL_LANES_PER_ROW);
    }
  }
  if (sub == 0 && row < nrows_x) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      if (t0 + t < b_size) {
        store_float(dst + static_cast<size_t>(t0 + t) * nrows_x + row, acc[t]);
      }
    }
  }
}

// Benchmark baseline: one warp per row, lane L owns word L % 7 of a block.
template <typename OutT, int TT>
static __global__ void
ptq1_0_matmul_kernel(const uint32_t *__restrict__ w,
                     const int8_t *__restrict__ xq,
                     const float *__restrict__ xscale, OutT *__restrict__ dst,
                     int ncols_x, int nrows_x, int b_size) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int row = blockIdx.x * MATMUL_WARPS + threadIdx.x / WARP_SIZE;
  const int t0 = blockIdx.y * TT;
  if (row >= nrows_x) {
    return;
  }
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;
  const uint32_t *wrow =
      w + static_cast<size_t>(row) * nblk * ptq1_0::BLOCK_WORDS;

  const int slot = lane / ptq1_0::BLOCK_WORDS;
  const int word_index = lane % ptq1_0::BLOCK_WORDS;
  const ptq1_0::DpLane dl = ptq1_0::dp_lane(word_index);

  const int *xrow[TT];
  const float *srow[TT];
  float acc[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    const int tok = min(t0 + t, b_size - 1);
    const int8_t *xbytes = xq + static_cast<size_t>(tok) * ncols_x;
    xrow[t] = reinterpret_cast<const int *>(xbytes);
    srow[t] = xscale + static_cast<size_t>(tok) * nblk;
    acc[t] = 0.0f;
  }

  for (int step = 0; step < nblk;
       step += BLOCKS_PER_WARP_STEP * MATMUL_PREFETCH) {
    uint32_t words[MATMUL_PREFETCH];
#pragma unroll
    for (int p = 0; p < MATMUL_PREFETCH; ++p) {
      const int b = step + p * BLOCKS_PER_WARP_STEP + slot;
      const size_t at =
          static_cast<size_t>(b) * ptq1_0::BLOCK_WORDS + word_index;
      words[p] = (slot < BLOCKS_PER_WARP_STEP && b < nblk) ? __ldg(wrow + at)
                                                           : 0u;
    }
#pragma unroll
    for (int p = 0; p < MATMUL_PREFETCH; ++p) {
      const int b = step + p * BLOCKS_PER_WARP_STEP + slot;
      const bool valid = slot < BLOCKS_PER_WARP_STEP && b < nblk;
      const int scale_lane = slot * ptq1_0::BLOCK_WORDS + SCALE_WORD;
      const uint32_t last = __shfl_sync(0xffffffffu, words[p], scale_lane);
      const float d = __half2float(
          __ushort_as_half(static_cast<unsigned short>(last >> 16)));

      uint32_t v_lo, v_hi;
      ptq1_0::init_state(words[p], word_index, v_lo, v_hi);
      int isum[TT] = {};
#pragma unroll
      for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
        const int q = ptq1_0::decode_step(v_lo, v_hi, dl.mult);
        if (n < dl.steps) {
          const int e = valid ? b * ptq1_0::BLOCK_ELEMS + dl.elem_base +
                                    n * dl.elem_stride
                              : 0;
#pragma unroll
          for (int t = 0; t < TT; ++t) {
            isum[t] = ptq1_0::dot4(q, __ldg(xrow[t] + (e >> 2)), isum[t]);
          }
        }
      }
      if (valid) {
#pragma unroll
        for (int t = 0; t < TT; ++t) {
          acc[t] += d * __ldg(srow[t] + b) * static_cast<float>(isum[t]);
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TT; ++t) {
    acc[t] = ptq1_0::warp_sum(acc[t]);
  }
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      if (t0 + t < b_size) {
        store_float(dst + static_cast<size_t>(t0 + t) * nrows_x + row, acc[t]);
      }
    }
  }
}

// `scratch` holds b_size * ncols_x int8 activations, then one f32 scale per
// 128 columns per token.
template <typename T>
static void ptq1_0_prepare_launch(const void *x, const void *signs,
                                  const void *gather, void *scratch,
                                  int ncols_x, int b_size, int do_fwht,
                                  cudaStream_t s) {
  int8_t *xq = static_cast<int8_t *>(scratch);
  float *xscale =
      reinterpret_cast<float *>(xq + static_cast<size_t>(b_size) * ncols_x);
  dim3 grid((ncols_x + ptq1_0::FWHT_BLOCK - 1) / ptq1_0::FWHT_BLOCK, b_size,
            1);
  ptq1_0_prepare_kernel<T><<<grid, PREPARE_THREADS, 0, s>>>(
      static_cast<const T *>(x), static_cast<const float *>(signs),
      static_cast<const uint32_t *>(gather), xq, xscale, ncols_x, do_fwht);
}

template <typename T>
static void ptq1_0_launch_lanes(const void *x, const void *w,
                                const void *signs, const void *gather,
                                void *scratch, void *dst, int ncols_x,
                                int nrows_x, int b_size, int do_fwht,
                                void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  ptq1_0_prepare_launch<T>(x, signs, gather, scratch, ncols_x, b_size, do_fwht,
                           s);
  const int8_t *xq = static_cast<const int8_t *>(scratch);
  const float *xscale = reinterpret_cast<const float *>(
      xq + static_cast<size_t>(b_size) * ncols_x);
  const int block = MATMUL_WARPS * WARP_SIZE;
  const unsigned int row_blocks = (nrows_x + MATMUL_WARPS - 1) / MATMUL_WARPS;
  const uint32_t *wp = static_cast<const uint32_t *>(w);
  if (b_size == 1) {
    ptq1_0_matmul_kernel<T, 1><<<dim3(row_blocks, 1, 1), block, 0, s>>>(
        wp, xq, xscale, static_cast<T *>(dst), ncols_x, nrows_x, b_size);
  } else {
    const unsigned int passes =
        (b_size + MATMUL_TOKENS_PER_PASS - 1) / MATMUL_TOKENS_PER_PASS;
    ptq1_0_matmul_kernel<T, MATMUL_TOKENS_PER_PASS>
        <<<dim3(row_blocks, passes, 1), block, 0, s>>>(
            wp, xq, xscale, static_cast<T *>(dst), ncols_x, nrows_x, b_size);
  }
}

template <typename T, int TT, bool XSMEM>
static void ptq1_0_launch_bpl(const void *x, const void *w, const void *signs,
                              const void *gather, void *scratch, void *dst,
                              int ncols_x, int nrows_x, int b_size,
                              int do_fwht, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  ptq1_0_prepare_launch<T>(x, signs, gather, scratch, ncols_x, b_size, do_fwht,
                           s);
  const int8_t *xq = static_cast<const int8_t *>(scratch);
  const float *xscale = reinterpret_cast<const float *>(
      xq + static_cast<size_t>(b_size) * ncols_x);
  const int block = MATMUL_WARPS * WARP_SIZE;
  const int rows_per_cta = MATMUL_WARPS * BPL_ROWS_PER_WARP;
  const unsigned int row_blocks = (nrows_x + rows_per_cta - 1) / rows_per_cta;
  const unsigned int passes = (b_size + TT - 1) / TT;
  const size_t smem = XSMEM ? static_cast<size_t>(TT) *
                                  (ncols_x / ptq1_0::BLOCK_ELEMS) *
                                  BPL_X_PITCH * sizeof(int4)
                            : 0;
  ptq1_0_matmul_bpl_kernel<T, TT, XSMEM>
      <<<dim3(row_blocks, passes, 1), block, smem, s>>>(
          static_cast<const uint32_t *>(w), xq, xscale, static_cast<T *>(dst),
          ncols_x, nrows_x, b_size);
}

// The thread-per-block kernel, one token per pass, wins for small batches;
// the lane kernel is only the fallback for larger ones on GPUs without the
// tensor-core GEMM. Activations are staged in shared memory when the row fits.
template <typename T>
static void ptq1_0_launch(const void *x, const void *w, const void *signs,
                          const void *gather, void *scratch, void *dst,
                          int ncols_x, int nrows_x, int b_size, int do_fwht,
                          void *stream) {
  if (b_size > BPL_MAX_TOKENS) {
    ptq1_0_launch_lanes<T>(x, w, signs, gather, scratch, dst, ncols_x, nrows_x,
                           b_size, do_fwht, stream);
    return;
  }
  const size_t stage_bytes = static_cast<size_t>(ncols_x) /
                             ptq1_0::BLOCK_ELEMS * BPL_X_PITCH * sizeof(int4);
  if (stage_bytes <= BPL_X_SMEM_MAX) {
    ptq1_0_launch_bpl<T, 1, true>(x, w, signs, gather, scratch, dst, ncols_x,
                                  nrows_x, b_size, do_fwht, stream);
  } else {
    ptq1_0_launch_bpl<T, 1, false>(x, w, signs, gather, scratch, dst, ncols_x,
                                   nrows_x, b_size, do_fwht, stream);
  }
}

// One CTA per token id: decodes the packed row, then the inverse fold (FWHT,
// then signs) that latent embedding rows need.
template <typename OutT>
static __global__ void
ptq1_0_embedding_kernel(const uint32_t *__restrict__ ids,
                        const uint8_t *__restrict__ w,
                        const float *__restrict__ signs,
                        OutT *__restrict__ dst, int ncols_x) {
  __shared__ float s[ptq1_0::FWHT_BLOCK];
  const int tid = threadIdx.x;
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;
  const uint8_t *row =
      w + static_cast<size_t>(ids[blockIdx.x]) * nblk * ptq1_0::BLOCK_BYTES;
  OutT *out = dst + static_cast<size_t>(blockIdx.x) * ncols_x;

  for (int col0 = 0; col0 < ncols_x; col0 += ptq1_0::FWHT_BLOCK) {
    for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += PREPARE_THREADS) {
      const int p = col0 + i;
      const size_t block_at = p / ptq1_0::BLOCK_ELEMS;
      const uint8_t *blk = row + block_at * ptq1_0::BLOCK_BYTES;
      s[i] = static_cast<float>(
                 ptq1_0::element_trit(blk, p % ptq1_0::BLOCK_ELEMS)) *
             block_scale(blk);
    }
    __syncthreads();
    fwht_shared(s, tid);
    for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += PREPARE_THREADS) {
      store_float(out + col0 + i, s[i] * FWHT_SCALE * signs[col0 + i]);
    }
    __syncthreads();
  }
}

// Host-side launchers used by `mistralrs-quant/src/gguf/ffi.rs`.

#define PTQ1_0_LAUNCHER(tag, c_type)                                           \
  extern "C" void launch_ptq1_0_matmul_##tag(                                  \
      const void *x, const void *w, const void *signs, const void *gather,     \
      void *scratch, void *dst, int ncols_x, int nrows_x, int b_size,          \
      int do_fwht, void *stream) {                                             \
    ptq1_0_launch<c_type>(x, w, signs, gather, scratch, dst, ncols_x,          \
                          nrows_x, b_size, do_fwht, stream);                   \
  }

PTQ1_0_LAUNCHER(f32, float)
PTQ1_0_LAUNCHER(f16, __half)
PTQ1_0_LAUNCHER(bf16, __nv_bfloat16)

#define PTQ1_0_EMBEDDING_LAUNCHER(tag, c_type)                                 \
  extern "C" void launch_ptq1_0_embedding_##tag(                               \
      const void *ids, const void *w, const void *signs, void *dst,            \
      int ncols_x, int n_ids, void *stream) {                                  \
    ptq1_0_embedding_kernel<c_type>                                            \
        <<<n_ids, PREPARE_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(    \
            static_cast<const uint32_t *>(ids),                                \
            static_cast<const uint8_t *>(w),                                   \
            static_cast<const float *>(signs), static_cast<c_type *>(dst),     \
            ncols_x);                                                          \
  }

PTQ1_0_EMBEDDING_LAUNCHER(f32, float)
PTQ1_0_EMBEDDING_LAUNCHER(f16, __half)
PTQ1_0_EMBEDDING_LAUNCHER(bf16, __nv_bfloat16)

// Benchmark entry: 0 lane-role kernel, 1/2/3 thread-per-block with 1/2/4
// tokens per pass, 4 the same with activations staged in shared memory (1 token
// per pass); anything else the production dispatch.
extern "C" void launch_ptq1_0_matmul_variant_bf16(
    const void *x, const void *w, const void *signs, const void *gather,
    void *scratch, void *dst, int ncols_x, int nrows_x, int b_size,
    int do_fwht, int variant, void *stream) {
  switch (variant) {
  case 0:
    ptq1_0_launch_lanes<__nv_bfloat16>(x, w, signs, gather, scratch, dst,
                                       ncols_x, nrows_x, b_size, do_fwht,
                                       stream);
    break;
  case 1:
    ptq1_0_launch_bpl<__nv_bfloat16, 1, false>(
        x, w, signs, gather, scratch, dst, ncols_x, nrows_x, b_size, do_fwht,
        stream);
    break;
  case 2:
    ptq1_0_launch_bpl<__nv_bfloat16, 2, false>(
        x, w, signs, gather, scratch, dst, ncols_x, nrows_x, b_size, do_fwht,
        stream);
    break;
  case 3:
    ptq1_0_launch_bpl<__nv_bfloat16, 4, false>(
        x, w, signs, gather, scratch, dst, ncols_x, nrows_x, b_size, do_fwht,
        stream);
    break;
  case 4:
    ptq1_0_launch_bpl<__nv_bfloat16, 1, true>(
        x, w, signs, gather, scratch, dst, ncols_x, nrows_x, b_size, do_fwht,
        stream);
    break;
  default:
    ptq1_0_launch<__nv_bfloat16>(x, w, signs, gather, scratch, dst, ncols_x,
                                 nrows_x, b_size, do_fwht, stream);
  }
}
