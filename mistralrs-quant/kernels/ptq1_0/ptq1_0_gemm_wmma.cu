// Prism ternary (PTQ1_0) tensor-core GEMM for prefill: activations are
// transformed to bf16, weights are decoded to bf16 tiles in shared memory.

#include <mma.h>

#include "ptq1_0_common.cuh"

using namespace nvcuda;

#define GEMM_BM 64 // tokens per CTA
#define GEMM_BN 64 // weight rows per CTA
#define GEMM_THREADS 256
#define GEMM_WARPS_N 4 // warps along the weight rows; 2 along the tokens
#define GEMM_PITCH 136 // bf16 per shared row: 128 + 8 padding
#define GEMM_C_PITCH 68 // floats per staged output row
#define GEMM_A_CHUNKS (GEMM_BM * ptq1_0::BLOCK_ELEMS / 8 / GEMM_THREADS)
#define GEMM_MIN_ARCH 800

// Gather, signs and FWHT as in the decode path, but the result stays bf16.
template <typename T>
static __global__ void
ptq1_0_prepare_bf16_kernel(const T *__restrict__ x,
                           const float *__restrict__ signs,
                           const uint32_t *__restrict__ gather,
                           __nv_bfloat16 *__restrict__ out, int k,
                           int do_fwht) {
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
  const float scale = do_fwht ? FWHT_SCALE : 1.0f;
  for (int i = tid; i < ptq1_0::FWHT_BLOCK; i += PREPARE_THREADS) {
    const int p = col0 + i;
    if (p < k) {
      out[base + p] = __float2bfloat16(s[i] * scale);
    }
  }
}

// Writes four consecutive decoded weights (`q` holds four signed trits) scaled
// by the block scale into a bf16 tile row.
static __device__ __forceinline__ void emit_weights(__nv_bfloat16 *row, int e,
                                                    int q, float d) {
  __nv_bfloat162 pair[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float lo = d * static_cast<float>(static_cast<int8_t>(q >> (16 * i)));
    const float hi =
        d * static_cast<float>(static_cast<int8_t>(q >> (16 * i + 8)));
    pair[i] = __floats2bfloat162_rn(lo, hi);
  }
  *reinterpret_cast<uint2 *>(row + e) = *reinterpret_cast<const uint2 *>(pair);
}

// Four threads share a weight row: part 0/1 the wide words 0-1 / 2-3, part 2
// the narrow words, part 3 the qh bytes.
static __device__ __forceinline__ void decode_part(int part, uint32_t wa,
                                                   uint32_t wb, float d,
                                                   __nv_bfloat16 *row) {
  if (part < 2) {
    const uint32_t words[2] = {wa, wb};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int g = part * 2 + j;
      uint32_t lo, hi;
      ptq1_0::init_state(words[j], g, lo, hi);
#pragma unroll
      for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
        emit_weights(row, n * 16 + 4 * g, ptq1_0::decode_step(lo, hi, 3u), d);
      }
    }
  } else if (part == 2) {
    const uint32_t words[2] = {wa, wb};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint32_t lo, hi;
      ptq1_0::init_state(words[j], ptq1_0::WIDE_WORDS + j, lo, hi);
#pragma unroll
      for (int n = 0; n < ptq1_0::DECODE_STEPS; ++n) {
        emit_weights(row, ptq1_0::NARROW_START + n * 8 + 4 * j,
                     ptq1_0::decode_step(lo, hi, 3u), d);
      }
    }
  } else {
    int first, second;
    ptq1_0::decode_qh(wa, first, second);
    emit_weights(row, ptq1_0::QH_START, first, d);
    emit_weights(row, ptq1_0::QH_START + 4, second, d);
  }
}

// CTA tile GEMM_BM tokens x GEMM_BN rows; K advances one weight block per step.
// The next step's global loads are issued before the tensor-core work.
template <typename OutT>
static __global__ void __launch_bounds__(GEMM_THREADS)
    ptq1_0_gemm_kernel(const uint32_t *__restrict__ w,
                       const __nv_bfloat16 *__restrict__ x,
                       OutT *__restrict__ dst, int ncols_x, int nrows_x,
                       int b_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GEMM_MIN_ARCH
  __shared__ __align__(128) __nv_bfloat16 a_tile[GEMM_BM * GEMM_PITCH];
  __shared__ __align__(128) __nv_bfloat16 b_tile[GEMM_BN * GEMM_PITCH];
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int wm = warp / GEMM_WARPS_N;
  const int wn = warp % GEMM_WARPS_N;
  const int n0 = blockIdx.x * GEMM_BN;
  const int m0 = blockIdx.y * GEMM_BM;
  const int nblk = ncols_x / ptq1_0::BLOCK_ELEMS;

  const int drow = tid >> 2;
  const int part = tid & 3;
  const bool row_ok = n0 + drow < nrows_x;
  const uint32_t *wrow =
      w + static_cast<size_t>(row_ok ? n0 + drow : 0) * nblk *
              ptq1_0::BLOCK_WORDS;

  uint4 a_reg[GEMM_A_CHUNKS];
  uint32_t wa = 0, wb = 0, wscale = 0;
  auto fetch = [&](int kb) {
#pragma unroll
    for (int i = 0; i < GEMM_A_CHUNKS; ++i) {
      const int c = tid + i * GEMM_THREADS;
      const int tok = m0 + (c >> 4);
      a_reg[i] = make_uint4(0, 0, 0, 0);
      if (tok < b_size) {
        a_reg[i] = __ldg(reinterpret_cast<const uint4 *>(
            x + static_cast<size_t>(tok) * ncols_x +
            static_cast<size_t>(kb) * ptq1_0::BLOCK_ELEMS + (c & 15) * 8));
      }
    }
    const uint32_t *blk = wrow + static_cast<size_t>(kb) * ptq1_0::BLOCK_WORDS;
    if (row_ok) {
      wscale = __ldg(blk + ptq1_0::BLOCK_WORDS - 1);
      wa = part < 3 ? __ldg(blk + part * 2) : wscale;
      wb = part < 3 ? __ldg(blk + part * 2 + 1) : 0u;
    } else {
      wscale = wa = wb = 0u;
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
  wmma::fill_fragment(acc[0], 0.0f);
  wmma::fill_fragment(acc[1], 0.0f);

  fetch(0);
  for (int kb = 0; kb < nblk; ++kb) {
#pragma unroll
    for (int i = 0; i < GEMM_A_CHUNKS; ++i) {
      const int c = tid + i * GEMM_THREADS;
      *reinterpret_cast<uint4 *>(&a_tile[(c >> 4) * GEMM_PITCH +
                                         (c & 15) * 8]) = a_reg[i];
    }
    const float d = __half2float(
        __ushort_as_half(static_cast<unsigned short>(wscale >> 16)));
    decode_part(part, wa, wb, d, &b_tile[drow * GEMM_PITCH]);
    if (kb + 1 < nblk) {
      fetch(kb + 1);
    }
    __syncthreads();

#pragma unroll
    for (int k16 = 0; k16 < ptq1_0::BLOCK_ELEMS / 16; ++k16) {
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(b_frag, &b_tile[(wn * 16) * GEMM_PITCH + k16 * 16],
                             GEMM_PITCH);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            a_frag;
        wmma::load_matrix_sync(
            a_frag, &a_tile[(wm * 32 + i * 16) * GEMM_PITCH + k16 * 16],
            GEMM_PITCH);
        wmma::mma_sync(acc[i], a_frag, b_frag, acc[i]);
      }
    }
    __syncthreads();
  }

  float *c_tile = reinterpret_cast<float *>(a_tile);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float *dst_tile = &c_tile[(wm * 32 + i * 16) * GEMM_C_PITCH + wn * 16];
    wmma::store_matrix_sync(dst_tile, acc[i], GEMM_C_PITCH,
                            wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < GEMM_BM * GEMM_BN; idx += GEMM_THREADS) {
    const int r = idx / GEMM_BN;
    const int c = idx % GEMM_BN;
    if (m0 + r < b_size && n0 + c < nrows_x) {
      store_float(dst + static_cast<size_t>(m0 + r) * nrows_x + n0 + c,
                  c_tile[r * GEMM_C_PITCH + c]);
    }
  }
#endif
}

// `scratch` holds b_size * ncols_x bf16 activations.
template <typename T>
static void ptq1_0_gemm_launch(const void *x, const void *w, const void *signs,
                               const void *gather, void *scratch, void *dst,
                               int ncols_x, int nrows_x, int b_size,
                               int do_fwht, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  __nv_bfloat16 *xb = static_cast<__nv_bfloat16 *>(scratch);
  dim3 prep_grid((ncols_x + ptq1_0::FWHT_BLOCK - 1) / ptq1_0::FWHT_BLOCK,
                 b_size, 1);
  ptq1_0_prepare_bf16_kernel<T><<<prep_grid, PREPARE_THREADS, 0, s>>>(
      static_cast<const T *>(x), static_cast<const float *>(signs),
      static_cast<const uint32_t *>(gather), xb, ncols_x, do_fwht);

  dim3 grid((nrows_x + GEMM_BN - 1) / GEMM_BN, (b_size + GEMM_BM - 1) / GEMM_BM,
            1);
  ptq1_0_gemm_kernel<T><<<grid, GEMM_THREADS, 0, s>>>(
      static_cast<const uint32_t *>(w), xb, static_cast<T *>(dst), ncols_x,
      nrows_x, b_size);
}

// Host-side launchers used by `mistralrs-quant/src/gguf/ffi.rs`.

#define PTQ1_0_GEMM_LAUNCHER(tag, c_type)                                      \
  extern "C" void launch_ptq1_0_gemm_##tag(                                    \
      const void *x, const void *w, const void *signs, const void *gather,     \
      void *scratch, void *dst, int ncols_x, int nrows_x, int b_size,          \
      int do_fwht, void *stream) {                                             \
    ptq1_0_gemm_launch<c_type>(x, w, signs, gather, scratch, dst, ncols_x,     \
                               nrows_x, b_size, do_fwht, stream);              \
  }

PTQ1_0_GEMM_LAUNCHER(f32, float)
PTQ1_0_GEMM_LAUNCHER(f16, __half)
PTQ1_0_GEMM_LAUNCHER(bf16, __nv_bfloat16)
