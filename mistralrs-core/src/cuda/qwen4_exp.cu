// Qwen4-Exp (Qwen3.8-Flash-Next) kernels: hyper-connections, hashed n-gram PLE
// and QSA sparse attention.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define Q4_PAD_SLOT 0xffffffffu
#define Q4_MAX_PLE_NGRAM 8
#define Q4_MAX_PLE_HEADS 64
#define Q4_MAX_CONV_STATE 16
#define Q4_QSA_MAX_INDEX_HEADS 8
#define Q4_QSA_HEAD_DIM 128
#define Q4_ATTN_HEAD_DIM 256
#define Q4_ATTN_WARPS 4
#define Q4_ATTN_MAX_HEADS_PER_WARP 4
#define Q4_ATTN_ITEMS 4
#define Q4_HC_THREADS 128
#define Q4_HC_WARP_SIZE 32
#define Q4_HC_HIDDEN 2560
#define Q4_HC_RANK 320
#define Q4_HC_STREAMS 4
#define Q4_HC_QUANT_GROUP 32
#define Q4_HC_QUANT_THREADS 256

struct Q4HcQ8Input {
  __half2 ds;
  int8_t qs[Q4_HC_QUANT_GROUP];
};

struct Q4HcQ8Weight {
  __half d;
  int8_t qs[Q4_HC_QUANT_GROUP];
};

struct Q4HcQ5Weight {
  __half2 dm;
  uint32_t high;
  int8_t qs[Q4_HC_QUANT_GROUP / 2];
};

template <typename T> __device__ __forceinline__ float q4_f(T v) {
  return (float)v;
}
template <typename T> __device__ __forceinline__ T q4_t(float v);
template <> __device__ __forceinline__ float q4_t<float>(float v) { return v; }
template <> __device__ __forceinline__ __half q4_t<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 q4_t<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ float q4_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float q4_silu(float x) { return x * q4_sigmoid(x); }

__device__ __forceinline__ float q4_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, o);
  }
  return v;
}

// All threads must call; blockDim.x must be a multiple of 32 and <= 1024.
__device__ __forceinline__ float q4_block_sum(float v, float *smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  v = q4_warp_sum(v);
  __syncthreads();
  if (lane == 0) {
    smem[warp] = v;
  }
  __syncthreads();
  v = threadIdx.x < warps ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
    v = q4_warp_sum(v);
    if (lane == 0) {
      smem[0] = v;
    }
  }
  __syncthreads();
  return smem[0];
}

template <typename T> struct Q4Vec8 {
  T v[8];
};

template <typename T>
__device__ __forceinline__ void q4_load8(const T *__restrict__ p, float *out) {
  const Q4Vec8<T> vec = *reinterpret_cast<const Q4Vec8<T> *>(p);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = q4_f(vec.v[i]);
  }
}

template <int FORMAT>
__device__ __forceinline__ float q4_hc_weight(const void *weight, size_t i) {
  if constexpr (FORMAT == 3) {
    const unsigned char *block = (const unsigned char *)weight + (i / 32) * 24;
    const int q = i % 32;
    const unsigned int high = *(const unsigned int *)(block + 4);
    const int value = ((block[8 + q % 16] >> (4 * (q / 16))) & 15) | (((high >> q) & 1) << 4);
    return __half2float(*(const __half *)block) * value + __half2float(*(const __half *)(block + 2));
  } else if constexpr (FORMAT == 2) {
    const unsigned char *block = (const unsigned char *)weight + (i / 32) * 34;
    return __half2float(*(const __half *)block) * (float)((const int8_t *)(block + 2))[i % 32];
  } else if constexpr (FORMAT == 1) {
    return __bfloat162float(((const __nv_bfloat16 *)weight)[i]);
  } else {
    return __half2float(((const __half *)weight)[i]);
  }
}

template <typename T, int FORMAT>
__global__ void q4_hc_up_mix_kernel(const T *__restrict__ lo, const void *up,
                                   const T *__restrict__ xn, T *__restrict__ out,
                                   int hidden, int rank, int hc) {
  const int warp = threadIdx.x / Q4_HC_WARP_SIZE;
  const int lane = threadIdx.x % Q4_HC_WARP_SIZE;
  const int h = blockIdx.x * (Q4_HC_THREADS / Q4_HC_WARP_SIZE) + warp;
  if (h >= hidden) {
    return;
  }
  const int token = blockIdx.y;
  const T *input = lo + (size_t)token * rank;
  float mixed = 0.0f;
  for (int stream = 0; stream < hc; ++stream) {
    const int row = stream * hidden + h;
    float sum = 0.0f;
    for (int k = lane; k < rank; k += Q4_HC_WARP_SIZE) {
      sum = fmaf(q4_f(input[k]), q4_hc_weight<FORMAT>(up, (size_t)row * rank + k), sum);
    }
    sum = q4_warp_sum(sum);
    if (lane == 0) {
      mixed += q4_f(xn[(size_t)token * hc * hidden + row]) * q4_sigmoid(q4_f(q4_t<T>(sum)));
    }
  }
  if (lane == 0) {
    out[(size_t)token * hidden + h] = q4_t<T>(mixed / hc);
  }
}

template <typename T>
__global__ void q4_hc_silu_kernel(const T *__restrict__ input, T *__restrict__ output,
                                 size_t elements, int hc) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < elements) {
    const float scaled = q4_f(q4_t<T>(q4_f(input[i]) / hc));
    output[i] = q4_t<T>(q4_silu(scaled));
  }
}

template <typename T>
__global__ void q4_hc_quantize_kernel(const T *input, Q4HcQ8Input *output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= Q4_HC_RANK) {
    return;
  }
  const int lane = index % Q4_HC_QUANT_GROUP;
  const float value = q4_f(input[index]);
  float amax = fabsf(value);
#pragma unroll
  for (int shift = Q4_HC_WARP_SIZE / 2; shift > 0; shift >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, shift));
  }
  const float sum = q4_warp_sum(value);
  const float scale = amax / 127.0f;
  auto &block = output[index / Q4_HC_QUANT_GROUP];
  block.qs[lane] =
      amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (lane == 0) {
    block.ds = __floats2half2_rn(scale, sum);
  }
}

__device__ __forceinline__ int q4_hc_quant_int(const int8_t *values,
                                               int index) {
  const auto *halves =
      reinterpret_cast<const uint16_t *>(values + sizeof(int) * index);
  return halves[0] | (static_cast<int>(halves[1]) << 16);
}

template <int FORMAT>
__device__ __forceinline__ float
q4_hc_quant_dot(const void *weights, const Q4HcQ8Input *input, int row) {
  constexpr int threads_per_group = FORMAT == 2 ? 4 : 2;
  constexpr int blocks_per_row = Q4_HC_RANK / Q4_HC_QUANT_GROUP;
  const int lane = threadIdx.x % Q4_HC_WARP_SIZE;
  float total = 0.0f;
  for (int block = lane / threads_per_group; block < blocks_per_row;
       block += Q4_HC_WARP_SIZE / threads_per_group) {
    const int part = 2 * (lane % threads_per_group);
    const auto *activation = reinterpret_cast<const int *>(input[block].qs);
    int sum = 0;
    if constexpr (FORMAT == 2) {
      const auto *weight = static_cast<const Q4HcQ8Weight *>(weights) +
                           row * blocks_per_row + block;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        sum = __dp4a(q4_hc_quant_int(weight->qs, part + i),
                     activation[part + i], sum);
      }
      // Match the separately rounded warp partials of the Q8_1 MMVQ reduction.
      const float scaled =
          __fmul_rn(static_cast<float>(sum), __half2float(weight->d));
      total = __fadd_rn(total, __fmul_rn(scaled, __low2float(input[block].ds)));
    } else {
      const auto *weight = static_cast<const Q4HcQ5Weight *>(weights) +
                           row * blocks_per_row + block;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int low_bits =
            reinterpret_cast<const int *>(weight->qs)[part + i];
        const int high_bits = weight->high >> (4 * (part + i));
        const int low = (low_bits & 0x0f0f0f0f) | ((high_bits << 4) & 0x10) |
                        ((high_bits << 11) & 0x1000) |
                        ((high_bits << 18) & 0x100000) |
                        ((high_bits << 25) & 0x10000000);
        const int high =
            ((low_bits >> 4) & 0x0f0f0f0f) | ((high_bits >> 12) & 0x10) |
            ((high_bits >> 5) & 0x1000) | ((high_bits << 2) & 0x100000) |
            ((high_bits << 9) & 0x10000000);
        sum = __dp4a(low, activation[part + i], sum);
        sum = __dp4a(high, activation[part + i + 4], sum);
      }
      const float2 dm = __half22float2(weight->dm);
      const float2 ds = __half22float2(input[block].ds);
      total += sum * (dm.x * ds.x) + (dm.y * ds.y) / 2;
    }
  }
  return q4_warp_sum(total);
}

template <typename T, int FORMAT>
__global__ void q4_hc_quant_up_mix_kernel(const Q4HcQ8Input *input,
                                          const void *weights, const T *xn,
                                          T *output) {
  const int warp = threadIdx.x / Q4_HC_WARP_SIZE;
  const int lane = threadIdx.x % Q4_HC_WARP_SIZE;
  const int hidden = blockIdx.x * (Q4_HC_THREADS / Q4_HC_WARP_SIZE) + warp;
  float value = 0.0f;
#pragma unroll
  for (int stream = 0; stream < Q4_HC_STREAMS; ++stream) {
    const int row = stream * Q4_HC_HIDDEN + hidden;
    const float gate =
        q4_f(q4_t<T>(q4_hc_quant_dot<FORMAT>(weights, input, row)));
    if (lane == 0) {
      value = fmaf(q4_f(xn[row]), q4_sigmoid(gate), value);
    }
  }
  if (lane == 0) {
    output[hidden] = q4_t<T>(value / Q4_HC_STREAMS);
  }
}

// Flattened token -> logical sequence mapping. A null tok_seq is the
// rectangular layout: every sequence has q_len consecutive new tokens (decode
// is q_len 1). tok_seq < 0 marks padding.
struct Q4Tokens {
  const int *tok_seq;
  const int *tok_local;
  const int *seq_start;
  const int *seq_len;
  int n_tokens;
  int n_seqs;
  int q_len;
};

__device__ __forceinline__ int q4_seq_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_seq[t] : t / l.q_len;
}
__device__ __forceinline__ int q4_local_of(const Q4Tokens &l, int t) {
  return l.tok_seq ? l.tok_local[t] : t % l.q_len;
}
__device__ __forceinline__ int q4_start_of(const Q4Tokens &l, int seq) {
  return l.seq_start ? l.seq_start[seq] : seq * l.q_len;
}
__device__ __forceinline__ int q4_len_of(const Q4Tokens &l, int seq) {
  return l.seq_len ? l.seq_len[seq] : l.q_len;
}

template <typename T>
__global__ void
q4_hc_norm_kernel(const T *__restrict__ x, const float *__restrict__ weight,
                  T *__restrict__ out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const int stream = row % hc;
  const T *xr = x + row * hidden;
  const float *wr = weight + (size_t)stream * hidden;
  T *orow = out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(xr[i]);
    ss = fmaf(v, v, ss);
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    orow[i] = q4_t<T>(q4_f(xr[i]) * inv * wr[i]);
  }
}

template <typename T>
__global__ void
q4_hc_mix_kernel(const T *__restrict__ xn, const T *__restrict__ gate,
                 T *__restrict__ out, size_t total, int hc, int hidden) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const size_t t = idx / hidden;
  const int i = idx - t * hidden;
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    const size_t o = (t * hc + c) * hidden + i;
    acc = fmaf(q4_f(xn[o]), q4_sigmoid(q4_f(gate[o])), acc);
  }
  out[idx] = q4_t<T>(acc / (float)hc);
}

// res_out[t, c] = res[t, c] + block_out[t] * 2 * sigmoid(inject[t, c] / hc);
// when next_weight is set, xn_out is the grouped RMSNorm of res_out for the
// next mixer.
template <typename T>
__global__ void
q4_hc_combine_kernel(const T *__restrict__ res, const T *__restrict__ block_out,
                     const T *__restrict__ inject, T *__restrict__ res_out,
                     const float *__restrict__ next_weight,
                     T *__restrict__ xn_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  const size_t row = blockIdx.x;
  const size_t t = row / hc;
  const int stream = row - t * hc;
  const float w = 2.0f * q4_sigmoid(q4_f(inject[row]) / (float)hc);
  const T *rr = res + row * hidden;
  const T *br = block_out + t * hidden;
  T *orow = res_out + row * hidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const T v = q4_t<T>(fmaf(q4_f(br[i]), w, q4_f(rr[i])));
    orow[i] = v;
    const float vf = q4_f(v);
    ss = fmaf(vf, vf, ss);
  }
  if (next_weight == nullptr) {
    return;
  }
  ss = q4_block_sum(ss, smem);
  const float inv = rsqrtf(ss / (float)hidden + eps);
  const float *wr = next_weight + (size_t)stream * hidden;
  T *xr = xn_out + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    xr[i] = q4_t<T>(q4_f(orow[i]) * inv * wr[i]);
  }
}

struct Q4PleHashParams {
  unsigned long long multipliers[Q4_MAX_PLE_NGRAM];
  unsigned long long vocab_sizes[Q4_MAX_PLE_HEADS];
  unsigned long long offsets[Q4_MAX_PLE_HEADS];
  int ngram_size;
  int heads_per_ngram;
  int num_heads;
  unsigned int eos;
};

// hist_pool rows hold the last `hist_len` token ids of each sequence, oldest
// first, stored as id + 1 so a zeroed slot reads as "no token" (EOS).
__global__ void q4_ple_hash_kernel(const unsigned int *__restrict__ tokens,
                                   Q4Tokens l,
                                   const float *__restrict__ hist_pool,
                                   const unsigned int *__restrict__ slots,
                                   int hist_len, Q4PleHashParams p,
                                   long long *__restrict__ rows_out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= l.n_tokens) {
    return;
  }
  long long *out = rows_out + (size_t)t * p.num_heads;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    for (int h = 0; h < p.num_heads; ++h) {
      out[h] = (long long)p.offsets[h];
    }
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  unsigned long long ctx[Q4_MAX_PLE_NGRAM];
  ctx[0] = tokens[t];
  bool cut = false;
  for (int s = 1; s < p.ngram_size; ++s) {
    const int j = local - s;
    bool none = false;
    unsigned int tok = p.eos;
    if (j >= 0) {
      tok = tokens[t - s];
    } else {
      const int h = hist_len + j;
      if (h < 0 || slot == Q4_PAD_SLOT) {
        none = true;
      } else {
        const float stored = hist_pool[(size_t)slot * hist_len + h];
        if (stored <= 0.0f) {
          none = true;
        } else {
          tok = (unsigned int)stored - 1u;
        }
      }
    }
    cut = cut || none || tok == p.eos;
    ctx[s] = cut ? p.eos : tok;
  }
  for (int n = 2; n <= p.ngram_size; ++n) {
    unsigned long long mixed = ctx[0] * p.multipliers[0];
    for (int j = 1; j < n; ++j) {
      mixed ^= ctx[j] * p.multipliers[j];
    }
    const int base = (n - 2) * p.heads_per_ngram;
    for (int g = 0; g < p.heads_per_ngram; ++g) {
      const int h = base + g;
      out[h] = (long long)(mixed % p.vocab_sizes[h] + p.offsets[h]);
    }
  }
}

// New history = last hist_len tokens of (history ++ this chunk), per sequence.
__global__ void
q4_ple_hist_update_kernel(const unsigned int *__restrict__ tokens, Q4Tokens l,
                          float *__restrict__ hist_pool,
                          const unsigned int *__restrict__ slots,
                          int hist_len) {
  const int seq = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq >= l.n_seqs) {
    return;
  }
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  float *h = hist_pool + (size_t)slot * hist_len;
  float next[Q4_MAX_PLE_NGRAM];
  for (int i = 0; i < hist_len; ++i) {
    const int e = len + i;
    next[i] = e < hist_len ? h[e] : (float)tokens[start + e - hist_len] + 1.0f;
  }
  for (int i = 0; i < hist_len; ++i) {
    h[i] = next[i];
  }
}

// One warp per (token, head): copy a 16-bit row of the n-gram table. The shards
// may be host memory the GPU reads through ATS/HMM.
__global__ void
q4_ple_gather_kernel(const long long *__restrict__ rows, int n_lookups,
                     const unsigned long long *__restrict__ shard_ptrs,
                     const long long *__restrict__ shard_starts, int n_shards,
                     int head_dim, unsigned short *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const long long r = rows[lookup];
  int lo = 0;
  int hi = n_shards - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (shard_starts[mid] <= r) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const unsigned short *src =
      reinterpret_cast<const unsigned short *>(shard_ptrs[lo]) +
      (size_t)(r - shard_starts[lo]) * head_dim;
  unsigned short *dst = out + (size_t)lookup * head_dim;
  for (int i = lane; i < head_dim; i += 32) {
    dst[i] = src[i];
  }
}

// Gather from the device-resident quantized table. Q8/Q4: per row, symmetric
// int8 or offset-8 nibbles with one f16 scale per Q4_PLE_GROUP values in
// `scales`. IQ4_NL: ggml block_iq4_nl rows (f16 scale + 16 nibble bytes per
// Q4_PLE_GROUP values, low nibbles first) straight from a GGUF; `scales` unused.
#define Q4_PLE_GROUP 32
#define Q4_PLE_FMT_Q8 0
#define Q4_PLE_FMT_Q4 1
#define Q4_PLE_FMT_IQ4_NL 2
#define Q4_IQ4_NL_BLOCK_BYTES 18
#define Q4_PLE_QUANT_WARPS 8
__constant__ signed char q4_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <typename T>
__global__ void q4_ple_quantize_table_kernel(
    const T *__restrict__ source, unsigned char *__restrict__ data,
    __half *__restrict__ scales, int groups, int bits) {
  const int group = blockIdx.x * Q4_PLE_QUANT_WARPS + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (group >= groups) {
    return;
  }
  const float value = q4_f(source[(size_t)group * Q4_PLE_GROUP + lane]);
  float amax = fabsf(value);
#pragma unroll
  for (int offset = Q4_PLE_GROUP / 2; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  }
  const float qmax = bits == 8 ? 127.0f : 7.0f;
  const __half scale = __float2half_rn(__fdiv_rn(amax, qmax));
  if (lane == 0) {
    scales[group] = scale;
  }
  const float d = __half2float(scale);
  const float inv = d > 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  // Match the host quantizer's ties away from zero and rounded f16 scale.
  const int q = (int)fminf(qmax, fmaxf(-qmax, roundf(__fmul_rn(value, inv))));
  if (bits == 8) {
    data[(size_t)group * Q4_PLE_GROUP + lane] = (unsigned char)(signed char)q;
  } else {
    const int next = __shfl_down_sync(0xffffffff, q, 1);
    if ((lane & 1) == 0) {
      data[(size_t)group * (Q4_PLE_GROUP / 2) + lane / 2] =
          (unsigned char)((q + 8) | ((next + 8) << 4));
    }
  }
}

__global__ void
q4_ple_gather_quant_kernel(const long long *__restrict__ rows, int n_lookups,
                           const unsigned char *__restrict__ data,
                           const __half *__restrict__ scales, int head_dim,
                           int format, __nv_bfloat16 *__restrict__ out) {
  const int lookup = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (lookup >= n_lookups) {
    return;
  }
  const size_t r = (size_t)rows[lookup];
  const int groups = head_dim / Q4_PLE_GROUP;
  __nv_bfloat16 *dst = out + (size_t)lookup * head_dim;
  if (format == Q4_PLE_FMT_IQ4_NL) {
    const unsigned char *src =
        data + r * (size_t)groups * Q4_IQ4_NL_BLOCK_BYTES;
    for (int i = lane; i < head_dim; i += 32) {
      const unsigned char *blk =
          src + (size_t)(i / Q4_PLE_GROUP) * Q4_IQ4_NL_BLOCK_BYTES;
      const int j = i % Q4_PLE_GROUP;
      const unsigned char byte = blk[2 + (j & 15)];
      const int idx = j < 16 ? (byte & 15) : (byte >> 4);
      const float d = __half2float(*reinterpret_cast<const __half *>(blk));
      dst[i] = __float2bfloat16_rn(d * (float)q4_kvalues_iq4nl[idx]);
    }
    return;
  }
  const int bits = format == Q4_PLE_FMT_Q8 ? 8 : 4;
  const size_t row_bytes = (size_t)head_dim * bits / 8;
  const unsigned char *src = data + r * row_bytes;
  const __half *sc = scales + r * groups;
  for (int i = lane; i < head_dim; i += 32) {
    int q;
    if (bits == 8) {
      q = (int)(signed char)src[i];
    } else {
      const unsigned char byte = src[i >> 1];
      q = (int)((i & 1) ? (byte >> 4) : (byte & 15)) - 8;
    }
    dst[i] = __float2bfloat16_rn((float)q * __half2float(sc[i / Q4_PLE_GROUP]));
  }
}

// Per token: gate each hyper-connection stream by the normalized key/query dot
// product, scale the shared value, and RMS-normalize the gated value for the
// conv. One block per token.
template <typename T>
__global__ void
q4_ple_gate_kernel(const T *__restrict__ key,
                   const T *__restrict__ hidden_states,
                   const T *__restrict__ value, const float *__restrict__ w_key,
                   const float *__restrict__ w_query,
                   const float *__restrict__ w_conv, T *__restrict__ gated_out,
                   T *__restrict__ normed_out, int hc, int hidden, float eps) {
  __shared__ float smem[32];
  __shared__ float stats[3 * 8 + 1];
  const size_t t = blockIdx.x;
  const T *kr = key + t * hc * hidden;
  const T *hr = hidden_states + t * hc * hidden;
  const T *vr = value + t * hidden;
  float vv = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = q4_f(vr[i]);
    vv = fmaf(v, v, vv);
  }
  vv = q4_block_sum(vv, smem);
  if (threadIdx.x == 0) {
    stats[3 * 8] = vv;
  }
  for (int c = 0; c < hc; ++c) {
    float kk = 0.0f, qq = 0.0f, kq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const size_t o = (size_t)c * hidden + i;
      const float k = q4_f(kr[o]);
      const float q = q4_f(hr[o]);
      kk = fmaf(k, k, kk);
      qq = fmaf(q, q, qq);
      kq = fmaf(k * w_key[o], q * w_query[o], kq);
    }
    kk = q4_block_sum(kk, smem);
    qq = q4_block_sum(qq, smem);
    kq = q4_block_sum(kq, smem);
    if (threadIdx.x == 0) {
      stats[3 * c] = kk;
      stats[3 * c + 1] = qq;
      stats[3 * c + 2] = kq;
    }
  }
  __syncthreads();
  const float mean_vv = stats[3 * 8] / (float)hidden;
  for (int c = 0; c < hc; ++c) {
    const float inv_k = rsqrtf(stats[3 * c] / (float)hidden + eps);
    const float inv_q = rsqrtf(stats[3 * c + 1] / (float)hidden + eps);
    const float g = stats[3 * c + 2] * inv_k * inv_q * rsqrtf((float)hidden);
    const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
    const float sig = q4_sigmoid(g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
    const float inv_g = rsqrtf(sig * sig * mean_vv + eps);
    T *gr = gated_out + (t * hc + c) * hidden;
    T *nr = normed_out + (t * hc + c) * hidden;
    const float *wc = w_conv + (size_t)c * hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      const float gv = sig * q4_f(vr[i]);
      gr[i] = q4_t<T>(gv);
      nr[i] = q4_t<T>(gv * inv_g * wc[i]);
    }
  }
}

// out = residual + gated + silu(dilated causal depthwise conv(normed)); history
// before the chunk comes from the pooled conv state `[slot, channels,
// state_len]` (oldest first).
template <typename T>
__global__ void
q4_ple_conv_kernel(const T *__restrict__ normed, const T *__restrict__ gated,
                   const T *__restrict__ residual, const T *__restrict__ weight,
                   const T *__restrict__ state_pool,
                   const unsigned int *__restrict__ slots, T *__restrict__ out,
                   Q4Tokens l, int channels, int kernel_size, int dilation,
                   int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_tokens * channels) {
    return;
  }
  const int t = idx / channels;
  const int c = idx - (size_t)t * channels;
  const int seq = q4_seq_of(l, t);
  if (seq < 0) {
    out[idx] = residual[idx];
    return;
  }
  const int local = q4_local_of(l, t);
  const unsigned int slot = slots[seq];
  float acc = 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    const int j = local - (kernel_size - 1 - k) * dilation;
    float x = 0.0f;
    if (j >= 0) {
      x = q4_f(normed[(size_t)(t - (local - j)) * channels + c]);
    } else if (slot != Q4_PAD_SLOT && state_len + j >= 0) {
      x = q4_f(state_pool[((size_t)slot * channels + c) * state_len +
                          state_len + j]);
    }
    acc = fmaf(q4_f(weight[(size_t)c * kernel_size + k]), x, acc);
  }
  out[idx] = q4_t<T>(q4_f(residual[idx]) + q4_f(gated[idx]) + q4_silu(acc));
}

template <typename T>
__global__ void
q4_ple_conv_state_update_kernel(const T *__restrict__ normed,
                                T *__restrict__ state_pool,
                                const unsigned int *__restrict__ slots,
                                Q4Tokens l, int channels, int state_len) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)l.n_seqs * channels) {
    return;
  }
  const int seq = idx / channels;
  const int c = idx - (size_t)seq * channels;
  const unsigned int slot = slots[seq];
  if (slot == Q4_PAD_SLOT) {
    return;
  }
  const int start = q4_start_of(l, seq);
  const int len = q4_len_of(l, seq);
  T *s = state_pool + ((size_t)slot * channels + c) * state_len;
  float next[Q4_MAX_CONV_STATE];
  for (int i = 0; i < state_len; ++i) {
    const int e = len + i;
    next[i] =
        e < state_len
            ? q4_f(s[e])
            : q4_f(normed[(size_t)(start + e - state_len) * channels + c]);
  }
  for (int i = 0; i < state_len; ++i) {
    s[i] = q4_t<T>(next[i]);
  }
}

// QSA aux cache rows are [raw indexer key | cos | sin] per token slot; once a
// block completes, its first token's key becomes the pooled, rotated block key.

struct Q4Paged {
  const int *block_tables;
  const unsigned int *kv_lens;
  int max_blocks_per_seq;
  int block_size;
};

__device__ __forceinline__ long long q4_slot(const Q4Paged &pg, int seq,
                                             int pos) {
  const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                  pos / pg.block_size];
  return (long long)blk * pg.block_size + pos % pg.block_size;
}

// Absolute kv position of flattened token t, or -1 for padding.
__device__ __forceinline__ int q4_pos_of(const Q4Tokens &l, const Q4Paged &pg,
                                         int t, int &seq) {
  seq = q4_seq_of(l, t);
  if (seq < 0) {
    return -1;
  }
  return (int)pg.kv_lens[seq] - q4_len_of(l, seq) + q4_local_of(l, t);
}

template <typename T>
__global__ void q4_qsa_aux_write_kernel(
    const T *__restrict__ raw_key, const T *__restrict__ cos,
    const T *__restrict__ sin, const long long *__restrict__ slot_mapping,
    T *__restrict__ aux, int n_tokens, int key_dim, int half_rot, int aux_dim) {
  const int t = blockIdx.x;
  if (t >= n_tokens) {
    return;
  }
  const long long slot = slot_mapping[t];
  if (slot < 0) {
    return;
  }
  T *dst = aux + slot * aux_dim;
  for (int i = threadIdx.x; i < key_dim; i += blockDim.x) {
    dst[i] = raw_key[(size_t)t * key_dim + i];
  }
  for (int i = threadIdx.x; i < half_rot; i += blockDim.x) {
    dst[key_dim + i] = cos[(size_t)t * half_rot + i];
    dst[key_dim + half_rot + i] = sin[(size_t)t * half_rot + i];
  }
}

// One warp per token; the token that completes a block writes the block key.
// key_dim must be 128.
template <typename T>
__global__ void
q4_qsa_finalize_kernel(const T *__restrict__ aux, T *__restrict__ block_keys,
                       Q4Tokens l, Q4Paged pg,
                       const float *__restrict__ norm_weight, int ratio,
                       int half_rot, int aux_dim, float eps) {
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (t >= l.n_tokens) {
    return;
  }
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  if (pos < 0 || (pos + 1) % ratio != 0) {
    return;
  }
  const int start = pos + 1 - ratio;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int m = 0; m < ratio; ++m) {
    const T *src = aux + q4_slot(pg, seq, start + m) * aux_dim;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] += q4_f(src[lane * 4 + i]);
    }
  }
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // HF pools in f32 and rounds to the checkpoint dtype before the norm
    v[i] = q4_f(q4_t<T>(v[i] / (float)ratio));
    ss = fmaf(v[i], v[i], ss);
  }
  ss = q4_warp_sum(ss);
  const float inv = rsqrtf(ss / (float)Q4_QSA_HEAD_DIM + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = q4_f(q4_t<T>(v[i] * inv * norm_weight[lane * 4 + i]));
  }
  const T *cs = aux + q4_slot(pg, seq, start) * aux_dim + Q4_QSA_HEAD_DIM;
  // Keep raw keys intact when speculative rollback reopens a completed block.
  T *dst = block_keys + q4_slot(pg, seq, start) / ratio * Q4_QSA_HEAD_DIM;
  const T *sn = cs + half_rot;
  float out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = v[i];
  }
  // rotate_half partner is half_rot dims away
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 4 + i;
    const float partner = __shfl_sync(
        0xffffffff, v[i],
        d < half_rot ? lane + half_rot / 4 : (lane - half_rot / 4) & 31);
    if (d < half_rot) {
      out[i] = v[i] * q4_f(cs[d]) - partner * q4_f(sn[d]);
    } else if (d < 2 * half_rot) {
      out[i] = v[i] * q4_f(cs[d - half_rot]) + partner * q4_f(sn[d - half_rot]);
    }
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    dst[lane * 4 + i] = q4_t<T>(out[i]);
  }
}

// scores[t, j] = sum_h relu(q[t, h] . block_key[j]) / sqrt(d) for rows with
// more complete blocks than the budget selects; other rows are skipped.
template <typename T>
__global__ void
q4_qsa_score_kernel(const T *__restrict__ q, const T *__restrict__ block_keys,
                    Q4Tokens l, Q4Paged pg, float *__restrict__ scores,
                    int score_stride, int n_heads, int ratio, int topk,
                    int key_dim) {
  __shared__ float qs[Q4_QSA_MAX_INDEX_HEADS * Q4_QSA_HEAD_DIM];
  const int t = blockIdx.y;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  if (nb <= topk || (int)(blockIdx.x * blockDim.x) >= nb) {
    return;
  }
  for (int i = threadIdx.x; i < n_heads * Q4_QSA_HEAD_DIM; i += blockDim.x) {
    qs[i] = q4_f(q[(size_t)t * n_heads * Q4_QSA_HEAD_DIM + i]);
  }
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nb) {
    return;
  }
  const T *key = block_keys + q4_slot(pg, seq, j * ratio) / ratio * key_dim;
  float dots[Q4_QSA_MAX_INDEX_HEADS];
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    dots[h] = 0.0f;
  }
  for (int d = 0; d < Q4_QSA_HEAD_DIM; d += 8) {
    float k[8];
    q4_load8(key + d, k);
#pragma unroll
    for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
      if (h < n_heads) {
        const float *qh = qs + h * Q4_QSA_HEAD_DIM + d;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          dots[h] = fmaf(qh[i], k[i], dots[h]);
        }
      }
    }
  }
  float total = 0.0f;
#pragma unroll
  for (int h = 0; h < Q4_QSA_MAX_INDEX_HEADS; ++h) {
    if (h < n_heads) {
      total += fmaxf(dots[h], 0.0f);
    }
  }
  scores[(size_t)t * score_stride + j] = total * rsqrtf((float)Q4_QSA_HEAD_DIM);
}

__device__ __forceinline__ unsigned int q4_ordered(float v) {
  if (v != v) {
    return 0u;
  }
  const unsigned int bits = __float_as_uint(v);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Per query row: the indices of the topk largest scores among the first nb
// complete blocks (all of them when nb <= topk). Radix select over the
// order-preserving float bits.
__global__ void q4_qsa_topk_kernel(const float *__restrict__ scores,
                                   int score_stride, Q4Tokens l, Q4Paged pg,
                                   int ratio, int topk,
                                   int *__restrict__ selected,
                                   int *__restrict__ n_selected) {
  __shared__ unsigned int hist[256];
  __shared__ unsigned int s_prefix, s_mask, s_remaining;
  __shared__ unsigned int s_gt, s_eq;
  const int t = blockIdx.x;
  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  const int nb = pos < 0 ? 0 : (pos + 1) / ratio;
  int *out = selected + (size_t)t * topk;
  if (nb <= topk) {
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
      out[i] = i;
    }
    if (threadIdx.x == 0) {
      n_selected[t] = nb;
    }
    return;
  }
  const float *row = scores + (size_t)t * score_stride;
  if (threadIdx.x == 0) {
    s_prefix = 0u;
    s_mask = 0u;
    s_remaining = topk;
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      hist[i] = 0u;
    }
    __syncthreads();
    const unsigned int prefix = s_prefix;
    const unsigned int mask = s_mask;
    for (int j = threadIdx.x; j < nb; j += blockDim.x) {
      const unsigned int key = q4_ordered(row[j]);
      if ((key & mask) == prefix) {
        atomicAdd(&hist[(key >> shift) & 255u], 1u);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int remaining = s_remaining;
      unsigned int above = 0u;
      int digit = 255;
      for (; digit > 0; --digit) {
        if (above + hist[digit] >= remaining) {
          break;
        }
        above += hist[digit];
      }
      s_remaining = remaining - above;
      s_prefix = prefix | ((unsigned int)digit << shift);
      s_mask = mask | (255u << shift);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_gt = 0u;
    s_eq = 0u;
  }
  __syncthreads();
  const unsigned int threshold = s_prefix;
  const unsigned int want_eq = s_remaining;
  const unsigned int n_gt = topk - want_eq;
  for (int j = threadIdx.x; j < nb; j += blockDim.x) {
    const unsigned int key = q4_ordered(row[j]);
    if (key > threshold) {
      out[atomicAdd(&s_gt, 1u)] = j;
    } else if (key == threshold) {
      const unsigned int e = atomicAdd(&s_eq, 1u);
      if (e < want_eq) {
        out[n_gt + e] = j;
      }
    }
  }
  if (threadIdx.x == 0) {
    n_selected[t] = topk;
  }
}

// Sparse GQA attention of each query over its selected blocks plus the
// incomplete tail. Grid: (tokens, kv_heads, splits); each split covers a
// contiguous chunk of the token's item list and writes partial (max, sum, acc)
// when splits > 1. Head dim is fixed at 256: each lane owns 8 dims.
// flashinfer_layout: K and V are [blocks, kv_heads, block_size, 256]; otherwise
// K is [blocks, kv_heads, 256/8, block_size, 8] and V is [blocks, kv_heads,
// 256, block_size].
template <typename T>
__global__ void __launch_bounds__(Q4_ATTN_WARPS * 32) q4_qsa_attention_kernel(
    const T *__restrict__ q, const T *__restrict__ k_cache,
    const T *__restrict__ v_cache, Q4Tokens l, Q4Paged pg,
    const int *__restrict__ selected, const int *__restrict__ n_selected,
    int topk, int ratio, int n_q_heads, int n_kv_heads, int items_per_split,
    float scale, int flashinfer_layout, T *__restrict__ out,
    float *__restrict__ part_acc, float *__restrict__ part_ml) {
  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int split = blockIdx.z;
  const int splits = gridDim.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group = n_q_heads / n_kv_heads;

  int seq;
  const int pos = q4_pos_of(l, pg, t, seq);
  int n_block_items = 0;
  int n_items = 0;
  int tail_start = 0;
  if (pos >= 0) {
    if (selected == nullptr) {
      n_items = pos + 1;
    } else {
      const int nb = (pos + 1) / ratio;
      n_block_items = n_selected[t] * ratio;
      tail_start = nb * ratio;
      n_items = n_block_items + (pos + 1 - tail_start);
    }
  }
  const int begin = split * items_per_split;
  const int end = min(n_items, begin + items_per_split);

  float qv[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float acc[Q4_ATTN_MAX_HEADS_PER_WARP][8];
  float m[Q4_ATTN_MAX_HEADS_PER_WARP];
  float s[Q4_ATTN_MAX_HEADS_PER_WARP];
  int n_heads_here = 0;
#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    const int h = warp + hh * Q4_ATTN_WARPS;
    m[hh] = -FLT_MAX;
    s[hh] = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[hh][i] = 0.0f;
      qv[hh][i] = 0.0f;
    }
    if (h < group) {
      n_heads_here = hh + 1;
      const int qh = g * group + h;
      q4_load8(q + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8,
               qv[hh]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        qv[hh][i] *= scale;
      }
    }
  }

  const size_t head_stride_fi = (size_t)pg.block_size * Q4_ATTN_HEAD_DIM;
  for (int it0 = begin; it0 < end; it0 += Q4_ATTN_ITEMS) {
    // Resolve and load a batch of items up front so the dependent loads overlap
    const T *kp[Q4_ATTN_ITEMS];
    const T *vp[Q4_ATTN_ITEMS];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      const int it = min(it0 + j, end - 1);
      int p;
      if (it < n_block_items) {
        p = selected[(size_t)t * topk + it / ratio] * ratio + it % ratio;
      } else {
        p = tail_start + (it - n_block_items);
      }
      const int blk = pg.block_tables[(size_t)seq * pg.max_blocks_per_seq +
                                      p / pg.block_size];
      const int off = p % pg.block_size;
      const size_t head_base = (size_t)blk * n_kv_heads + g;
      if (flashinfer_layout) {
        const size_t base = head_base * head_stride_fi +
                            (size_t)off * Q4_ATTN_HEAD_DIM + lane * 8;
        kp[j] = k_cache + base;
        vp[j] = v_cache + base;
      } else {
        kp[j] =
            k_cache +
            (head_base * (Q4_ATTN_HEAD_DIM / 8) + lane) * pg.block_size * 8 +
            (size_t)off * 8;
        vp[j] = v_cache +
                (head_base * Q4_ATTN_HEAD_DIM + lane * 8) * pg.block_size + off;
      }
    }
    float kf[Q4_ATTN_ITEMS][8];
    float vf[Q4_ATTN_ITEMS][8];
#pragma unroll
    for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
      q4_load8(kp[j], kf[j]);
      if (flashinfer_layout) {
        q4_load8(vp[j], vf[j]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          vf[j][i] = q4_f(vp[j][(size_t)i * pg.block_size]);
        }
      }
    }
#pragma unroll
    for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
      if (hh < n_heads_here) {
        float dot[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          dot[j] = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dot[j] = fmaf(qv[hh][i], kf[j][i], dot[j]);
          }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            dot[j] += __shfl_xor_sync(0xffffffff, dot[j], o);
          }
        }
        float m_new = m[hh];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          if (it0 + j < end) {
            m_new = fmaxf(m_new, dot[j]);
          }
        }
        const float corr = __expf(m[hh] - m_new);
        float psum = 0.0f;
        float pj[Q4_ATTN_ITEMS];
#pragma unroll
        for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
          pj[j] = it0 + j < end ? __expf(dot[j] - m_new) : 0.0f;
          psum += pj[j];
        }
        s[hh] = s[hh] * corr + psum;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float a = acc[hh][i] * corr;
#pragma unroll
          for (int j = 0; j < Q4_ATTN_ITEMS; ++j) {
            a = fmaf(pj[j], vf[j][i], a);
          }
          acc[hh][i] = a;
        }
        m[hh] = m_new;
      }
    }
  }

#pragma unroll
  for (int hh = 0; hh < Q4_ATTN_MAX_HEADS_PER_WARP; ++hh) {
    if (hh >= n_heads_here) {
      continue;
    }
    const int qh = g * (n_q_heads / n_kv_heads) + warp + hh * Q4_ATTN_WARPS;
    if (splits == 1) {
      const float inv = s[hh] > 0.0f ? 1.0f / s[hh] : 0.0f;
      T *o = out + ((size_t)t * n_q_heads + qh) * Q4_ATTN_HEAD_DIM + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        o[i] = q4_t<T>(acc[hh][i] * inv);
      }
    } else {
      const size_t row = ((size_t)t * n_q_heads + qh) * splits + split;
      float *a = part_acc + row * Q4_ATTN_HEAD_DIM + lane * 8;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        a[i] = acc[hh][i];
      }
      if (lane == 0) {
        part_ml[row * 2] = m[hh];
        part_ml[row * 2 + 1] = s[hh];
      }
    }
  }
}

template <typename T>
__global__ void
q4_qsa_attention_reduce_kernel(const float *__restrict__ part_acc,
                               const float *__restrict__ part_ml, int splits,
                               T *__restrict__ out) {
  const size_t row = blockIdx.x;
  const int d = threadIdx.x;
  float m = -FLT_MAX;
  for (int sp = 0; sp < splits; ++sp) {
    const float ms = part_ml[(row * splits + sp) * 2];
    if (part_ml[(row * splits + sp) * 2 + 1] > 0.0f) {
      m = fmaxf(m, ms);
    }
  }
  float total = 0.0f;
  float acc = 0.0f;
  for (int sp = 0; sp < splits; ++sp) {
    const float ss = part_ml[(row * splits + sp) * 2 + 1];
    if (ss <= 0.0f) {
      continue;
    }
    const float w = __expf(part_ml[(row * splits + sp) * 2] - m);
    total = fmaf(ss, w, total);
    acc = fmaf(part_acc[(row * splits + sp) * Q4_ATTN_HEAD_DIM + d], w, acc);
  }
  out[row * Q4_ATTN_HEAD_DIM + d] = q4_t<T>(total > 0.0f ? acc / total : 0.0f);
}

// Launcher dtype codes: 0 = f16, 1 = bf16, 2 = f32.

#define Q4_DISPATCH(dtype, ...)                                                \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else if ((dtype) == 1) {                                                 \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = float;                                                  \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

#define Q4_DISPATCH_HALF(dtype, ...)                                           \
  do {                                                                         \
    if ((dtype) == 0) {                                                        \
      using scalar_t = __half;                                                 \
      __VA_ARGS__;                                                             \
    } else {                                                                   \
      using scalar_t = __nv_bfloat16;                                          \
      __VA_ARGS__;                                                             \
    }                                                                          \
  } while (0)

static inline int q4_row_threads(int hidden) {
  int threads = 32;
  while (threads < 512 && threads * 4 < hidden) {
    threads <<= 1;
  }
  return threads;
}

extern "C" void qwen4_hc_up_mix(const void *lo, const void *up, const void *xn,
                                void *out, int tokens, int hidden, int rank,
                                int hc, int format, int dtype, int64_t stream) {
  const cudaStream_t s = (cudaStream_t)stream;
  const dim3 grid((hidden + Q4_HC_THREADS / Q4_HC_WARP_SIZE - 1) /
                      (Q4_HC_THREADS / Q4_HC_WARP_SIZE), tokens);
#define Q4_HC_UP(FORMAT)                                                        \
  q4_hc_up_mix_kernel<scalar_t, FORMAT><<<grid, Q4_HC_THREADS, 0, s>>>(            \
      (const scalar_t *)lo, up, (const scalar_t *)xn, (scalar_t *)out, hidden,    \
      rank, hc)
  Q4_DISPATCH_HALF(dtype, {
    if (format == 3) {
      Q4_HC_UP(3);
    } else if (format == 2) {
      Q4_HC_UP(2);
    } else if (format == 1) {
      Q4_HC_UP(1);
    } else {
      Q4_HC_UP(0);
    }
  });
#undef Q4_HC_UP
}

extern "C" void qwen4_hc_quant_up_mix(const void *lo, const void *up,
                                      const void *xn, void *scratch, void *out,
                                      int format, int dtype, int64_t stream) {
  const cudaStream_t s =
      static_cast<cudaStream_t>(reinterpret_cast<void *>(stream));
  auto *quantized = static_cast<Q4HcQ8Input *>(scratch);
  const int blocks =
      (Q4_HC_RANK + Q4_HC_QUANT_THREADS - 1) / Q4_HC_QUANT_THREADS;
  const int output_blocks = Q4_HC_HIDDEN / (Q4_HC_THREADS / Q4_HC_WARP_SIZE);
  Q4_DISPATCH_HALF(dtype, {
    q4_hc_quantize_kernel<scalar_t><<<blocks, Q4_HC_QUANT_THREADS, 0, s>>>(
        static_cast<const scalar_t *>(lo), quantized);
    if (format == 2) {
      q4_hc_quant_up_mix_kernel<scalar_t, 2>
          <<<output_blocks, Q4_HC_THREADS, 0, s>>>(
              quantized, up, static_cast<const scalar_t *>(xn),
              static_cast<scalar_t *>(out));
    } else {
      q4_hc_quant_up_mix_kernel<scalar_t, 3>
          <<<output_blocks, Q4_HC_THREADS, 0, s>>>(
              quantized, up, static_cast<const scalar_t *>(xn),
              static_cast<scalar_t *>(out));
    }
  });
}

extern "C" void qwen4_hc_silu(const void *input, void *output, size_t elements,
                              int hc, int dtype, int64_t stream) {
  const int threads = 256;
  Q4_DISPATCH(dtype, q4_hc_silu_kernel<scalar_t>
              <<<(elements + threads - 1) / threads, threads, 0, (cudaStream_t)stream>>>(
                  (const scalar_t *)input, (scalar_t *)output, elements, hc));
}

extern "C" void qwen4_hc_norm(const void *x, const float *weight, void *out,
                              int rows, int hc, int hidden, float eps,
                              int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_norm_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>((const scalar_t *)x,
                                                       weight, (scalar_t *)out,
                                                       hc, hidden, eps));
}

extern "C" void qwen4_hc_mix(const void *xn, const void *gate, void *out,
                             int tokens, int hc, int hidden, int dtype,
                             int64_t stream) {
  const size_t total = (size_t)tokens * hidden;
  if (total == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const unsigned int blocks = (total + threads - 1) / threads;
  Q4_DISPATCH(dtype, q4_hc_mix_kernel<scalar_t><<<blocks, threads, 0, s>>>(
                         (const scalar_t *)xn, (const scalar_t *)gate,
                         (scalar_t *)out, total, hc, hidden));
}

extern "C" void qwen4_hc_combine(const void *res, const void *block_out,
                                 const void *inject, void *res_out,
                                 const float *next_weight, void *xn_out,
                                 int rows, int hc, int hidden, float eps,
                                 int dtype, int64_t stream) {
  if (rows == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_hc_combine_kernel<scalar_t>
              <<<rows, q4_row_threads(hidden), 0, s>>>(
                  (const scalar_t *)res, (const scalar_t *)block_out,
                  (const scalar_t *)inject, (scalar_t *)res_out, next_weight,
                  (scalar_t *)xn_out, hc, hidden, eps));
}

extern "C" void qwen4_ple_hash(const unsigned int *tokens, Q4Tokens layout,
                               const float *hist_pool,
                               const unsigned int *slots, int hist_len,
                               const Q4PleHashParams *params,
                               long long *rows_out, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 128;
  q4_ple_hash_kernel<<<(layout.n_tokens + threads - 1) / threads, threads, 0,
                       s>>>(tokens, layout, hist_pool, slots, hist_len, *params,
                            rows_out);
}

extern "C" void qwen4_ple_hist_update(const unsigned int *tokens,
                                      Q4Tokens layout, float *hist_pool,
                                      const unsigned int *slots, int hist_len,
                                      int64_t stream) {
  if (layout.n_seqs == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  q4_ple_hist_update_kernel<<<(layout.n_seqs + 63) / 64, 64, 0, s>>>(
      tokens, layout, hist_pool, slots, hist_len);
}

extern "C" void qwen4_ple_gather(const long long *rows, int n_lookups,
                                 const unsigned long long *shard_ptrs,
                                 const long long *shard_starts, int n_shards,
                                 int head_dim, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0, s>>>(
      rows, n_lookups, shard_ptrs, shard_starts, n_shards, head_dim,
      (unsigned short *)out);
}

extern "C" void qwen4_ple_gather_quant(const long long *rows, int n_lookups,
                                       const unsigned char *data,
                                       const void *scales, int head_dim,
                                       int format, void *out, int64_t stream) {
  if (n_lookups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  q4_ple_gather_quant_kernel<<<(n_lookups + warps - 1) / warps, warps * 32, 0,
                               s>>>(rows, n_lookups, data,
                                    (const __half *)scales, head_dim, format,
                                    (__nv_bfloat16 *)out);
}

extern "C" void qwen4_ple_quantize_table(const void *source, unsigned char *data,
                                         void *scales, int groups, int bits,
                                         int dtype, int64_t stream) {
  if (groups == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int blocks = (groups + Q4_PLE_QUANT_WARPS - 1) / Q4_PLE_QUANT_WARPS;
  Q4_DISPATCH(dtype, q4_ple_quantize_table_kernel<scalar_t>
              <<<blocks, Q4_PLE_QUANT_WARPS * Q4_PLE_GROUP, 0, s>>>(
                  (const scalar_t *)source, data, (__half *)scales, groups, bits));
}

extern "C" void qwen4_ple_gate(const void *key, const void *hidden_states,
                               const void *value, const float *w_key,
                               const float *w_query, const float *w_conv,
                               void *gated_out, void *normed_out, int tokens,
                               int hc, int hidden, float eps, int dtype,
                               int64_t stream) {
  if (tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH(dtype, q4_ple_gate_kernel<scalar_t><<<tokens, 512, 0, s>>>(
                         (const scalar_t *)key, (const scalar_t *)hidden_states,
                         (const scalar_t *)value, w_key, w_query, w_conv,
                         (scalar_t *)gated_out, (scalar_t *)normed_out, hc,
                         hidden, eps));
}

extern "C" void qwen4_ple_conv(const void *normed, const void *gated,
                               const void *residual, const void *weight,
                               void *state_pool, const unsigned int *slots,
                               void *out, Q4Tokens layout, int channels,
                               int kernel_size, int dilation, int state_len,
                               int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int threads = 256;
  const size_t total = (size_t)layout.n_tokens * channels;
  const unsigned int blocks = (total + threads - 1) / threads;
  const size_t state_total = (size_t)layout.n_seqs * channels;
  const unsigned int state_blocks = (state_total + threads - 1) / threads;
  Q4_DISPATCH(dtype, {
    q4_ple_conv_kernel<scalar_t><<<blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (const scalar_t *)gated,
        (const scalar_t *)residual, (const scalar_t *)weight,
        (const scalar_t *)state_pool, slots, (scalar_t *)out, layout, channels,
        kernel_size, dilation, state_len);
    q4_ple_conv_state_update_kernel<scalar_t><<<state_blocks, threads, 0, s>>>(
        (const scalar_t *)normed, (scalar_t *)state_pool, slots, layout,
        channels, state_len);
  });
}

extern "C" void qwen4_qsa_aux_write(const void *raw_key, const void *cos,
                                    const void *sin,
                                    const long long *slot_mapping, void *aux,
                                    int n_tokens, int key_dim, int half_rot,
                                    int aux_dim, int dtype, int64_t stream) {
  if (n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  Q4_DISPATCH_HALF(dtype,
                   q4_qsa_aux_write_kernel<scalar_t><<<n_tokens, 128, 0, s>>>(
                       (const scalar_t *)raw_key, (const scalar_t *)cos,
                       (const scalar_t *)sin, slot_mapping, (scalar_t *)aux,
                       n_tokens, key_dim, half_rot, aux_dim));
}

extern "C" void qwen4_qsa_finalize(const void *aux, void *block_keys,
                                   Q4Tokens layout, Q4Paged paged,
                                   const float *norm_weight, int ratio,
                                   int half_rot, int aux_dim, float eps,
                                   int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const int warps = 8;
  Q4_DISPATCH_HALF(
      dtype, q4_qsa_finalize_kernel<scalar_t>
      <<<(layout.n_tokens + warps - 1) / warps, warps * 32, 0, s>>>(
          (const scalar_t *)aux, (scalar_t *)block_keys, layout, paged,
          norm_weight, ratio, half_rot, aux_dim, eps));
}

extern "C" void qwen4_qsa_select(const void *q, const void *block_keys,
                                 Q4Tokens layout, Q4Paged paged, float *scores,
                                 int score_stride, int n_heads, int ratio,
                                 int topk, int key_dim, int *selected,
                                 int *n_selected, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  if (score_stride > topk) {
    const int threads = 128;
    const dim3 grid((score_stride + threads - 1) / threads, layout.n_tokens);
    Q4_DISPATCH_HALF(
        dtype, q4_qsa_score_kernel<scalar_t><<<grid, threads, 0, s>>>(
                   (const scalar_t *)q, (const scalar_t *)block_keys, layout,
                   paged, scores, score_stride, n_heads, ratio, topk, key_dim));
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, s>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

// Top-k over scores produced elsewhere (the cuTile scoring kernel).
extern "C" void qwen4_qsa_topk(const float *scores, int score_stride,
                               Q4Tokens layout, Q4Paged paged, int ratio,
                               int topk, int *selected, int *n_selected,
                               int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  q4_qsa_topk_kernel<<<layout.n_tokens, 1024, 0, (cudaStream_t)stream>>>(
      scores, score_stride, layout, paged, ratio, topk, selected, n_selected);
}

extern "C" void qwen4_qsa_attention(
    const void *q, const void *k_cache, const void *v_cache, Q4Tokens layout,
    Q4Paged paged, const int *selected, const int *n_selected, int topk,
    int ratio, int n_q_heads, int n_kv_heads, int splits, int items_per_split,
    float scale, int flashinfer_layout, void *out, float *part_acc,
    float *part_ml, int dtype, int64_t stream) {
  if (layout.n_tokens == 0) {
    return;
  }
  const cudaStream_t s = (cudaStream_t)stream;
  const dim3 grid(layout.n_tokens, n_kv_heads, splits);
  Q4_DISPATCH_HALF(dtype, {
    q4_qsa_attention_kernel<scalar_t><<<grid, Q4_ATTN_WARPS * 32, 0, s>>>(
        (const scalar_t *)q, (const scalar_t *)k_cache,
        (const scalar_t *)v_cache, layout, paged, selected, n_selected, topk,
        ratio, n_q_heads, n_kv_heads, items_per_split, scale, flashinfer_layout,
        (scalar_t *)out, part_acc, part_ml);
    if (splits > 1) {
      q4_qsa_attention_reduce_kernel<scalar_t>
          <<<layout.n_tokens * n_q_heads, Q4_ATTN_HEAD_DIM, 0, s>>>(
              part_acc, part_ml, splits, (scalar_t *)out);
    }
  });
}
