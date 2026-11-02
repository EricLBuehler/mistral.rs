#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../common/fp_codecs.cuh"

namespace {
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kUnit = 8;
constexpr int kShardFields = 3;
constexpr int kQ8KSuperBlock = 256;

// Codec ids mirror `RowCodec::id` in src/embedding/mod.rs.
enum Codec : int {
  DENSE_F32 = 0,
  DENSE_F16 = 1,
  DENSE_BF16 = 2,
  Q4_0 = 3,
  Q4_1 = 4,
  Q5_0 = 5,
  Q5_1 = 6,
  Q8_0 = 7,
  Q8_1 = 8,
  Q2K = 9,
  Q3K = 10,
  Q4K = 11,
  Q5K = 12,
  Q6K = 13,
  Q8K = 14,
  IQ4_NL = 15,
  FP4 = 16,
  FP8 = 17,
  AFQ = 18,
  SYM_INT4 = 19,
  SYM_INT8 = 20,
  MXFP4 = 21,
};

// Scalar encodings mirror `ScaleType` in src/embedding/mod.rs.
enum ScaleType : int {
  SCALE_F32 = 0,
  SCALE_F16 = 1,
  SCALE_BF16 = 2,
  SCALE_E4M3 = 3,
  SCALE_E8M0 = 4,
  SCALE_E8M0_RAW = 5,
};

enum RowScaleMode : int { ROW_SCALE_NONE = 0, ROW_SCALE_SCALAR = 1, ROW_SCALE_PER_ROW = 2 };

struct GatherArgs {
  const void *ids;
  const uint8_t *data;
  const uint8_t *scales;
  const uint8_t *aux;
  const uint64_t *shards;
  const void *row_scales;
  void *out;
  uint64_t lookups;
  uint64_t rows;
  uint64_t columns;
  uint64_t data_stride;
  uint64_t scale_stride;
  uint64_t scale_row_div;
  uint64_t group;
  float out_scale;
  int32_t ids_i64;
  int32_t shard_count;
  int32_t codec;
  int32_t scale_type;
  int32_t row_scale_mode;
  int32_t row_scale_type;
  int32_t bits;
  int32_t out_dtype;
};

__constant__ int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                1,    13,   25,  38,  53,  69,  89,  113};

struct Row {
  const uint8_t *data;
  const uint8_t *scales;
  const uint8_t *aux;
  float mul;
};

__device__ __forceinline__ float half_at(const uint8_t *p) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(p[0] | (p[1] << 8))));
}

__device__ __forceinline__ float float_at(const uint8_t *p) {
  return __uint_as_float(uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                         (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24));
}

__device__ __forceinline__ float load_scalar(const void *base, uint64_t index, int type) {
  const auto *p = static_cast<const uint8_t *>(base);
  switch (type) {
  case SCALE_F32:
    return reinterpret_cast<const float *>(p)[index];
  case SCALE_F16:
    return __half2float(reinterpret_cast<const __half *>(p)[index]);
  case SCALE_BF16:
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(p)[index]);
  case SCALE_E4M3:
    return e4m3_to_float(p[index]);
  case SCALE_E8M0:
    return e8m0_to_float_ocp(p[index]);
  default:
    return e8m0_to_float(p[index]);
  }
}

__device__ __forceinline__ void k4_scale_min(int j, const uint8_t *q, int &d, int &m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Decodes kUnit consecutive columns starting at `col`, a multiple of kUnit.
template <int C>
__device__ __forceinline__ void decode(const GatherArgs &a, const Row &r, uint64_t col,
                                       float v[kUnit]) {
  if constexpr (C == DENSE_F32) {
    const auto *p = reinterpret_cast<const float4 *>(r.data + col * 4);
    const float4 x = p[0], y = p[1];
    v[0] = x.x, v[1] = x.y, v[2] = x.z, v[3] = x.w;
    v[4] = y.x, v[5] = y.y, v[6] = y.z, v[7] = y.w;
  } else if constexpr (C == DENSE_F16 || C == DENSE_BF16) {
    const uint4 x = *reinterpret_cast<const uint4 *>(r.data + col * 2);
    const auto *h = reinterpret_cast<const unsigned short *>(&x);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = C == DENSE_F16 ? __half2float(__ushort_as_half(h[i]))
                            : __bfloat162float(__ushort_as_bfloat16(h[i]));
    }
  } else if constexpr (C == Q4_0 || C == Q4_1 || C == IQ4_NL) {
    constexpr int bytes = C == Q4_1 ? 20 : 18;
    constexpr int qs = C == Q4_1 ? 4 : 2;
    const uint8_t *b = r.data + (col / 32) * bytes;
    const int j = col % 32;
    const float d = half_at(b);
    const float m = C == Q4_1 ? half_at(b + 2) : 0.0f;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int k = j + i;
      const int q = (b[qs + (k % 16)] >> (k < 16 ? 0 : 4)) & 15;
      if constexpr (C == Q4_0)
        v[i] = (q - 8) * d;
      else if constexpr (C == Q4_1)
        v[i] = q * d + m;
      else
        v[i] = kIq4[q] * d;
    }
  } else if constexpr (C == Q5_0 || C == Q5_1) {
    constexpr int bytes = C == Q5_1 ? 24 : 22;
    constexpr int qh_at = C == Q5_1 ? 4 : 2;
    const uint8_t *b = r.data + (col / 32) * bytes;
    const int j = col % 32;
    const float d = half_at(b);
    const float m = C == Q5_1 ? half_at(b + 2) : 0.0f;
    const uint32_t qh = b[qh_at] | (b[qh_at + 1] << 8) | (b[qh_at + 2] << 16) |
                        (static_cast<uint32_t>(b[qh_at + 3]) << 24);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int k = j + i;
      const int q = ((b[qh_at + 4 + (k % 16)] >> (k < 16 ? 0 : 4)) & 15) | (((qh >> k) & 1) << 4);
      v[i] = C == Q5_0 ? (q - 16) * d : q * d + m;
    }
  } else if constexpr (C == Q8_0 || C == Q8_1) {
    constexpr int bytes = C == Q8_1 ? 36 : 34;
    constexpr int qs = C == Q8_1 ? 4 : 2;
    const uint8_t *b = r.data + (col / 32) * bytes;
    const auto *q = reinterpret_cast<const int8_t *>(b + qs) + col % 32;
    const float d = half_at(b);
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = q[i] * d;
  } else if constexpr (C == Q8K) {
    const uint8_t *b = r.data + (col / kQ8KSuperBlock) * 292;
    const float d = float_at(b);
    const auto *q = reinterpret_cast<const int8_t *>(b + 4) + col % kQ8KSuperBlock;
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = q[i] * d;
  } else if constexpr (C == Q2K) {
    const uint8_t *b = r.data + (col / 256) * 84;
    const int c = col % 256, n = c / 128, rem = c % 128, j = rem / 32, half = (rem % 32) / 16;
    const int sc = b[8 * n + 2 * j + half];
    const float d = half_at(b + 80), dmin = half_at(b + 82);
    const float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
    const uint8_t *q = b + 16 + 32 * n + 16 * half + rem % 16;
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = dl * ((q[i] >> (2 * j)) & 3) - ml;
  } else if constexpr (C == Q3K) {
    const uint8_t *b = r.data + (col / 256) * 110;
    const int c = col % 256, n = c / 128, rem = c % 128, j = rem / 32, half = (rem % 32) / 16;
    const int is = 8 * n + 2 * j + half;
    const uint8_t *sb = b + 96;
    const int low = is < 8 ? (sb[is] & 0xF) : (sb[is - 8] >> 4);
    const int high = (sb[8 + is % 4] >> (2 * (is / 4))) & 3;
    const float dl = half_at(b + 108) * ((low | (high << 4)) - 32);
    const int l = rem % 16;
    const uint8_t *q = b + 32 + 32 * n + 16 * half + l;
    const uint8_t *hm = b + 16 * half + l;
    const int m = 1 << (4 * n + j);
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = dl * (((q[i] >> (2 * j)) & 3) - ((hm[i] & m) ? 0 : 4));
  } else if constexpr (C == Q4K || C == Q5K) {
    constexpr int bytes = C == Q4K ? 144 : 176;
    constexpr int qs = C == Q4K ? 16 : 48;
    const uint8_t *b = r.data + (col / 256) * bytes;
    const int c = col % 256, chunk = c / 64, hi = (c % 64) / 32, l = c % 32;
    int sc, mn;
    k4_scale_min(2 * chunk + hi, b + 4, sc, mn);
    const float d = half_at(b) * sc, m = half_at(b + 2) * mn;
    const uint8_t *q = b + qs + 32 * chunk + l;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      int x = hi ? (q[i] >> 4) : (q[i] & 0xF);
      if constexpr (C == Q5K)
        x += ((b[16 + l + i] >> (2 * chunk + hi)) & 1) << 4;
      v[i] = d * x - m;
    }
  } else if constexpr (C == Q6K) {
    const uint8_t *b = r.data + (col / 256) * 210;
    const int c = col % 256, n = c / 128, quarter = (c % 128) / 32, l = c % 32;
    const auto *sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * n;
    const float d = half_at(b + 208) * sc[l / 16 + 2 * quarter];
    const uint8_t *ql = b + 64 * n + l + 32 * (quarter & 1);
    const uint8_t *qh = b + 128 + 32 * n + l;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int lo = quarter >= 2 ? (ql[i] >> 4) : (ql[i] & 0xF);
      v[i] = d * ((lo | (((qh[i] >> (2 * quarter)) & 3) << 4)) - 32);
    }
  } else if constexpr (C == MXFP4) {
    const uint8_t *b = r.data + (col / 32) * 17;
    const uint32_t e = b[0];
    const float d = __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int c = col % 32 + i;
      const int code = (b[1 + c % 16] >> (c < 16 ? 0 : 4)) & 15;
      v[i] = (2.0f * e2m1_to_float(code)) * d;
    }
  } else if constexpr (C == FP4) {
    const uint8_t *q = r.data + col / 2;
    const float s = load_scalar(r.scales, col / a.group, a.scale_type);
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = __fmul_rn(e2m1_to_float((q[i / 2] >> (4 * (i % 2))) & 15), s);
  } else if constexpr (C == FP8) {
    const uint8_t *q = r.data + col;
    const float s = r.scales ? load_scalar(r.scales, col / a.group, a.scale_type) : 1.0f;
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] = e4m3_to_float(q[i]) * s;
  } else if constexpr (C == AFQ) {
    const uint64_t g = col / a.group;
    const float s = load_scalar(r.scales, g, a.scale_type);
    const float z = load_scalar(r.aux, g, a.scale_type);
    const int bits = a.bits;
    const uint32_t mask = (1u << bits) - 1;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const uint64_t bit = (col + i) * bits;
      const uint8_t *p = r.data + bit / 8;
      const uint32_t word = p[0] | (bit % 8 + bits > 8 ? p[1] << 8 : 0);
      v[i] = fmaf(static_cast<float>((word >> (bit % 8)) & mask), s, z);
    }
  } else if constexpr (C == SYM_INT4 || C == SYM_INT8) {
    const float s = load_scalar(r.scales, col / a.group, a.scale_type);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      int q;
      if constexpr (C == SYM_INT8) {
        q = reinterpret_cast<const int8_t *>(r.data)[col + i];
      } else {
        const uint8_t byte = r.data[(col + i) / 2];
        q = static_cast<int>((i & 1) ? (byte >> 4) : (byte & 15)) - 8;
      }
      v[i] = q * s;
    }
  }
}

template <typename T> __device__ __forceinline__ T cast_out(float x);
template <> __device__ __forceinline__ float cast_out(float x) { return x; }
template <> __device__ __forceinline__ __half cast_out(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_out(float x) {
  return __float2bfloat16_rn(x);
}

template <typename T> __device__ __forceinline__ void store(T *dst, const float v[kUnit]) {
  if constexpr (sizeof(T) == 4) {
    auto *p = reinterpret_cast<float4 *>(dst);
    p[0] = make_float4(v[0], v[1], v[2], v[3]);
    p[1] = make_float4(v[4], v[5], v[6], v[7]);
  } else {
    uint4 packed;
    T *h = reinterpret_cast<T *>(&packed);
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      h[i] = cast_out<T>(v[i]);
    *reinterpret_cast<uint4 *>(dst) = packed;
  }
}

__device__ __forceinline__ Row locate(const GatherArgs &a, uint64_t row) {
  const uint8_t *data = a.data;
  const uint8_t *scales = a.scales;
  uint64_t local = row;
  if (a.shard_count > 0) {
    int lo = 0, hi = a.shard_count - 1;
    while (lo < hi) {
      const int mid = (lo + hi + 1) / 2;
      if (a.shards[mid * kShardFields + 2] <= row)
        lo = mid;
      else
        hi = mid - 1;
    }
    data = reinterpret_cast<const uint8_t *>(a.shards[lo * kShardFields]);
    scales = reinterpret_cast<const uint8_t *>(a.shards[lo * kShardFields + 1]);
    local = row - a.shards[lo * kShardFields + 2];
  }
  Row r;
  r.data = data + local * a.data_stride;
  r.scales = scales ? scales + (local / a.scale_row_div) * a.scale_stride : nullptr;
  r.aux = a.aux ? a.aux + local * a.scale_stride : nullptr;
  float mul = a.out_scale;
  if (a.row_scale_mode == ROW_SCALE_SCALAR)
    mul *= load_scalar(a.row_scales, 0, a.row_scale_type);
  else if (a.row_scale_mode == ROW_SCALE_PER_ROW)
    mul *= load_scalar(a.row_scales, row, a.row_scale_type);
  r.mul = mul;
  return r;
}

// One lookup per `threads_per_row` threads; the host guarantees columns % kUnit == 0 and 16-byte rows.
template <int C, typename T>
__global__ void gather_rows(GatherArgs a, int threads_per_row) {
  const int rows_per_block = kThreads / threads_per_row;
  const uint64_t lookup =
      static_cast<uint64_t>(blockIdx.x) * rows_per_block + threadIdx.x / threads_per_row;
  if (lookup >= a.lookups)
    return;
  const int lane = threadIdx.x % threads_per_row;
  const uint64_t row = a.ids_i64 ? static_cast<const int64_t *>(a.ids)[lookup]
                                 : static_cast<const uint32_t *>(a.ids)[lookup];
  T *out = static_cast<T *>(a.out) + lookup * a.columns;
  const uint64_t units = a.columns / kUnit;
  if (row >= a.rows) {
    for (uint64_t c = lane; c < a.columns; c += threads_per_row)
      out[c] = cast_out<T>(0.0f);
    return;
  }
  const Row r = locate(a, row);
  for (uint64_t u = lane; u < units; u += threads_per_row) {
    float v[kUnit];
    decode<C>(a, r, u * kUnit, v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i)
      v[i] *= r.mul;
    store<T>(out + u * kUnit, v);
  }
}

// Dense and FP8 rows whose width or alignment rules out kUnit vectors decode one element per thread.
template <int C, typename T> __global__ void gather_rows_scalar(GatherArgs a, int threads_per_row) {
  const int rows_per_block = kThreads / threads_per_row;
  const uint64_t lookup =
      static_cast<uint64_t>(blockIdx.x) * rows_per_block + threadIdx.x / threads_per_row;
  if (lookup >= a.lookups)
    return;
  const int lane = threadIdx.x % threads_per_row;
  const uint64_t row = a.ids_i64 ? static_cast<const int64_t *>(a.ids)[lookup]
                                 : static_cast<const uint32_t *>(a.ids)[lookup];
  T *out = static_cast<T *>(a.out) + lookup * a.columns;
  if (row >= a.rows) {
    for (uint64_t c = lane; c < a.columns; c += threads_per_row)
      out[c] = cast_out<T>(0.0f);
    return;
  }
  const Row r = locate(a, row);
  for (uint64_t c = lane; c < a.columns; c += threads_per_row) {
    float x;
    if constexpr (C == FP8) {
      const float s = r.scales ? load_scalar(r.scales, c / a.group, a.scale_type) : 1.0f;
      x = e4m3_to_float(r.data[c]) * s;
    } else {
      x = load_scalar(r.data, c, C - DENSE_F32);
    }
    out[c] = cast_out<T>(x * r.mul);
  }
}

int threads_per_row(uint64_t work) {
  int t = kWarp;
  while (t < kThreads && static_cast<uint64_t>(t) < work)
    t *= 2;
  return t;
}

template <int C, typename T> void launch_codec(const GatherArgs &a, bool scalar, cudaStream_t s) {
  const uint64_t work = scalar ? a.columns : a.columns / kUnit;
  const int tpr = threads_per_row(work);
  const uint64_t per_block = kThreads / tpr;
  const unsigned blocks = static_cast<unsigned>((a.lookups + per_block - 1) / per_block);
  if constexpr (C == DENSE_F32 || C == DENSE_F16 || C == DENSE_BF16 || C == FP8) {
    if (scalar) {
      gather_rows_scalar<C, T><<<blocks, kThreads, 0, s>>>(a, tpr);
      return;
    }
  }
  gather_rows<C, T><<<blocks, kThreads, 0, s>>>(a, tpr);
}

template <typename T> void launch(const GatherArgs &a, bool scalar, cudaStream_t s) {
#define GATHER_CASE(C)                                                                             \
  case C:                                                                                          \
    launch_codec<C, T>(a, scalar, s);                                                              \
    break;
  switch (a.codec) {
    GATHER_CASE(DENSE_F32)
    GATHER_CASE(DENSE_F16)
    GATHER_CASE(DENSE_BF16)
    GATHER_CASE(Q4_0)
    GATHER_CASE(Q4_1)
    GATHER_CASE(Q5_0)
    GATHER_CASE(Q5_1)
    GATHER_CASE(Q8_0)
    GATHER_CASE(Q8_1)
    GATHER_CASE(Q2K)
    GATHER_CASE(Q3K)
    GATHER_CASE(Q4K)
    GATHER_CASE(Q5K)
    GATHER_CASE(Q6K)
    GATHER_CASE(Q8K)
    GATHER_CASE(IQ4_NL)
    GATHER_CASE(MXFP4)
    GATHER_CASE(FP4)
    GATHER_CASE(FP8)
    GATHER_CASE(AFQ)
    GATHER_CASE(SYM_INT4)
    GATHER_CASE(SYM_INT8)
  default:
    break;
  }
#undef GATHER_CASE
}
} // namespace

extern "C" void mistralrs_gather_rows(const GatherArgs *args, int32_t scalar, cudaStream_t stream) {
  if (args->lookups == 0 || args->columns == 0)
    return;
  switch (args->out_dtype) {
  case 0:
    launch<__nv_bfloat16>(*args, scalar != 0, stream);
    break;
  case 1:
    launch<__half>(*args, scalar != 0, stream);
    break;
  default:
    launch<float>(*args, scalar != 0, stream);
    break;
  }
}
