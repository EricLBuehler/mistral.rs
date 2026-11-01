#include "utils.metal"
#include "float8.metal"

namespace embedding {

constant uint unit = 8;
constant float fp4_values[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
constant int iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1, 13, 25, 38, 53, 69, 89, 113};

// Codec and scale IDs follow RowCodec and ScaleType in embedding/mod.rs.
enum Codec : int {
  DENSE_F32 = 0, DENSE_F16 = 1, DENSE_BF16 = 2,
  Q4_0 = 3, Q4_1 = 4, Q5_0 = 5, Q5_1 = 6, Q8_0 = 7, Q8_1 = 8,
  Q2K = 9, Q3K = 10, Q4K = 11, Q5K = 12, Q6K = 13, Q8K = 14,
  IQ4_NL = 15, FP4 = 16, FP8 = 17, AFQ = 18, SYM_INT4 = 19, SYM_INT8 = 20,
};

struct EmbeddingGatherConfig {
  ulong lookups;
  ulong rows;
  ulong columns;
  ulong data_stride;
  ulong scale_stride;
  ulong scale_row_div;
  ulong group;
  float out_scale;
  uint scale_type;
  uint row_scale_mode;
  uint row_scale_type;
  uint bits;
  uint has_scales;
  uint ids_i64;
  uint padding;
};

inline ushort read_u16(const device uchar *p) {
  return ushort(p[0]) | (ushort(p[1]) << 8);
}

inline uint read_u32(const device uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

inline float read_half(const device uchar *p) {
  return float(as_type<half>(read_u16(p)));
}

inline float scale_at(const device uchar *data, ulong i, uint type) {
  switch (type) {
  case 0: return as_type<float>(read_u32(data + i * 4));
  case 1: return read_half(data + i * 2);
  case 2: return as_type<float>(uint(read_u16(data + i * 2)) << 16);
  case 3: return fp8_e4m3_to_float(data[i]);
  case 4:
    if (data[i] == 0) return as_type<float>(uint(1) << 22);
    if (data[i] == 255) return NAN;
    return as_type<float>(uint(data[i]) << 23);
  default: return as_type<float>(uint(data[i]) << 23);
  }
}

inline int k4_scale(int j, const device uchar *q) {
  return j < 4 ? q[j] & 63 : (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
}

inline int k4_min(int j, const device uchar *q) {
  return j < 4 ? q[j + 4] & 63 : (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}

template <int C>
inline float decode(constant EmbeddingGatherConfig &a, const device uchar *data,
                    const device uchar *scales, const device uchar *aux, ulong col) {
  if (C <= DENSE_BF16) {
    return scale_at(data, col, C);
  } else if (C == Q4_0 || C == Q4_1 || C == IQ4_NL) {
    constexpr int bytes = C == Q4_1 ? 20 : 18;
    constexpr int qs = C == Q4_1 ? 4 : 2;
    const device uchar *b = data + (col / 32) * bytes;
    const int c = col % 32;
    const int q = (b[qs + c % 16] >> (c < 16 ? 0 : 4)) & 15;
    const float d = read_half(b);
    if (C == Q4_0) return (q - 8) * d;
    if (C == Q4_1) return q * d + read_half(b + 2);
    return iq4_values[q] * d;
  } else if (C == Q5_0 || C == Q5_1) {
    constexpr int bytes = C == Q5_1 ? 24 : 22;
    constexpr int qh_offset = C == Q5_1 ? 4 : 2;
    const device uchar *b = data + (col / 32) * bytes;
    const int c = col % 32;
    const uint qh = read_u32(b + qh_offset);
    const int ql = (b[qh_offset + 4 + c % 16] >> (c < 16 ? 0 : 4)) & 15;
    const int q = ql | (((qh >> c) & 1) << 4);
    const float d = read_half(b);
    return C == Q5_0 ? (q - 16) * d : q * d + read_half(b + 2);
  } else if (C == Q8_0 || C == Q8_1) {
    constexpr int bytes = C == Q8_1 ? 36 : 34;
    constexpr int qs = C == Q8_1 ? 4 : 2;
    const device uchar *b = data + (col / 32) * bytes;
    return float(as_type<char>(b[qs + col % 32])) * read_half(b);
  } else if (C == Q8K) {
    const device uchar *b = data + (col / 256) * 292;
    return float(as_type<char>(b[4 + col % 256])) * as_type<float>(read_u32(b));
  } else if (C == Q2K) {
    const device uchar *b = data + (col / 256) * 84;
    const int c = col % 256, n = c / 128, rem = c % 128;
    const int j = rem / 32, h = (rem % 32) / 16;
    const int sc = b[8 * n + 2 * j + h];
    const int q = (b[16 + 32 * n + 16 * h + rem % 16] >> (2 * j)) & 3;
    return read_half(b + 80) * (sc & 15) * q - read_half(b + 82) * (sc >> 4);
  } else if (C == Q3K) {
    const device uchar *b = data + (col / 256) * 110;
    const int c = col % 256, n = c / 128, rem = c % 128;
    const int j = rem / 32, h = (rem % 32) / 16, l = rem % 16;
    const int s = 8 * n + 2 * j + h;
    const device uchar *sb = b + 96;
    const int low = s < 8 ? (sb[s] & 15) : (sb[s - 8] >> 4);
    const int high = (sb[8 + s % 4] >> (2 * (s / 4))) & 3;
    const int q = (b[32 + 32 * n + 16 * h + l] >> (2 * j)) & 3;
    const int sign = (b[16 * h + l] & (1 << (4 * n + j))) ? 0 : 4;
    return read_half(b + 108) * ((low | (high << 4)) - 32) * (q - sign);
  } else if (C == Q4K || C == Q5K) {
    constexpr int bytes = C == Q4K ? 144 : 176;
    constexpr int qs = C == Q4K ? 16 : 48;
    const device uchar *b = data + (col / 256) * bytes;
    const int c = col % 256, chunk = c / 64, hi = (c % 64) / 32, l = c % 32;
    const int ql = (b[qs + 32 * chunk + l] >> (4 * hi)) & 15;
    const int qh = C == Q5K ? ((b[16 + l] >> (2 * chunk + hi)) & 1) << 4 : 0;
    const int s = 2 * chunk + hi;
    return read_half(b) * k4_scale(s, b + 4) * (ql | qh)
         - read_half(b + 2) * k4_min(s, b + 4);
  } else if (C == Q6K) {
    const device uchar *b = data + (col / 256) * 210;
    const int c = col % 256, n = c / 128, quarter = (c % 128) / 32, l = c % 32;
    const int sc = as_type<char>(b[192 + 8 * n + l / 16 + 2 * quarter]);
    const int ql = (b[64 * n + l + 32 * (quarter & 1)] >> (quarter >= 2 ? 4 : 0)) & 15;
    const int qh = (b[128 + 32 * n + l] >> (2 * quarter)) & 3;
    return read_half(b + 208) * sc * ((ql | (qh << 4)) - 32);
  } else if (C == FP4) {
    const int code = (data[col / 2] >> (4 * (col % 2))) & 15;
    const float q = (code & 8) ? -fp4_values[code & 7] : fp4_values[code & 7];
    return q * scale_at(scales, col / a.group, a.scale_type);
  } else if (C == FP8) {
    const float s = a.has_scales ? scale_at(scales, col / a.group, a.scale_type) : 1.f;
    return fp8_e4m3_to_float(data[col]) * s;
  } else if (C == AFQ) {
    const ulong bit = col * a.bits;
    const device uchar *p = data + bit / 8;
    const uint word = p[0] | (bit % 8 + a.bits > 8 ? uint(p[1]) << 8 : 0);
    const uint q = (word >> (bit % 8)) & ((1u << a.bits) - 1);
    const ulong g = col / a.group;
    return fma(float(q), scale_at(scales, g, a.scale_type), scale_at(aux, g, a.scale_type));
  } else {
    const int q = C == SYM_INT8 ? int(as_type<char>(data[col]))
                              : int((data[col / 2] >> (4 * (col % 2))) & 15) - 8;
    return q * scale_at(scales, col / a.group, a.scale_type);
  }
}

template <int C, typename T>
kernel void gather(const device uchar *data [[buffer(0)]],
                   const device uchar *scales [[buffer(1)]],
                   const device uchar *aux [[buffer(2)]],
                   const device uchar *row_scales [[buffer(3)]],
                   const device uchar *ids [[buffer(4)]],
                   device T *out [[buffer(5)]],
                   constant EmbeddingGatherConfig &a [[buffer(6)]],
                   uint tid [[thread_position_in_grid]]) {
  const ulong units = (a.columns + unit - 1) / unit;
  const ulong lookup = ulong(tid) / units;
  if (lookup >= a.lookups) return;
  const ulong row = a.ids_i64 ? ulong(reinterpret_cast<const device long *>(ids)[lookup])
                            : ulong(reinterpret_cast<const device uint *>(ids)[lookup]);
  const ulong first = (ulong(tid) % units) * unit;
  if (row >= a.rows) {
    for (uint i = 0; i < unit && first + i < a.columns; ++i)
      out[lookup * a.columns + first + i] = T(0.f);
    return;
  }
  float multiplier = a.out_scale;
  if (a.row_scale_mode == 1)
    multiplier *= scale_at(row_scales, 0, a.row_scale_type);
  else if (a.row_scale_mode == 2)
    multiplier *= scale_at(row_scales, row, a.row_scale_type);
  data += row * a.data_stride;
  scales += (row / a.scale_row_div) * a.scale_stride;
  if (C == AFQ) aux += (row / a.scale_row_div) * a.scale_stride;
  for (uint i = 0; i < unit && first + i < a.columns; ++i)
    out[lookup * a.columns + first + i] = T(decode<C>(a, data, scales, aux, first + i) * multiplier);
}

#define EMBEDDING_INSTANCE(C, T, NAME) \
  template [[host_name("embedding_gather_" #C "_" #NAME)]] kernel void gather<C, T>( \
      const device uchar *, const device uchar *, const device uchar *, \
      const device uchar *, const device uchar *, device T *, \
      constant EmbeddingGatherConfig &, uint);
#define EMBEDDING_CODEC(C) \
  EMBEDDING_INSTANCE(C, float, float) \
  EMBEDDING_INSTANCE(C, half, half) \
  EMBEDDING_INSTANCE(C, bfloat16_t, bfloat)
EMBEDDING_CODEC(0)
EMBEDDING_CODEC(1)
EMBEDDING_CODEC(2)
EMBEDDING_CODEC(3)
EMBEDDING_CODEC(4)
EMBEDDING_CODEC(5)
EMBEDDING_CODEC(6)
EMBEDDING_CODEC(7)
EMBEDDING_CODEC(8)
EMBEDDING_CODEC(9)
EMBEDDING_CODEC(10)
EMBEDDING_CODEC(11)
EMBEDDING_CODEC(12)
EMBEDDING_CODEC(13)
EMBEDDING_CODEC(14)
EMBEDDING_CODEC(15)
EMBEDDING_CODEC(16)
EMBEDDING_CODEC(17)
EMBEDDING_CODEC(18)
EMBEDDING_CODEC(19)
EMBEDDING_CODEC(20)
#undef EMBEDDING_CODEC
#undef EMBEDDING_INSTANCE

} // namespace embedding
