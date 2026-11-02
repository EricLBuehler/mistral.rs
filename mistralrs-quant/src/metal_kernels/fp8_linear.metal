#include "float8.metal"
#include "utils.metal"
#include <metal_simdgroup_matrix>

namespace fp8_linear {

constant constexpr ushort kScaleBlock128 = 1;
constant constexpr ushort kScaleTensor = 2;
constant constexpr ushort kScaleChannel = 3;
constant ushort scale_mode [[function_constant(500)]];
constant bool aligned [[function_constant(501)]];

struct Params {
  uint m;
  uint n;
  uint k;
  uint block_y;
  uint block_x;
  uint scale_stride;
};

inline float load_scale(const device float *scales, constant Params &p, uint n, uint k) {
  size_t scale;
  if (scale_mode == kScaleBlock128) {
    scale = size_t(n >> 7) * p.scale_stride + (k >> 7);
  } else if (scale_mode == kScaleTensor) {
    scale = 0;
  } else if (scale_mode == kScaleChannel) {
    scale = n;
  } else {
    scale = size_t(n / p.block_y) * p.scale_stride + k / p.block_x;
  }
  return scales[scale];
}

template <typename T>
inline T load_weight(const device uchar *weight, const device float *scales,
                     constant Params &p, uint n, uint k) {
  return T(fp8_e4m3_to_float(weight[size_t(n) * p.k + k]) * load_scale(scales, p, n, k));
}

constant constexpr uint kSimdWidth = 32;
constant constexpr uint kSimdgroups = 4;
constant constexpr uint kThreads = kSimdWidth * kSimdgroups;
constant constexpr uint kGemvSimdgroups = 8;
constant constexpr uint kOutputsPerSimd = 4;
constant constexpr uint kValuesPerLane = 4;
constant constexpr uint kVectorChunk = kSimdWidth * kValuesPerLane;
constant constexpr uint kTileK = 32;
constant constexpr uint kFragment = 8;
template <typename T, uint Count> struct alignas(sizeof(T)) TileVector {
  uchar bytes[sizeof(T) * Count];
};

template <typename T> struct MmaOperand {
  using type = T;
};
template <typename T> struct InputVector {
  using type = vec<T, 4>;
  static type load(const device T *input) {
    // Packed vectors preserve scalar alignment for sliced tensors.
    return *reinterpret_cast<const device packed_vec<T, 4> *>(input);
  }
};
#if !defined(__HAVE_BFLOAT__)
template <> struct MmaOperand<bfloat16_t> {
  using type = float;
};
template <> struct InputVector<bfloat16_t> {
  using type = float4;
  static type load(const device bfloat16_t *input) {
    const ushort4 bits = *reinterpret_cast<const device packed_ushort4 *>(input);
    return as_type<float4>(uint4(bits) << 16);
  }
};
#endif

template <typename T, uint Rows>
[[kernel]] void gemv(const device T *input [[buffer(0)]],
                    const device uchar *weight [[buffer(1)]],
                    const device float *scales [[buffer(2)]],
                    device T *output [[buffer(3)]],
                    constant Params &p [[buffer(4)]],
                    uint3 group [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (group.x * kGemvSimdgroups + simd) * kOutputsPerSimd;
  if (!aligned && n0 >= p.n) {
    return;
  }
  float sums[Rows][kOutputsPerSimd] = {{0}};
  for (uint base = lane * kValuesPerLane; base < p.k; base += kVectorChunk) {
    const float block_scale = scale_mode == kScaleBlock128
                                  ? scales[size_t(n0 >> 7) * p.scale_stride + (base >> 7)]
                                  : 0.f;
    float x[Rows][kValuesPerLane];
#pragma clang loop unroll(full)
    for (uint m = 0; m < Rows; ++m) {
      if (aligned) {
        const float4 values = float4(InputVector<T>::load(input + size_t(m) * p.k + base));
#pragma clang loop unroll(full)
        for (uint i = 0; i < kValuesPerLane; ++i) {
          x[m][i] = values[i];
        }
      } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < kValuesPerLane; ++i) {
          x[m][i] = m < p.m && base + i < p.k
                         ? float(input[size_t(m) * p.k + base + i])
                         : 0.f;
        }
      }
    }
#pragma clang loop unroll(full)
    for (uint o = 0; o < kOutputsPerSimd; ++o) {
      if (aligned || n0 + o < p.n) {
        uchar4 packed;
        if (aligned) {
          packed = *reinterpret_cast<const device packed_uchar4 *>(weight + size_t(n0 + o) * p.k + base);
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < kValuesPerLane; ++i) {
          const uint k = base + i;
          if (aligned || k < p.k) {
            const uchar bits = aligned ? packed[i] : weight[size_t(n0 + o) * p.k + k];
            const float scale = scale_mode == kScaleBlock128 ? block_scale : load_scale(scales, p, n0 + o, k);
            const T scaled = T(fp8_e4m3_to_float(bits) * scale);
            const float w = float(scaled);
#pragma clang loop unroll(full)
            for (uint m = 0; m < Rows; ++m) {
              sums[m][o] = fma(x[m][i], w, sums[m][o]);
            }
          }
        }
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint m = 0; m < Rows; ++m) {
#pragma clang loop unroll(full)
    for (uint o = 0; o < kOutputsPerSimd; ++o) {
      const float sum = simd_sum(sums[m][o]);
      if (lane == 0 && (aligned || (m < p.m && n0 + o < p.n))) {
        output[size_t(m) * p.n + n0 + o] = T(sum);
      }
    }
  }
}

template <typename T, uint TileM, uint TileN>
[[kernel]] void gemm(const device T *input [[buffer(0)]],
                    const device uchar *weight [[buffer(1)]],
                    const device float *scales [[buffer(2)]],
                    device T *output [[buffer(3)]],
                    constant Params &p [[buffer(4)]],
                    uint3 group [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint stride = kTileK + 16 / sizeof(T);
  constexpr uint simd_cols_count = TileM == kFragment ? kSimdgroups : 2;
  constexpr uint simd_rows = TileM / (kSimdgroups / simd_cols_count);
  constexpr uint simd_cols = TileN / simd_cols_count;
  constexpr uint fragments_m = simd_rows / kFragment;
  constexpr uint fragments_n = simd_cols / kFragment;
  using Operand = typename MmaOperand<T>::type;
  threadgroup T xs[TileM * stride];
  threadgroup T ws[TileN * stride];
  const uint m0 = group.y * TileM;
  const uint n0 = group.x * TileN;
  const uint sm = (simd / simd_cols_count) * simd_rows;
  const uint sn = (simd % simd_cols_count) * simd_cols;
  const uint quad = lane / 4;
  const uint fm = (quad & 4) + ((lane / 2) % 4);
  const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
  simdgroup_matrix<float, kFragment, kFragment> accum[fragments_m][fragments_n];
  simdgroup_matrix<Operand, kFragment, kFragment> a[fragments_m];
  simdgroup_matrix<Operand, kFragment, kFragment> b[fragments_n];
  // Full unrolling keeps fragment indices constant so the compiler retains register storage.
#pragma clang loop unroll(full)
  for (uint i = 0; i < fragments_m; ++i) {
#pragma clang loop unroll(full)
    for (uint j = 0; j < fragments_n; ++j) {
      accum[i][j] = make_filled_simdgroup_matrix<float, kFragment, kFragment>(0.f);
    }
  }

  for (uint base = 0; base < p.k; base += kTileK) {
    const float block_scale = scale_mode == kScaleBlock128
                                  ? scales[size_t(n0 >> 7) * p.scale_stride + (base >> 7)]
                                  : 0.f;
    if (aligned) {
      constexpr uint input_reads = TileM * kTileK / kThreads;
      constexpr uint weight_reads = TileN * kTileK / kThreads;
      const uint input_row = tid / (kTileK / input_reads);
      const uint input_col = (tid % (kTileK / input_reads)) * input_reads;
      using InputTile = TileVector<T, input_reads>;
      auto destination = reinterpret_cast<threadgroup InputTile *>(xs + input_row * stride + input_col);
      if (m0 + input_row < p.m) {
        *destination = *reinterpret_cast<const device InputTile *>(input + size_t(m0 + input_row) * p.k + base + input_col);
      } else {
        *destination = InputTile{};
      }
      const uint weight_row = tid / (kTileK / weight_reads);
      const uint weight_col = (tid % (kTileK / weight_reads)) * weight_reads;
      const auto values = *reinterpret_cast<const device TileVector<uchar, weight_reads> *>(weight + size_t(n0 + weight_row) * p.k + base + weight_col);
#pragma clang loop unroll(full)
      for (uint j = 0; j < weight_reads; ++j) {
        const float scale = scale_mode == kScaleBlock128 ? block_scale : load_scale(scales, p, n0 + weight_row, base + weight_col + j);
        ws[weight_row * stride + weight_col + j] = T(fp8_e4m3_to_float(values.bytes[j]) * scale);
      }
    } else {
#pragma clang loop unroll(full)
      for (uint chunk = 0; chunk < TileM * kTileK / kThreads; ++chunk) {
        const uint i = tid + chunk * kThreads;
        const uint row = i / kTileK;
        const uint col = i % kTileK;
        const uint k = base + col;
        xs[row * stride + col] = m0 + row < p.m && k < p.k
                                     ? input[size_t(m0 + row) * p.k + k]
                                     : T(0);
      }
#pragma clang loop unroll(full)
      for (uint chunk = 0; chunk < TileN * kTileK / kThreads; ++chunk) {
        const uint i = tid + chunk * kThreads;
        const uint row = i / kTileK;
        const uint col = i % kTileK;
        const uint k = base + col;
        T value = T(0);
        if (n0 + row < p.n && k < p.k) {
          value = scale_mode == kScaleBlock128
                      ? T(fp8_e4m3_to_float(weight[size_t(n0 + row) * p.k + k]) * block_scale)
                      : load_weight<T>(weight, scales, p, n0 + row, k);
        }
        ws[row * stride + col] = value;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint k = 0; k < kTileK; k += kFragment) {
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (uint i = 0; i < fragments_m; ++i) {
        a[i].thread_elements()[0] = Operand(xs[(sm + i * kFragment + fm) * stride + k + fn]);
        a[i].thread_elements()[1] = Operand(xs[(sm + i * kFragment + fm) * stride + k + fn + 1]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (uint j = 0; j < fragments_n; ++j) {
        b[j].thread_elements()[0] = Operand(ws[(sn + j * kFragment + fn) * stride + k + fm]);
        b[j].thread_elements()[1] = Operand(ws[(sn + j * kFragment + fn + 1) * stride + k + fm]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (uint i = 0; i < fragments_m; ++i) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < fragments_n; ++j) {
          const uint column = i % 2 == 0 ? j : fragments_n - 1 - j;
          simdgroup_multiply_accumulate(accum[i][column], a[i], b[column], accum[i][column]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma clang loop unroll(full)
  for (uint i = 0; i < fragments_m; ++i) {
    const uint m = m0 + sm + i * kFragment + fm;
#pragma clang loop unroll(full)
    for (uint j = 0; j < fragments_n; ++j) {
      const uint n = n0 + sn + j * kFragment + fn;
      if (m < p.m && (aligned || n < p.n)) {
        output[size_t(m) * p.n + n] = T(accum[i][j].thread_elements()[0]);
      }
      if (m < p.m && (aligned || n + 1 < p.n)) {
        output[size_t(m) * p.n + n + 1] = T(accum[i][j].thread_elements()[1]);
      }
    }
  }
}

#define INSTANTIATE_FP8_LINEAR(T, suffix)                                      \
  template [[host_name("fp8_gemv_" #suffix "_r1")]] [[kernel]] decltype(gemv<T, 1>) gemv<T, 1>; \
  template [[host_name("fp8_gemv_" #suffix "_r2")]] [[kernel]] decltype(gemv<T, 2>) gemv<T, 2>; \
  template [[host_name("fp8_gemv_" #suffix "_r4")]] [[kernel]] decltype(gemv<T, 4>) gemv<T, 4>; \
  template [[host_name("fp8_gemm_" #suffix "_8")]] [[kernel]] decltype(gemm<T, 8, 32>) gemm<T, 8, 32>; \
  template [[host_name("fp8_gemm_" #suffix "_16")]] [[kernel]] decltype(gemm<T, 16, 32>) gemm<T, 16, 32>; \
  template [[host_name("fp8_gemm_" #suffix "_32")]] [[kernel]] decltype(gemm<T, 32, 32>) gemm<T, 32, 32>; \
  template [[host_name("fp8_gemm_" #suffix "_64")]] [[kernel]] decltype(gemm<T, 64, 32>) gemm<T, 64, 32>;

INSTANTIATE_FP8_LINEAR(half, f16)
INSTANTIATE_FP8_LINEAR(bfloat16_t, bf16)
INSTANTIATE_FP8_LINEAR(float, f32)

} // namespace fp8_linear
