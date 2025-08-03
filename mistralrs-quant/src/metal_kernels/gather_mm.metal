// Copyright © 2024 Apple Inc.
// Combined gather MM kernel from MLX for mistralrs-quant

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ========== Start of bf16.metal ==========
// BFloat16 support
#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}
#else

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  // Check for nan
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  // Take bits
  uint32_t float_bits = as_type<uint32_t>(x);

  // Round to nearest even
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);

  // Take upper 16 bits
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  // Upper 16 bits are the data and lower 16 bits are 0s
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

// Binary operators
#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }                                                                            \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

// Arithmetic Operators
#define bfloat_binop(_op_, _operator_)                                         \
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \
                    _MLX_BFloat16, float);                                     \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

// Comparison ops
#define bfloat_compop(__op__, __operator__)                                    \
  bfloat_binop_base(__op__, __operator__, bool, _MLX_BFloat16, _MLX_BFloat16,  \
                    float);                                                    \
  bfloat_binop_helper(__op__, __operator__, bool, float, float);               \
  bfloat_binop_helper(__op__, __operator__, bool, half, float);                \
  bfloat_binop_helper(__op__, __operator__, bool, int32_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint32_t, float);            \
  bfloat_binop_helper(__op__, __operator__, bool, int64_t, float);             \
  bfloat_binop_helper(__op__, __operator__, bool, uint64_t, float);

bfloat_compop(>, operator>);
bfloat_compop(<, operator<);
bfloat_compop(>=, operator>=);
bfloat_compop(<=, operator<=);
bfloat_compop(==, operator==);
bfloat_compop(!=, operator!=);

#undef bfloat_compop
#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

typedef struct _MLX_BFloat16 bfloat16_t;

inline uint16_t bfloat16_to_uint16(const bfloat16_t x) { return x.bits_; }

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return _MLX_BFloat16(x, _MLX_BFloat16::bits_to_bfloat());
}

#endif
// ========== End of bf16.metal ==========

// ========== Start of defines.h ==========
#define STEEL_CONST constexpr constant static const

template <typename T>
struct Limits {
  static const constant T max = metal::numeric_limits<T>::max();
  static const constant T min = metal::numeric_limits<T>::min();
  static const constant T finite_max = metal::numeric_limits<T>::max();
  static const constant T finite_min = metal::numeric_limits<T>::min();
};

template <>
struct Limits<half> {
  static constexpr constant half max = 65504.0h;
  static constexpr constant half min = -65504.0h;
  static constexpr constant half finite_max = 65504.0h;
  static constexpr constant half finite_min = -65504.0h;
};

template <>
struct Limits<float> {
  static constexpr constant float max = metal::numeric_limits<float>::max();
  static constexpr constant float min = -metal::numeric_limits<float>::max();
  static constexpr constant float finite_max =
      metal::numeric_limits<float>::max();
  static constexpr constant float finite_min =
      -metal::numeric_limits<float>::max();
};

template <>
struct Limits<bfloat16_t> {
  static constexpr constant float max = metal::numeric_limits<float>::max();
  static constexpr constant float min = -metal::numeric_limits<float>::max();
  static constexpr constant float finite_max =
      metal::numeric_limits<float>::max();
  static constexpr constant float finite_min =
      -metal::numeric_limits<float>::max();
};
// ========== End of defines.h ==========

// ========== Start of bf16_math.h ==========
#define METAL_FUNC constexpr
#define METAL_INLINE inline

METAL_FUNC bfloat16_t abs(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::abs(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t ceil(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::ceil(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t cos(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::cos(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t exp(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::exp(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t exp2(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::exp2(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t floor(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::floor(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t log(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::log(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t log2(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::log2(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t rint(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::rint(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t rsqrt(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::rsqrt(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t sin(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::sin(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t sqrt(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::sqrt(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t round(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::round(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t trunc(bfloat16_t x) {
  return static_cast<bfloat16_t>(metal::trunc(static_cast<float>(x)));
}

METAL_FUNC bfloat16_t min(bfloat16_t x, bfloat16_t y) {
  return static_cast<bfloat16_t>(metal::min(static_cast<float>(x), static_cast<float>(y)));
}

METAL_FUNC bfloat16_t max(bfloat16_t x, bfloat16_t y) {
  return static_cast<bfloat16_t>(metal::max(static_cast<float>(x), static_cast<float>(y)));
}
// ========== End of bf16_math.h ==========

// ========== Start of complex.h ==========
template <typename T>
struct complex_t {
  T real;
  T imag;
};

typedef struct complex_t<float> complex64_t;
typedef struct complex_t<half> complex32_t;
typedef struct complex_t<bfloat16_t> complex32bf16_t;
// ========== End of complex.h ==========

// ========== Start of utils.h ==========
METAL_FUNC ulong elem_to_loc(
    uint elem, 
    constant const int* shape,
    constant const int64_t* strides, 
    int ndim) {
  ulong loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc += pos_in_dim * strides[i];
  }
  return loc;
}

template <typename T, typename U>
U static_cast_with_checks(T x) {
  return static_cast<U>(x);
}

template <>
METAL_INLINE uint8_t static_cast_with_checks<float, uint8_t>(float x) {
  return x < 0 ? uint8_t(0) : static_cast<uint8_t>(x);
}

template <>
METAL_INLINE uint8_t static_cast_with_checks<bfloat16_t, uint8_t>(bfloat16_t x) {
  return x < 0 ? uint8_t(0) : static_cast<uint8_t>(x);
}

template <>
METAL_INLINE uint8_t static_cast_with_checks<half, uint8_t>(half x) {
  return x < 0 ? uint8_t(0) : static_cast<uint8_t>(x);
}

template <>
METAL_INLINE uint16_t static_cast_with_checks<float, uint16_t>(float x) {
  return x < 0 ? uint16_t(0) : static_cast<uint16_t>(x);
}

template <>
METAL_INLINE uint16_t static_cast_with_checks<bfloat16_t, uint16_t>(bfloat16_t x) {
  return x < 0 ? uint16_t(0) : static_cast<uint16_t>(x);
}

template <>
METAL_INLINE uint16_t static_cast_with_checks<half, uint16_t>(half x) {
  return x < 0 ? uint16_t(0) : static_cast<uint16_t>(x);
}

template <>
METAL_INLINE uint32_t static_cast_with_checks<float, uint32_t>(float x) {
  return x < 0 ? uint32_t(0) : static_cast<uint32_t>(x);
}

template <>
METAL_INLINE uint32_t static_cast_with_checks<bfloat16_t, uint32_t>(bfloat16_t x) {
  return x < 0 ? uint32_t(0) : static_cast<uint32_t>(x);
}

template <>
METAL_INLINE uint32_t static_cast_with_checks<half, uint32_t>(half x) {
  return x < 0 ? uint32_t(0) : static_cast<uint32_t>(x);
}

template <>
METAL_INLINE uint64_t static_cast_with_checks<float, uint64_t>(float x) {
  return x < 0 ? uint64_t(0) : static_cast<uint64_t>(x);
}

template <>
METAL_INLINE uint64_t static_cast_with_checks<bfloat16_t, uint64_t>(bfloat16_t x) {
  return x < 0 ? uint64_t(0) : static_cast<uint64_t>(x);
}

template <>
METAL_INLINE uint64_t static_cast_with_checks<half, uint64_t>(half x) {
  return x < 0 ? uint64_t(0) : static_cast<uint64_t>(x);
}
// ========== End of utils.h ==========

// ========== Start of steel/utils.h ==========
namespace mlx {
namespace steel {

METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

} // namespace steel
} // namespace mlx
// ========== End of steel/utils.h ==========

// ========== Start of steel/gemm/params.h ==========
namespace mlx {
namespace steel {

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int ldc;
  const int fdc;

  const int64_t batch_stride_c;

  const float alpha;
  const float beta;
};

} // namespace steel
} // namespace mlx
// ========== End of steel/gemm/params.h ==========

// ========== Start of steel/gemm/transforms.h ==========
namespace mlx {
namespace steel {

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  static METAL_FUNC OutT apply(InT x, OutT) { return static_cast<OutT>(x); }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  float alpha;
  float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  METAL_FUNC OutT apply(InT x) const { return static_cast<OutT>(x * alpha); }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x * alpha + c * beta);
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

} // namespace steel
} // namespace mlx
// ========== End of steel/gemm/transforms.h ==========

// ========== Start of steel/gemm/loader.h ==========
namespace mlx {
namespace steel {

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short src_ld,
    short tile_stride = BCOLS>
struct BlockLoader {
  const int src_offset;
  const int dst_offset;

  typedef T src_dtype;

  STEEL_CONST short n_rows = BROWS;
  STEEL_CONST short n_cols = BCOLS;

  const device T* src;
  const int64_t ld;
  threadgroup T* dst;

  struct alignas(BCOLS * sizeof(T)) {
    T row[BCOLS];
  } tiles[BROWS / tile_stride];

  const short thread_idx;
  const short bi;
  const short bj;

  const short ti;
  const short tj;

  const short warp_size = 32;
  const short n_threads = (BROWS < warp_size) ? BROWS : warp_size;
  const short n_tile_rows = BROWS / n_threads;

  METAL_FUNC BlockLoader(
      const device T* src_,
      const int64_t ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_offset(0),
        dst_offset(0),
        src(src_),
        ld(ld_),
        dst(dst_),
        thread_idx(simd_group_id * warp_size + simd_lane_id),
        bi(n_tile_rows * thread_idx),
        bj(0),
        ti(thread_idx),
        tj(0) {}

  METAL_FUNC void load_unsafe() const {
    #pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += n_threads) {
      #pragma clang loop unroll(full)
      for (short j = 0; j < BCOLS; j++) {
        dst[i * dst_ld + thread_idx * dst_ld + j] = src[i * src_ld + thread_idx * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dims) const {
    src_tile_dims = min(src_tile_dims, short2(BROWS, BCOLS));

    #pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += n_threads) {
      if (i + ti < src_tile_dims.x) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < BCOLS; j++) {
          if (j < src_tile_dims.y) {
            dst[i * dst_ld + thread_idx * dst_ld + j] = src[i * src_ld + thread_idx * src_ld + j];
          } else {
            dst[i * dst_ld + thread_idx * dst_ld + j] = T(0);
          }
        }
      } else {
        #pragma clang loop unroll(full)
        for (short j = 0; j < BCOLS; j++) {
          dst[i * dst_ld + thread_idx * dst_ld + j] = T(0);
        }
      }
    }
  }

  METAL_FUNC void next() {
    src += src_offset;
  }
};

} // namespace steel
} // namespace mlx
// ========== End of steel/gemm/loader.h ==========

// ========== Start of steel/gemm/mma.h ==========
namespace mlx {
namespace steel {

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  typedef AccumType accum_type;
  typedef T lhs_type;
  typedef T rhs_type;

  STEEL_CONST short TM_stride = 8 * WM;
  STEEL_CONST short TN_stride = 8 * WN;

  STEEL_CONST short TM = BM / TM_stride;
  STEEL_CONST short TN = BN / TN_stride;

  STEEL_CONST short simd_stride = 8;

  simdgroup_matrix<AccumType, TM_stride, TN_stride> C[TM][TN] = {
      simdgroup_matrix<AccumType, TM_stride, TN_stride>(0)};

  short sm;
  short sn;

  const short tm;
  const short tn;

  METAL_FUNC BlockMMA(
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : sm(simd_group_id / WN),
        sn(simd_group_id % WN),
        tm(simd_lane_id / (TN_stride / simd_stride)),
        tn(simd_lane_id % (TN_stride / simd_stride)) {
    sm = WM - sm - 1;
  }

  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
    As += sm * TM_stride * (BK + 8);
    Bs += sn * TN_stride * (BK + 8);

    simdgroup_matrix<T, TM_stride, simd_stride> A[TM];
    simdgroup_matrix<T, TN_stride, simd_stride> B[TN];

    #pragma clang loop unroll(full)
    for (short kk = 0; kk < BK; kk += simd_stride) {
      threadgroup_barrier(mem_flags::mem_none);

      #pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
        simdgroup_load(A[i], As + i * TM_stride * (BK + 8) + kk, BK + 8, false);
      }

      #pragma clang loop unroll(full)
      for (short j = 0; j < TN; j++) {
        simdgroup_load(B[j], Bs + j * TN_stride * (BK + 8) + kk, BK + 8, false);
      }

      #pragma clang loop unroll(full)
      for (short i = 0; i < TM; i++) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < TN; j++) {
          simdgroup_multiply_accumulate(
              C[i][j], A[i], B[j], C[i][j]);
        }
      }
    }
  }

  METAL_FUNC void store_result(device U* D, const int ldd) const {
    D += sm * TM_stride * ldd + sn * TN_stride;

    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < TN; j++) {
        simdgroup_store(
            transform_output(C[i][j]),
            D + i * TM_stride * ldd + j * TN_stride,
            ldd);
      }
    }
  }

  METAL_FUNC void store_result_safe(
      device U* D,
      const int ldd,
      short2 dst_tile_dims) const {
    D += sm * TM_stride * ldd + sn * TN_stride;

    dst_tile_dims -= short2(sn * TN_stride, sm * TM_stride);

    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < TN; j++) {
        auto dst_offset = i * TM_stride * ldd + j * TN_stride;

        if (tm + i * TM_stride < dst_tile_dims.y &&
            tn + j * TN_stride < dst_tile_dims.x) {
          simdgroup_store(
              transform_output(C[i][j]),
              D + dst_offset,
              ldd);
        }
      }
    }
  }

  METAL_FUNC void store_result_slice(
      device U* D,
      const int ldd,
      short2 offset,
      short2 out_dims) const {
    // Only called by gather_mm_rhs which needs special slice handling
    D += sm * TM_stride * ldd + sn * TN_stride;

    if (sm == 0) {
      out_dims.y = out_dims.y - offset.y;
      offset.y = offset.y % TM_stride;
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < TM; i++) {
      auto row = tm + i * TM_stride;
      if (row < out_dims.y && row >= offset.y) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < TN; j++) {
          D[(row - offset.y) * ldd + tn + j * TN_stride] = static_cast<U>(C[i][j][tm % TM_stride][tn]);
        }
      }
    }
  }

  // Helper to apply epilogue transformation
  METAL_FUNC auto transform_output(const simdgroup_matrix<AccumType, TM_stride, TN_stride>& mat) const {
    simdgroup_matrix<U, TM_stride, TN_stride> result;
    #pragma clang loop unroll(full)
    for (int i = 0; i < TM_stride; i++) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < TN_stride; j++) {
        result[i][j] = Epilogue::apply(mat[i][j]);
      }
    }
    return result;
  }
};

} // namespace steel
} // namespace mlx
// ========== End of steel/gemm/mma.h ==========

// ========== Start of steel/gemm/gemm.h ==========
namespace mlx {
namespace steel {

template <bool M_aligned, bool N_aligned, bool K_aligned>
struct LoopAlignment {};

template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned,
    typename AccumType = typename AccumHelper<T>::accum_type,
    typename Epilogue = TransformNone<U, AccumType>>
struct GEMMKernel {
  STEEL_CONST short tgp_padding_a = 16 / sizeof(T);
  STEEL_CONST short tgp_padding_b = 16 / sizeof(T);
  STEEL_CONST short tgp_mem_size_a =
      transpose_a ? BK * (BM + tgp_padding_a) : BM * (BK + tgp_padding_a);
  STEEL_CONST short tgp_mem_size_b =
      transpose_b ? BN * (BK + tgp_padding_b) : BK * (BN + tgp_padding_b);
  STEEL_CONST short tgp_mem_size = tgp_mem_size_a + tgp_mem_size_b;

  STEEL_CONST short tgp_size = WM * WN * 32;

  using loader_a_t = BlockLoader<
      T,
      transpose_a ? BK : BM,
      transpose_a ? BM : BK,
      transpose_a ? BM + tgp_padding_a : BK + tgp_padding_a,
      transpose_a ? BM : BK>;

  using loader_b_t = BlockLoader<
      T,
      transpose_b ? BN : BK,
      transpose_b ? BK : BN,
      transpose_b ? BK + tgp_padding_b : BN + tgp_padding_b,
      transpose_b ? BK : BN>;

  using mma_t = BlockMMA<T, U, BM, BN, BK, WM, WN, AccumType, Epilogue>;

  template <typename alignments>
  static METAL_FUNC void gemm_loop(
      threadgroup T* As,
      threadgroup T* Bs,
      int gemm_k_iterations,
      thread loader_a_t& loader_a,
      thread loader_b_t& loader_b,
      thread mma_t& mma_op,
      short tgp_bm,
      short tgp_bn,
      short tgp_bk,
      alignments) {
    constexpr const bool M_aligned = alignments::M_aligned;
    constexpr const bool N_aligned = alignments::N_aligned;
    constexpr const bool K_aligned = alignments::K_aligned;

    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      loader_a.load_safe(short2(K_aligned ? BK : tgp_bk, M_aligned ? BM : tgp_bm));
      loader_b.load_safe(short2(K_aligned ? BK : tgp_bk, N_aligned ? BN : tgp_bn));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      mma_op.mma(As, Bs);

      loader_a.next();
      loader_b.next();
    }
  }
};

} // namespace steel
} // namespace mlx
// ========== End of steel/gemm/gemm.h ==========

// ========== Start of steel_gemm_gather.h kernel definitions ==========
using namespace mlx::steel;

constant bool has_batch [[function_constant(10)]];
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gather_mm_rhs(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* rhs_indices [[buffer(2)]],
    device T* C [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Find the block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = rhs_indices[c_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (rhs_indices[c_row + n] != index) {
        offset_next = n;
        index_next = rhs_indices[c_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Prepare threadgroup mma operation
    thread mma_t mma_op(simd_group_id, simd_lane_id);

    // Prepare threadgroup loading operations
    thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(
        B + index * params->batch_stride_b,
        params->ldb,
        Bs,
        simd_group_id,
        simd_lane_id);

    // Prepare iterations
    const int gemm_k_iterations = params->gemm_k_iterations_aligned;

    // Do unaligned K iterations first
    if (!align_K) {
      const int k_last = params->gemm_k_iterations_aligned * BK;
      const int k_remain = params->K - k_last;
      const size_t k_jump_a =
          transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
      const size_t k_jump_b =
          transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

      // Move loader source ahead to end
      loader_a.src += k_jump_a;
      loader_b.src += k_jump_b;

      // Load tile
      const short2 tile_dims_A =
          transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
      const short2 tile_dims_B =
          transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

      loader_a.load_safe(tile_dims_A);
      loader_b.load_safe(tile_dims_B);

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Do matmul
      mma_op.mma(As, Bs);

      // Reset source back to start
      loader_a.src -= k_jump_a;
      loader_b.src -= k_jump_b;
    }

    // Matrix level aligned never check
    if (align_M && align_N) {
      for (int k = 0; k < gemm_k_iterations; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load elements into threadgroup
        loader_a.load_unsafe();
        loader_b.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply and accumulate threadgroup elements
        mma_op.mma(As, Bs);

        // Prepare for next iteration
        loader_a.next();
        loader_b.next();
      }

      // Store results to device memory
      if (offset_next - offset == BM) {
        mma_op.store_result(C, params->ldd);
      } else {
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(BN, offset_next));
      }
    } else {
      const short lbk = 0;

      // Tile aligned don't check
      if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<true, true, true>{});
        if (offset_next - offset == BM) {
          mma_op.store_result(C, params->ldd);
        } else {
          mma_op.store_result_slice(
              C, params->ldd, short2(0, offset), short2(BN, offset_next));
        }
      }

      // Tile partially aligned check rows
      else if (align_N || tgp_bn == BN) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<false, true, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(BN, offset_next));
      }

      // Tile partially aligned check cols
      else if (align_M || tgp_bm == BM) {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<true, false, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(tgp_bn, offset_next));
      }

      // Nothing aligned so check both rows and cols
      else {
        gemm_kernel::gemm_loop(
            As,
            Bs,
            gemm_k_iterations,
            loader_a,
            loader_b,
            mma_op,
            tgp_bm,
            tgp_bn,
            lbk,
            LoopAlignment<false, false, true>{});
        mma_op.store_result_slice(
            C, params->ldd, short2(0, offset), short2(tgp_bn, offset_next));
      }
    }
  }
}

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gather_mm(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device uint32_t* lhs_indices [[buffer(2)]],
    const device uint32_t* rhs_indices [[buffer(3)]],
    device T* C [[buffer(4)]],
    const constant GEMMParams* params [[buffer(5)]],
    const constant int* indices_shape [[buffer(6)]],
    const constant int64_t* lhs_strides [[buffer(7)]],
    const constant int64_t* rhs_strides [[buffer(8)]],
    const constant int& batch_ndim_a [[buffer(9)]],
    const constant int* batch_shape_a [[buffer(10)]],
    const constant int64_t* batch_strides_a [[buffer(11)]],
    const constant int& batch_ndim_b [[buffer(12)]],
    const constant int* batch_shape_b [[buffer(13)]],
    const constant int64_t* batch_strides_b [[buffer(14)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  if (params->tiles_n <= static_cast<int>(tid.x) ||
      params->tiles_m <= static_cast<int>(tid.y)) {
    return;
  }

  // Move A and B to the locations pointed by lhs_indices and rhs_indices.
  uint32_t indx_A, indx_B;
  if (has_batch) {
    ulong2 indices_offsets = elem_to_loc_broadcast(
        tid.z, indices_shape, lhs_strides, rhs_strides, params->batch_ndim);
    indx_A = lhs_indices[indices_offsets.x];
    indx_B = rhs_indices[indices_offsets.y];
  } else {
    indx_A = lhs_indices[params->batch_stride_a * tid.z];
    indx_B = rhs_indices[params->batch_stride_b * tid.z];
  }
  A += elem_to_loc(indx_A, batch_shape_a, batch_strides_a, batch_ndim_a);
  B += elem_to_loc(indx_B, batch_shape_b, batch_strides_b, batch_ndim_b);
  C += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  // Just make sure everybody's finished with the indexing math above.
  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid.y * BM;
  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * params->ldd + c_col_long;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  // Prepare iterations
  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;
    const size_t k_jump_a =
        transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
    const size_t k_jump_b =
        transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

    // Move loader source ahead to end
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    // Load tile
    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    mma_op.mma(As, Bs);

    // Reset source back to start
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  // Matrix level aligned never check
  if (align_M && align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    // Store results to device memory
    mma_op.store_result(C, params->ldd);
  } else {
    const short lbk = 0;

    // Tile aligned don't check
    if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<true, true, true>{});
      mma_op.store_result(C, params->ldd);
    }

    // Tile partially aligned check rows
    else if (align_N || tgp_bn == BN) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<false, true, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Tile partially aligned check cols
    else if (align_M || tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<true, false, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }

    // Nothing aligned so check both rows and cols
    else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          lbk,
          LoopAlignment<false, false, true>{});
      mma_op.store_result_safe(C, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}

// ========== Start of kernel instantiations ==========
#define instantiate_gather_mm_rhs(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  template [[host_name("gather_mm_rhs_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]] \
  [[kernel, max_total_threads_per_threadgroup(wm * wn * 32)]] \
  void gather_mm_rhs<itype, bm, bn, bk, wm, wn, trans_a, trans_b, float>( \
      const device itype* A [[buffer(0)]], \
      const device itype* B [[buffer(1)]], \
      const device uint32_t* rhs_indices [[buffer(2)]], \
      device itype* C [[buffer(3)]], \
      const constant GEMMParams* params [[buffer(4)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]]);

#define instantiate_gather_mm(tname, trans_a, trans_b, iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  template [[host_name("gather_mm_" #tname "_" #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]] \
  [[kernel, max_total_threads_per_threadgroup(wm * wn * 32)]] \
  void gather_mm<itype, bm, bn, bk, wm, wn, trans_a, trans_b, float>( \
      const device itype* A [[buffer(0)]], \
      const device itype* B [[buffer(1)]], \
      const device uint32_t* lhs_indices [[buffer(2)]], \
      const device uint32_t* rhs_indices [[buffer(3)]], \
      device itype* C [[buffer(4)]], \
      const constant GEMMParams* params [[buffer(5)]], \
      const constant int* indices_shape [[buffer(6)]], \
      const constant int64_t* lhs_strides [[buffer(7)]], \
      const constant int64_t* rhs_strides [[buffer(8)]], \
      const constant int& batch_ndim_a [[buffer(9)]], \
      const constant int* batch_shape_a [[buffer(10)]], \
      const constant int64_t* batch_strides_a [[buffer(11)]], \
      const constant int& batch_ndim_b [[buffer(12)]], \
      const constant int* batch_shape_b [[buffer(13)]], \
      const constant int64_t* batch_strides_b [[buffer(14)]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint3 tid [[threadgroup_position_in_grid]]);

#define instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm_rhs(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)  \
  instantiate_gather_mm_rhs(nt, false,  true, iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, bm, bn, bk, wm, wn) \
  instantiate_gather_mm(nn, false, false, iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_gather_mm(nt, false, true , iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_gather_mm(tn, true , false, iname, itype, oname, otype, bm, bn, bk, wm, wn)      \
  instantiate_gather_mm(tt, true , true , iname, itype, oname, otype, bm, bn, bk, wm, wn)

#define instantiate_gather_mm_shapes_helper(iname, itype, oname, otype)                     \
  instantiate_gather_mm_rhs_transpose_helper(iname, itype, oname, otype, 16, 64, 16, 1, 2)  \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2)      \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 1, 2)      \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2)      \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 32, 64, 16, 1, 2)      \
  instantiate_gather_mm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2)

instantiate_gather_mm_shapes_helper(float16, half, float16, half);
instantiate_gather_mm_shapes_helper(bfloat16, bfloat16_t, bfloat16, bfloat16_t);
instantiate_gather_mm_shapes_helper(float32, float, float32, float);