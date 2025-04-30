#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

typedef half float16_t;

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}
#else

/////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////
// Bfloat struct
/////////////////////////////////////////////////////////////////////////////

struct _MLX_BFloat16 {
  /////////////////////////////////////////////////////////////////////////////
  // Constructors
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

  /////////////////////////////////////////////////////////////////////////////
  // Conversions to bfloat

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

  /////////////////////////////////////////////////////////////////////////////
  // Conversions from bfloat

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

/////////////////////////////////////////////////////////////////////////////
// Bfloat operators
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Unary ops
constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

/////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////
// Inplace Operators
#define bfloat_inplace_op_helper(__op__, __operator__, itype, addr_space)      \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, itype rhs) {                              \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }                                                                            \
  constexpr METAL_FUNC addr_space itype &__operator__(addr_space itype &lhs,   \
                                                      _MLX_BFloat16 rhs) {     \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__, itype)       \
  bfloat_inplace_op_helper(__op__, __operator__, itype, device);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, thread);               \
  bfloat_inplace_op_helper(__op__, __operator__, itype, threadgroup);

#define bfloat_inplace_op(itype)                                               \
  bfloat_inplace_op_addr_space_helper(+, operator+=, itype);                   \
  bfloat_inplace_op_addr_space_helper(-, operator-=, itype);                   \
  bfloat_inplace_op_addr_space_helper(*, operator*=, itype);                   \
  bfloat_inplace_op_addr_space_helper(/, operator/=, itype);

bfloat_inplace_op(float);
bfloat_inplace_op(half);
bfloat_inplace_op(int16_t);
bfloat_inplace_op(int32_t);
bfloat_inplace_op(int64_t);
bfloat_inplace_op(uint16_t);
bfloat_inplace_op(uint32_t);
bfloat_inplace_op(uint64_t);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper
#undef bfloat_inplace_op

#define bfloat_inplace_op_helper(__op__, __operator__, addr_space)             \
  constexpr METAL_FUNC addr_space _MLX_BFloat16 &__operator__(                 \
      addr_space _MLX_BFloat16 &lhs, _MLX_BFloat16 rhs) {                      \
    lhs = static_cast<float>(lhs) __op__ static_cast<float>(rhs);              \
    return lhs;                                                                \
  }

#define bfloat_inplace_op_addr_space_helper(__op__, __operator__)              \
  bfloat_inplace_op_helper(__op__, __operator__, device);                      \
  bfloat_inplace_op_helper(__op__, __operator__, thread);                      \
  bfloat_inplace_op_helper(__op__, __operator__, threadgroup);

bfloat_inplace_op_addr_space_helper(+, operator+=);
bfloat_inplace_op_addr_space_helper(-, operator-=);
bfloat_inplace_op_addr_space_helper(*, operator*=);
bfloat_inplace_op_addr_space_helper(/, operator/=);

#undef bfloat_inplace_op_helper
#undef bfloat_inplace_op_addr_space_helper

/////////////////////////////////////////////////////////////////////////////
// Bfloat typedef
/////////////////////////////////////////////////////////////////////////////

typedef struct _MLX_BFloat16 bfloat16_t;

inline uint16_t bfloat16_to_uint16(const bfloat16_t x) { return x.bits_; }

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return _MLX_BFloat16(x, _MLX_BFloat16::bits_to_bfloat());
}

#endif

///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(static_cast<ctype>(x),             \
                                            static_cast<ctype>(y),             \
                                            static_cast<ctype>(z), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(static_cast<ctype>(x),          \
                                               static_cast<ctype>(y),          \
                                               static_cast<ctype>(z), mfast)); \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(static_cast<ctype>(x),             \
                                            static_cast<ctype>(y),             \
                                            static_cast<ctype>(z), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int &exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(static_cast<ctype>(x),             \
                                            static_cast<ctype>(y),             \
                                            static_cast<ctype>(z), mfast));    \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(static_cast<ctype>(x),          \
                                               static_cast<ctype>(y),          \
                                               static_cast<ctype>(z), mfast)); \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(static_cast<ctype>(x),             \
                                            static_cast<ctype>(y),             \
                                            static_cast<ctype>(z), mfast));    \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(bfloat16_t, bfloat16_t, float,
                             __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(bfloat16_t, bfloat16_t, float,
                             __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(bfloat16_t, bfloat16_t, float,
                             __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(itype, otype, ctype, itype_to_ctype, \
                                          ctype_to_otype)                      \
                                                                               \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {      \
    return ctype_to_otype(                                                     \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {             \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));             \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_and_fill_down(itype data, itype filling_data,  \
                                              ushort delta, ushort modulo) {   \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo));   \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_and_fill_down(itype data, itype filling_data,  \
                                              ushort delta) {                  \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta,             \
        __metal_get_simdgroup_size(ushort())));                                \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_and_fill_up(itype data, itype filling_data,    \
                                            ushort delta, ushort modulo) {     \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                    \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo));   \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_and_fill_up(itype data, itype filling_data,    \
                                            ushort delta) {                    \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                    \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta,             \
        __metal_get_simdgroup_size(ushort())));                                \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {               \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));               \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {        \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));        \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {          \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {                 \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));                 \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {                 \
    return ctype_to_otype(                                                     \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));                 \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(bfloat16_t, bfloat16_t, uint16_t,
                                  bfloat16_to_uint16, uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename OutT, typename InT> struct TransformNone {
  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  static METAL_FUNC OutT apply(InT x, OutT) { return static_cast<OutT>(x); }
};

template <typename OutT, typename InT> struct TransformAdd {
  TransformAdd(const float, const float) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT> struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_)
      : alpha(alpha_), beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }

  METAL_FUNC OutT apply(InT x, OutT c) const {
    return static_cast<OutT>(x * alpha + (beta * c));
  }
};

template <typename T> struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2 swizzle(uint3 tid [[threadgroup_position_in_grid]],
                                 const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

} // namespace steel
} // namespace mlx

METAL_FUNC ulong2 elem_to_loc_broadcast(uint elem, constant const int *shape,
                                        constant const int64_t *a_strides,
                                        constant const int64_t *b_strides,
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

METAL_FUNC ulong3 elem_to_loc_broadcast(uint elem, constant const int *shape,
                                        constant const int64_t *a_strides,
                                        constant const int64_t *b_strides,
                                        constant const int64_t *c_strides,
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

template <int val> using Int = integral_constant<int, val>;

#pragma METAL internals : enable

namespace metal {

template <typename T> struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T> constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts> struct make_void {
  typedef void type;
};

template <typename... Ts> using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T> struct pointer_element {};

template <typename T> struct pointer_element<thread T *> {
  using type = remove_cv_t<T>;
};
template <typename T> struct pointer_element<device T *> {
  using type = remove_cv_t<T>;
};
template <typename T> struct pointer_element<constant T *> {
  using type = remove_cv_t<T>;
};
template <typename T> struct pointer_element<threadgroup T *> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename RInt, typename CInt> struct Shape2D {
  RInt r;
  CInt c;

  Shape2D(RInt r_, CInt c_) : r(r_), c(c_) {}
};

template <typename Shape, typename Layout> struct Layout2D {
  Shape shape;
  Layout layout;
};

template <typename T, int kFragRows_, int kFragCols_> struct BaseMMAFrag {
  static_assert(kFragRows_ == 8,
                "Only 8 x 8 fragment matrices are currently supported");
  static_assert(kFragCols_ == 8,
                "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T> struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;

  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST int kElemRows = 1;
  STEEL_CONST int kElemCols = 2;

  static_assert(kElemRows * kElemCols == kElemsPerFrag,
                "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void load(thread frag_type &dst, SrcPtrType src,
                                        StrX str_x, StrY str_y) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <typename SrcPtrType, typename StrX, typename StrY, typename LimX,
            typename LimY, typename OffX, typename OffY>
  METAL_FUNC static constexpr void
  load_safe(thread frag_type &dst, SrcPtrType src, StrX str_x, StrY str_y,
            LimX lim_x, LimY lim_y, OffX off_x = Int<0>{},
            OffY off_y = Int<0>{}) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[(off_x + i) * str_x + (off_x + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void
  store(const thread frag_type &src, DstPtrType dst, StrX str_x, StrY str_y) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] =
            static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY, typename LimX,
            typename LimY, typename OffX, typename OffY>
  METAL_FUNC static constexpr void
  store_safe(const thread frag_type &src, DstPtrType dst, StrX str_x,
             StrY str_y, LimX lim_x, LimY lim_y, OffX off_x = Int<0>{},
             OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  METAL_FUNC static constexpr void mma(thread frag_type &D, thread frag_type &A,
                                       thread frag_type &B,
                                       thread frag_type &C) {
    mat_type D_mat;
    mat_type A_mat;
    mat_type B_mat;
    mat_type C_mat;

    reinterpret_cast<thread frag_type &>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread frag_type &>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread frag_type &>(C_mat.thread_elements()) = C;

    mma(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type &>(D_mat.thread_elements());
  }

  METAL_FUNC static constexpr void mma(thread mat_type &D, thread mat_type &A,
                                       thread mat_type &B, thread mat_type &C) {
    simdgroup_multiply_accumulate(D, A, B, C);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(thread const frag_type &inp_vals,
                                              thread T *reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(thread frag_type &inp_vals,
                                              thread T *row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <typename T, int kTileRows_, int kTileCols_,
          class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
  STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
  STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  STEEL_CONST int kTileRows = kTileRows_;
  STEEL_CONST int kTileCols = kTileCols_;

  STEEL_CONST int kRows = kTileRows * kFragRows;
  STEEL_CONST int kCols = kTileCols * kFragCols;

  STEEL_CONST int kNumFrags = kTileRows * kTileCols;
  STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  STEEL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags] = {frag_type(0)};

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type &frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type &frag_at(const short i,
                                                       const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC mat_type mat_at(const short i, const short j) {
    mat_type val_mat;
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < kElemsPerFrag; ++ii) {
      val_mat.thread_elements()[ii] = frag_at(i, j)[ii];
    }
    return val_mat;
  }

  METAL_FUNC thread elem_type *elems() {
    return reinterpret_cast<thread elem_type *>(val_frags);
  }

  METAL_FUNC const thread elem_type *elems() const {
    return reinterpret_cast<const thread elem_type *>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(frag_at(i, j),
                                           &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(frag_at(i, j),
                                           &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U *src) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(frag_at(i, j),
                        &(src[(i * kFragRows) * w_x * str_x +
                              (j * kFragCols) * w_y * str_y]),
                        Int<str_x>{}, Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U *dst) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(frag_at(i, j),
                         &(dst[(i * kFragRows) * w_x * str_x +
                               (j * kFragCols) * w_y * str_y]),
                         Int<str_x>{}, Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load(const device U *src, const int ld) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(src[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]), ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U *dst, const int ld) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]), ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void load_safe(const device U *src, const int ld,
                            const short2 src_tile_dims) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load_safe(frag_at(i, j), src, ld, Int<1>{}, src_tile_dims.y,
                             src_tile_dims.x, (i * kFragRows) * w_x,
                             (j * kFragCols) * w_y);
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(device U *dst, const int ld,
                             const short2 dst_tile_dims) const {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(frag_at(i, j), dst, ld, Int<1>{}, dst_tile_dims.y,
                              dst_tile_dims.x, (i * kFragRows) * w_x,
                              (j * kFragCols) * w_y);
      }
    }
  }
};

template <typename T, typename U, int M, int N, int K>
METAL_FUNC void
tile_matmad(thread MMATile<T, M, N> &D, thread MMATile<U, M, K> &A,
            thread MMATile<U, K, N> &B, thread MMATile<T, M, N> &C) {
  STEEL_PRAGMA_UNROLL
  for (short k = 0; k < K; ++k) {
    STEEL_PRAGMA_UNROLL
    for (short m = 0; m < M; ++m) {
      STEEL_PRAGMA_UNROLL
      for (short n = 0; n < N; ++n) {
        short n_serp = (m % 2) ? (N - 1 - n) : n;
        MMATile<T, M, N>::MMAFrag_t::mma(D.frag_at(m, n_serp), A.frag_at(m, k),
                                         B.frag_at(k, n_serp),
                                         C.frag_at(m, n_serp));
      }
    }
  }
}

template <typename T, typename U, int BM, int BN, int BK, int WM, int WN,
          bool transpose_a, bool transpose_b, short lda_tgp, short ldb_tgp,
          typename AccumType = float,
          typename Epilogue = TransformNone<U, AccumType>>
struct BlockMMA {
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TM_stride = kFragSize * WM;
  // Warp tile simdgroup matrix strides along M
  STEEL_CONST short TN_stride = kFragSize * WN;

  // Warp tile size along M
  STEEL_CONST short TM = BM / TM_stride;
  // Warp tile size along N
  STEEL_CONST short TN = BN / TN_stride;

  // Threadgroup A strides
  STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp; // M
  STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1; // K

  // Threadgroup B strides
  STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp; // K
  STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1; // N

  // Threadgroup strides along K
  STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
  STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

  // Offsets within threadgroup
  short sm;
  short sn;

  short As_offset;
  short Bs_offset;

  /* Constructor */
  METAL_FUNC BlockMMA(ushort simd_group_id [[simdgroup_index_in_threadgroup]],
                      ushort simd_lane_id [[thread_index_in_simdgroup]]) {
    // Determine thread position in simdgroup matrix
    short tm = kFragSize * (simd_group_id / WN);
    short tn = kFragSize * (simd_group_id % WN);

    short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = simd_coord.y;
    sn = simd_coord.x;

    // Determine thread and simdgroup offset
    As_offset = (tm + sm) * A_str_m + (sn)*A_str_k; // M, K
    Bs_offset = (sm)*B_str_k + (tn + sn) * B_str_n; // K, N

    sm += tm;
    sn += tn;
  }

  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T *As, const threadgroup T *Bs) {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);

      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);

      simdgroup_barrier(mem_flags::mem_none);

      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U *D, const int ldd) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;

    Ctile.template store<U, WM, WN>(D, ldd);
  }

  METAL_FUNC void store_result_safe(device U *D, const int ldd,
                                    short2 dst_tile_dims) {
    // Apply epilogue
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
    }

    // Adjust for simdgroup and thread location
    D += sm * ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
  }

  /* Apply epilogue */
  template <typename UnaryEpilogue>
  METAL_FUNC void apply_epilogue(thread const UnaryEpilogue &epilogue_op) {
    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
      Ctile.elems()[i] = epilogue_op.apply(Ctile.elems()[i]);
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void apply_epilogue(const device U *C, const int ldc,
                                 const int fdc,
                                 thread const BinaryEpilogue &epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto &accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < decltype(Ctile)::kElemsPerFrag; k++) {
          accum[k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  /* Apply epilogue */
  template <typename BinaryEpilogue>
  METAL_FUNC void
  apply_epilogue_safe(const device U *C, const int ldc, const int fdc,
                      short2 dst_tile_dims,
                      thread const BinaryEpilogue &epilogue_op) {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread auto &accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;

        constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

        // Read C
        U c_elems[kelems] = {0};

        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          if ((j * TN_stride + k) < dst_tile_dims.x) {
            c_elems[k] = C[offset_c + k * fdc];
          }
        }

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          accum[k] = epilogue_op.apply(accum[k], c_elems[k]);
        }
      }
    }
  }

  /* Store results from simdgroup_matrix results into device memory */
  METAL_FUNC void store_result(device U *D, const int ldd, const device U *C,
                               const int ldc, const int fdc,
                               thread const Epilogue &epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    // Loop over all simdgroup tiles
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        // Get accumulated result and associated offset in C
        thread const auto &accum = Ctile.frag_at(i, j);
        int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
        int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

        // Apply epilogue
        STEEL_PRAGMA_UNROLL
        for (short k = 0; k < kelems; k++) {
          D[offset_d + k] = epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
        }
      }
    }
  }

  METAL_FUNC void store_result_safe(device U *D, const int ldd,
                                    const device U *C, const int ldc,
                                    const int fdc, short2 dst_tile_dims,
                                    thread const Epilogue &epilogue_op) const {
    // Adjust for simdgroup and thread location
    C += (sm)*ldc + (sn)*fdc;
    D += (sm)*ldd + sn;
    dst_tile_dims -= short2(sn, sm);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    constexpr short kelems = decltype(Ctile)::kElemsPerFrag;

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      if (i * TM_stride < dst_tile_dims.y) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          // Get accumulated result and associated offset in C
          thread const auto &accum = Ctile.frag_at(i, j);
          int offset_c = (i * TM_stride) * ldc + (j * TN_stride) * fdc;
          int offset_d = (i * TM_stride) * ldd + (j * TN_stride);

          // Apply epilogue
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < kelems; k++) {
            if ((j * TN_stride + k) < dst_tile_dims.x) {
              D[offset_d + k] =
                  epilogue_op.apply(accum[k], C[offset_c + k * fdc]);
            }
          }
        }
      }
    }
  }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename T, short BROWS, short BCOLS, short dst_ld,
          short reduction_dim, short tgp_size, short alignment = 1,
          short n_reads = (BCOLS * BROWS) / (tgp_size),
          short TCOLS = BCOLS / n_reads, short TROWS = tgp_size / TCOLS>
struct BlockLoader {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T *dst;
  const device T *src;

  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  /* Constructor */
  METAL_FUNC
  BlockLoader(const device T *src_, const int src_ld_, threadgroup T *dst_,
              ushort simd_group_id [[simdgroup_index_in_threadgroup]],
              ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_), tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id), bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)), dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp &op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = op.apply(dst[i * dst_ld + j]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector *)(&dst[i * dst_ld])) =
          *((const device ReadVector *)(&src[i * src_ld]));
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() { src += tile_stride; }
};

template <int R, int C> struct CShape {
  STEEL_CONST int kRows = R;
  STEEL_CONST int kCols = C;
};

template <typename T, short BROWS, short BCOLS, short kDstStrRow,
          short kDstStrCol, short reduction_dim, short tgp_size,
          short n_reads = (BCOLS * BROWS) / (tgp_size),
          short TCOLS = BCOLS / n_reads, short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short thread_idx;
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T *dst;
  const device T *src;

  /* Constructor */
  METAL_FUNC
  BlockLoaderT(const device T *src_, const int src_ld_, threadgroup T *dst_,
               ushort simd_group_id [[simdgroup_index_in_threadgroup]],
               ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_), tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
        thread_idx(simd_group_id * 32 + simd_lane_id), bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(src_ + bi * src_ld + bj) {}

  /* Apply operation to threadgroup without bound checking */
  template <typename UnaryOp>
  METAL_FUNC void apply_inplace_op(thread const UnaryOp &op) const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] =
            op.apply(dst[i * kDstStrRow + j * kDstStrCol]);
      }
    }
  }

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    // Skip loading if thread has no valid reads
    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    // Use fast thread memory for bound checks
    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      // Make sure tmp_idx only contains valid indices
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }

      // Read valid indices into tmp_val
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }

      // Zero out uneeded values
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }

      // Copy values to threadgroup memory
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() { src += tile_stride; }
};

} // namespace steel
} // namespace mlx

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U> struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                        \
  template <> struct Limits<type> {                                            \
    static constexpr constant type max = metal::numeric_limits<type>::max();   \
    static constexpr constant type min = metal::numeric_limits<type>::min();   \
    static constexpr constant type finite_max =                                \
        metal::numeric_limits<type>::max();                                    \
    static constexpr constant type finite_min =                                \
        metal::numeric_limits<type>::min();                                    \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)                                          \
  template <> struct Limits<type> {                                            \
    static constexpr constant type max =                                       \
        metal::numeric_limits<type>::infinity();                               \
    static constexpr constant type min =                                       \
        -metal::numeric_limits<type>::infinity();                              \
    static constexpr constant type finite_max =                                \
        metal::numeric_limits<type>::max();                                    \
    static constexpr constant type finite_min =                                \
        -metal::numeric_limits<type>::max();                                   \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <> struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(IdxT elem, constant const int *shape,
                            constant const int64_t *strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(uint3 elem, constant const int *shape,
                            constant const int64_t *strides, int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_1(uint elem, constant const int64_t &stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_2(uint2 elem, constant const int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_3(uint3 elem, constant const int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
         elem.z * IdxT(strides[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 2> elem_to_loc_2_nd(uint3 elem, constant const int *shape,
                                         constant const int64_t *a_strides,
                                         constant const int64_t *b_strides,
                                         int ndim) {
  vec<IdxT, 2> loc = {IdxT(elem.x * IdxT(a_strides[ndim - 1]) +
                           IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
                      IdxT(elem.x * IdxT(b_strides[ndim - 1]) +
                           elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 3> elem_to_loc_3_nd(uint3 elem, constant const int *shape,
                                         constant const int64_t *a_strides,
                                         constant const int64_t *b_strides,
                                         constant const int64_t *c_strides,
                                         int ndim) {
  vec<IdxT, 3> loc = {IdxT(elem.x * IdxT(a_strides[ndim - 1])) +
                          IdxT(elem.y * IdxT(a_strides[ndim - 2])),
                      IdxT(elem.x * IdxT(b_strides[ndim - 1])) +
                          IdxT(elem.y * IdxT(b_strides[ndim - 2])),
                      IdxT(elem.x * IdxT(c_strides[ndim - 1])) +
                          IdxT(elem.y * IdxT(c_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    loc.z += l * IdxT(c_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int *shape, const constant int64_t *strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int *shape, const constant int64_t *strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() { return offset; }
};

template <typename OffsetT> struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) : dim(dim) {}

  void next(const constant int *shape, const constant int64_t *strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int *shape, const constant int64_t *strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() { return offset; }
};

template <typename OffsetT> struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) {}

  void next(const constant int *, const constant int64_t *strides) {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int *, const constant int64_t *strides) {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() { return offset; }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U> inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}

inline uint64_t simd_shuffle_up(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_up(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_up(bool data, uint16_t delta) {
  return simd_shuffle_up(static_cast<uint32_t>(data), delta);
}

inline uint64_t simd_shuffle_and_fill_up(uint64_t data, uint64_t filling,
                                         uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline int64_t simd_shuffle_and_fill_up(int64_t data, int64_t filling,
                                        uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(static_cast<uint32_t>(data),
                                  static_cast<uint32_t>(filling), delta);
}

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

// std::conditional is not included with Metal
template <bool condition, typename T, typename U> struct ConditionalType {
  using type = U;
};

template <typename T, typename U> struct ConditionalType<true, T, U> {
  using type = T;
};

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T *x, thread U *x_thread) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
             x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T *x, thread U *x_thread, int N) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
             x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t *w, const thread U *x_thread, U scale,
              U bias, U sum) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4 * i] * (w[i] & 0x03) +
                x_thread[4 * i + 1] * (w[i] & 0x0c) +
                x_thread[4 * i + 2] * (w[i] & 0x30) +
                x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t *ws = (const device uint16_t *)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) +
                x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(const device uint8_t *w, const thread U *x_thread, U scale,
                   U bias, U sum, int N) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum += (x_thread[4 * i] * (w[i] & 0x03) +
                x_thread[4 * i + 1] * (w[i] & 0x0c) +
                x_thread[4 * i + 2] * (w[i] & 0x30) +
                x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t *ws = (const device uint16_t *)w;
    for (int i = 0; i < (N / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) +
                x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
                x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void qouter(const thread uint8_t *w, U x, U scale, U bias,
                   thread U *result) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }

  } else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int N, int bits>
inline void dequantize(const device uint8_t *w, U scale, U bias,
                       threadgroup U *w_local) {
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / static_cast<U>(4.0f),
              scale / static_cast<U>(16.0f), scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x7) * scale + bias;
      w_local[1] = ((w[0] & 0x38) >> 3) * scale + bias;
      w_local[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
      w_local[3] = ((w[1] & 0xe) >> 1) * scale + bias;
      w_local[4] = ((w[1] & 0x70) >> 4) * scale + bias;
      w_local[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
      w_local[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
      w_local[7] = ((w[2] & 0xe0) >> 5) * scale + bias;
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;

      w_local[0] = (w[0] & 0x3f) * scale + bias;
      w_local[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
      w_local[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
      w_local[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <typename T, short BROWS, short BCOLS, short dst_ld,
          short reduction_dim, short tgp_size, short group_size, short bits>
struct QuantizedBlockLoader {
  static_assert(BCOLS <= group_size,
                "The group size should be larger than the columns");
  static_assert(group_size % BCOLS == 0,
                "The group size should be divisible by the columns");
  static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8,
                "Template undefined for bits not in {2, 3, 4, 6, 8}");

  MLX_MTL_CONST short pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  MLX_MTL_CONST short bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T *dst;
  const device uint8_t *src;
  const device T *scales;
  const device T *biases;

  QuantizedBlockLoader(const device uint8_t *src_, const device T *scales_,
                       const device T *biases_, const int src_ld_,
                       threadgroup T *dst_,
                       ushort simd_group_id [[simdgroup_index_in_threadgroup]],
                       ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(reduction_dim
                        ? BCOLS_PACKED * bytes_per_pack
                        : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0), group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(src + i * bytes_per_pack, scale, bias,
                                       dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t *)(src + i * bytes_per_pack), scale, bias,
          dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T, int group_size, int bits, int D>
METAL_FUNC void qmv_quad_impl(const device uint32_t *w, const device T *scales,
                              const device T *biases, const device T *x,
                              device T *y, constant int &in_vec_size,
                              const constant int &out_vec_size,
                              uint3 tid [[threadgroup_position_in_grid]],
                              uint quad_gid [[quadgroup_index_in_threadgroup]],
                              uint quad_lid [[thread_index_in_quadgroup]]) {
  constexpr int quads_per_simd = SIMD_SIZE / QUAD_SIZE;
  constexpr int pack_factor = 32 / bits;
  constexpr int values_per_thread = D / QUAD_SIZE;
  constexpr int packs_per_thread = values_per_thread / pack_factor;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int results_per_quadgroup = 8;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_quadgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.x * quads_per_simd * results_per_quadgroup + quad_gid;

  w += out_row * in_vec_size_w + quad_lid * packs_per_thread;
  scales += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + quad_lid / scale_step_per_thread;
  x += tid.y * in_vec_size + quad_lid * values_per_thread;
  y += tid.y * out_vec_size + out_row;

  U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

  for (int row = 0; row < results_per_quadgroup; row++) {
    auto wl =
        (const device uint8_t *)(w + row * in_vec_size_w * quads_per_simd);
    const device T *sl = scales + row * in_vec_size_g * quads_per_simd;
    const device T *bl = biases + row * in_vec_size_g * quads_per_simd;

    U s = sl[0];
    U b = bl[0];
    if (row * quads_per_simd + out_row < out_vec_size) {
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
  }

  for (int row = 0; row < results_per_quadgroup; row++) {
    result[row] = quad_sum(result[row]);
    if (quad_lid == 0 && row * quads_per_simd + out_row < out_vec_size) {
      y[row * quads_per_simd] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_fast_impl(const device uint32_t *w, const device T *scales,
                              const device T *biases, const device T *x,
                              device T *y, const constant int &in_vec_size,
                              const constant int &out_vec_size,
                              uint3 tid [[threadgroup_position_in_grid]],
                              uint simd_gid [[simdgroup_index_in_threadgroup]],
                              uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t *ws = (const device uint8_t *)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.x * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.y * in_vec_size + simd_lid * values_per_thread;
  y += tid.y * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t *)(ws + row * in_vec_size_w);
      const device T *sl = scales + row * in_vec_size_g;
      const device T *bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void qmv_impl(const device uint32_t *w, const device T *scales,
                         const device T *biases, const device T *x, device T *y,
                         const constant int &in_vec_size,
                         const constant int &out_vec_size,
                         uint3 tid [[threadgroup_position_in_grid]],
                         uint simd_gid [[simdgroup_index_in_threadgroup]],
                         uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 4 : 3;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t *ws = (const device uint8_t *)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.x * (num_simdgroups * results_per_simdgroup) +
                      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.y * in_vec_size + simd_lid * values_per_thread;
    y += tid.y * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; out_row + row < out_vec_size; row++) {
        auto wl = (const device uint8_t *)(ws + row * in_vec_size_w);
        const device T *sl = scales + row * in_vec_size_g;
        const device T *bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining =
        clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
              0, values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread,
                                                              remaining);

      for (int row = 0; out_row + row < out_vec_size; row++) {
        auto wl = (const device uint8_t *)(ws + row * in_vec_size_w);
        const device T *sl = scales + row * in_vec_size_g;
        const device T *bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }
    }

    for (int row = 0; out_row + row < out_vec_size; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
          simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.y * in_vec_size + simd_lid * values_per_thread;
    y += tid.y * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t *)(ws + row * in_vec_size_w);
        const device T *sl = scales + row * in_vec_size_g;
        const device T *bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining =
        clamp(static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
              0, values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread,
                                                              remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t *)(ws + row * in_vec_size_w);
        const device T *sl = scales + row * in_vec_size_g;
        const device T *bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b,
                                                             sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

template <typename T, const int group_size, const int bits>
METAL_FUNC void qvm_impl(const device uint32_t *w, const device T *scales,
                         const device T *biases, const device T *x, device T *y,
                         const int in_vec_size, const int out_vec_size,
                         uint3 tid [[threadgroup_position_in_grid]],
                         uint simd_gid [[simdgroup_index_in_threadgroup]],
                         uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int num_simdgroups = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 32 / bits;
  constexpr int bytes_per_pack = power_of_2_bits ? 1 : 3;
  constexpr int tn = 32 / pack_factor;
  constexpr int block_size = SIMD_SIZE;

  using W_T =
      typename ConditionalType<power_of_2_bits, uint32_t, uint8_t>::type;
  const device W_T *ws = (const device W_T *)w;

  typedef float U;
  typedef struct {
    W_T wi[tn * bytes_per_pack];
  } vec_w;

  thread vec_w w_local;
  thread U result[tn * pack_factor] = {0};
  thread U scale = 1;
  thread U bias = 0;
  thread U x_local = 0;

  // Adjust positions
  const int out_vec_size_w = out_vec_size * bytes_per_pack / pack_factor;
  const int out_vec_size_g = out_vec_size / group_size;
  int out_col = pack_factor * tn * (tid.x * num_simdgroups + simd_gid);
  ws += out_col * bytes_per_pack / pack_factor + simd_lid * out_vec_size_w;
  scales += out_col / group_size + simd_lid * out_vec_size_g;
  biases += out_col / group_size + simd_lid * out_vec_size_g;
  x += tid.y * in_vec_size + simd_lid;
  y += tid.y * out_vec_size + out_col;

  if (out_col >= out_vec_size) {
    return;
  }

  // Loop over in_vec in blocks of block_size
  int remaining = in_vec_size % block_size;
  if (remaining == 0) {
    for (int i = 0; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w *)ws);
      qouter<U, tn * pack_factor, bits>((thread uint8_t *)&w_local, x_local,
                                        scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
  } else {
    for (int i = block_size; i < in_vec_size; i += block_size) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w *)ws);

      qouter<U, tn * pack_factor, bits>((thread uint8_t *)&w_local, x_local,
                                        scale, bias, result);

      x += block_size;
      scales += block_size * out_vec_size_g;
      biases += block_size * out_vec_size_g;
      ws += block_size * out_vec_size_w;
    }
    if (static_cast<int>(simd_lid) < remaining) {
      x_local = *x;
      scale = *scales;
      bias = *biases;
      w_local = *((device vec_w *)ws);
    } else {
      x_local = 0;
      scale = 0;
      bias = 0;
    }
    qouter<U, tn * pack_factor, bits>((thread uint8_t *)&w_local, x_local,
                                      scale, bias, result);
  }

// Accumulate in the simdgroup
#pragma clang loop unroll(full)
  for (int k = 0; k < tn * pack_factor; k++) {
    result[k] = simd_sum(result[k]);
  }

  // Store the result
  if (simd_lid == 0) {
#pragma clang loop unroll(full)
    for (int k = 0; k < tn * pack_factor; k++) {
      y[k] = static_cast<T>(result[k]);
    }
  }
}

template <typename T, const int group_size, const int bits,
          const bool aligned_N, const int BM = 32, const int BK = 32,
          const int BN = 32>
METAL_FUNC void qmm_t_impl(const device uint32_t *w, const device T *scales,
                           const device T *biases, const device T *x,
                           device T *y, threadgroup T *Xs, threadgroup T *Ws,
                           const constant int &K, const constant int &N,
                           const constant int &M,
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int bytes_per_pack = (bits == 3 || bits == 6) ? 3 : 1;

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true,
                                     BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE,
                           group_size, bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t *)w;

  x += y_row * K;
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * N + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, N, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T, const int group_size, const int bits, const int BM = 32,
          const int BK = 32, const int BN = 32>
METAL_FUNC void qmm_n_impl(const device uint32_t *w, const device T *scales,
                           const device T *biases, const device T *x,
                           device T *y, threadgroup T *Xs, threadgroup T *Ws,
                           const constant int &K, const constant int &N,
                           const constant int &M,
                           uint3 tid [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint simd_gid [[simdgroup_index_in_threadgroup]],
                           uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int bytes_per_pack = power_of_2_bits ? 1 : 3;

  // Instantiate the appropriate BlockMMA and Loader
  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, false,
                                     BK_padded, BN_padded>;
  using loader_x_t = mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1,
                                             WM * WN * SIMD_SIZE, 1, 4>;
  using loader_w_t =
      QuantizedBlockLoader<T, BK, BN, BN_padded, 0, WM * WN * SIMD_SIZE,
                           group_size, bits>;

  auto wl = (const device uint8_t *)w;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  x += y_row * K;
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * N + y_col;

  // Make the x loader and mma operation
  const short num_els = min(BM, M - y_row);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, num_els));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if ((K % BK) != 0) {
      const int k_blocks = K / BK;
      for (int k = 0; k < k_blocks; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
      const short num_k = K - k_blocks * BK;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_x.load_safe(short2(num_k, BM));
      loader_w.load_safe(short2(BN, num_k));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(Xs, Ws);
    } else {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM) {
    mma_op.store_result_safe(y, N, short2(BN, num_els));
  } else {
    mma_op.store_result(y, N);
  }
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T *&x, const device uint32_t *&w, const device T *&scales,
    const device T *&biases, device T *&y, int output_stride,
    const constant int &x_batch_ndims, const constant int *x_shape,
    const constant int64_t *x_strides, const constant int &w_batch_ndims,
    const constant int *w_shape, const constant int64_t *w_strides,
    const constant int64_t *s_strides, const constant int64_t *b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(w_idx, w_shape, w_strides, s_strides,
                                       b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T *&x, const device uint32_t *&w, const device T *&scales,
    const device T *&biases, const device uint32_t *lhs_indices,
    const device uint32_t *rhs_indices, device T *&y, int output_stride,
    const constant int &batch_ndims, const constant int *batch_shape,
    const constant int64_t *lhs_strides, const constant int64_t *rhs_strides,
    const constant int &x_batch_ndims, const constant int *x_shape,
    const constant int64_t *x_strides, const constant int &w_batch_ndims,
    const constant int *w_shape, const constant int64_t *w_strides,
    const constant int64_t *s_strides, const constant int64_t *b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(tid.z, batch_shape, lhs_strides,
                                       rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(w_idx, w_shape, w_strides, s_strides,
                                       b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T, int group_size, int bits, int D, bool batched>
[[kernel]] void qmv_quad(const device uint32_t *w [[buffer(0)]],
                         const device T *scales [[buffer(1)]],
                         const device T *biases [[buffer(2)]],
                         const device T *x [[buffer(3)]],
                         device T *y [[buffer(4)]],
                         const constant int &in_vec_size [[buffer(5)]],
                         const constant int &out_vec_size [[buffer(6)]],
                         const constant int &x_batch_ndims [[buffer(7)]],
                         const constant int *x_shape [[buffer(8)]],
                         const constant int64_t *x_strides [[buffer(9)]],
                         const constant int &w_batch_ndims [[buffer(10)]],
                         const constant int *w_shape [[buffer(11)]],
                         const constant int64_t *w_strides [[buffer(12)]],
                         const constant int64_t *s_strides [[buffer(13)]],
                         const constant int64_t *b_strides [[buffer(14)]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         uint quad_gid [[quadgroup_index_in_threadgroup]],
                         uint quad_lid [[thread_index_in_quadgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(x, w, scales, biases, y, out_vec_size * M,
                             x_batch_ndims, x_shape, x_strides, w_batch_ndims,
                             w_shape, w_strides, s_strides, b_strides, tid);
  }
  qmv_quad_impl<T, group_size, bits, D>(w, scales, biases, x, y, in_vec_size,
                                        out_vec_size, tid, quad_gid, quad_lid);
}

template <typename T, int group_size, int bits, bool batched>
[[kernel]] void qmv_fast(const device uint32_t *w [[buffer(0)]],
                         const device T *scales [[buffer(1)]],
                         const device T *biases [[buffer(2)]],
                         const device T *x [[buffer(3)]],
                         device T *y [[buffer(4)]],
                         const constant int &in_vec_size [[buffer(5)]],
                         const constant int &out_vec_size [[buffer(6)]],
                         const constant int &x_batch_ndims [[buffer(7)]],
                         const constant int *x_shape [[buffer(8)]],
                         const constant int64_t *x_strides [[buffer(9)]],
                         const constant int &w_batch_ndims [[buffer(10)]],
                         const constant int *w_shape [[buffer(11)]],
                         const constant int64_t *w_strides [[buffer(12)]],
                         const constant int64_t *s_strides [[buffer(13)]],
                         const constant int64_t *b_strides [[buffer(14)]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         uint simd_gid [[simdgroup_index_in_threadgroup]],
                         uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(x, w, scales, biases, y, out_vec_size * M,
                             x_batch_ndims, x_shape, x_strides, w_batch_ndims,
                             w_shape, w_strides, s_strides, b_strides, tid);
  }
  qmv_fast_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                     out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void qmv(const device uint32_t *w [[buffer(0)]],
                    const device T *scales [[buffer(1)]],
                    const device T *biases [[buffer(2)]],
                    const device T *x [[buffer(3)]], device T *y [[buffer(4)]],
                    const constant int &in_vec_size [[buffer(5)]],
                    const constant int &out_vec_size [[buffer(6)]],
                    const constant int &x_batch_ndims [[buffer(7)]],
                    const constant int *x_shape [[buffer(8)]],
                    const constant int64_t *x_strides [[buffer(9)]],
                    const constant int &w_batch_ndims [[buffer(10)]],
                    const constant int *w_shape [[buffer(11)]],
                    const constant int64_t *w_strides [[buffer(12)]],
                    const constant int64_t *s_strides [[buffer(13)]],
                    const constant int64_t *b_strides [[buffer(14)]],
                    uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(x, w, scales, biases, y, out_vec_size * M,
                             x_batch_ndims, x_shape, x_strides, w_batch_ndims,
                             w_shape, w_strides, s_strides, b_strides, tid);
  }
  qmv_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, bool batched>
[[kernel]] void qvm(const device uint32_t *w [[buffer(0)]],
                    const device T *scales [[buffer(1)]],
                    const device T *biases [[buffer(2)]],
                    const device T *x [[buffer(3)]], device T *y [[buffer(4)]],
                    const constant int &in_vec_size [[buffer(5)]],
                    const constant int &out_vec_size [[buffer(6)]],
                    const constant int &x_batch_ndims [[buffer(7)]],
                    const constant int *x_shape [[buffer(8)]],
                    const constant int64_t *x_strides [[buffer(9)]],
                    const constant int &w_batch_ndims [[buffer(10)]],
                    const constant int *w_shape [[buffer(11)]],
                    const constant int64_t *w_strides [[buffer(12)]],
                    const constant int64_t *s_strides [[buffer(13)]],
                    const constant int64_t *b_strides [[buffer(14)]],
                    uint3 tid [[threadgroup_position_in_grid]],
                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                    uint simd_lid [[thread_index_in_simdgroup]]) {
  if (batched) {
    int M = x_shape[x_batch_ndims];
    adjust_matrix_offsets<T>(x, w, scales, biases, y, out_vec_size * M,
                             x_batch_ndims, x_shape, x_strides, w_batch_ndims,
                             w_shape, w_strides, s_strides, b_strides, tid);
  }
  qvm_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, int split_k = 32>
[[kernel]] void qvm_split_k(const device uint32_t *w [[buffer(0)]],
                            const device T *scales [[buffer(1)]],
                            const device T *biases [[buffer(2)]],
                            const device T *x [[buffer(3)]],
                            device T *y [[buffer(4)]],
                            const constant int &in_vec_size [[buffer(5)]],
                            const constant int &out_vec_size [[buffer(6)]],
                            const constant int &x_batch_ndims [[buffer(7)]],
                            const constant int *x_shape [[buffer(8)]],
                            const constant int64_t *x_strides [[buffer(9)]],
                            const constant int &w_batch_ndims [[buffer(10)]],
                            const constant int *w_shape [[buffer(11)]],
                            const constant int64_t *w_strides [[buffer(12)]],
                            const constant int64_t *s_strides [[buffer(13)]],
                            const constant int64_t *b_strides [[buffer(14)]],
                            const constant int &final_block_size [[buffer(15)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(x, w, scales, biases, y, out_vec_size * M,
                           x_batch_ndims, x_shape, x_strides, w_batch_ndims,
                           w_shape, w_strides, s_strides, b_strides, tid);

  // When (in_vec_size % split_k != 0) the final block needs to be smaller
  int in_vec_size_adj =
      tid.z % split_k == split_k - 1 ? final_block_size : in_vec_size;

  qvm_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size_adj,
                                out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits,
          const bool aligned_N, const bool batched, const int BM = 32,
          const int BK = 32, const int BN = 32>
[[kernel]] void
qmm_t(const device uint32_t *w [[buffer(0)]],
      const device T *scales [[buffer(1)]],
      const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
      device T *y [[buffer(4)]], const constant int &K [[buffer(5)]],
      const constant int &N [[buffer(6)]], const constant int &M [[buffer(7)]],
      const constant int &x_batch_ndims [[buffer(8)]],
      const constant int *x_shape [[buffer(9)]],
      const constant int64_t *x_strides [[buffer(10)]],
      const constant int &w_batch_ndims [[buffer(11)]],
      const constant int *w_shape [[buffer(12)]],
      const constant int64_t *w_strides [[buffer(13)]],
      const constant int64_t *s_strides [[buffer(14)]],
      const constant int64_t *b_strides [[buffer(15)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint lid [[thread_index_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(x, w, scales, biases, y, M * N, x_batch_ndims,
                             x_shape, x_strides, w_batch_ndims, w_shape,
                             w_strides, s_strides, b_strides, tid);
  }
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, const bool batched,
          const int BM = 32, const int BK = 32, const int BN = 32>
[[kernel]] void
qmm_n(const device uint32_t *w [[buffer(0)]],
      const device T *scales [[buffer(1)]],
      const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
      device T *y [[buffer(4)]], const constant int &K [[buffer(5)]],
      const constant int &N [[buffer(6)]], const constant int &M [[buffer(7)]],
      const constant int &x_batch_ndims [[buffer(8)]],
      const constant int *x_shape [[buffer(9)]],
      const constant int64_t *x_strides [[buffer(10)]],
      const constant int &w_batch_ndims [[buffer(11)]],
      const constant int *w_shape [[buffer(12)]],
      const constant int64_t *w_strides [[buffer(13)]],
      const constant int64_t *s_strides [[buffer(14)]],
      const constant int64_t *b_strides [[buffer(15)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint lid [[thread_index_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(x, w, scales, biases, y, M * N, x_batch_ndims,
                             x_shape, x_strides, w_batch_ndims, w_shape,
                             w_strides, s_strides, b_strides, tid);
  }

  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void bs_qmv_fast(const device uint32_t *w [[buffer(0)]],
                            const device T *scales [[buffer(1)]],
                            const device T *biases [[buffer(2)]],
                            const device T *x [[buffer(3)]],
                            device T *y [[buffer(4)]],
                            const constant int &in_vec_size [[buffer(5)]],
                            const constant int &out_vec_size [[buffer(6)]],
                            const constant int &x_batch_ndims [[buffer(7)]],
                            const constant int *x_shape [[buffer(8)]],
                            const constant int64_t *x_strides [[buffer(9)]],
                            const constant int &w_batch_ndims [[buffer(10)]],
                            const constant int *w_shape [[buffer(11)]],
                            const constant int64_t *w_strides [[buffer(12)]],
                            const constant int64_t *s_strides [[buffer(13)]],
                            const constant int64_t *b_strides [[buffer(14)]],
                            const constant int &batch_ndims [[buffer(15)]],
                            const constant int *batch_shape [[buffer(16)]],
                            const device uint32_t *lhs_indices [[buffer(17)]],
                            const device uint32_t *rhs_indices [[buffer(18)]],
                            const constant int64_t *lhs_strides [[buffer(19)]],
                            const constant int64_t *rhs_strides [[buffer(20)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(x, w, scales, biases, lhs_indices, rhs_indices, y,
                           out_vec_size * M, batch_ndims, batch_shape,
                           lhs_strides, rhs_strides, x_batch_ndims, x_shape,
                           x_strides, w_batch_ndims, w_shape, w_strides,
                           s_strides, b_strides, tid);
  qmv_fast_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                     out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void
bs_qmv(const device uint32_t *w [[buffer(0)]],
       const device T *scales [[buffer(1)]],
       const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
       device T *y [[buffer(4)]], const constant int &in_vec_size [[buffer(5)]],
       const constant int &out_vec_size [[buffer(6)]],
       const constant int &x_batch_ndims [[buffer(7)]],
       const constant int *x_shape [[buffer(8)]],
       const constant int64_t *x_strides [[buffer(9)]],
       const constant int &w_batch_ndims [[buffer(10)]],
       const constant int *w_shape [[buffer(11)]],
       const constant int64_t *w_strides [[buffer(12)]],
       const constant int64_t *s_strides [[buffer(13)]],
       const constant int64_t *b_strides [[buffer(14)]],
       const constant int &batch_ndims [[buffer(15)]],
       const constant int *batch_shape [[buffer(16)]],
       const device uint32_t *lhs_indices [[buffer(17)]],
       const device uint32_t *rhs_indices [[buffer(18)]],
       const constant int64_t *lhs_strides [[buffer(19)]],
       const constant int64_t *rhs_strides [[buffer(20)]],
       uint3 tid [[threadgroup_position_in_grid]],
       uint simd_gid [[simdgroup_index_in_threadgroup]],
       uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(x, w, scales, biases, lhs_indices, rhs_indices, y,
                           out_vec_size * M, batch_ndims, batch_shape,
                           lhs_strides, rhs_strides, x_batch_ndims, x_shape,
                           x_strides, w_batch_ndims, w_shape, w_strides,
                           s_strides, b_strides, tid);
  qmv_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int bits>
[[kernel]] void
bs_qvm(const device uint32_t *w [[buffer(0)]],
       const device T *scales [[buffer(1)]],
       const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
       device T *y [[buffer(4)]], const constant int &in_vec_size [[buffer(5)]],
       const constant int &out_vec_size [[buffer(6)]],
       const constant int &x_batch_ndims [[buffer(7)]],
       const constant int *x_shape [[buffer(8)]],
       const constant int64_t *x_strides [[buffer(9)]],
       const constant int &w_batch_ndims [[buffer(10)]],
       const constant int *w_shape [[buffer(11)]],
       const constant int64_t *w_strides [[buffer(12)]],
       const constant int64_t *s_strides [[buffer(13)]],
       const constant int64_t *b_strides [[buffer(14)]],
       const constant int &batch_ndims [[buffer(15)]],
       const constant int *batch_shape [[buffer(16)]],
       const device uint32_t *lhs_indices [[buffer(17)]],
       const device uint32_t *rhs_indices [[buffer(18)]],
       const constant int64_t *lhs_strides [[buffer(19)]],
       const constant int64_t *rhs_strides [[buffer(20)]],
       uint3 tid [[threadgroup_position_in_grid]],
       uint simd_gid [[simdgroup_index_in_threadgroup]],
       uint simd_lid [[thread_index_in_simdgroup]]) {
  int M = x_shape[x_batch_ndims];
  adjust_matrix_offsets<T>(x, w, scales, biases, lhs_indices, rhs_indices, y,
                           out_vec_size * M, batch_ndims, batch_shape,
                           lhs_strides, rhs_strides, x_batch_ndims, x_shape,
                           x_strides, w_batch_ndims, w_shape, w_strides,
                           s_strides, b_strides, tid);
  qvm_impl<T, group_size, bits>(w, scales, biases, x, y, in_vec_size,
                                out_vec_size, tid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits,
          const bool aligned_N, const int BM = 32, const int BK = 32,
          const int BN = 32>
[[kernel]] void
bs_qmm_t(const device uint32_t *w [[buffer(0)]],
         const device T *scales [[buffer(1)]],
         const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
         device T *y [[buffer(4)]], const constant int &K [[buffer(5)]],
         const constant int &N [[buffer(6)]],
         const constant int &M [[buffer(7)]],
         const constant int &x_batch_ndims [[buffer(8)]],
         const constant int *x_shape [[buffer(9)]],
         const constant int64_t *x_strides [[buffer(10)]],
         const constant int &w_batch_ndims [[buffer(11)]],
         const constant int *w_shape [[buffer(12)]],
         const constant int64_t *w_strides [[buffer(13)]],
         const constant int64_t *s_strides [[buffer(14)]],
         const constant int64_t *b_strides [[buffer(15)]],
         const constant int &batch_ndims [[buffer(16)]],
         const constant int *batch_shape [[buffer(17)]],
         const device uint32_t *lhs_indices [[buffer(18)]],
         const device uint32_t *rhs_indices [[buffer(19)]],
         const constant int64_t *lhs_strides [[buffer(20)]],
         const constant int64_t *rhs_strides [[buffer(21)]],
         uint3 tid [[threadgroup_position_in_grid]],
         uint lid [[thread_index_in_threadgroup]],
         uint simd_gid [[simdgroup_index_in_threadgroup]],
         uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x, w, scales, biases, lhs_indices, rhs_indices, y, M * N, batch_ndims,
      batch_shape, lhs_strides, rhs_strides, x_batch_ndims, x_shape, x_strides,
      w_batch_ndims, w_shape, w_strides, s_strides, b_strides, tid);
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits, const int BM = 32,
          const int BK = 32, const int BN = 32>
[[kernel]] void
bs_qmm_n(const device uint32_t *w [[buffer(0)]],
         const device T *scales [[buffer(1)]],
         const device T *biases [[buffer(2)]], const device T *x [[buffer(3)]],
         device T *y [[buffer(4)]], const constant int &K [[buffer(5)]],
         const constant int &N [[buffer(6)]],
         const constant int &M [[buffer(7)]],
         const constant int &x_batch_ndims [[buffer(8)]],
         const constant int *x_shape [[buffer(9)]],
         const constant int64_t *x_strides [[buffer(10)]],
         const constant int &w_batch_ndims [[buffer(11)]],
         const constant int *w_shape [[buffer(12)]],
         const constant int64_t *w_strides [[buffer(13)]],
         const constant int64_t *s_strides [[buffer(14)]],
         const constant int64_t *b_strides [[buffer(15)]],
         const constant int &batch_ndims [[buffer(16)]],
         const constant int *batch_shape [[buffer(17)]],
         const device uint32_t *lhs_indices [[buffer(18)]],
         const device uint32_t *rhs_indices [[buffer(19)]],
         const constant int64_t *lhs_strides [[buffer(20)]],
         const constant int64_t *rhs_strides [[buffer(21)]],
         uint3 tid [[threadgroup_position_in_grid]],
         uint lid [[thread_index_in_threadgroup]],
         uint simd_gid [[simdgroup_index_in_threadgroup]],
         uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x, w, scales, biases, lhs_indices, rhs_indices, y, M * N, batch_ndims,
      batch_shape, lhs_strides, rhs_strides, x_batch_ndims, x_shape, x_strides,
      w_batch_ndims, w_shape, w_strides, s_strides, b_strides, tid);
  qmm_n_impl<T, group_size, bits, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_quantize(const device T *w [[buffer(0)]],
                                device uint8_t *out [[buffer(1)]],
                                device T *scales [[buffer(2)]],
                                device T *biases [[buffer(3)]],
                                uint2 index [[thread_position_in_grid]],
                                uint2 grid_dim [[threads_per_grid]]) {
  constexpr T eps = T(1e-7);
  constexpr int simd_size = 32;
  constexpr T n_bins = (1 << bits) - 1;
  constexpr int packs_per_int = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int writes_per_reduce = packs_per_int / values_per_reduce;
  constexpr int writes_per_pack =
      writes_per_reduce > 1 ? 1 : values_per_reduce / packs_per_int;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int bytes_per_pack = power_of_2_bits ? 1 : 3;

  static_assert(group_size % simd_size == 0,
                "Group size must be divisible by simd size.");

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t in_index = offset * values_per_reduce;
  size_t out_index = power_of_2_bits
                         ? offset * writes_per_pack
                         : offset * bytes_per_pack / writes_per_reduce;

  T w_thread[values_per_reduce];
  T w_min = Limits<T>::max;
  T w_max = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    T val = w[in_index + i];
    w_thread[i] = val;
    w_min = min(w_min, val);
    w_max = max(w_max, val);
  }

  w_min = simd_min(w_min);
  w_max = simd_max(w_max);

  T scale = max((w_max - w_min) / n_bins, eps);
  bool side = abs(w_min) > abs(w_max);
  scale = side ? scale : -scale;
  T edge = side ? w_min : w_max;
  T q0 = round(edge / scale);
  bool at_zero = q0 == 0.0f;
  scale = at_zero ? scale : edge / q0;
  T bias = at_zero ? T(0) : edge;

  // Write out the scales and biases
  size_t gindex = in_index / group_size;
  if (in_index % group_size == 0) {
    scales[gindex] = scale;
    biases[gindex] = bias;
  }

  // We accumulate 3 bytes worth for 3/6 bit so we need a uint32_t
  uint32_t output = 0;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);
    if (bits == 8) {
      output = val;
    } else {
      output += val << (bits * (i % packs_per_int));
    }

    if (packs_per_int < values_per_reduce &&
        i % packs_per_int == packs_per_int - 1) {
      out[out_index + i / packs_per_int] = output;
      output = 0;
    } else {
#pragma clang loop unroll(full)
      for (int j = 1; j < writes_per_reduce; j++) {
        uint8_t sval = simd_shuffle_down(val, j);
        output += sval << (bits * (j * values_per_reduce + i));
      }
    }
  }
  if (bits == 3 || bits == 6) {
    if (in_index % packs_per_int == 0 && out_index % bytes_per_pack == 0) {
      out[out_index] = output & 0xff;
      out[out_index + 1] = (output & 0xff00) >> 8;
      out[out_index + 2] = (output & 0xff0000) >> 16;
    }
  } else {
    if (writes_per_reduce > 0 && out_index % writes_per_reduce == 0) {
      out[out_index / writes_per_reduce] = output;
    }
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void affine_dequantize(const device uint8_t *w [[buffer(0)]],
                                  const device T *scales [[buffer(1)]],
                                  const device T *biases [[buffer(2)]],
                                  device T *out [[buffer(3)]],
                                  uint2 index [[thread_position_in_grid]],
                                  uint2 grid_dim [[threads_per_grid]]) {
  constexpr int packs_per_int = bits == 3 ? 8 : bits == 6 ? 4 : 8 / bits;
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  constexpr int bytes_per_pack = power_of_2_bits ? 1 : 3;

  size_t offset = index.x + grid_dim.x * size_t(index.y);
  size_t oindex = offset * packs_per_int;
  size_t gindex = oindex / group_size;
  T scale = scales[gindex];
  T bias = biases[gindex];

  out += oindex;

  if (bits == 3) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x7) * scale + bias;
    out[1] = ((w[0] & 0x38) >> 3) * scale + bias;
    out[2] = (((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * scale + bias;
    out[3] = ((w[1] & 0xe) >> 1) * scale + bias;
    out[4] = ((w[1] & 0x70) >> 4) * scale + bias;
    out[5] = (((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * scale + bias;
    out[6] = ((w[2] & 0x1c) >> 2) * scale + bias;
    out[7] = ((w[2] & 0xe0) >> 5) * scale + bias;

  } else if (bits == 6) {
    w += offset * bytes_per_pack;
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    uint val = w[offset];
#pragma clang loop unroll(full)
    for (int i = 0; i < packs_per_int; i++) {
      uint8_t d;
      if (bits == 2) {
        d = (val >> (bits * i)) & 0x03;
      } else if (bits == 4) {
        d = (val >> (bits * i)) & 0x0f;
      } else if (bits == 8) {
        d = val;
      }
      out[i] = scale * d + bias;
    }
  }
}

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...)                                    \
  template [[host_name(                                                        \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define instantiate_quantized(name, type, group_size, bits)                    \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits, name,     \
                     type, group_size, bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched)   \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits            \
                           "_batch_" #batched,                                 \
                     name, type, group_size, bits, batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned)   \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits            \
                           "_alN_" #aligned,                                   \
                     name, type, group_size, bits, aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits,    \
                                              aligned, batched)                \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits            \
                           "_alN_" #aligned "_batch_" #batched,                \
                     name, type, group_size, bits, aligned, batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)   \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D   \
                           "_batch_" #batched,                                 \
                     name, type, group_size, bits, D, batched)

#define instantiate_quantized_split_k(name, type, group_size, bits, split_k)   \
  instantiate_kernel(#name "_" #type "_gs_" #group_size "_b_" #bits            \
                           "_spk_" #split_k,                                   \
                     name, type, group_size, bits, split_k)

#define instantiate_quantized_batched_wrap(name, type, group_size, bits)       \
  instantiate_quantized_batched(name, type, group_size, bits, 1)               \
      instantiate_quantized_batched(name, type, group_size, bits, 0)

#define instantiate_quantized_all_batched(type, group_size, bits)              \
  instantiate_quantized_batched_wrap(qmv_fast, type, group_size, bits)         \
      instantiate_quantized_batched_wrap(qmv, type, group_size, bits)          \
          instantiate_quantized_batched_wrap(qvm, type, group_size, bits)      \
              instantiate_quantized_batched_wrap(qmm_n, type, group_size,      \
                                                 bits)

#define instantiate_quantized_all_single(type, group_size, bits)               \
  instantiate_quantized(affine_quantize, type, group_size, bits)               \
      instantiate_quantized(affine_dequantize, type, group_size, bits)         \
          instantiate_quantized(bs_qmv_fast, type, group_size, bits)           \
              instantiate_quantized(bs_qmv, type, group_size, bits)            \
                  instantiate_quantized(bs_qvm, type, group_size, bits)        \
                      instantiate_quantized(bs_qmm_n, type, group_size, bits)

#define instantiate_quantized_all_aligned(type, group_size, bits)              \
  instantiate_quantized_aligned(bs_qmm_t, type, group_size, bits, true)        \
      instantiate_quantized_aligned(bs_qmm_t, type, group_size, bits, false)   \
          instantiate_quantized_aligned_batched(qmm_t, type, group_size, bits, \
                                                true, 1)                       \
              instantiate_quantized_aligned_batched(qmm_t, type, group_size,   \
                                                    bits, true, 0)             \
                  instantiate_quantized_aligned_batched(                       \
                      qmm_t, type, group_size, bits, false, 1)                 \
                      instantiate_quantized_aligned_batched(                   \
                          qmm_t, type, group_size, bits, false, 0)

#define instantiate_quantized_all_quad(type, group_size, bits)                 \
  instantiate_quantized_quad(qmv_quad, type, group_size, bits, 64, 1)          \
      instantiate_quantized_quad(qmv_quad, type, group_size, bits, 64, 0)      \
          instantiate_quantized_quad(qmv_quad, type, group_size, bits, 128, 1) \
              instantiate_quantized_quad(qmv_quad, type, group_size, bits,     \
                                         128, 0)

#define instantiate_quantized_all_splitk(type, group_size, bits)               \
  instantiate_quantized_split_k(qvm_split_k, type, group_size, bits, 8)        \
      instantiate_quantized_split_k(qvm_split_k, type, group_size, bits, 32)

#define instantiate_quantized_funcs(type, group_size, bits)                    \
  instantiate_quantized_all_single(type, group_size, bits)                     \
      instantiate_quantized_all_batched(type, group_size, bits)                \
          instantiate_quantized_all_aligned(type, group_size, bits)            \
              instantiate_quantized_all_quad(type, group_size, bits)           \
                  instantiate_quantized_all_splitk(type, group_size, bits)

#define instantiate_quantized_types(group_size, bits)                          \
  instantiate_quantized_funcs(float, group_size, bits)                         \
      instantiate_quantized_funcs(float16_t, group_size, bits)                 \
          instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits)                                     \
  instantiate_quantized_types(128, bits) instantiate_quantized_types(64, bits) \
      instantiate_quantized_types(32, bits)

#define instantiate_quantized_all()                                            \
  instantiate_quantized_groups(2) instantiate_quantized_groups(3)              \
      instantiate_quantized_groups(4) instantiate_quantized_groups(6)          \
          instantiate_quantized_groups(8)

instantiate_quantized_all() // clang-format on
