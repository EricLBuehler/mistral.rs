#include <metal_stdlib>

using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

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

#endif

float dequantize_fp4_tree(unsigned char val, float absmax) {
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if ((val & 0b0100) == 4) {                // 0
    if ((val & 0b0010) == 2) {              // 01
      if ((val & 0b0001) == 1) {            // 111
        return 0.25000000f * absmax * sign; // 1111
      } else {
        return 0.16666667f * absmax * sign; // 1110
      }
    } else {
      if ((val & 0b0001) == 1) {            // 110
        return 0.50000000f * absmax * sign; // 1101
      } else {
        return 0.33333333f * absmax * sign; // 1100
      }
    }
  } else {
    if ((val & 0b0010) == 2) {              // 10
      if ((val & 0b0001) == 1) {            // 101
        return 1.00000000f * absmax * sign; // 1011
      } else {
        return 0.66666667f * absmax * sign; // 1010
      }
    } else {
      if ((val & 0b0001) == 1) {                 // 100
        return 5.208333333e-03f * absmax * sign; // 1001
      } else {
        return 0.00000000f * absmax * sign; // 1000
      }
    }
  }
}

float dequantize_nf4(unsigned char val) {
  if ((val & 0b1000) == 8) {
    if ((val & 0b0100) == 4) {     // 1
      if ((val & 0b0010) == 2) {   // 11
        if ((val & 0b0001) == 1) { // 111
          return 1.0f;
        } else {
          return 0.7229568362236023f;
        }
      } else {
        if ((val & 0b0001) == 1) { // 110
          return 0.5626170039176941f;
        } else {
          return 0.44070982933044434f;
        }
      }
    } else {
      if ((val & 0b0010) == 2) {   // 10
        if ((val & 0b0001) == 1) { // 101
          return 0.33791524171829224f;
        } else {
          return 0.24611230194568634f;
        }
      } else {
        if ((val & 0b0001) == 1) { // 100
          return 0.16093020141124725f;
        } else {
          return 0.07958029955625534f;
        }
      }
    }
  } else {
    if ((val & 0b0100) == 4) {     // 0
      if ((val & 0b0010) == 2) {   // 01
        if ((val & 0b0001) == 1) { // 011
          return 0.0f;
        } else {
          return -0.09105003625154495f;
        }
      } else {
        if ((val & 0b0001) == 1) { // 010
          return -0.18477343022823334f;
        } else {
          return -0.28444138169288635f;
        }
      }
    } else {
      if ((val & 0b0010) == 2) {   // 00
        if ((val & 0b0001) == 1) { // 001
          return -0.39491748809814453f;
        } else {
          return -0.5250730514526367f;
        }
      } else {
        if ((val & 0b0001) == 1) { // 000
          return -0.6961928009986877f;
        } else {
          return -1.0f;
        }
      }
    }
  }
}

template <typename T>
[[kernel]] void kernel_dequantize_nf4(const device float *code [[buffer(0)]],
                                      const device uchar *input [[buffer(1)]],
                                      const device float *absmax [[buffer(2)]],
                                      device T *out [[buffer(3)]],
                                      device const int &blocksize,
                                      device const int &n,
                                      uint id [[thread_position_in_grid]]) {

  int block_idx = id * blocksize;
  int valid_items = (n > blocksize + block_idx) ? blocksize : (n - block_idx);
  int block_end = block_idx + valid_items;

  for (int i = block_idx; i < block_end; ++i) {
    float local_abs_max = absmax[block_idx / (blocksize / 2)];

    uint8_t input_value = static_cast<uint8_t>(input[i]);
    float high_nibble = dequantize_nf4(input_value >> 4);
    float low_nibble = dequantize_nf4(input_value & 0x0F);

    out[i * 2] = static_cast<T>(high_nibble * local_abs_max);
    out[i * 2 + 1] = static_cast<T>(low_nibble * local_abs_max);
  }
}

template <typename T>
[[kernel]] void kernel_dequantize_fp4(const device float *code [[buffer(0)]],
                                      const device uchar *input [[buffer(1)]],
                                      const device float *absmax [[buffer(2)]],
                                      device T *out [[buffer(3)]],
                                      device const int &blocksize,
                                      device const int &n,
                                      uint id [[thread_position_in_grid]]) {

  int block_idx = id * blocksize;
  int valid_items = (n > blocksize + block_idx) ? blocksize : (n - block_idx);
  int block_end = block_idx + valid_items;

  for (int i = block_idx; i < block_end; ++i) {
    float local_abs_max = absmax[block_idx / (blocksize / 2)];

    // Extract the high and low nibbles from the input value
    uint8_t input_value = static_cast<uint8_t>(input[i]);
    float high_nibble = dequantize_fp4_tree(input_value >> 4, local_abs_max);
    float low_nibble = dequantize_fp4_tree(input_value & 0x0F, local_abs_max);

    out[i * 2] = static_cast<T>(high_nibble);
    out[i * 2 + 1] = static_cast<T>(low_nibble);
  }
}

template <typename T>
[[kernel]] void kernel_dequantize_int8(const device float *code [[buffer(0)]],
                                       const device uchar *input [[buffer(1)]],
                                       const device float *absmax [[buffer(2)]],
                                       device T *out [[buffer(3)]],
                                       device const int &blocksize,
                                       device const int &n,
                                       uint id [[thread_position_in_grid]]) {

  int block_idx = id * blocksize;
  int valid_items = (n > blocksize + block_idx) ? blocksize : (n - block_idx);
  int block_end = block_idx + valid_items;

  for (int i = block_idx; i < block_end; ++i) {
    float local_abs_max = absmax[block_idx / blocksize];

    out[i] = static_cast<T>(code[input[i]] * local_abs_max);
  }
}

#define instantiate_dequantize_nf4(type)                                       \
  template [[host_name("kernel_dequantize_nf4_" #type)]] [[kernel]] void       \
  kernel_dequantize_nf4<type>(                                                 \
      const device float *code [[buffer(0)]],                                  \
      const device uchar *input [[buffer(1)]],                                 \
      const device float *absmax [[buffer(2)]],                                \
      device type *out [[buffer(3)]], device const int &blocksize,             \
      device const int &n, uint id [[thread_position_in_grid]]);

instantiate_dequantize_nf4(float) instantiate_dequantize_nf4(bfloat16_t)
    instantiate_dequantize_nf4(half)

#define instantiate_dequantize_fp4(type)                                       \
  template [[host_name("kernel_dequantize_fp4_" #type)]] [[kernel]] void       \
  kernel_dequantize_fp4<type>(                                                 \
      const device float *code [[buffer(0)]],                                  \
      const device uchar *input [[buffer(1)]],                                 \
      const device float *absmax [[buffer(2)]],                                \
      device type *out [[buffer(3)]], device const int &blocksize,             \
      device const int &n, uint id [[thread_position_in_grid]]);

        instantiate_dequantize_fp4(float) instantiate_dequantize_fp4(bfloat16_t)
            instantiate_dequantize_fp4(half)

#define instantiate_dequantize_int8(type)                                      \
  template [[host_name("kernel_dequantize_int8_" #type)]] [[kernel]] void      \
  kernel_dequantize_int8<type>(                                                \
      const device float *code [[buffer(0)]],                                  \
      const device uchar *input [[buffer(1)]],                                 \
      const device float *absmax [[buffer(2)]],                                \
      device type *out [[buffer(3)]], device const int &blocksize,             \
      device const int &n, uint id [[thread_position_in_grid]]);

                instantiate_dequantize_int8(float)
                    instantiate_dequantize_int8(bfloat16_t)
                        instantiate_dequantize_int8(half)
