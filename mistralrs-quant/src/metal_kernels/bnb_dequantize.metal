#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

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
