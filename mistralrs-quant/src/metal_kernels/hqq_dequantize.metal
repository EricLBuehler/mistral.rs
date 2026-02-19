#include <metal_stdlib>

/*********************************/
/************* 8-bit *************/
//********************************/

template <typename T>
[[kernel]] void dequantize_8bit(const device char *weight [[buffer(0)]],
                                const device T *scale [[buffer(1)]],
                                const device T *zero [[buffer(2)]],
                                device T *output [[buffer(3)]],
                                device const uint &h, device const uint &w,
                                uint tid [[thread_position_in_grid]]) {
  uint j = tid % w;
  output[tid] = ((T)(weight[tid]) - zero[j]) * scale[j];
}

#define instantiate_dequantize_8bit(type)                                      \
  template [[host_name("dequantize_8bit_" #type)]] [[kernel]] void             \
  dequantize_8bit<type>(const device char *weight [[buffer(0)]],               \
                        const device type *scale [[buffer(1)]],                \
                        const device type *zero [[buffer(2)]],                 \
                        device type *output [[buffer(3)]],                     \
                        device const uint &h, device const uint &w,            \
                        uint tid [[thread_position_in_grid]]);

instantiate_dequantize_8bit(float)
#if defined(__HAVE_BFLOAT__)
    instantiate_dequantize_8bit(bfloat)
#endif
        instantiate_dequantize_8bit(half)

    /*********************************/
    /************* 4-bit *************/
    //********************************/

    template <typename T>
    [[kernel]] void dequantize_4bit(const device char *weight [[buffer(0)]],
                                    const device T *scale [[buffer(1)]],
                                    const device T *zero [[buffer(2)]],
                                    device T *output [[buffer(3)]],
                                    device const uint &h, device const uint &w,
                                    uint tid [[thread_position_in_grid]]) {
  uint n = h * w;
  uint j = tid % w;
  output[tid] =
      ((T)((weight[tid] & 0xF0) >> 4) - zero[j]) * scale[j]; // First chunk
  output[tid + n] =
      ((T)((weight[tid] & 0x0F)) - zero[j]) * scale[j]; // Second chunk
}

#define instantiate_dequantize_4bit(type)                                      \
  template [[host_name("dequantize_4bit_" #type)]] [[kernel]] void             \
  dequantize_4bit<type>(const device char *weight [[buffer(0)]],               \
                        const device type *scale [[buffer(1)]],                \
                        const device type *zero [[buffer(2)]],                 \
                        device type *output [[buffer(3)]],                     \
                        device const uint &h, device const uint &w,            \
                        uint tid [[thread_position_in_grid]]);

instantiate_dequantize_4bit(float)
#if defined(__HAVE_BFLOAT__)
    instantiate_dequantize_4bit(bfloat)
#endif
        instantiate_dequantize_4bit(half)

    /*********************************/
    /************* 2-bit *************/
    //********************************/

    template <typename T>
    [[kernel]] void dequantize_2bit(const device char *weight [[buffer(0)]],
                                    const device T *scale [[buffer(1)]],
                                    const device T *zero [[buffer(2)]],
                                    device T *output [[buffer(3)]],
                                    device const uint &h, device const uint &w,
                                    uint tid [[thread_position_in_grid]]) {
  uint n = h * w;
  uint j = tid % w;
  output[tid] =
      ((T)((weight[tid] & 0xC0) >> 6) - zero[j]) * scale[j]; // 1st chunk
  output[tid + n] =
      ((T)((weight[tid] & 0x30) >> 4) - zero[j]) * scale[j]; // 2nd chunk
  output[tid + n * 2] =
      ((T)((weight[tid] & 0x0C) >> 2) - zero[j]) * scale[j]; // 3rd chunk
  output[tid + n * 3] =
      ((T)((weight[tid] & 0x03)) - zero[j]) * scale[j]; // 4th chunk
}

#define instantiate_dequantize_2bit(type)                                      \
  template [[host_name("dequantize_2bit_" #type)]] [[kernel]] void             \
  dequantize_2bit<type>(const device char *weight [[buffer(0)]],               \
                        const device type *scale [[buffer(1)]],                \
                        const device type *zero [[buffer(2)]],                 \
                        device type *output [[buffer(3)]],                     \
                        device const uint &h, device const uint &w,            \
                        uint tid [[thread_position_in_grid]]);

instantiate_dequantize_2bit(float)
#if defined(__HAVE_BFLOAT__)
    instantiate_dequantize_2bit(bfloat)
#endif
        instantiate_dequantize_2bit(half)

    /*********************************/
    /************* 1-bit *************/
    //********************************/

    template <typename T>
    [[kernel]] void dequantize_1bit(const device char *weight [[buffer(0)]],
                                    const device T *scale [[buffer(1)]],
                                    const device T *zero [[buffer(2)]],
                                    device T *output [[buffer(3)]],
                                    device const uint &h, device const uint &w,
                                    uint tid [[thread_position_in_grid]]) {
  uint n = h * w;
  uint j = tid % w;
  output[tid] =
      ((T)((weight[tid] & 0x80) >> 7) - zero[j]) * scale[j]; // 1st chunk
  output[tid + n] =
      ((T)((weight[tid] & 0x40) >> 6) - zero[j]) * scale[j]; // 2nd chunk
  output[tid + n * 2] =
      ((T)((weight[tid] & 0x20) >> 5) - zero[j]) * scale[j]; // 3rd chunk
  output[tid + n * 3] =
      ((T)((weight[tid] & 0x10) >> 4) - zero[j]) * scale[j]; // 4th chunk
  output[tid + n * 4] =
      ((T)((weight[tid] & 0x08) >> 3) - zero[j]) * scale[j]; // 5th chunk
  output[tid + n * 5] =
      ((T)((weight[tid] & 0x04) >> 2) - zero[j]) * scale[j]; // 6th chunk
  output[tid + n * 6] =
      ((T)((weight[tid] & 0x02) >> 1) - zero[j]) * scale[j]; // 7th chunk
  output[tid + n * 7] =
      ((T)((weight[tid] & 0x01)) - zero[j]) * scale[j]; // 8th chunk
}

#define instantiate_dequantize_1bit(type)                                      \
  template [[host_name("dequantize_1bit_" #type)]] [[kernel]] void             \
  dequantize_1bit<type>(const device char *weight [[buffer(0)]],               \
                        const device type *scale [[buffer(1)]],                \
                        const device type *zero [[buffer(2)]],                 \
                        device type *output [[buffer(3)]],                     \
                        device const uint &h, device const uint &w,            \
                        uint tid [[thread_position_in_grid]]);

instantiate_dequantize_1bit(float)
#if defined(__HAVE_BFLOAT__)
    instantiate_dequantize_1bit(bfloat)
#endif
        instantiate_dequantize_1bit(half)

    /*********************************/
    /************* 3-bit *************/
    //********************************/

    template <typename T>
    [[kernel]] void dequantize_3bit(const device int *weight [[buffer(0)]],
                                    const device T *scale [[buffer(1)]],
                                    const device T *zero [[buffer(2)]],
                                    device T *output [[buffer(3)]],
                                    device const uint &h, device const uint &w,
                                    uint tid [[thread_position_in_grid]]) {
  uint n = h * w;
  uint j = tid % w;
  output[tid] =
      ((T)((weight[tid] & 0x38000000) >> 27) - zero[j]) * scale[j]; // 1st chunk
  output[tid + n] =
      ((T)((weight[tid] & 0x07000000) >> 24) - zero[j]) * scale[j]; // 2nd chunk
  output[tid + n * 2] =
      ((T)((weight[tid] & 0x00E00000) >> 21) - zero[j]) * scale[j]; // 3rd chunk
  output[tid + n * 3] =
      ((T)((weight[tid] & 0x001C0000) >> 18) - zero[j]) * scale[j]; // 4th chunk
  output[tid + n * 4] =
      ((T)((weight[tid] & 0x00038000) >> 15) - zero[j]) * scale[j]; // 5th chunk
  output[tid + n * 5] =
      ((T)((weight[tid] & 0x00007000) >> 12) - zero[j]) * scale[j]; // 6th chunk
  output[tid + n * 6] =
      ((T)((weight[tid] & 0x00000E00) >> 9) - zero[j]) * scale[j]; // 7th chunk
  output[tid + n * 7] =
      ((T)((weight[tid] & 0x000001C0) >> 6) - zero[j]) * scale[j]; // 8th chunk
  output[tid + n * 8] =
      ((T)((weight[tid] & 0x00000038) >> 3) - zero[j]) * scale[j]; // 9th chunk
  output[tid + n * 9] =
      ((T)((weight[tid] & 0x00000007)) - zero[j]) * scale[j]; // 10th chunk
}

#define instantiate_dequantize_3bit(type)                                      \
  template [[host_name("dequantize_3bit_" #type)]] [[kernel]] void             \
  dequantize_3bit<type>(const device int *weight [[buffer(0)]],                \
                        const device type *scale [[buffer(1)]],                \
                        const device type *zero [[buffer(2)]],                 \
                        device type *output [[buffer(3)]],                     \
                        device const uint &h, device const uint &w,            \
                        uint tid [[thread_position_in_grid]]);

instantiate_dequantize_3bit(float)
#if defined(__HAVE_BFLOAT__)
    instantiate_dequantize_3bit(bfloat)
#endif
        instantiate_dequantize_3bit(half)
