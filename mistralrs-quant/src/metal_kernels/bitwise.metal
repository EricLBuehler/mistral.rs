#include <metal_stdlib>

template <typename T>
[[kernel]] void bitwise_or(const device T *a [[buffer(0)]],
                           const device T *b [[buffer(1)]],
                           device T *output [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  output[tid] = a[tid] | b[tid];
}

#define instantiate_bitwise_or(type)                                           \
  template [[host_name("bitwise_or_" #type)]] [[kernel]] void                  \
  bitwise_or<type>(                                                            \
      const device type *a [[buffer(0)]], const device type *b [[buffer(1)]],  \
      device type *out [[buffer(2)]], uint tid [[thread_position_in_grid]]);

instantiate_bitwise_or(uint8_t);
instantiate_bitwise_or(uint32_t);
instantiate_bitwise_or(int64_t);
instantiate_bitwise_or(int);

template <typename T>
[[kernel]] void bitwise_xor(const device T *a [[buffer(0)]],
                            const device T *b [[buffer(1)]],
                            device T *output [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
  output[tid] = a[tid] ^ b[tid];
}

#define instantiate_bitwise_xor(type)                                          \
  template [[host_name("bitwise_xor_" #type)]] [[kernel]] void                 \
  bitwise_xor<type>(                                                           \
      const device type *a [[buffer(0)]], const device type *b [[buffer(1)]],  \
      device type *out [[buffer(2)]], uint tid [[thread_position_in_grid]]);

instantiate_bitwise_xor(uint8_t);
instantiate_bitwise_xor(uint32_t);
instantiate_bitwise_xor(int64_t);
instantiate_bitwise_xor(int);

template <typename T>
[[kernel]] void bitwise_and(const device T *a [[buffer(0)]],
                            const device T *b [[buffer(1)]],
                            device T *output [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
  output[tid] = a[tid] & b[tid];
}

#define instantiate_bitwise_and(type)                                          \
  template [[host_name("bitwise_and_" #type)]] [[kernel]] void                 \
  bitwise_and<type>(                                                           \
      const device type *a [[buffer(0)]], const device type *b [[buffer(1)]],  \
      device type *out [[buffer(2)]], uint tid [[thread_position_in_grid]]);

instantiate_bitwise_and(uint8_t);
instantiate_bitwise_and(uint32_t);
instantiate_bitwise_and(int64_t);
instantiate_bitwise_and(int);

template <typename T>
[[kernel]] void bitwise_leftshift(const device T *a [[buffer(0)]],
                                  device T *output [[buffer(1)]],
                                  device const uint &k,
                                  uint tid [[thread_position_in_grid]]) {
  output[tid] = a[tid] << k;
}

#define instantiate_bitwise_leftshift(type)                                    \
  template [[host_name("bitwise_leftshift_" #type)]] [[kernel]] void           \
  bitwise_leftshift<type>(                                                     \
      const device type *a [[buffer(0)]], device type *out [[buffer(1)]],      \
      device const uint &k, uint tid [[thread_position_in_grid]]);

instantiate_bitwise_leftshift(uint8_t);
instantiate_bitwise_leftshift(uint32_t);
instantiate_bitwise_leftshift(int64_t);
instantiate_bitwise_leftshift(int);

template <typename T>
[[kernel]] void bitwise_not(const device T *a [[buffer(0)]],
                            device T *output [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
  output[tid] = ~a[tid];
}

#define instantiate_bitwise_not(type)                                          \
  template [[host_name("bitwise_not_" #type)]] [[kernel]] void                 \
  bitwise_not<type>(const device type *a [[buffer(0)]],                        \
                    device type *out [[buffer(1)]],                            \
                    uint tid [[thread_position_in_grid]]);

instantiate_bitwise_not(uint8_t);
instantiate_bitwise_not(uint32_t);
instantiate_bitwise_not(int64_t);
instantiate_bitwise_not(int);