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
  bitwise_or<type>(const device type *a [[buffer(0)]],                         \
                   const device type *b [[buffer(1)]],                         \
                   device type *out [[buffer(2)]],                            \            
    uint tid [[thread_position_in_grid]]);

instantiate_bitwise_or(uint8_t) instantiate_bitwise_or(uint32_t)
    instantiate_bitwise_or(int64_t) instantiate_bitwise_or(int)

        template <typename T>
        [[kernel]] void bitwise_leftshift(const device T *a [[buffer(0)]],
                                          device T *output [[buffer(1)]],
                                          device const uint &k,
                                          uint tid
                                          [[thread_position_in_grid]]) {
  output[tid] = a[tid] << k;
}

#define instantiate_bitwise_leftshift(type)                                    \
  template [[host_name("bitwise_leftshift_" #type)]] [[kernel]] void           \
  bitwise_leftshift<type>(                                                     \
      const device type *a [[buffer(0)]], device type *out [[buffer(1)]],      \
      device const uint &k, uint tid [[thread_position_in_grid]]);

instantiate_bitwise_leftshift(uint8_t) instantiate_bitwise_leftshift(uint32_t)
    instantiate_bitwise_leftshift(int64_t) instantiate_bitwise_leftshift(int)
