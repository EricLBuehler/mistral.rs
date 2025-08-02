#include <metal_stdlib>
using namespace metal;

// Pack 8-bit values kernel (essentially a copy with type conversion)
kernel void pack_8bit(constant int32_t *input [[buffer(0)]],
                      device uint8_t *output [[buffer(1)]],
                      constant uint64_t &num_elems [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  if (tid >= num_elems)
    return;

  // Extract lower 8 bits from int32
  output[tid] = uint8_t(input[tid] & 0xFF);
}

// Pack 4-bit values kernel - packs 2 4-bit values into 1 byte
kernel void pack_4bit(constant uint8_t *input [[buffer(0)]],
                      device uint8_t *output [[buffer(1)]],
                      constant uint64_t &height [[buffer(2)]],
                      constant uint64_t &width [[buffer(3)]],
                      uint2 tid [[thread_position_in_grid]]) {
  uint step = height / 2;
  if (tid.x >= step || tid.y >= width)
    return;

  uint idx_a = tid.x * width + tid.y;
  uint idx_b = (tid.x + step) * width + tid.y;

  uint8_t a = input[idx_a] & 0xF;
  uint8_t b = input[idx_b] & 0xF;

  output[idx_a] = (a << 4) | b;
}

// Pack 2-bit values kernel - packs 4 2-bit values into 1 byte
kernel void pack_2bit(constant uint8_t *input [[buffer(0)]],
                      device uint8_t *output [[buffer(1)]],
                      constant uint64_t &height [[buffer(2)]],
                      constant uint64_t &width [[buffer(3)]],
                      uint2 tid [[thread_position_in_grid]]) {
  uint step = height / 4;
  if (tid.x >= step || tid.y >= width)
    return;

  uint out_idx = tid.x * width + tid.y;

  uint8_t packed = 0;
  for (int i = 0; i < 4; i++) {
    uint idx = (tid.x + i * step) * width + tid.y;
    uint8_t val = input[idx] & 0x3;
    packed |= (val << (6 - i * 2));
  }

  output[out_idx] = packed;
}

// Pack 3-bit values kernel - packs 10 3-bit values into 32 bits
kernel void pack_3bit(constant uint32_t *input [[buffer(0)]],
                      device int32_t *output [[buffer(1)]],
                      constant uint64_t &height [[buffer(2)]],
                      constant uint64_t &width [[buffer(3)]],
                      uint2 tid [[thread_position_in_grid]]) {
  uint step = height / 10;
  if (tid.x >= step || tid.y >= width)
    return;

  uint out_idx = tid.x * width + tid.y;

  int32_t packed = 0;
  for (int i = 0; i < 10; i++) {
    uint idx = (tid.x + i * step) * width + tid.y;
    int32_t val = int32_t(input[idx] & 0x7);
    packed |= (val << (27 - i * 3));
  }

  output[out_idx] = packed;
}

// Pack 1-bit values kernel - packs 8 1-bit values into 1 byte
kernel void pack_1bit(constant uint8_t *input [[buffer(0)]],
                      device uint8_t *output [[buffer(1)]],
                      constant uint64_t &height [[buffer(2)]],
                      constant uint64_t &width [[buffer(3)]],
                      uint2 tid [[thread_position_in_grid]]) {
  uint step = height / 8;
  if (tid.x >= step || tid.y >= width)
    return;

  uint out_idx = tid.x * width + tid.y;

  uint8_t packed = 0;
  for (int i = 0; i < 8; i++) {
    uint idx = (tid.x + i * step) * width + tid.y;
    uint8_t bit = input[idx] & 0x1;
    packed |= (bit << (7 - i));
  }

  output[out_idx] = packed;
}