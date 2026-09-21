// Index and decode math for PTQ1_0 (Prism ternary, group 128); host-compilable
// so it can be tested without a GPU.
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define PTQ1_0_HD __host__ __device__ __forceinline__
#else
#define PTQ1_0_HD inline
#endif

namespace ptq1_0 {

constexpr int BLOCK_ELEMS = 128;
constexpr int BLOCK_BYTES = 28;
constexpr int BLOCK_WORDS = 7; // 4 wide qs words, 2 narrow qs words, qh + scale
constexpr int WIDE_WORDS = 4;
constexpr int NARROW_START = 80;
constexpr int QH_START = 120;
constexpr int DECODE_STEPS = 5;
constexpr int FWHT_BLOCK = 1024;
constexpr uint32_t LANE_MASK = 0x00FF00FFu;
constexpr uint32_t ONES = 0x01010101u;

// What one weight word contributes: `steps` groups of 4 consecutive weights at
// elements elem_base + step * elem_stride.
struct DpLane {
  int elem_base;
  int elem_stride;
  uint32_t mult;
  int steps;
};

PTQ1_0_HD DpLane dp_lane(int word) {
  if (word < WIDE_WORDS) {
    return {4 * word, 16, 3u, DECODE_STEPS};
  }
  if (word < BLOCK_WORDS - 1) {
    return {NARROW_START + 4 * (word - WIDE_WORDS), 8, 3u, DECODE_STEPS};
  }
  return {QH_START, 4, 9u, 2}; // qh: two trits per step, so advance by 3 * 3
}

// Lower index of butterfly pair `pair` at stride `h` (h a power of two).
PTQ1_0_HD int fwht_low_index(int pair, int h) {
  return ((pair & ~(h - 1)) << 1) | (pair & (h - 1));
}

PTQ1_0_HD uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t sel) {
#ifdef __CUDA_ARCH__
  return __byte_perm(x, y, sel);
#else
  const uint64_t v = (static_cast<uint64_t>(y) << 32) | x;
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    r |= static_cast<uint32_t>((v >> (8 * ((sel >> (4 * i)) & 7))) & 0xFF)
         << (8 * i);
  }
  return r;
#endif
}

PTQ1_0_HD uint32_t sub_bytes(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  return __vsub4(a, b);
#else
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    r |= (((a >> (8 * i)) - (b >> (8 * i))) & 0xFFu) << (8 * i);
  }
  return r;
#endif
}

PTQ1_0_HD int dot4(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  for (int i = 0; i < 4; ++i) {
    c += static_cast<int>(static_cast<int8_t>(a >> (8 * i))) *
         static_cast<int>(static_cast<int8_t>(b >> (8 * i)));
  }
  return c;
#endif
}

// Trit state for a weight word. Bytes go to 16-bit lanes so multiplying by 3
// cannot carry between them.
PTQ1_0_HD void init_state(uint32_t word, int word_index, uint32_t &v_lo,
                          uint32_t &v_hi) {
  v_lo = byte_perm(word, 0, 0x4140);
  v_hi = word_index < BLOCK_WORDS - 1 ? byte_perm(word, 0, 0x4342)
                                      : (v_lo * 3u) & LANE_MASK;
}

// Four trit digits 0..2 (one per byte, weight + 1) for the next step,
// advancing the state.
PTQ1_0_HD int decode_digits(uint32_t &v_lo, uint32_t &v_hi, uint32_t mult) {
  const uint32_t w_lo = v_lo * 3u;
  const uint32_t w_hi = v_hi * 3u;
  v_lo = (v_lo * mult) & LANE_MASK;
  v_hi = (v_hi * mult) & LANE_MASK;
  return static_cast<int>(byte_perm(w_lo, w_hi, 0x7531));
}

// The same four trits as signed weights -1..1.
PTQ1_0_HD int decode_step(uint32_t &v_lo, uint32_t &v_hi, uint32_t mult) {
  return static_cast<int>(
      sub_bytes(static_cast<uint32_t>(decode_digits(v_lo, v_hi, mult)), ONES));
}

// The 8 qh weights as two dp4a groups (elements 120..123 and 124..127) of
// signed trits; the second group starts two digits further on.
PTQ1_0_HD void decode_qh(uint32_t word, int &first, int &second) {
  uint32_t lo, hi;
  init_state(word, BLOCK_WORDS - 1, lo, hi);
  const uint32_t w1_lo = ((lo * 9u) & LANE_MASK) * 3u;
  const uint32_t w1_hi = ((lo * 27u) & LANE_MASK) * 3u;
  first = static_cast<int>(
      sub_bytes(byte_perm(lo * 3u, hi * 3u, 0x7531), ONES));
  second = static_cast<int>(sub_bytes(byte_perm(w1_lo, w1_hi, 0x7531), ONES));
}

// Weight `e` (0..127) of a block as -1, 0 or 1, following the packed element
// order of the qs and qh bytes.
PTQ1_0_HD int element_trit(const uint8_t *blk, int e) {
  int byte;
  int n;
  if (e < NARROW_START) {
    byte = e & 15;
    n = e >> 4;
  } else if (e < QH_START) {
    byte = 16 + ((e - NARROW_START) & 7);
    n = (e - NARROW_START) >> 3;
  } else {
    byte = 24 + ((e - QH_START) & 1);
    n = (e - QH_START) >> 1;
  }
  uint32_t v = blk[byte];
  for (int i = 0; i < n; ++i) {
    v = (v * 3u) & 255u;
  }
  return static_cast<int>((v * 3u) >> 8) - 1;
}

#ifdef __CUDACC__
static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

static __device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
  }
  return x;
}
#endif

} // namespace ptq1_0
