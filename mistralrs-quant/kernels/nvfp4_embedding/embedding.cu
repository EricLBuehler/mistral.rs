#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {
constexpr int kWarpSize = 32;
constexpr int kWarps = 8;
constexpr int kBlockSize = 16;
constexpr int kIndexFields = 3;

struct EmbeddingLaunch {
  const int64_t *rows;
  const uint64_t *shards;
  void *output;
  size_t lookups;
  size_t columns;
  int shard_count;
  float global_scale;
  int dtype;
};

__device__ float e4m3(uint8_t bits) {
  const int exponent = (bits >> 3) & 15;
  const int mantissa = bits & 7;
  if (exponent == 0) {
    const float value = mantissa * (1.0f / 512.0f);
    return bits & 128 ? -value : value;
  }
  if (exponent == 15 && mantissa == 7) {
    return __int_as_float(0x7fffffff);
  }
  return __int_as_float(((bits & 128) << 24) | ((exponent + 120) << 23) |
                        (mantissa << 20));
}

template <typename T> __device__ T cast_output(float value);
template <> __device__ __nv_bfloat16 cast_output(float value) {
  return __float2bfloat16_rn(value);
}
template <> __device__ __half cast_output(float value) {
  return __float2half_rn(value);
}
template <> __device__ float cast_output(float value) { return value; }

template <typename T> __global__ void gather(EmbeddingLaunch args) {
  const size_t lookup = blockIdx.x * kWarps + (threadIdx.x / kWarpSize);
  if (lookup >= args.lookups) {
    return;
  }
  const int lane = threadIdx.x % kWarpSize;
  const uint64_t row = args.rows[lookup];
  int lo = 0;
  int hi = args.shard_count - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (args.shards[mid * kIndexFields + 2] <= row) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const uint64_t local = row - args.shards[lo * kIndexFields + 2];
  const auto *data =
      reinterpret_cast<const uint8_t *>(args.shards[lo * kIndexFields]) +
      local * (args.columns / 2);
  const auto *scales =
      reinterpret_cast<const uint8_t *>(args.shards[lo * kIndexFields + 1]) +
      local * (args.columns / kBlockSize);
  auto *out = reinterpret_cast<T *>(args.output) + lookup * args.columns;
  constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (size_t column = lane; column < args.columns; column += kWarpSize) {
    const uint8_t code = (data[column / 2] >> (4 * (column % 2))) & 15;
    const float q = (code & 8) ? -values[code & 7] : values[code & 7];
    const float value = __fmul_rn(
        __fmul_rn(q, e4m3(scales[column / kBlockSize])), args.global_scale);
    out[column] = cast_output<T>(value);
  }
}
} // namespace

extern "C" void mistralrs_nvfp4_embedding_gather(const EmbeddingLaunch *args,
                                                 cudaStream_t stream) {
  const int blocks = (args->lookups + kWarps - 1) / kWarps;
  if (args->dtype == 0) {
    gather<__nv_bfloat16><<<blocks, kWarps * kWarpSize, 0, stream>>>(*args);
  } else if (args->dtype == 1) {
    gather<__half><<<blocks, kWarps * kWarpSize, 0, stream>>>(*args);
  } else {
    gather<float><<<blocks, kWarps * kWarpSize, 0, stream>>>(*args);
  }
}
