// W4A16 NVFP4 GEMM for GPUs without block-scaled FP4 MMA: E2M1 codes and E4M3
// block scales are expanded in registers and fed to bf16/f16 mma.sync. Serves
// dense rows and moe_align-sorted expert routes.

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

struct Nvfp4W4a16Args {
  const void *input;
  const uint8_t *weight;
  const uint8_t *scales;
  const float *global;
  const int32_t *sorted_ids;
  const int32_t *expert_ids;
  const int32_t *num_padded;
  void *output;
  float *partial;
  int rows;
  int n;
  int k;
  int input_stride;
  int splits;
  int k_tiles_per_split;
};

namespace {
constexpr int kWarp = 32;
constexpr int kBlock = 16;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kStages = 3;
constexpr int kWarpsN = 4;
constexpr int kWarpTilesN = 4;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;
constexpr int kKSteps = kTileK / kMmaK;
constexpr int kAChunk = 8;
constexpr int kAChunksPerRow = kTileK / kAChunk;
constexpr int kBRowBytes = kTileK / 2;
constexpr int kBUnit = 8;
constexpr int kBUnitsPerRow = kBRowBytes / kBUnit;
constexpr int kScalesPerTile = kTileK / kBlock;
constexpr int kReduceThreads = 256;
constexpr int kDefaultSmem = 48 * 1024;

// LUT bytes are 2 * e2m1 + 16 so one magic exponent turns them into exact
// 16-bit floats.
constexpr uint32_t kLutPositiveLow = 0x13121110u;
constexpr uint32_t kLutPositiveHigh = 0x1c181614u;
constexpr uint32_t kLutNegativeLow = 0x0d0e0f10u;
constexpr uint32_t kLutNegativeHigh = 0x04080a0cu;
constexpr uint32_t kNibbleIndex = 0x7777u;
constexpr uint32_t kNibbleSign = 0x8888u;
constexpr uint32_t kIdentitySelect = 0x3210u;
constexpr uint32_t kLowPairSelect = 0x4140u;
constexpr uint32_t kHighPairSelect = 0x4342u;
constexpr uint32_t kScaleSign = 0x00800080u;
constexpr uint32_t kScaleMagnitude = 0x007f007fu;
constexpr int kScaleSignShift = 8;
constexpr int kScaleMagnitudeShift = 7;
// E4M3 bits placed in an f16 read as scale / 256; this restores scale / 2 to
// undo the doubled LUT.
constexpr float kHalfScaleFix = 128.0f;

enum Config { kRows16 = 0, kRows64 = 1, kRows128 = 2 };

template <typename T> struct Traits;

template <> struct Traits<__half> {
  using T2 = __half2;
  static constexpr uint32_t kMagic = 0x64646464u;
  static constexpr float kBias = 1040.0f;
  static __device__ __forceinline__ T2 pair(float value) {
    return __float2half2_rn(value);
  }
  static __device__ __forceinline__ T2 from_halves(__half2 value) {
    return value;
  }
  static __device__ __forceinline__ T2 low(T2 value) {
    return __low2half2(value);
  }
  static __device__ __forceinline__ T2 high(T2 value) {
    return __high2half2(value);
  }
  static __device__ __forceinline__ void mma(float *c, const uint32_t *a,
                                             uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                   "r"(b1));
  }
  static __device__ __forceinline__ void store2(__half *out, float x, float y) {
    *reinterpret_cast<__half2 *>(out) = __floats2half2_rn(x, y);
  }
};

template <> struct Traits<__nv_bfloat16> {
  using T2 = __nv_bfloat162;
  static constexpr uint32_t kMagic = 0x43434343u;
  static constexpr float kBias = 144.0f;
  static __device__ __forceinline__ T2 pair(float value) {
    return __float2bfloat162_rn(value);
  }
  static __device__ __forceinline__ T2 from_halves(__half2 value) {
    return __float22bfloat162_rn(__half22float2(value));
  }
  static __device__ __forceinline__ T2 low(T2 value) {
    return __low2bfloat162(value);
  }
  static __device__ __forceinline__ T2 high(T2 value) {
    return __high2bfloat162(value);
  }
  static __device__ __forceinline__ void mma(float *c, const uint32_t *a,
                                             uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                   "r"(b1));
  }
  static __device__ __forceinline__ void store2(__nv_bfloat16 *out, float x,
                                                float y) {
    *reinterpret_cast<__nv_bfloat162 *>(out) = __floats2bfloat162_rn(x, y);
  }
};

template <int WarpsM, int TilesM> struct Shape {
  static constexpr int kRows = WarpsM * TilesM * kMmaM;
  static constexpr int kThreads = WarpsM * kWarpsN * kWarp;
  static constexpr int kStageA = kRows * kTileK;
  static constexpr int kStageB = kTileN * kBRowBytes;
  static constexpr size_t kSmem =
      size_t(kStages) * (kStageA * 2 + kStageB) + 2 * kRows * sizeof(int);
};

__device__ __forceinline__ uint32_t smem_address(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void copy16(void *dst, const void *src, bool valid) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                   smem_address(dst)),
               "l"(src), "r"(valid ? 16 : 0));
#endif
}

__device__ __forceinline__ void copy8(void *dst, const void *src, bool valid) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_address(dst)),
      "l"(src), "r"(valid ? 8 : 0));
#endif
}

__device__ __forceinline__ void commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N> __device__ __forceinline__ void wait_stages() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void load_matrix(uint32_t *a, const void *pointer) {
#if __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(smem_address(pointer)));
#endif
}

__device__ __forceinline__ uint32_t lookup(uint32_t nibbles) {
  const uint32_t index = nibbles & kNibbleIndex;
  const uint32_t positive =
      __byte_perm(kLutPositiveLow, kLutPositiveHigh, index);
  const uint32_t negative =
      __byte_perm(kLutNegativeLow, kLutNegativeHigh, index);
  return __byte_perm(positive, negative,
                     kIdentitySelect | ((nibbles & kNibbleSign) >> 1));
}

// Two E4M3 scales from `word` (bytes picked by `select`) as exact halves of
// scale / 2.
__device__ __forceinline__ __half2 half_scales(uint32_t word, uint32_t select) {
  const uint32_t spread = __byte_perm(word, 0, select);
  const uint32_t bits = ((spread & kScaleSign) << kScaleSignShift) |
                        ((spread & kScaleMagnitude) << kScaleMagnitudeShift);
  return __hmul2(*reinterpret_cast<const __half2 *>(&bits),
                 __float2half2_rn(kHalfScaleFix));
}

__device__ __forceinline__ uint32_t load_scales(const uint8_t *row, int tile,
                                                int k_blocks, bool aligned) {
  const int first = tile * kScalesPerTile;
  if (aligned)
    return __ldg(reinterpret_cast<const uint32_t *>(row + first));
  uint32_t value = 0;
#pragma unroll
  for (int i = 0; i < kScalesPerTile; ++i)
    if (first + i < k_blocks)
      value |= uint32_t(__ldg(row + first + i)) << (8 * i);
  return value;
}

template <typename T, int WarpsM, int TilesM>
__global__ void __launch_bounds__(Shape<WarpsM, TilesM>::kThreads)
    gemm(Nvfp4W4a16Args a) {
#if __CUDA_ARCH__ >= 800
  using S = Shape<WarpsM, TilesM>;
  using Tr = Traits<T>;
  using T2 = typename Tr::T2;
  extern __shared__ uint4 shared[];
  T *a_shared = reinterpret_cast<T *>(shared);
  uint8_t *b_shared =
      reinterpret_cast<uint8_t *>(a_shared + kStages * S::kStageA);
  int *source_rows = reinterpret_cast<int *>(b_shared + kStages * S::kStageB);
  int *output_rows = source_rows + S::kRows;

  const int tile_m = blockIdx.y;
  const int tile_n = blockIdx.x * kTileN;
  int expert = 0;
  if (a.sorted_ids) {
    if (tile_m * S::kRows >= *a.num_padded)
      return;
    expert = a.expert_ids[tile_m];
    if (expert < 0)
      return;
  }
  for (int r = threadIdx.x; r < S::kRows; r += S::kThreads) {
    int source = -1;
    int output = -1;
    if (a.sorted_ids) {
      const int route = a.sorted_ids[tile_m * S::kRows + r];
      if (route < a.rows) {
        source = route / a.input_stride;
        output = route;
      }
    } else if (tile_m * S::kRows + r < a.rows) {
      source = output = tile_m * S::kRows + r;
    }
    source_rows[r] = source;
    output_rows[r] = output;
  }
  __syncthreads();

  const int packed_k = a.k / 2;
  const int k_blocks = a.k / kBlock;
  const int k_tiles = (a.k + kTileK - 1) / kTileK;
  const int tile_begin = blockIdx.z * a.k_tiles_per_split;
  const int tile_count =
      min(k_tiles, tile_begin + a.k_tiles_per_split) - tile_begin;
  const T *input = static_cast<const T *>(a.input);
  const uint8_t *weight = a.weight + size_t(expert) * a.n * packed_k;
  const uint8_t *scales = a.scales + size_t(expert) * a.n * k_blocks;
  const float *global = a.global + size_t(expert) * a.n;

  auto load_tile = [&](int tile, int stage) {
    T *a_stage = a_shared + stage * S::kStageA;
    for (int i = threadIdx.x; i < S::kRows * kAChunksPerRow; i += S::kThreads) {
      const int r = i / kAChunksPerRow;
      const int chunk = i % kAChunksPerRow;
      const int col = tile * kTileK + chunk * kAChunk;
      const int source = source_rows[r];
      const bool valid = source >= 0 && col < a.k;
      const T *src = valid ? input + size_t(source) * a.k + col : input;
      copy16(a_stage + r * kTileK + ((chunk ^ (r & 7)) * kAChunk), src, valid);
    }
    uint8_t *b_stage = b_shared + stage * S::kStageB;
    for (int i = threadIdx.x; i < kTileN * kBUnitsPerRow; i += S::kThreads) {
      const int r = i / kBUnitsPerRow;
      const int unit = i % kBUnitsPerRow;
      const int row = tile_n + r;
      const int col = tile * kTileK + unit * 2 * kAChunk;
      const bool valid = row < a.n && col < a.k;
      const uint8_t *src =
          valid ? weight + size_t(row) * packed_k + col / 2 : weight;
      const int swizzle = ((r >> 2) & 1) << 1;
      copy8(b_stage + r * kBRowBytes + ((unit ^ swizzle) * kBUnit), src, valid);
    }
  };

  const int warp = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const int pair = lane % 4;
  const bool aligned_scales = k_blocks % kScalesPerTile == 0;
  const uint8_t *scale_rows[kWarpTilesN];
  bool scale_valid[kWarpTilesN];
#pragma unroll
  for (int j = 0; j < kWarpTilesN; ++j) {
    const int row =
        tile_n + warp_n * kWarpTilesN * kMmaN + j * kMmaN + lane / 4;
    scale_valid[j] = row < a.n;
    scale_rows[j] = scales + size_t(scale_valid[j] ? row : 0) * k_blocks;
  }
  auto fetch_scales = [&](uint32_t *out, int tile) {
#pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j)
      out[j] = scale_valid[j]
                   ? load_scales(scale_rows[j], tile, k_blocks, aligned_scales)
                   : 0u;
  };

  float acc[TilesM][kWarpTilesN][4] = {};
  const T2 negative_bias = Tr::pair(-Tr::kBias);

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < tile_count)
      load_tile(tile_begin + s, s);
    commit();
  }
  uint32_t current[kWarpTilesN];
  if (tile_count > 0)
    fetch_scales(current, tile_begin);

  for (int t = 0; t < tile_count; ++t) {
    wait_stages<kStages - 2>();
    __syncthreads();
    const int next = t + kStages - 1;
    if (next < tile_count)
      load_tile(tile_begin + next, next % kStages);
    commit();
    uint32_t upcoming[kWarpTilesN] = {};
    if (t + 1 < tile_count)
      fetch_scales(upcoming, tile_begin + t + 1);

    const int stage = t % kStages;
    const T *a_stage = a_shared + stage * S::kStageA;
    const uint8_t *b_stage = b_shared + stage * S::kStageB;
    T2 scale_pairs[kWarpTilesN][kKSteps / 2];
#pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      scale_pairs[j][0] =
          Tr::from_halves(half_scales(current[j], kLowPairSelect));
      scale_pairs[j][1] =
          Tr::from_halves(half_scales(current[j], kHighPairSelect));
    }
#pragma unroll
    for (int step = 0; step < kKSteps; ++step) {
      uint32_t fragments[TilesM][4];
#pragma unroll
      for (int i = 0; i < TilesM; ++i) {
        const int r = warp_m * TilesM * kMmaM + i * kMmaM + lane % 16;
        const int chunk = step * 2 + lane / 16;
        load_matrix(fragments[i],
                    a_stage + r * kTileK + ((chunk ^ (r & 7)) * kAChunk));
      }
#pragma unroll
      for (int j = 0; j < kWarpTilesN; ++j) {
        const int r = warp_n * kWarpTilesN * kMmaN + j * kMmaN + lane / 4;
        const int swizzle = ((r >> 2) & 1) << 1;
        const uint2 words = *reinterpret_cast<const uint2 *>(
            b_stage + r * kBRowBytes + ((step ^ swizzle) * kBUnit));
        const uint32_t codes =
            __byte_perm(words.x, words.y, pair | ((pair + 4) << 4));
        const uint32_t bytes = lookup(codes);
        const T2 scale = step % 2 ? Tr::high(scale_pairs[j][step / 2])
                                  : Tr::low(scale_pairs[j][step / 2]);
        uint32_t low = __byte_perm(bytes, Tr::kMagic, kLowPairSelect);
        uint32_t high = __byte_perm(bytes, Tr::kMagic, kHighPairSelect);
        T2 b0 = *reinterpret_cast<T2 *>(&low);
        T2 b1 = *reinterpret_cast<T2 *>(&high);
        if constexpr (std::is_same_v<T, __half>) {
          // Valid E4M3 scales overflow bias * scale in F16; removing the bias first stays exact.
          b0 = __hmul2(__hadd2(b0, negative_bias), scale);
          b1 = __hmul2(__hadd2(b1, negative_bias), scale);
        } else {
          const T2 offset = __hmul2(scale, negative_bias);
          b0 = __hfma2(b0, scale, offset);
          b1 = __hfma2(b1, scale, offset);
        }
#pragma unroll
        for (int i = 0; i < TilesM; ++i)
          Tr::mma(acc[i][j], fragments[i], *reinterpret_cast<uint32_t *>(&b0),
                  *reinterpret_cast<uint32_t *>(&b1));
      }
    }
#pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j)
      current[j] = upcoming[j];
  }
  wait_stages<0>();

  T *output = static_cast<T *>(a.output);
  const bool even_n = a.n % 2 == 0;
#pragma unroll
  for (int j = 0; j < kWarpTilesN; ++j) {
    const int col =
        tile_n + warp_n * kWarpTilesN * kMmaN + j * kMmaN + pair * 2;
    if (col >= a.n)
      continue;
    const bool pair_valid = col + 1 < a.n;
    const float g0 = global[col];
    const float g1 = pair_valid ? global[col + 1] : 0.0f;
#pragma unroll
    for (int i = 0; i < TilesM; ++i) {
#pragma unroll
      for (int half_row = 0; half_row < 2; ++half_row) {
        const int r =
            warp_m * TilesM * kMmaM + i * kMmaM + lane / 4 + half_row * 8;
        const int row = output_rows[r];
        if (row < 0)
          continue;
        const float x = acc[i][j][half_row * 2] * g0;
        const float y = acc[i][j][half_row * 2 + 1] * g1;
        if (a.partial) {
          float *dst =
              a.partial + (size_t(blockIdx.z) * a.rows + row) * a.n + col;
          dst[0] = x;
          if (pair_valid)
            dst[1] = y;
        } else {
          T *dst = output + size_t(row) * a.n + col;
          if (pair_valid && even_n) {
            Tr::store2(dst, x, y);
          } else {
            dst[0] = static_cast<T>(x);
            if (pair_valid)
              dst[1] = static_cast<T>(y);
          }
        }
      }
    }
  }
#else
  __trap();
#endif
}

template <typename T>
__global__ void reduce(const float *partial, T *output, size_t elements,
                       int splits) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < elements;
       i += size_t(gridDim.x) * blockDim.x) {
    float sum = 0.0f;
    for (int s = 0; s < splits; ++s)
      sum += partial[s * elements + i];
    output[i] = static_cast<T>(sum);
  }
}

template <typename T, int WarpsM, int TilesM>
cudaError_t launch_shape(const Nvfp4W4a16Args &a, int row_tiles,
                         cudaStream_t stream) {
  using S = Shape<WarpsM, TilesM>;
  auto kernel = gemm<T, WarpsM, TilesM>;
  if (S::kSmem > kDefaultSmem) {
    const cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(S::kSmem));
    if (status != cudaSuccess)
      return status;
  }
  const dim3 grid((a.n + kTileN - 1) / kTileN, row_tiles, a.splits);
  kernel<<<grid, S::kThreads, S::kSmem, stream>>>(a);
  return cudaGetLastError();
}

template <typename T>
int run(const Nvfp4W4a16Args *args, int config, int row_tiles,
        cudaStream_t stream) {
  const Nvfp4W4a16Args &a = *args;
  cudaError_t status;
  switch (config) {
  case kRows16:
    status = launch_shape<T, 1, 1>(a, row_tiles, stream);
    break;
  case kRows64:
    status = launch_shape<T, 2, 2>(a, row_tiles, stream);
    break;
  case kRows128:
    status = launch_shape<T, 2, 4>(a, row_tiles, stream);
    break;
  default:
    return cudaErrorInvalidValue;
  }
  if (status != cudaSuccess || !a.partial)
    return status;
  const size_t elements = size_t(a.rows) * a.n;
  const int blocks =
      int(min((elements + kReduceThreads - 1) / kReduceThreads, size_t(65535)));
  reduce<T><<<blocks, kReduceThreads, 0, stream>>>(
      a.partial, static_cast<T *>(a.output), elements, a.splits);
  return cudaGetLastError();
}
} // namespace

extern "C" int mistralrs_nvfp4_w4a16_f16(const Nvfp4W4a16Args *args, int config,
                                         int row_tiles, cudaStream_t stream) {
  return run<__half>(args, config, row_tiles, stream);
}

extern "C" int mistralrs_nvfp4_w4a16_bf16(const Nvfp4W4a16Args *args,
                                          int config, int row_tiles,
                                          cudaStream_t stream) {
  return run<__nv_bfloat16>(args, config, row_tiles, stream);
}
