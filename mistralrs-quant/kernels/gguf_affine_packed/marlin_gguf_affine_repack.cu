// Portions of this file are adapted from the vLLM project
// (https://github.com/vllm-project/vllm)
// Licensed under the Apache License 2.0
// Copyright contributors to the vLLM project

#include "gguf_affine_packed.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int QK32 = 32;
constexpr int QK_K = 256;
constexpr int K_SCALE_SIZE = 12;
constexpr int MARLIN_TILE_K = 16;
constexpr int MARLIN_TILE_N = 64;
constexpr int MARLIN_WIDE_TILE = 128;
constexpr int CUDA_THREADS = 256;

struct BlockQ4_0 {
  __half d;
  uint8_t qs[QK32 / 2];
};

struct __align__(4) BlockQ4_1 {
  __half2 dm;
  uint8_t qs[QK32 / 2];
};

struct BlockQ5_0 {
  __half d;
  uint8_t qh[4];
  uint8_t qs[QK32 / 2];
};

struct __align__(4) BlockQ5_1 {
  __half2 dm;
  uint8_t qh[4];
  uint8_t qs[QK32 / 2];
};

struct BlockQ8_0 {
  __half d;
  int8_t qs[QK32];
};

struct __align__(4) BlockQ8_1 {
  __half2 ds;
  int8_t qs[QK32];
};

struct __align__(4) BlockQ2K {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  __half2 dm;
};

struct BlockQ3K {
  uint8_t hmask[QK_K / 8];
  uint8_t qs[QK_K / 4];
  uint8_t scales[K_SCALE_SIZE];
  __half d;
};

struct __align__(4) BlockQ4K {
  __half2 dm;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qs[QK_K / 2];
};

struct __align__(4) BlockQ5K {
  __half2 dm;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
};

struct BlockQ6K {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  __half d;
};

struct BlockQ8K {
  float d;
  int8_t qs[QK_K];
  int16_t bsums[QK_K / 16];
};

static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ4_1) == 20);
static_assert(sizeof(BlockQ5_0) == 22);
static_assert(sizeof(BlockQ5_1) == 24);
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(sizeof(BlockQ8_1) == 36);
static_assert(sizeof(BlockQ2K) == 84);
static_assert(sizeof(BlockQ3K) == 110);
static_assert(sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ5K) == 176);
static_assert(sizeof(BlockQ6K) == 210);
static_assert(sizeof(BlockQ8K) == 292);

template <int Format> struct FormatTraits;

#define MRS_FORMAT_TRAITS(FORMAT, BLOCK, BLOCK_SIZE, BITS, GROUP_SIZE)         \
  template <> struct FormatTraits<FORMAT> {                                    \
    using Block = BLOCK;                                                       \
    static constexpr int block_size = BLOCK_SIZE;                              \
    static constexpr int bits = BITS;                                          \
    static constexpr int group_size = GROUP_SIZE;                              \
  }

MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q4_0, BlockQ4_0, QK32, 4, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q4_1, BlockQ4_1, QK32, 4, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q5_0, BlockQ5_0, QK32, 8, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q5_1, BlockQ5_1, QK32, 8, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q8_0, BlockQ8_0, QK32, 8, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q8_1, BlockQ8_1, QK32, 8, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q2_K, BlockQ2K, QK_K, 4, 16);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q3_K, BlockQ3K, QK_K, 4, 16);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q4_K, BlockQ4K, QK_K, 4, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q5_K, BlockQ5K, QK_K, 8, 32);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q6_K, BlockQ6K, QK_K, 8, 16);
MRS_FORMAT_TRAITS(MRS_GGUF_AFFINE_Q8_K, BlockQ8K, QK_K, 8, 32);

#undef MRS_FORMAT_TRAITS

template <int Format>
__device__ __forceinline__ const typename FormatTraits<Format>::Block &
get_block(const void *src, int k, int row, int column) {
  using Traits = FormatTraits<Format>;
  using Block = typename Traits::Block;
  const int blocks_per_row = k / Traits::block_size;
  return static_cast<const Block *>(
      src)[row * blocks_per_row + column / Traits::block_size];
}

__device__ __forceinline__ uint8_t get_nibble(const uint8_t *qs, int local) {
  const uint8_t packed = qs[local % (QK32 / 2)];
  return local < QK32 / 2 ? packed & 15 : packed >> 4;
}

template <int Format>
__device__ __forceinline__ uint8_t get_quant(const void *src, int k, int row,
                                             int column) {
  const auto &block = get_block<Format>(src, k, row, column);
  constexpr int block_size = FormatTraits<Format>::block_size;
  const int local = column % block_size;

  if constexpr (Format == MRS_GGUF_AFFINE_Q4_0 ||
                Format == MRS_GGUF_AFFINE_Q4_1) {
    return get_nibble(block.qs, local);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q5_0 ||
                       Format == MRS_GGUF_AFFINE_Q5_1) {
    const uint8_t high = (block.qh[local / 8] >> (local % 8)) & 1;
    return get_nibble(block.qs, local) | (high << 4);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q8_0 ||
                       Format == MRS_GGUF_AFFINE_Q8_1 ||
                       Format == MRS_GGUF_AFFINE_Q8_K) {
    return static_cast<uint8_t>(static_cast<int>(block.qs[local]) + 128);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q2_K) {
    const int packed_index = local / 128 * 32 + local % 32;
    const int shift = local % 128 / 32 * 2;
    return (block.qs[packed_index] >> shift) & 3;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q3_K) {
    const int packed_index = local / 128 * 32 + local % 32;
    const int shift = local % 128 / 32 * 2;
    const uint8_t low = (block.qs[packed_index] >> shift) & 3;
    const uint8_t high = (block.hmask[local % 32] >> (local / 32)) & 1;
    return low | (high << 2);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q4_K) {
    const int chunk = local / 64;
    const int position = local % 64;
    const uint8_t packed = block.qs[chunk * 32 + position % 32];
    return position < 32 ? packed & 15 : packed >> 4;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q5_K) {
    const int chunk = local / 64;
    const int position = local % 64;
    const uint8_t packed = block.qs[chunk * 32 + position % 32];
    const uint8_t low = position < 32 ? packed & 15 : packed >> 4;
    const int high_shift = chunk * 2 + position / 32;
    const uint8_t high = (block.qh[position % 32] >> high_shift) & 1;
    return low | (high << 4);
  } else {
    const int half = local / 128;
    const int position = local % 32;
    const int quarter = local % 128 / 32;
    const int packed_index = half * 64 + position + (quarter % 2) * 32;
    const int high_shift = quarter * 2;
    const uint8_t low =
        quarter < 2 ? block.ql[packed_index] & 15 : block.ql[packed_index] >> 4;
    const uint8_t high = (block.qh[half * 32 + position] >> high_shift) & 3;
    return low | (high << 4);
  }
}

__device__ __forceinline__ void get_scale_min_k4(int group,
                                                 const uint8_t *packed,
                                                 uint8_t &scale, uint8_t &min) {
  if (group < 4) {
    scale = packed[group] & 63;
    min = packed[group + 4] & 63;
  } else {
    scale = (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4);
    min = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
  }
}

__device__ __forceinline__ int get_q3_scale(const uint8_t *packed, int group) {
  const uint8_t low = group < 8 ? packed[group] & 15 : packed[group - 8] >> 4;
  const uint8_t high = (packed[8 + group % 4] >> (2 * (group / 4))) & 3;
  return static_cast<int>(low | (high << 4)) - 32;
}

template <int Format>
__device__ __forceinline__ void get_affine_params(const void *src, int k,
                                                  int row, int group,
                                                  float &scale, float &offset) {
  constexpr int group_size = FormatTraits<Format>::group_size;
  const int column = group * group_size;
  const auto &block = get_block<Format>(src, k, row, column);
  constexpr int block_size = FormatTraits<Format>::block_size;
  const int local_group = column % block_size / group_size;

  if constexpr (Format == MRS_GGUF_AFFINE_Q4_0) {
    scale = __half2float(block.d);
    offset = 8.0f * scale;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q4_1) {
    scale = __half2float(__low2half(block.dm));
    offset = -__half2float(__high2half(block.dm));
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q5_0) {
    scale = __half2float(block.d);
    offset = 16.0f * scale;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q5_1) {
    scale = __half2float(__low2half(block.dm));
    offset = -__half2float(__high2half(block.dm));
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q8_0 ||
                       Format == MRS_GGUF_AFFINE_Q8_1) {
    if constexpr (Format == MRS_GGUF_AFFINE_Q8_0) {
      scale = __half2float(block.d);
    } else {
      scale = __half2float(__low2half(block.ds));
    }
    offset = 128.0f * scale;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q2_K) {
    const uint8_t packed = block.scales[local_group];
    const float d = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));
    scale = d * static_cast<float>(packed & 15);
    offset = dmin * static_cast<float>(packed >> 4);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q3_K) {
    scale = __half2float(block.d) *
            static_cast<float>(get_q3_scale(block.scales, local_group));
    offset = 4.0f * scale;
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q4_K ||
                       Format == MRS_GGUF_AFFINE_Q5_K) {
    uint8_t quant_scale;
    uint8_t quant_min;
    get_scale_min_k4(local_group, block.scales, quant_scale, quant_min);
    const float d = __half2float(__low2half(block.dm));
    const float dmin = __half2float(__high2half(block.dm));
    scale = d * static_cast<float>(quant_scale);
    offset = dmin * static_cast<float>(quant_min);
  } else if constexpr (Format == MRS_GGUF_AFFINE_Q6_K) {
    scale =
        __half2float(block.d) * static_cast<float>(block.scales[local_group]);
    offset = 32.0f * scale;
  } else {
    scale = block.d;
    offset = 128.0f * scale;
  }
}

template <typename T> struct Scalar;

template <> struct Scalar<__half> {
  __device__ static __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
  }
};

template <> struct Scalar<__nv_bfloat16> {
  __device__ static __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <int Format>
__global__ void repack_payload_u4_kernel(const void *__restrict__ src,
                                         uint32_t *__restrict__ payload, int k,
                                         int n, int padded_n) {
  constexpr int tile_words = MARLIN_TILE_K * MARLIN_TILE_N / 8;
  const size_t payload_words = static_cast<size_t>(k) * padded_n / 8;
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_index >= payload_words) {
    return;
  }

  const int n_tiles = padded_n / MARLIN_TILE_N;
  const size_t tile = word_index / tile_words;
  const int tile_word = word_index % tile_words;
  const int k_tile = tile / n_tiles;
  const int n_tile = tile % n_tiles;
  const int marlin_thread = tile_word / 4;
  const int marlin_warp = tile_word % 4;
  const int tensor_column = marlin_thread / 4;
  const int tensor_row = marlin_thread % 4 * 2;
  const int global_k = k_tile * MARLIN_TILE_K + tensor_row;
  const int global_n =
      n_tile * MARLIN_TILE_N + marlin_warp * 16 + tensor_column;

  uint8_t values[8] = {};
  if (global_n < n) {
    values[0] = get_quant<Format>(src, k, global_n, global_k);
    values[1] = get_quant<Format>(src, k, global_n, global_k + 1);
    values[2] = get_quant<Format>(src, k, global_n, global_k + 8);
    values[3] = get_quant<Format>(src, k, global_n, global_k + 9);
  }
  if (global_n + 8 < n) {
    values[4] = get_quant<Format>(src, k, global_n + 8, global_k);
    values[5] = get_quant<Format>(src, k, global_n + 8, global_k + 1);
    values[6] = get_quant<Format>(src, k, global_n + 8, global_k + 8);
    values[7] = get_quant<Format>(src, k, global_n + 8, global_k + 9);
  }

  constexpr int pack_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    packed |= static_cast<uint32_t>(values[pack_order[index]]) << (4 * index);
  }
  payload[word_index] = packed;
}

template <int Format>
__global__ void repack_payload_u8_kernel(const void *__restrict__ src,
                                         uint32_t *__restrict__ payload, int k,
                                         int n, int padded_n) {
  constexpr int tile_words = MARLIN_TILE_K * MARLIN_TILE_N / 4;
  const size_t payload_words = static_cast<size_t>(k) * padded_n / 4;
  const size_t word_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_index >= payload_words) {
    return;
  }

  const int n_tiles = padded_n / MARLIN_TILE_N;
  const size_t tile = word_index / tile_words;
  const int tile_word = word_index % tile_words;
  const int k_tile = tile / n_tiles;
  const int n_tile = tile % n_tiles;
  const int marlin_thread = tile_word / 8;
  const int remainder = tile_word % 8;
  const int marlin_warp = remainder / 2;
  const int column_half = remainder % 2;
  const int tensor_column = marlin_thread / 4;
  const int tensor_row = marlin_thread % 4 * 2;
  const int global_k = k_tile * MARLIN_TILE_K + tensor_row;
  const int global_n = n_tile * MARLIN_TILE_N + marlin_warp * 16 +
                       tensor_column + column_half * 8;

  uint8_t values[4] = {};
  if (global_n < n) {
    values[0] = get_quant<Format>(src, k, global_n, global_k);
    values[1] = get_quant<Format>(src, k, global_n, global_k + 1);
    values[2] = get_quant<Format>(src, k, global_n, global_k + 8);
    values[3] = get_quant<Format>(src, k, global_n, global_k + 9);
  }

  constexpr int pack_order[4] = {0, 2, 1, 3};
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    packed |= static_cast<uint32_t>(values[pack_order[index]]) << (8 * index);
  }
  payload[word_index] = packed;
}

template <typename T, int Format>
__global__ void
repack_metadata_kernel(const void *__restrict__ src, T *__restrict__ scales,
                       T *__restrict__ offsets, int k, int n, int padded_n) {
  constexpr int group_size = FormatTraits<Format>::group_size;
  const int groups = k / group_size;
  const size_t values = static_cast<size_t>(groups) * padded_n;
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= values) {
    return;
  }

  const int group = index / padded_n;
  const int row = index % padded_n;
  const int permuted_row = row / MARLIN_TILE_N * MARLIN_TILE_N + row % 8 * 8 +
                           row % MARLIN_TILE_N / 8;
  const size_t output_index =
      static_cast<size_t>(group) * padded_n + permuted_row;
  if (row >= n) {
    scales[output_index] = Scalar<T>::from_float(0.0f);
    offsets[output_index] = Scalar<T>::from_float(0.0f);
    return;
  }

  float scale;
  float offset;
  get_affine_params<Format>(src, k, row, group, scale, offset);
  scales[output_index] = Scalar<T>::from_float(scale);
  offsets[output_index] = Scalar<T>::from_float(offset);
}

inline cudaStream_t as_stream(uintptr_t stream) {
  return reinterpret_cast<cudaStream_t>(stream);
}

struct RepackArgs {
  const void *src;
  void *payload;
  void *scales;
  void *offsets;
  int k;
  int n;
  int padded_n;
  uintptr_t stream;
};

template <int Format> inline bool valid_shape(const RepackArgs &args) {
  constexpr int block_size = FormatTraits<Format>::block_size;
  return args.src != nullptr && args.payload != nullptr &&
         args.scales != nullptr && args.offsets != nullptr && args.k > 0 &&
         args.n > 0 && args.k % block_size == 0 && args.padded_n >= args.n &&
         ((args.k % MARLIN_WIDE_TILE == 0 &&
           args.padded_n % MARLIN_TILE_N == 0) ||
          (args.k % MARLIN_TILE_N == 0 &&
           args.padded_n % MARLIN_WIDE_TILE == 0));
}

template <typename T, int Format>
int launch_repack_format(const RepackArgs &args) {
  if (!valid_shape<Format>(args)) {
    return MRS_GGUF_AFFINE_INVALID_ARGUMENT;
  }

  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  constexpr int bits = FormatTraits<Format>::bits;
  constexpr int group_size = FormatTraits<Format>::group_size;
  const int output_n = args.padded_n;
  const size_t payload_words =
      static_cast<size_t>(args.k) * output_n * bits / 32;
  const size_t metadata_values =
      static_cast<size_t>(args.k / group_size) * output_n;
  const int payload_blocks =
      static_cast<int>((payload_words + CUDA_THREADS - 1) / CUDA_THREADS);
  const int metadata_blocks =
      static_cast<int>((metadata_values + CUDA_THREADS - 1) / CUDA_THREADS);
  cudaStream_t stream = as_stream(args.stream);

  if constexpr (bits == 4) {
    repack_payload_u4_kernel<Format>
        <<<payload_blocks, CUDA_THREADS, 0, stream>>>(
            args.src, static_cast<uint32_t *>(args.payload), args.k, args.n,
            output_n);
  } else {
    repack_payload_u8_kernel<Format>
        <<<payload_blocks, CUDA_THREADS, 0, stream>>>(
            args.src, static_cast<uint32_t *>(args.payload), args.k, args.n,
            output_n);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  repack_metadata_kernel<T, Format>
      <<<metadata_blocks, CUDA_THREADS, 0, stream>>>(
          args.src, static_cast<T *>(args.scales),
          static_cast<T *>(args.offsets), args.k, args.n, output_n);
  return static_cast<int>(cudaGetLastError());
}

template <typename T> int launch_repack(int format, const RepackArgs &args) {
  switch (format) {
  case MRS_GGUF_AFFINE_Q4_0:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q4_0>(args);
  case MRS_GGUF_AFFINE_Q4_1:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q4_1>(args);
  case MRS_GGUF_AFFINE_Q5_0:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q5_0>(args);
  case MRS_GGUF_AFFINE_Q5_1:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q5_1>(args);
  case MRS_GGUF_AFFINE_Q8_0:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q8_0>(args);
  case MRS_GGUF_AFFINE_Q8_1:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q8_1>(args);
  case MRS_GGUF_AFFINE_Q2_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q2_K>(args);
  case MRS_GGUF_AFFINE_Q3_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q3_K>(args);
  case MRS_GGUF_AFFINE_Q4_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q4_K>(args);
  case MRS_GGUF_AFFINE_Q5_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q5_K>(args);
  case MRS_GGUF_AFFINE_Q6_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q6_K>(args);
  case MRS_GGUF_AFFINE_Q8_K:
    return launch_repack_format<T, MRS_GGUF_AFFINE_Q8_K>(args);
  default:
    return MRS_GGUF_AFFINE_INVALID_ARGUMENT;
  }
}

} // namespace

extern "C" int mrs_gguf_affine_repack_f16(int format, const void *src,
                                          void *payload, void *scales,
                                          void *offsets, int k, int n,
                                          int padded_n, uintptr_t stream) {
  return launch_repack<__half>(
      format, {src, payload, scales, offsets, k, n, padded_n, stream});
}

extern "C" int mrs_gguf_affine_repack_bf16(int format, const void *src,
                                           void *payload, void *scales,
                                           void *offsets, int k, int n,
                                           int padded_n, uintptr_t stream) {
  return launch_repack<__nv_bfloat16>(
      format, {src, payload, scales, offsets, k, n, padded_n, stream});
}
