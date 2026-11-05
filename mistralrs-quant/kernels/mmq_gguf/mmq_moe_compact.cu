#include "mmq_gguf.cuh"

constexpr int COMPACT_MAX_EXPERT_THREADS = 1024;
constexpr int COMPACT_COLUMNS = 16;
constexpr int COMPACT_WIDE_COLUMNS = 128;
constexpr int COMPACT_MEDIUM_COLUMNS = 64;
constexpr int COMPACT_ROWS = 128;
constexpr int COMPACT_THREADS = 256;
constexpr int COMPACT_WARP = 32;
constexpr int COMPACT_HEADER_WORDS = 2;

__device__ __forceinline__ int compact_columns(int routed, int maximum) {
  int columns = COMPACT_COLUMNS;
  while (columns < maximum && columns < routed) {
    columns *= 2;
  }
  return columns;
}

struct CompactMmqArgs {
  const void *weight;
  const void *activations;
  const int32_t *ids_dst;
  const int32_t *expert_bounds;
  float *output;
  int32_t input_cols;
  int32_t output_rows;
  int32_t assignments;
  int32_t output_stride;
  int32_t quant_type;
  int32_t experts;
  int32_t wide_columns;
};

__global__ void compact_mmq_worklist(const int32_t *bounds, int2 *tiles,
                                     int32_t *tile_count, int experts,
                                     int wide_columns) {
  __shared__ int warp_totals[COMPACT_WARP];
  __shared__ int preceding;
  const int lane = threadIdx.x % COMPACT_WARP;
  const int warp = threadIdx.x / COMPACT_WARP;
  if (threadIdx.x == 0) {
    preceding = 0;
  }
  __syncthreads();
  for (int base = 0; base < experts; base += blockDim.x) {
    const int expert = base + threadIdx.x;
    const int routed =
        expert < experts ? bounds[expert + 1] - bounds[expert] : 0;
    const int columns = compact_columns(routed, wide_columns);
    const int count = (routed + columns - 1) / columns;
    int offset = count;
#pragma unroll
    for (int shift = 1; shift < COMPACT_WARP; shift *= 2) {
      const int value = __shfl_up_sync(0xffffffff, offset, shift);
      if (lane >= shift) {
        offset += value;
      }
    }
    if (lane == COMPACT_WARP - 1) {
      warp_totals[warp] = offset;
    }
    __syncthreads();
    if (warp == 0) {
      int value = lane < blockDim.x / COMPACT_WARP ? warp_totals[lane] : 0;
#pragma unroll
      for (int shift = 1; shift < COMPACT_WARP; shift *= 2) {
        const int previous = __shfl_up_sync(0xffffffff, value, shift);
        if (lane >= shift) {
          value += previous;
        }
      }
      warp_totals[lane] = value;
    }
    __syncthreads();
    if (warp > 0) {
      offset += warp_totals[warp - 1];
    }
    offset += preceding;
    for (int tile = 0; tile < count; ++tile) {
      tiles[offset - count + tile] = make_int2(expert, tile);
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
      preceding = offset;
      *tile_count = offset;
    }
    __syncthreads();
  }
}

template <ggml_type type, int wide_columns, bool need_check>
__global__ __launch_bounds__(COMPACT_THREADS,
                             1) void compact_mmq(CompactMmqArgs args,
                                                 const int2 *tiles,
                                                 const int32_t *tile_count) {
  if (blockIdx.y >= *tile_count) {
    return;
  }
  const int2 tile = tiles[blockIdx.y];
  const int first = args.expert_bounds[tile.x];
  const int count = args.expert_bounds[tile.x + 1] - first;
  const int columns = compact_columns(count, wide_columns);
  const int column = tile.y * columns;
  const int thread = threadIdx.y * COMPACT_WARP + threadIdx.x;
  extern __shared__ int ids_shared[];
  if (thread < columns) {
    ids_shared[thread] =
        column + thread < count ? args.ids_dst[first + column + thread] : 0;
  }
  __syncthreads();
  constexpr int qk = ggml_cuda_type_traits<type>::qk;
  const int row = blockIdx.x * COMPACT_ROWS;
  const int weight_offset =
      (tile.x * args.output_rows + row) * (args.input_cols / qk);
  const int activation_offset =
      (first + column) * sizeof(block_q8_1_mmq) / sizeof(int);
#define COMPACT_TILE(COLUMNS)                                                  \
  mul_mat_q_process_tile<type, COLUMNS, need_check, false>(                    \
      static_cast<const char *>(args.weight), weight_offset,                   \
      static_cast<const int *>(args.activations) + activation_offset,          \
      ids_shared, args.output + row, GGML_TYPE_F32, nullptr,                   \
      args.input_cols / qk, args.assignments, args.output_stride,              \
      args.output_rows - row - 1, count - column - 1, 0, args.input_cols / qk, \
      true)
  if (columns == COMPACT_COLUMNS) {
    COMPACT_TILE(COMPACT_COLUMNS);
  } else if (columns == 2 * COMPACT_COLUMNS) {
    COMPACT_TILE(2 * COMPACT_COLUMNS);
  } else if (columns == COMPACT_MEDIUM_COLUMNS) {
    COMPACT_TILE(COMPACT_MEDIUM_COLUMNS);
  } else {
    COMPACT_TILE(wide_columns);
  }
#undef COMPACT_TILE
}

static int compact_device_capability() {
  static int capability[GGML_CUDA_MAX_DEVICES] = {};
  int device;
  cudaGetDevice(&device);
  if (capability[device] == 0) {
    int major;
    int minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    capability[device] = 100 * major + 10 * minor;
  }
  return capability[device];
}

template <ggml_type type, int wide_columns, bool need_check>
void launch_compact_mmq(const CompactMmqArgs &args, void *workspace,
                        cudaStream_t stream) {
  const int bytes = mmq_get_nbytes_shared<type>(
      wide_columns, COMPACT_ROWS, compact_device_capability(), COMPACT_WARP,
      COMPACT_THREADS / COMPACT_WARP);
  CUDA_SET_SHARED_MEMORY_LIMIT((compact_mmq<type, wide_columns, need_check>),
                               bytes);
  // The sum of rounded expert counts is bounded without a host readback.
  const int capacity = args.assignments / COMPACT_COLUMNS + args.experts;
  const dim3 grid((args.output_rows + COMPACT_ROWS - 1) / COMPACT_ROWS,
                  capacity);
  const dim3 block(COMPACT_WARP, COMPACT_THREADS / COMPACT_WARP);
  const auto *count = static_cast<const int32_t *>(workspace);
  const auto *tiles =
      reinterpret_cast<const int2 *>(count + COMPACT_HEADER_WORDS);
  compact_mmq<type, wide_columns, need_check>
      <<<grid, block, bytes, stream>>>(args, tiles, count);
}

template <ggml_type type, int wide_columns>
void launch_compact_rows(const CompactMmqArgs &args, void *workspace,
                         cudaStream_t stream) {
  if (args.output_rows % COMPACT_ROWS == 0) {
    launch_compact_mmq<type, wide_columns, false>(args, workspace, stream);
  } else {
    launch_compact_mmq<type, wide_columns, true>(args, workspace, stream);
  }
}

template <ggml_type type>
void launch_compact_format(const CompactMmqArgs &args, void *workspace,
                           cudaStream_t stream) {
  if (args.wide_columns == COMPACT_WIDE_COLUMNS) {
    launch_compact_rows<type, COMPACT_WIDE_COLUMNS>(args, workspace, stream);
  } else {
    launch_compact_rows<type, COMPACT_MEDIUM_COLUMNS>(args, workspace, stream);
  }
}

extern "C" void launch_mmq_gguf_compact_prepare(const int32_t *bounds,
                                                void *workspace,
                                                int32_t experts,
                                                int32_t wide_columns,
                                                void *stream) {
  auto *count = static_cast<int32_t *>(workspace);
  auto *tiles = reinterpret_cast<int2 *>(count + COMPACT_HEADER_WORDS);
  const int threads =
      std::min(COMPACT_MAX_EXPERT_THREADS,
               (experts + COMPACT_WARP - 1) / COMPACT_WARP * COMPACT_WARP);
  compact_mmq_worklist<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      bounds, tiles, count, experts, wide_columns);
}

extern "C" void launch_mmq_gguf_compact(const CompactMmqArgs *args,
                                        void *workspace, void *stream) {
#define COMPACT_FORMAT(TYPE)                                                   \
  case TYPE:                                                                   \
    launch_compact_format<TYPE>(*args, workspace,                              \
                                static_cast<cudaStream_t>(stream));            \
    break;
  switch (args->quant_type) {
    COMPACT_FORMAT(GGML_TYPE_Q4_0)
    COMPACT_FORMAT(GGML_TYPE_Q4_1)
    COMPACT_FORMAT(GGML_TYPE_Q5_0)
    COMPACT_FORMAT(GGML_TYPE_Q5_1)
    COMPACT_FORMAT(GGML_TYPE_Q8_0)
    COMPACT_FORMAT(GGML_TYPE_Q2_K)
    COMPACT_FORMAT(GGML_TYPE_Q3_K)
    COMPACT_FORMAT(GGML_TYPE_Q4_K)
    COMPACT_FORMAT(GGML_TYPE_Q5_K)
    COMPACT_FORMAT(GGML_TYPE_Q6_K)
  default:
    break;
  }
#undef COMPACT_FORMAT
}
