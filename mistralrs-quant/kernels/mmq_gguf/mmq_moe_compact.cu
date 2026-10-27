#include "mmq_gguf.cuh"

constexpr int COMPACT_EXPERTS = 512;
constexpr int COMPACT_COLUMNS = 16;
constexpr int COMPACT_GATE_COLUMNS = 128;
constexpr int COMPACT_DOWN_COLUMNS = 64;
constexpr int COMPACT_ROWS = 128;
constexpr int COMPACT_THREADS = 256;
constexpr int COMPACT_WARP = 32;
constexpr int COMPACT_CAPABILITY = 1210;
constexpr int COMPACT_HEADER_WORDS = 2;

__host__ __device__ constexpr int compact_wide_columns(ggml_type type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K
             ? COMPACT_GATE_COLUMNS
             : COMPACT_DOWN_COLUMNS;
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
};

__global__ void compact_mmq_worklist(const int32_t *bounds, int2 *tiles,
                                     int32_t *tile_count, int wide_columns) {
  __shared__ int warp_totals[COMPACT_WARP];
  const int expert = threadIdx.x;
  const int lane = expert % COMPACT_WARP;
  const int warp = expert / COMPACT_WARP;
  const int routed = bounds[expert + 1] - bounds[expert];
  const int columns = routed >= wide_columns ? wide_columns : COMPACT_COLUMNS;
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
    int value = lane < COMPACT_EXPERTS / COMPACT_WARP ? warp_totals[lane] : 0;
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
  if (expert == COMPACT_EXPERTS - 1) {
    *tile_count = offset;
  }
  for (int tile = 0; tile < count; ++tile) {
    tiles[offset - count + tile] = make_int2(expert, tile);
  }
}

template <ggml_type type>
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
  constexpr int wide_columns = compact_wide_columns(type);
  const int columns = count >= wide_columns ? wide_columns : COMPACT_COLUMNS;
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
  if (columns == wide_columns) {
    mul_mat_q_process_tile<type, wide_columns, false, false>(
        static_cast<const char *>(args.weight), weight_offset,
        static_cast<const int *>(args.activations) + activation_offset,
        ids_shared, args.output + row, GGML_TYPE_F32, nullptr,
        args.input_cols / qk, args.assignments, args.output_stride,
        args.output_rows - row - 1, count - column - 1, 0, args.input_cols / qk,
        true);
  } else {
    mul_mat_q_process_tile<type, COMPACT_COLUMNS, false, false>(
        static_cast<const char *>(args.weight), weight_offset,
        static_cast<const int *>(args.activations) + activation_offset,
        ids_shared, args.output + row, GGML_TYPE_F32, nullptr,
        args.input_cols / qk, args.assignments, args.output_stride,
        args.output_rows - row - 1, count - column - 1, 0, args.input_cols / qk,
        true);
  }
}

template <ggml_type type>
void launch_compact_mmq(const CompactMmqArgs &args, void *workspace,
                        cudaStream_t stream) {
  const int bytes = mmq_get_nbytes_shared<type>(
      compact_wide_columns(type), COMPACT_ROWS, COMPACT_CAPABILITY,
      COMPACT_WARP, COMPACT_THREADS / COMPACT_WARP);
  CUDA_SET_SHARED_MEMORY_LIMIT((compact_mmq<type>), bytes);
  // The sum of rounded expert counts is bounded without a host readback.
  const int capacity = args.assignments / COMPACT_COLUMNS + COMPACT_EXPERTS;
  const dim3 grid(args.output_rows / COMPACT_ROWS, capacity);
  const dim3 block(COMPACT_WARP, COMPACT_THREADS / COMPACT_WARP);
  const auto *count = static_cast<const int32_t *>(workspace);
  const auto *tiles =
      reinterpret_cast<const int2 *>(count + COMPACT_HEADER_WORDS);
  compact_mmq<type><<<grid, block, bytes, stream>>>(args, tiles, count);
}

extern "C" void launch_mmq_gguf_compact_prepare(const int32_t *bounds,
                                                void *workspace,
                                                int32_t quant_type,
                                                void *stream) {
  auto *count = static_cast<int32_t *>(workspace);
  auto *tiles = reinterpret_cast<int2 *>(count + COMPACT_HEADER_WORDS);
  const int wide_columns =
      compact_wide_columns(static_cast<ggml_type>(quant_type));
  compact_mmq_worklist<<<1, COMPACT_EXPERTS, 0,
                         static_cast<cudaStream_t>(stream)>>>(
      bounds, tiles, count, wide_columns);
}

extern "C" void launch_mmq_gguf_compact(const CompactMmqArgs *args,
                                        void *workspace, void *stream) {
  if (args->quant_type == GGML_TYPE_Q4_K) {
    launch_compact_mmq<GGML_TYPE_Q4_K>(*args, workspace,
                                       static_cast<cudaStream_t>(stream));
  } else if (args->quant_type == GGML_TYPE_Q5_K) {
    launch_compact_mmq<GGML_TYPE_Q5_K>(*args, workspace,
                                       static_cast<cudaStream_t>(stream));
  } else if (args->quant_type == GGML_TYPE_Q4_1) {
    launch_compact_mmq<GGML_TYPE_Q4_1>(*args, workspace,
                                       static_cast<cudaStream_t>(stream));
  } else if (args->quant_type == GGML_TYPE_Q5_1) {
    launch_compact_mmq<GGML_TYPE_Q5_1>(*args, workspace,
                                       static_cast<cudaStream_t>(stream));
  } else {
    launch_compact_mmq<GGML_TYPE_Q8_0>(*args, workspace,
                                       static_cast<cudaStream_t>(stream));
  }
}
