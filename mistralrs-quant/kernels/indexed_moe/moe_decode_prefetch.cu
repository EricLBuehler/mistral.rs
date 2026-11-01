#include "../gguf/vecdot.cuh"
#include <cuda_runtime.h>

constexpr int MOE_WARPS = 8;
constexpr int MOE_ROWS_PER_WARP = 2;
constexpr size_t MOE_STAGED_OPERAND_BYTES = 24 * 1024;

struct MoeDecodeArgs {
  const void *weights;
  const void *up;
  const block_q8_1 *input;
  const unsigned int *indices;
  const float *routing;
  float *output;
  int rows;
  int columns;
  int topk;
  int input_stride;
  int activation;
};

template <int qk, int qi, typename Weight, int vdr, vec_dot_q_cuda_t dot,
          bool gate>
__global__ void moe_decode_prefetch(MoeDecodeArgs args) {
  const int blocks_per_row = args.columns / qk;
  constexpr int blocks_per_iteration = vdr * WARP_SIZE / qi;
  const int lane = threadIdx.x;
  const int row = (blockIdx.x * MOE_WARPS + threadIdx.y) * MOE_ROWS_PER_WARP;
  const int slot = blockIdx.z * args.topk + blockIdx.y;
  const size_t expert_offset =
      (size_t)args.indices[slot] * args.rows * blocks_per_row;
  const auto *weight =
      static_cast<const Weight *>(args.weights) + expert_offset;
  const auto *up_weight = static_cast<const Weight *>(args.up) + expert_offset;
  const auto *activation =
      args.input + (gate ? blockIdx.z : slot) * args.input_stride / QK8_1;
  float gate_sums[MOE_ROWS_PER_WARP] = {};
  float up_sums[MOE_ROWS_PER_WARP] = {};
  for (int block = lane / (qi / vdr); block < blocks_per_row;
       block += blocks_per_iteration) {
    const int part = vdr * (lane % (qi / vdr));
#pragma unroll
    for (int r = 0; r < MOE_ROWS_PER_WARP; ++r) {
      if (row + r >= args.rows) {
        continue;
      }
      const int future = block + blocks_per_iteration;
      const size_t offset = (size_t)(row + r) * blocks_per_row;
      if (future < blocks_per_row) {
        asm volatile(
            "prefetch.global.L2 [%0];" ::"l"(weight + offset + future));
        if constexpr (gate) {
          asm volatile(
              "prefetch.global.L2 [%0];" ::"l"(up_weight + offset + future));
        }
      }
      gate_sums[r] +=
          dot(weight + offset + block, activation + block * (qk / QK8_1), part);
      if constexpr (gate) {
        up_sums[r] += dot(up_weight + offset + block,
                          activation + block * (qk / QK8_1), part);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < MOE_ROWS_PER_WARP; ++r) {
    gate_sums[r] = warp_reduce_sum(gate_sums[r]);
    if constexpr (gate)
      up_sums[r] = warp_reduce_sum(up_sums[r]);
    if (lane == 0 && row + r < args.rows) {
      if constexpr (gate) {
        const float g = gate_sums[r];
        const float activated =
            args.activation == 0
                ? 0.5f * g *
                      (1.0f +
                       tanhf(0.7978845608f * (g + 0.044715f * g * g * g)))
                : g / (1.0f + expf(-g));
        args.output[slot * args.rows + row + r] = up_sums[r] * activated;
      } else {
        atomicAdd(args.output + blockIdx.z * args.rows + row + r,
                  gate_sums[r] * args.routing[slot]);
      }
    }
  }
}

#define MOE_PREFETCH_LAUNCHERS(TAG, DTYPE, ID, QK, QI, WEIGHT, VDR, DOT)       \
  extern "C" void launch_moe_gemv_fused_gate_up_##TAG##_prefetch(              \
      const void *gate, const void *up, const void *input,                     \
      const unsigned int *ids, float *output, int rows, int columns,           \
      int batch, int topk, int stride, int activation, void *stream) {         \
    const dim3 grid((rows + MOE_WARPS * MOE_ROWS_PER_WARP - 1) /               \
                        (MOE_WARPS * MOE_ROWS_PER_WARP),                       \
                    topk, batch);                                              \
    const MoeDecodeArgs args{                                                  \
        gate,   up,        static_cast<const block_q8_1 *>(input),             \
        ids,    nullptr,   output,                                             \
        rows,   columns,   topk,                                               \
        stride, activation};                                                   \
    moe_decode_prefetch<QK, QI, WEIGHT, VDR, DOT, true>                        \
        <<<grid, dim3(WARP_SIZE, MOE_WARPS), 0,                                \
           static_cast<cudaStream_t>(stream)>>>(args);                         \
  }                                                                            \
  extern "C" void launch_moe_gemv_down_aggregate_##TAG##_prefetch(             \
      const void *weight, const void *input, const unsigned int *ids,          \
      const float *routing, float *output, int rows, int columns, int batch,   \
      int topk, int stride, void *stream) {                                    \
    const dim3 grid((rows + MOE_WARPS * MOE_ROWS_PER_WARP - 1) /               \
                        (MOE_WARPS * MOE_ROWS_PER_WARP),                       \
                    topk, batch);                                              \
    const MoeDecodeArgs args{                                                  \
        weight, weight,  static_cast<const block_q8_1 *>(input),               \
        ids,    routing, output,                                               \
        rows,   columns, topk,                                                 \
        stride, 0};                                                            \
    moe_decode_prefetch<QK, QI, WEIGHT, VDR, DOT, false>                       \
        <<<grid, dim3(WARP_SIZE, MOE_WARPS), 0,                                \
           static_cast<cudaStream_t>(stream)>>>(args);                         \
  }

GGUF_VECTOR_FORMATS(MOE_PREFETCH_LAUNCHERS)

template <int Bytes>
__device__ __forceinline__ void copy_moe_stage(unsigned char *dst,
                                               const unsigned char *src) {
#if __CUDA_ARCH__ >= 800
  const unsigned address = unsigned(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(address),
               "l"(src), "n"(Bytes));
#else
  if constexpr (Bytes == 16)
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
  else if constexpr (Bytes == 8)
    *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
  else
    *reinterpret_cast<unsigned *>(dst) =
        *reinterpret_cast<const unsigned *>(src);
#endif
}

template <int QK, int QI, typename Weight, int VDR, vec_dot_q_cuda_t Dot,
          bool Gate, int RowsPerWarp, int Bytes>
__global__ void moe_decode_staged(MoeDecodeArgs args) {
  extern __shared__ __align__(16) unsigned char shared[];
  constexpr int tile_rows = MOE_WARPS * RowsPerWarp;
  const int blocks_per_row = args.columns / QK;
  const int row_bytes = blocks_per_row * sizeof(Weight);
  const int first_row = blockIdx.x * tile_rows;
  const int valid_rows = min(tile_rows, args.rows - first_row);
  const int panel_bytes = valid_rows * row_bytes;
  const int up_offset = (tile_rows * row_bytes + 15) / 16 * 16;
  const int slot = blockIdx.z * args.topk + blockIdx.y;
  const size_t expert_offset =
      size_t(args.indices[slot]) * args.rows * blocks_per_row;
  const auto *weight = static_cast<const Weight *>(args.weights) +
                       expert_offset + size_t(first_row) * blocks_per_row;
  const auto *up_weight = static_cast<const Weight *>(args.up) + expert_offset +
                          size_t(first_row) * blocks_per_row;
  const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int offset = tid * Bytes; offset < panel_bytes;
       offset += MOE_WARPS * WARP_SIZE * Bytes) {
    copy_moe_stage<Bytes>(shared + offset,
                          reinterpret_cast<const unsigned char *>(weight) +
                              offset);
    if constexpr (Gate) {
      copy_moe_stage<Bytes>(shared + up_offset + offset,
                            reinterpret_cast<const unsigned char *>(up_weight) +
                                offset);
    }
  }
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
#endif
  __syncthreads();
  weight = reinterpret_cast<const Weight *>(shared);
  up_weight = reinterpret_cast<const Weight *>(shared + up_offset);
  const auto *activation =
      args.input + (Gate ? blockIdx.z : slot) * args.input_stride / QK8_1;
  const int lane = threadIdx.x;
  const int row = threadIdx.y * RowsPerWarp;
  constexpr int blocks_per_iteration = VDR * WARP_SIZE / QI;
  float gate_sums[RowsPerWarp] = {};
  float up_sums[RowsPerWarp] = {};
  for (int block = lane / (QI / VDR); block < blocks_per_row;
       block += blocks_per_iteration) {
    const int part = VDR * (lane % (QI / VDR));
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      if (row + r >= valid_rows)
        continue;
      const size_t offset = size_t(row + r) * blocks_per_row;
      gate_sums[r] +=
          Dot(weight + offset + block, activation + block * (QK / QK8_1), part);
      if constexpr (Gate) {
        up_sums[r] += Dot(up_weight + offset + block,
                          activation + block * (QK / QK8_1), part);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    gate_sums[r] = warp_reduce_sum(gate_sums[r]);
    if constexpr (Gate)
      up_sums[r] = warp_reduce_sum(up_sums[r]);
    if (lane == 0 && row + r < valid_rows) {
      if constexpr (Gate) {
        const float g = gate_sums[r];
        const float activated =
            args.activation == 0
                ? 0.5f * g *
                      (1.0f +
                       tanhf(0.7978845608f * (g + 0.044715f * g * g * g)))
                : g / (1.0f + expf(-g));
        args.output[slot * args.rows + first_row + row + r] =
            up_sums[r] * activated;
      } else {
        atomicAdd(args.output + blockIdx.z * args.rows + first_row + row + r,
                  gate_sums[r] * args.routing[slot]);
      }
    }
  }
}

template <int QK, int QI, typename Weight, int VDR, vec_dot_q_cuda_t Dot,
          bool Gate, int RowsPerWarp, int Bytes>
bool launch_moe_staged_tile(MoeDecodeArgs args, cudaStream_t stream) {
  constexpr int tile_rows = MOE_WARPS * RowsPerWarp;
  const size_t panel = size_t(tile_rows) * (args.columns / QK) * sizeof(Weight);
  if (panel > MOE_STAGED_OPERAND_BYTES)
    return false;
  const size_t shared = Gate ? (panel + 15) / 16 * 16 + panel : panel;
  const dim3 grid((args.rows + tile_rows - 1) / tile_rows, args.topk);
  moe_decode_staged<QK, QI, Weight, VDR, Dot, Gate, RowsPerWarp, Bytes>
      <<<grid, dim3(WARP_SIZE, MOE_WARPS), shared, stream>>>(args);
  return true;
}

template <int QK, int QI, typename Weight, int VDR, vec_dot_q_cuda_t Dot,
          bool Gate>
bool launch_moe_staged_typed(MoeDecodeArgs args, cudaStream_t stream) {
  const size_t row_bytes = size_t(args.columns / QK) * sizeof(Weight);
  const size_t alignment = row_bytes |
                           reinterpret_cast<uintptr_t>(args.weights) |
                           reinterpret_cast<uintptr_t>(args.up);
  if ((alignment & 15) == 0)
    return launch_moe_staged_tile<QK, QI, Weight, VDR, Dot, Gate, 1, 16>(
        args, stream);
  if ((alignment & 7) == 0)
    return launch_moe_staged_tile<QK, QI, Weight, VDR, Dot, Gate, 1, 8>(args,
                                                                        stream);
  // Four-byte copies use more rows to amortize staging within the same footprint cap.
  if ((alignment & 3) == 0)
    return launch_moe_staged_tile<QK, QI, Weight, VDR, Dot, Gate, 2, 4>(args,
                                                                        stream);
  return false;
}

extern "C" bool launch_gguf_moe_staged(const MoeDecodeArgs *args, int format,
                                       bool gate, void *stream) {
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
#define STAGED_FORMAT(TAG, DTYPE, ID, QK, QI, WEIGHT, VDR, DOT)                \
  case ID:                                                                     \
    return gate ? launch_moe_staged_typed<QK, QI, WEIGHT, VDR, DOT, true>(     \
                      *args, cuda_stream)                                      \
                : launch_moe_staged_typed<QK, QI, WEIGHT, VDR, DOT, false>(    \
                      *args, cuda_stream);
  switch (format) {
    GGUF_VECTOR_FORMATS(STAGED_FORMAT)
  default:
    return false;
  }
#undef STAGED_FORMAT
}
