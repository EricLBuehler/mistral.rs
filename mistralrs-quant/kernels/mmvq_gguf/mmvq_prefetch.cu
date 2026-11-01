#include "prefetch_row.cuh"

struct PrefetchProjections {
  const void *q;
  const void *k;
  const void *v;
  const block_q8_1 *input;
  void *q_out;
  void *k_out;
  void *v_out;
  int q_rows;
  int k_rows;
  int columns;
};

template <typename T, int qk, int qi, typename Weight, int vdr,
          vec_dot_q_cuda_t dot>
__global__ __launch_bounds__(PREFETCH_WARPS *WARP_SIZE, 1) void mmvq_prefetch(
    const Weight *__restrict__ weights, const block_q8_1 *__restrict__ input,
    T *__restrict__ output, int columns) {
  prefetch_row<T, qk, qi, Weight, vdr, dot>(
      weights + (size_t)blockIdx.x * (columns / qk), input, output + blockIdx.x,
      columns);
}

template <typename T, int qk, int qi, typename Weight, int vdr,
          vec_dot_q_cuda_t dot>
__global__
__launch_bounds__(PREFETCH_WARPS *WARP_SIZE,
                  1) void mmvq_prefetch_projected(PrefetchProjections args) {
  int row = blockIdx.x;
  const void *weight = args.q;
  void *output = args.q_out;
  if (row >= args.q_rows) {
    row -= args.q_rows;
    weight = args.k;
    output = args.k_out;
    if (row >= args.k_rows) {
      row -= args.k_rows;
      weight = args.v;
      output = args.v_out;
    }
  }
  prefetch_row<T, qk, qi, Weight, vdr, dot>(
      static_cast<const Weight *>(weight) + (size_t)row * (args.columns / qk),
      args.input, static_cast<T *>(output) + row, args.columns);
}

#define PREFETCH_LAUNCHER(TAG, SUFFIX, T, QK, QI, WEIGHT, VDR, DOT)            \
  extern "C" void launch_mmvq_gguf_##TAG##_##SUFFIX##_prefetch(                \
      const void *weight, const void *input, void *output, int columns,        \
      int rows, int, int, int, void *stream) {                                 \
    mmvq_prefetch<T, QK, QI, WEIGHT, VDR, DOT>                                 \
        <<<rows, dim3(WARP_SIZE, PREFETCH_WARPS), 0,                           \
           static_cast<cudaStream_t>(stream)>>>(                               \
            static_cast<const WEIGHT *>(weight),                               \
            static_cast<const block_q8_1 *>(input), static_cast<T *>(output),  \
            columns);                                                          \
  }                                                                            \
  extern "C" void launch_mmvq_gguf_##TAG##_##SUFFIX##_qkv_prefetch(            \
      const void *q, const void *k, const void *v, const void *input,          \
      void *q_out, void *k_out, void *v_out, int columns, int q_rows,          \
      int k_rows, int v_rows, int, int, void *stream) {                        \
    const PrefetchProjections args{                                            \
        q,      k,      v,     static_cast<const block_q8_1 *>(input),         \
        q_out,  k_out,  v_out, q_rows,                                         \
        k_rows, columns};                                                      \
    mmvq_prefetch_projected<T, QK, QI, WEIGHT, VDR, DOT>                       \
        <<<q_rows + k_rows + v_rows, dim3(WARP_SIZE, PREFETCH_WARPS), 0,       \
           static_cast<cudaStream_t>(stream)>>>(args);                         \
  }

#define PREFETCH_FORMAT(TAG, DTYPE, ID, QK, QI, WEIGHT, VDR, DOT)              \
  PREFETCH_LAUNCHER(TAG, bf16, __nv_bfloat16, QK, QI, WEIGHT, VDR, DOT)        \
  PREFETCH_LAUNCHER(TAG, f16, half, QK, QI, WEIGHT, VDR, DOT)                  \
  PREFETCH_LAUNCHER(TAG, f32, float, QK, QI, WEIGHT, VDR, DOT)

GGUF_DENSE_PREFETCH_FORMATS(PREFETCH_FORMAT)
