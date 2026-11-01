// GGUF matvec kernels with Q8_1-quantized activations.
// Adapted from llama.cpp's CUDA mmvq path for the GGUF types used here.

#include "../gemv/row.cuh"
#include "../gguf/vecdot.cuh"
#include "prefetch_row.cuh"

#define MMVQ_NWARPS_SINGLE_COL_PLAIN 8
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN 2
#define MMVQ_NWARPS_SINGLE_COL_FUSED_GLU 4
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU 2
#define MMVQ_NWARPS_SINGLE_COL_FUSED_QKV 8
#define MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV 2
#define MMVQ_FUSED_QKV_PACKED_MIN_ROW_RATIO 2

static __device__ __forceinline__ float warp_reduce_sum_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
  }
  return x;
}

static __device__ __forceinline__ float warp_reduce_max_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE));
  }
  return x;
}

enum GluActivation {
  GLU_SILU = 0,
  GLU_GELU = 1,
  GLU_RELU = 2,
  GLU_GELU_ERF = 3,
  GLU_SIGMOID = 4
};

static __device__ __forceinline__ float glu_silu(float x) {
  return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float glu_gelu(float x) {
  const float kSqrt2OverPi = 0.7978845608f;
  const float kCoeff = 0.044715f;
  const float x3 = x * x * x;
  const float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float glu_relu(float x) {
  return fmaxf(x, 0.0f);
}

static __device__ __forceinline__ float glu_gelu_erf(float x) {
  return x * normcdff(x);
}

static __device__ __forceinline__ float apply_glu_activation(float x, int act) {
  switch (act) {
  case GLU_SILU:
    return glu_silu(x);
  case GLU_GELU:
    return glu_gelu(x);
  case GLU_RELU:
    return glu_relu(x);
  case GLU_GELU_ERF:
    return glu_gelu_erf(x);
  case GLU_SIGMOID:
    return 1.0f / (1.0f + expf(-x));
  default:
    return glu_silu(x);
  }
}

static constexpr __device__ int mmvq_nwarps_plain_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_PLAIN;
  }
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int
mmvq_rows_per_cuda_block_plain_for(int ncols_dst) {
  return (ncols_dst == 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN : 2;
}

static constexpr __device__ int mmvq_nwarps_fused_glu_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_FUSED_GLU;
  }
  return 2;
}

static constexpr __device__ int
mmvq_rows_per_cuda_block_fused_glu_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU;
  }
  return (ncols_dst <= 4) ? 1 : 2;
}

static constexpr __device__ int mmvq_nwarps_fused_qkv_for(int ncols_dst) {
  if (ncols_dst == 1) {
    return MMVQ_NWARPS_SINGLE_COL_FUSED_QKV;
  }
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int
mmvq_rows_per_cuda_block_fused_qkv_for(int ncols_dst) {
  return (ncols_dst == 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV : 2;
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void
mmvq_core_impl(const void *__restrict__ vx, const block_q8_1 *__restrict__ y,
               dst_t *__restrict__ dst, const int ncols_x, const int nrows_x,
               const int stride_col_y, const int stride_col_dst,
               const int row0) {

  constexpr int nwarps = mmvq_nwarps_plain_for(ncols_dst);
  constexpr int rows_per_cuda_block =
      mmvq_rows_per_cuda_block_plain_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        const int weight_kbx = row * blocks_per_row_x + kbx;
        tmp[j][i] +=
            vec_dot_q_cuda(static_cast<const block_q_t *>(vx) + weight_kbx,
                           &y[j * stride_col_y + kby], kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum_f32(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      dst[j * stride_col_dst + row0 + threadIdx.x] = (dst_t)tmp[j][threadIdx.x];
    }
  }
}

struct MixedMmvqArgs {
  const void *quant_weight;
  const void *quant_input;
  const void *dense_weight;
  const void *input;
  void *quant_output;
  void *dense_output;
  int columns;
  int quant_rows;
  int dense_rows;
  int prefetch;
};

template <typename T, typename Vec2, int QK, int QI, typename Weight, int VDR,
          vec_dot_q_cuda_t Dot, bool Prefetch>
__global__ __launch_bounds__(MMVQ_NWARPS_SINGLE_COL_PLAIN *WARP_SIZE,
                             1) void mmvq_dense_pair(MixedMmvqArgs args) {
  constexpr int threads = MMVQ_NWARPS_SINGLE_COL_PLAIN * WARP_SIZE;
  if (blockIdx.x < args.dense_rows) {
    const mistralrs_gemv::RowArgs<T> dense{
        static_cast<const T *>(args.dense_weight),
        static_cast<const T *>(args.input),
        nullptr,
        static_cast<T *>(args.dense_output),
        args.dense_rows,
        args.columns,
        false};
    mistralrs_gemv::row<T, Vec2, threads, 1>(
        dense, blockIdx.x, threadIdx.y * WARP_SIZE + threadIdx.x);
  } else if constexpr (Prefetch) {
    const int row = blockIdx.x - args.dense_rows;
    prefetch_row<T, QK, QI, Weight, VDR, Dot>(
        static_cast<const Weight *>(args.quant_weight) +
            (size_t)row * (args.columns / QK),
        static_cast<const block_q8_1 *>(args.quant_input),
        static_cast<T *>(args.quant_output) + row, args.columns);
  } else {
    mmvq_core_impl<T, QK, QI, Weight, VDR, Dot, 1>(
        args.quant_weight, static_cast<const block_q8_1 *>(args.quant_input),
        static_cast<T *>(args.quant_output), args.columns, args.quant_rows,
        args.columns / QK8_1, args.quant_rows,
        MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN * (blockIdx.x - args.dense_rows));
  }
}

template <typename T, typename Vec2>
void launch_dense_pair(MixedMmvqArgs args, int format, cudaStream_t stream) {
  const int quant_blocks =
      args.prefetch ? args.quant_rows
                    : args.quant_rows / MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN;
  const int blocks = quant_blocks + args.dense_rows;
#define MIXED_FORMAT(tag, rust_tag, code, qk, qi, weight, vdr, dot)            \
  case code:                                                                   \
    if (args.prefetch) {                                                       \
      mmvq_dense_pair<T, Vec2, qk, qi, weight, vdr, dot, true>                 \
          <<<blocks, dim3(WARP_SIZE, MMVQ_NWARPS_SINGLE_COL_PLAIN), 0,         \
             stream>>>(args);                                                  \
    } else {                                                                   \
      mmvq_dense_pair<T, Vec2, qk, qi, weight, vdr, dot, false>                \
          <<<blocks, dim3(WARP_SIZE, MMVQ_NWARPS_SINGLE_COL_PLAIN), 0,         \
             stream>>>(args);                                                  \
    }                                                                          \
    break;
  switch (format) { GGUF_DENSE_VECTOR_FORMATS(MIXED_FORMAT) }
#undef MIXED_FORMAT
}

extern "C" void launch_mmvq_gguf_dense_pair(MixedMmvqArgs args, int format,
                                            int dtype, void *stream) {
  const auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (dtype == 0) {
    launch_dense_pair<half, half2>(args, format, cuda_stream);
  } else if (dtype == 1) {
    launch_dense_pair<__nv_bfloat16, __nv_bfloat162>(args, format, cuda_stream);
  } else {
    launch_dense_pair<float, float2>(args, format, cuda_stream);
  }
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void mmvq_core_fused_glu_impl(
    const void *__restrict__ vx_gate, const void *__restrict__ vx_up,
    const block_q8_1 *__restrict__ y, dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x, const int stride_col_y,
    const int stride_col_dst, const int activation) {

  constexpr int nwarps = mmvq_nwarps_fused_glu_for(ncols_dst);
  constexpr int rows_per_cuda_block =
      mmvq_rows_per_cuda_block_fused_glu_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};
  float tmp_up[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        const int weight_kbx = row * blocks_per_row_x + kbx;
        const block_q8_1 *yj = &y[j * stride_col_y + kby];
        tmp_gate[j][i] += vec_dot_q_cuda(
            static_cast<const block_q_t *>(vx_gate) + weight_kbx, yj, kqs);
        tmp_up[j][i] += vec_dot_q_cuda(
            static_cast<const block_q_t *>(vx_up) + weight_kbx, yj, kqs);
      }
    }
  }

  __shared__ float tmp_shared_gate[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                                  [rows_per_cuda_block][WARP_SIZE];
  __shared__ float tmp_shared_up[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                                [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared_gate[threadIdx.y - 1][j][i][threadIdx.x] = tmp_gate[j][i];
        tmp_shared_up[threadIdx.y - 1][j][i][threadIdx.x] = tmp_up[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
        tmp_up[j][i] += tmp_shared_up[l][j][i][threadIdx.x];
      }
      tmp_gate[j][i] = warp_reduce_sum_f32(tmp_gate[j][i]);
      tmp_up[j][i] = warp_reduce_sum_f32(tmp_up[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      const dst_t gate_val = (dst_t)tmp_gate[j][threadIdx.x];
      const dst_t up_val = (dst_t)tmp_up[j][threadIdx.x];
      const dst_t activated =
          (dst_t)apply_glu_activation((float)gate_val, activation);
      dst[j * stride_col_dst + row0 + threadIdx.x] = activated * up_val;
    }
  }
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void mmvq_core_fused_qkv_impl(
    const void *__restrict__ vx_q, const void *__restrict__ vx_k,
    const void *__restrict__ vx_v, const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ q_dst, dst_t *__restrict__ k_dst,
    dst_t *__restrict__ v_dst, const int ncols_x, const int nrows_q,
    const int nrows_k, const int nrows_v, const int stride_col_y) {

  constexpr int nwarps = mmvq_nwarps_fused_qkv_for(ncols_dst);
  constexpr int rows_per_cuda_block =
      mmvq_rows_per_cuda_block_fused_qkv_for(ncols_dst);

  const void *vx;
  dst_t *dst;
  int nrows_x;
  int row_block;
  if (gridDim.y == 1) {
    const int q_blocks =
        (nrows_q + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const int k_blocks =
        (nrows_k + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const int block = blockIdx.x;
    if (block < q_blocks) {
      row_block = block;
      vx = vx_q;
      dst = q_dst;
      nrows_x = nrows_q;
    } else if (block < q_blocks + k_blocks) {
      row_block = block - q_blocks;
      vx = vx_k;
      dst = k_dst;
      nrows_x = nrows_k;
    } else {
      row_block = block - q_blocks - k_blocks;
      vx = vx_v;
      dst = v_dst;
      nrows_x = nrows_v;
    }
  } else {
    row_block = blockIdx.x;
    if (blockIdx.y == 0) {
      vx = vx_q;
      dst = q_dst;
      nrows_x = nrows_q;
    } else if (blockIdx.y == 1) {
      vx = vx_k;
      dst = k_dst;
      nrows_x = nrows_k;
    } else {
      vx = vx_v;
      dst = v_dst;
      nrows_x = nrows_v;
    }
  }

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * row_block;
  if (row0 >= nrows_x) {
    return;
  }

  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        if (row >= nrows_x) {
          continue;
        }
        const int weight_kbx = row * blocks_per_row_x + kbx;
        tmp[j][i] +=
            vec_dot_q_cuda(static_cast<const block_q_t *>(vx) + weight_kbx,
                           &y[j * stride_col_y + kby], kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum_f32(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      dst[j * nrows_x + row0 + threadIdx.x] = (dst_t)tmp[j][threadIdx.x];
    }
  }
}

#define MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,     \
                         dst_tag, dst_c_type, ncols)                           \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_plain_for(ncols) * WARP_SIZE, 1)                             \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda##ncols(                         \
          const void *__restrict__ vx, const void *__restrict__ vy,            \
          dst_c_type *__restrict__ dst, const int ncols_x, const int nrows_x,  \
          const int stride_col_y, const int stride_col_dst) {                  \
    mmvq_core_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val, vec_dot,    \
                   ncols>(vx, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,   \
                          stride_col_y, stride_col_dst,                        \
                          mmvq_rows_per_cuda_block_plain_for(ncols) *          \
                              blockIdx.x);                                     \
  }

#define MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, \
                             dst_tag, dst_c_type, ncols)                       \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_fused_glu_for(ncols) * WARP_SIZE, 1)                         \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda##ncols(                     \
          const void *__restrict__ vx_gate, const void *__restrict__ vx_up,    \
          const void *__restrict__ vy, dst_c_type *__restrict__ dst,           \
          const int ncols_x, const int nrows_x, const int stride_col_y,        \
          const int stride_col_dst, const int activation) {                    \
    mmvq_core_fused_glu_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val,   \
                             vec_dot, ncols>(                                  \
        vx_gate, vx_up, (const block_q8_1 *)vy, dst, ncols_x, nrows_x,         \
        stride_col_y, stride_col_dst, activation);                             \
  }

#define MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, \
                             dst_tag, dst_c_type, ncols)                       \
  extern "C" __global__ void __launch_bounds__(                                \
      mmvq_nwarps_fused_qkv_for(ncols) * WARP_SIZE, 1)                         \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda##ncols(                     \
          const void *__restrict__ vx_q, const void *__restrict__ vx_k,        \
          const void *__restrict__ vx_v, const void *__restrict__ vy,          \
          dst_c_type *__restrict__ q_dst, dst_c_type *__restrict__ k_dst,      \
          dst_c_type *__restrict__ v_dst, const int ncols_x,                   \
          const int nrows_q, const int nrows_k, const int nrows_v,             \
          const int stride_col_y) {                                            \
    mmvq_core_fused_qkv_impl<dst_c_type, qk_val, qi_val, block_q_t, vdr_val,   \
                             vec_dot, ncols>(                                  \
        vx_q, vx_k, vx_v, (const block_q8_1 *)vy, q_dst, k_dst, v_dst,         \
        ncols_x, nrows_q, nrows_k, nrows_v, stride_col_y);                     \
  }

#define MMVQ_PLAIN_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot) \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 1)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 2)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 3)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 4)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 5)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 6)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 7)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, bf16,     \
                   __nv_bfloat16, 8)                                           \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 1)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 2)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 3)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 4)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 5)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 6)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 7)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f16,      \
                   half, 8)                                                    \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 1)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 2)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 3)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 4)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 5)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 6)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 7)                                                   \
  MMVQ_PLAIN_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot, f32,      \
                   float, 8)

#define MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val,      \
                                 vec_dot, dst_tag, dst_c_type)                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 1)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 2)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 3)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 4)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 5)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 6)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 7)                                 \
  MMVQ_FUSED_QKV_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 8)

#define MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val,      \
                                 vec_dot, dst_tag, dst_c_type)                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 1)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 2)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 3)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 4)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 5)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 6)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 7)                                 \
  MMVQ_FUSED_GLU_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,       \
                       dst_tag, dst_c_type, 8)

MMVQ_PLAIN_BATCH_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                     vec_dot_q5_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1)
MMVQ_PLAIN_BATCH_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
MMVQ_PLAIN_BATCH_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                     vec_dot_q3_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                     vec_dot_q5_K_q8_1)
MMVQ_PLAIN_BATCH_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)

#define MMVQ_FUSED_GLU_TYPE_SET(tag, block_q_t, qk_val, qi_val, vdr_val,       \
                                vec_dot)                                       \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           bf16, __nv_bfloat16)                                \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f16, half)                                          \
  MMVQ_FUSED_GLU_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f32, float)

MMVQ_FUSED_GLU_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                        vec_dot_q5_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                        vec_dot_q3_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                        vec_dot_q5_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)

#define MMVQ_FUSED_QKV_TYPE_SET(tag, block_q_t, qk_val, qi_val, vdr_val,       \
                                vec_dot)                                       \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           bf16, __nv_bfloat16)                                \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f16, half)                                          \
  MMVQ_FUSED_QKV_BATCH_SET(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot,   \
                           f32, float)

MMVQ_FUSED_QKV_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ,
                        vec_dot_q5_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ,
                        vec_dot_q3_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ,
                        vec_dot_q5_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_bf16(const __nv_bfloat16 *__restrict__ x,
                             void *__restrict__ vy, const int kx,
                             const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __bfloat162float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f16(const half *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f32(const float *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

#define MMVQ_LAUNCHER_PLAIN(tag, dst_tag, dst_c_type)                          \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_plain(                  \
      const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,     \
      int stride_col_y, int stride_col_dst, int b_size, void *stream) {        \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_PLAIN : 2;              \
    const unsigned int nblocks =                                               \
        (unsigned int)((nrows_x + rows_per_block - 1) / rows_per_block);       \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_PLAIN;                                   \
    } else if (b_size <= 4) {                                                  \
      nwarps = 4;                                                              \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, 1, 1);                                                  \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda1<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda2<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda3<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda4<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda5<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda6<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda7<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_plain_cuda8<<<grid, block, 0, s>>>(        \
          vx, vy, (dst_c_type *)dst, ncols_x, nrows_x, stride_col_y,           \
          stride_col_dst);                                                     \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

#define MMVQ_LAUNCHER_FUSED_GLU(tag, dst_tag, dst_c_type)                      \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_fused_glu(              \
      const void *vx_gate, const void *vx_up, const void *vy, void *dst,       \
      int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst,          \
      int b_size, int activation, void *stream) {                              \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_GLU               \
                      : ((b_size <= 4) ? 1 : 2);                               \
    const unsigned int nblocks =                                               \
        (unsigned int)((nrows_x + rows_per_block - 1) / rows_per_block);       \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_FUSED_GLU;                               \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, 1, 1);                                                  \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda1<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda2<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda3<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda4<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda5<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda6<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda7<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_glu_cuda8<<<grid, block, 0, s>>>(    \
          vx_gate, vx_up, vy, (dst_c_type *)dst, ncols_x, nrows_x,             \
          stride_col_y, stride_col_dst, activation);                           \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

#define MMVQ_LAUNCHER_FUSED_QKV(tag, dst_tag, dst_c_type)                      \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_fused_qkv(              \
      const void *vx_q, const void *vx_k, const void *vx_v, const void *vy,    \
      void *q_dst, void *k_dst, void *v_dst, int ncols_x, int nrows_q,         \
      int nrows_k, int nrows_v, int stride_col_y, int b_size, void *stream) {  \
    const unsigned int rows_per_block =                                        \
        (b_size <= 1) ? MMVQ_ROWS_PER_BLOCK_SINGLE_COL_FUSED_QKV : 2;          \
    int max_nrows = nrows_q > nrows_k ? nrows_q : nrows_k;                     \
    max_nrows = max_nrows > nrows_v ? max_nrows : nrows_v;                     \
    int min_nrows = nrows_q < nrows_k ? nrows_q : nrows_k;                     \
    min_nrows = min_nrows < nrows_v ? min_nrows : nrows_v;                     \
    const bool use_packed =                                                    \
        max_nrows >= MMVQ_FUSED_QKV_PACKED_MIN_ROW_RATIO * min_nrows;          \
    const unsigned int q_blocks =                                              \
        (unsigned int)((nrows_q + rows_per_block - 1) / rows_per_block);       \
    const unsigned int k_blocks =                                              \
        (unsigned int)((nrows_k + rows_per_block - 1) / rows_per_block);       \
    const unsigned int v_blocks =                                              \
        (unsigned int)((nrows_v + rows_per_block - 1) / rows_per_block);       \
    const unsigned int nblocks =                                               \
        use_packed ? q_blocks + k_blocks + v_blocks                            \
                   : (unsigned int)((max_nrows + rows_per_block - 1) /         \
                                    rows_per_block);                           \
    unsigned int nwarps;                                                       \
    if (b_size == 1) {                                                         \
      nwarps = MMVQ_NWARPS_SINGLE_COL_FUSED_QKV;                               \
    } else if (b_size <= 4) {                                                  \
      nwarps = 4;                                                              \
    } else {                                                                   \
      nwarps = 2;                                                              \
    }                                                                          \
    dim3 grid(nblocks, use_packed ? 1 : 3, 1);                                 \
    dim3 block(WARP_SIZE, nwarps, 1);                                          \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda1<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 2:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda2<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 3:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda3<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 4:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda4<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 5:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda5<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 6:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda6<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 7:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda7<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    case 8:                                                                    \
      mmvq_gguf_##tag##_##dst_tag##_fused_qkv_cuda8<<<grid, block, 0, s>>>(    \
          vx_q, vx_k, vx_v, vy, (dst_c_type *)q_dst, (dst_c_type *)k_dst,      \
          (dst_c_type *)v_dst, ncols_x, nrows_q, nrows_k, nrows_v,             \
          stride_col_y);                                                       \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

MMVQ_LAUNCHER_PLAIN(q4_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q8_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q2_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q3_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q6_k, bf16, __nv_bfloat16)

MMVQ_LAUNCHER_PLAIN(q4_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q8_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q2_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q3_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q6_k, f16, half)

MMVQ_LAUNCHER_PLAIN(q4_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q4_1, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_1, f32, float)
MMVQ_LAUNCHER_PLAIN(q8_0, f32, float)
MMVQ_LAUNCHER_PLAIN(q2_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q3_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q4_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q5_k, f32, float)
MMVQ_LAUNCHER_PLAIN(q6_k, f32, float)

#define MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(tag)                                  \
  MMVQ_LAUNCHER_FUSED_GLU(tag, bf16, __nv_bfloat16)                            \
  MMVQ_LAUNCHER_FUSED_GLU(tag, f16, half)                                      \
  MMVQ_LAUNCHER_FUSED_GLU(tag, f32, float)

MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_1)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_1)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q2_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q3_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_k)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q6_k)

#define MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(tag)                                  \
  MMVQ_LAUNCHER_FUSED_QKV(tag, bf16, __nv_bfloat16)                            \
  MMVQ_LAUNCHER_FUSED_QKV(tag, f16, half)                                      \
  MMVQ_LAUNCHER_FUSED_QKV(tag, f32, float)

MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q2_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q3_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q6_k)

extern "C" void launch_mmvq_gguf_quantize_q8_1_bf16(const void *x, void *vy,
                                                    int kx, int kx_padded,
                                                    int num_rows,
                                                    void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_bf16<<<grid, block, 0, s>>>((const __nv_bfloat16 *)x,
                                                      vy, kx, kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f16(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f16<<<grid, block, 0, s>>>((const half *)x, vy, kx,
                                                     kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f32(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f32<<<grid, block, 0, s>>>((const float *)x, vy, kx,
                                                     kx_padded);
}
