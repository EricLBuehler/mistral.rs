/*
 * AFQ (Affine Fast Quantization) CUDA Kernels
 *
 * Implements dequantization and quantization operations for AFQ format.
 * Supports 2, 3, 4, 6, 8-bit quantization with group sizes 32, 64, 128.
 */

#include "afq_utils.cuh"
#include <cfloat>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// Dequantization Kernels
// ============================================================================

// Generic dequantization kernel for power-of-2 bit widths (2, 4, 8)
// Weight layout: [rows, cols * bits / 32] as u32
// Output layout: [rows, cols]
template <typename T, int bits, int group_size>
__global__ void afq_dequantize_kernel(const uint32_t *__restrict__ w_q,
                                      const T *__restrict__ scales,
                                      const T *__restrict__ biases,
                                      T *__restrict__ output, int rows,
                                      int cols) {
  constexpr int values_per_u32 = 32 / bits;
  const int packed_cols = cols * bits / 32;
  const int groups_per_row = cols / group_size;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = rows * cols;

  if (tid >= total_elements)
    return;

  int row = tid / cols;
  int col = tid % cols;

  // Calculate packed position
  int packed_col = col / values_per_u32;
  int value_idx = col % values_per_u32;

  // Load packed value
  uint32_t packed = w_q[row * packed_cols + packed_col];

  // Extract quantized value
  uint32_t q = extract_bits<bits>(packed, value_idx);

  // Get scale and bias for this group
  int group_idx = col / group_size;
  T scale = scales[row * groups_per_row + group_idx];
  T bias = biases[row * groups_per_row + group_idx];

  // Dequantize: w = q * scale + bias
  output[tid] = dequant_value<T>(q, scale, bias);
}

// Specialized dequantization kernel for 3-bit (non-power-of-2)
// 8 values packed into 3 bytes (24 bits)
template <typename T, int group_size>
__global__ void afq_dequantize_3bit_kernel(const uint8_t *__restrict__ w_q,
                                           const T *__restrict__ scales,
                                           const T *__restrict__ biases,
                                           T *__restrict__ output, int rows,
                                           int cols) {
  const int groups_per_row = cols / group_size;
  // 8 values per 3 bytes
  const int packed_cols = (cols * 3 + 7) / 8;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = rows * cols;

  if (tid >= total_elements)
    return;

  int row = tid / cols;
  int col = tid % cols;

  // Calculate byte position for 3-bit extraction
  const uint8_t *row_data = w_q + row * packed_cols;
  uint32_t q = extract_3bit(row_data, col);

  // Get scale and bias
  int group_idx = col / group_size;
  T scale = scales[row * groups_per_row + group_idx];
  T bias = biases[row * groups_per_row + group_idx];

  output[tid] = dequant_value<T>(q, scale, bias);
}

// Specialized dequantization kernel for 6-bit (non-power-of-2)
// 4 values packed into 3 bytes (24 bits)
template <typename T, int group_size>
__global__ void afq_dequantize_6bit_kernel(const uint8_t *__restrict__ w_q,
                                           const T *__restrict__ scales,
                                           const T *__restrict__ biases,
                                           T *__restrict__ output, int rows,
                                           int cols) {
  const int groups_per_row = cols / group_size;
  // 4 values per 3 bytes
  const int packed_cols = (cols * 6 + 7) / 8;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = rows * cols;

  if (tid >= total_elements)
    return;

  int row = tid / cols;
  int col = tid % cols;

  // Calculate byte position for 6-bit extraction
  const uint8_t *row_data = w_q + row * packed_cols;
  uint32_t q = extract_6bit(row_data, col);

  // Get scale and bias
  int group_idx = col / group_size;
  T scale = scales[row * groups_per_row + group_idx];
  T bias = biases[row * groups_per_row + group_idx];

  output[tid] = dequant_value<T>(q, scale, bias);
}

// ============================================================================
// Quantization Kernels
// ============================================================================

// Compute scale and bias for a group using warp reduction
template <typename T>
__device__ void compute_scale_bias_warp(const T *w, int group_start, int cols,
                                        int group_size, float &scale,
                                        float &bias) {
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  // Each lane processes multiple elements if group_size > warp_size
  for (int i = lane; i < group_size; i += AFQ_WARP_SIZE) {
    int col = group_start + i;
    if (col < cols) {
      float val;
      if constexpr (std::is_same_v<T, float>) {
        val = w[col];
      } else if constexpr (std::is_same_v<T, __half>) {
        val = __half2float(w[col]);
      }
#if __CUDA_ARCH__ >= 800
      else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        val = __bfloat162float(w[col]);
      }
#endif
      local_min = fminf(local_min, val);
      local_max = fmaxf(local_max, val);
    }
  }

  // Warp reduce to find global min/max
  float group_min = warp_reduce_min(local_min);
  float group_max = warp_reduce_max(local_max);

  // Broadcast from lane 0
  group_min = __shfl_sync(0xffffffff, group_min, 0);
  group_max = __shfl_sync(0xffffffff, group_max, 0);

  // Compute scale and bias
  float range = group_max - group_min;
  scale = fmaxf(range, 1e-7f);
  bias = group_min;
}

// Generic quantization kernel for power-of-2 bit widths
template <typename T, int bits, int group_size>
__global__ void
afq_quantize_kernel(const T *__restrict__ w, uint32_t *__restrict__ w_q,
                    T *__restrict__ scales, T *__restrict__ biases, int rows,
                    int cols) {
  constexpr int values_per_u32 = 32 / bits;
  constexpr uint32_t max_q = (1u << bits) - 1;
  const int packed_cols = cols * bits / 32;
  const int groups_per_row = cols / group_size;

  // Each warp processes one group
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / AFQ_WARP_SIZE;
  int lane = threadIdx.x % AFQ_WARP_SIZE;

  int total_groups = rows * groups_per_row;
  if (warp_id >= total_groups)
    return;

  int row = warp_id / groups_per_row;
  int group_idx = warp_id % groups_per_row;
  int group_start = group_idx * group_size;

  const T *row_w = w + row * cols;

  // Compute scale and bias
  float scale, bias;
  compute_scale_bias_warp(row_w, group_start, cols, group_size, scale, bias);

  // Write scale and bias (only lane 0)
  if (lane == 0) {
    float n_bins = (float)max_q;
    float final_scale = scale / n_bins;

    if constexpr (std::is_same_v<T, float>) {
      scales[row * groups_per_row + group_idx] = final_scale;
      biases[row * groups_per_row + group_idx] = bias;
    } else if constexpr (std::is_same_v<T, __half>) {
      scales[row * groups_per_row + group_idx] = __float2half(final_scale);
      biases[row * groups_per_row + group_idx] = __float2half(bias);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scales[row * groups_per_row + group_idx] = __float2bfloat16(final_scale);
      biases[row * groups_per_row + group_idx] = __float2bfloat16(bias);
    }
#endif
  }

  // Quantize and pack values
  float inv_scale = (float)max_q / scale;

  for (int i = lane; i < group_size; i += AFQ_WARP_SIZE) {
    int col = group_start + i;
    if (col >= cols)
      continue;

    float val;
    if constexpr (std::is_same_v<T, float>) {
      val = row_w[col];
    } else if constexpr (std::is_same_v<T, __half>) {
      val = __half2float(row_w[col]);
    }
#if __CUDA_ARCH__ >= 800
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      val = __bfloat162float(row_w[col]);
    }
#endif

    // Quantize
    float q_float = roundf((val - bias) * inv_scale);
    uint32_t q = min(max((int)q_float, 0), (int)max_q);

    // Pack into u32 using atomic OR
    int packed_col = col / values_per_u32;
    int shift = (col % values_per_u32) * bits;
    atomicOr(&w_q[row * packed_cols + packed_col], q << shift);
  }
}

// ============================================================================
// Extern "C" Launch Functions - Dequantize
// ============================================================================

#define DEFINE_DEQUANT_LAUNCHER(bits, gs, dtype, dtype_name)                   \
  extern "C" void afq_dequantize_##bits##bit_gs##gs##_##dtype_name(            \
      const uint32_t *w_q, const dtype *scales, const dtype *biases,           \
      dtype *output, int rows, int cols) {                                     \
    int total = rows * cols;                                                   \
    int blocks = cdiv(total, AFQ_BLOCK_SIZE);                                  \
    afq_dequantize_kernel<dtype, bits, gs>                                     \
        <<<blocks, AFQ_BLOCK_SIZE>>>(w_q, scales, biases, output, rows, cols); \
  }

#define DEFINE_DEQUANT_3BIT_LAUNCHER(gs, dtype, dtype_name)                    \
  extern "C" void afq_dequantize_3bit_gs##gs##_##dtype_name(                   \
      const uint8_t *w_q, const dtype *scales, const dtype *biases,            \
      dtype *output, int rows, int cols) {                                     \
    int total = rows * cols;                                                   \
    int blocks = cdiv(total, AFQ_BLOCK_SIZE);                                  \
    afq_dequantize_3bit_kernel<dtype, gs>                                      \
        <<<blocks, AFQ_BLOCK_SIZE>>>(w_q, scales, biases, output, rows, cols); \
  }

#define DEFINE_DEQUANT_6BIT_LAUNCHER(gs, dtype, dtype_name)                    \
  extern "C" void afq_dequantize_6bit_gs##gs##_##dtype_name(                   \
      const uint8_t *w_q, const dtype *scales, const dtype *biases,            \
      dtype *output, int rows, int cols) {                                     \
    int total = rows * cols;                                                   \
    int blocks = cdiv(total, AFQ_BLOCK_SIZE);                                  \
    afq_dequantize_6bit_kernel<dtype, gs>                                      \
        <<<blocks, AFQ_BLOCK_SIZE>>>(w_q, scales, biases, output, rows, cols); \
  }

// 2-bit dequantize launchers
DEFINE_DEQUANT_LAUNCHER(2, 32, float, f32)
DEFINE_DEQUANT_LAUNCHER(2, 64, float, f32)
DEFINE_DEQUANT_LAUNCHER(2, 128, float, f32)
DEFINE_DEQUANT_LAUNCHER(2, 32, __half, f16)
DEFINE_DEQUANT_LAUNCHER(2, 64, __half, f16)
DEFINE_DEQUANT_LAUNCHER(2, 128, __half, f16)

// 3-bit dequantize launchers
DEFINE_DEQUANT_3BIT_LAUNCHER(32, float, f32)
DEFINE_DEQUANT_3BIT_LAUNCHER(64, float, f32)
DEFINE_DEQUANT_3BIT_LAUNCHER(128, float, f32)
DEFINE_DEQUANT_3BIT_LAUNCHER(32, __half, f16)
DEFINE_DEQUANT_3BIT_LAUNCHER(64, __half, f16)
DEFINE_DEQUANT_3BIT_LAUNCHER(128, __half, f16)

// 4-bit dequantize launchers
DEFINE_DEQUANT_LAUNCHER(4, 32, float, f32)
DEFINE_DEQUANT_LAUNCHER(4, 64, float, f32)
DEFINE_DEQUANT_LAUNCHER(4, 128, float, f32)
DEFINE_DEQUANT_LAUNCHER(4, 32, __half, f16)
DEFINE_DEQUANT_LAUNCHER(4, 64, __half, f16)
DEFINE_DEQUANT_LAUNCHER(4, 128, __half, f16)

// 6-bit dequantize launchers
DEFINE_DEQUANT_6BIT_LAUNCHER(32, float, f32)
DEFINE_DEQUANT_6BIT_LAUNCHER(64, float, f32)
DEFINE_DEQUANT_6BIT_LAUNCHER(128, float, f32)
DEFINE_DEQUANT_6BIT_LAUNCHER(32, __half, f16)
DEFINE_DEQUANT_6BIT_LAUNCHER(64, __half, f16)
DEFINE_DEQUANT_6BIT_LAUNCHER(128, __half, f16)

// 8-bit dequantize launchers
DEFINE_DEQUANT_LAUNCHER(8, 32, float, f32)
DEFINE_DEQUANT_LAUNCHER(8, 64, float, f32)
DEFINE_DEQUANT_LAUNCHER(8, 128, float, f32)
DEFINE_DEQUANT_LAUNCHER(8, 32, __half, f16)
DEFINE_DEQUANT_LAUNCHER(8, 64, __half, f16)
DEFINE_DEQUANT_LAUNCHER(8, 128, __half, f16)

// BFloat16 versions
DEFINE_DEQUANT_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(2, 128, __nv_bfloat16, bf16)
DEFINE_DEQUANT_3BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_DEQUANT_3BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_DEQUANT_3BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_DEQUANT_6BIT_LAUNCHER(32, __nv_bfloat16, bf16)
DEFINE_DEQUANT_6BIT_LAUNCHER(64, __nv_bfloat16, bf16)
DEFINE_DEQUANT_6BIT_LAUNCHER(128, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_DEQUANT_LAUNCHER(8, 128, __nv_bfloat16, bf16)

// ============================================================================
// Extern "C" Launch Functions - Quantize
// ============================================================================

#define DEFINE_QUANT_LAUNCHER(bits, gs, dtype, dtype_name)                     \
  extern "C" void afq_quantize_##bits##bit_gs##gs##_##dtype_name(              \
      const dtype *w, uint32_t *w_q, dtype *scales, dtype *biases, int rows,   \
      int cols) {                                                              \
    int groups_per_row = cols / gs;                                            \
    int total_groups = rows * groups_per_row;                                  \
    int warps_needed = total_groups;                                           \
    int threads = warps_needed * AFQ_WARP_SIZE;                                \
    int blocks = cdiv(threads, AFQ_BLOCK_SIZE);                                \
    /* Zero out w_q first */                                                   \
    int packed_cols = cols * bits / 32;                                        \
    cudaMemset(w_q, 0, rows * packed_cols * sizeof(uint32_t));                 \
    afq_quantize_kernel<dtype, bits, gs>                                       \
        <<<blocks, AFQ_BLOCK_SIZE>>>(w, w_q, scales, biases, rows, cols);      \
  }

// 2-bit quantize launchers
DEFINE_QUANT_LAUNCHER(2, 32, float, f32)
DEFINE_QUANT_LAUNCHER(2, 64, float, f32)
DEFINE_QUANT_LAUNCHER(2, 128, float, f32)
DEFINE_QUANT_LAUNCHER(2, 32, __half, f16)
DEFINE_QUANT_LAUNCHER(2, 64, __half, f16)
DEFINE_QUANT_LAUNCHER(2, 128, __half, f16)

// 4-bit quantize launchers
DEFINE_QUANT_LAUNCHER(4, 32, float, f32)
DEFINE_QUANT_LAUNCHER(4, 64, float, f32)
DEFINE_QUANT_LAUNCHER(4, 128, float, f32)
DEFINE_QUANT_LAUNCHER(4, 32, __half, f16)
DEFINE_QUANT_LAUNCHER(4, 64, __half, f16)
DEFINE_QUANT_LAUNCHER(4, 128, __half, f16)

// 8-bit quantize launchers
DEFINE_QUANT_LAUNCHER(8, 32, float, f32)
DEFINE_QUANT_LAUNCHER(8, 64, float, f32)
DEFINE_QUANT_LAUNCHER(8, 128, float, f32)
DEFINE_QUANT_LAUNCHER(8, 32, __half, f16)
DEFINE_QUANT_LAUNCHER(8, 64, __half, f16)
DEFINE_QUANT_LAUNCHER(8, 128, __half, f16)

// BFloat16 quantize
DEFINE_QUANT_LAUNCHER(2, 32, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(2, 64, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(2, 128, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(4, 32, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(4, 64, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(4, 128, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(8, 32, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(8, 64, __nv_bfloat16, bf16)
DEFINE_QUANT_LAUNCHER(8, 128, __nv_bfloat16, bf16)

// Note: 3-bit and 6-bit quantization kernels require special byte packing
// and are more complex. For now, these are handled by the CPU fallback
// or can be added later with specialized kernels.
