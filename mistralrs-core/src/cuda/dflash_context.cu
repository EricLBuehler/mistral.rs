#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kMaxDFlashContextTaps = 64;

template <typename T> struct DFlashTapPackArgs {
  const T *inputs[kMaxDFlashContextTaps];
  int widths[kMaxDFlashContextTaps];
  int offsets[kMaxDFlashContextTaps];
};

template <typename T>
__global__ void dflash_pack_taps_kernel(
    DFlashTapPackArgs<T> args, const uint32_t *__restrict__ row_indices,
    T *__restrict__ output, int taps, int output_rows, int output_width,
    int row_start) {
  const int tap = blockIdx.y;
  if (tap >= taps) {
    return;
  }
  const int width = args.widths[tap];
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(output_rows) * width;
  if (index >= elements) {
    return;
  }
  const int output_row = static_cast<int>(index / width);
  const int column = static_cast<int>(index % width);
  const int source_row = row_indices == nullptr
                             ? row_start + output_row
                             : static_cast<int>(row_indices[output_row]);
  output[static_cast<int64_t>(output_row) * output_width + args.offsets[tap] +
         column] = args.inputs[tap][static_cast<int64_t>(source_row) * width +
                                    column];
}

template <typename T>
int launch_dflash_pack_taps(const void *const *inputs, const int *widths,
                            int taps, const void *row_indices, void *output,
                            int output_rows, int output_width, int row_start,
                            int64_t stream) {
  if (inputs == nullptr || widths == nullptr || output == nullptr || taps <= 0 ||
      taps > kMaxDFlashContextTaps || output_rows <= 0 || output_width <= 0 ||
      row_start < 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  DFlashTapPackArgs<T> args{};
  int offset = 0;
  int max_width = 0;
  for (int tap = 0; tap < taps; ++tap) {
    if (inputs[tap] == nullptr || widths[tap] <= 0) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    args.inputs[tap] = static_cast<const T *>(inputs[tap]);
    args.widths[tap] = widths[tap];
    args.offsets[tap] = offset;
    offset += widths[tap];
    max_width = max_width > widths[tap] ? max_width : widths[tap];
  }
  if (offset != output_width) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  constexpr int threads = 256;
  const int64_t max_elements = static_cast<int64_t>(output_rows) * max_width;
  const int blocks = static_cast<int>((max_elements + threads - 1) / threads);
  const dim3 grid(blocks, taps);
  dflash_pack_taps_kernel<T><<<grid, threads, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
      args, static_cast<const uint32_t *>(row_indices), static_cast<T *>(output),
      taps, output_rows, output_width, row_start);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ float dflash_context_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float
dflash_context_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float
dflash_context_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T dflash_context_from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half
dflash_context_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
dflash_context_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T>
__global__ void dflash_context_keys_kernel(
    const T *__restrict__ input, const T *__restrict__ norm_weights,
    const T *__restrict__ cos, const T *__restrict__ sin,
    const uint32_t *__restrict__ positions, T *__restrict__ output, int heads,
    int rows, int head_dim, int rot_dim, float eps) {
  __shared__ float reduce[1024];

  const int row = blockIdx.x;
  const int token = row % rows;
  const int tmp = row / rows;
  const int layer = tmp / heads;
  const int64_t base = static_cast<int64_t>(row) * head_dim;

  float sum = 0.0f;
  for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
    const float value = dflash_context_to_float(input[base + column]);
    sum += value * value;
  }
  reduce[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms =
      rsqrtf(reduce[0] / static_cast<float>(head_dim) + eps);
  const T *weight = norm_weights + static_cast<int64_t>(layer) * head_dim;
  const int64_t rope_base = static_cast<int64_t>(positions[token]) * rot_dim;
  for (int offset = threadIdx.x; offset < rot_dim; offset += blockDim.x) {
    const int x_index = offset;
    const int y_index = rot_dim + offset;
    const T normalized_x = dflash_context_from_float<T>(
        dflash_context_to_float(input[base + x_index]) * inv_rms *
        dflash_context_to_float(weight[x_index]));
    const T normalized_y = dflash_context_from_float<T>(
        dflash_context_to_float(input[base + y_index]) * inv_rms *
        dflash_context_to_float(weight[y_index]));
    const float x = dflash_context_to_float(normalized_x);
    const float y = dflash_context_to_float(normalized_y);
    const float c = dflash_context_to_float(cos[rope_base + offset]);
    const float s = dflash_context_to_float(sin[rope_base + offset]);
    output[base + x_index] = dflash_context_from_float<T>(x * c - y * s);
    output[base + y_index] = dflash_context_from_float<T>(y * c + x * s);
  }
  for (int column = rot_dim * 2 + threadIdx.x; column < head_dim;
       column += blockDim.x) {
    const float value = dflash_context_to_float(input[base + column]) *
                        inv_rms * dflash_context_to_float(weight[column]);
    output[base + column] = dflash_context_from_float<T>(value);
  }
}

template <typename T>
int launch_dflash_context_keys(const void *input, const void *norm_weights,
                               const void *cos, const void *sin,
                               const void *positions, void *output, int layers,
                               int heads, int rows, int head_dim, int rot_dim,
                               float eps, int64_t stream) {
  if (input == nullptr || norm_weights == nullptr || cos == nullptr ||
      sin == nullptr || positions == nullptr || output == nullptr ||
      layers <= 0 || heads <= 0 || rows <= 0 || head_dim <= 0 || rot_dim <= 0 ||
      rot_dim * 2 > head_dim) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int threads = 32;
  while (threads < head_dim && threads < 1024) {
    threads <<= 1;
  }
  const int blocks = layers * heads * rows;
  dflash_context_keys_kernel<T><<<blocks, threads, 0,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const T *>(input), static_cast<const T *>(norm_weights),
      static_cast<const T *>(cos), static_cast<const T *>(sin),
      static_cast<const uint32_t *>(positions), static_cast<T *>(output), heads,
      rows, head_dim, rot_dim, eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dflash_pack_taps_f16(
    const void *const *inputs, const int *widths, int taps,
    const void *row_indices, void *output, int output_rows, int output_width,
    int row_start, int64_t stream) {
  return launch_dflash_pack_taps<__half>(
      inputs, widths, taps, row_indices, output, output_rows, output_width,
      row_start, stream);
}

extern "C" int dflash_pack_taps_bf16(
    const void *const *inputs, const int *widths, int taps,
    const void *row_indices, void *output, int output_rows, int output_width,
    int row_start, int64_t stream) {
  return launch_dflash_pack_taps<__nv_bfloat16>(
      inputs, widths, taps, row_indices, output, output_rows, output_width,
      row_start, stream);
}

extern "C" int dflash_pack_taps_f32(
    const void *const *inputs, const int *widths, int taps,
    const void *row_indices, void *output, int output_rows, int output_width,
    int row_start, int64_t stream) {
  return launch_dflash_pack_taps<float>(
      inputs, widths, taps, row_indices, output, output_rows, output_width,
      row_start, stream);
}

extern "C" int dflash_context_keys_f16(
    const void *input, const void *norm_weights, const void *cos,
    const void *sin, const void *positions, void *output, int layers,
    int heads, int rows, int head_dim, int rot_dim, float eps, int64_t stream) {
  return launch_dflash_context_keys<__half>(
      input, norm_weights, cos, sin, positions, output, layers, heads, rows,
      head_dim, rot_dim, eps, stream);
}

extern "C" int dflash_context_keys_bf16(
    const void *input, const void *norm_weights, const void *cos,
    const void *sin, const void *positions, void *output, int layers,
    int heads, int rows, int head_dim, int rot_dim, float eps, int64_t stream) {
  return launch_dflash_context_keys<__nv_bfloat16>(
      input, norm_weights, cos, sin, positions, output, layers, heads, rows,
      head_dim, rot_dim, eps, stream);
}

extern "C" int dflash_context_keys_f32(
    const void *input, const void *norm_weights, const void *cos,
    const void *sin, const void *positions, void *output, int layers,
    int heads, int rows, int head_dim, int rot_dim, float eps, int64_t stream) {
  return launch_dflash_context_keys<float>(
      input, norm_weights, cos, sin, positions, output, layers, heads, rows,
      head_dim, rot_dim, eps, stream);
}
