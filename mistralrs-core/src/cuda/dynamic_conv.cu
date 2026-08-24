#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kMaxDynamicConvKernelSize = 8;

template <typename T> __device__ __forceinline__ float dynamic_conv_to_float(T value) {
  return static_cast<float>(value);
}

template <typename T> __device__ __forceinline__ T dynamic_conv_from_float(float value) {
  return static_cast<T>(value);
}

template <typename T, int KernelSize>
__global__ void dynamic_conv_kernel(const T *hidden, const T *dynamic,
                                    const T *base, T *output, int batch,
                                    int sequence_length, int hidden_size,
                                    int group_size, int kernel_size,
                                    int64_t hidden_stride_b,
                                    int64_t hidden_stride_s,
                                    int64_t hidden_stride_h,
                                    int64_t dynamic_stride_b,
                                    int64_t dynamic_stride_s,
                                    int64_t dynamic_stride_k,
                                    int64_t dynamic_stride_g,
                                    int64_t base_stride_k,
                                    int64_t base_stride_h) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(batch) * sequence_length *
                           hidden_size;
  if (index >= elements) {
    return;
  }

  const int channel = static_cast<int>(index % hidden_size);
  const int64_t row = index / hidden_size;
  const int position = static_cast<int>(row % sequence_length);
  const int batch_index = static_cast<int>(row / sequence_length);
  const int group = channel / group_size;
  const int taps = KernelSize == 0 ? kernel_size : KernelSize;

  float result = 0.0f;
#pragma unroll
  for (int offset = 0;
       offset < (KernelSize == 0 ? kMaxDynamicConvKernelSize : KernelSize);
       ++offset) {
    if (offset >= taps || offset > position) {
      continue;
    }
    const int64_t hidden_index = static_cast<int64_t>(batch_index) *
                                     hidden_stride_b +
                                 static_cast<int64_t>(position - offset) *
                                     hidden_stride_s +
                                 static_cast<int64_t>(channel) * hidden_stride_h;
    const int64_t dynamic_index =
        static_cast<int64_t>(batch_index) * dynamic_stride_b +
        static_cast<int64_t>(position) * dynamic_stride_s +
        static_cast<int64_t>(offset) * dynamic_stride_k +
        static_cast<int64_t>(group) * dynamic_stride_g;
    const int64_t base_index =
        static_cast<int64_t>(offset) * base_stride_k +
        static_cast<int64_t>(channel) * base_stride_h;
    const float value = dynamic_conv_to_float(hidden[hidden_index]);
    const float weight = dynamic_conv_to_float(base[base_index]) +
                         dynamic_conv_to_float(dynamic[dynamic_index]);
    result += value * weight;
  }
  output[index] = dynamic_conv_from_float<T>(result);
}

template <typename T>
int launch_dynamic_conv(const void *hidden, const void *dynamic,
                        const void *base, void *output, int batch,
                        int sequence_length, int hidden_size, int group_size,
                        int kernel_size, int64_t hidden_stride_b,
                        int64_t hidden_stride_s, int64_t hidden_stride_h,
                        int64_t dynamic_stride_b, int64_t dynamic_stride_s,
                        int64_t dynamic_stride_k, int64_t dynamic_stride_g,
                        int64_t base_stride_k, int64_t base_stride_h,
                        int64_t stream) {
  if (batch <= 0 || sequence_length <= 0 || hidden_size <= 0 ||
      group_size <= 0 || hidden_size % group_size != 0 || kernel_size <= 0 ||
      kernel_size > kMaxDynamicConvKernelSize) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = static_cast<int64_t>(batch) * sequence_length *
                           hidden_size;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (kernel_size == 2) {
    dynamic_conv_kernel<T, 2><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const T *>(hidden), static_cast<const T *>(dynamic),
        static_cast<const T *>(base), static_cast<T *>(output), batch,
        sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b,
        hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s,
        dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
  } else {
    dynamic_conv_kernel<T, 0><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const T *>(hidden), static_cast<const T *>(dynamic),
        static_cast<const T *>(base), static_cast<T *>(output), batch,
        sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b,
        hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s,
        dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dynamic_conv_bf16(const void *hidden, const void *dynamic,
                                  const void *base, void *output, int batch,
                                  int sequence_length, int hidden_size,
                                  int group_size, int kernel_size,
                                  int64_t hidden_stride_b,
                                  int64_t hidden_stride_s,
                                  int64_t hidden_stride_h,
                                  int64_t dynamic_stride_b,
                                  int64_t dynamic_stride_s,
                                  int64_t dynamic_stride_k,
                                  int64_t dynamic_stride_g,
                                  int64_t base_stride_k,
                                  int64_t base_stride_h,
                                  int64_t stream) {
  return launch_dynamic_conv<__nv_bfloat16>(
      hidden, dynamic, base, output, batch, sequence_length, hidden_size,
      group_size, kernel_size, hidden_stride_b, hidden_stride_s,
      hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k,
      dynamic_stride_g, base_stride_k, base_stride_h, stream);
}

extern "C" int dynamic_conv_f16(const void *hidden, const void *dynamic,
                                 const void *base, void *output, int batch,
                                 int sequence_length, int hidden_size,
                                 int group_size, int kernel_size,
                                 int64_t hidden_stride_b,
                                 int64_t hidden_stride_s,
                                 int64_t hidden_stride_h,
                                 int64_t dynamic_stride_b,
                                 int64_t dynamic_stride_s,
                                 int64_t dynamic_stride_k,
                                 int64_t dynamic_stride_g,
                                 int64_t base_stride_k,
                                 int64_t base_stride_h,
                                 int64_t stream) {
  return launch_dynamic_conv<__half>(hidden, dynamic, base, output, batch,
                                     sequence_length, hidden_size, group_size,
                                     kernel_size, hidden_stride_b,
                                     hidden_stride_s, hidden_stride_h,
                                     dynamic_stride_b, dynamic_stride_s,
                                     dynamic_stride_k, dynamic_stride_g,
                                     base_stride_k, base_stride_h, stream);
}

extern "C" int dynamic_conv_f32(const void *hidden, const void *dynamic,
                                 const void *base, void *output, int batch,
                                 int sequence_length, int hidden_size,
                                 int group_size, int kernel_size,
                                 int64_t hidden_stride_b,
                                 int64_t hidden_stride_s,
                                 int64_t hidden_stride_h,
                                 int64_t dynamic_stride_b,
                                 int64_t dynamic_stride_s,
                                 int64_t dynamic_stride_k,
                                 int64_t dynamic_stride_g,
                                 int64_t base_stride_k,
                                 int64_t base_stride_h,
                                 int64_t stream) {
  return launch_dynamic_conv<float>(hidden, dynamic, base, output, batch,
                                    sequence_length, hidden_size, group_size,
                                    kernel_size, hidden_stride_b,
                                    hidden_stride_s, hidden_stride_h,
                                    dynamic_stride_b, dynamic_stride_s,
                                    dynamic_stride_k, dynamic_stride_g,
                                    base_stride_k, base_stride_h, stream);
}
