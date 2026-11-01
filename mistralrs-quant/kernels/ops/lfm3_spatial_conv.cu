#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void lfm3_spatial_conv_kernel(const T *p, const T *weights,
                                         T *output, size_t elements,
                                         uint32_t height, uint32_t width,
                                         uint32_t merge, uint32_t channels,
                                         uint32_t kernel_size) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  uint32_t channel = index % channels;
  size_t row = index / channels;
  size_t frame = row / (height * width);
  uint32_t patch = row % (height * width);
  uint32_t block = patch / (merge * merge);
  uint32_t y = block / (width / merge) * merge + patch / merge % merge;
  uint32_t x = block % (width / merge) * merge + patch % merge;
  float sum = 0;
  for (uint32_t ky = 0; ky < kernel_size; ++ky) {
    int iy = int(y) + int(ky) - int(kernel_size / 2);
    if (iy < 0 || iy >= int(height))
      continue;
    for (uint32_t kx = 0; kx < kernel_size; ++kx) {
      int ix = int(x) + int(kx) - int(kernel_size / 2);
      if (ix < 0 || ix >= int(width))
        continue;
      size_t token =
          frame * height * width +
          (uint32_t(iy) / merge * (width / merge) + uint32_t(ix) / merge) *
              merge * merge +
          uint32_t(iy) % merge * merge + uint32_t(ix) % merge;
      size_t base = token * 3 * channels + channel;
      float gated = float(T(float(p[base]) * float(p[base + 2 * channels])));
      sum += gated *
             float(weights[(channel * kernel_size + ky) * kernel_size + kx]);
    }
  }
  output[index] =
      T(float(T(sum)) * float(p[row * 3 * channels + channels + channel]));
}

constexpr uint32_t kSpatialConvThreads = 256;

template <typename T>
int launch_lfm3_spatial_conv(const void *p, const void *w, void *output,
                             size_t elements, uint32_t height, uint32_t width,
                             uint32_t merge, uint32_t channels, uint32_t kernel,
                             void *stream) {
  lfm3_spatial_conv_kernel<T>
      <<<(elements + kSpatialConvThreads - 1) / kSpatialConvThreads,
         kSpatialConvThreads, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const T *>(p), static_cast<const T *>(w),
          static_cast<T *>(output), elements, height, width, merge, channels,
          kernel);
  return int(cudaGetLastError());
}

#define WRAPPER(T, NAME)                                                       \
  extern "C" int lfm3_spatial_conv_##NAME(                                     \
      const void *p, const void *w, void *output, size_t elements,             \
      uint32_t height, uint32_t width, uint32_t merge, uint32_t channels,      \
      uint32_t kernel, void *stream) {                                         \
    return launch_lfm3_spatial_conv<T>(p, w, output, elements, height, width,  \
                                       merge, channels, kernel, stream);       \
  }
WRAPPER(float, f32)
WRAPPER(__half, f16)
WRAPPER(__nv_bfloat16, bf16)
