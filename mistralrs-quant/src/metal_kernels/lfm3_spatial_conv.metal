#include <metal_stdlib>
using namespace metal;
#if defined(__HAVE_BFLOAT__)
typedef bfloat bfloat16_t;
#endif

template <typename T>
[[kernel]] void lfm3_spatial_conv(
    const device T *p [[buffer(0)]], const device T *weights [[buffer(1)]],
    device T *output [[buffer(2)]], constant uint &elements [[buffer(3)]],
    constant uint &height [[buffer(4)]], constant uint &width [[buffer(5)]],
    constant uint &merge [[buffer(6)]], constant uint &channels [[buffer(7)]],
    constant uint &kernel_size [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= elements)
    return;
  const uint channel = index % channels;
  const uint row = index / channels;
  const uint frame = row / (height * width);
  const uint patch = row % (height * width);
  const uint block = patch / (merge * merge);
  const uint y = block / (width / merge) * merge + patch / merge % merge;
  const uint x = block % (width / merge) * merge + patch % merge;
  float sum = 0;
  for (uint ky = 0; ky < kernel_size; ++ky) {
    int iy = int(y) + int(ky) - int(kernel_size / 2);
    if (iy < 0 || iy >= int(height))
      continue;
    for (uint kx = 0; kx < kernel_size; ++kx) {
      int ix = int(x) + int(kx) - int(kernel_size / 2);
      if (ix < 0 || ix >= int(width))
        continue;
      uint token = frame * height * width +
                   (uint(iy) / merge * (width / merge) + uint(ix) / merge) *
                       merge * merge +
                   uint(iy) % merge * merge + uint(ix) % merge;
      uint base = token * 3 * channels + channel;
      volatile T gated = T(float(p[base]) * float(p[base + 2 * channels]));
      sum += float(gated) *
             float(weights[(channel * kernel_size + ky) * kernel_size + kx]);
    }
  }
  volatile T convolved = T(sum);
  output[index] =
      T(float(convolved) * float(p[row * 3 * channels + channels + channel]));
}

#define INSTANTIATE(T)                                                         \
  template [[host_name("lfm3_spatial_conv_" #T)]] [[kernel]] void              \
  lfm3_spatial_conv<T>(const device T *, const device T *, device T *,         \
                       constant uint &, constant uint &, constant uint &,      \
                       constant uint &, constant uint &, constant uint &,      \
                       uint);
INSTANTIATE(float)
INSTANTIATE(half)
#if defined(__HAVE_BFLOAT__)
INSTANTIATE(bfloat16_t)
#endif
