#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int PARAM_COUNT = 15;
constexpr int THREADS = 256;
constexpr int OPTIMIZED_WIDTH = 7;

struct ConvParams {
  uint64_t values[PARAM_COUNT];
};

template <typename T>
__device__ __forceinline__ T gated_input(const T *__restrict__ x,
                                         const ConvParams &params,
                                         uint64_t batch, uint64_t token,
                                         uint64_t channel) {
  const auto *p = params.values;
  const uint64_t offset = batch * p[4] + token * p[5] + channel * p[6];
  return T(float(x[offset]) * float(x[offset + 2 * p[2] * p[6]]));
}

template <typename T, int WIDTH>
__global__ void gated_conv(const T *__restrict__ x,
                           const T *__restrict__ weight,
                           const T *__restrict__ bias,
                           const T *__restrict__ state, T *__restrict__ output,
                           T *__restrict__ next_state, ConvParams params) {
  const auto *p = params.values;
  const uint64_t seq = p[1], hidden = p[2], width = WIDTH ? WIDTH : p[3];
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= p[0] * seq * hidden) {
    return;
  }
  const uint64_t channel = index % hidden;
  const uint64_t token = index / hidden % seq;
  const uint64_t batch = index / (hidden * seq);
  const uint64_t state_base = batch * p[7] + channel * p[8];
  float sum = 0.0f;
#pragma unroll
  for (int tap = 0; tap < width; tap++) {
    T value = T(0.0f);
    if (token + tap + 1 >= width) {
      value = gated_input(x, params, batch, token + tap + 1 - width, channel);
    } else if (p[13]) {
      value = state[state_base + (token + tap + 1) * p[9]];
    }
    sum += float(value) * float(weight[channel * p[10] + tap * p[11]]);
  }
  if (p[14]) {
    sum += float(bias[channel * p[12]]);
  }
  const T convolved = T(sum);
  const uint64_t gate = batch * p[4] + token * p[5] + (hidden + channel) * p[6];
  output[index] = T(float(convolved) * float(x[gate]));
  if (token + 1 == seq) {
#pragma unroll
    for (int tap = 0; tap < width; tap++) {
      T value = T(0.0f);
      if (seq + tap >= width) {
        value = gated_input(x, params, batch, seq + tap - width, channel);
      } else if (p[13]) {
        value = state[state_base + (seq + tap) * p[9]];
      }
      next_state[(batch * hidden + channel) * width + tap] = value;
    }
  }
}

template <typename T>
int launch(const void *x, const void *weight, const void *bias,
           const void *state, void *output, void *next_state,
           const uint64_t *params, cudaStream_t stream) {
  ConvParams p;
  for (int i = 0; i < PARAM_COUNT; i++) {
    p.values[i] = params[i];
  }
  const uint64_t count = params[0] * params[1] * params[2];
  const uint32_t blocks = (count + THREADS - 1) / THREADS;
  if (params[3] == OPTIMIZED_WIDTH) {
    gated_conv<T, OPTIMIZED_WIDTH><<<blocks, THREADS, 0, stream>>>(
        static_cast<const T *>(x), static_cast<const T *>(weight),
        static_cast<const T *>(bias), static_cast<const T *>(state),
        static_cast<T *>(output), static_cast<T *>(next_state), p);
  } else {
    gated_conv<T, 0><<<blocks, THREADS, 0, stream>>>(
        static_cast<const T *>(x), static_cast<const T *>(weight),
        static_cast<const T *>(bias), static_cast<const T *>(state),
        static_cast<T *>(output), static_cast<T *>(next_state), p);
  }
  return int(cudaGetLastError());
}

} // namespace

#define LFM3_CONV_EXPORT(name, type)                                           \
  extern "C" int name(const void *x, const void *weight, const void *bias,     \
                      const void *state, void *output, void *next_state,       \
                      const uint64_t *params, cudaStream_t stream) {           \
    return launch<type>(x, weight, bias, state, output, next_state, params,    \
                        stream);                                               \
  }

LFM3_CONV_EXPORT(lfm3_conv_f32, float)
LFM3_CONV_EXPORT(lfm3_conv_f16, __half)
LFM3_CONV_EXPORT(lfm3_conv_bf16, __nv_bfloat16)
