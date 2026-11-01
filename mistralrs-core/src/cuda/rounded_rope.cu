#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int NORM_ROPE_WARP = 32;
constexpr int NORM_ROPE_BLOCK = 1024;

struct NormRopeArgs {
  const void *input;
  const void *weight;
  const void *cos;
  const void *sin;
  void *output;
  int batch;
  int sequence;
  int heads;
  int columns;
  int half_rot;
  int64_t strides[4];
  float epsilon;
};

__device__ __forceinline__ float rounded_mul(float a, float b) {
  float out;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}
__device__ __forceinline__ float rounded_add(float a, float b) {
  float out;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}
__device__ __forceinline__ float rounded_sub(float a, float b) {
  float out;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}
__device__ __forceinline__ float norm_scale(float sum, int columns, float eps) {
  float mean;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(float(columns)));
  float inverse;
  const float value = rounded_add(mean, eps);
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(inverse) : "f"(value));
  return inverse;
}
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = NORM_ROPE_WARP / 2; offset > 0; offset >>= 1)
    value = rounded_add(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

// RMSNorm rounds to the input dtype before the separate F32 rotary operations.
template <typename T>
__device__ __forceinline__ float normalized(const T *input, const T *weight,
                                            int64_t base, int64_t stride,
                                            int column, float scale) {
  const float value =
      rounded_mul(rounded_mul(scale, float(input[base + column * stride])),
                  float(weight[column]));
  return float(T(value));
}

template <typename T>
__global__ void rounded_norm_partial_rope(NormRopeArgs args) {
  const int row = blockIdx.x;
  const int head = row % args.heads;
  const int token = row / args.heads;
  const int batch = token / args.sequence;
  const int sequence = token % args.sequence;
  const int64_t base = batch * args.strides[0] + sequence * args.strides[1] +
                       head * args.strides[2];
  const T *input = static_cast<const T *>(args.input);
  const T *weight = static_cast<const T *>(args.weight);
  const T *cos =
      static_cast<const T *>(args.cos) + int64_t(token) * args.half_rot;
  const T *sin =
      static_cast<const T *>(args.sin) + int64_t(token) * args.half_rot;
  T *output = static_cast<T *>(args.output) + int64_t(row) * args.columns;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < args.columns; col += blockDim.x) {
    const float value = float(input[base + col * args.strides[3]]);
    asm("fma.rn.f32 %0, %1, %1, %2;" : "=f"(sum) : "f"(value), "f"(sum));
  }
  sum = warp_sum(sum);
  if (blockDim.x > NORM_ROPE_WARP) {
    __shared__ float sums[NORM_ROPE_WARP];
    if (threadIdx.x % NORM_ROPE_WARP == 0)
      sums[threadIdx.x / NORM_ROPE_WARP] = sum;
    __syncthreads();
    sum = warp_sum(sums[threadIdx.x % NORM_ROPE_WARP]);
  }
  const float scale = norm_scale(sum, args.columns, args.epsilon);
  for (int col = threadIdx.x; col < args.half_rot; col += blockDim.x) {
    const float x =
        normalized(input, weight, base, args.strides[3], col, scale);
    const float y = normalized(input, weight, base, args.strides[3],
                               col + args.half_rot, scale);
    const float c = float(cos[col]);
    const float s = float(sin[col]);
    output[col] = T(rounded_sub(rounded_mul(x, c), rounded_mul(y, s)));
    output[col + args.half_rot] =
        T(rounded_add(rounded_mul(y, c), rounded_mul(x, s)));
  }
  for (int col = 2 * args.half_rot + threadIdx.x; col < args.columns;
       col += blockDim.x)
    output[col] =
        T(normalized(input, weight, base, args.strides[3], col, scale));
}

extern "C" void launch_rounded_norm_rope(NormRopeArgs args, int dtype,
                                         void *stream) {
  const int rows = args.batch * args.sequence * args.heads;
  const int threads =
      args.columns < NORM_ROPE_BLOCK ? NORM_ROPE_WARP : NORM_ROPE_BLOCK;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (dtype == 0)
    rounded_norm_partial_rope<float><<<rows, threads, 0, s>>>(args);
  else if (dtype == 1)
    rounded_norm_partial_rope<__half><<<rows, threads, 0, s>>>(args);
  else
    rounded_norm_partial_rope<__nv_bfloat16><<<rows, threads, 0, s>>>(args);
}
