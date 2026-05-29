// Derived from vLLM (Apache-2.0): https://github.com/vllm-project/vllm

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

namespace {

__device__ __forceinline__ float gelu_tanh_f32(float f) {
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715f;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return 0.5f * f * (1.0f + ::tanhf(inner));
}

template <typename T> __device__ __forceinline__ float to_f32(T x);
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <> __device__ __forceinline__ float to_f32<__half>(__half x) {
  return __half2float(x);
}

template <typename T> __device__ __forceinline__ T from_f32(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <> __device__ __forceinline__ __half from_f32<__half>(float x) {
  return __float2half(x);
}

template <typename T>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input,
                                   const int d) {
  const T* x_ptr = input + (size_t)blockIdx.x * 2 * d;  // gate
  const T* y_ptr = x_ptr + d;                           // up
  T* out_ptr = out + (size_t)blockIdx.x * d;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float gate = to_f32<T>(x_ptr[idx]);
    float up = to_f32<T>(y_ptr[idx]);
    out_ptr[idx] = from_f32<T>(gelu_tanh_f32(gate) * up);
  }
}

}  // namespace

extern "C" void launch_gelu_tanh_and_mul_bf16(void* out, const void* input,
                                              int32_t num_tokens, int32_t d,
                                              cudaStream_t stream) {
  if (num_tokens == 0) return;
  dim3 grid(num_tokens);
  dim3 block(d < 1024 ? d : 1024);
  act_and_mul_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<const __nv_bfloat16*>(input), d);
}

extern "C" void launch_gelu_tanh_and_mul_f16(void* out, const void* input,
                                             int32_t num_tokens, int32_t d,
                                             cudaStream_t stream) {
  if (num_tokens == 0) return;
  dim3 grid(num_tokens);
  dim3 block(d < 1024 ? d : 1024);
  act_and_mul_kernel<__half><<<grid, block, 0, stream>>>(
      reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(input), d);
}

__global__ void moe_sum_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                    const __nv_bfloat16* __restrict__ input,
                                    int32_t num_tokens, int32_t hidden,
                                    int32_t topk) {
  const int token = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= num_tokens || col >= hidden) return;

  const size_t base = ((size_t)token * topk * hidden) + col;
  float acc = 0.0f;
  for (int slot = 0; slot < topk; ++slot) {
    acc += __bfloat162float(input[base + (size_t)slot * hidden]);
  }
  out[(size_t)token * hidden + col] = __float2bfloat16(acc);
}

extern "C" void launch_moe_sum_bf16(void* out, const void* input,
                                     int32_t num_tokens, int32_t hidden,
                                     int32_t topk, cudaStream_t stream) {
  if (num_tokens == 0 || hidden == 0 || topk == 0) return;
  const int threads = 256;
  dim3 grid(num_tokens, (hidden + threads - 1) / threads);
  moe_sum_bf16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<const __nv_bfloat16*>(input), num_tokens, hidden, topk);
}
