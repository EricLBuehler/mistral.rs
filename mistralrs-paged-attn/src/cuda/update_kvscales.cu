#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define DIV_CONST 240.0f;

template <typename T> __device__ __forceinline__ float to_float_abs(T x);

template <> __device__ __forceinline__ float to_float_abs<float>(float x) {
  return fabsf(x);
}

template <> __device__ __forceinline__ float to_float_abs<__half>(__half x) {
  return fabsf(__half2float(x));
}

template <>
__device__ __forceinline__ float to_float_abs<__nv_bfloat16>(__nv_bfloat16 x) {
  return fabsf(__bfloat162float(x));
}

// --- atomicMax for floats using atomicCAS on int bits ---
// Works for non-negative floats (we only use absolute values divided by 240)
__device__ __forceinline__ float atomicMaxFloat(float *address, float val) {
  int *addr_i = reinterpret_cast<int *>(address);
  int old_i = *addr_i;
  float old_f = __int_as_float(old_i);

  while (old_f < val) {
    int assumed = old_i;
    int new_i = __float_as_int(val);
    int prev = atomicCAS(addr_i, assumed, new_i);
    if (prev == assumed) {
      return __int_as_float(prev); // return previous value
    }
    old_i = prev;
    old_f = __int_as_float(old_i);
  }
  return old_f;
}

// find abs-max of k and v, divide by 240, atomically update corresponding kv
// scales --- Reference implementation:
// https://github.com/guoqingbao/attention.rs/tree/main/src/kernels/src/update_kvscales.cu
template <typename T>
__global__ void compute_and_update_scales_kernel(
    const T *__restrict__ k, const T *__restrict__ v, long num_elements,
    float *__restrict__ k_scales, // single-float pointer in device memory
    float *__restrict__ v_scales  // single-float pointer in device memory
) {
  extern __shared__ float sdata[]; // 2 * blockDim.x floats
  float *s_k = sdata;
  float *s_v = sdata + blockDim.x;

  const int tid = threadIdx.x;
  const int bdim = blockDim.x;
  const int gdim = gridDim.x;
  long global_thread_index = (long)blockIdx.x * bdim + tid;

  // per-thread local maxima
  float local_max_k = 0.0f;
  float local_max_v = 0.0f;

  // strided loop covering entire array
  long idx = global_thread_index;
  long stride = (long)bdim * (long)gdim;
  while (idx < num_elements) {
    float avk = to_float_abs<T>(k[idx]);
    float avv = to_float_abs<T>(v[idx]);
    if (avk > local_max_k)
      local_max_k = avk;
    if (avv > local_max_v)
      local_max_v = avv;
    idx += stride;
  }

  // store per-thread maxima to shared memory
  s_k[tid] = local_max_k;
  s_v[tid] = local_max_v;
  __syncthreads();

  // parallel reduction in shared memory to find block maxima
  for (int s = bdim >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      float other_k = s_k[tid + s];
      if (other_k > s_k[tid])
        s_k[tid] = other_k;
      float other_v = s_v[tid + s];
      if (other_v > s_v[tid])
        s_v[tid] = other_v;
    }
    __syncthreads();
  }

  // thread 0 of block updates global scales atomically
  if (tid == 0) {
    float candidate_k_scale = s_k[0] / DIV_CONST;
    float candidate_v_scale = s_v[0] / DIV_CONST;
    // only attempt update if candidate > 0 (guard; optional)
    if (candidate_k_scale > 0.0f)
      atomicMaxFloat(k_scales, candidate_k_scale);
    if (candidate_v_scale > 0.0f)
      atomicMaxFloat(v_scales, candidate_v_scale);
  }
}

template <typename T>
void update_scales_typed(T *k, T *v, long num_elements,
                         float *k_scales, // device single float pointer
                         float *v_scales, // device single float pointer
                         int64_t stream_) {
  const int threads = 512;
  long blocks = (num_elements + threads - 1) / threads;
  if (blocks == 0)
    blocks = 1;
  if (blocks > 65535)
    blocks = 65535;
  const cudaStream_t stream = (cudaStream_t)stream_;
  size_t shared_bytes = 2 * threads * sizeof(float);
  compute_and_update_scales_kernel<T>
      <<<(int)blocks, threads, shared_bytes, stream>>>(k, v, num_elements,
                                                       k_scales, v_scales);
}

extern "C" void update_kv_scales_f32(void *k, void *v, const long num_elements,
                                     float *k_scales, float *v_scales,
                                     int64_t stream_) {
  update_scales_typed<float>(reinterpret_cast<float *>(k),
                             reinterpret_cast<float *>(v), num_elements,
                             k_scales, v_scales, stream_);
}

extern "C" void update_kv_scales_f16(void *k, void *v, const long num_elements,
                                     float *k_scales, float *v_scales,
                                     int64_t stream_) {
  update_scales_typed<__half>(reinterpret_cast<__half *>(k),
                              reinterpret_cast<__half *>(v), num_elements,
                              k_scales, v_scales, stream_);
}

extern "C" void update_kv_scales_bf16(void *k, void *v, const long num_elements,
                                      float *k_scales, float *v_scales,
                                      int64_t stream_) {
  update_scales_typed<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(k),
                                     reinterpret_cast<__nv_bfloat16 *>(v),
                                     num_elements, k_scales, v_scales, stream_);
}
