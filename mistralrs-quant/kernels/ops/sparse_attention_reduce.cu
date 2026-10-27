#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename T>
__global__ void
sparse_attention_reduce_kernel(const float *__restrict__ partial,
                               const float *__restrict__ stats,
                               T *__restrict__ out, int splits, int head_dim) {
  const int row = blockIdx.x;
  const int dim = threadIdx.x;
  float maximum = -FLT_MAX;
  for (int split = 0; split < splits; ++split) {
    const int offset = (row * splits + split) * 2;
    if (stats[offset + 1] > 0.0f) {
      maximum = fmaxf(maximum, stats[offset]);
    }
  }
  float total = 0.0f;
  float acc = 0.0f;
  for (int split = 0; split < splits; ++split) {
    const int offset = (row * splits + split) * 2;
    const float sum = stats[offset + 1];
    if (sum <= 0.0f) {
      continue;
    }
    const float weight = __expf(stats[offset] - maximum);
    total = fmaf(sum, weight, total);
    acc = fmaf(partial[(row * splits + split) * head_dim + dim], weight, acc);
  }
  out[row * head_dim + dim] = T(total > 0.0f ? acc / total : 0.0f);
}

extern "C" void launch_sparse_attention_reduce(const float *partial,
                                               const float *stats, void *out,
                                               int rows, int splits,
                                               int head_dim, int dtype,
                                               cudaStream_t stream) {
  if (dtype == 0) {
    sparse_attention_reduce_kernel<<<rows, head_dim, 0, stream>>>(
        partial, stats, (__half *)out, splits, head_dim);
  } else {
    sparse_attention_reduce_kernel<<<rows, head_dim, 0, stream>>>(
        partial, stats, (__nv_bfloat16 *)out, splits, head_dim);
  }
}
