#include "mmq_common.cuh"
#include <cuda_runtime.h>

constexpr int GROUPED_WARPS = 4;
constexpr int GATE_FEATURES_PER_WARP = 2;
constexpr int DOWN_FEATURES_PER_WARP = 4;
constexpr int FRAGMENT_VDR = 2;

struct GroupedGemvArgs {
  const void *gate;
  const void *up;
  const void *x;
  const uint32_t *bounds;
  const uint32_t *sorted_ids;
  const float *route_weights;
  float *output;
  int n;
  int k;
  int k_padded;
  int topk;
  int experts;
};

struct Q4KFragment {
  int v[2];
  uint16_t scales[2];
  half2 dm;
};

static __device__ __forceinline__ Q4KFragment load_q4k_fragment(
    const block_q4_K *w, int iqs) {
  Q4KFragment f;
  const int q8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q = (const int *)(w->qs + 16 * q8_offset + 4 * ((iqs / 2) % 4));
  f.v[0] = q[0];
  f.v[1] = q[4];
  const uint16_t *scales = (const uint16_t *)w->scales;
  const int j = q8_offset / 2;
  if (j < 2) {
    f.scales[0] = scales[j] & 0x3f3f;
    f.scales[1] = scales[j + 2] & 0x3f3f;
  } else {
    f.scales[0] = (scales[j + 2] & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    f.scales[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j] & 0xc0c0) >> 2);
  }
  f.dm = w->dm;
  return f;
}

struct Q8KFragment {
  int u[2 * QR4_K];
  float d[QR4_K];
};

static __device__ __forceinline__ Q8KFragment load_q8k_fragment(
    const block_q8_1 *x, int iqs) {
  Q8KFragment f;
  const int q8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *b = x + q8_offset + i;
    const int *q = (const int *)b->qs + ((iqs / 2) % 4);
    f.u[2 * i] = q[0];
    f.u[2 * i + 1] = q[4];
    f.d[i] = __low2float(b->ds);
  }
  return f;
}

static __device__ __forceinline__ float dot_q4k_fragment(
    const Q4KFragment &w, const Q8KFragment &x) {
  const uint8_t *sc = (const uint8_t *)w.scales;
  const uint8_t *m = sc + 2;
  float sum_d = 0.0f;
  float sum_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0 = (w.v[0] >> (4 * i)) & 0x0f0f0f0f;
    const int v1 = (w.v[1] >> (4 * i)) & 0x0f0f0f0f;
    const int dot = ggml_cuda_dp4a(v1, x.u[2 * i + 1], ggml_cuda_dp4a(v0, x.u[2 * i], 0));
    const int sum = ggml_cuda_dp4a(0x01010101, x.u[2 * i + 1], ggml_cuda_dp4a(0x01010101, x.u[2 * i], 0));
    sum_d += x.d[i] * (dot * sc[i]);
    sum_m += x.d[i] * (sum * m[i]);
  }
  const float2 dm = __half22float2(w.dm);
  return dm.x * sum_d - dm.y * sum_m;
}

struct Q41Fragment {
  int v[FRAGMENT_VDR];
  half2 dm;
};

static __device__ __forceinline__ Q41Fragment load_q41_fragment(
    const block_q4_1 *w, int iqs) {
  Q41Fragment f;
#pragma unroll
  for (int i = 0; i < FRAGMENT_VDR; ++i) {
    f.v[i] = ((const int *)w->qs)[iqs + i];
  }
  f.dm = w->dm;
  return f;
}

static __device__ __forceinline__ float dot_q41_fragment(
    const Q41Fragment &w, const block_q8_1 *x, int iqs) {
  int sum = 0;
#pragma unroll
  for (int i = 0; i < FRAGMENT_VDR; ++i) {
    const int lo = w.v[i] & 0x0f0f0f0f;
    const int hi = (w.v[i] >> 4) & 0x0f0f0f0f;
    const int *q = (const int *)x->qs;
    sum = ggml_cuda_dp4a(lo, q[iqs + i], sum);
    sum = ggml_cuda_dp4a(hi, q[iqs + i + QI4_1], sum);
  }
  const float2 dm = __half22float2(w.dm);
  const float2 ds = __half22float2(x->ds);
  const float dd = dm.x * ds.x;
  const float ms = dm.y * ds.y;
  return sum * dd + ms / (QI8_1 / (FRAGMENT_VDR * QR4_1));
}

template <int GROUP>
__global__ void grouped_gemv_gate_up(GroupedGemvArgs a) {
  const int expert = blockIdx.y;
  const int begin = a.bounds[expert];
  const int end = a.bounds[expert + 1];
  const int feature0 = (blockIdx.x * GROUPED_WARPS + threadIdx.y) * GATE_FEATURES_PER_WARP;
  if (begin == end || feature0 >= a.n) return;
  const int blocks_per_row = a.k / QK_K;
  const int input_stride = a.k_padded / QK8_1;
  const size_t expert_offset = (size_t)expert * a.n * blocks_per_row;
  const block_q4_K *gate = (const block_q4_K *)a.gate + expert_offset;
  const block_q4_K *up = (const block_q4_K *)a.up + expert_offset;
  const block_q8_1 *input = (const block_q8_1 *)a.x;
  constexpr int blocks_per_iter = FRAGMENT_VDR * WARP_SIZE / QI4_K;
  const int kqs = FRAGMENT_VDR * (threadIdx.x % (QI4_K / FRAGMENT_VDR));
  for (int pos = begin; pos < end; pos += GROUP) {
    int assignment[GROUP];
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      assignment[g] = pos + g < end ? a.sorted_ids[pos + g] : 0;
    }
    for (int r = 0; r < GATE_FEATURES_PER_WARP && feature0 + r < a.n; ++r) {
      const int feature = feature0 + r;
      float gs[GROUP] = {};
      float us[GROUP] = {};
      for (int kb = threadIdx.x / (QI4_K / FRAGMENT_VDR); kb < blocks_per_row; kb += blocks_per_iter) {
        const size_t wi = (size_t)feature * blocks_per_row + kb;
        const Q4KFragment gw = load_q4k_fragment(gate + wi, kqs);
        const Q4KFragment uw = load_q4k_fragment(up + wi, kqs);
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
          if (pos + g < end) {
            const block_q8_1 *x = input + (size_t)(assignment[g] / a.topk) * input_stride + kb * (QK_K / QK8_1);
            const Q8KFragment xf = load_q8k_fragment(x, kqs);
            gs[g] += dot_q4k_fragment(gw, xf);
            us[g] += dot_q4k_fragment(uw, xf);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        if (pos + g < end) {
          const float gate_sum = warp_reduce_sum(gs[g]);
          const float up_sum = warp_reduce_sum(us[g]);
          if (threadIdx.x == 0) {
            a.output[(size_t)assignment[g] * a.n + feature] = up_sum * (gate_sum / (1.0f + expf(-gate_sum)));
          }
        }
      }
    }
  }
}

template <int GROUP>
__global__ void grouped_gemv_down(GroupedGemvArgs a) {
  const int expert = blockIdx.y;
  const int begin = a.bounds[expert];
  const int end = a.bounds[expert + 1];
  const int feature0 = (blockIdx.x * GROUPED_WARPS + threadIdx.y) * DOWN_FEATURES_PER_WARP;
  if (begin == end || feature0 >= a.n) return;
  const int blocks_per_row = a.k / QK4_1;
  const int input_stride = a.k_padded / QK8_1;
  const block_q4_1 *w = (const block_q4_1 *)a.gate + (size_t)expert * a.n * blocks_per_row;
  const block_q8_1 *input = (const block_q8_1 *)a.x;
  constexpr int blocks_per_iter = FRAGMENT_VDR * WARP_SIZE / QI4_1;
  const int kqs = FRAGMENT_VDR * (threadIdx.x % (QI4_1 / FRAGMENT_VDR));
  for (int pos = begin; pos < end; pos += GROUP) {
    int assignment[GROUP];
#pragma unroll
    for (int g = 0; g < GROUP; ++g) {
      assignment[g] = pos + g < end ? a.sorted_ids[pos + g] : 0;
    }
    for (int r = 0; r < DOWN_FEATURES_PER_WARP && feature0 + r < a.n; ++r) {
      const int feature = feature0 + r;
      float sums[GROUP] = {};
      for (int kb = threadIdx.x / (QI4_1 / FRAGMENT_VDR); kb < blocks_per_row; kb += blocks_per_iter) {
        const Q41Fragment wf = load_q41_fragment(w + (size_t)feature * blocks_per_row + kb, kqs);
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
          if (pos + g < end) {
            const block_q8_1 *x = input + (size_t)assignment[g] * input_stride + kb;
            sums[g] += dot_q41_fragment(wf, x, kqs);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < GROUP; ++g) {
        if (pos + g < end) {
          const float sum = warp_reduce_sum(sums[g]);
          if (threadIdx.x == 0) {
            const int id = assignment[g];
            atomicAdd(a.output + (size_t)(id / a.topk) * a.n + feature, sum * a.route_weights[id]);
          }
        }
      }
    }
  }
}

extern "C" int launch_grouped_gemv_gate_up(const GroupedGemvArgs *args, int width, void *stream) {
  const dim3 block(WARP_SIZE, GROUPED_WARPS);
  const dim3 grid((args->n + GROUPED_WARPS * GATE_FEATURES_PER_WARP - 1) / (GROUPED_WARPS * GATE_FEATURES_PER_WARP), args->experts);
  if (width == 2) grouped_gemv_gate_up<2><<<grid, block, 0, (cudaStream_t)stream>>>(*args);
  else if (width == 4) grouped_gemv_gate_up<4><<<grid, block, 0, (cudaStream_t)stream>>>(*args);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}

extern "C" int launch_grouped_gemv_down(const GroupedGemvArgs *args, int width, int batch, void *stream) {
  const cudaError_t cleared = cudaMemsetAsync(args->output, 0, (size_t)batch * args->n * sizeof(float), (cudaStream_t)stream);
  if (cleared != cudaSuccess) return (int)cleared;
  const dim3 block(WARP_SIZE, GROUPED_WARPS);
  const dim3 grid((args->n + GROUPED_WARPS * DOWN_FEATURES_PER_WARP - 1) / (GROUPED_WARPS * DOWN_FEATURES_PER_WARP), args->experts);
  if (width == 2) grouped_gemv_down<2><<<grid, block, 0, (cudaStream_t)stream>>>(*args);
  else if (width == 4) grouped_gemv_down<4><<<grid, block, 0, (cudaStream_t)stream>>>(*args);
  else return (int)cudaErrorInvalidValue;
  return (int)cudaGetLastError();
}
