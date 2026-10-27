// Adapted from llama.cpp MMVQ; see third_party/ggml_mmvq/LICENSE.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int MOE_WARP = 32;
constexpr int MOE_WARPS = 8;
constexpr int MOE_ROWS_PER_WARP = 2;
constexpr int MOE_HIDDEN = 2560;
constexpr int MOE_INTERMEDIATE = 640;
constexpr int MOE_TOPK = 10;
constexpr int MOE_DOWN_PADDED = 1024;
constexpr int QK8_1 = 32;
constexpr int QI8_1 = 8;
constexpr int QK_K = 256;
constexpr int QR4_K = 2;
constexpr int QR5_K = 2;
constexpr int QI4_K = 32;
constexpr int QI5_K = 32;
constexpr int QI4_1 = 4;
constexpr int QR4_1 = 2;
constexpr int QI5_1 = 4;
constexpr int QI8_0 = 8;
constexpr int VDR_Q4_1_Q8_1_MMVQ = 2;
constexpr int VDR_Q5_1_Q8_1_MMVQ = 2;
constexpr int VDR_Q8_0_Q8_1_MMVQ = 2;

struct block_q8_1 {
  half2 ds;
  int8_t qs[QK8_1];
};
struct block_q4_K {
  half2 dm;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct block_q5_K {
  half2 dm;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
struct block_q4_1 {
  half2 dm;
  uint8_t qs[16];
};
struct block_q5_1 {
  half2 dm;
  uint8_t qh[4];
  uint8_t qs[16];
};
struct block_q8_0 {
  half d;
  int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q5_K) == 176);
static_assert(sizeof(block_q4_1) == 20);
static_assert(sizeof(block_q5_1) == 24);
static_assert(sizeof(block_q8_0) == 34);

enum class MoeQuant { Q4K, Q5K, Q4_1, Q5_1, Q8_0 };
template <MoeQuant> struct MoeFormat;
template <> struct MoeFormat<MoeQuant::Q4K> {
  using Weight = block_q4_K;
};
template <> struct MoeFormat<MoeQuant::Q5K> {
  using Weight = block_q5_K;
};
template <> struct MoeFormat<MoeQuant::Q4_1> {
  using Weight = block_q4_1;
};
template <> struct MoeFormat<MoeQuant::Q5_1> {
  using Weight = block_q5_1;
};
template <> struct MoeFormat<MoeQuant::Q8_0> {
  using Weight = block_q8_0;
};

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = MOE_WARP / 2; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, MOE_WARP);
  }
  return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t *x8,
                                                        const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8,
                                                         const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int
get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

static __device__ __forceinline__ int
get_int_from_uint8_aligned(const uint8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
  return __dp4a(a, b, c);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_1_q8_1_impl(const int *v, const int *u, const half2 &dm4,
                       const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm4f = __half22float2(dm4);
  const float2 ds8f = __half22float2(ds8);
  const float d4d8 = dm4f.x * ds8f.x;
  const float m4s8 = dm4f.y * ds8f.y;
  return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const half2 &dm5, const half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm5f = __half22float2(dm5);
  const float2 ds8f = __half22float2(ds8);
  const float d5d8 = dm5f.x * ds8f.x;
  const float m5s8 = dm5f.y * ds8f.y;
  return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q8_0_q8_1_impl(const int *v, const int *u, const half &d8_0,
                       const half &d8_1) {

  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return sumi * __half2float(d8_0) * __half2float(d8_1);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const half2 &dm4, const float *__restrict__ d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 =
        ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = __half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const half2 &dm5,
    const float *__restrict__ d8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 =
        ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm5f = __half22float2(dm5);
  return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q4_1 *bq4_1 = (const block_q4_1 *)vbq;
  int v[VDR_Q4_1_Q8_1_MMVQ];
  int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
  }
  return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {
  const block_q5_1 *bq5_1 = (const block_q5_1 *)vbq;
  int vl[VDR_Q5_1_Q8_1_MMVQ];
  int vh[VDR_Q5_1_Q8_1_MMVQ];
  int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
    vh[i] = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
  }
  return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm,
                                                    bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq;
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
  }
  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                    __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q4_K *bq4_K = (const block_q4_K *)vbq;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 =
      (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t *scales = (const uint16_t *)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &iqs) {

  const block_q5_K *bq5_K = (const block_q5_K *)vbq;
  int vl[2];
  int vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];

  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int *ql =
      (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));

  vl[0] = ql[0];
  vl[1] = ql[4];
  vh[0] = qh[0] >> bq8_offset;
  vh[1] = qh[4] >> bq8_offset;

  const uint16_t *scales = (const uint16_t *)bq5_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;

#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = __low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

template <MoeQuant TYPE>
__device__ __forceinline__ float
moe_dot(const void *weight, const block_q8_1 *activation, int part) {
  if constexpr (TYPE == MoeQuant::Q4K)
    return vec_dot_q4_K_q8_1(weight, activation, part);
  else if constexpr (TYPE == MoeQuant::Q5K)
    return vec_dot_q5_K_q8_1(weight, activation, part);
  else if constexpr (TYPE == MoeQuant::Q4_1)
    return vec_dot_q4_1_q8_1(weight, activation, part);
  else if constexpr (TYPE == MoeQuant::Q5_1)
    return vec_dot_q5_1_q8_1(weight, activation, part);
  else
    return vec_dot_q8_0_q8_1(weight, activation, part);
}

template <MoeQuant TYPE>
__global__ void moe_decode_sm121(const void *__restrict__ weights,
                                 const void *__restrict__ up,
                                 const block_q8_1 *__restrict__ input,
                                 const unsigned int *__restrict__ indices,
                                 const float *__restrict__ routing,
                                 float *__restrict__ output) {
  constexpr bool gate = TYPE == MoeQuant::Q4K || TYPE == MoeQuant::Q5K;
  constexpr int rows = gate ? MOE_INTERMEDIATE : MOE_HIDDEN;
  constexpr int columns = gate ? MOE_HIDDEN : MOE_INTERMEDIATE;
  constexpr int group = gate ? QK_K : QK8_1;
  constexpr int qi = gate ? QI4_K : (TYPE == MoeQuant::Q8_0 ? QI8_0 : QI5_1);
  constexpr int values_per_dot = 2;
  constexpr int blocks_per_row = columns / group;
  constexpr int blocks_per_iteration = values_per_dot * MOE_WARP / qi;
  using Weight = typename MoeFormat<TYPE>::Weight;
  const int lane = threadIdx.x;
  const int row = (blockIdx.x * MOE_WARPS + threadIdx.y) * MOE_ROWS_PER_WARP;
  const int slot = blockIdx.y;
  const size_t expert_offset = (size_t)indices[slot] * rows * blocks_per_row;
  const auto *weight = static_cast<const Weight *>(weights) + expert_offset;
  const auto *up_weight = static_cast<const Weight *>(up) + expert_offset;
  const auto *activation = input + (gate ? 0 : slot * MOE_DOWN_PADDED / QK8_1);
  float gate_sums[MOE_ROWS_PER_WARP] = {};
  float up_sums[MOE_ROWS_PER_WARP] = {};
  for (int block = lane / (qi / values_per_dot); block < blocks_per_row;
       block += blocks_per_iteration) {
    const int part = values_per_dot * (lane % (qi / values_per_dot));
#pragma unroll
    for (int r = 0; r < MOE_ROWS_PER_WARP; ++r) {
      const int future = block + blocks_per_iteration;
      const int offset = (row + r) * blocks_per_row;
      if (future < blocks_per_row) {
        asm volatile(
            "prefetch.global.L2 [%0];" ::"l"(weight + offset + future));
        if constexpr (gate) {
          asm volatile(
              "prefetch.global.L2 [%0];" ::"l"(up_weight + offset + future));
        }
      }
      gate_sums[r] += moe_dot<TYPE>(weight + offset + block,
                                    activation + block * (group / QK8_1), part);
      if constexpr (gate) {
        up_sums[r] += moe_dot<TYPE>(up_weight + offset + block,
                                    activation + block * (group / QK8_1), part);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < MOE_ROWS_PER_WARP; ++r) {
    gate_sums[r] = warp_reduce_sum(gate_sums[r]);
    if constexpr (gate)
      up_sums[r] = warp_reduce_sum(up_sums[r]);
    if (lane == 0) {
      if constexpr (gate) {
        output[slot * rows + row + r] =
            up_sums[r] * (gate_sums[r] / (1.0f + expf(-gate_sums[r])));
      } else {
        atomicAdd(output + row + r, gate_sums[r] * routing[slot]);
      }
    }
  }
}

#define MOE_GATE_LAUNCHER(TAG, TYPE)                                           \
  extern "C" void launch_moe_gemv_fused_gate_up_##TAG##_sm121(                 \
      const void *gate, const void *up, const void *input,                     \
      const unsigned int *ids, float *output, int, int, int, int, int, int,    \
      void *stream) {                                                          \
    const dim3 grid(MOE_INTERMEDIATE / (MOE_WARPS * MOE_ROWS_PER_WARP),        \
                    MOE_TOPK);                                                 \
    moe_decode_sm121<MoeQuant::TYPE><<<grid, dim3(MOE_WARP, MOE_WARPS), 0,     \
                                       static_cast<cudaStream_t>(stream)>>>(   \
        gate, up, static_cast<const block_q8_1 *>(input), ids, nullptr,        \
        output);                                                               \
  }

#define MOE_DOWN_LAUNCHER(TAG, TYPE)                                           \
  extern "C" void launch_moe_gemv_down_aggregate_##TAG##_sm121(                \
      const void *weight, const void *input, const unsigned int *ids,          \
      const float *routing, float *output, int, int, int, int, int,            \
      void *stream) {                                                          \
    const dim3 grid(MOE_HIDDEN / (MOE_WARPS * MOE_ROWS_PER_WARP), MOE_TOPK);   \
    moe_decode_sm121<MoeQuant::TYPE><<<grid, dim3(MOE_WARP, MOE_WARPS), 0,     \
                                       static_cast<cudaStream_t>(stream)>>>(   \
        weight, weight, static_cast<const block_q8_1 *>(input), ids, routing,  \
        output);                                                               \
  }

MOE_GATE_LAUNCHER(q4k, Q4K)
MOE_GATE_LAUNCHER(q5k, Q5K)
MOE_DOWN_LAUNCHER(q4_1, Q4_1)
MOE_DOWN_LAUNCHER(q5_1, Q5_1)
MOE_DOWN_LAUNCHER(q8_0, Q8_0)
