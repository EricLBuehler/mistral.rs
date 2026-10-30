// E4M3 W8A16 mma.sync GEMM over moe_align-sorted routes or dense rows

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "../common/fp_codecs.cuh"

namespace {

constexpr int kBN = 128;
constexpr int kStages = 3;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = kWarpsM * kWarpsN * 32;
constexpr int kWarpN = kBN / kWarpsN;
constexpr int kNTiles = kWarpN / 8;
constexpr int kAPad = 8;
constexpr int kChunk = 16;
constexpr int kBankLine = 128;
constexpr int kGroupM = 8;
constexpr int kMmaK = 16;
constexpr int kStaticSmemLimit = 48 * 1024;
constexpr int kMxGroup = 32;

enum ScaleKind : int { kScaleF32 = 0, kScaleE8M0 = 1 };

struct Params {
  const void *a;
  const uint8_t *w;
  const void *scales;
  const int32_t *sorted_ids;
  const int32_t *expert_ids;
  const int32_t *num_post_pad;
  void *out;
  int n;
  int k;
  int em;
  int num_valid;
  int top_k;
  int scale_block_n;
  int scale_block_k;
  int scale_rows;
  int scale_cols;
};

template <typename T> struct Vec2;
template <> struct Vec2<__nv_bfloat16> {
  using type = __nv_bfloat162;
  static __device__ __forceinline__ type from(float2 v) {
    return __float22bfloat162_rn(v);
  }
};
template <> struct Vec2<half> {
  using type = half2;
  static __device__ __forceinline__ type from(float2 v) {
    return __float22half2_rn(v);
  }
};

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem,
                                           bool pred) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  int size = pred ? kChunk : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(addr),
               "l"(gmem), "r"(size));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

template <typename T>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    uint32_t b0, uint32_t b1) {
  if constexpr (std::is_same_v<T, half>) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                   "r"(b1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0),
                   "r"(b1));
  }
}

// XOR chunks by bank line so the 8 rows of a B fragment hit distinct banks
template <int BK> __device__ __forceinline__ int swizzle_b(int row, int col) {
  constexpr int chunks = BK / kChunk;
  constexpr int rows_per_line = kBankLine / BK > 0 ? kBankLine / BK : 1;
  return col ^ (((row / rows_per_line) % chunks) * kChunk);
}

template <int SK>
__device__ __forceinline__ float load_scale(const Params &p, size_t index) {
  if constexpr (SK == kScaleE8M0) {
    return e8m0_to_float_subnormal(
        __ldg(static_cast<const uint8_t *>(p.scales) + index));
  } else {
    return __ldg(static_cast<const float *>(p.scales) + index);
  }
}

template <typename T, bool EPI>
__device__ __forceinline__ uint32_t dequant_pair(const uint8_t *src,
                                                 float scale) {
  __nv_fp8x2_storage_t raw = *reinterpret_cast<const uint16_t *>(src);
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(raw, __NV_E4M3);
  typename Vec2<T>::type v;
  if constexpr (EPI && std::is_same_v<T, half>) {
    v = *reinterpret_cast<half2 *>(&h);
  } else {
    float2 f = __half22float2(*reinterpret_cast<half2 *>(&h));
    if constexpr (!EPI) {
      f.x *= scale;
      f.y *= scale;
    }
    v = Vec2<T>::from(f);
  }
  return *reinterpret_cast<uint32_t *>(&v);
}

template <typename T, int MT, int BK> constexpr int smem_bytes() {
  return kStages *
         (kWarpsM * MT * 16 * (BK + kAPad) * int(sizeof(T)) + kBN * BK);
}

template <typename T, int MT, int BK, bool EPI, int SK>
__global__ void __launch_bounds__(kThreads)
    w8a16_grouped_kernel(const Params p) {
  constexpr int BM = kWarpsM * MT * 16;
  constexpr int a_stride = BK + kAPad;
  constexpr int a_stage = BM * a_stride;
  constexpr int b_stage = kBN * BK;
  extern __shared__ __align__(16) uint8_t smem[];
  T *sa = reinterpret_cast<T *>(smem);
  uint8_t *sb = smem + kStages * a_stage * sizeof(T);
  __shared__ int s_route[BM];

  const int num_pid_m = (p.em + BM - 1) / BM;
  const int num_pid_n = (p.n + kBN - 1) / kBN;
  const int pid = blockIdx.x;
  const int in_group = kGroupM * num_pid_n;
  const int first_m = (pid / in_group) * kGroupM;
  const int group_rows = min(num_pid_m - first_m, kGroupM);
  const int pid_m = first_m + (pid % in_group) % group_rows;
  const int pid_n = (pid % in_group) / group_rows;
  const bool routed = p.sorted_ids != nullptr;
  if (routed && pid_m * BM >= __ldg(p.num_post_pad))
    return;
  const int expert = routed ? __ldg(p.expert_ids + pid_m) : 0;

  const int tid = threadIdx.x;
  for (int i = tid; i < BM; i += kThreads) {
    int index = pid_m * BM + i;
    int route = routed
                    ? (index < p.em ? __ldg(p.sorted_ids + index) : p.num_valid)
                    : index;
    s_route[i] = route < p.num_valid ? route : -1;
  }
  __syncthreads();

  const int warp = tid / 32;
  const int lane = tid % 32;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const int n_block = pid_n * kBN;
  const T *a = static_cast<const T *>(p.a);

  float acc[MT][kNTiles][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < kNTiles; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        acc[i][j][c] = 0.0f;

  if (expert >= 0) {
    const uint8_t *w = p.w + static_cast<size_t>(expert) * p.n * p.k;
    const int k_tiles = (p.k + BK - 1) / BK;
    constexpr int a_chunks_per_row = BK * sizeof(T) / kChunk;
    constexpr int b_chunks_per_row = BK / kChunk;
    size_t scale_row[kNTiles];
#pragma unroll
    for (int j = 0; j < kNTiles; ++j) {
      int n = min(n_block + warp_n * kWarpN + j * 8 + lane / 4, p.n - 1);
      scale_row[j] =
          (static_cast<size_t>(expert) * p.scale_rows + n / p.scale_block_n) *
          p.scale_cols;
    }
    // per-lane E8M0 rows are scattered, so prefetch them a K tile ahead
    constexpr bool kPrefetchMx = SK == kScaleE8M0 && !EPI;
    constexpr int kMxGroups = kPrefetchMx ? BK / kMxGroup : 1;
    uint8_t mx_cur[kNTiles][kMxGroups];
    uint8_t mx_next[kNTiles][kMxGroups];
    auto fetch_mx = [&](uint8_t(&dst)[kNTiles][kMxGroups], int k_tile) {
#pragma unroll
      for (int j = 0; j < kNTiles; ++j)
#pragma unroll
        for (int g = 0; g < kMxGroups; ++g) {
          int group = k_tile * kMxGroups + g;
          dst[j][g] = group < p.scale_cols
                          ? __ldg(static_cast<const uint8_t *>(p.scales) +
                                  scale_row[j] + group)
                          : 0;
        }
    };
    if constexpr (kPrefetchMx)
      fetch_mx(mx_cur, 0);

    auto load_stage = [&](int stage, int k_tile) {
      const int k0 = k_tile * BK;
      for (int c = tid; c < BM * a_chunks_per_row; c += kThreads) {
        int row = c / a_chunks_per_row;
        int col = (c % a_chunks_per_row) * (kChunk / sizeof(T));
        int route = s_route[row];
        bool pred = route >= 0 && k0 + col < p.k;
        const T *src =
            pred ? a + static_cast<size_t>(route / p.top_k) * p.k + k0 + col
                 : a;
        cp_async16(sa + stage * a_stage + row * a_stride + col, src, pred);
      }
      for (int c = tid; c < kBN * b_chunks_per_row; c += kThreads) {
        int row = c / b_chunks_per_row;
        int col = (c % b_chunks_per_row) * kChunk;
        int n = n_block + row;
        bool pred = n < p.n && k0 + col < p.k;
        const uint8_t *src =
            pred ? w + static_cast<size_t>(n) * p.k + k0 + col : p.w;
        cp_async16(sb + stage * b_stage + row * BK + swizzle_b<BK>(row, col),
                   src, pred);
      }
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < k_tiles)
        load_stage(s, s);
      cp_async_commit();
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
      cp_async_wait<kStages - 2>();
      __syncthreads();
      int next = kt + kStages - 1;
      if (next < k_tiles)
        load_stage(next % kStages, next);
      cp_async_commit();
      if constexpr (kPrefetchMx) {
        if (kt + 1 < k_tiles)
          fetch_mx(mx_next, kt + 1);
      }

      const int stage = kt % kStages;
      const T *sa_s = sa + stage * a_stage;
      const uint8_t *sb_s = sb + stage * b_stage;
#pragma unroll
      for (int kk = 0; kk < BK; kk += kMmaK) {
        uint32_t af[MT][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          int row = warp_m * MT * 16 + i * 16 + (lane % 16);
          int col = kk + (lane / 16) * 8;
          ldmatrix_x4(af[i], sa_s + row * a_stride + col);
        }
        // scale groups are multiples of the mma K
        int k_group = 0;
        bool k_live = true;
        if constexpr (!EPI && !kPrefetchMx) {
          int k = kt * BK + kk;
          k_live = k < p.k;
          k_group = k_live ? k / p.scale_block_k : 0;
        }
#pragma unroll
        for (int j = 0; j < kNTiles; ++j) {
          int n_local = warp_n * kWarpN + j * 8 + lane / 4;
          int k_local = kk + (lane % 4) * 2;
          const uint8_t *src =
              sb_s + n_local * BK + swizzle_b<BK>(n_local, k_local);
          float scale = 1.0f;
          if constexpr (kPrefetchMx) {
            scale = e8m0_to_float_subnormal(mx_cur[j][kk / kMxGroup]);
          } else if constexpr (!EPI) {
            scale = k_live ? load_scale<SK>(p, scale_row[j] + k_group) : 0.0f;
          }
          uint32_t b0 = dequant_pair<T, EPI>(src, scale);
          uint32_t b1 = dequant_pair<T, EPI>(src + 8, scale);
#pragma unroll
          for (int i = 0; i < MT; ++i)
            mma<T>(acc[i][j], af[i], b0, b1);
        }
      }
      if constexpr (kPrefetchMx) {
#pragma unroll
        for (int jj = 0; jj < kNTiles; ++jj)
#pragma unroll
          for (int g = 0; g < kMxGroups; ++g)
            mx_cur[jj][g] = mx_next[jj][g];
      }
    }
    cp_async_wait<0>();
  }

  T *out = static_cast<T *>(p.out);
  const bool paired = p.n % 2 == 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
      int row = warp_m * MT * 16 + i * 16 + half_row * 8 + lane / 4;
      int route = s_route[row];
      if (route < 0)
        continue;
#pragma unroll
      for (int j = 0; j < kNTiles; ++j) {
        int n = n_block + warp_n * kWarpN + j * 8 + (lane % 4) * 2;
        float v0 = acc[i][j][half_row * 2];
        float v1 = acc[i][j][half_row * 2 + 1];
        if constexpr (EPI) {
          size_t base =
              static_cast<size_t>(expert < 0 ? 0 : expert) * p.scale_rows;
          if (n < p.n)
            v0 *= load_scale<SK>(p, base + n / p.scale_block_n);
          if (n + 1 < p.n)
            v1 *= load_scale<SK>(p, base + (n + 1) / p.scale_block_n);
        }
        T *dst = out + static_cast<size_t>(route) * p.n + n;
        if (paired && n + 1 < p.n) {
          *reinterpret_cast<typename Vec2<T>::type *>(dst) =
              Vec2<T>::from(make_float2(v0, v1));
        } else {
          if (n < p.n)
            dst[0] = static_cast<T>(v0);
          if (n + 1 < p.n)
            dst[1] = static_cast<T>(v1);
        }
      }
    }
  }
}

template <typename T, int MT, int BK, bool EPI, int SK>
int launch_typed(const Params &p, cudaStream_t stream) {
  constexpr int BM = kWarpsM * MT * 16;
  constexpr int bytes = smem_bytes<T, MT, BK>();
  int blocks = ((p.em + BM - 1) / BM) * ((p.n + kBN - 1) / kBN);
  if (blocks == 0)
    return 0;
  auto kernel = w8a16_grouped_kernel<T, MT, BK, EPI, SK>;
  if constexpr (bytes > kStaticSmemLimit) {
    static const cudaError_t opt_in = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (opt_in != cudaSuccess)
      return static_cast<int>(opt_in);
  }
  kernel<<<blocks, kThreads, bytes, stream>>>(p);
  return static_cast<int>(cudaGetLastError());
}

template <typename T, int BK, bool EPI, int SK>
int launch_tiles(const Params &p, int block_m, cudaStream_t stream) {
  switch (block_m) {
  case 32:
    return launch_typed<T, 1, BK, EPI, SK>(p, stream);
  case 64:
    return launch_typed<T, 2, BK, EPI, SK>(p, stream);
  case 128:
    return launch_typed<T, 4, BK, EPI, SK>(p, stream);
  default:
    return static_cast<int>(cudaErrorInvalidValue);
  }
}

template <typename T, bool EPI, int SK>
int launch_k(const Params &p, int block_m, int block_k, cudaStream_t stream) {
  switch (block_k) {
  case 64:
    return launch_tiles<T, 64, EPI, SK>(p, block_m, stream);
  case 128:
    return launch_tiles<T, 128, EPI, SK>(p, block_m, stream);
  default:
    return static_cast<int>(cudaErrorInvalidValue);
  }
}

template <typename T>
int launch(const Params &p, int block_m, int block_k, int scale_kind,
           cudaStream_t stream) {
  bool epilogue = p.scale_cols == 1;
  if (scale_kind == kScaleE8M0) {
    return epilogue
               ? launch_k<T, true, kScaleE8M0>(p, block_m, block_k, stream)
               : launch_k<T, false, kScaleE8M0>(p, block_m, block_k, stream);
  }
  return epilogue ? launch_k<T, true, kScaleF32>(p, block_m, block_k, stream)
                  : launch_k<T, false, kScaleF32>(p, block_m, block_k, stream);
}

} // namespace

#define W8A16_EXPORT(T, SUFFIX)                                                \
  extern "C" int mistralrs_w8a16_grouped_##SUFFIX(                             \
      const void *a, const uint8_t *w, const void *scales,                     \
      const int32_t *sorted_ids, const int32_t *expert_ids,                    \
      const int32_t *num_post_pad, void *out, int n, int k, int em,            \
      int num_valid, int top_k, int scale_block_n, int scale_block_k,          \
      int scale_kind, int block_m, int block_k, cudaStream_t stream) {         \
    Params p{a,                                                                \
             w,                                                                \
             scales,                                                           \
             sorted_ids,                                                       \
             expert_ids,                                                       \
             num_post_pad,                                                     \
             out,                                                              \
             n,                                                                \
             k,                                                                \
             em,                                                               \
             num_valid,                                                        \
             top_k,                                                            \
             scale_block_n,                                                    \
             scale_block_k,                                                    \
             (n + scale_block_n - 1) / scale_block_n,                          \
             (k + scale_block_k - 1) / scale_block_k};                         \
    return launch<T>(p, block_m, block_k, scale_kind, stream);                 \
  }

W8A16_EXPORT(__nv_bfloat16, bf16)
W8A16_EXPORT(half, f16)
