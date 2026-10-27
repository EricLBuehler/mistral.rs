#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int EMBEDDING_WARP = 32;
constexpr int EMBEDDING_WARPS = 8;
constexpr int IQ4_BLOCK = 32;
constexpr int IQ4_BYTES = 18;
constexpr int Q8_1_BYTES = 36;
constexpr int Q8K_BLOCK = 256;
constexpr int Q8K_BYTES = 292;
constexpr int Q8_1 = 9;
constexpr int Q8_K = 15;
constexpr int IQ4_NL = 20;

__device__ __constant__ int8_t embedding_iq4_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

template <int FORMAT>
__global__ void embedding_gather_extra(const uint8_t *data, const uint32_t *ids,
                                       float *out, size_t count,
                                       size_t columns) {
  const size_t lookup = blockIdx.x * EMBEDDING_WARPS + threadIdx.y;
  if (lookup >= count)
    return;
  constexpr int block_size = FORMAT == Q8_K ? Q8K_BLOCK : IQ4_BLOCK;
  constexpr int block_bytes =
      FORMAT == Q8_K ? Q8K_BYTES : (FORMAT == Q8_1 ? Q8_1_BYTES : IQ4_BYTES);
  const uint8_t *row =
      data + size_t(ids[lookup]) * (columns / block_size) * block_bytes;
  for (size_t column = threadIdx.x; column < columns;
       column += EMBEDDING_WARP) {
    const uint8_t *block = row + (column / block_size) * block_bytes;
    const int local = column % block_size;
    float value;
    if constexpr (FORMAT == IQ4_NL) {
      const uint8_t packed = block[2 + local % (IQ4_BLOCK / 2)];
      const int q = (packed >> (4 * (local / (IQ4_BLOCK / 2)))) & 15;
      value = __half2float(*reinterpret_cast<const half *>(block)) *
              embedding_iq4_values[q];
    } else if constexpr (FORMAT == Q8_1) {
      value = __half2float(*reinterpret_cast<const half *>(block)) *
              reinterpret_cast<const int8_t *>(block + 4)[local];
    } else {
      value = *reinterpret_cast<const float *>(block) *
              reinterpret_cast<const int8_t *>(block + 4)[local];
    }
    out[lookup * columns + column] = value;
  }
}

extern "C" void gguf_embedding_gather_extra(const uint8_t *data,
                                            const uint32_t *ids, float *out,
                                            size_t count, size_t columns,
                                            uint32_t format, void *stream) {
  const dim3 threads(EMBEDDING_WARP, EMBEDDING_WARPS);
  const int blocks = (count + EMBEDDING_WARPS - 1) / EMBEDDING_WARPS;
  const auto s = static_cast<cudaStream_t>(stream);
  switch (format) {
  case Q8_1:
    embedding_gather_extra<Q8_1>
        <<<blocks, threads, 0, s>>>(data, ids, out, count, columns);
    break;
  case Q8_K:
    embedding_gather_extra<Q8_K>
        <<<blocks, threads, 0, s>>>(data, ids, out, count, columns);
    break;
  case IQ4_NL:
    embedding_gather_extra<IQ4_NL>
        <<<blocks, threads, 0, s>>>(data, ids, out, count, columns);
    break;
  }
}
