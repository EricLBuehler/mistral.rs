#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kPackedRowsPerLaunch = 64;

struct PackedInputRows {
  const uint32_t *values[kPackedRowsPerLaunch];
};

__global__ void pack_completion_input_kernel(const uint32_t *host,
                                             PackedInputRows staged,
                                             uint32_t *output, int rows,
                                             int host_width,
                                             int staged_width) {
  const int64_t row_width = static_cast<int64_t>(host_width) + staged_width;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(rows) * row_width;
  if (index >= elements) {
    return;
  }
  const int row = static_cast<int>(index / row_width);
  const int column = static_cast<int>(index - row * row_width);
  output[index] = column < host_width
                      ? host[static_cast<int64_t>(row) * host_width + column]
                      : staged.values[row][column - host_width];
}

__global__ void pad_decode_input_kernel(const uint32_t *input,
                                        uint32_t *output, int input_rows,
                                        int output_rows, int width) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(output_rows) * width;
  if (index >= elements) {
    return;
  }
  const int row = static_cast<int>(index / width);
  const int column = static_cast<int>(index - static_cast<int64_t>(row) * width);
  const int source_row = row < input_rows ? row : 0;
  output[index] = input[static_cast<int64_t>(source_row) * width + column];
}

}

extern "C" int pack_completion_input_u32(
    const void *host, const void *const *staged_rows, void *output, int rows,
    int host_width, int staged_width, int64_t stream) {
  if (host == nullptr || staged_rows == nullptr || output == nullptr ||
      rows <= 0 || host_width <= 0 || staged_width <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const auto *host_values = static_cast<const uint32_t *>(host);
  auto *output_values = static_cast<uint32_t *>(output);
  for (int row_start = 0; row_start < rows;
       row_start += kPackedRowsPerLaunch) {
    const int remaining_rows = rows - row_start;
    const int chunk_rows = remaining_rows < kPackedRowsPerLaunch
                               ? remaining_rows
                               : kPackedRowsPerLaunch;
    PackedInputRows packed{};
    for (int row = 0; row < chunk_rows; ++row) {
      packed.values[row] =
          static_cast<const uint32_t *>(staged_rows[row_start + row]);
    }
    const int64_t row_width =
        static_cast<int64_t>(host_width) + staged_width;
    const int64_t elements = static_cast<int64_t>(chunk_rows) * row_width;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    pack_completion_input_kernel<<<blocks, threads, 0, cuda_stream>>>(
        host_values + static_cast<int64_t>(row_start) * host_width, packed,
        output_values + static_cast<int64_t>(row_start) * row_width,
        chunk_rows, host_width, staged_width);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

extern "C" int pad_decode_input_u32(const void *input, void *output,
                                      int input_rows, int output_rows,
                                      int width, int64_t stream) {
  if (input == nullptr || output == nullptr || input_rows <= 0 ||
      output_rows < input_rows || width <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = static_cast<int64_t>(output_rows) * width;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  pad_decode_input_kernel<<<blocks, threads, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t *>(input), static_cast<uint32_t *>(output),
      input_rows, output_rows, width);
  return static_cast<int>(cudaGetLastError());
}
