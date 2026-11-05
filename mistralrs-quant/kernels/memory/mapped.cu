#include <cuda_runtime.h>
#include <stdint.h>

constexpr size_t PREFAULT_STRIDE_BYTES = 4096;
constexpr int PREFAULT_THREADS = 256;
constexpr int PREFAULT_MAX_BLOCKS = 4096;

__global__ void prefault_mapped_pages(const uint8_t *data, size_t bytes) {
  const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t offset = thread * PREFAULT_STRIDE_BYTES; offset < bytes;
       offset += size_t(blockDim.x) * gridDim.x * PREFAULT_STRIDE_BYTES) {
    const volatile uint8_t *address = data + offset;
    (void)*address;
  }
  if (thread == 0) {
    const volatile uint8_t *last = data + bytes - 1;
    (void)*last;
  }
}

extern "C" int mapped_memory_prefault(const uint8_t *data, size_t bytes,
                                      void *stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  const size_t pages =
      (bytes + PREFAULT_STRIDE_BYTES - 1) / PREFAULT_STRIDE_BYTES;
  const size_t requested = (pages + PREFAULT_THREADS - 1) / PREFAULT_THREADS;
  const int blocks =
      requested < PREFAULT_MAX_BLOCKS ? requested : PREFAULT_MAX_BLOCKS;
  prefault_mapped_pages<<<blocks, PREFAULT_THREADS, 0, s>>>(data, bytes);
  const auto launch_status = cudaGetLastError();
  const auto sync_status = cudaStreamSynchronize(s);
  return static_cast<int>(launch_status == cudaSuccess ? sync_status
                                                       : launch_status);
}
