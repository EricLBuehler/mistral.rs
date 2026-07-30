#include <cuda_runtime.h>
#include <stdint.h>

extern "C" int cuda_graph_copy_bytes(const void *src, void *dst, int64_t n, int64_t stream) {
    if (n < 0) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    return static_cast<int>(cudaMemcpyAsync(
        dst, src, static_cast<size_t>(n), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
}
