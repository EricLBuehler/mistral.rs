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

extern "C" int cuda_graph_copy_2d_bytes(const void *src, void *dst, int64_t width,
                                         int64_t height, int64_t src_pitch,
                                         int64_t dst_pitch, int64_t stream) {
    if (width < 0 || height < 0 || src_pitch < width || dst_pitch < width) {
        return 1;
    }
    if (width == 0 || height == 0) {
        return 0;
    }
    return static_cast<int>(cudaMemcpy2DAsync(
        dst, static_cast<size_t>(dst_pitch), src, static_cast<size_t>(src_pitch),
        static_cast<size_t>(width), static_cast<size_t>(height), cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(stream)));
}
