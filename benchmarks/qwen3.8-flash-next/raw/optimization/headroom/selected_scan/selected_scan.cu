#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>

constexpr unsigned THREADS = 256;
constexpr unsigned WARPS = THREADS / 32;
constexpr size_t MIN_CHUNK = 8 * 1024;
constexpr size_t MAX_CHUNK = 256 * 1024;
constexpr size_t MIN_FLUSH = 64 * 1024 * 1024;
constexpr unsigned FLUSH_BLOCKS = 1024;

struct ScanContext {
    void *weights[3] = {};
    size_t expert_bytes[3] = {};
    uint32_t *selected = nullptr;
    uint4 *checksums = nullptr;
    uint4 *flush = nullptr;
    size_t checksum_stride = 0;
    size_t flush_bytes = 0;
    int experts = 0;
    int count = 0;
    unsigned epoch = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

__device__ __forceinline__ uint4 xor4(uint4 a, uint4 b) {
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

__device__ __forceinline__ uint4 warp_xor(uint4 value) {
    for (unsigned delta = 16; delta; delta >>= 1) {
        value.x ^= __shfl_down_sync(0xffffffff, value.x, delta);
        value.y ^= __shfl_down_sync(0xffffffff, value.y, delta);
        value.z ^= __shfl_down_sync(0xffffffff, value.z, delta);
        value.w ^= __shfl_down_sync(0xffffffff, value.w, delta);
    }
    return value;
}

__global__ void scan_selected_weights(const uint4 *__restrict__ weights,
                                     const uint32_t *__restrict__ selected,
                                     uint4 *__restrict__ checksums,
                                     size_t expert_vectors, size_t chunk_vectors) {
    const size_t begin = size_t(blockIdx.x) * chunk_vectors;
    const size_t length = min(chunk_vectors, expert_vectors - begin);
    const uint4 *source = weights + size_t(selected[blockIdx.y]) * expert_vectors + begin;
    uint4 value = make_uint4(0, 0, 0, 0);
    for (size_t index = threadIdx.x; index < length; index += blockDim.x) {
        value = xor4(value, source[index]);
    }
    value = warp_xor(value);
    __shared__ uint4 partial[WARPS];
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < WARPS ? partial[threadIdx.x] : make_uint4(0, 0, 0, 0);
        value = warp_xor(value);
        if (threadIdx.x == 0) checksums[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = value;
    }
}

__global__ void flush_cache_lines(uint4 *buffer, size_t vectors, unsigned epoch) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < vectors; index += size_t(gridDim.x) * blockDim.x) {
        const unsigned value = unsigned(index) ^ epoch;
        buffer[index] = make_uint4(value, value + 1, value + 2, value + 3);
    }
}

static cudaError_t launch_scan(ScanContext *context, int projection, size_t chunk) {
    const int first = projection < 0 ? 0 : projection;
    const int last = projection < 0 ? 3 : projection + 1;
    for (int p = first; p < last; ++p) {
        const size_t blocks = (context->expert_bytes[p] + chunk - 1) / chunk;
        scan_selected_weights<<<dim3(unsigned(blocks), unsigned(context->count)), THREADS, 0, context->stream>>>(
            static_cast<const uint4 *>(context->weights[p]), context->selected,
            context->checksums + p * context->checksum_stride,
            context->expert_bytes[p] / sizeof(uint4), chunk / sizeof(uint4));
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }
    return cudaSuccess;
}

extern "C" const char *scan_error(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}

extern "C" int scan_destroy(ScanContext *context) {
    if (!context) return 0;
    cudaError_t first = cudaSuccess;
    const auto record = [&first](cudaError_t error) { if (first == cudaSuccess) first = error; };
    if (context->stream) record(cudaStreamSynchronize(context->stream));
    for (void *weight : context->weights) if (weight) record(cudaFree(weight));
    if (context->selected) record(cudaFree(context->selected));
    if (context->checksums) record(cudaFree(context->checksums));
    if (context->flush) record(cudaFree(context->flush));
    if (context->start) record(cudaEventDestroy(context->start));
    if (context->stop) record(cudaEventDestroy(context->stop));
    if (context->stream) record(cudaStreamDestroy(context->stream));
    delete context;
    return int(first);
}

extern "C" int scan_create(const void *const *sources, const size_t *expert_bytes,
                           int experts, ScanContext **out, uint64_t *info) {
    if (!out || experts <= 0 || experts > 512) return int(cudaErrorInvalidValue);
    *out = nullptr;
    auto *context = new (std::nothrow) ScanContext;
    if (!context) return int(cudaErrorMemoryAllocation);
    const auto fail = [context](cudaError_t error) { scan_destroy(context); return int(error); };
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) return fail(error);
    info[0] = free_bytes; info[1] = total_bytes;
    int l2 = 0;
    error = cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
    if (error != cudaSuccess) return fail(error);
    info[2] = uint64_t(l2);
    context->flush_bytes = std::max(MIN_FLUSH, size_t(l2) * 4);
    info[3] = context->flush_bytes;
    context->experts = experts;
    size_t largest = 0;
    for (int p = 0; p < 3; ++p) {
        if (expert_bytes[p] == 0 || expert_bytes[p] % sizeof(uint4)) return fail(cudaErrorInvalidValue);
        context->expert_bytes[p] = expert_bytes[p];
        largest = std::max(largest, expert_bytes[p]);
    }
    context->checksum_stride = size_t(experts) * ((largest + MIN_CHUNK - 1) / MIN_CHUNK);
    error = cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) return fail(error);
    error = cudaEventCreate(&context->start);
    if (error != cudaSuccess) return fail(error);
    error = cudaEventCreate(&context->stop);
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&context->selected, size_t(experts) * sizeof(uint32_t));
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&context->checksums, 3 * context->checksum_stride * sizeof(uint4));
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&context->flush, context->flush_bytes);
    if (error != cudaSuccess) return fail(error);
    for (int p = 0; p < 3; ++p) {
        error = cudaMalloc(&context->weights[p], size_t(experts) * expert_bytes[p]);
        if (error != cudaSuccess) return fail(error);
        error = cudaMemcpyAsync(context->weights[p], sources[p], size_t(experts) * expert_bytes[p], cudaMemcpyHostToDevice, context->stream);
        if (error != cudaSuccess) return fail(error);
    }
    error = cudaStreamSynchronize(context->stream);
    if (error != cudaSuccess) return fail(error);
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) return fail(error);
    info[4] = free_bytes;
    *out = context;
    return 0;
}

extern "C" int scan_select(ScanContext *context, const uint32_t *experts, int count) {
    if (!context || count <= 0 || count > context->experts) return int(cudaErrorInvalidValue);
    for (int i = 0; i < count; ++i) {
        if (experts[i] >= unsigned(context->experts) || (i && experts[i] <= experts[i - 1])) return int(cudaErrorInvalidValue);
    }
    context->count = count;
    return int(cudaMemcpyAsync(context->selected, experts, size_t(count) * sizeof(uint32_t), cudaMemcpyHostToDevice, context->stream));
}

extern "C" int scan_measure(ScanContext *context, int projection, size_t chunk,
                            int repeats, int evict, float *mean_ms) {
    if (!context || context->count == 0 || projection < -1 || projection > 2 ||
        chunk < MIN_CHUNK || chunk > MAX_CHUNK || chunk % sizeof(uint4) || repeats <= 0 || repeats > 1000)
        return int(cudaErrorInvalidValue);
    float total = 0;
    const int windows = evict ? repeats : 1;
    for (int window = 0; window < windows; ++window) {
        if (evict) {
            flush_cache_lines<<<FLUSH_BLOCKS, THREADS, 0, context->stream>>>(context->flush, context->flush_bytes / sizeof(uint4), ++context->epoch);
            const cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) return int(error);
        }
        cudaError_t error = cudaEventRecord(context->start, context->stream);
        if (error != cudaSuccess) return int(error);
        for (int repeat = 0; repeat < (evict ? 1 : repeats); ++repeat) {
            error = launch_scan(context, projection, chunk);
            if (error != cudaSuccess) return int(error);
        }
        error = cudaEventRecord(context->stop, context->stream);
        if (error != cudaSuccess) return int(error);
        error = cudaEventSynchronize(context->stop);
        if (error != cudaSuccess) return int(error);
        float elapsed = 0;
        error = cudaEventElapsedTime(&elapsed, context->start, context->stop);
        if (error != cudaSuccess) return int(error);
        total += elapsed;
    }
    *mean_ms = total / repeats;
    return 0;
}

extern "C" int scan_checksums(ScanContext *context, int projection, size_t chunk,
                              uint32_t *host_output, size_t words) {
    if (!context || context->count == 0 || projection < 0 || projection > 2 ||
        chunk < MIN_CHUNK || chunk > MAX_CHUNK || chunk % sizeof(uint4)) return int(cudaErrorInvalidValue);
    const size_t blocks = (context->expert_bytes[projection] + chunk - 1) / chunk;
    const size_t needed = size_t(context->count) * blocks * 4;
    if (words != needed) return int(cudaErrorInvalidValue);
    cudaError_t error = launch_scan(context, projection, chunk);
    if (error != cudaSuccess) return int(error);
    error = cudaMemcpyAsync(host_output, context->checksums + projection * context->checksum_stride, needed * sizeof(uint32_t), cudaMemcpyDeviceToHost, context->stream);
    if (error != cudaSuccess) return int(error);
    return int(cudaStreamSynchronize(context->stream));
}
