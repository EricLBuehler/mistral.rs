/*
 * Custom GEMV (General Matrix-Vector multiplication) CUDA Kernel
 *
 * Optimized for LLM decode-phase inference (batch_size=1).
 * Computes: y = A * x + bias where:
 *   - A: [M, K] weight matrix (row-major)
 *   - x: [K] input vector
 *   - bias: [M] optional bias vector
 *   - y: [M] output vector
 *
 * Design follows llama.cpp mmvf.cu approach:
 *   - Simple loop without heavy unrolling
 *   - Vectorized loads (float2, half2, nv_bfloat162)
 *   - __ldg() for read-only cache path
 *   - Warp-level reduction using XOR shuffle
 *   - Minimal register pressure for better occupancy
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

// Warp-level reduction sum using XOR shuffle (butterfly pattern)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Type conversion helpers
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
#endif

// ============================================================================
// Simple GEMV Kernel - follows llama.cpp approach
// ============================================================================

template <typename T, typename Vec2, int BLOCK_SIZE>
__global__ void gemv_kernel(
    const T* __restrict__ A,      // [M, K] weights (row-major)
    const T* __restrict__ x,      // [K] input vector
    const T* __restrict__ bias,   // [M] optional bias
    T* __restrict__ y,            // [M] output vector
    int M, int K, bool has_bias
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int K2 = K / 2;

    // Static shared memory for block-level reduction
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    // Direct pointers
    const Vec2* A_vec = reinterpret_cast<const Vec2*>(A + row * K);
    const Vec2* x_vec = reinterpret_cast<const Vec2*>(x);

    // Single accumulator - simple and efficient
    float acc = 0.0f;

    // Simple loop - let compiler handle scheduling
    for (int col2 = tid; col2 < K2; col2 += BLOCK_SIZE) {
        Vec2 a_val = __ldg(A_vec + col2);
        Vec2 x_val = __ldg(x_vec + col2);

        if constexpr (std::is_same_v<T, float>) {
            acc = fmaf(a_val.x, x_val.x, acc);
            acc = fmaf(a_val.y, x_val.y, acc);
        } else if constexpr (std::is_same_v<T, __half>) {
            float2 a_f = __half22float2(a_val);
            float2 x_f = __half22float2(x_val);
            acc = fmaf(a_f.x, x_f.x, acc);
            acc = fmaf(a_f.y, x_f.y, acc);
        }
#if __CUDA_ARCH__ >= 800
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            acc = fmaf(__bfloat162float(a_val.x), __bfloat162float(x_val.x), acc);
            acc = fmaf(__bfloat162float(a_val.y), __bfloat162float(x_val.y), acc);
        }
#endif
    }

    // Handle remainder if K is odd
    if (K % 2 != 0 && tid == 0) {
        int last_idx = K - 1;
        acc = fmaf(to_float(__ldg(A + row * K + last_idx)), to_float(__ldg(x + last_idx)), acc);
    }

    // Warp-level reduction
    acc = warp_reduce_sum(acc);

    // Block-level reduction via shared memory
    if constexpr (BLOCK_SIZE > WARP_SIZE) {
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
            warp_sums[warp_id] = acc;
        }
        __syncthreads();

        if (warp_id == 0) {
            acc = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
            acc = warp_reduce_sum(acc);
        }
    }

    // Thread 0 writes the final result
    if (tid == 0) {
        if (has_bias) {
            acc += to_float(__ldg(bias + row));
        }

        if constexpr (std::is_same_v<T, float>) {
            y[row] = acc;
        } else if constexpr (std::is_same_v<T, __half>) {
            y[row] = __float2half(acc);
        }
#if __CUDA_ARCH__ >= 800
        else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            y[row] = __float2bfloat16(acc);
        }
#endif
    }
}

// ============================================================================
// Block Size Selection - minimize iterations
// ============================================================================

__host__ int get_optimal_block_size(int K) {
    // Following llama.cpp: choose block size to minimize iterations
    // iterations = ceil(K2 / block_size)
    const int K2 = K / 2;

    // For small K, use smaller blocks
    if (K2 <= 32) return 32;
    if (K2 <= 64) return 64;
    if (K2 <= 128) return 128;

    // For larger K, 256 threads gives good occupancy
    return 256;
}

// ============================================================================
// Launch Functions - BF16
// ============================================================================

extern "C" void launch_gemv_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    __nv_bfloat16* y,
    int M, int K,
    bool has_bias,
    cudaStream_t stream
) {
    int block_size = get_optimal_block_size(K);
    dim3 grid(M);
    dim3 block(block_size);

    switch (block_size) {
        case 32:
            gemv_kernel<__nv_bfloat16, __nv_bfloat162, 32>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 64:
            gemv_kernel<__nv_bfloat16, __nv_bfloat162, 64>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 128:
            gemv_kernel<__nv_bfloat16, __nv_bfloat162, 128>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 256:
            gemv_kernel<__nv_bfloat16, __nv_bfloat162, 256>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        default:
            gemv_kernel<__nv_bfloat16, __nv_bfloat162, 256>
                <<<grid, dim3(256), 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
    }
}

// ============================================================================
// Launch Functions - F16
// ============================================================================

extern "C" void launch_gemv_f16(
    const __half* A,
    const __half* x,
    const __half* bias,
    __half* y,
    int M, int K,
    bool has_bias,
    cudaStream_t stream
) {
    int block_size = get_optimal_block_size(K);
    dim3 grid(M);
    dim3 block(block_size);

    switch (block_size) {
        case 32:
            gemv_kernel<__half, half2, 32>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 64:
            gemv_kernel<__half, half2, 64>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 128:
            gemv_kernel<__half, half2, 128>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 256:
            gemv_kernel<__half, half2, 256>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        default:
            gemv_kernel<__half, half2, 256>
                <<<grid, dim3(256), 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
    }
}

// ============================================================================
// Launch Functions - F32
// ============================================================================

extern "C" void launch_gemv_f32(
    const float* A,
    const float* x,
    const float* bias,
    float* y,
    int M, int K,
    bool has_bias,
    cudaStream_t stream
) {
    int block_size = get_optimal_block_size(K);
    dim3 grid(M);
    dim3 block(block_size);

    switch (block_size) {
        case 32:
            gemv_kernel<float, float2, 32>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 64:
            gemv_kernel<float, float2, 64>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 128:
            gemv_kernel<float, float2, 128>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        case 256:
            gemv_kernel<float, float2, 256>
                <<<grid, block, 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
        default:
            gemv_kernel<float, float2, 256>
                <<<grid, dim3(256), 0, stream>>>(A, x, bias, y, M, K, has_bias);
            break;
    }
}
