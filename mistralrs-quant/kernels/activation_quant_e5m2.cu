#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h> // For FLT_MAX
#include <cuda_fp8.h> // For __nv_fp8_e5m2

// E5M2_MAX: The maximum representable value for E5M2.
// __NV_FP8_E5M2_MAX is 57344.0f (available in cuda_fp8.h from CUDA 11.8+)
// If using an older toolkit, this might need to be defined manually.
// For safety, let's use the direct value if the define isn't available for the toolchain.
#ifndef __NV_FP8_E5M2_MAX__
#define FP8_E5M2_MAX_VAL 57344.0f
#else
#define FP8_E5M2_MAX_VAL __NV_FP8_E5M2_MAX__
#endif

template <typename T_IN, typename T_SCALE>
__global__ void quantize_per_token_e5m2_kernel_impl(
    const T_IN* __restrict__ activations,          // Input activations (FP16 or BF16)
    __nv_fp8_e5m2* __restrict__ out_quantized_act, // Output E5M2 activations
    T_SCALE* __restrict__ out_scales,              // Output per-token scales (FP16 or BF16)
    int num_tokens,                   // Number of tokens
    int token_size)                   // Size of each token (features per token)
{
    // Grid: num_tokens, Block: threads_per_token (e.g., 256, 512)
    // Each block processes one token.
    int token_idx = blockIdx.x;

    if (token_idx >= num_tokens) {
        return;
    }

    const T_IN* current_token_activations = activations + token_idx * token_size;
    __nv_fp8_e5m2* current_out_quant_act = out_quantized_act + token_idx * token_size;

    extern __shared__ float shared_mem[]; // Shared memory for reduction

    // 1. Find absmax for the current token using parallel reduction within the block
    float thread_abs_max = 0.0f;
    for (int i = threadIdx.x; i < token_size; i += blockDim.x) {
        float val = static_cast<float>(current_token_activations[i]);
        thread_abs_max = max(thread_abs_max, fabsf(val));
    }
    shared_mem[threadIdx.x] = thread_abs_max;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // The final absmax is in shared_mem[0] for this token
    if (threadIdx.x == 0) {
        float final_token_abs_max = shared_mem[0];

        // 2. Calculate scale
        T_SCALE scale = (final_token_abs_max == 0.0f) ? static_cast<T_SCALE>(1.0f) : static_cast<T_SCALE>(FP8_E5M2_MAX_VAL / final_token_abs_max);
        out_scales[token_idx] = scale;

        // Store scale in shared memory for all threads in block to use for quantization
        shared_mem[0] = static_cast<float>(scale);
    }
    __syncthreads(); // Ensure scale is visible to all threads

    float common_scale = shared_mem[0]; // All threads read the common scale

    // 3. Quantize values in parallel
    for (int i = threadIdx.x; i < token_size; i += blockDim.x) {
        float val_fp32 = static_cast<float>(current_token_activations[i]);
        val_fp32 *= common_scale;
        // The __nv_fp8_e5m2 constructor handles rounding and saturation.
        current_out_quant_act[i] = __nv_fp8_e5m2(val_fp32);
    }
}

extern "C" __global__ void quantize_fp16_to_e5m2_per_token_kernel(
    const __half* activations,
    __nv_fp8_e5m2* out_quantized_act,
    __half* out_scales,
    int num_tokens,
    int token_size)
{
    quantize_per_token_e5m2_kernel_impl<__half, __half>(
        activations, out_quantized_act, out_scales, num_tokens, token_size);
}

extern "C" __global__ void quantize_bf16_to_e5m2_per_token_kernel(
    const __nv_bfloat16* activations,
    __nv_fp8_e5m2* out_quantized_act,
    __nv_bfloat16* out_scales,
    int num_tokens,
    int token_size)
{
    quantize_per_token_e5m2_kernel_impl<__nv_bfloat16, __nv_bfloat16>(
        activations, out_quantized_act, out_scales, num_tokens, token_size);
}

// It might also be useful to have versions where T_SCALE is always __half (FP16)
// regardless of T_IN, as scales are often preferred in FP16.
extern "C" __global__ void quantize_bf16_to_e5m2_fp16scales_kernel(
    const __nv_bfloat16* activations,
    __nv_fp8_e5m2* out_quantized_act,
    __half* out_scales, // Scales are FP16
    int num_tokens,
    int token_size)
{
    quantize_per_token_e5m2_kernel_impl<__nv_bfloat16, __half>( // Note T_SCALE is __half
        activations, out_quantized_act, out_scales, num_tokens, token_size);
}

```

Next, I will add the FFI definitions in Rust and update the build script. I'll create a new file `mistralrs-quant/src/kernels.rs` for the FFI definitions to keep things organized.
