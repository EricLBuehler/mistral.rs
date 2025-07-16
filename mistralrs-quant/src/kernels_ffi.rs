//! FFI bindings for custom CUDA kernels.

#[cfg(feature = "cuda")]
mod cuda_bindings {
    use candle_core::cuda::ffi; // For CUdeviceptr, cudaStream_t

    extern "C" {
        /// CUDA kernel to quantize FP16 activations to FP8 E5M2 per token.
        ///
        /// # Arguments
        /// * `activations`: Device pointer to input FP16 activations.
        /// * `out_quantized_act`: Device pointer to output E5M2 quantized activations.
        /// * `out_scales`: Device pointer to output FP16 per-token scales.
        /// * `num_tokens`: Number of tokens to process.
        /// * `token_size`: Number of features per token.
        /// * `stream`: CUDA stream for the operation.
        pub fn quantize_fp16_to_e5m2_per_token_kernel(
            activations: ffi::CUdeviceptr,      // const __half*
            out_quantized_act: ffi::CUdeviceptr, // __nv_fp8_e5m2*
            out_scales: ffi::CUdeviceptr,       // __half*
            num_tokens: i32,
            token_size: i32,
            stream: ffi::cudaStream_t,
        );

        /// CUDA kernel to quantize BF16 activations to FP8 E5M2 per token, with BF16 scales.
        ///
        /// # Arguments
        /// * `activations`: Device pointer to input BF16 activations.
        /// * `out_quantized_act`: Device pointer to output E5M2 quantized activations.
        /// * `out_scales`: Device pointer to output BF16 per-token scales.
        /// * `num_tokens`: Number of tokens to process.
        /// * `token_size`: Number of features per token.
        /// * `stream`: CUDA stream for the operation.
        pub fn quantize_bf16_to_e5m2_per_token_kernel(
            activations: ffi::CUdeviceptr,      // const __nv_bfloat16*
            out_quantized_act: ffi::CUdeviceptr, // __nv_fp8_e5m2*
            out_scales: ffi::CUdeviceptr,       // __nv_bfloat16*
            num_tokens: i32,
            token_size: i32,
            stream: ffi::cudaStream_t,
        );

        /// CUDA kernel to quantize BF16 activations to FP8 E5M2 per token, with FP16 scales.
        ///
        /// This is useful if FP16 is the preferred precision for scales regardless of input type.
        /// # Arguments
        /// * `activations`: Device pointer to input BF16 activations.
        /// * `out_quantized_act`: Device pointer to output E5M2 quantized activations.
        /// * `out_scales`: Device pointer to output FP16 per-token scales.
        /// * `num_tokens`: Number of tokens to process.
        /// * `token_size`: Number of features per token.
        /// * `stream`: CUDA stream for the operation.
        pub fn quantize_bf16_to_e5m2_fp16scales_kernel(
            activations: ffi::CUdeviceptr,      // const __nv_bfloat16*
            out_quantized_act: ffi::CUdeviceptr, // __nv_fp8_e5m2*
            out_scales: ffi::CUdeviceptr,       // __half* (FP16 scales)
            num_tokens: i32,
            token_size: i32,
            stream: ffi::cudaStream_t,
        );
    }
}

#[cfg(feature = "cuda")]
pub use cuda_bindings::*;
