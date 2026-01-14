use candle_core::{DType, Result, Tensor};

use crate::scalar_fp8::ops::fp8_to_dtype;

/// Per-tensor FP8 dequantization: result = weight_fp8 * scale_inv
///
/// For per-tensor quantization, the scale is a single scalar that applies
/// to the entire weight tensor.
pub fn fp8_pertensor_dequantize(
    weight: &Tensor,
    scale_inv: &Tensor,
    out_dtype: DType,
) -> Result<Tensor> {
    // Convert FP8 weight to F32 using custom CUDA kernel
    let weight_f32 = fp8_to_dtype(weight, DType::F32)?;
    let scale_inv_f32 = scale_inv.to_dtype(DType::F32)?;
    // Multiply and convert to output dtype
    (weight_f32.broadcast_mul(&scale_inv_f32))?.to_dtype(out_dtype)
}

/// Per-tensor FP8 quantization: result = x / scale (clamped to FP8 range)
///
/// Quantizes input tensor using a static activation scale.
#[allow(dead_code)]
pub fn fp8_pertensor_quantize(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let scale_f32 = scale.to_dtype(DType::F32)?;
    let x_scaled = x_f32.broadcast_div(&scale_f32)?;
    // Clamp to FP8 E4M3 range: [-448, 448]
    let clamped = x_scaled.clamp(-448.0f64, 448.0f64)?;
    clamped.to_dtype(DType::F8E4M3)
}
