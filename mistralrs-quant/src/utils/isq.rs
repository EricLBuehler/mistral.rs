use candle_core::{quantized::GgmlDType, Tensor};

pub enum QuantizationBehaviour {
    Quantize(GgmlDType),
    Skip,
}

/// Return the fallback dtype for the given dtype.
fn get_fallback(dtype: GgmlDType) -> QuantizationBehaviour {
    // The normal `Q` quants are a bit more lenient than the `K` quants.
    // => Try to fallback to a similar `Q` quant.
    // If that's not possible, skip this tensor.
    match dtype {
        GgmlDType::Q2K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q3K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q4K => QuantizationBehaviour::Quantize(GgmlDType::Q4_1),
        GgmlDType::Q5K => QuantizationBehaviour::Quantize(GgmlDType::Q5_0),
        GgmlDType::Q6K => QuantizationBehaviour::Quantize(GgmlDType::Q5_1),
        GgmlDType::Q8K => QuantizationBehaviour::Quantize(GgmlDType::Q8_1),
        _ => QuantizationBehaviour::Skip,
    }
}

/// Check if the tensor can be quantized with the given dtype.
fn can_quantize(tensor: &Tensor, dtype: GgmlDType) -> bool {
    let dims = tensor.shape().dims();
    // The tensor must not be empty and the last dimension must be a multiple of the block size.
    !(dims.is_empty() || (dims[dims.len() - 1] % dtype.block_size() != 0))
}

/// Check if we should quantize the tensor and if so, with which dtype.
pub(crate) fn get_quantization_behaviour(
    tensor: &Tensor,
    dtype: GgmlDType,
) -> QuantizationBehaviour {
    if dtype == GgmlDType::F32 {
        return QuantizationBehaviour::Skip;
    }

    if can_quantize(tensor, dtype) {
        return QuantizationBehaviour::Quantize(dtype);
    }
    let fallback = get_fallback(dtype);
    match fallback {
        QuantizationBehaviour::Skip => fallback,
        QuantizationBehaviour::Quantize(new_dtype) => get_quantization_behaviour(tensor, new_dtype),
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq {
    ($tensor:expr, $device:expr, $dtype:expr, $n_quantized:expr) => {
        {
            let quantization_behaviour = $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
            match quantization_behaviour{
                $crate::utils::isq::QuantizationBehaviour::Skip => {
                    let shape = $tensor.shape();
                    tracing::warn!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable.");
                    Arc::new(candle_core::quantized::QTensor::quantize_onto(&$tensor, GgmlDType::F32, &$device)?)
                },
                $crate::utils::isq::QuantizationBehaviour::Quantize(dtype) => {
                    $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Arc::new(candle_core::quantized::QTensor::quantize_onto(&$tensor, dtype, &$device)?)
                }
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq_imatrix {
    ($tensor:expr, $imatrix:expr, $device:expr, $dtype:expr, $n_quantized:expr) => {
        {
            let quantization_behaviour = $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
            match quantization_behaviour{
                $crate::utils::isq::QuantizationBehaviour::Skip => {
                    let shape = $tensor.shape();
                    tracing::warn!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable.");
                    if $tensor.device().is_cpu() {
                        Arc::new(candle_core::quantized::QTensor::quantize_imatrix_onto(&$tensor, &$imatrix, GgmlDType::F32, &$device)?)
                    } else {
                        Arc::new(candle_core::quantized::QTensor::quantize_imatrix(&$tensor, &$imatrix, GgmlDType::F32)?)
                    }
                },
                $crate::utils::isq::QuantizationBehaviour::Quantize(dtype) => {
                    $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if $tensor.device().is_cpu() {
                        Arc::new(candle_core::quantized::QTensor::quantize_imatrix_onto(&$tensor, &$imatrix, dtype, &$device)?)
                    } else {
                        Arc::new(candle_core::quantized::QTensor::quantize_imatrix(&$tensor, &$imatrix, dtype)?)
                    }
                }
            }
        }
    };
}
