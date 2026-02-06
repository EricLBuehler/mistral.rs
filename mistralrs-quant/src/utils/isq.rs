use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};

use crate::{
    get_immediate_isq, ImmediateIsqMatch, ImmediateIsqParams, QuantMethod, ShardedVarBuilder,
};

pub enum QuantizationBehavior {
    Quantize(GgmlDType),
    Skip,
}

pub fn apply_immediate_isq(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(params) = get_immediate_isq() else {
        return Ok(layer);
    };
    let prefix = format!("{}.weight", vb.prefix());
    if let Some(ImmediateIsqMatch { ty, device }) = crate::resolve_immediate_isq(&params, &prefix) {
        let device = device.unwrap_or_else(|| vb.device().clone());
        tracing::info!("Immediate ISQ: quantizing `{prefix}` to {ty:?} on {device:?}");
        let result = layer.clone().apply_isq(
            Some(ty),
            device,
            &AtomicUsize::new(0),
            None,
            params.guard.clone(),
        );
        tracing::info!("Immediate ISQ: done `{prefix}`");
        result
    } else {
        Ok(layer)
    }
}

pub(crate) fn apply_immediate_isq_always(
    layer: Arc<dyn QuantMethod>,
    device: &Device,
) -> Result<Arc<dyn QuantMethod>> {
    if let Some(ImmediateIsqParams {
        guard,
        ty: Some(immediate_isq),
        predicates: _,
        ..
    }) = get_immediate_isq()
    {
        layer.clone().apply_isq(
            Some(immediate_isq),
            device.clone(),
            &AtomicUsize::new(0),
            None,
            guard,
        )
    } else {
        Ok(layer)
    }
}

/// Return the fallback dtype for the given dtype.
fn get_fallback(dtype: GgmlDType) -> QuantizationBehavior {
    // The normal `Q` quants are a bit more lenient than the `K` quants.
    // => Try to fallback to a similar `Q` quant.
    // If that's not possible, skip this tensor.
    match dtype {
        GgmlDType::Q2K => QuantizationBehavior::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q3K => QuantizationBehavior::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q4K => QuantizationBehavior::Quantize(GgmlDType::Q4_1),
        GgmlDType::Q5K => QuantizationBehavior::Quantize(GgmlDType::Q5_0),
        GgmlDType::Q6K => QuantizationBehavior::Quantize(GgmlDType::Q5_1),
        GgmlDType::Q8K => QuantizationBehavior::Quantize(GgmlDType::Q8_1),
        _ => QuantizationBehavior::Skip,
    }
}

/// Check if the tensor can be quantized with the given dtype.
fn can_quantize(tensor: &Tensor, dtype: GgmlDType) -> bool {
    let dims = tensor.shape().dims();
    // The tensor must not be empty and the last dimension must be a multiple of the block size.
    !dims.is_empty() && dims[dims.len() - 1].is_multiple_of(dtype.block_size())
}

/// Check if we should quantize the tensor and if so, with which dtype.
pub(crate) fn get_quantization_behaviour(
    tensor: &Tensor,
    dtype: GgmlDType,
) -> QuantizationBehavior {
    if dtype == GgmlDType::F32 {
        return QuantizationBehavior::Skip;
    }

    if can_quantize(tensor, dtype) {
        return QuantizationBehavior::Quantize(dtype);
    }
    let fallback = get_fallback(dtype);
    match fallback {
        QuantizationBehavior::Skip => fallback,
        QuantizationBehavior::Quantize(new_dtype) => get_quantization_behaviour(tensor, new_dtype),
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq {
    ($tensor:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {
        {
            let quantization_behaviour = $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
            let dtype = match quantization_behaviour{
                $crate::utils::isq::QuantizationBehavior::Skip => {
                    let shape = $tensor.shape();
                    $crate::log::once_log_warn(&format!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable."));
                    GgmlDType::F32
                },
                $crate::utils::isq::QuantizationBehavior::Quantize(dtype) => {
                    $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    dtype
                }
            };

            let initial = candle_core::quantized::QTensor::quantize(&$tensor, dtype)?;
            let data = initial.data()?;

            let _acquired_quantize_guard = $guard.acquire(&$device);
            let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

            Arc::new(candle_core::quantized::QTensor::new(qstorage, $tensor.shape())?)
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq_imatrix {
    ($tensor:expr, $imatrix:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {
        {
            let quantization_behaviour = $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
            let dtype = match quantization_behaviour{
                $crate::utils::isq::QuantizationBehavior::Skip => {
                    let shape = $tensor.shape();
                    $crate::log::once_log_warn(&format!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable."));
                    GgmlDType::F32
                },
                $crate::utils::isq::QuantizationBehavior::Quantize(dtype) => {
                    $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    dtype
                }
            };

            let initial = candle_core::quantized::QTensor::quantize_imatrix(&$tensor, &$imatrix, dtype)?;
            if !$tensor.device().is_cpu() {
                // Short-circuit here, no need for fancy
                Arc::new(initial)
            } else {
                let data = initial.data()?;

                let _acquired_quantize_guard = $guard.acquire(&$device);
                let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

                Arc::new(candle_core::quantized::QTensor::new(qstorage, $tensor.shape())?)
            }
        }
    };
}
