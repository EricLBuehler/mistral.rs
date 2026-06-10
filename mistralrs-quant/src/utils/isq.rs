use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};

use crate::{
    get_immediate_isq, pending_layer, ImmediateIsqMatch, ImmediateIsqParams, IsqType,
    PendingIsqLayer, QuantMethod, ShardedVarBuilder, TrackedModule,
};

pub enum QuantizationBehavior {
    Quantize(GgmlDType),
    Skip,
}

pub fn apply_immediate_isq(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_with_key(layer, vb, None)
}

pub fn apply_immediate_isq_with_key(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    key: Option<String>,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(params) = get_immediate_isq() else {
        return Ok(layer);
    };
    let prefix = format!("{}.weight", vb.prefix());
    if let Some(ImmediateIsqMatch { ty, device }) = crate::resolve_immediate_isq(&params, &prefix) {
        let device = device.unwrap_or_else(|| vb.device().clone());

        // Capture modes keep the layer unquantized; the resolved ty is recorded for later.
        let spawn_ty = match params.capture {
            crate::IsqCaptureMode::Immediate => ty,
            _ => None,
        };
        let layer = spawn_pending_isq(layer, spawn_ty, device, &params);
        vb.tracker().add_module(TrackedModule {
            key: key.unwrap_or_else(|| vb.prefix()),
            ct: layer.clone(),
            ty,
        });
        Ok(layer)
    } else {
        Ok(layer)
    }
}

pub(crate) fn spawn_pending_isq(
    layer: Arc<dyn QuantMethod>,
    ty: Option<IsqType>,
    device: Device,
    params: &ImmediateIsqParams,
) -> Arc<PendingIsqLayer> {
    params.backpressure.acquire();
    let backpressure = params.backpressure.clone();
    let guard = params.guard.clone();
    let (tx, rx) = pending_layer::pending_isq_channel();
    params.pool.spawn(move || {
        let result = layer
            .clone()
            .apply_isq(ty, device, &AtomicUsize::new(0), None, guard);
        let _ = tx.send(result);
        backpressure.release();
    });
    Arc::new(PendingIsqLayer::new(rx))
}

/// In-flight parallel requantization; receivers are in the same order as the input modules.
/// Holds the pool so spawned jobs outlive the call.
pub struct RequantizeHandles {
    _pool: rayon::ThreadPool,
    pub receivers: Vec<pending_layer::IsqReceiver>,
}

/// Where requantized layers should live.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RequantizeResults {
    /// On each module's device, ready to swap into the live model (imatrix, re-ISQ).
    Resident,
    /// Raw-block types stage on CPU so their serialized bytes are plain memory (UQFF writes)
    CpuStaged,
}

/// Quantize every tracked module on a fresh pool sized for `pool_ty`. The per-module type is
/// the caller's policy: `|m| m.ty.unwrap_or(default)` honors the load-time plan (topology pins),
/// `|_| ty` forces a uniform type.
pub fn requantize_tracked(
    modules: &[TrackedModule],
    pool_ty: IsqType,
    results: RequantizeResults,
    ty_for: impl Fn(&TrackedModule) -> IsqType,
    imatrix_for: &dyn Fn(&str) -> Option<Vec<f32>>,
) -> Result<RequantizeHandles> {
    let (pool, _) = crate::create_isq_thread_pool(Some(pool_ty));
    let guard = crate::QuantizeOntoGuard::new();
    let mut receivers = Vec::with_capacity(modules.len());
    for module in modules {
        let layer = module.ct.resolve()?;
        let ty = ty_for(module);
        let imatrix = if ty.supports_imatrix() {
            imatrix_for(&module.key)
        } else {
            if imatrix_for(&module.key).is_some() {
                crate::log::once_log_warn(format!(
                    "{ty} does not consume imatrix weights; quantizing without them."
                ));
            }
            None
        };
        // Types convertible to GgmlDType quantize into raw blocks; everything else is tensor-backed.
        let device = if results == RequantizeResults::CpuStaged
            && candle_core::quantized::GgmlDType::try_from(ty).is_ok()
        {
            Device::Cpu
        } else {
            layer.dtype_and_device().1
        };
        let guard = guard.clone();
        let (tx, rx) = pending_layer::pending_isq_channel();
        pool.spawn(move || {
            let result =
                layer
                    .clone()
                    .apply_isq(Some(ty), device, &AtomicUsize::new(0), imatrix, guard);
            let _ = tx.send(result);
        });
        receivers.push(rx);
    }
    Ok(RequantizeHandles {
        _pool: pool,
        receivers,
    })
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

            // Quantize from a CPU copy: byte extraction from a device-resident QTensor races
            // in-flight device work, and quantization is CPU-bound anyway.
            let cpu_src = $tensor.to_device(&candle_core::Device::Cpu)?;
            let initial = candle_core::quantized::QTensor::quantize(&cpu_src, dtype)?;
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

            // Quantize from a CPU copy: byte extraction from a device-resident QTensor races
            // in-flight device work, and quantization is CPU-bound anyway.
            let cpu_src = $tensor.to_device(&candle_core::Device::Cpu)?;
            // Fallback dtypes (legacy Q, F32) have no imatrix quantizer; quantize plainly.
            let initial = if matches!(dtype, GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K) {
                candle_core::quantized::QTensor::quantize_imatrix(&cpu_src, &$imatrix, dtype)?
            } else {
                candle_core::quantized::QTensor::quantize(&cpu_src, dtype)?
            };
            let data = initial.data()?;

            let _acquired_quantize_guard = $guard.acquire(&$device);
            let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

            Arc::new(candle_core::quantized::QTensor::new(qstorage, $tensor.shape())?)
        }
    };
}
