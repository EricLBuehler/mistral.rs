use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, Device, Result, Tensor};

use crate::{
    get_immediate_isq, pending_layer, ImmediateIsqMatch, ImmediateIsqParams, IsqConsumer,
    IsqRequest, IsqType, PendingIsqLayer, QuantMethod, ShardedVarBuilder, TrackedModule,
};

pub enum QuantizationBehavior {
    Quantize(GgmlDType),
    Skip,
}

pub fn apply_immediate_isq(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_sharded(layer, vb, Some(crate::Shard::default()))
}

/// Like [`apply_immediate_isq`], recording the rank slice so from-source requantization can
/// re-slice; pass None when the load applied a transform a shard cannot express.
pub fn apply_immediate_isq_sharded(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_inner(layer, vb, None, shard)
}

pub fn apply_immediate_isq_with_key(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    key: Option<String>,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    apply_immediate_isq_inner(layer, vb, key, shard)
}

fn apply_immediate_isq_inner(
    layer: Arc<dyn QuantMethod>,
    vb: ShardedVarBuilder,
    key: Option<String>,
    shard: Option<crate::Shard>,
) -> Result<Arc<dyn QuantMethod>> {
    let Some(params) = get_immediate_isq() else {
        return Ok(layer);
    };
    let prefix = format!("{}.weight", vb.prefix());
    if let Some(ImmediateIsqMatch {
        ty,
        device,
        promote_default,
    }) = crate::resolve_immediate_isq(&params, &prefix)
    {
        let device = if params.capture == crate::IsqCaptureMode::CaptureAll {
            Device::Cpu
        } else {
            device.unwrap_or_else(|| vb.device().clone())
        };

        // Capture modes keep the layer unquantized; the resolved ty is recorded for later.
        let spawn_ty = match params.capture {
            crate::IsqCaptureMode::Immediate => ty,
            _ => None,
        };
        let module_key = key.unwrap_or_else(|| vb.prefix());
        let layer = spawn_pending_isq(layer, spawn_ty, device, &params, module_key.clone());
        vb.tracker().add_module(TrackedModule {
            key: module_key,
            ct: layer.clone(),
            ty,
            promote_default,
            shard,
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
    module_key: String,
) -> Arc<PendingIsqLayer> {
    let guard = params.guard.clone().with_module_key(module_key.clone());
    let request = IsqRequest {
        ty,
        device: device.clone(),
        has_imatrix: false,
        capture: params.capture,
        consumer: IsqConsumer::ImmediateLoad,
        module_key,
    };
    let rx = match layer.plan_isq(&request) {
        Ok(plan) => params.executor.submit(plan, request.consumer, move || {
            layer
                .clone()
                .apply_isq(ty, device, &AtomicUsize::new(0), None, guard)
        }),
        Err(e) => {
            let (tx, rx) = pending_layer::pending_isq_channel();
            let _ = tx.send(Err(e));
            rx
        }
    };
    Arc::new(PendingIsqLayer::new(rx))
}

/// In-flight parallel requantization; receivers are in the same order as the input modules.
/// Holds the pool so spawned jobs outlive the call.
pub struct RequantizeHandles {
    _executor: crate::IsqExecutor,
    pub receivers: Vec<pending_layer::IsqReceiver>,
}

/// Quantize a rebuilt `[E, out, in]` expert stack to `ty`: GGML types go slab-by-slab so each
/// expert can take its own importance vector; other types quantize the whole stack.
pub fn quantize_expert_stack(
    stack: Tensor,
    ty: IsqType,
    imatrix: Option<Vec<f32>>,
    device: &Device,
    guard: crate::QuantizeOntoGuard,
) -> Result<Arc<dyn QuantMethod>> {
    if candle_core::quantized::GgmlDType::try_from(ty).is_ok() {
        let w = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            ty,
            imatrix.as_deref(),
            device,
            guard,
        )?;
        return Ok(Arc::new(crate::GgufMatMul::from_qtensor(w, None)));
    }
    let unquant = Arc::new(crate::UnquantLinear::new(
        crate::QuantMethodConfig::Unquantized(candle_nn::Linear::new(stack, None)),
    )?) as Arc<dyn QuantMethod>;
    unquant.apply_isq(
        Some(ty),
        device.clone(),
        &AtomicUsize::new(0),
        imatrix,
        guard,
    )
}

/// Quantize every tracked module on an executor sized for `pool_ty`.
pub fn requantize_tracked(
    modules: &[TrackedModule],
    pool_ty: IsqType,
    ty_for: impl Fn(&TrackedModule) -> IsqType,
    imatrix_for: &dyn Fn(&str) -> Option<Vec<f32>>,
    consumer: IsqConsumer,
    extra_host_reserve_bytes: usize,
    report: Option<crate::QuantizationReport>,
) -> Result<RequantizeHandles> {
    let config = crate::IsqExecutorConfig::new(Some(pool_ty))
        .with_external_reserved_host_bytes(extra_host_reserve_bytes);
    let (executor, _) = crate::create_isq_executor(config);
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
        let device = layer.dtype_and_device().1;
        let mut guard = guard
            .clone()
            .with_module_key(module.key.clone())
            .with_requested(ty.to_string());
        if let Some(report) = &report {
            guard = guard.with_report(report.clone());
        }
        let request = IsqRequest {
            ty: Some(ty),
            device: device.clone(),
            has_imatrix: imatrix.is_some(),
            capture: crate::IsqCaptureMode::Immediate,
            consumer,
            module_key: module.key.clone(),
        };
        let plan = layer.plan_isq(&request)?;
        let rx = executor.submit(plan, consumer, move || {
            layer
                .clone()
                .apply_isq(Some(ty), device, &AtomicUsize::new(0), imatrix, guard)
        });
        receivers.push(rx);
    }
    Ok(RequantizeHandles {
        _executor: executor,
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

pub(crate) fn warn_skip_quantization(
    guard: Option<&crate::QuantizeOntoGuard>,
    module_key: Option<&str>,
    quant: Option<&str>,
    shape: &[usize],
    reason: &str,
) {
    if let Some(report) = guard.and_then(|guard| guard.report()) {
        report.record_skip(
            module_key.unwrap_or("<unknown>"),
            guard
                .and_then(|guard| guard.requested())
                .map(ToString::to_string)
                .or_else(|| quant.map(ToString::to_string)),
            shape.to_vec(),
            reason,
        );
        return;
    }

    let quant = quant.map(|quant| format!("{quant} ")).unwrap_or_default();
    match module_key {
        Some(module_key) => crate::log::once_log_warn(format!(
            "Skipping {quant} quantization of `{module_key}` with tensor shape {shape:?}: {reason}."
        )),
        None => crate::log::once_log_warn(format!(
            "Skipping {quant} quantization of tensor with shape {shape:?}: {reason}."
        )),
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq {
    ($tensor:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {{
        let quantization_behaviour =
            $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
        let dtype = match quantization_behaviour {
            $crate::utils::isq::QuantizationBehavior::Skip => {
                let shape = $tensor.dims().to_vec();
                let quant = format!("{:?}", $dtype);
                $crate::utils::isq::warn_skip_quantization(
                    Some(&$guard),
                    $guard.module_key(),
                    Some(&quant),
                    &shape,
                    "tensor is not quantizable",
                );
                GgmlDType::F32
            }
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

        Arc::new(candle_core::quantized::QTensor::new(
            qstorage,
            $tensor.shape(),
        )?)
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! generate_isq_imatrix {
    ($tensor:expr, $imatrix:expr, $device:expr, $dtype:expr, $n_quantized:expr, $guard:expr) => {{
        let quantization_behaviour =
            $crate::utils::isq::get_quantization_behaviour(&$tensor, $dtype);
        let dtype = match quantization_behaviour {
            $crate::utils::isq::QuantizationBehavior::Skip => {
                let shape = $tensor.dims().to_vec();
                let quant = format!("{:?}", $dtype);
                $crate::utils::isq::warn_skip_quantization(
                    Some(&$guard),
                    $guard.module_key(),
                    Some(&quant),
                    &shape,
                    "tensor is not quantizable",
                );
                GgmlDType::F32
            }
            $crate::utils::isq::QuantizationBehavior::Quantize(dtype) => {
                $n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                dtype
            }
        };

        // Quantize from a CPU copy: byte extraction from a device-resident QTensor races
        // in-flight device work, and quantization is CPU-bound anyway.
        let cpu_src = $tensor.to_device(&candle_core::Device::Cpu)?;
        // Fallback dtypes (legacy Q, F32) have no imatrix quantizer; quantize plainly.
        let initial = if matches!(
            dtype,
            GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
        ) {
            candle_core::quantized::QTensor::quantize_imatrix(&cpu_src, &$imatrix, dtype)?
        } else {
            candle_core::quantized::QTensor::quantize(&cpu_src, dtype)?
        };
        let data = initial.data()?;

        let _acquired_quantize_guard = $guard.acquire(&$device);
        let qstorage = candle_core::quantized::QStorage::from_data(data, &$device, dtype)?;

        Arc::new(candle_core::quantized::QTensor::new(
            qstorage,
            $tensor.shape(),
        )?)
    }};
}
