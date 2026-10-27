use super::*;
use crate::{
    try_forward_with_shared_quantized_activation, try_fused_quantized_pair, DistributedKind,
    GgufMatMul, IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig,
    QuantizeOntoGuard, QuantizedSerde, UnquantLinear,
};
use candle_core::{cuda::cudarc::driver::sys, Var};
use std::sync::{atomic::AtomicUsize, Arc};

const COLUMNS: usize = 512;
const LEFT_ROWS: usize = 47;
const RIGHT_ROWS: usize = 65;
const MIXED_DENSE_ROWS: usize = 4;
const FORMATS: [GgmlDType; 10] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

fn layer(format: GgmlDType, rows: usize, device: &Device) -> Result<Arc<dyn QuantMethod>> {
    let values = (0..rows * COLUMNS)
        .map(|index| ((index * 37 + rows * 19) % 211) as f32 / 10500.0 - 0.01)
        .collect::<Vec<_>>();
    let weight = QTensor::quantize_onto(
        &Tensor::from_vec(values, (rows, COLUMNS), &Device::Cpu)?,
        format,
        device,
    )?;
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(weight),
        b: None,
    })?))
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

#[test]
#[ignore = "requires a CUDA device"]
fn shared_mmvq_replays_same_and_mixed_formats_with_workspace_reuse() -> Result<()> {
    for _ in 0..2 {
        let device = Device::new_cuda_with_stream(0)?;
        let cuda = device.as_cuda_device()?;
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();
        let stream = cuda.cuda_stream();
        for (format_index, format) in FORMATS.into_iter().enumerate() {
            let left = layer(format, LEFT_ROWS, &device)?;
            for right_format in [format, FORMATS[(format_index + 1) % FORMATS.len()]] {
                let right = layer(right_format, RIGHT_ROWS, &device)?;
                let methods = [left.as_ref(), right.as_ref()];
                let quantized = [left.get_qtensor().unwrap(), right.get_qtensor().unwrap()];
                let weights = quantized.iter().map(AsRef::as_ref).collect::<Vec<_>>();
                for dtype in [DType::F32, DType::F16, DType::BF16] {
                    for batch in [1, 2, 4, 8] {
                        let backing = Var::zeros((1, batch, COLUMNS + 2), dtype, &device)?;
                        let input = backing.narrow(2, 1, COLUMNS)?;
                        let other =
                            Tensor::full(0.125f32, input.shape(), &device)?.to_dtype(dtype)?;
                        let public_available =
                            try_forward_with_shared_quantized_activation(&input, &methods)?
                                .is_some();
                        if !public_available {
                            #[cfg(has_marlin_kernels)]
                            {
                                let mut prefers_affine = false;
                                for method in methods {
                                    prefers_affine |=
                                        method.prepare_gguf_affine_raw(batch, dtype, &device)?;
                                }
                                assert!(
                                    prefers_affine && batch >= super::super::GGUF_AFFINE_MIN_BATCH
                                );
                            }
                            #[cfg(not(has_marlin_kernels))]
                            panic!("supported MMVQ formats should share the input");
                        }
                        let project = |input: &Tensor| {
                            if public_available {
                                Ok(
                                    try_forward_with_shared_quantized_activation(input, &methods)?
                                        .expect("eligible formats"),
                                )
                            } else {
                                shared_lhs(&weights, input)
                            }
                        };
                        let run = || -> Result<Vec<Tensor>> {
                            let mut outputs = project(&input)?;
                            outputs.push(left.forward(&other)?);
                            outputs.extend(project(&other)?);
                            outputs.extend(project(&input)?);
                            Ok(outputs)
                        };
                        let _ = run()?;
                        device.synchronize()?;
                        let tracking = stream.context().is_event_tracking();
                        if tracking {
                            unsafe { stream.context().disable_event_tracking() };
                        }
                        if let Err(error) = stream
                            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                        {
                            if tracking {
                                unsafe { stream.context().enable_event_tracking() };
                            }
                            return Err(candle_core::Error::msg(error));
                        }
                        let outputs = run();
                        let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
                        if tracking {
                            unsafe { stream.context().enable_event_tracking() };
                        }
                        let outputs = outputs?;
                        let graph = graph
                            .map_err(candle_core::Error::msg)?
                            .expect("captured graph");
                        for factor in [0.0f32, 0.05, -0.09] {
                            let input_values = (0..batch * (COLUMNS + 2))
                                .map(|index| ((index * 7 % 19) as f32 - 9.0) * factor)
                                .collect::<Vec<_>>();
                            backing.set(
                                &Tensor::from_vec(input_values, (1, batch, COLUMNS + 2), &device)?
                                    .to_dtype(dtype)?,
                            )?;
                            graph.launch().map_err(candle_core::Error::msg)?;
                            for (projection, weight) in weights.iter().enumerate() {
                                let expected = plain(weight, &input)?;
                                for actual in [&outputs[projection], &outputs[projection + 5]] {
                                    assert_eq!(actual.dims(), expected.dims());
                                    assert_eq!(actual.dtype(), expected.dtype());
                                    assert_eq!(values(actual)?, values(&expected)?);
                                }
                                assert_eq!(
                                    values(&outputs[projection + 3])?,
                                    values(&plain(weight, &other)?)?
                                );
                            }
                            assert_eq!(values(&outputs[2])?, values(&left.forward(&other)?)?);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum BlockedCapability {
    Bias,
    Stats,
    Lora,
    Collective,
}

#[derive(Debug)]
struct GuardedWeight {
    inner: Arc<dyn QuantMethod>,
    blocked: BlockedCapability,
}

impl QuantizedSerde for GuardedWeight {
    fn name(&self) -> &'static str {
        "guarded-shared-mmvq-weight"
    }
}

impl QuantMethod for GuardedWeight {
    fn new(_method: QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("guarded weight requires an existing projection")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.inner.dequantize_w()
    }

    fn forward_raw(&self, input: &Tensor) -> Result<Tensor> {
        self.inner.forward(input)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        self.inner.quantized_act_type()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        self.inner.dtype_and_device()
    }

    fn get_qtensor(&self) -> Option<Arc<QTensor>> {
        if matches!(self.blocked, BlockedCapability::Collective) {
            None
        } else {
            self.inner.get_qtensor()
        }
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        if matches!(self.blocked, BlockedCapability::Collective) {
            None
        } else {
            self.inner.unquant_weight_bias()
        }
    }

    fn unquant_forward_weight(&self) -> Option<Tensor> {
        if matches!(self.blocked, BlockedCapability::Collective) {
            None
        } else {
            self.inner.unquant_forward_weight()
        }
    }

    fn has_bias(&self) -> bool {
        matches!(self.blocked, BlockedCapability::Bias)
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        matches!(self.blocked, BlockedCapability::Stats).then_some((0, 0))
    }

    fn is_dynamic_lora_active(&self) -> bool {
        matches!(self.blocked, BlockedCapability::Lora)
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        matches!(self.blocked, BlockedCapability::Collective)
            .then_some(DistributedKind::RowParallel)
    }

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams> {
        self.inner.plan_isq(request)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        self.inner.add_delta_w(delta)
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(self)
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn shared_mmvq_preserves_projection_and_backend_capabilities() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let left = layer(GgmlDType::Q8_0, LEFT_ROWS, &device)?;
    let right = layer(GgmlDType::Q4K, RIGHT_ROWS, &device)?;
    let input = Tensor::zeros((1, COLUMNS), DType::BF16, &device)?;
    let mixed_quant = layer(GgmlDType::Q8_0, LEFT_ROWS + 1, &device)?;
    let mixed_dense: Arc<dyn QuantMethod> = Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(candle_nn::Linear::new(
            Tensor::ones((MIXED_DENSE_ROWS, COLUMNS), DType::BF16, &device)?,
            None,
        )),
    )?);
    assert!(
        try_fused_quantized_pair(&input, mixed_quant.as_ref(), mixed_dense.as_ref())?.is_some()
    );
    assert!(
        try_fused_quantized_pair(&input, mixed_dense.as_ref(), mixed_quant.as_ref())?.is_some()
    );
    for blocked in [
        BlockedCapability::Bias,
        BlockedCapability::Stats,
        BlockedCapability::Lora,
        BlockedCapability::Collective,
    ] {
        let guarded = GuardedWeight {
            inner: left.clone(),
            blocked,
        };
        assert!(
            try_forward_with_shared_quantized_activation(&input, &[&guarded, right.as_ref()])?
                .is_none()
        );
        assert!(
            try_forward_with_shared_quantized_activation(&input, &[right.as_ref(), &guarded])?
                .is_none()
        );
        for (blocked_weight, other) in [
            (mixed_quant.clone(), mixed_dense.as_ref()),
            (mixed_dense.clone(), mixed_quant.as_ref()),
        ] {
            let guarded = GuardedWeight {
                inner: blocked_weight,
                blocked,
            };
            assert!(try_fused_quantized_pair(&input, &guarded, other)?.is_none());
            assert!(try_fused_quantized_pair(&input, other, &guarded)?.is_none());
        }
    }
    for format in [GgmlDType::Q8_1, GgmlDType::Q8K] {
        let unsupported = layer(format, LEFT_ROWS, &device)?;
        assert!(try_forward_with_shared_quantized_activation(
            &input,
            &[unsupported.as_ref(), right.as_ref()]
        )?
        .is_none());
    }
    let prefill = Tensor::zeros((MMVQ_MAX_BATCH + 1, COLUMNS), DType::BF16, &device)?;
    assert!(try_forward_with_shared_quantized_activation(
        &prefill,
        &[left.as_ref(), right.as_ref()]
    )?
    .is_none());
    assert!(try_forward_with_shared_quantized_activation(&input, &[left.as_ref()])?.is_none());
    Ok(())
}
