use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, DType, Device, DeviceLocation, Result, Shape, Tensor, D};
use candle_nn::Linear;

use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_CONTROLLER},
    generate_isq, generate_isq_imatrix,
    hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer, ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, ImatrixLayerStats, IsqType,
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
};

#[derive(Debug)]
pub struct UnquantLinear {
    w: Tensor,
    b: Option<Tensor>,
    stats: ImatrixLayerStats,
}

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
            QuantMethodConfig::Unquantized(l) => Ok(Self {
                w: l.weight().clone(),
                b: l.bias().cloned(),
                stats: ImatrixLayerStats::empty(),
            }),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.w.clone())
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper(a.device().clone());

        // Try custom GEMV for single-token decode (batch_size=1)
        #[cfg(feature = "cuda")]
        if crate::gemv::should_use_gemv(a, &self.w) {
            return crate::gemv::gemv(a, &self.w, self.b.as_ref());
        }

        let w = match *a.dims() {
            [b1, b2, _, _] => self.w.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.w.broadcast_left(bsize)?,
            _ => self.w.clone(),
        };

        self.stats.process(a)?;

        if let Some(b) = self.b.as_ref() {
            let mut tgt_shape = a.dims().to_vec();
            tgt_shape[a.dims().len() - 1] = w.dim(D::Minus2)?;
            let b = b.broadcast_as(Shape::from_dims(&tgt_shape))?;

            match a.device().location() {
                DeviceLocation::Cuda { .. } => {
                    // Try to use cublaslt, otherwise fallback to gemm
                    if let (Device::Cuda(_), Some(cublaslt)) =
                        (a.device(), CUBLASLT_CONTROLLER.get_for_device(a.device()))
                    {
                        cublaslt
                            .batch_matmul(
                                a,
                                &w,
                                Some(&b.t()?.contiguous()?),
                                None,
                                Some(1.0),
                                None,
                                None,
                            )?
                            .t()
                    } else {
                        let matmul_result = a.matmul(&w.t()?)?;
                        matmul_result.broadcast_add(&b)
                    }
                }
                DeviceLocation::Metal { .. } => {
                    let matmul_result = a.matmul(&w.t()?)?;
                    matmul_result.broadcast_add(&b)
                }
                DeviceLocation::Cpu => {
                    #[cfg(feature = "accelerate")]
                    {
                        let original_dtype = a.dtype();
                        let a_f32 = a.to_dtype(DType::F32)?;
                        let w_f32 = w.t()?.to_dtype(DType::F32)?;
                        let b_f32 = b.to_dtype(DType::F32)?;
                        let matmul_result = a_f32.matmul(&w_f32)?;
                        matmul_result
                            .broadcast_add(&b_f32)?
                            .to_dtype(original_dtype)
                    }
                    #[cfg(not(feature = "accelerate"))]
                    {
                        let matmul_result = a.matmul(&w.t()?)?;
                        matmul_result.broadcast_add(&b)
                    }
                }
            }
        } else {
            match a.device().location() {
                DeviceLocation::Cuda { .. } => {
                    if let (Device::Cuda(_), Some(cublaslt)) =
                        (a.device(), CUBLASLT_CONTROLLER.get_for_device(a.device()))
                    {
                        // cuBLAS batch_matmul requires 3D tensors, fall back to regular matmul for 2D.
                        if a.rank() >= 3 && w.rank() >= 3 {
                            cublaslt
                                .batch_matmul(a, &w, None, None, None, None, None)?
                                .t()
                        } else {
                            a.matmul(&w.t()?)
                        }
                    } else {
                        a.matmul(&w.t()?)
                    }
                }
                DeviceLocation::Metal { .. } => a.matmul(&w.t()?),
                DeviceLocation::Cpu => {
                    #[cfg(feature = "accelerate")]
                    {
                        let original_dtype = a.dtype();
                        a.to_dtype(DType::F32)?
                            .matmul(&w.t()?.to_dtype(DType::F32)?)?
                            .to_dtype(original_dtype)
                    }
                    #[cfg(not(feature = "accelerate"))]
                    {
                        a.matmul(&w.t()?)
                    }
                }
            }
        }
    }

    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Weights are [num_experts, out_features, in_features]
        // For Metal path:
        //   - a: (b_size, seq_len, 1, 1, hidden_dim) - 5D
        //   - indices: (b_size, seq_len, num_experts_per_tok) - 3D
        // For CUDA path:
        //   - a: (num_tokens, 1, hidden_dim) - 3D
        //   - indices: (num_tokens, num_experts_per_tok) - 2D

        let w = &self.w;
        let (_num_experts, out_features, _in_features) = w.dims3()?;

        match a.dims() {
            // Metal path: 5D input (b_size, seq_len, 1, 1, hidden_dim)
            &[b_size, seq_len, 1, 1, hidden_dim] => {
                let (_b, _s, num_experts_per_tok) = indices.dims3()?;
                // Flatten indices to select experts
                let flat_indices = indices.reshape((b_size * seq_len * num_experts_per_tok,))?;

                // Select expert weights: [b*s*k, out_features, in_features]
                let selected_w = w.index_select(&flat_indices, 0)?;

                // Reshape input: [b*s, hidden_dim]
                let a_flat = a.reshape((b_size * seq_len, hidden_dim))?;

                // For each token, we need to compute with each selected expert
                // Broadcast a to match: [b*s, 1, hidden_dim] -> [b*s, k, hidden_dim]
                let a_expanded = a_flat
                    .unsqueeze(1)?
                    .broadcast_as((b_size * seq_len, num_experts_per_tok, hidden_dim))?
                    .reshape((b_size * seq_len * num_experts_per_tok, hidden_dim))?;

                // Matmul: [b*s*k, hidden_dim] @ [b*s*k, hidden_dim, out_features] -> [b*s*k, out_features]
                let result = a_expanded
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                // Reshape back to [b, s, k, out_features]
                result.reshape((b_size, seq_len, num_experts_per_tok, out_features))
            }
            // CUDA path: 3D input (num_tokens, 1, hidden_dim)
            &[num_tokens, 1, hidden_dim] => {
                let (_, num_experts_per_tok) = indices.dims2()?;

                // Flatten indices
                let flat_indices = indices.reshape((num_tokens * num_experts_per_tok,))?;

                // Select expert weights: [n*k, out_features, in_features]
                let selected_w = w.index_select(&flat_indices, 0)?;

                // Broadcast input: [n, 1, hidden] -> [n, k, hidden] -> [n*k, hidden]
                let a_expanded = a
                    .broadcast_as((num_tokens, num_experts_per_tok, hidden_dim))?
                    .reshape((num_tokens * num_experts_per_tok, hidden_dim))?;

                // Matmul: [n*k, hidden] @ [n*k, hidden, out] -> [n*k, out]
                let result = a_expanded
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                // Reshape to [n, k, out]
                result.reshape((num_tokens, num_experts_per_tok, out_features))
            }
            &[num_tokens, num_experts_per_tok, hidden_dim] => {
                let (indices_num_tokens, indices_num_experts_per_tok) = indices.dims2()?;
                if num_tokens != indices_num_tokens
                    || num_experts_per_tok != indices_num_experts_per_tok
                {
                    candle_core::bail!(
                        "UnquantLinear::gather_forward: input shape {:?} does not match indices shape {:?}",
                        a.dims(),
                        indices.dims()
                    );
                }

                let flat_indices = indices.reshape((num_tokens * num_experts_per_tok,))?;
                let selected_w = w.index_select(&flat_indices, 0)?;
                let a_flat = a.reshape((num_tokens * num_experts_per_tok, hidden_dim))?;

                let result = a_flat
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                result.reshape((num_tokens, num_experts_per_tok, out_features))
            }
            // Metal path stage 2: per-slot inputs (b, s, k, hidden) from a prior gather's output
            &[b_size, seq_len, num_experts_per_tok, hidden_dim] => {
                let (ib, is, ik) = indices.dims3()?;
                if (b_size, seq_len, num_experts_per_tok) != (ib, is, ik) {
                    candle_core::bail!(
                        "UnquantLinear::gather_forward: input shape {:?} does not match indices shape {:?}",
                        a.dims(),
                        indices.dims()
                    );
                }
                let flat = b_size * seq_len * num_experts_per_tok;
                let flat_indices = indices.reshape((flat,))?;
                let selected_w = w.index_select(&flat_indices, 0)?;
                let a_flat = a.reshape((flat, hidden_dim))?;

                let result = a_flat
                    .unsqueeze(1)?
                    .matmul(&selected_w.transpose(1, 2)?)?
                    .squeeze(1)?;

                result.reshape((b_size, seq_len, num_experts_per_tok, out_features))
            }
            dims => {
                candle_core::bail!(
                    "UnquantLinear::gather_forward: unsupported input shape {:?}",
                    dims
                );
            }
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self {
            w: (&self.w + delta)?,
            b: self.b.clone(),
            stats: self.stats.clone(),
        }))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.w.dtype(), self.w.device().clone())
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            /*Some(IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    // IsqType::HQQ3 => HqqBits::Three,
                    // IsqType::HQQ2 => HqqBits::Two,
                    // IsqType::HQQ1 => HqqBits::One,
                    _ => unreachable!(),
                };
                let cfg = HqqConfig {
                    bits,
                    group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                    axis: HqqAxis::Zero,
                    optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                    round_zeros: false,
                    channel_wise: true,
                };
                let res = HqqLayer::quantize(&self.w.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.b {
                    let bias = bias
                        .to_device(&device)?
                        .to_dtype(res.dtype_and_device().0)?;
                    Ok(Arc::new(res.with_bias(bias)))
                } else {
                    Ok(Arc::new(res))
                }
            }
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("AFQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::AFQ8 => AfqBits::Eight,
                    IsqType::AFQ6 => AfqBits::Six,
                    IsqType::AFQ4 => AfqBits::Four,
                    IsqType::AFQ3 => AfqBits::Three,
                    IsqType::AFQ2 => AfqBits::Two,
                    _ => unreachable!(),
                };

                Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: self.w.to_device(&device)?,
                    bias: self.b.as_ref().map(|b| b.to_device(&device).unwrap()),
                    bits,
                    group_size: AfqGroupSize::default(),
                })?))
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let ty = dtype.unwrap();
                // routed imatrix vectors are per-expert; stacks quantize slab-by-slab
                if self.w.rank() == 3 && imatrix_weight.is_some() {
                    n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let w = GgufMatMul::quantize_expert_stack(
                        &self.w,
                        ty,
                        imatrix_weight.as_deref(),
                        &device,
                        guard,
                    )?;
                    let b = match &self.b {
                        Some(b) => Some(b.to_dtype(DType::F32)?.to_device(&device)?),
                        None => None,
                    };
                    return Ok(Arc::new(GgufMatMul::from_qtensor(w, b)));
                }
                let dtype: GgmlDType = ty.try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(self.w, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(self.w, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .b
                        .as_ref()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("F8E4M3 does not support imatrix.");
                }

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                    lin: Linear::new(w, b),
                    dtype: DType::F8E4M3,
                })?))
            }
            Some(IsqType::MXFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("MXFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = self.w.to_device(&device)?;
                let b = self.b.as_ref().map(|b| b.to_device(&device)).transpose()?;
                crate::MXFP4Layer::quantize(&w, b, &device)
            }
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8Q8 does not support imatrix.");
                }

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);
                // Ignore imatrix altogether

                let w = self.w.to_device(&device)?;
                let b = if let Some(b) = &self.b {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        Some((self.w.clone(), self.b.clone()))
    }

    fn begin_track_stats(&self) -> Result<()> {
        // Stacked [E, out, in] expert weights collect per expert via the routed path.
        if self.w.dims().len() == 3 {
            self.stats.enable_routed(
                self.w.dim(0)?,
                self.w.dim(candle_core::D::Minus1)?,
                self.w.device(),
            )
        } else {
            self.stats
                .enable(self.w.dim(candle_core::D::Minus1)?, self.w.device())
        }
    }

    fn process_routed_stats(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        self.stats.process_routed(x, ids)
    }

    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        self.stats.snapshot()
    }
    fn end_track_stats(&self) -> Result<Tensor> {
        if self.stats.is_enabled() {
            let imatrix = self.stats.compute_imatrix();
            self.stats.clear()?;
            imatrix
        } else {
            candle_core::bail!("`{}` is not tracking stats.", self.name())
        }
    }
}

impl QuantizedSerde for UnquantLinear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "unquant-linear"
    }
    fn serialize_directly(&self, prefix: &str, ty: IsqType) -> Result<Vec<crate::UqffTensor>> {
        if !ty.supports_uqff() {
            candle_core::bail!("UQFF direct serialization does not support {ty}.");
        }

        let layer = Arc::new(Self {
            w: self.w.clone(),
            b: self.b.clone(),
            stats: self.stats.clone(),
        })
        .apply_isq(
            Some(ty),
            self.w.device().clone(),
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        )?;

        layer.serialize_directly(prefix, ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layer(device: &Device) -> Result<UnquantLinear> {
        let weight = Tensor::from_vec(
            vec![1f32, 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.],
            (2, 2, 3),
            device,
        )?;
        <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(Linear::new(
            weight, None,
        )))
    }

    #[test]
    fn gather_forward_expands_single_route_input() -> Result<()> {
        let device = Device::Cpu;
        let layer = test_layer(&device)?;
        let input = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 1, 3), &device)?;
        let indices = Tensor::from_vec(vec![0u32, 1, 1, 0], (2, 2), &device)?;

        let output = layer.gather_forward(&input, &indices)?;

        assert_eq!(output.dims(), &[2, 2, 2]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            &[1., 2., 3., 6., 6., 15., 4., 5.]
        );
        Ok(())
    }

    #[test]
    fn gather_forward_accepts_per_route_input() -> Result<()> {
        let device = Device::Cpu;
        let layer = test_layer(&device)?;
        let input = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &device)?;
        let indices = Tensor::from_vec(vec![0u32, 1], (1, 2), &device)?;

        let output = layer.gather_forward(&input, &indices)?;

        assert_eq!(output.dims(), &[1, 2, 2]);
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, &[1., 2., 6., 15.]);
        Ok(())
    }
}
