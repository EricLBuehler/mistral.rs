use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{quantized::GgmlDType, DType, Device, DeviceLocation, Result, Shape, Tensor, D};
use candle_nn::Linear;

use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_CONTROLLER},
    generate_isq, generate_isq_imatrix,
    hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer, ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, ImatrixLayerStats, IsqType, MatMul,
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

#[derive(Debug)]
pub struct UnquantLinear {
    w: Tensor,
    b: Option<Tensor>,
    stats: Option<ImatrixLayerStats>,
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
                stats: None,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.w.clone())
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
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

        if let Some(stats) = &self.stats {
            stats.process(a)?;
        }

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
        } else if let (Device::Cuda(_), Some(cublaslt)) =
            (a.device(), CUBLASLT_CONTROLLER.get_for_device(a.device()))
        {
            // cuBLAS batch_matmul requires 3D tensors, fall back to regular matmul for 2D
            if a.rank() >= 3 && w.rank() >= 3 {
                cublaslt
                    .batch_matmul(a, &w, None, None, None, None, None)?
                    .t()
            } else {
                MatMul.matmul(a, &w.t()?)
            }
        } else {
            MatMul.matmul(a, &w.t()?)
        }
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
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
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
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

    fn begin_track_stats(&mut self) -> Result<()> {
        self.stats = Some(ImatrixLayerStats::new(&self.w, self.w.device())?);
        Ok(())
    }

    fn end_track_stats(&self) -> Result<Tensor> {
        if let Some(stats) = &self.stats {
            let imatrix = stats.compute_imatrix()?;
            stats.clear()?;
            Ok(imatrix)
        } else {
            candle_core::bail!("`{}` does not support tracking stats.", self.name())
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (1 for unquantized), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for UnquantLinear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "unquant-linear"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.b.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        // Version is always first!

        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for unquant is 1
        buffer.push(QuantizedSerdeType::Unquant as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight
        serialize_tensor(&mut buffer, &self.w)?;

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Unquant as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Unquant as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let w = deserialize_tensor(&mut buffer, device)?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok(Arc::new(Self { w, b, stats: None }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Unquant as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Unquant as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let w = deserialize_tensor(&mut buffer, device)?;

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        Ok((
            Arc::new(Self {
                w,
                b: None,
                stats: None,
            }),
            b,
        ))
    }
}
