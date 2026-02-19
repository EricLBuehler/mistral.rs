use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

mod ops;
pub use ops::{fp8_blockwise_dequantize, fp8_blockwise_quantize};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use ops::{fp8_blockwise_matmul, fp8_indexed_moe_gemm};

#[cfg(feature = "cuda")]
mod ffi;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    AfqBits, AfqGroupSize, AfqLayer, DummyLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits,
    HqqConfig, HqqLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
};

#[derive(Debug)]
pub struct BlockwiseFP8Linear {
    weight: Tensor,
    weight_scale_inv: Tensor,
    bias: Option<Tensor>,
    dequant_dtype: DType,
    weight_block_size: Vec<usize>,
}

impl QuantMethod for BlockwiseFP8Linear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
            QuantMethodConfig::BlockwiseFP8 {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
            } => Ok(Self {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
            }),
        }
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Try to use native FP8 GEMM kernel on CUDA
        #[cfg(feature = "cuda")]
        {
            if matches!(x.device(), candle_core::Device::Cuda(_))
                && ffi::HAVE_BLOCKWISE_GEMM_KERNELS
            {
                // Handle batched inputs by flattening to 2D
                let orig_dims = x.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    // Flatten all but last dim: [batch, seq, features] -> [batch*seq, features]
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    x.reshape((batch_size, features))?
                } else {
                    x.clone()
                };

                // Use native FP8 GEMM kernel
                let result = ops::fp8_blockwise_matmul(
                    &x_2d,
                    &self.weight,
                    &self.weight_scale_inv,
                    &self.weight_block_size,
                )?;

                // Reshape back to original batch dimensions
                let result = if orig_dims.len() > 2 {
                    let out_features = result.dim(1)?;
                    let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                    new_dims.push(out_features);
                    result.reshape(new_dims)?
                } else {
                    result
                };

                // Apply bias if present
                if let Some(ref bias) = self.bias {
                    return result.broadcast_add(bias);
                }
                return Ok(result);
            }
        }

        // Fallback: dequantize and use unquantized matmul
        let weight = self.dequantize_w()?;
        // Dispatch to unquant. This uses some cublaslt for bias & on cuda always, so it is better
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight,
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Try to use native FP8 indexed MoE GEMM kernel on CUDA
        #[cfg(feature = "cuda")]
        {
            if matches!(x.device(), candle_core::Device::Cuda(_))
                && ffi::HAVE_BLOCKWISE_GEMM_KERNELS
            {
                // Use native FP8 indexed MoE GEMM kernel (expects U32 indices)
                let result = ops::fp8_indexed_moe_gemm(
                    x,
                    &self.weight,
                    &self.weight_scale_inv,
                    indices,
                    &self.weight_block_size,
                )?;
                // Apply bias if present (broadcast over tokens and topk)
                if let Some(ref bias) = self.bias {
                    return result.broadcast_add(bias);
                }
                return Ok(result);
            }
        }

        // Fallback: dequantize weights and compute manually
        let weight = self.dequantize_w()?;

        // Expected shapes:
        // - x: (n_tokens, 1, hidden_dim) or (n_tokens, n_experts_per_tok, hidden_dim)
        // - indices: (n_tokens, n_experts_per_tok)
        // - weight: (n_experts, out_features, in_features)

        let (n_tokens, n_experts_per_tok) = indices.dims2()?;
        let (_n_experts, out_features, _in_features) = weight.dims3()?;

        // Flatten indices to select expert weights
        let flat_indices = indices.flatten_all()?;

        // Select weights for each (token, expert) pair
        // weight_selected: (n_tokens * n_experts_per_tok, out_features, in_features)
        let weight_selected = weight.index_select(&flat_indices, 0)?;

        // Reshape x for batched matmul
        let x_expanded = if x.dims().len() == 3 && x.dim(1)? == 1 {
            // x is (n_tokens, 1, hidden_dim) - broadcast to (n_tokens * n_experts_per_tok, 1, hidden_dim)
            x.squeeze(1)?
                .unsqueeze(1)?
                .broadcast_as((n_tokens * n_experts_per_tok, 1, x.dim(2)?))?
                .contiguous()?
        } else if x.dims().len() == 3 {
            // x is (n_tokens, n_experts_per_tok, hidden_dim)
            x.reshape((n_tokens * n_experts_per_tok, 1, x.dim(2)?))?
        } else {
            // x is (n_tokens, hidden_dim)
            x.unsqueeze(1)?
                .broadcast_as((n_tokens * n_experts_per_tok, 1, x.dim(1)?))?
                .contiguous()?
        };

        // Batched matmul: (batch, 1, k) @ (batch, k, n).T = (batch, 1, n)
        // weight_selected is (batch, n, k), so we need to transpose last two dims
        let weight_t = weight_selected.transpose(1, 2)?;
        let result = x_expanded.matmul(&weight_t)?;

        // Reshape result to (n_tokens, n_experts_per_tok, out_features)
        let result = result.reshape((n_tokens, n_experts_per_tok, out_features))?;

        // Apply bias if present
        if let Some(ref bias) = self.bias {
            result.broadcast_add(bias)
        } else {
            Ok(result)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("BlockwiseFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::F8E4M3, self.weight.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )?;
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
                let res = HqqLayer::quantize(&weight.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.bias {
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
                    weight: weight.to_device(&device)?,
                    bias: self.bias.as_ref().map(|b| b.to_device(&device).unwrap()),
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
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(weight, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .bias
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

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
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

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);
                // Ignore imatrix altogether

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
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
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (3 for fp8), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// Dequant W scalar, f32, little endian
// -----------------------
// Dequant X scalar, f32, little endian
// -----------------------
// Quant scalar, f32, little endian
// -----------------------
// Quantization type, u32, little endian
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for BlockwiseFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "blockwise-fp8-linear"
    }
}

/// Create a BlockwiseFP8Linear for MoE with 3D weights [num_experts, N, K].
/// This is used by FusedExperts to enable gather_forward with native FP8 GEMM.
pub fn blockwise_fp8_moe(
    weight: Tensor,
    weight_scale_inv: Tensor,
    weight_block_size: Vec<usize>,
    dequant_dtype: DType,
) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype,
        weight_block_size,
    }))
}

pub fn blockwise_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    bias: bool,
    hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::Fp8 { weight_block_size } = config else {
        candle_core::bail!("Unexpected quantization config.")
    };

    // Handle the case where we actually have an unquantized layer
    if vb.contains_tensor("weight") && !vb.contains_tensor("weight_scale_inv") {
        return crate::linear_b(in_dim, out_dim, bias, &None, vb);
    }

    // Handle the case where the layer is dummy (no tensors)
    if !(vb.contains_tensor("weight") && vb.contains_tensor("weight_scale_inv")) {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    // Blockwise FP8 requires weight_block_size to be set
    let Some(weight_block_size) = weight_block_size else {
        candle_core::bail!("Blockwise FP8 requires weight_block_size to be set. Use per-tensor FP8 for models without block sizes.")
    };
    if weight_block_size.len() != 2 {
        candle_core::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        hints,
        DType::F32,
    )?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_block_size: weight_block_size.clone(),
        weight_scale_inv,
        bias,
        dequant_dtype: vb.dtype(),
    }))
}
