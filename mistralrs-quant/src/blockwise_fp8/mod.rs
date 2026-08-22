use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
mod deepgemm;
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
    make_dummy_or_error, ActivationQuantizationScheme, AfqBits, AfqGroupSize, AfqLayer, FP8Linear,
    Fp8ActivationScheme, GgufMatMul, HqqAxis, HqqBits, HqqConfig, HqqLayer, IsqType, QuantMethod,
    QuantMethodConfig, QuantizeOntoGuard, QuantizedActivation, QuantizedConfig, QuantizedSerde,
    Shard, ShardedVarBuilder, UnquantLinear,
};

#[derive(Debug)]
pub struct BlockwiseFP8Linear {
    weight: Tensor,
    weight_scale_inv: Tensor,
    bias: Option<Tensor>,
    dequant_dtype: DType,
    weight_block_size: Vec<usize>,
    activation_scheme: Option<Fp8ActivationScheme>,
    provider: BlockwiseFp8Provider,
}

#[derive(Clone, Debug)]
enum BlockwiseFp8Provider {
    Legacy,
    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    CutlassSm90,
    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    DeepGemmSm90(Arc<deepgemm::Prepared>),
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
static CUTLASS_FP8_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
static DEEPGEMM_FP8_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
static DEEPGEMM_FP8_FALLBACK_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
const FP8_SM90_PROVIDER_ENV: &str = "MISTRALRS_FP8_SM90_PROVIDER";

impl BlockwiseFp8Provider {
    fn supports_shared_activation(&self) -> bool {
        match self {
            Self::Legacy => false,
            #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
            Self::CutlassSm90 => true,
            #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
            Self::DeepGemmSm90(_) => true,
        }
    }
}

impl BlockwiseFP8Linear {
    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    fn deepgemm_enabled() -> bool {
        match std::env::var(FP8_SM90_PROVIDER_ENV) {
            Err(std::env::VarError::NotPresent) => true,
            Ok(provider) => {
                provider.eq_ignore_ascii_case("deepgemm") || provider.eq_ignore_ascii_case("auto")
            }
            Err(std::env::VarError::NotUnicode(_)) => false,
        }
    }

    fn prepare_provider(&mut self) -> Result<()> {
        self.provider = BlockwiseFp8Provider::Legacy;
        if self.activation_scheme != Some(Fp8ActivationScheme::Dynamic) {
            return Ok(());
        }
        #[cfg(all(
            feature = "cuda",
            has_cutlass_fp8_sm90_kernels,
            has_deepgemm_fp8_sm90_provider
        ))]
        if Self::deepgemm_enabled()
            && self.dequant_dtype == DType::BF16
            && deepgemm::supported(
                &self.weight,
                &self.weight_scale_inv,
                &self.weight_block_size,
            )
        {
            match deepgemm::prepare(
                &self.weight,
                &self.weight_scale_inv,
                &self.weight_block_size,
            ) {
                Ok(prepared) => {
                    self.provider = BlockwiseFp8Provider::DeepGemmSm90(prepared);
                    DEEPGEMM_FP8_PROVIDER_LOG.call_once(|| {
                        tracing::info!("Using DeepGEMM SM90 blockwise FP8 serving provider");
                    });
                    return Ok(());
                }
                Err(error) => {
                    DEEPGEMM_FP8_FALLBACK_LOG.call_once(|| {
                        tracing::warn!(
                            "DeepGEMM SM90 FP8 provider is unavailable ({error}); using CUTLASS"
                        );
                    });
                }
            }
        }
        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        if ops::cutlass_fp8_blockwise_supported(
            &self.weight,
            &self.weight_scale_inv,
            &self.weight_block_size,
        ) {
            let Device::Cuda(dev) = self.weight.device() else {
                unreachable!()
            };
            let _ = ops::prepare_cutlass_fp8(dev)?;
            self.provider = BlockwiseFp8Provider::CutlassSm90;
            CUTLASS_FP8_PROVIDER_LOG.call_once(|| {
                tracing::info!(
                    "Using CUTLASS SM90 blockwise FP8 GEMM with dynamic 1x128 activations"
                );
            });
        }
        Ok(())
    }
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
                activation_scheme,
            } => {
                let mut layer = Self {
                    weight,
                    weight_scale_inv,
                    bias,
                    dequant_dtype,
                    weight_block_size,
                    activation_scheme,
                    provider: BlockwiseFp8Provider::Legacy,
                };
                layer.prepare_provider()?;
                Ok(layer)
            }
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

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(all(
            feature = "cuda",
            has_cutlass_fp8_sm90_kernels,
            has_deepgemm_fp8_sm90_provider
        ))]
        if matches!(x.dtype(), DType::F16 | DType::BF16) {
            if let BlockwiseFp8Provider::DeepGemmSm90(prepared) = &self.provider {
                let original_shape = x.dims().to_vec();
                let features = original_shape
                    .last()
                    .copied()
                    .ok_or_else(|| candle_core::Error::msg("FP8 activation cannot be scalar"))?;
                let rows = original_shape[..original_shape.len() - 1]
                    .iter()
                    .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
                    .ok_or_else(|| {
                        candle_core::Error::msg("FP8 activation shape overflows usize")
                    })?;
                let input = x.reshape((rows, features))?;
                let result = if deepgemm::serving_supported(&input) {
                    deepgemm::matmul(prepared, &input, &self.weight, &self.weight_scale_inv)?
                } else {
                    let (activation, scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                    ops::fp8_blockwise_matmul_cutlass(
                        &activation,
                        &scales,
                        &self.weight,
                        &self.weight_scale_inv,
                        x.dtype(),
                    )?
                };
                let mut output_shape = original_shape[..original_shape.len() - 1].to_vec();
                output_shape.push(result.dim(1)?);
                let result = result.reshape(output_shape)?;
                if let Some(bias) = &self.bias {
                    return result.broadcast_add(bias);
                }
                return Ok(result);
            }
        }

        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        if self.activation_quantization_scheme().is_some()
            && matches!(x.dtype(), DType::F16 | DType::BF16)
        {
            let activation = self.quantize_activation(x)?;
            return self.forward_quantized(&activation);
        }

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
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
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

    fn activation_quantization_scheme(&self) -> Option<ActivationQuantizationScheme> {
        if self.activation_scheme != Some(Fp8ActivationScheme::Dynamic)
            || !self.provider.supports_shared_activation()
        {
            return None;
        }
        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        {
            Some(ActivationQuantizationScheme {
                dtype: DType::F8E4M3,
                block_shape: [1, self.weight_block_size[1]],
            })
        }
        #[cfg(not(all(feature = "cuda", has_cutlass_fp8_sm90_kernels)))]
        {
            None
        }
    }

    fn activation_quantization_scheme_for(
        &self,
        _x: &Tensor,
    ) -> Option<ActivationQuantizationScheme> {
        let scheme = self.activation_quantization_scheme()?;
        #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
        if matches!(&self.provider, BlockwiseFp8Provider::DeepGemmSm90(_))
            && _x.dtype() == DType::BF16
        {
            let (_, batch_dims) = _x.dims().split_last()?;
            let rows = batch_dims
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))?;
            if deepgemm::serving_shape_supported(_x.dtype(), rows) {
                return None;
            }
        }
        Some(scheme)
    }

    fn quantize_activation(&self, x: &Tensor) -> Result<QuantizedActivation> {
        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        {
            let scheme = self.activation_quantization_scheme().ok_or_else(|| {
                candle_core::Error::msg("blockwise FP8 activation quantization is unavailable")
            })?;
            let source_shape = x.dims().to_vec();
            let source_dtype = x.dtype();
            let features = source_shape
                .last()
                .copied()
                .ok_or_else(|| candle_core::Error::msg("FP8 activation cannot be scalar"))?;
            let rows = source_shape[..source_shape.len() - 1]
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
                .ok_or_else(|| candle_core::Error::msg("FP8 activation shape overflows usize"))?;
            let x = x.reshape((rows, features))?.contiguous()?;
            let (quantized, scales) = ops::fp8_quantize_activation_cutlass(&x)?;
            QuantizedActivation::new(quantized, scales, source_shape, source_dtype, scheme)
        }

        #[cfg(not(all(feature = "cuda", has_cutlass_fp8_sm90_kernels)))]
        {
            let _ = x;
            candle_core::bail!("blockwise FP8 activation quantization is unavailable")
        }
    }

    fn forward_quantized(&self, activation: &QuantizedActivation) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        {
            let scheme = self.activation_quantization_scheme().ok_or_else(|| {
                candle_core::Error::msg("blockwise FP8 activation quantization is unavailable")
            })?;
            if activation.scheme() != scheme {
                candle_core::bail!(
                    "FP8 activation scheme {:?} does not match layer scheme {:?}",
                    activation.scheme(),
                    scheme
                )
            }
            let result = ops::fp8_blockwise_matmul_cutlass(
                activation.quantized(),
                activation.scales(),
                &self.weight,
                &self.weight_scale_inv,
                activation.source_dtype(),
            )?;
            let output_features = result.dim(1)?;
            let mut output_shape =
                activation.source_shape()[..activation.source_shape().len() - 1].to_vec();
            output_shape.push(output_features);
            let result = result.reshape(output_shape)?;
            if let Some(bias) = &self.bias {
                return result.broadcast_add(bias);
            }
            Ok(result)
        }

        #[cfg(not(all(feature = "cuda", has_cutlass_fp8_sm90_kernels)))]
        {
            let _ = activation;
            candle_core::bail!("blockwise FP8 prequantized forward is unavailable")
        }
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("BlockwiseFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::F8E4M3, self.weight.device().clone())
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dequant_dtype,
            self.weight.device().clone(),
            self.weight.dims().to_vec(),
            request,
            true,
        ))
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
                    bias: self
                        .bias
                        .as_ref()
                        .map(|b| b.to_device(&device))
                        .transpose()?,
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
                        .map(|b| b.to_dtype(DType::F32)?.to_device(&device))
                        .transpose()?,
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
            Some(IsqType::MXFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("MXFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = weight.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                crate::MXFP4Layer::quantize(&w, b, &device)
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

impl QuantizedSerde for BlockwiseFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "blockwise-fp8-linear"
    }
}

/// Creates a blockwise FP8 layer for MoE models.
pub fn blockwise_fp8_moe(
    weight: Tensor,
    weight_scale_inv: Tensor,
    weight_block_size: Vec<usize>,
    activation_scheme: Option<Fp8ActivationScheme>,
    dequant_dtype: DType,
) -> Result<Arc<dyn QuantMethod>> {
    let mut layer = BlockwiseFP8Linear {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype,
        weight_block_size,
        activation_scheme,
        provider: BlockwiseFp8Provider::Legacy,
    };
    layer.prepare_provider()?;
    Ok(Arc::new(layer))
}

pub fn blockwise_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    bias: bool,
    hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::Fp8 {
        weight_block_size,
        activation_scheme,
        fmt,
        ..
    } = config
    else {
        candle_core::bail!("Unexpected quantization config.")
    };

    match blockwise_fp8_module_kind(config, &vb)? {
        BlockwiseFp8ModuleKind::Missing => {
            return make_dummy_or_error("blockwise_fp8_linear", &vb, &["weight"]);
        }
        BlockwiseFp8ModuleKind::Unquantized => {
            return unquantized_linear_b_with_hints(in_dim, out_dim, bias, hints, vb);
        }
        BlockwiseFp8ModuleKind::Quantized => {}
    }

    // Blockwise FP8 requires weight_block_size to be set
    let Some(weight_block_size) = weight_block_size else {
        candle_core::bail!("Blockwise FP8 requires weight_block_size to be set. Use per-tensor FP8 for models without block sizes.")
    };
    if weight_block_size.len() != 2 {
        candle_core::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    if weight_block_size.contains(&0) {
        candle_core::bail!("Expected nonzero weight_block_size, got {weight_block_size:?}")
    }
    if fmt.as_deref().is_some_and(|fmt| fmt != "e4m3") {
        candle_core::bail!("Unsupported blockwise FP8 format {fmt:?}; expected `e4m3`")
    }

    let scale_hints = scale_shard_from_weight_shard(
        [out_dim, in_dim],
        [weight_block_size[0], weight_block_size[1]],
        hints,
    )?;
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        scale_hints,
        DType::F32,
    )?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    let mut layer = BlockwiseFP8Linear {
        weight,
        weight_block_size: weight_block_size.clone(),
        weight_scale_inv,
        bias,
        dequant_dtype: vb.dtype(),
        activation_scheme: *activation_scheme,
        provider: BlockwiseFp8Provider::Legacy,
    };
    layer.prepare_provider()?;
    Ok(Arc::new(layer))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlockwiseFp8ModuleKind {
    Missing,
    Unquantized,
    Quantized,
}

pub(crate) fn blockwise_fp8_module_kind(
    config: &QuantizedConfig,
    vb: &ShardedVarBuilder,
) -> Result<BlockwiseFp8ModuleKind> {
    let QuantizedConfig::Fp8 {
        modules_to_not_convert,
        ..
    } = config
    else {
        candle_core::bail!("Unexpected quantization config.")
    };

    let has_weight = vb.contains_tensor("weight");
    let has_scale = vb.contains_tensor("weight_scale_inv");
    let prefix = vb.prefix();
    let module_path = canonical_language_model_module_path(&prefix);
    let is_excluded = modules_to_not_convert
        .iter()
        .any(|module| canonical_language_model_module_path(module) == module_path);

    if is_excluded {
        if has_scale {
            candle_core::bail!(
                "FP8-excluded module `{}` unexpectedly has `weight_scale_inv`",
                vb.prefix()
            );
        }
        return Ok(if has_weight {
            BlockwiseFp8ModuleKind::Unquantized
        } else {
            BlockwiseFp8ModuleKind::Missing
        });
    }

    if has_weight && !has_scale {
        if !modules_to_not_convert.is_empty() {
            candle_core::bail!(
                "FP8 module `{}` has no `weight_scale_inv` and is not listed in `modules_to_not_convert`",
                vb.prefix()
            );
        }
        return Ok(BlockwiseFp8ModuleKind::Unquantized);
    }

    Ok(if has_weight && has_scale {
        BlockwiseFp8ModuleKind::Quantized
    } else {
        BlockwiseFp8ModuleKind::Missing
    })
}

fn canonical_language_model_module_path(path: &str) -> Cow<'_, str> {
    for prefix in ["model.language_model.", "language_model.model."] {
        if let Some(suffix) = path.strip_prefix(prefix) {
            return Cow::Owned(format!("model.{suffix}"));
        }
    }
    Cow::Borrowed(path)
}

fn unquantized_linear_b_with_hints(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", hints)?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };
    Ok(Arc::new(UnquantLinear::new(
        QuantMethodConfig::Unquantized(Linear::new(weight, bias)),
    )?))
}

pub(crate) fn scale_shard_from_weight_shard(
    weight_shape: [usize; 2],
    weight_block_size: [usize; 2],
    weight_shard: Shard,
) -> Result<Shard> {
    let (dim, offset, len) = match weight_shard {
        Shard::Simple {
            dim,
            rank,
            world_size,
        } => {
            if world_size == 0 || rank >= world_size {
                candle_core::bail!(
                    "Invalid FP8 weight shard rank {rank} for world size {world_size}"
                );
            }
            let logical_dim = *weight_shape.get(dim).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Cannot shard rank-2 FP8 weight along dimension {dim}"
                ))
            })?;
            if !logical_dim.is_multiple_of(world_size) {
                candle_core::bail!(
                    "FP8 weight dimension {logical_dim} is not divisible by world size {world_size}"
                );
            }
            let len = logical_dim / world_size;
            (dim, rank * len, len)
        }
        Shard::Offset { dim, offset, len } => (dim, offset, len),
    };

    let logical_dim = *weight_shape.get(dim).ok_or_else(|| {
        candle_core::Error::msg(format!(
            "Cannot shard rank-2 FP8 weight along dimension {dim}"
        ))
    })?;
    let block_size = weight_block_size[dim];
    if block_size == 0 {
        candle_core::bail!("FP8 weight block size must be nonzero")
    }
    let end = offset
        .checked_add(len)
        .ok_or_else(|| candle_core::Error::msg("FP8 weight shard range overflowed".to_string()))?;
    if end > logical_dim {
        candle_core::bail!("FP8 weight shard {offset}..{end} exceeds dimension size {logical_dim}")
    }
    if !offset.is_multiple_of(block_size) || (end != logical_dim && !end.is_multiple_of(block_size))
    {
        candle_core::bail!(
            "FP8 weight shard {offset}..{end} is not aligned to block size {block_size}"
        )
    }

    let scale_offset = offset / block_size;
    let scale_end = end.div_ceil(block_size);
    Ok(Shard::Offset {
        dim,
        offset: scale_offset,
        len: scale_end - scale_offset,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::ShardedSafeTensors;

    fn fp8_config(exclusions: &[&str]) -> QuantizedConfig {
        QuantizedConfig::Fp8 {
            weight_block_size: Some(vec![128, 128]),
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
            fmt: Some("e4m3".to_string()),
            modules_to_not_convert: exclusions
                .iter()
                .map(|module| (*module).to_string())
                .collect(),
        }
    }

    #[test]
    fn scale_shard_follows_simple_weight_partition() -> Result<()> {
        let shard = scale_shard_from_weight_shard(
            [12_288, 5_120],
            [128, 128],
            Shard::Simple {
                dim: 0,
                rank: 3,
                world_size: 8,
            },
        )?;
        assert_eq!(
            shard,
            Shard::Offset {
                dim: 0,
                offset: 36,
                len: 12,
            }
        );

        let shard = scale_shard_from_weight_shard(
            [5_120, 6_144],
            [128, 128],
            Shard::Simple {
                dim: 1,
                rank: 7,
                world_size: 8,
            },
        )?;
        assert_eq!(
            shard,
            Shard::Offset {
                dim: 1,
                offset: 42,
                len: 6,
            }
        );
        Ok(())
    }

    #[test]
    fn scale_shard_maps_replicated_kv_offset_to_scale_rows() -> Result<()> {
        let shard = scale_shard_from_weight_shard(
            [1_024, 5_120],
            [128, 128],
            Shard::Offset {
                dim: 0,
                offset: 512,
                len: 256,
            },
        )?;
        assert_eq!(
            shard,
            Shard::Offset {
                dim: 0,
                offset: 4,
                len: 2,
            }
        );
        Ok(())
    }

    #[test]
    fn derived_scale_offset_selects_matching_scale_blocks() -> Result<()> {
        let scales =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (4, 2), &Device::Cpu)?;
        let scale_shard = scale_shard_from_weight_shard(
            [8, 4],
            [2, 2],
            Shard::Offset {
                dim: 0,
                offset: 4,
                len: 2,
            },
        )?;
        assert_eq!(
            scale_shard.apply_to(&scales)?.to_vec2::<f32>()?,
            [vec![5., 6.]]
        );
        Ok(())
    }

    #[test]
    fn scale_shard_rejects_unaligned_weight_partition() {
        let err = scale_shard_from_weight_shard(
            [1_024, 5_120],
            [128, 128],
            Shard::Offset {
                dim: 0,
                offset: 64,
                len: 256,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("not aligned"));
    }

    #[test]
    fn excluded_fp8_linear_preserves_weight_shard() -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "foo.weight".to_string(),
            Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
        );
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp("foo");
        let layer = blockwise_fp8_linear_b(
            4,
            8,
            &fp8_config(&["foo"]),
            false,
            Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            },
            vb,
        )?;
        assert_eq!(layer.dequantize_w()?.dims(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn official_language_model_alias_is_excluded() -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
        );
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu)
            .pp("model")
            .pp("language_model")
            .pp("embed_tokens");
        let layer = blockwise_fp8_linear_b(
            4,
            8,
            &fp8_config(&["model.embed_tokens"]),
            false,
            Shard::default(),
            vb,
        )?;
        assert_eq!(layer.dequantize_w()?.dims(), &[8, 4]);
        assert_ne!(
            canonical_language_model_module_path("other.model.embed_tokens"),
            canonical_language_model_module_path("model.embed_tokens")
        );
        Ok(())
    }

    #[test]
    fn declared_exclusions_make_unlisted_missing_scale_an_error() -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "foo.weight".to_string(),
            Tensor::zeros((8, 4), DType::F32, &Device::Cpu)?,
        );
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu).pp("foo");
        let err = blockwise_fp8_linear_b(4, 8, &fp8_config(&["bar"]), false, Shard::default(), vb)
            .unwrap_err();
        assert!(err.to_string().contains("not listed"));
        Ok(())
    }
}
