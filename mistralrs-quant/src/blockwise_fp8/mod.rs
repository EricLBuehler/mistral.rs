use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
mod deepgemm;
#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
pub(crate) mod mma;
pub(crate) mod ops;
pub use ops::{
    fp8_blockwise_dequantize, fp8_blockwise_quantize, fused_add_rms_norm_quantized,
    fused_add_rms_norm_quantized_with_normalized,
};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use ops::{fp8_blockwise_matmul, fp8_indexed_moe_gemm};

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
use crate::GluActivationType;
use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    ActivationQuantizationScheme, ActivationScaleLayout, AfqBits, AfqGroupSize, AfqLayer,
    FP8Linear, Fp8ActivationScheme, GgufMatMul, HqqAxis, HqqBits, HqqConfig, HqqLayer, IsqType,
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedActivation, QuantizedConfig,
    QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
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
    #[cfg(all(feature = "cuda", feature = "cutile"))]
    CutileW8A16,
    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    CutlassSm90,
    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    DeepGemmSm90(Arc<deepgemm::Prepared>),
    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
    TensorCoreGemv,
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
static CUTLASS_FP8_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
static DEEPGEMM_FP8_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
static DEEPGEMM_FP8_FALLBACK_LOG: std::sync::Once = std::sync::Once::new();
#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
const FP8_SM90_PROVIDER_ENV: &str = "MISTRALRS_FP8_SM90_PROVIDER";
#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
static TENSOR_CORE_GEMV_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();

impl BlockwiseFp8Provider {
    fn supports_shared_activation(&self) -> bool {
        match self {
            Self::Legacy => false,
            #[cfg(all(feature = "cuda", feature = "cutile"))]
            Self::CutileW8A16 => false,
            #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
            Self::CutlassSm90 => true,
            #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
            Self::DeepGemmSm90(_) => true,
            #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
            Self::TensorCoreGemv => true,
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
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if self.activation_scheme.is_none()
            && self.weight_block_size == [128, 128]
            && matches!(self.dequant_dtype, DType::BF16 | DType::F16)
        {
            if let (Device::Cuda(dev), Ok((n, k))) = (self.weight.device(), self.weight.dims2()) {
                if crate::cutile::fp8_w8a16_supported(dev, n, k, self.dequant_dtype) {
                    crate::cutile::register_fp8_w8a16_shape(
                        &self.weight,
                        &self.weight_scale_inv,
                        crate::Fp8WeightScaleLayout::Block([128, 128]),
                        self.dequant_dtype,
                    );
                    self.provider = BlockwiseFp8Provider::CutileW8A16;
                    return Ok(());
                }
            }
        }
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
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if self.dequant_dtype == DType::BF16
            && mma::weight_supported(
                &self.weight,
                &self.weight_scale_inv,
                &self.weight_block_size,
            )
        {
            self.provider = BlockwiseFp8Provider::TensorCoreGemv;
            #[cfg(feature = "cutile")]
            if let (Device::Cuda(dev), Ok((n, k))) = (self.weight.device(), self.weight.dims2()) {
                if crate::cutile::fp8_gemm_supported(dev, n, k) {
                    crate::cutile::register_fp8_gemm_shape(&self.weight, &self.weight_scale_inv);
                }
            }
            TENSOR_CORE_GEMV_PROVIDER_LOG.call_once(|| {
                tracing::info!(
                    "Using FP8 tensor-core W8A8 GEMV for decode with dynamic 1x128 activations"
                );
            });
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    fn cutile_scale_layout() -> ActivationScaleLayout {
        ActivationScaleLayout::GroupMajor {
            row_alignment: std::num::NonZeroUsize::new(crate::cutile::FP8_GEMM_BLOCK_ROWS)
                .expect("cuTile GEMM block rows are nonzero"),
        }
    }

    /// Rows padded to the GEMM block when `x` takes the cuTile GEMM instead of the tensor-core GEMV.
    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    fn cutile_padded_rows(&self, x: &Tensor) -> Option<usize> {
        if !matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv)
            || x.dtype() != DType::BF16
        {
            return None;
        }
        let Device::Cuda(dev) = x.device() else {
            return None;
        };
        let (n, k) = self.weight.dims2().ok()?;
        let (_, batch_dims) = x.dims().split_last()?;
        let rows = batch_dims
            .iter()
            .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))?;
        let block_rows = crate::cutile::FP8_GEMM_BLOCK_ROWS;
        (rows > mma::MMA_GEMV_MAX_ROWS && crate::cutile::fp8_gemm_supported(dev, n, k))
            .then(|| rows.div_ceil(block_rows) * block_rows)
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    fn forward_cutile_quantized(&self, activation: &QuantizedActivation) -> Result<Tensor> {
        let dev = activation.quantized().device().as_cuda_device()?;
        let (rows, _) = activation.quantized().dims2()?;
        let result = crate::cutile::cutile_fp8_gemm(
            activation.quantized(),
            activation.scales(),
            &self.weight,
            &self.weight_scale_inv,
            dev,
        )?
        .narrow(0, 0, rows)?;
        let source_shape = activation.source_shape();
        let mut output_shape = source_shape[..source_shape.len() - 1].to_vec();
        output_shape.push(result.dim(1)?);
        let result = result.reshape(output_shape)?;
        match &self.bias {
            Some(bias) => result.broadcast_add(bias),
            None => Ok(result),
        }
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
        #[cfg(all(feature = "cuda", feature = "cutile"))]
        if matches!(self.provider, BlockwiseFp8Provider::CutileW8A16)
            && matches!(x.dtype(), DType::BF16 | DType::F16)
        {
            let original_shape = x.dims().to_vec();
            let features = original_shape
                .last()
                .copied()
                .ok_or_else(|| candle_core::Error::msg("FP8 activation cannot be scalar"))?;
            let rows = original_shape[..original_shape.len() - 1]
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
                .ok_or_else(|| candle_core::Error::msg("FP8 activation shape overflows usize"))?;
            let (output_features, weight_features) = self.weight.dims2()?;
            if features != weight_features {
                candle_core::bail!(
                    "FP8 activation K={features} does not match weight K={weight_features}"
                )
            }
            if !x.device().same_device(self.weight.device()) {
                candle_core::bail!("FP8 weight and activation must be on the same device")
            }
            if rows == 0 {
                let mut output_shape = original_shape[..original_shape.len() - 1].to_vec();
                output_shape.push(output_features);
                return Tensor::zeros(output_shape, x.dtype(), x.device());
            }
            let input = x.reshape((rows, features))?;
            let result = crate::cutile::cutile_fp8_w8a16(
                &input,
                &self.weight,
                &self.weight_scale_inv,
                crate::Fp8WeightScaleLayout::Block([128, 128]),
            )?;
            let mut output_shape = original_shape[..original_shape.len() - 1].to_vec();
            output_shape.push(result.dim(1)?);
            let result = result.reshape(output_shape)?;
            return match &self.bias {
                Some(bias) => result.broadcast_add(bias),
                None => Ok(result),
            };
        }

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

        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) && x.dtype() == DType::BF16
        {
            if self.activation_quantization_scheme_for(x).is_some() {
                let activation = self.quantize_activation(x)?;
                return self.forward_quantized(&activation);
            }
            let weight = self.dequantize_w()?;
            let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                weight,
                self.bias.clone(),
            )))?;
            return unquant.forward(x);
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
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) {
            return Some(ActivationQuantizationScheme {
                dtype: DType::F8E4M3,
                block_shape: [1, self.weight_block_size[1]],
            });
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
        x: &Tensor,
    ) -> Option<ActivationQuantizationScheme> {
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) {
            #[cfg(feature = "cutile")]
            if self.cutile_padded_rows(x).is_some() {
                return self.activation_quantization_scheme();
            }
            let (_, batch_dims) = x.dims().split_last()?;
            let rows = batch_dims.iter().product::<usize>();
            if x.dtype() != DType::BF16 || rows == 0 || rows > mma::MMA_GEMV_MAX_ROWS {
                return None;
            }
        }
        let _ = x;
        self.activation_quantization_scheme()
    }

    fn preferred_activation_scale_layout_for(&self, x: &Tensor) -> Option<ActivationScaleLayout> {
        self.activation_quantization_scheme_for(x)?;
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
        if self.cutile_padded_rows(x).is_some() {
            return Some(Self::cutile_scale_layout());
        }
        #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
        if matches!(&self.provider, BlockwiseFp8Provider::DeepGemmSm90(_))
            && x.dtype() == DType::BF16
        {
            let (_, batch_dims) = x.dims().split_last()?;
            let rows = batch_dims
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))?;
            if deepgemm::serving_shape_supported(x.dtype(), rows) {
                return Some(ActivationScaleLayout::GroupMajor {
                    row_alignment: std::num::NonZeroUsize::new(
                        deepgemm::DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT,
                    )
                    .expect("DeepGEMM row alignment is nonzero"),
                });
            }
        }
        Some(ActivationScaleLayout::RowMajor)
    }

    fn quantize_activation(&self, x: &Tensor) -> Result<QuantizedActivation> {
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) {
            let scheme = self.activation_quantization_scheme_for(x).ok_or_else(|| {
                candle_core::Error::msg("FP8 tensor-core activation quantization is unavailable")
            })?;
            let source_shape = x.dims().to_vec();
            let features = source_shape[source_shape.len() - 1];
            let rows = source_shape[..source_shape.len() - 1]
                .iter()
                .product::<usize>();
            #[cfg(feature = "cutile")]
            if let Some(padded_rows) = self.cutile_padded_rows(x) {
                let (quantized, scales) =
                    mma::quantize_activation_padded(&x.reshape((rows, features))?, padded_rows)?;
                return QuantizedActivation::new_with_scale_layout(
                    quantized.narrow(0, 0, rows)?,
                    scales,
                    source_shape,
                    x.dtype(),
                    scheme,
                    Self::cutile_scale_layout(),
                );
            }
            let layout = ActivationScaleLayout::RowMajor;
            let (quantized, scales) =
                mma::quantize_activation(&x.reshape((rows, features))?, layout)?;
            return QuantizedActivation::new_with_scale_layout(
                quantized,
                scales,
                source_shape,
                x.dtype(),
                scheme,
                layout,
            );
        }
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
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) {
            #[cfg(feature = "cutile")]
            if activation.scale_layout() == Self::cutile_scale_layout() {
                return self.forward_cutile_quantized(activation);
            }
            let result = mma::gemv(
                activation.quantized(),
                activation.scales(),
                activation.scale_layout(),
                &self.weight,
                &self.weight_scale_inv,
            )?;
            let source_shape = activation.source_shape();
            let mut output_shape = source_shape[..source_shape.len() - 1].to_vec();
            output_shape.push(result.dim(1)?);
            let result = result.reshape(output_shape)?;
            if let Some(bias) = &self.bias {
                return result.broadcast_add(bias);
            }
            return Ok(result);
        }
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
            let result = match activation.scale_layout() {
                ActivationScaleLayout::RowMajor => ops::fp8_blockwise_matmul_cutlass(
                    activation.quantized(),
                    activation.scales(),
                    &self.weight,
                    &self.weight_scale_inv,
                    activation.source_dtype(),
                )?,
                ActivationScaleLayout::GroupMajor { row_alignment } => {
                    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
                    {
                        if row_alignment.get() != deepgemm::DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT {
                            candle_core::bail!(
                                "DeepGEMM activation scales require row alignment {}",
                                deepgemm::DEEPGEMM_ACTIVATION_SCALE_M_ALIGNMENT
                            )
                        }
                        if activation.source_dtype() != DType::BF16 {
                            candle_core::bail!(
                                "DeepGEMM prequantized activation requires a BF16 source"
                            )
                        }
                        let BlockwiseFp8Provider::DeepGemmSm90(prepared) = &self.provider else {
                            candle_core::bail!(
                                "group-major FP8 activation scales require the DeepGEMM provider"
                            )
                        };
                        deepgemm::matmul_prequantized(
                            prepared,
                            activation.quantized(),
                            activation.scales(),
                            &self.weight,
                            &self.weight_scale_inv,
                        )?
                    }
                    #[cfg(not(all(feature = "cuda", has_deepgemm_fp8_sm90_provider)))]
                    {
                        let _ = row_alignment;
                        candle_core::bail!(
                            "group-major FP8 activation scales require the DeepGEMM provider"
                        )
                    }
                }
            };
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

    #[cfg(feature = "cuda")]
    fn try_forward_fused_split_glu(
        &self,
        input: &Tensor,
        split_size: usize,
        activation: GluActivationType,
    ) -> Result<Option<Tensor>> {
        #[cfg(all(has_blockwise_fp8_kernels, feature = "cutile"))]
        if matches!(self.provider, BlockwiseFp8Provider::TensorCoreGemv) {
            if self.activation_scheme != Some(Fp8ActivationScheme::Dynamic)
                || input.dtype() != DType::BF16
            {
                return Ok(None);
            }
            let Some((&packed_features, batch_dims)) = input.dims().split_last() else {
                return Ok(None);
            };
            let Device::Cuda(dev) = input.device() else {
                return Ok(None);
            };
            let (n, k) = self.weight.dims2()?;
            let rows = batch_dims
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
                .ok_or_else(|| {
                    candle_core::Error::msg("fused split GLU activation shape overflows usize")
                })?;
            if packed_features != split_size.saturating_mul(2)
                || k != split_size
                || rows <= mma::MMA_GEMV_MAX_ROWS
                || !crate::cutile::fp8_gemm_supported(dev, n, k)
            {
                return Ok(None);
            }
            let block_rows = crate::cutile::FP8_GEMM_BLOCK_ROWS;
            let padded_rows = rows.div_ceil(block_rows) * block_rows;
            let input = input.reshape((rows, packed_features))?;
            let (quantized, scales) = crate::utils::fused_split_glu_quantized_bf16(
                &input,
                split_size,
                self.weight_block_size[1],
                padded_rows,
                activation,
            )?;
            let result = crate::cutile::cutile_fp8_gemm(
                &quantized,
                &scales,
                &self.weight,
                &self.weight_scale_inv,
                dev,
            )?
            .narrow(0, 0, rows)?;
            let mut output_shape = batch_dims.to_vec();
            output_shape.push(result.dim(1)?);
            let result = result.reshape(output_shape)?;
            return match &self.bias {
                Some(bias) => result.broadcast_add(bias).map(Some),
                None => Ok(Some(result)),
            };
        }
        #[cfg(all(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider))]
        {
            let BlockwiseFp8Provider::DeepGemmSm90(prepared) = &self.provider else {
                return Ok(None);
            };
            if self.activation_scheme != Some(Fp8ActivationScheme::Dynamic)
                || input.dtype() != DType::BF16
                || !input.device().is_cuda()
            {
                return Ok(None);
            }
            let Some((&packed_features, batch_dims)) = input.dims().split_last() else {
                return Ok(None);
            };
            if packed_features != split_size.saturating_mul(2) {
                return Ok(None);
            }
            let rows = batch_dims
                .iter()
                .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
                .ok_or_else(|| {
                    candle_core::Error::msg("fused split GLU activation shape overflows usize")
                })?;
            if !deepgemm::serving_shape_supported(input.dtype(), rows) {
                return Ok(None);
            }
            let scale_shape = deepgemm::activation_scale_shape(rows, split_size)?;
            let input = input.reshape((rows, packed_features))?;
            let (quantized, scales) = crate::utils::fused_split_glu_quantized_bf16(
                &input,
                split_size,
                self.weight_block_size[1],
                scale_shape[1],
                activation,
            )?;
            let quantized = quantized.reshape((rows, split_size))?;
            let result = deepgemm::matmul_prequantized(
                prepared,
                &quantized,
                &scales,
                &self.weight,
                &self.weight_scale_inv,
            )?;
            let mut output_shape = batch_dims.to_vec();
            output_shape.push(result.dim(1)?);
            let result = result.reshape(output_shape)?;
            match &self.bias {
                Some(bias) => result.broadcast_add(bias).map(Some),
                None => Ok(Some(result)),
            }
        }

        #[cfg(not(all(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)))]
        {
            let _ = (input, split_size, activation);
            Ok(None)
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

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    fn deepgemm_fused_glu_test_layer(
        dev: &Device,
        output_features: usize,
        input_features: usize,
    ) -> Result<BlockwiseFP8Linear> {
        const BLOCK_SIZE: usize = 128;

        let weight_values = (0..output_features * input_features)
            .map(|index| {
                let row = index / input_features;
                let column = index % input_features;
                let block = row / BLOCK_SIZE * (input_features / BLOCK_SIZE) + column / BLOCK_SIZE;
                let amplitude = [0.04, 0.18, 0.75, 1.6][block % 4];
                let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
                value * amplitude
            })
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weight_values, (output_features, input_features), dev)?
            .to_dtype(DType::BF16)?;
        let (weight, weight_scale_inv) =
            ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight,
            weight_scale_inv,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        })?;
        assert!(matches!(
            &layer.provider,
            BlockwiseFp8Provider::DeepGemmSm90(_)
        ));
        Ok(layer)
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    fn deepgemm_fused_glu_test_input(
        rows: usize,
        split_size: usize,
        phase: usize,
        dev: &Device,
    ) -> Result<Tensor> {
        const GROUP_SIZE: usize = 128;

        let packed_features = split_size * 2;
        let values = (0..rows * packed_features)
            .map(|index| {
                let row = index / packed_features;
                let packed_column = index % packed_features;
                let column = packed_column % split_size;
                let half = packed_column / split_size;
                let amplitude =
                    [0.2, 0.75][column / GROUP_SIZE % 2] * if half == 0 { 1.0 } else { 0.6 };
                let value = ((row * 37 + column * 19 + half * 11 + phase * 23) % 101) as f32 - 50.0;
                value * amplitude / 17.0
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, packed_features), dev)?.to_dtype(DType::BF16)
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    fn assert_deepgemm_fused_glu_close(
        label: &str,
        reference: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let reference = reference
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let output = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let max_reference = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let max_error = reference
            .iter()
            .zip(&output)
            .map(|(reference, output)| (reference - output).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error <= 0.02 + 0.02 * max_reference,
            "{label}: max error {max_error}, max reference {max_reference}"
        );
        Ok(())
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    fn copy_cuda_bf16(source: &Tensor, destination: &Tensor) -> Result<()> {
        use candle_core::{cuda::cudarc::driver::sys, Storage};
        use half::bf16;

        if source.dims() != destination.dims()
            || source.dtype() != DType::BF16
            || destination.dtype() != DType::BF16
            || !source.device().same_device(destination.device())
        {
            candle_core::bail!("CUDA BF16 test copy requires matching tensors")
        }
        let Device::Cuda(dev) = source.device() else {
            candle_core::bail!("CUDA BF16 test copy requires CUDA tensors")
        };
        let stream = dev.cuda_stream();
        let (source_storage, source_layout) = source.storage_and_layout();
        let Storage::Cuda(source_storage) = &*source_storage else {
            unreachable!()
        };
        let (destination_storage, destination_layout) = destination.storage_and_layout();
        let Storage::Cuda(destination_storage) = &*destination_storage else {
            unreachable!()
        };
        let (source_ptr, source_guard) = crate::utils::slice_ptr_on_stream(
            source_storage.as_cuda_slice::<bf16>()?,
            source_layout.start_offset(),
            &stream,
        );
        let (destination_ptr, destination_guard) = crate::utils::slice_ptr_on_stream(
            destination_storage.as_cuda_slice::<bf16>()?,
            destination_layout.start_offset(),
            &stream,
        );
        let status = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                destination_ptr,
                source_ptr,
                source.elem_count() * std::mem::size_of::<bf16>(),
                stream.cu_stream(),
            )
        };
        drop((source_guard, destination_guard));
        if status != sys::cudaError_enum::CUDA_SUCCESS {
            candle_core::bail!("CUDA BF16 test copy failed: {status:?}")
        }
        Ok(())
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn deepgemm_fused_glu_matches_unfused_sm90() -> Result<()> {
        const SPLIT_SIZE: usize = 256;
        const OUTPUT_FEATURES: usize = 256;
        const ROW_COUNTS: [usize; 8] = [1, 3, 4, 5, 16, 33, 129, 257];

        let dev = Device::new_cuda(0)?;
        let layer = deepgemm_fused_glu_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
        for rows in ROW_COUNTS {
            let input = deepgemm_fused_glu_test_input(rows, SPLIT_SIZE, rows, &dev)?;
            let intermediate =
                crate::utils::fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?;
            let reference = layer.forward(&intermediate)?;
            let output = layer
                .try_forward_fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?
                .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
            dev.synchronize()?;
            assert_eq!(output.dims(), [rows, OUTPUT_FEATURES]);
            assert_deepgemm_fused_glu_close(&format!("rows={rows}"), &reference, &output)?;
        }

        let input = deepgemm_fused_glu_test_input(6, SPLIT_SIZE, 91, &dev)?.reshape((2, 3, 512))?;
        let intermediate =
            crate::utils::fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?;
        let reference = layer.forward(&intermediate)?;
        let output = layer
            .try_forward_fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
        dev.synchronize()?;
        assert_eq!(output.dims(), [2, 3, OUTPUT_FEATURES]);
        assert_deepgemm_fused_glu_close("rank=3", &reference, &output)
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn deepgemm_consumes_fused_rms_norm_group_major_activation() -> Result<()> {
        use float8::F8E4M3;

        const ROWS: usize = 5;
        const FEATURES: usize = 256;
        const OUTPUT_FEATURES: usize = 256;
        const EPSILON: f32 = 1.0e-6;

        let device = Device::new_cuda(0)?;
        let layer = deepgemm_fused_glu_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
        let input_values = (0..ROWS * FEATURES)
            .map(|index| ((index * 17 + index / FEATURES * 13) % 79) as f32 / 23.0 - 1.5)
            .collect::<Vec<_>>();
        let residual_values = (0..ROWS * FEATURES)
            .map(|index| ((index * 11 + index / FEATURES * 7) % 47) as f32 / 31.0 - 0.7)
            .collect::<Vec<_>>();
        let norm_weight_values = (0..FEATURES)
            .map(|column| 0.8 + (column % 29) as f32 / 64.0)
            .collect::<Vec<_>>();
        let input =
            Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
        let residual =
            Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
        let norm_weight =
            Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
        let scheme = layer
            .activation_quantization_scheme_for(&input)
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM activation scheme is unavailable"))?;
        let scale_layout = layer
            .preferred_activation_scale_layout_for(&input)
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM scale layout is unavailable"))?;
        assert_eq!(
            scale_layout,
            ActivationScaleLayout::GroupMajor {
                row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
            }
        );
        let fused = fused_add_rms_norm_quantized(
            &input,
            &residual,
            &norm_weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        assert_eq!(fused.activation().scales().dims(), &[2, 8]);
        let output = layer.forward_quantized(fused.activation())?;
        device.synchronize()?;

        let activation = fused
            .activation()
            .quantized()
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<F8E4M3>()?;
        let activation_scales = fused
            .activation()
            .scales()
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let weight = ops::fp8_blockwise_dequantize(
            &layer.weight,
            &layer.weight_scale_inv,
            vec![128, 128],
            DType::F32,
        )?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
        let output = output
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let mut max_reference = 0.0f32;
        let mut max_error = 0.0f32;
        for row in 0..ROWS {
            for output_column in 0..OUTPUT_FEATURES {
                let mut reference = 0.0f32;
                for column in 0..FEATURES {
                    let activation_value = activation[row * FEATURES + column].to_f32()
                        * activation_scales[column / 128][row];
                    reference += activation_value * weight[output_column][column];
                }
                max_reference = max_reference.max(reference.abs());
                max_error = max_error.max((output[row][output_column] - reference).abs());
            }
        }
        assert!(
            max_error <= 0.02 + 0.02 * max_reference,
            "DeepGEMM group-major prequantized output error {max_error}, max reference {max_reference}"
        );
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    fn cutile_test_layer(
        dev: &Device,
        output_features: usize,
        input_features: usize,
    ) -> Result<BlockwiseFP8Linear> {
        const BLOCK_SIZE: usize = 128;

        let weight_values = (0..output_features * input_features)
            .map(|index| {
                let row = index / input_features;
                let column = index % input_features;
                let block = row / BLOCK_SIZE * (input_features / BLOCK_SIZE) + column / BLOCK_SIZE;
                let amplitude = [0.04, 0.18, 0.75, 1.6][block % 4];
                let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
                value * amplitude
            })
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weight_values, (output_features, input_features), dev)?
            .to_dtype(DType::BF16)?;
        let (weight, weight_scale_inv) =
            ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight,
            weight_scale_inv,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        })?;
        assert!(matches!(
            &layer.provider,
            BlockwiseFp8Provider::TensorCoreGemv
        ));
        Ok(layer)
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_w8a16_preserves_empty_leading_dimensions() -> Result<()> {
        const FEATURES: usize = 256;

        let device = Device::new_cuda(0)?;
        let weight = Tensor::zeros((FEATURES, FEATURES), DType::BF16, &device)?;
        let (weight, weight_scale_inv) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;
        let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
            weight,
            weight_scale_inv,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![128, 128],
            activation_scheme: None,
        })?;
        assert!(matches!(layer.provider, BlockwiseFp8Provider::CutileW8A16));
        let output = layer.forward(&Tensor::zeros((2, 0, FEATURES), DType::BF16, &device)?)?;
        assert_eq!(output.dims(), [2, 0, FEATURES]);
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    fn assert_close(label: &str, reference: &Tensor, output: &Tensor) -> Result<()> {
        let reference = reference.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let output = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let max_error = output
            .sub(&reference)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_error <= 0.02 + 0.02 * max_reference,
            "{label}: error {max_error}, max reference {max_reference}"
        );
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fused_glu_matches_unfused() -> Result<()> {
        const OUTPUT_FEATURES: usize = 256;
        const SPLIT_SIZE: usize = 512;

        let dev = Device::new_cuda(0)?;
        let layer = cutile_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
        let input_of = |shape: &[usize], phase: usize| -> Result<Tensor> {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index * 13 + phase * 7) % 89) as f32 / 29.0 - 1.4)
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, &dev)?.to_dtype(DType::BF16)
        };
        for shape in [
            &[64usize, 2 * SPLIT_SIZE][..],
            &[300, 2 * SPLIT_SIZE],
            &[2, 40, 2 * SPLIT_SIZE],
        ] {
            let input = input_of(shape, shape.len())?;
            let reference = layer.forward(&crate::utils::fused_split_glu(
                &input,
                SPLIT_SIZE,
                GluActivationType::Silu,
            )?)?;
            let output = layer
                .try_forward_fused_split_glu(&input, SPLIT_SIZE, GluActivationType::Silu)?
                .ok_or_else(|| candle_core::Error::msg("cuTile fused GLU was not selected"))?;
            assert_eq!(output.dims(), reference.dims());
            assert_close(&format!("shape {shape:?}"), &reference, &output)?;
        }
        let small = input_of(&[8, 2 * SPLIT_SIZE], 5)?;
        assert!(layer
            .try_forward_fused_split_glu(&small, SPLIT_SIZE, GluActivationType::Silu)?
            .is_none());
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels, feature = "cutile"))]
    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_consumes_fused_rms_norm_group_major_activation() -> Result<()> {
        const ROWS: usize = 300;
        const FEATURES: usize = 256;
        const OUTPUT_FEATURES: usize = 256;
        const EPSILON: f32 = 1.0e-6;

        let device = Device::new_cuda(0)?;
        let layer = cutile_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
        let input_values = (0..ROWS * FEATURES)
            .map(|index| ((index * 17 + index / FEATURES * 13) % 79) as f32 / 23.0 - 1.5)
            .collect::<Vec<_>>();
        let residual_values = (0..ROWS * FEATURES)
            .map(|index| ((index * 11 + index / FEATURES * 7) % 47) as f32 / 31.0 - 0.7)
            .collect::<Vec<_>>();
        let norm_weight_values = (0..FEATURES)
            .map(|column| 0.8 + (column % 29) as f32 / 64.0)
            .collect::<Vec<_>>();
        let input =
            Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
        let residual =
            Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?;
        let norm_weight =
            Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
        assert_eq!(
            layer.preferred_activation_scale_layout_for(&input.narrow(0, 0, 8)?),
            Some(ActivationScaleLayout::RowMajor)
        );
        let scheme = layer
            .activation_quantization_scheme_for(&input)
            .ok_or_else(|| candle_core::Error::msg("cuTile activation scheme is unavailable"))?;
        let scale_layout = layer
            .preferred_activation_scale_layout_for(&input)
            .ok_or_else(|| candle_core::Error::msg("cuTile scale layout is unavailable"))?;
        assert_eq!(
            scale_layout,
            ActivationScaleLayout::GroupMajor {
                row_alignment: std::num::NonZeroUsize::new(crate::cutile::FP8_GEMM_BLOCK_ROWS)
                    .unwrap(),
            }
        );
        let fused = fused_add_rms_norm_quantized(
            &input,
            &residual,
            &norm_weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        assert_eq!(fused.activation().scales().dims(), &[2, 384]);
        let output = layer.forward_quantized(fused.activation())?;
        let residual_ref = (input.to_dtype(DType::F32)? + residual.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)?;
        assert_close("residual", &residual_ref, fused.residual())?;
        // the GEMM must read the producer's padded storage right: compare against the dequantized
        // activation it was handed rather than re-quantizing a bf16 normalization
        let cpu = Device::Cpu;
        let values = fused
            .activation()
            .quantized()
            .to_device(&cpu)?
            .to_dtype(DType::F32)?;
        let row_scales = fused
            .activation()
            .scales()
            .to_device(&cpu)?
            .t()?
            .narrow(0, 0, ROWS)?
            .contiguous()?;
        let dequantized = values
            .reshape((ROWS, FEATURES / 128, 128))?
            .broadcast_mul(&row_scales.reshape((ROWS, FEATURES / 128, 1))?)?
            .reshape((ROWS, FEATURES))?;
        let weight = ops::fp8_blockwise_dequantize(
            &layer.weight,
            &layer.weight_scale_inv,
            vec![128, 128],
            DType::F32,
        )?
        .to_device(&cpu)?;
        assert_close("output", &dequantized.matmul(&weight.t()?)?, &output)?;
        Ok(())
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn deepgemm_fused_rms_norm_cuda_graph_replays_sm90() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;
        use half::bf16;

        const ROWS: usize = 5;
        const FEATURES: usize = 256;
        const OUTPUT_FEATURES: usize = 256;
        const EPSILON: f32 = 1.0e-6;

        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda_device) = &device else {
            unreachable!()
        };
        let stream = cuda_device.cuda_stream();
        let layer = deepgemm_fused_glu_test_layer(&device, OUTPUT_FEATURES, FEATURES)?;
        let norm_weight_values = (0..FEATURES)
            .map(|column| 0.8 + (column % 29) as f32 / 64.0)
            .collect::<Vec<_>>();
        let norm_weight =
            Tensor::from_vec(norm_weight_values, FEATURES, &device)?.to_dtype(DType::BF16)?;
        let make_inputs = |phase: usize| -> Result<(Tensor, Tensor)> {
            let input_values = (0..ROWS * FEATURES)
                .map(|index| {
                    ((index * 17 + index / FEATURES * 13 + phase * 19) % 79) as f32 / 23.0 - 1.5
                })
                .collect::<Vec<_>>();
            let residual_values = (0..ROWS * FEATURES)
                .map(|index| {
                    ((index * 11 + index / FEATURES * 7 + phase * 23) % 47) as f32 / 31.0 - 0.7
                })
                .collect::<Vec<_>>();
            Ok((
                Tensor::from_vec(input_values, (ROWS, FEATURES), &device)?.to_dtype(DType::BF16)?,
                Tensor::from_vec(residual_values, (ROWS, FEATURES), &device)?
                    .to_dtype(DType::BF16)?,
            ))
        };
        let (input_a, residual_a) = make_inputs(7)?;
        let (input_b, residual_b) = make_inputs(53)?;
        let make_reference = |input: &Tensor, residual: &Tensor| -> Result<(Tensor, Tensor)> {
            let residual_output = (input + residual)?;
            let normalized =
                candle_nn::ops::rms_norm(&residual_output.contiguous()?, &norm_weight, EPSILON)?;
            let output = layer.forward(&normalized)?;
            Ok((residual_output, output))
        };
        let (reference_residual_a, reference_output_a) = make_reference(&input_a, &residual_a)?;
        let (reference_residual_b, reference_output_b) = make_reference(&input_b, &residual_b)?;

        let graph_input = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
        let graph_residual = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
        let graph_residual_output = Tensor::zeros((ROWS, FEATURES), DType::BF16, &device)?;
        let graph_output = Tensor::zeros((ROWS, OUTPUT_FEATURES), DType::BF16, &device)?;
        copy_cuda_bf16(&input_a, &graph_input)?;
        copy_cuda_bf16(&residual_a, &graph_residual)?;
        let scheme = layer
            .activation_quantization_scheme_for(&graph_input)
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM activation scheme is unavailable"))?;
        let scale_layout = layer
            .preferred_activation_scale_layout_for(&graph_input)
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM scale layout is unavailable"))?;
        assert_eq!(
            scale_layout,
            ActivationScaleLayout::GroupMajor {
                row_alignment: std::num::NonZeroUsize::new(4).unwrap(),
            }
        );
        let warmup = fused_add_rms_norm_quantized(
            &graph_input,
            &graph_residual,
            &norm_weight,
            EPSILON,
            scheme,
            scale_layout,
        )?;
        assert_eq!(warmup.activation().scales().dims(), &[2, 8]);
        let warmup_output = layer.forward_quantized(warmup.activation())?;
        drop((warmup_output, warmup));
        device.synchronize()?;

        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = (|| -> Result<()> {
            let fused = fused_add_rms_norm_quantized(
                &graph_input,
                &graph_residual,
                &norm_weight,
                EPSILON,
                scheme,
                scale_layout,
            )?;
            let output = layer.forward_quantized(fused.activation())?;
            copy_cuda_bf16(fused.residual(), &graph_residual_output)?;
            copy_cuda_bf16(&output, &graph_output)?;
            drop((output, fused));
            Ok(())
        })();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;

        let expected_residual_a = reference_residual_a
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let expected_residual_b = reference_residual_b
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<bf16>()?;
        let replays = [
            (
                &input_a,
                &residual_a,
                &reference_output_a,
                &expected_residual_a,
            ),
            (
                &input_a,
                &residual_a,
                &reference_output_a,
                &expected_residual_a,
            ),
            (
                &input_b,
                &residual_b,
                &reference_output_b,
                &expected_residual_b,
            ),
            (
                &input_a,
                &residual_a,
                &reference_output_a,
                &expected_residual_a,
            ),
        ];
        let mut first_output = None;
        for (replay, (input, residual, reference_output, expected_residual)) in
            replays.into_iter().enumerate()
        {
            copy_cuda_bf16(input, &graph_input)?;
            copy_cuda_bf16(residual, &graph_residual)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            stream
                .synchronize()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;

            let actual_residual = graph_residual_output
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            assert_eq!(actual_residual.as_slice(), expected_residual.as_slice());
            assert_deepgemm_fused_glu_close(
                &format!("fused RMSNorm graph replay {replay}"),
                reference_output,
                &graph_output,
            )?;
            let actual_output = graph_output
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            match replay {
                0 => first_output = Some(actual_output),
                1 | 3 => assert_eq!(
                    actual_output.as_slice(),
                    first_output.as_ref().unwrap().as_slice()
                ),
                2 => assert_ne!(
                    actual_output.as_slice(),
                    first_output.as_ref().unwrap().as_slice()
                ),
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    #[cfg(all(
        feature = "cuda",
        has_cutlass_fp8_sm90_kernels,
        has_deepgemm_fp8_sm90_provider
    ))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn deepgemm_fused_glu_cuda_graph_replays_sm90() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;

        const ROWS: usize = 16;
        const SPLIT_SIZE: usize = 256;
        const OUTPUT_FEATURES: usize = 256;
        const REPLAY_COUNT: usize = 3;

        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!()
        };
        let stream = cuda_dev.cuda_stream();
        let layer = deepgemm_fused_glu_test_layer(&dev, OUTPUT_FEATURES, SPLIT_SIZE)?;
        let input_a = deepgemm_fused_glu_test_input(ROWS, SPLIT_SIZE, 7, &dev)?;
        let input_b = deepgemm_fused_glu_test_input(ROWS, SPLIT_SIZE, 53, &dev)?;
        let graph_input = input_a.copy()?;
        let graph_output = Tensor::zeros((ROWS, OUTPUT_FEATURES), DType::BF16, &dev)?;
        let reference_a = layer.forward(&crate::utils::fused_split_glu(
            &input_a,
            SPLIT_SIZE,
            GluActivationType::Silu,
        )?)?;
        let reference_b = layer.forward(&crate::utils::fused_split_glu(
            &input_b,
            SPLIT_SIZE,
            GluActivationType::Silu,
        )?)?;
        let warmup = layer
            .try_forward_fused_split_glu(&graph_input, SPLIT_SIZE, GluActivationType::Silu)?
            .ok_or_else(|| candle_core::Error::msg("DeepGEMM fused GLU was not selected"))?;
        drop(warmup);
        dev.synchronize()?;

        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = layer
            .try_forward_fused_split_glu(&graph_input, SPLIT_SIZE, GluActivationType::Silu)
            .and_then(|output| {
                let output = output.ok_or_else(|| {
                    candle_core::Error::msg("DeepGEMM fused GLU was not selected")
                })?;
                copy_cuda_bf16(&output, &graph_output)?;
                drop(output);
                Ok(())
            });
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;

        for replay in 0..REPLAY_COUNT {
            let (source, reference) = if replay == 1 {
                (&input_b, &reference_b)
            } else {
                (&input_a, &reference_a)
            };
            let churn = (0..32)
                .map(|_| Tensor::zeros((ROWS, SPLIT_SIZE), DType::BF16, &dev))
                .collect::<Result<Vec<_>>>()?;
            drop(churn);
            copy_cuda_bf16(source, &graph_input)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            stream
                .synchronize()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            assert_deepgemm_fused_glu_close(
                &format!("graph replay {replay}"),
                reference,
                &graph_output,
            )?;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn deepgemm_production_shapes_use_linear_dispatch_sm90() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const M: usize = 512;
        const OUTPUT_TOLERANCE: f32 = 0.02;
        const SHAPES: [(usize, usize); 5] = [
            (16_384, 5_120),
            (14_336, 5_120),
            (5_120, 6_144),
            (34_816, 5_120),
            (5_120, 17_408),
        ];

        let dev = Device::new_cuda(0)?;
        for (n, k) in SHAPES {
            let weight = Tensor::ones((n, k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
            let weight_scales = Tensor::ones((n / BLOCK_SIZE, k / BLOCK_SIZE), DType::F32, &dev)?
                .affine(1.0 / k as f64, 0.0)?;
            let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
                weight,
                weight_scale_inv: weight_scales,
                bias: None,
                dequant_dtype: DType::BF16,
                weight_block_size: vec![BLOCK_SIZE, BLOCK_SIZE],
                activation_scheme: Some(Fp8ActivationScheme::Dynamic),
            })?;
            assert!(matches!(
                &layer.provider,
                BlockwiseFp8Provider::DeepGemmSm90(_)
            ));

            let input = Tensor::ones((M, k), DType::BF16, &dev)?;
            let output = layer.forward_raw(&input)?;
            dev.synchronize()?;
            let max_error = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|value| (value - 1.0).abs())
                .fold(0.0, f32::max);
            assert!(
                max_error <= OUTPUT_TOLERANCE,
                "M={M}, N={n}, K={k}: output error {max_error}"
            );
        }
        Ok(())
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
        let layer = crate::fp8_config::fp8_checkpoint_linear_b(
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
        let layer = crate::fp8_config::fp8_checkpoint_linear_b(
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
        let err = crate::fp8_config::fp8_checkpoint_linear_b(
            4,
            8,
            &fp8_config(&["bar"]),
            false,
            Shard::default(),
            vb,
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing FP8 weight scale"));
        Ok(())
    }
}
