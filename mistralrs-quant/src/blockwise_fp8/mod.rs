use std::sync::{atomic::AtomicUsize, Arc};

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
    QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedActivation, QuantizedSerde, Shard,
    UnquantLinear,
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
    #[cfg(cuda_cutile)]
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
// The fused GLU quantizer kernel emits one F32 scale per 128 columns.
#[cfg(feature = "cuda")]
const FUSED_GLU_QUANT_BLOCK_COLUMNS: usize = 128;
#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
static TENSOR_CORE_GEMV_PROVIDER_LOG: std::sync::Once = std::sync::Once::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DynamicFp8Provider {
    Legacy,
    CutlassSm90,
    TensorCoreGemv,
}

// The tensor-core GEMV covers non-Hopper parts; on SM90 the CUTLASS GEMM serves every row count.
fn select_dynamic_provider(cutlass_sm90: bool, tensor_core_gemv: bool) -> DynamicFp8Provider {
    if cutlass_sm90 {
        DynamicFp8Provider::CutlassSm90
    } else if tensor_core_gemv {
        DynamicFp8Provider::TensorCoreGemv
    } else {
        DynamicFp8Provider::Legacy
    }
}

impl BlockwiseFp8Provider {
    fn supports_shared_activation(&self) -> bool {
        match self {
            Self::Legacy => false,
            #[cfg(cuda_cutile)]
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
        #[cfg(cuda_cutile)]
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
        let cutlass_sm90 = ops::cutlass_fp8_blockwise_supported(
            &self.weight,
            &self.weight_scale_inv,
            &self.weight_block_size,
        );
        #[cfg(not(all(feature = "cuda", has_cutlass_fp8_sm90_kernels)))]
        let cutlass_sm90 = false;
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        let tensor_core_gemv = self.dequant_dtype == DType::BF16
            && mma::weight_supported(
                &self.weight,
                &self.weight_scale_inv,
                &self.weight_block_size,
            );
        #[cfg(not(all(feature = "cuda", has_blockwise_fp8_kernels)))]
        let tensor_core_gemv = false;

        match select_dynamic_provider(cutlass_sm90, tensor_core_gemv) {
            DynamicFp8Provider::Legacy => {}
            #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
            DynamicFp8Provider::CutlassSm90 => {
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
            #[cfg(not(all(feature = "cuda", has_cutlass_fp8_sm90_kernels)))]
            DynamicFp8Provider::CutlassSm90 => unreachable!(),
            #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
            DynamicFp8Provider::TensorCoreGemv => {
                self.provider = BlockwiseFp8Provider::TensorCoreGemv;
                #[cfg(feature = "cutile")]
                if let (Device::Cuda(dev), Ok((n, k))) = (self.weight.device(), self.weight.dims2())
                {
                    if crate::cutile::fp8_gemm_supported(dev, n, k) {
                        crate::cutile::register_fp8_gemm_shape(
                            &self.weight,
                            &self.weight_scale_inv,
                        );
                    }
                }
                TENSOR_CORE_GEMV_PROVIDER_LOG.call_once(|| {
                    tracing::info!(
                        "Using FP8 tensor-core W8A8 GEMV for decode with dynamic 1x128 activations"
                    );
                });
            }
            #[cfg(not(all(feature = "cuda", has_blockwise_fp8_kernels)))]
            DynamicFp8Provider::TensorCoreGemv => unreachable!(),
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

impl BlockwiseFP8Linear {
    #[cfg(feature = "cuda")]
    fn forward_packed_glu(
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

    #[cfg(feature = "cuda")]
    fn forward_split_glu(
        &self,
        gate: &Tensor,
        value: &Tensor,
        activation: GluActivationType,
    ) -> Result<Option<Tensor>> {
        if gate.dtype() != DType::BF16 || !gate.device().is_cuda() {
            return Ok(None);
        }
        let Some(scheme) = self.activation_quantization_scheme_for(value) else {
            return Ok(None);
        };
        if scheme.dtype != DType::F8E4M3
            || scheme.block_shape != [1, FUSED_GLU_QUANT_BLOCK_COLUMNS]
            || scheme.scale_encoding != crate::ActivationScaleEncoding::F32
        {
            return Ok(None);
        }
        let Some(scale_layout @ ActivationScaleLayout::GroupMajor { row_alignment }) =
            self.preferred_activation_scale_layout_for(value)
        else {
            return Ok(None);
        };
        let Some((&columns, batch_dims)) = value.dims().split_last() else {
            return Ok(None);
        };
        let rows = batch_dims
            .iter()
            .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
            .ok_or_else(|| {
                candle_core::Error::msg("fused GLU activation row count overflows usize")
            })?;
        if rows == 0 || columns == 0 {
            return Ok(None);
        }
        let scale_stride_m = crate::aligned_activation_scale_rows(rows, row_alignment)?;
        let Some((quantized, scales)) = crate::utils::fused_glu_quantized_bf16(
            gate,
            value,
            scheme.block_shape[1],
            scale_stride_m,
            activation,
        )?
        else {
            return Ok(None);
        };
        let quantized = QuantizedActivation::new_with_scale_layout(
            quantized,
            scales,
            value.dims().to_vec(),
            value.dtype(),
            scheme,
            scale_layout,
        )?;
        self.forward_quantized(&quantized).map(Some)
    }
}

impl QuantMethod for BlockwiseFP8Linear {
    #[cfg(feature = "metal")]
    fn prepare_metal_kernels(&self, activation_dtype: DType) -> Result<()> {
        if activation_dtype == self.dequant_dtype
            && crate::metal_fp8::supported(&self.weight, &self.weight_scale_inv, self.dequant_dtype)
        {
            if let [block_n, block_k] = self.weight_block_size.as_slice() {
                crate::metal_fp8::prepare(&self.weight, [*block_n, *block_k], activation_dtype)?;
            }
        }
        Ok(())
    }

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
    fn embedding_table(&self) -> Option<crate::EmbeddingTable<'_>> {
        use crate::embedding::{RowCodec, ScaleGrid, ScaleType};
        let (rows, columns) = self.weight.dims2().ok()?;
        let &[block_rows, block_columns] = self.weight_block_size.as_slice() else {
            return None;
        };
        let grid = ScaleGrid {
            ty: ScaleType::from_dtype(self.weight_scale_inv.dtype()).ok()?,
            group: block_columns,
            rows_per_scale: block_rows,
        };
        Some(
            crate::EmbeddingTable::new(RowCodec::Fp8(Some(grid)), rows, columns, &self.weight)
                .with_scales(&self.weight_scale_inv),
        )
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        if self.weight.rank() == 3 {
            let experts = (0..self.weight.dim(0)?)
                .map(|expert| {
                    ops::fp8_blockwise_dequantize(
                        &self.weight.get(expert)?,
                        &self.weight_scale_inv.get(expert)?,
                        self.weight_block_size.to_vec(),
                        self.dequant_dtype,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Tensor::stack(&experts, 0);
        }
        ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let [block_n, block_k] = self.weight_block_size.as_slice() {
            if let Some(result) = crate::metal_fp8::linear(
                x,
                &self.weight,
                &self.weight_scale_inv,
                [*block_n, *block_k],
                self.dequant_dtype,
            )? {
                return match &self.bias {
                    Some(bias) => result.broadcast_add(bias),
                    None => Ok(result),
                };
            }
        }

        #[cfg(cuda_cutile)]
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
    fn routed_weights(&self) -> Option<crate::RoutedWeights> {
        let (experts, n, k) = self.weight.dims3().ok()?;
        let [block_rows, block_cols] = self.weight_block_size[..] else {
            return None;
        };
        Some(crate::RoutedWeights::new(
            crate::RoutedCodec::Fp8 {
                block_rows,
                block_cols,
            },
            self.weight.clone(),
            self.weight_scale_inv.clone(),
            None,
            self.bias.clone(),
            [experts, n, k],
        ))
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(routed) = self.routed_weights() {
            if let Some(output) = crate::try_routed_gather(x, indices, &routed)? {
                return Ok(output);
            }
        }
        #[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]
        if let [block_n, block_k] = *self.weight_block_size.as_slice() {
            let view = crate::moe::grouped_fp8::Fp8WeightView {
                weight: &self.weight,
                scales: &self.weight_scale_inv,
                encoding: crate::moe::grouped_fp8::Fp8ScaleEncoding::F32,
                block: [block_n, block_k],
            };
            if let Some(result) = crate::moe::grouped_fp8::gather(x, indices, view)? {
                return crate::routed_gemv::add_routed_bias(result, self.bias.as_ref(), indices);
            }
        }
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
                return crate::routed_gemv::add_routed_bias(result, self.bias.as_ref(), indices);
            }
        }

        let weight = self.dequantize_w()?;
        crate::routed_gemv::dense_gather(&weight, self.bias.as_ref(), x, indices)
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
                scale_encoding: crate::ActivationScaleEncoding::F32,
                block_shape: [1, self.weight_block_size[1]],
            });
        }
        #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
        {
            Some(ActivationQuantizationScheme {
                dtype: DType::F8E4M3,
                scale_encoding: crate::ActivationScaleEncoding::F32,
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

    fn forward_glu(
        &self,
        input: crate::GluInput<'_>,
        activation: crate::GluActivationType,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "cuda")]
        match input {
            crate::GluInput::Packed { input, split_size } => {
                self.forward_packed_glu(input, split_size, activation)
            }
            crate::GluInput::Split { gate, value } => {
                self.forward_split_glu(gate, value, activation)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (input, activation);
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
#[path = "tests.rs"]
mod tests;
