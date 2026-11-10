use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationQuantizationScheme {
    pub dtype: DType,
    pub block_shape: [usize; 2],
    pub scale_encoding: ActivationScaleEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationScaleEncoding {
    F32,
    E4M3,
    E8M0,
}

impl ActivationScaleEncoding {
    pub(crate) fn storage_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::E4M3 => DType::F8E4M3,
            Self::E8M0 => DType::U8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationScaleLayout {
    RowMajor,
    GroupMajor { row_alignment: NonZeroUsize },
}

pub(crate) fn aligned_activation_scale_rows(
    rows: usize,
    row_alignment: NonZeroUsize,
) -> Result<usize> {
    let alignment = row_alignment.get();
    rows.checked_add(alignment - 1)
        .ok_or_else(|| {
            candle_core::Error::msg("group-major activation scale row count overflows usize")
        })
        .map(|rows| rows / alignment * alignment)
}

#[derive(Clone, Debug)]
pub struct QuantizedActivation {
    pub(crate) quantized: Tensor,
    pub(crate) scales: Tensor,
    pub(crate) source_shape: Vec<usize>,
    pub(crate) source_dtype: DType,
    pub(crate) scheme: ActivationQuantizationScheme,
    pub(crate) scale_layout: ActivationScaleLayout,
    pub(crate) global_scale: Option<f32>,
}

impl QuantizedActivation {
    pub fn new(
        quantized: Tensor,
        scales: Tensor,
        source_shape: Vec<usize>,
        source_dtype: DType,
        scheme: ActivationQuantizationScheme,
    ) -> Result<Self> {
        Self::new_with_scale_layout(
            quantized,
            scales,
            source_shape,
            source_dtype,
            scheme,
            ActivationScaleLayout::RowMajor,
        )
    }

    pub fn new_with_scale_layout(
        quantized: Tensor,
        scales: Tensor,
        source_shape: Vec<usize>,
        source_dtype: DType,
        scheme: ActivationQuantizationScheme,
        scale_layout: ActivationScaleLayout,
    ) -> Result<Self> {
        let [row_block, col_block] = scheme.block_shape;
        if row_block == 0 || col_block == 0 {
            candle_core::bail!("activation quantization block dimensions must be nonzero");
        }
        if quantized.dtype() != scheme.dtype {
            candle_core::bail!(
                "quantized activation has dtype {:?}, expected {:?}",
                quantized.dtype(),
                scheme.dtype
            );
        }
        if scales.dtype() != scheme.scale_encoding.storage_dtype() {
            candle_core::bail!(
                "quantized activation scales for {:?} must be {:?}, got {:?}",
                scheme.scale_encoding,
                scheme.scale_encoding.storage_dtype(),
                scales.dtype()
            );
        }
        if !quantized.device().same_device(scales.device()) {
            candle_core::bail!("quantized activation values and scales are on different devices");
        }
        let (rows, cols) = quantized.dims2()?;
        let source_cols = source_shape
            .last()
            .copied()
            .ok_or_else(|| candle_core::Error::msg("activation source shape cannot be empty"))?;
        let source_rows = source_shape[..source_shape.len() - 1]
            .iter()
            .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
            .ok_or_else(|| candle_core::Error::msg("activation source shape overflows usize"))?;
        if (rows, cols) != (source_rows, source_cols) {
            candle_core::bail!(
                "quantized activation shape ({rows}, {cols}) does not match source shape {:?}",
                source_shape
            );
        }
        let expected_scale_shape = match scale_layout {
            ActivationScaleLayout::RowMajor => (rows.div_ceil(row_block), cols.div_ceil(col_block)),
            ActivationScaleLayout::GroupMajor { row_alignment } => {
                if row_block != 1 {
                    candle_core::bail!(
                        "group-major activation scales require a row block size of 1, got {row_block}"
                    );
                }
                let scale_rows = aligned_activation_scale_rows(rows, row_alignment)?;
                (cols.div_ceil(col_block), scale_rows)
            }
        };
        if scales.dims2()? != expected_scale_shape {
            candle_core::bail!(
                "quantized activation scale shape {:?} does not match expected {:?} for {:?}",
                scales.dims(),
                expected_scale_shape,
                scale_layout
            );
        }
        Ok(Self {
            quantized,
            scales,
            source_shape,
            source_dtype,
            scheme,
            scale_layout,
            global_scale: None,
        })
    }

    pub fn new_nvfp4(
        quantized: Tensor,
        scales: Tensor,
        source: &Tensor,
        global_scale: f32,
    ) -> Result<Self> {
        if source.rank() < 2 || !matches!(source.dtype(), DType::BF16 | DType::F16) {
            candle_core::bail!("NVFP4 activation source must be rank >= 2 and BF16 or F16");
        }
        let columns = source.dim(candle_core::D::Minus1)?;
        if columns == 0 || !columns.is_multiple_of(NVFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "NVFP4 activation columns must be a nonzero multiple of {NVFP4_BLOCK_SIZE}"
            );
        }
        let rows = source.elem_count() / columns;
        if quantized.dtype() != DType::U8 || quantized.dims() != [rows, columns / 2] {
            candle_core::bail!("NVFP4 activation values must be packed U8 [rows, columns / 2]");
        }
        if scales.dtype() != DType::F8E4M3 || scales.dims() != [rows, columns / NVFP4_BLOCK_SIZE] {
            candle_core::bail!("NVFP4 activation block scales must be E4M3 [rows, columns / 16]");
        }
        if !quantized.device().same_device(source.device())
            || !scales.device().same_device(source.device())
        {
            candle_core::bail!("NVFP4 activation tensors must be on the same device");
        }
        if !global_scale.is_finite() || global_scale <= 0.0 {
            candle_core::bail!("NVFP4 activation global scale must be finite and positive");
        }
        Ok(Self {
            quantized,
            scales,
            source_shape: source.dims().to_vec(),
            source_dtype: source.dtype(),
            scheme: nvfp4::NVFP4_ACTIVATION_SCHEME,
            scale_layout: ActivationScaleLayout::RowMajor,
            global_scale: Some(global_scale),
        })
    }

    pub fn global_scale(&self) -> Option<f32> {
        self.global_scale
    }

    pub fn quantized(&self) -> &Tensor {
        &self.quantized
    }

    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    pub fn source_shape(&self) -> &[usize] {
        &self.source_shape
    }

    pub fn source_dtype(&self) -> DType {
        self.source_dtype
    }

    pub fn scheme(&self) -> ActivationQuantizationScheme {
        self.scheme
    }

    pub fn scale_layout(&self) -> ActivationScaleLayout {
        self.scale_layout
    }
}

#[derive(Clone, Debug)]
pub struct FusedRmsNormQuantized {
    pub(crate) residual: Tensor,
    pub(crate) activation: QuantizedActivation,
}

impl FusedRmsNormQuantized {
    pub fn new(residual: Tensor, activation: QuantizedActivation) -> Result<Self> {
        if residual.dtype() != activation.source_dtype()
            || residual.dims() != activation.source_shape()
        {
            candle_core::bail!(
                "fused RMSNorm residual shape {:?} and dtype {:?} do not match activation source shape {:?} and dtype {:?}",
                residual.dims(),
                residual.dtype(),
                activation.source_shape(),
                activation.source_dtype()
            );
        }
        if !residual
            .device()
            .same_device(activation.quantized().device())
        {
            candle_core::bail!("fused RMSNorm residual and activation are on different devices");
        }
        Ok(Self {
            residual,
            activation,
        })
    }

    pub fn residual(&self) -> &Tensor {
        &self.residual
    }

    pub fn activation(&self) -> &QuantizedActivation {
        &self.activation
    }

    pub fn into_parts(self) -> (Tensor, QuantizedActivation) {
        (self.residual, self.activation)
    }
}

/// How a layer's input activation scales are calibrated; layers share one quantized activation only when equal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationCalibration {
    Dynamic,
    /// `source` pins the scale tensor's storage when only that exact storage may be shared.
    Static {
        global_scale: f32,
        source: Option<candle_core::TensorId>,
    },
}

impl ActivationCalibration {
    pub fn global_scale(&self) -> Option<f32> {
        match self {
            Self::Dynamic => None,
            Self::Static { global_scale, .. } => Some(*global_scale),
        }
    }
}
