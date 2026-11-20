use super::*;

/// In-situ quantization type specifying the format to apply to model weights.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq, Serialize, Deserialize)]
pub enum IsqType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    HQQ8,
    HQQ4,
    // HQQ3,
    // HQQ2,
    // HQQ1,
    F8E4M3,
    AFQ8,
    AFQ6,
    AFQ4,
    AFQ3,
    AFQ2,
    F8Q8,
    MXFP4,
}

/// Target bit width for automatic ISQ quantization.
///
/// On Metal, these select AFQ variants; on CUDA/CPU, they select Q*K variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqBits {
    /// 2-bit quantization (AFQ2 on Metal, Q2K otherwise).
    Two,
    /// 3-bit quantization (AFQ3 on Metal, Q3K otherwise).
    Three,
    /// 4-bit quantization (AFQ4 on Metal, Q4K otherwise).
    Four,
    /// 5-bit quantization (Q5K on all platforms).
    Five,
    /// 6-bit quantization (AFQ6 on Metal, Q6K otherwise).
    Six,
    /// 8-bit quantization (AFQ8 on Metal, Q8_0 otherwise).
    Eight,
}

impl IsqBits {
    /// Resolve to the platform-appropriate `IsqType` for the given device.
    pub fn resolve(self, device: &Device) -> IsqType {
        match (self, device.is_metal()) {
            (Self::Two, true) => IsqType::AFQ2,
            (Self::Two, false) => IsqType::Q2K,
            (Self::Three, true) => IsqType::AFQ3,
            (Self::Three, false) => IsqType::Q3K,
            (Self::Four, true) => IsqType::AFQ4,
            (Self::Four, false) => IsqType::Q4K,
            (Self::Five, _) => IsqType::Q5K,
            (Self::Six, true) => IsqType::AFQ6,
            (Self::Six, false) => IsqType::Q6K,
            (Self::Eight, true) => IsqType::AFQ8,
            (Self::Eight, false) => IsqType::Q8_0,
        }
    }

    /// Return all platform variants, with the current platform's preferred variant first.
    /// On Metal, AFQ variants come first; on other platforms, GGUF/Q variants come first.
    pub fn expand(self) -> Vec<IsqType> {
        #[cfg(feature = "metal")]
        match self {
            Self::Two => vec![IsqType::AFQ2, IsqType::Q2K],
            Self::Three => vec![IsqType::AFQ3, IsqType::Q3K],
            Self::Four => vec![IsqType::AFQ4, IsqType::Q4K],
            Self::Five => vec![IsqType::Q5K],
            Self::Six => vec![IsqType::AFQ6, IsqType::Q6K],
            Self::Eight => vec![IsqType::AFQ8, IsqType::Q8_0],
        }
        #[cfg(not(feature = "metal"))]
        match self {
            Self::Two => vec![IsqType::Q2K, IsqType::AFQ2],
            Self::Three => vec![IsqType::Q3K, IsqType::AFQ3],
            Self::Four => vec![IsqType::Q4K, IsqType::AFQ4],
            Self::Five => vec![IsqType::Q5K],
            Self::Six => vec![IsqType::Q6K, IsqType::AFQ6],
            Self::Eight => vec![IsqType::Q8_0, IsqType::AFQ8],
        }
    }
}

impl TryFrom<&str> for IsqBits {
    type Error = ();
    fn try_from(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "2" => Ok(Self::Two),
            "3" => Ok(Self::Three),
            "4" => Ok(Self::Four),
            "5" => Ok(Self::Five),
            "6" => Ok(Self::Six),
            "8" => Ok(Self::Eight),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for IsqType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Q4_1 => write!(f, "q4_1"),
            Self::Q5_0 => write!(f, "q5_0"),
            Self::Q5_1 => write!(f, "q5_1"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q8_1 => write!(f, "q8_1"),
            Self::Q2K => write!(f, "q2k"),
            Self::Q3K => write!(f, "q3k"),
            Self::Q4K => write!(f, "q4k"),
            Self::Q5K => write!(f, "q5k"),
            Self::Q6K => write!(f, "q6k"),
            Self::Q8K => write!(f, "q8k"),
            Self::HQQ8 => write!(f, "hqq8"),
            Self::HQQ4 => write!(f, "hqq4"),
            Self::F8E4M3 => write!(f, "fp8"),
            Self::AFQ8 => write!(f, "afq8"),
            Self::AFQ6 => write!(f, "afq6"),
            Self::AFQ4 => write!(f, "afq4"),
            Self::AFQ3 => write!(f, "afq3"),
            Self::AFQ2 => write!(f, "afq2"),
            Self::F8Q8 => write!(f, "f8q8"),
            Self::MXFP4 => write!(f, "mxfp4"),
        }
    }
}

pub(crate) const AFFINE_METADATA_TENSOR_COUNT: usize = 2;

pub(crate) fn integer_pack_factor(dense_bytes: usize, packed_bytes: usize) -> usize {
    (dense_bytes / packed_bytes.max(1)).max(1)
}

pub(crate) fn block_pack_factor(block_elements: usize, dtype: DType, packed_bytes: usize) -> usize {
    let dtype_bytes = dtype.size_in_bytes();
    let mut factor = integer_pack_factor(block_elements * dtype_bytes, packed_bytes);
    while factor > 1 && block_elements / factor * dtype_bytes < packed_bytes {
        factor -= 1;
    }
    factor
}

pub(crate) fn afq_pack_factor(dtype: DType, bits: usize, group_size: usize) -> usize {
    affine_pack_factor(dtype, bits, group_size, dtype.size_in_bytes())
}

pub(crate) fn hqq_pack_factor(dtype: DType, bits: usize, group_size: usize) -> usize {
    affine_pack_factor(dtype, bits, group_size, DType::F32.size_in_bytes())
}

pub(crate) fn affine_pack_factor(
    dtype: DType,
    bits: usize,
    group_size: usize,
    metadata_element_bytes: usize,
) -> usize {
    let packed_bytes =
        (group_size * bits).div_ceil(8) + AFFINE_METADATA_TENSOR_COUNT * metadata_element_bytes;
    block_pack_factor(group_size, dtype, packed_bytes)
}

impl IsqType {
    pub fn promote_for_sensitive_tensor(self) -> Self {
        match self {
            Self::AFQ2 | Self::AFQ3 | Self::AFQ4 => Self::AFQ6,
            Self::AFQ6 | Self::AFQ8 => Self::AFQ8,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q4_0 | Self::Q4_1 => Self::Q6K,
            Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1 => Self::Q8_0,
            ty => ty,
        }
    }

    /// Integer factor used to estimate the weight-size reduction over the given dtype.
    pub fn pack_factor(&self, dtype: DType) -> usize {
        let ggml = |quantized_dtype: GgmlDType| {
            block_pack_factor(
                quantized_dtype.block_size(),
                dtype,
                quantized_dtype.type_size(),
            )
        };
        match self {
            Self::Q4_0 => ggml(GgmlDType::Q4_0),
            Self::Q4_1 => ggml(GgmlDType::Q4_1),
            Self::Q5_0 => ggml(GgmlDType::Q5_0),
            Self::Q5_1 => ggml(GgmlDType::Q5_1),
            Self::Q8_0 => ggml(GgmlDType::Q8_0),
            Self::Q8_1 => ggml(GgmlDType::Q8_1),
            Self::Q2K => ggml(GgmlDType::Q2K),
            Self::Q3K => ggml(GgmlDType::Q3K),
            Self::Q4K => ggml(GgmlDType::Q4K),
            Self::Q5K => ggml(GgmlDType::Q5K),
            Self::Q6K => ggml(GgmlDType::Q6K),
            Self::Q8K => ggml(GgmlDType::Q8K),
            Self::AFQ2 => afq_pack_factor(dtype, 2, AfqGroupSize::Low as usize),
            Self::AFQ3 => afq_pack_factor(dtype, 3, AfqGroupSize::Low as usize),
            Self::AFQ4 => afq_pack_factor(dtype, 4, AfqGroupSize::Low as usize),
            Self::AFQ6 => afq_pack_factor(dtype, 6, AfqGroupSize::Low as usize),
            Self::AFQ8 => afq_pack_factor(dtype, 8, AfqGroupSize::Low as usize),
            Self::F8Q8 => {
                block_pack_factor(f8q8::QK8_0, dtype, std::mem::size_of::<f8q8::BlockF8Q8>())
            }
            Self::HQQ4 => hqq_pack_factor(dtype, 4, hqq::ISQ_HQQ_GROUP_SIZE),
            Self::HQQ8 => hqq_pack_factor(dtype, 8, hqq::ISQ_HQQ_GROUP_SIZE),
            Self::F8E4M3 => 1,
            Self::MXFP4 => block_pack_factor(
                mxfp4::MXFP4_BLOCK_SIZE,
                dtype,
                mxfp4::MXFP4_BLOCK_SIZE * mxfp4::N_BITS / u8::BITS as usize
                    + DType::U8.size_in_bytes(),
            ),
        }
    }

    /// Only the K-quant formats consume importance weights; the rest quantize without them.
    pub fn supports_imatrix(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K
        )
    }

    pub fn supports_uqff(self) -> bool {
        matches!(
            self,
            Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5K
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q6K
                | Self::Q8K
                | Self::Q8_0
                | Self::Q8_1
                | Self::HQQ4
                | Self::HQQ8
                | Self::F8E4M3
                | Self::AFQ2
                | Self::AFQ3
                | Self::AFQ4
                | Self::AFQ6
                | Self::AFQ8
                | Self::F8Q8
                | Self::MXFP4
        )
    }

    pub fn supports_stacked_gather(self) -> bool {
        GgmlDType::try_from(self).is_ok()
            || matches!(
                self,
                Self::AFQ2
                    | Self::AFQ3
                    | Self::AFQ4
                    | Self::AFQ6
                    | Self::AFQ8
                    | Self::HQQ4
                    | Self::HQQ8
                    | Self::F8E4M3
                    | Self::MXFP4
            )
    }

    pub fn get_max_isq_cpu_threads(&self) -> Option<NonZeroUsize> {
        match self {
            /*IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            IsqType::HQQ4
            | IsqType::HQQ8
            | IsqType::AFQ2
            | IsqType::AFQ3
            | IsqType::AFQ4
            | IsqType::AFQ6
            | IsqType::AFQ8
            | IsqType::MXFP4 => {
                // Use 1 because our HQQ quantizes on the GPU
                Some(1.try_into().unwrap())
            }
            IsqType::F8E4M3 | IsqType::F8Q8 => None,
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
            | IsqType::Q8_1 => None,
        }
    }
}

impl TryFrom<IsqType> for GgmlDType {
    type Error = candle_core::Error;

    fn try_from(value: IsqType) -> Result<Self> {
        let tp = match value {
            IsqType::Q2K => Self::Q2K,
            IsqType::Q3K => Self::Q3K,
            IsqType::Q4K => Self::Q4K,
            IsqType::Q4_0 => Self::Q4_0,
            IsqType::Q4_1 => Self::Q4_1,
            IsqType::Q5K => Self::Q5K,
            IsqType::Q5_0 => Self::Q5_0,
            IsqType::Q5_1 => Self::Q5_1,
            IsqType::Q6K => Self::Q6K,
            IsqType::Q8K => Self::Q8K,
            IsqType::Q8_0 => Self::Q8_0,
            IsqType::Q8_1 => Self::Q8_1,
            _ => candle_core::bail!("Expected valid GGML ISQ type."),
        };
        #[cfg(feature = "cuda")]
        {
            if !matches!(
                tp,
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
            ) {
                candle_core::bail!("GGML ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`")
            }
        }
        Ok(tp)
    }
}

impl TryFrom<GgmlDType> for IsqType {
    type Error = candle_core::Error;

    fn try_from(value: GgmlDType) -> Result<Self> {
        match value {
            GgmlDType::Q2K => Ok(Self::Q2K),
            GgmlDType::Q3K => Ok(Self::Q3K),
            GgmlDType::Q4K => Ok(Self::Q4K),
            GgmlDType::Q5K => Ok(Self::Q5K),
            GgmlDType::Q6K => Ok(Self::Q6K),
            GgmlDType::Q4_0 => Ok(Self::Q4_0),
            GgmlDType::Q4_1 => Ok(Self::Q4_1),
            GgmlDType::Q5_0 => Ok(Self::Q5_0),
            GgmlDType::Q5_1 => Ok(Self::Q5_1),
            GgmlDType::Q8_0 => Ok(Self::Q8_0),
            GgmlDType::Q8_1 => Ok(Self::Q8_1),
            GgmlDType::Q8K => Ok(Self::Q8K),
            GgmlDType::BF16 | GgmlDType::F32 | GgmlDType::F16 => {
                candle_core::bail!("Expected valid GGML ISQ type.")
            }
        }
    }
}
