use mistralrs_core::{parse_isq_value, IsqType};

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
    fn as_str(self) -> &'static str {
        match self {
            Self::Two => "2",
            Self::Three => "3",
            Self::Four => "4",
            Self::Five => "5",
            Self::Six => "6",
            Self::Eight => "8",
        }
    }
}

/// Specifies how ISQ (in-situ quantization) should be configured.
///
/// Use [`IsqSetting::Auto`] to let the engine pick the best quantization type for
/// the target platform, or [`IsqSetting::Specific`] (via [`TextModelBuilder::with_isq`](crate::TextModelBuilder::with_isq))
/// to choose an exact type.
///
/// # Examples
///
/// ```no_run
/// # use mistralrs::*;
/// # async fn example() -> anyhow::Result<()> {
/// // Auto-select the best 4-bit quantization for the current platform:
/// let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
///     .with_auto_isq(IsqBits::Four)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IsqSetting {
    /// Auto-select the best ISQ type for the target platform at the given bit width.
    /// On Metal this selects AFQ variants; on CUDA/CPU this selects Q*K variants.
    Auto(IsqBits),
    /// Use a specific ISQ type directly.
    Specific(IsqType),
}

impl From<IsqType> for IsqSetting {
    fn from(isq: IsqType) -> Self {
        IsqSetting::Specific(isq)
    }
}

/// Resolve an [`IsqSetting`] to a concrete [`IsqType`] given the target device.
pub(crate) fn resolve_isq(
    setting: &IsqSetting,
    device: &candle_core::Device,
) -> anyhow::Result<IsqType> {
    match setting {
        IsqSetting::Auto(bits) => {
            parse_isq_value(bits.as_str(), Some(device)).map_err(|e| anyhow::anyhow!(e))
        }
        IsqSetting::Specific(ty) => Ok(*ty),
    }
}
