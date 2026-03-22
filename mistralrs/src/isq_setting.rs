pub use mistralrs_core::{IsqBits, IsqType};

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
        IsqSetting::Auto(bits) => Ok(bits.resolve(device)),
        IsqSetting::Specific(ty) => Ok(*ty),
    }
}
