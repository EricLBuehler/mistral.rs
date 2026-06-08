use std::{borrow::Cow, path::PathBuf, str::FromStr};
/// Wrapper around a `Cow<'a, [u8]>` buffer that implements
/// `safetensors::tensor::View`.
///
/// *Purpose*: lets us pass raw byte buffers to
/// `safetensors::serialize_to_file` without cloning them into a `Vec<u8>` or
/// converting to a higher‑level tensor type.
/// We expose the buffer as a 1‑D `u8` tensor of shape `[len]`.
#[derive(Clone)]
pub struct CowBytesView<'a> {
    data: Cow<'a, [u8]>,
    shape: [usize; 1],
}

impl<'a> CowBytesView<'a> {
    /// Convenience constructor.
    pub fn new(data: Cow<'a, [u8]>) -> Self {
        let len = data.len();
        Self { data, shape: [len] }
    }
}

impl safetensors::tensor::View for CowBytesView<'_> {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        // Serialize as raw bytes
        safetensors::tensor::Dtype::U8
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        assert!(matches!(self.data, Cow::Borrowed(_)));
        // Cloning a `Cow` is cheap (only clones the enum, not the data).
        self.data.clone()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_quant::{IsqBits, IsqType};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::pipeline::EmbeddingModulePaths;

pub(crate) const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";
pub const UQFF_MULTI_FILE_DELIMITER: &str = ";";

pub(crate) struct WeightLoadingState {
    pub(crate) from_uqff: bool,
    pub(crate) loading_isq: bool,
    pub(crate) immediate_isq: bool,
    pub(crate) write_uqff: bool,
}

pub(crate) enum WeightLoadingMode {
    Uqff,
    ImmediateIsq,
    PostLoadIsq,
    UqffSerialization,
    Plain,
}

impl From<WeightLoadingState> for WeightLoadingMode {
    fn from(state: WeightLoadingState) -> Self {
        if state.from_uqff {
            Self::Uqff
        } else if state.immediate_isq {
            Self::ImmediateIsq
        } else if state.loading_isq {
            Self::PostLoadIsq
        } else if state.write_uqff {
            Self::UqffSerialization
        } else {
            Self::Plain
        }
    }
}

impl WeightLoadingMode {
    pub(crate) fn message(self, target: &'static str) -> Cow<'static, str> {
        match self {
            Self::Uqff => {
                Cow::Borrowed("Loading residual weights and preparing UQFF placeholders.")
            }
            Self::ImmediateIsq => {
                Cow::Owned(format!("Loading {target} weights with immediate ISQ."))
            }
            Self::PostLoadIsq => Cow::Owned(format!(
                "Loading full-precision {target} weights for post-load ISQ."
            )),
            Self::UqffSerialization => {
                Cow::Owned(format!("Loading {target} weights for UQFF serialization."))
            }
            Self::Plain => Cow::Owned(format!("Loading {target} weights.")),
        }
    }
}

/// Parse ISQ value.
///
/// If the provided value is a valid integer (one of 2,3,4,5,6,8), the best quantization type will be chosen.
/// Note that the fallback is always a Q/K quantization but on Metal 2,3,4,6,8 uses the fast AFQ.
///
/// One of:
/// - `Q4_0`
/// - `Q4_1`
/// - `Q5_0`
/// - `Q5_1`
/// - `Q8_0`
/// - `Q8_1`
/// - `Q2K`
/// - `Q3K`
/// - `Q4K`
/// - `Q5K`
/// - `Q6K`
/// - `Q8K`
/// - `HQQ1`
/// - `HQQ2`
/// - `HQQ3`
/// - `HQQ4`
/// - `HQQ8`
/// - `AFQ2`
/// - `AFQ3`
/// - `AFQ4`
/// - `AFQ6`
/// - `AFQ8`
pub fn parse_isq_value(s: &str, device: Option<&Device>) -> Result<IsqType, String> {
    let lowered = s.to_lowercase();

    // Numeric shorthands resolve via IsqBits
    if let Ok(bits) = IsqBits::try_from(lowered.as_str()) {
        let tp = match device {
            Some(dev) => bits.resolve(dev),
            None => bits.resolve(&Device::Cpu),
        };
        #[cfg(feature = "cuda")]
        {
            // All IsqBits resolutions are CUDA-safe, so no extra check needed.
        }
        return Ok(tp);
    }

    let tp = match lowered.as_str() {
        "q4_0" => IsqType::Q4_0,
        "q4_1" => IsqType::Q4_1,
        "q5_0" => IsqType::Q5_0,
        "q5_1" => IsqType::Q5_1,
        "q8_0" => IsqType::Q8_0,
        "q8_1" => IsqType::Q8_1,
        "q2k" => IsqType::Q2K,
        "q3k" => IsqType::Q3K,
        "q4k" => IsqType::Q4K,
        "q5k" => IsqType::Q5K,
        "q6k" => IsqType::Q6K,
        "q8k" => IsqType::Q8K,
        "hqq8" => IsqType::HQQ8,
        "hqq4" => IsqType::HQQ4,
        "fp8" => IsqType::F8E4M3,
        "afq8" => IsqType::AFQ8,
        "afq6" => IsqType::AFQ6,
        "afq4" => IsqType::AFQ4,
        "afq3" => IsqType::AFQ3,
        "afq2" => IsqType::AFQ2,
        "f8q8" => IsqType::F8Q8,
        "mxfp4" => IsqType::MXFP4,
        // "hqq3" => IsqType::HQQ3,
        // "hqq2" => IsqType::HQQ2,
        // "hqq1" => IsqType::HQQ1,
        _ => return Err(format!("ISQ type {s} unknown, choose one of `2`, `3`, `4`, `5`, `6`, `8`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`, `MXFP4`.")),
    };
    #[cfg(feature = "cuda")]
    {
        if !matches!(
            tp,
            IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q8_0
                | IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q5K
                | IsqType::Q6K
                | IsqType::HQQ8
                | IsqType::HQQ4
                | IsqType::F8E4M3
                | IsqType::AFQ2
                | IsqType::AFQ3
                | IsqType::AFQ4
                | IsqType::AFQ6
                | IsqType::AFQ8
                | IsqType::F8Q8
                | IsqType::MXFP4 // | IsqType::HQQ3
                                 // | IsqType::HQQ2
                                 // | IsqType::HQQ1
        ) {
            return Err("ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`, `FP8`, `AFQ8`, `AFQ6`, `AFQ4`, `AFQ3`, `AFQ2`, `F8Q8`, `MXFP4`".to_string());
        }
    }
    Ok(tp)
}

/// Expand an ISQ specifier into concrete `IsqType` variants.
/// Numeric shorthands (2-8) produce both the non-Metal and Metal variants;
/// explicit method names resolve to a single variant.
pub fn expand_isq_value(s: &str) -> anyhow::Result<Vec<IsqType>> {
    if let Ok(bits) = IsqBits::try_from(s.to_lowercase().as_str()) {
        return Ok(bits.expand());
    }
    let isq = parse_isq_value(s, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(vec![isq])
}

/// Given a UQFF filename like `"q4k-0.uqff"`, returns `Some(("q4k", 0))`.
/// Returns `None` for non-sharded filenames like `"model.uqff"` where the
/// suffix after the last `-` is not a number.
pub fn parse_uqff_shard(filename: &str) -> Option<(String, u64)> {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())?;
    let (prefix, suffix) = stem.rsplit_once('-')?;
    let index = suffix.parse::<u64>().ok()?;
    Some((prefix.to_string(), index))
}

/// Expand a single UQFF filename to include all sibling shards.
///
/// Given `"q4k-0.uqff"` and a list of available files, returns
/// `["q4k-0.uqff", "q4k-1.uqff", ...]` for all sequential indices found.
/// Non-sharded filenames (those not matching `{prefix}-{N}.uqff`) are returned as-is.
pub fn expand_uqff_shards(first_file: &str, available_files: &[String]) -> Vec<String> {
    let Some((prefix, _)) = parse_uqff_shard(first_file) else {
        return vec![first_file.to_string()];
    };
    let mut shards = Vec::new();
    for index in 0u64.. {
        let candidate = format!("{prefix}-{index}.uqff");
        if available_files.iter().any(|f| f == &candidate) {
            shards.push(candidate);
        } else {
            break;
        }
    }
    if shards.is_empty() {
        vec![first_file.to_string()]
    } else {
        shards
    }
}

/// Resolve a UQFF shorthand (numeric like `"8"` or ISQ name like `"q4k"`) to an
/// actual UQFF filename from the available files list.
///
/// Returns `Some("q8_0-0.uqff")` if a matching file is found, `None` otherwise.
/// For numeric shorthands, tries all platform variants via `IsqBits::expand()`.
pub fn resolve_uqff_shorthand(input: &str, available_files: &[String]) -> Option<String> {
    let lowered = input.to_lowercase();

    // Try numeric shorthand first (2/3/4/5/6/8)
    if let Ok(bits) = IsqBits::try_from(lowered.as_str()) {
        for isq_type in bits.expand() {
            let candidate = format!("{isq_type}-0.uqff");
            if available_files.iter().any(|f| f == &candidate) {
                return Some(candidate);
            }
        }
        return None;
    }

    // Try explicit ISQ type name (e.g., "q4k", "afq8", "q8_0")
    if let Ok(isq_type) = parse_isq_value(&lowered, None) {
        let candidate = format!("{isq_type}-0.uqff");
        if available_files.iter().any(|f| f == &candidate) {
            return Some(candidate);
        }
    }

    None
}

#[derive(Clone, Debug, Copy, Default, Deserialize, serde::Serialize)]
pub enum IsqOrganization {
    #[default]
    #[serde(rename = "default")]
    Default,
    /// Only quantize MoE experts, if applicable. The enables MoQE.
    /// <https://arxiv.org/abs/2310.02410>
    #[serde(rename = "moqe")]
    MoeExpertsOnly,
}

impl FromStr for IsqOrganization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Self::Default),
            "moqe" => Ok(Self::MoeExpertsOnly),
            other => Err(format!(
                "Expected ISQ organization `default` or `moqe`, got `{other}`"
            )),
        }
    }
}

pub struct UqffFullSer<'a> {
    pub tokenizer: &'a Tokenizer,
    pub template_filename: &'a Option<PathBuf>,
    pub modules: Option<&'a String>,
    pub module_paths: Option<&'a [EmbeddingModulePaths]>,
    pub generation_config: Option<&'a PathBuf>,
    pub config: String,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

pub trait IsqModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        None
    }
}

/// Trait for loading models with ISQ.
pub(crate) trait IsqModelLoader {
    /// Regex to match layers which will have standard *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE *immediate* ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    /// Regex to match layers which will have standard ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }

    /// Regex to match layers which will have standard MoQE ISQ applied.
    ///
    /// Only called on non-adapter models!
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q8() {
        let files = vec!["q8_0-0.uqff".to_string(), "config.json".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some("q8_0-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_afq8() {
        let files = vec!["afq8-0.uqff".to_string(), "config.json".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some("afq8-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_prefers_platform_variant() {
        // expand() returns platform-preferred variant first:
        // Metal: [AFQ8, Q8_0], non-Metal: [Q8_0, AFQ8]
        let files = vec!["q8_0-0.uqff".to_string(), "afq8-0.uqff".to_string()];
        let expected = if cfg!(feature = "metal") {
            "afq8-0.uqff"
        } else {
            "q8_0-0.uqff"
        };
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some(expected.to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q4() {
        let files = vec!["q4k-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("4", &files),
            Some("q4k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q5() {
        let files = vec!["q5k-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("5", &files),
            Some("q5k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_isq_name() {
        let files = vec!["q4k-0.uqff".to_string(), "q8_0-0.uqff".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("q4k", &files),
            Some("q4k-0.uqff".to_string())
        );
    }

    #[test]
    fn test_resolve_uqff_shorthand_explicit_filename_returns_none() {
        let files = vec!["q8_0-0.uqff".to_string()];
        assert_eq!(resolve_uqff_shorthand("q8_0-0.uqff", &files), None);
    }

    #[test]
    fn test_resolve_uqff_shorthand_no_match() {
        let files = vec!["config.json".to_string()];
        assert_eq!(resolve_uqff_shorthand("8", &files), None);
    }
}
