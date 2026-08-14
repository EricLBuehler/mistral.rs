use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use mistralrs_quant::{IsqBits, IsqType, TrackedModule, UqffOutputReport, UqffReport, UqffTensor};
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

use crate::pipeline::{ChatTemplate, EmbeddingModulePaths};
use crate::utils::progress::configure_progress_bar;

pub(crate) const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";
pub const UQFF_MULTI_FILE_DELIMITER: &str = ";";
const UQFF_METADATA_PRODUCER: &str = "uqff.producer";
const UQFF_METADATA_VERSION: &str = "uqff.version";
const UQFF_METADATA_MISTRALRS_VERSION: &str = "uqff.producer.mistralrs.version";
const UQFF_METADATA_MISTRALRS_GIT_REVISION: &str = "uqff.producer.mistralrs.git_revision";
const UQFF_STAGING_PREFIX: &str = ".mistralrs-uqff-";
const UQFF_STAGING_PAYLOAD: &str = "payload";
const UQFF_STAGING_BACKUP: &str = "backup";
const UQFF_MANAGED_METADATA_FILES: &[&str] = &[
    UQFF_RESIDUAL_SAFETENSORS,
    "config.json",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    mistralrs_quant::UQFF_REPORT_JSON,
];

struct UqffArtifactStaging {
    tempdir: tempfile::TempDir,
    payload: PathBuf,
    final_parent: PathBuf,
    shard_stems: Vec<String>,
    dedicated_output: bool,
}

impl UqffArtifactStaging {
    fn new(
        final_parent: PathBuf,
        staging_parent: &Path,
        shard_stems: Vec<String>,
        dedicated_output: bool,
    ) -> Result<Self> {
        if final_parent.exists() && !final_parent.is_dir() {
            anyhow::bail!(
                "UQFF output directory `{}` is not a directory.",
                final_parent.display()
            );
        }
        std::fs::create_dir_all(staging_parent)?;
        let tempdir = tempfile::Builder::new()
            .prefix(UQFF_STAGING_PREFIX)
            .tempdir_in(staging_parent)?;
        let payload = tempdir.path().join(UQFF_STAGING_PAYLOAD);
        std::fs::create_dir(&payload)?;
        Ok(Self {
            tempdir,
            payload,
            final_parent,
            shard_stems,
            dedicated_output,
        })
    }

    fn staged_path(&self, final_path: &Path) -> Result<PathBuf> {
        let file_name = final_path
            .file_name()
            .context("Target UQFF path must have a filename!")?;
        Ok(self.payload.join(file_name))
    }

    fn publish(self) -> Result<()> {
        std::fs::create_dir_all(&self.final_parent)?;
        let staged_entries = immediate_child_names(&self.payload)?;
        let mut managed = staged_entries.iter().cloned().collect::<BTreeSet<_>>();
        let (report_matches_output, report_shards) =
            existing_report_artifacts(&self.final_parent, &self.shard_stems);
        if self.dedicated_output || report_matches_output {
            managed.extend(UQFF_MANAGED_METADATA_FILES.iter().map(PathBuf::from));
            managed.extend(report_shards);
        }
        for entry in std::fs::read_dir(&self.final_parent)? {
            let name = PathBuf::from(entry?.file_name());
            if is_indexed_uqff_shard(&name, &self.shard_stems) {
                managed.insert(name);
            }
        }

        let backup = self.tempdir.path().join(UQFF_STAGING_BACKUP);
        std::fs::create_dir(&backup)?;
        let mut backed_up = Vec::new();
        for name in &managed {
            let destination = self.final_parent.join(name);
            if !path_exists(&destination) {
                continue;
            }
            let backup_path = backup.join(name);
            if let Err(error) = std::fs::rename(&destination, &backup_path) {
                let rollback = rollback_uqff_publish(&self.final_parent, &backup, &[], &backed_up);
                return Err(publish_error(
                    format!(
                        "Failed to stage existing UQFF artifact `{}`",
                        destination.display()
                    ),
                    error,
                    rollback,
                ));
            }
            backed_up.push(name.clone());
        }

        let mut installed = Vec::new();
        for name in staged_entries {
            let source = self.payload.join(&name);
            let destination = self.final_parent.join(&name);
            if let Err(error) = std::fs::rename(&source, &destination) {
                let rollback =
                    rollback_uqff_publish(&self.final_parent, &backup, &installed, &backed_up);
                return Err(publish_error(
                    format!(
                        "Failed to publish UQFF artifact `{}`",
                        destination.display()
                    ),
                    error,
                    rollback,
                ));
            }
            installed.push(name);
        }
        Ok(())
    }
}

fn normalized_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn immediate_child_names(parent: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = std::fs::read_dir(parent)?
        .map(|entry| entry.map(|entry| PathBuf::from(entry.file_name())))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    Ok(entries)
}

fn existing_report_artifacts(parent: &Path, stems: &[String]) -> (bool, Vec<PathBuf>) {
    let report_path = parent.join(mistralrs_quant::UQFF_REPORT_JSON);
    let Ok(contents) = std::fs::read(report_path) else {
        return (false, Vec::new());
    };
    let Ok(report) = serde_json::from_slice::<mistralrs_quant::UqffReport>(&contents) else {
        return (false, Vec::new());
    };
    let shards = report
        .outputs
        .into_iter()
        .flat_map(|output| output.shards)
        .map(PathBuf::from)
        .filter(|path| path.components().count() == 1 && path.file_name().is_some())
        .collect::<Vec<_>>();
    let matches_output = shards.iter().any(|path| is_indexed_uqff_shard(path, stems));
    (matches_output, shards)
}

fn is_indexed_uqff_shard(path: &Path, stems: &[String]) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(base) = name.strip_suffix(".uqff") else {
        return false;
    };
    stems.iter().any(|stem| {
        base.strip_prefix(stem)
            .and_then(|suffix| suffix.strip_prefix('-'))
            .is_some_and(|index| {
                !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit())
            })
    })
}

fn path_exists(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

fn remove_path(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if metadata.file_type().is_dir() {
        std::fs::remove_dir_all(path)?;
    } else {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn rollback_uqff_publish(
    final_parent: &Path,
    backup: &Path,
    installed: &[PathBuf],
    backed_up: &[PathBuf],
) -> Result<()> {
    for name in installed.iter().rev() {
        remove_path(&final_parent.join(name))?;
    }
    for name in backed_up.iter().rev() {
        let destination = final_parent.join(name);
        remove_path(&destination)?;
        std::fs::rename(backup.join(name), destination)?;
    }
    Ok(())
}

fn publish_error(context: String, error: std::io::Error, rollback: Result<()>) -> anyhow::Error {
    match rollback {
        Ok(()) => anyhow::anyhow!("{context}: {error}"),
        Err(rollback_error) => {
            anyhow::anyhow!("{context}: {error}; rollback also failed: {rollback_error}")
        }
    }
}

fn is_tracked_module_tensor(name: &str, tracked_keys: &HashSet<&str>) -> bool {
    name.strip_suffix(".weight")
        .or_else(|| name.strip_suffix(".bias"))
        .is_some_and(|module| tracked_keys.contains(module))
}

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
            Self::Uqff => Cow::Borrowed("Loading UQFF model weights."),
            Self::ImmediateIsq => Cow::Owned(format!("Loading {target} weights for quantization.")),
            Self::PostLoadIsq => Cow::Owned(format!("Loading {target} weights for quantization.")),
            Self::UqffSerialization => {
                Cow::Owned(format!("Loading {target} weights for UQFF output."))
            }
            Self::Plain => Cow::Owned(format!("Loading {target} weights.")),
        }
    }
}

pub(crate) fn format_isq_types(types: &[IsqType]) -> String {
    types
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
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
    if tp == IsqType::F8Q8 && device.is_some_and(|device| !device.is_cpu()) {
        return Err("F8Q8 is CPU-only; choose `fp8` or another accelerator ISQ type.".to_string());
    }
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

pub fn resolve_uqff_report_output<'a>(
    input: &str,
    available_files: &[String],
    report: &'a UqffReport,
) -> Result<Option<&'a UqffOutputReport>> {
    validate_uqff_report_identity(report, available_files)?;

    if let Some(output) = report
        .outputs
        .iter()
        .find(|output| output.shards.iter().any(|shard| shard == input))
    {
        return Ok(Some(output));
    }
    if let Some(output) = report
        .outputs
        .iter()
        .find(|output| output.quant.eq_ignore_ascii_case(input))
    {
        return Ok(Some(output));
    }

    let aliases = report
        .outputs
        .iter()
        .map(|output| format!("{}-0.uqff", output.quant.to_ascii_lowercase()))
        .collect::<Vec<_>>();
    let Some(alias) = resolve_uqff_shorthand(input, &aliases) else {
        return Ok(None);
    };
    Ok(report
        .outputs
        .iter()
        .find(|output| format!("{}-0.uqff", output.quant.to_ascii_lowercase()) == alias))
}

fn validate_uqff_report_identity(report: &UqffReport, available_files: &[String]) -> Result<()> {
    if report.outputs.is_empty() {
        anyhow::bail!(
            "{}: report has no outputs",
            mistralrs_quant::UQFF_REPORT_JSON
        );
    }

    let mut quants = HashSet::new();
    let mut claimed = HashMap::new();
    for output in &report.outputs {
        if output.quant.trim().is_empty() {
            anyhow::bail!(
                "{}: output quant must not be empty",
                mistralrs_quant::UQFF_REPORT_JSON
            );
        }
        if !quants.insert(output.quant.to_ascii_lowercase()) {
            anyhow::bail!(
                "{}: duplicate output quant `{}`",
                mistralrs_quant::UQFF_REPORT_JSON,
                output.quant
            );
        }
        if output.shards.is_empty() {
            anyhow::bail!(
                "{}: output `{}` has no shards",
                mistralrs_quant::UQFF_REPORT_JSON,
                output.quant
            );
        }
        for shard in &output.shards {
            if !available_files.iter().any(|file| file == shard) {
                anyhow::bail!(
                    "{}: output `{}` references missing shard `{shard}`",
                    mistralrs_quant::UQFF_REPORT_JSON,
                    output.quant
                );
            }
            if let Some(previous_quant) = claimed.insert(shard, output.quant.as_str()) {
                anyhow::bail!(
                    "{}: shard `{shard}` belongs to both `{previous_quant}` and `{}`",
                    mistralrs_quant::UQFF_REPORT_JSON,
                    output.quant
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn resolve_uqff_input_files(
    input: &str,
    available_files: &[String],
    report: Option<&UqffReport>,
) -> Result<Vec<String>> {
    if let Some(report) = report {
        if let Some(output) = resolve_uqff_report_output(input, available_files, report)? {
            return Ok(output.shards.clone());
        }
    }

    if available_files.iter().any(|file| file == input) {
        return Ok(expand_uqff_shards(input, available_files));
    }

    let first = resolve_uqff_shorthand(input, available_files).unwrap_or_else(|| input.to_string());
    Ok(expand_uqff_shards(&first, available_files))
}

pub(crate) fn read_uqff_report_file(path: &Path) -> Result<UqffReport> {
    let contents = std::fs::read(path)
        .with_context(|| format!("Failed to read UQFF report `{}`", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("Failed to parse UQFF report `{}`", path.display()))
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
    pub effective_chat_template: Option<&'a ChatTemplate>,
    pub modules: Option<&'a String>,
    pub module_paths: Option<&'a [EmbeddingModulePaths]>,
    pub generation_config: Option<&'a PathBuf>,
    pub config: String,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct UqffWriteConfig {
    pub output: PathBuf,
    #[serde(default)]
    pub types: Vec<IsqType>,
    #[serde(default)]
    pub base_model: Option<String>,
    #[serde(default)]
    pub repo_id: Option<String>,
}

impl<'de> Deserialize<'de> for UqffWriteConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Path(PathBuf),
            Config {
                output: PathBuf,
                #[serde(default)]
                types: Vec<IsqType>,
                #[serde(default)]
                base_model: Option<String>,
                #[serde(default)]
                repo_id: Option<String>,
            },
        }

        match Repr::deserialize(deserializer)? {
            Repr::Path(output) => Ok(Self::from_output(output)),
            Repr::Config {
                output,
                types,
                base_model,
                repo_id,
            } => Ok(Self::with_types(output, types).with_report_metadata(base_model, repo_id)),
        }
    }
}

impl FromStr for UqffWriteConfig {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::from_output(PathBuf::from(s)))
    }
}

impl From<PathBuf> for UqffWriteConfig {
    fn from(output: PathBuf) -> Self {
        Self::from_output(output)
    }
}

impl UqffWriteConfig {
    pub fn from_output(output: PathBuf) -> Self {
        Self {
            output,
            types: Vec::new(),
            base_model: None,
            repo_id: None,
        }
    }

    pub fn with_types(output: PathBuf, types: Vec<IsqType>) -> Self {
        Self {
            output,
            types,
            base_model: None,
            repo_id: None,
        }
    }

    pub fn with_report_metadata(
        mut self,
        base_model: Option<String>,
        repo_id: Option<String>,
    ) -> Self {
        self.base_model = base_model;
        self.repo_id = repo_id;
        self
    }

    /// Build from an ISQ specifier; numeric shorthands expand to all variants, platform-preferred first.
    pub fn expand_from_str(output: PathBuf, spec: &str) -> anyhow::Result<Self> {
        Ok(Self::with_types(output, expand_isq_value(spec)?))
    }
}

pub(crate) struct UqffWriteRequest<'a> {
    pub output: PathBuf,
    pub types: Vec<IsqType>,
    pub base_model: Option<String>,
    pub repo_id: Option<String>,
    pub layers: Vec<TrackedModule>,
    pub quantize_predicates: Option<Vec<Regex>>,
    pub residual: Vec<(String, Tensor)>,
    pub full_ser: UqffFullSer<'a>,
    pub imatrix: std::collections::HashMap<String, Vec<f32>>,
}

const MAX_UQFF_SIZE_BYTES: usize = 10 * 1024 * 1024 * 1024;

struct UqffTypeWriteContext<'a> {
    ty: IsqType,
    serialized: &'a Path,
    display_path: &'a Path,
    layers: &'a [TrackedModule],
    swap_runtime: bool,
    imatrix: &'a HashMap<String, Vec<f32>>,
    quantize_predicates: Option<&'a [Regex]>,
    type_index: usize,
    type_count: usize,
}

struct UqffShardPaths<'a> {
    parent: &'a Path,
    display_parent: &'a Path,
    file_stem: &'a str,
}

pub(crate) fn write_uqff_artifacts(request: UqffWriteRequest<'_>) -> Result<()> {
    let UqffWriteRequest {
        output,
        types,
        base_model,
        repo_id,
        mut layers,
        quantize_predicates,
        mut residual,
        full_ser,
        imatrix,
    } = request;

    if types.is_empty() {
        anyhow::bail!("UQFF serialization requires at least one ISQ type.");
    }
    let mut seen_types = HashSet::new();
    for ty in &types {
        if !seen_types.insert(*ty) {
            anyhow::bail!("Duplicate UQFF output type `{ty}` was requested.");
        }
        if !ty.supports_uqff() {
            anyhow::bail!("UQFF serialization does not support {ty}.");
        }
    }
    layers.sort_by(|a, b| a.key.cmp(&b.key));
    let tracked_keys = layers
        .iter()
        .map(|module| module.key.as_str())
        .collect::<HashSet<_>>();
    residual.retain(|(name, _)| !is_tracked_module_tensor(name, &tracked_keys));

    let final_output_paths = if types.len() == 1 {
        if output.extension().is_none_or(|ext| ext != "uqff") {
            anyhow::bail!("UQFF output path extension must be `.uqff`");
        }
        vec![(types[0], output.clone())]
    } else {
        if output.extension().is_some_and(|ext| ext == "uqff") {
            anyhow::bail!(
                "Multiple UQFF output types require a directory path, not a `.uqff` file."
            );
        }
        types
            .iter()
            .map(|ty| (*ty, output.join(format!("{ty}.uqff"))))
            .collect::<Vec<_>>()
    };

    let (final_parent, staging_parent) = if final_output_paths.len() == 1 {
        let parent = normalized_parent(&final_output_paths[0].1).to_path_buf();
        (parent.clone(), parent)
    } else {
        (output.clone(), normalized_parent(&output).to_path_buf())
    };
    let shard_stems = final_output_paths
        .iter()
        .map(|(_, path)| {
            path.file_stem()
                .context("Target UQFF path must have a file stem!")
                .map(|stem| stem.to_string_lossy().to_string())
        })
        .collect::<Result<Vec<_>>>()?;
    let staging = UqffArtifactStaging::new(
        final_parent.clone(),
        &staging_parent,
        shard_stems,
        types.len() > 1,
    )?;
    let mut output_paths = final_output_paths
        .into_iter()
        .map(|(ty, final_path)| {
            staging
                .staged_path(&final_path)
                .map(|staged_path| (ty, staged_path, final_path))
        })
        .collect::<Result<Vec<_>>>()?;

    // Serialize the runtime type last so earlier passes still see the original sources.
    let runtime_ty = types[0];
    output_paths.sort_by_key(|(ty, _, _)| *ty == runtime_ty);

    let total_types = output_paths.len();
    let mut report_outputs = Vec::with_capacity(total_types);
    for (idx, (ty, path, display_path)) in output_paths.into_iter().enumerate() {
        let output_report = write_uqff_type(UqffTypeWriteContext {
            ty,
            serialized: &path,
            display_path: &display_path,
            layers: &layers,
            swap_runtime: ty == runtime_ty,
            imatrix: &imatrix,
            quantize_predicates: quantize_predicates.as_deref(),
            type_index: idx + 1,
            type_count: total_types,
        })?;
        report_outputs.push(output_report);
    }
    write_uqff_metadata(&staging.payload, &final_parent, residual, full_ser)?;
    let report = mistralrs_quant::UqffReport {
        schema: 1,
        generated_by: mistralrs_quant::UqffGeneratedBy {
            tool: "mistralrs quantize".to_string(),
            mistralrs_version: Some(crate::MISTRALRS_VERSION.to_string()),
            git_revision: Some(crate::MISTRALRS_GIT_REVISION.to_string()),
        },
        base_model,
        repo_id,
        uqff_version: format!(
            "{}.{}.{}",
            mistralrs_quant::UQFF_VERSION_MAJOR,
            mistralrs_quant::UQFF_VERSION_MINOR,
            mistralrs_quant::UQFF_VERSION_PATCH
        ),
        outputs: report_outputs,
    };
    mistralrs_quant::write_uqff_report(&staging.payload, &report)?;
    staging.publish()?;
    info!("In-memory model is quantized with {runtime_ty} as its default type.");
    info!(
        "Wrote UQFF report to `{}`.",
        final_parent
            .join(mistralrs_quant::UQFF_REPORT_JSON)
            .display()
    );
    Ok(())
}

fn write_uqff_type(ctx: UqffTypeWriteContext<'_>) -> Result<mistralrs_quant::UqffOutputReport> {
    let UqffTypeWriteContext {
        ty,
        serialized,
        display_path,
        layers,
        swap_runtime,
        imatrix,
        quantize_predicates,
        type_index,
        type_count,
    } = ctx;
    tracing::info!(
        "Serializing {type_index}/{type_count}: {} {ty} UQFF layers to `{}`.",
        layers.len(),
        display_path.display()
    );

    let parent = serialized
        .parent()
        .context("Target UQFF path must have a filename!")?;
    let display_parent = display_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;

    let file_stem = serialized
        .file_stem()
        .context("Target UQFF path must have a file stem!")?
        .to_string_lossy()
        .to_string();
    let shard_paths = UqffShardPaths {
        parent,
        display_parent,
        file_stem: &file_stem,
    };

    let bar = ProgressBar::new(layers.len() as u64);
    configure_progress_bar(&bar);
    bar.set_prefix(format!("{type_index}/{type_count} {ty}"));
    bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{prefix} [{elapsed_precise}] [{bar:40.red/magenta}] {pos}/{len} ({eta}) {msg}",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    bar.set_message("starting");
    bar.tick();

    let mut seen = HashSet::new();
    let mut current_chunk = Vec::new();
    let mut current_bytes = 0usize;
    let mut shard_index = 0u64;
    let mut shards = Vec::new();
    let quant_report = mistralrs_quant::QuantizationReport::default();
    let mut layer_reports = Vec::with_capacity(layers.len());

    for version in mistralrs_quant::uqff_version_tensors() {
        seen.insert(version.name().to_string());
        current_bytes += version.nbytes();
        current_chunk.push(version);
    }

    // Quantization runs on the pool; the writer consumes results in key order so the shard
    // layout stays deterministic while quantize-N+1 overlaps with write-N.
    // Topology-pinned layers keep their type; `ty` is the default for the rest.
    let quantized_layers = layers
        .iter()
        .filter(|module| uqff_should_quantize(module, quantize_predicates))
        .cloned()
        .collect::<Vec<_>>();
    let handles = mistralrs_quant::requantize_tracked(
        &quantized_layers,
        ty,
        |m| m.resolve_type(ty),
        &|key| imatrix.get(key).cloned(),
        mistralrs_quant::IsqConsumer::UqffWrite,
        MAX_UQFF_SIZE_BYTES,
        Some(quant_report.clone()),
    )?;
    let mut receivers = handles.receivers.into_iter();
    let guard = mistralrs_quant::QuantizeOntoGuard::new();
    for module in layers {
        bar.set_message(module.key.clone());
        bar.tick();
        let quantize = uqff_should_quantize(module, quantize_predicates);
        let resolved_ty = quantize.then(|| module.resolve_type(ty));
        let layer = if quantize {
            receivers
                .next()
                .expect("requantize receiver count must match selected UQFF layers")
                .recv()
                .map_err(|e| anyhow::anyhow!("Requantize channel error: {e}"))??
                .value
        } else {
            module.ct.resolve()?
        };
        let serialize_ty = resolved_ty.or_else(|| layer.uqff_type()).unwrap_or(ty);
        let serialized_tensors = layer.serialize_uqff(&module.key, serialize_ty)?;
        let (stored, shape) =
            mistralrs_quant::stored_type_from_tensors(&serialized_tensors, &module.key)?;
        layer_reports.push(mistralrs_quant::UqffLayerReport {
            module: module.key.clone(),
            default_target: quantize.then(|| ty.to_string()),
            resolved_target: resolved_ty.map(|resolved_ty| resolved_ty.to_string()),
            stored,
            shape,
        });
        for tensor in serialized_tensors {
            let name = tensor.name().to_string();
            if !seen.insert(name.clone()) {
                anyhow::bail!("Duplicate UQFF tensor key `{name}`.");
            }
            let tensor_bytes = tensor.nbytes();
            if !current_chunk.is_empty() && current_bytes + tensor_bytes > MAX_UQFF_SIZE_BYTES {
                flush_uqff_shard(
                    &bar,
                    &shard_paths,
                    &mut shard_index,
                    &mut shards,
                    &mut current_chunk,
                    &mut current_bytes,
                )?;
            }
            current_bytes += tensor_bytes;
            current_chunk.push(tensor);
            if current_bytes >= MAX_UQFF_SIZE_BYTES {
                flush_uqff_shard(
                    &bar,
                    &shard_paths,
                    &mut shard_index,
                    &mut shards,
                    &mut current_chunk,
                    &mut current_bytes,
                )?;
            }
        }
        if swap_runtime && quantize {
            let target = module.ct.resolve()?.dtype_and_device().1;
            let layer = if layer.dtype_and_device().1.same_device(&target) {
                layer
            } else {
                layer.apply_isq(
                    None,
                    target,
                    &std::sync::atomic::AtomicUsize::new(0),
                    None,
                    guard.clone().with_module_key(module.key.clone()),
                )?
            };
            module.ct.replace(layer);
        }
        bar.inc(1);
    }
    debug_assert!(receivers.next().is_none());

    flush_uqff_shard(
        &bar,
        &shard_paths,
        &mut shard_index,
        &mut shards,
        &mut current_chunk,
        &mut current_bytes,
    )?;
    bar.finish_and_clear();
    let output_report = mistralrs_quant::build_output_report_from_layers(
        ty.to_string(),
        shards,
        layer_reports,
        &quant_report,
    );
    if output_report.fallback_count == 0 {
        info!("UQFF {ty}: {} layers, no fallbacks.", output_report.layers);
    } else {
        info!(
            "UQFF {ty}: {} layers, {} fallback layer{}.",
            output_report.layers,
            output_report.fallback_count,
            if output_report.fallback_count == 1 {
                ""
            } else {
                "s"
            }
        );
    }
    info!(
        "Finished serializing {type_index}/{type_count}: {ty} UQFF to `{}` ({} shard{}).",
        display_path.display(),
        shard_index,
        if shard_index == 1 { "" } else { "s" }
    );
    Ok(output_report)
}

fn uqff_should_quantize(module: &TrackedModule, predicates: Option<&[Regex]>) -> bool {
    if module.ty.is_some() {
        return true;
    }
    predicates.is_none_or(|predicates| {
        let weight_key = format!("{}.weight", module.key);
        predicates
            .iter()
            .any(|predicate| predicate.is_match(&weight_key))
    })
}

fn flush_uqff_shard(
    bar: &ProgressBar,
    paths: &UqffShardPaths<'_>,
    shard_index: &mut u64,
    shards: &mut Vec<String>,
    current_chunk: &mut Vec<UqffTensor>,
    current_bytes: &mut usize,
) -> Result<()> {
    if current_chunk.is_empty() {
        return Ok(());
    }

    let shard_name = format!("{}-{shard_index}.uqff", paths.file_stem);
    let shard_path = paths.parent.join(&shard_name);
    let display_path = paths.display_parent.join(&shard_name);
    bar.suspend(|| {
        info!(
            "Writing shard {} to `{}`",
            shard_index,
            display_path.display()
        );
    });
    safetensors::serialize_to_file(
        current_chunk.iter().map(|tensor| (tensor.name(), tensor)),
        Some(uqff_safetensors_metadata()),
        &shard_path,
    )?;
    if let Some(name) = shard_path.file_name().and_then(|name| name.to_str()) {
        shards.push(name.to_string());
    }
    *shard_index += 1;
    current_chunk.clear();
    *current_bytes = 0;
    Ok(())
}

fn write_uqff_metadata(
    metadata_parent: &Path,
    display_parent: &Path,
    residual: Vec<(String, Tensor)>,
    full_ser: UqffFullSer<'_>,
) -> Result<()> {
    let residual_out = metadata_parent.join(UQFF_RESIDUAL_SAFETENSORS);
    let config_out = metadata_parent.join("config.json");
    let modules_out = metadata_parent.join("modules.json");
    let tokenizer_out = metadata_parent.join("tokenizer.json");
    let gen_cfg_out = metadata_parent.join("generation_config.json");
    let processor_out = metadata_parent.join("processor_config.json");
    let preprocessor_out = metadata_parent.join("preprocessor_config.json");

    info!(
        "Serializing {} residual tensors to `{}`.",
        residual.len(),
        display_parent.join(UQFF_RESIDUAL_SAFETENSORS).display()
    );
    safetensors::serialize_to_file(residual, Some(uqff_safetensors_metadata()), &residual_out)?;

    let UqffFullSer {
        tokenizer,
        template_filename,
        effective_chat_template,
        modules,
        module_paths,
        generation_config,
        config,
        processor_filename,
        preprocessor_filename,
    } = full_ser;

    info!(
        "Serializing configuration to `{}`.",
        display_parent.join("config.json").display()
    );
    std::fs::write(
        config_out,
        sanitize_quantized_weight_source_config(&config)?,
    )?;

    info!(
        "Serializing tokenizer to `{}`.",
        display_parent.join("tokenizer.json").display()
    );
    serde_json::to_writer_pretty(File::create(&tokenizer_out)?, tokenizer)
        .map_err(candle_core::Error::msg)?;

    write_uqff_chat_metadata(
        metadata_parent,
        display_parent,
        template_filename.as_deref(),
        effective_chat_template,
    )?;

    if let Some(generation_config) = generation_config {
        info!(
            "Serializing generation config to `{}`.",
            display_parent.join("generation_config.json").display()
        );
        let cfg = std::fs::read(generation_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&gen_cfg_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(processor_config) = processor_filename {
        info!(
            "Serializing processor config to `{}`.",
            display_parent.join("processor_config.json").display()
        );
        let cfg = std::fs::read(processor_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&processor_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(preprocessor_config) = preprocessor_filename {
        info!(
            "Serializing preprocessor config to `{}`.",
            display_parent.join("preprocessor_config.json").display()
        );
        let cfg = std::fs::read(preprocessor_config).map_err(candle_core::Error::msg)?;
        std::fs::write(&preprocessor_out, cfg).map_err(candle_core::Error::msg)?;
    }

    if let Some(modules) = modules {
        info!(
            "Serializing modules manifest to `{}`.",
            display_parent.join("modules.json").display()
        );
        std::fs::write(&modules_out, modules).map_err(candle_core::Error::msg)?;

        if let Some(module_paths) = module_paths {
            for module in module_paths {
                match module {
                    EmbeddingModulePaths::Transformer { path }
                    | EmbeddingModulePaths::Pooling { path, .. }
                    | EmbeddingModulePaths::Dense { path, .. }
                    | EmbeddingModulePaths::Normalize { path } => {
                        if path.is_empty() {
                            continue;
                        }
                        let module_dir = metadata_parent.join(path.as_str());
                        std::fs::create_dir_all(&module_dir).map_err(candle_core::Error::msg)?;

                        match module {
                            EmbeddingModulePaths::Pooling { config, .. } => {
                                let dest = module_dir.join("config.json");
                                if config != &dest {
                                    std::fs::copy(config, &dest)
                                        .map_err(candle_core::Error::msg)?;
                                }
                            }
                            EmbeddingModulePaths::Dense { config, model, .. } => {
                                let dest_cfg = module_dir.join("config.json");
                                if config != &dest_cfg {
                                    std::fs::copy(config, &dest_cfg)
                                        .map_err(candle_core::Error::msg)?;
                                }
                                let dest_model = module_dir.join("model.safetensors");
                                if model != &dest_model {
                                    std::fs::copy(model, &dest_model)
                                        .map_err(candle_core::Error::msg)?;
                                }
                            }
                            EmbeddingModulePaths::Transformer { .. }
                            | EmbeddingModulePaths::Normalize { .. } => {}
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn sanitize_quantized_weight_source_config(config: &str) -> Result<String> {
    fn remove_source_quantization(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                object.remove("quantization_config");
                for value in object.values_mut() {
                    remove_source_quantization(value);
                }
            }
            serde_json::Value::Array(array) => {
                for value in array {
                    remove_source_quantization(value);
                }
            }
            _ => {}
        }
    }

    let mut config: serde_json::Value = serde_json::from_str(config)
        .context("Failed to parse model config for quantized weight source")?;
    remove_source_quantization(&mut config);
    serde_json::to_string_pretty(&config)
        .context("Failed to serialize model config for quantized weight source")
}

fn write_uqff_chat_metadata(
    metadata_parent: &Path,
    display_parent: &Path,
    template_filename: Option<&Path>,
    effective_chat_template: Option<&ChatTemplate>,
) -> Result<()> {
    let tokenizer_cfg_out = metadata_parent.join("tokenizer_config.json");
    let chat_template_jinja_out = metadata_parent.join("chat_template.jinja");

    if let Some(chat_template) = effective_chat_template {
        info!(
            "Serializing tokenizer config to `{}`.",
            display_parent.join("tokenizer_config.json").display()
        );
        serde_json::to_writer_pretty(File::create(&tokenizer_cfg_out)?, chat_template)?;
    } else if let Some(template_filename) = template_filename {
        let template = std::fs::read(template_filename).map_err(candle_core::Error::msg)?;

        if template_filename.extension().map(|e| e.to_str()) == Some(Some("jinja")) {
            info!(
                "Serializing chat template to `{}`.",
                display_parent.join("chat_template.jinja").display()
            );
            std::fs::write(&chat_template_jinja_out, template).map_err(candle_core::Error::msg)?;

            let sibling_cfg = template_filename
                .parent()
                .map(|dir| dir.join("tokenizer_config.json"));
            if let Some(cfg_path) = sibling_cfg.filter(|p| p.exists()) {
                info!(
                    "Serializing tokenizer config to `{}`.",
                    display_parent.join("tokenizer_config.json").display()
                );
                std::fs::copy(&cfg_path, &tokenizer_cfg_out).map_err(candle_core::Error::msg)?;
            }
        } else {
            info!(
                "Serializing tokenizer config to `{}`.",
                display_parent.join("tokenizer_config.json").display()
            );
            std::fs::write(&tokenizer_cfg_out, template).map_err(candle_core::Error::msg)?;
        }
    }
    Ok(())
}

fn uqff_safetensors_metadata() -> HashMap<String, String> {
    HashMap::from([
        (UQFF_METADATA_PRODUCER.to_string(), "mistral.rs".to_string()),
        (
            UQFF_METADATA_VERSION.to_string(),
            format!(
                "{}.{}.{}",
                mistralrs_quant::UQFF_VERSION_MAJOR,
                mistralrs_quant::UQFF_VERSION_MINOR,
                mistralrs_quant::UQFF_VERSION_PATCH
            ),
        ),
        (
            UQFF_METADATA_MISTRALRS_VERSION.to_string(),
            crate::MISTRALRS_VERSION.to_string(),
        ),
        (
            UQFF_METADATA_MISTRALRS_GIT_REVISION.to_string(),
            crate::MISTRALRS_GIT_REVISION.to_string(),
        ),
    ])
}

pub trait IsqModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)>;

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        None
    }
}

/// Trait for loading models with ISQ.
pub(crate) trait IsqModelLoader {
    /// Exact checkpoint tensor paths whose default ISQ type should be promoted.
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>>;

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
        self.isq_layer_regexes_moqe(config)
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
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
}

/// Map a layer tracking key to candidate llama.cpp imatrix entries.
fn gguf_imatrix_names(key: &str) -> Vec<String> {
    if key == "lm_head" || key.ends_with(".lm_head") || key == "output" {
        return vec!["output.weight".to_string()];
    }
    let Some(index) = mistralrs_quant::layer_index_from_prefix(key) else {
        return Vec::new();
    };
    let per_expert = key
        .split_once(".experts.")
        .and_then(|(_, suffix)| suffix.split('.').next())
        .is_some_and(|segment| segment.parse::<usize>().is_ok());
    if per_expert {
        return Vec::new();
    }
    let is_expert = key.contains(".experts.") || key.contains(".block_sparse_moe.");
    let is_shared = key.contains(".shared_expert.")
        || key.contains(".shared_experts.")
        || key.contains(".shared_mlp.");
    let names = if key.ends_with(".self_attn.q_proj") {
        vec!["attn_q", "attn_qkv"]
    } else if key.ends_with(".self_attn.k_proj") {
        vec!["attn_k", "attn_qkv"]
    } else if key.ends_with(".self_attn.v_proj") {
        vec!["attn_v", "attn_qkv"]
    } else if key.ends_with(".self_attn.qkv_proj") {
        vec!["attn_qkv", "attn_q", "attn_k", "attn_v"]
    } else if key.ends_with(".self_attn.o_proj")
        || key.ends_with(".self_attn.out_proj")
        || key.ends_with(".self_attn.dense")
        || key.ends_with(".attention.wo")
    {
        vec!["attn_output"]
    } else if key.ends_with(".attention.wq") {
        vec!["attn_q"]
    } else if key.ends_with(".attention.wk") {
        vec!["attn_k"]
    } else if key.ends_with(".attention.wv") {
        vec!["attn_v"]
    } else if key.ends_with(".self_attn.q_a_proj") {
        vec!["attn_q_a"]
    } else if key.ends_with(".self_attn.q_b_proj") {
        vec!["attn_q_b"]
    } else if key.ends_with(".self_attn.kv_a_proj_with_mqa") {
        vec!["attn_kv_a_mqa"]
    } else if key.ends_with(".self_attn.kv_b_proj") {
        vec!["attn_kv_b"]
    } else if key.ends_with(".self_attn.k_b_proj") {
        vec!["attn_k_b", "attn_kv_b"]
    } else if key.ends_with(".self_attn.v_b_proj") {
        vec!["attn_v_b", "attn_kv_b"]
    } else if key.ends_with(".linear_attn.in_proj_qkvz") {
        vec!["ssm_in", "attn_qkv", "attn_gate"]
    } else if key.ends_with(".linear_attn.in_proj_qkv") {
        vec!["attn_qkv"]
    } else if key.ends_with(".linear_attn.in_proj_z") {
        vec!["attn_gate"]
    } else if key.ends_with(".linear_attn.in_proj_ba") {
        vec!["ssm_ba", "ssm_beta", "ssm_alpha"]
    } else if key.ends_with(".linear_attn.in_proj_b") {
        vec!["ssm_beta", "ssm_ba"]
    } else if key.ends_with(".linear_attn.in_proj_a") {
        vec!["ssm_alpha", "ssm_ba"]
    } else if key.ends_with(".linear_attn.out_proj") {
        vec!["ssm_out"]
    } else if key.ends_with(".mamba.in_proj") {
        vec!["ssm_in"]
    } else if key.ends_with(".mamba.out_proj") {
        vec!["ssm_out"]
    } else if key.ends_with(".conv.in_proj") {
        vec!["shortconv.in_proj"]
    } else if key.ends_with(".conv.out_proj") {
        vec!["shortconv.out_proj"]
    } else if key.ends_with(".gate_up_proj") {
        if is_shared {
            vec!["ffn_gate_shexp", "ffn_up_shexp"]
        } else if is_expert {
            vec!["ffn_gate_up_exps", "ffn_gate_exps", "ffn_up_exps"]
        } else {
            vec!["ffn_gate", "ffn_up"]
        }
    } else if key.ends_with(".c_fc") || key.ends_with(".fc1") {
        vec!["ffn_up"]
    } else if key.ends_with(".c_proj") || key.ends_with(".fc2") {
        vec!["ffn_down"]
    } else if key.ends_with(".w1") {
        vec!["ffn_gate"]
    } else if key.ends_with(".w3") {
        vec!["ffn_up"]
    } else if key.ends_with(".w2") {
        vec!["ffn_down"]
    } else if key.ends_with(".shared_mlp.input_linear") {
        vec!["ffn_gate_shexp", "ffn_up_shexp", "ffn_gate", "ffn_up"]
    } else if key.ends_with(".shared_mlp.output_linear") {
        vec!["ffn_down_shexp", "ffn_down"]
    } else if key.ends_with(".block_sparse_moe.input_linear") {
        vec!["ffn_gate_exps", "ffn_up_exps"]
    } else if key.ends_with(".block_sparse_moe.output_linear") {
        vec!["ffn_down_exps"]
    } else if key.ends_with(".self_attn.gate_proj") {
        vec!["attn_gate"]
    } else if key.ends_with(".gate_proj") {
        if is_shared {
            vec!["ffn_gate_shexp"]
        } else if is_expert {
            vec!["ffn_gate_exps"]
        } else {
            vec!["ffn_gate"]
        }
    } else if key.ends_with(".up_proj") {
        if is_shared {
            vec!["ffn_up_shexp"]
        } else if is_expert {
            vec!["ffn_up_exps"]
        } else {
            vec!["ffn_up"]
        }
    } else if key.ends_with(".down_proj") {
        if is_shared {
            vec!["ffn_down_shexp"]
        } else if is_expert {
            vec!["ffn_down_exps"]
        } else {
            vec!["ffn_down"]
        }
    } else if key.ends_with(".shared_expert_gate") {
        vec!["ffn_gate_inp_shexp"]
    } else if key.ends_with(".gate")
        || key.ends_with(".gate.wg")
        || key.ends_with(".router")
        || key.ends_with(".router.layer")
    {
        vec!["ffn_gate_inp"]
    } else {
        Vec::new()
    };
    names
        .into_iter()
        .map(|name| format!("blk.{index}.{name}.weight"))
        .collect()
}

/// Load per-layer imatrix weights for `modules` from a `.cimatrix` (tracking-key keyed) or
/// llama.cpp `.imatrix` file.
pub(crate) fn load_imatrix_map(
    path: &Path,
    modules: &[TrackedModule],
) -> Result<std::collections::HashMap<String, Vec<f32>>> {
    if path.extension().is_some_and(|ext| ext == "cimatrix") {
        info!("Loading collected imatrix file `{}`.", path.display());
        return Ok(mistralrs_quant::CollectedImatrixData::load_imatrix(path)?.0);
    }
    info!("Loading GGUF-format imatrix file `{}`.", path.display());
    let data = candle_core::quantized::imatrix_file::load_imatrix(path)?;
    let mut map = std::collections::HashMap::new();
    for module in modules {
        for name in gguf_imatrix_names(&module.key) {
            if let Some(values) = data.get(&name) {
                map.insert(module.key.clone(), values.clone());
                break;
            }
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{get_chat_template, AdapterPaths, LocalModelPaths};
    use candle_core::DType;
    use candle_nn::Linear;
    use mistralrs_quant::{
        pending_isq_channel, PendingIsqLayer, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
        UnquantLinear,
    };
    use std::sync::{atomic::AtomicUsize, Arc};

    #[test]
    fn uqff_config_removes_source_quantization_metadata_recursively() -> Result<()> {
        let source = serde_json::json!({
            "architectures": ["Qwen3NextForCausalLM"],
            "model_type": "test",
            "_mistralrs_gdn_v_head_layout": "tiled",
            "_mistralrs_qk_rope_layout": "adjacent",
            "quantization_config": {"quant_method": "gptq", "bits": 4},
            "text_config": {
                "hidden_size": 128,
                "quantization_config": {"quant_method": "fp8"}
            },
            "submodels": [
                {"quantization_config": {"quant_method": "awq"}, "layers": 2}
            ]
        });

        let sanitized = sanitize_quantized_weight_source_config(&serde_json::to_string(&source)?)?;
        let sanitized: serde_json::Value = serde_json::from_str(&sanitized)?;
        assert_eq!(sanitized["architectures"][0], "Qwen3NextForCausalLM");
        assert_eq!(sanitized["model_type"], "test");
        assert_eq!(sanitized["_mistralrs_gdn_v_head_layout"], "tiled");
        assert_eq!(sanitized["_mistralrs_qk_rope_layout"], "adjacent");
        assert_eq!(sanitized["text_config"]["hidden_size"], 128);
        assert_eq!(sanitized["submodels"][0]["layers"], 2);
        assert!(sanitized.get("quantization_config").is_none());
        assert!(sanitized["text_config"]
            .get("quantization_config")
            .is_none());
        assert!(sanitized["submodels"][0]
            .get("quantization_config")
            .is_none());
        Ok(())
    }

    #[test]
    fn effective_chat_template_round_trips_through_uqff_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let source_template = dir.path().join("source.jinja");
        std::fs::write(&source_template, "source template")?;
        let effective: ChatTemplate = serde_json::from_value(serde_json::json!({
            "chat_template": "{{ bos_token }}effective{{ eos_token }}",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>"
        }))?;

        write_uqff_chat_metadata(
            dir.path(),
            dir.path(),
            Some(source_template.as_path()),
            Some(&effective),
        )?;

        assert!(!dir.path().join("chat_template.jinja").exists());
        let paths = LocalModelPaths {
            tokenizer_filename: dir.path().join("tokenizer.json"),
            config_filename: dir.path().join("config.json"),
            template_filename: Some(dir.path().join("tokenizer_config.json")),
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::None,
            gen_conf: None,
            preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename: None,
        };
        let reloaded = get_chat_template(&paths, None, None, None, None);
        assert_eq!(
            reloaded.get_template_contents(),
            ["{{ bos_token }}effective{{ eos_token }}"]
        );
        assert_eq!(reloaded.bos_tok().as_deref(), Some("<bos>"));
        assert_eq!(reloaded.eos_tok().as_deref(), Some("<eos>"));
        assert_eq!(reloaded.unk_tok().as_deref(), Some("<unk>"));
        Ok(())
    }

    fn tracked_unquant(key: &str, stacked: bool) -> Result<TrackedModule> {
        let weight = if stacked {
            Tensor::zeros((2, 2, 32), DType::F32, &Device::Cpu)?
        } else {
            Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?
        };
        let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(weight, None),
        ))?) as Arc<dyn QuantMethod>;
        let (_tx, rx) = pending_isq_channel();
        let ct = Arc::new(PendingIsqLayer::new(rx));
        ct.replace(layer);
        Ok(TrackedModule {
            key: key.to_string(),
            ct,
            ty: None,
            promote_default: false,
            shard: None,
        })
    }

    fn tracked_afq4(key: &str) -> Result<TrackedModule> {
        let weight = Tensor::zeros((2, 64), DType::F32, &Device::Cpu)?;
        let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(weight, None),
        ))?) as Arc<dyn QuantMethod>;
        let layer = layer.apply_isq(
            Some(IsqType::AFQ4),
            Device::Cpu,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        )?;
        let (_tx, rx) = pending_isq_channel();
        let ct = Arc::new(PendingIsqLayer::new(rx));
        ct.replace(layer);
        Ok(TrackedModule {
            key: key.to_string(),
            ct,
            ty: None,
            promote_default: false,
            shard: None,
        })
    }

    fn write_test_uqff_report(parent: &Path, shards: &[&str]) -> Result<()> {
        let report = serde_json::json!({
            "schema": 1,
            "generated_by": { "tool": "test" },
            "uqff_version": "0.1.0",
            "outputs": [{
                "quant": "q4k",
                "shards": shards,
                "layers": 0,
                "actual_counts": {},
                "fallback_count": 0
            }]
        });
        std::fs::write(
            parent.join(mistralrs_quant::UQFF_REPORT_JSON),
            serde_json::to_vec(&report)?,
        )?;
        Ok(())
    }

    fn custom_named_uqff_report() -> UqffReport {
        serde_json::from_value(serde_json::json!({
            "schema": 1,
            "generated_by": { "tool": "test" },
            "uqff_version": "0.1.0",
            "outputs": [{
                "quant": "q8_0",
                "shards": ["release-part-a.uqff", "release-part-b.uqff"],
                "layers": 0,
                "actual_counts": {},
                "fallback_count": 0
            }]
        }))
        .unwrap()
    }

    #[test]
    fn uqff_staging_failure_preserves_existing_artifacts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let final_parent = dir.path().join("artifact");
        std::fs::create_dir(&final_parent)?;
        std::fs::write(final_parent.join("model-0.uqff"), b"old shard")?;
        std::fs::write(final_parent.join("config.json"), b"old config")?;

        {
            let staging = UqffArtifactStaging::new(
                final_parent.clone(),
                dir.path(),
                vec!["model".to_string()],
                false,
            )?;
            std::fs::write(staging.payload.join("model-0.uqff"), b"new shard")?;
            std::fs::write(staging.payload.join("config.json"), b"new config")?;
        }

        assert_eq!(
            std::fs::read(final_parent.join("model-0.uqff"))?,
            b"old shard"
        );
        assert_eq!(
            std::fs::read(final_parent.join("config.json"))?,
            b"old config"
        );
        assert!(std::fs::read_dir(dir.path())?.all(|entry| {
            entry.is_ok_and(|entry| {
                !entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(UQFF_STAGING_PREFIX)
            })
        }));
        Ok(())
    }

    #[test]
    fn uqff_publish_replaces_complete_shard_set_and_preserves_unrelated_files() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let final_parent = dir.path().join("artifact");
        std::fs::create_dir(&final_parent)?;
        std::fs::write(final_parent.join("model-0.uqff"), b"old shard")?;
        std::fs::write(final_parent.join("model-7.uqff"), b"stale shard")?;
        std::fs::write(final_parent.join("model-extra.uqff"), b"unrelated shard")?;
        std::fs::write(final_parent.join("config.json"), b"unrelated config")?;
        std::fs::write(final_parent.join("tokenizer.json"), b"unrelated tokenizer")?;
        write_test_uqff_report(&final_parent, &["other-0.uqff"])?;
        let unrelated_report = std::fs::read(final_parent.join(mistralrs_quant::UQFF_REPORT_JSON))?;
        std::fs::write(final_parent.join("notes.txt"), b"keep me")?;

        let staging = UqffArtifactStaging::new(
            final_parent.clone(),
            dir.path(),
            vec!["model".to_string()],
            false,
        )?;
        std::fs::write(staging.payload.join("model-0.uqff"), b"new shard")?;
        staging.publish()?;

        assert_eq!(
            std::fs::read(final_parent.join("model-0.uqff"))?,
            b"new shard"
        );
        assert!(!final_parent.join("model-7.uqff").exists());
        assert_eq!(
            std::fs::read(final_parent.join("model-extra.uqff"))?,
            b"unrelated shard"
        );
        assert_eq!(
            std::fs::read(final_parent.join("config.json"))?,
            b"unrelated config"
        );
        assert_eq!(
            std::fs::read(final_parent.join("tokenizer.json"))?,
            b"unrelated tokenizer"
        );
        assert_eq!(
            std::fs::read(final_parent.join(mistralrs_quant::UQFF_REPORT_JSON))?,
            unrelated_report
        );
        assert_eq!(std::fs::read(final_parent.join("notes.txt"))?, b"keep me");
        Ok(())
    }

    #[test]
    fn uqff_publish_cleans_metadata_owned_by_same_stem_report() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let final_parent = dir.path().join("artifact");
        std::fs::create_dir(&final_parent)?;
        std::fs::write(final_parent.join("model-0.uqff"), b"old shard")?;
        std::fs::write(final_parent.join("model-4.uqff"), b"stale shard")?;
        std::fs::write(final_parent.join("legacy-0.uqff"), b"reported shard")?;
        std::fs::write(
            final_parent.join("generation_config.json"),
            b"stale metadata",
        )?;
        std::fs::write(final_parent.join("notes.txt"), b"keep me")?;
        write_test_uqff_report(
            &final_parent,
            &["model-0.uqff", "model-4.uqff", "legacy-0.uqff"],
        )?;

        let staging = UqffArtifactStaging::new(
            final_parent.clone(),
            dir.path(),
            vec!["model".to_string()],
            false,
        )?;
        std::fs::write(staging.payload.join("model-0.uqff"), b"new shard")?;
        staging.publish()?;

        assert_eq!(
            std::fs::read(final_parent.join("model-0.uqff"))?,
            b"new shard"
        );
        assert!(!final_parent.join("model-4.uqff").exists());
        assert!(!final_parent.join("legacy-0.uqff").exists());
        assert!(!final_parent.join("generation_config.json").exists());
        assert!(!final_parent
            .join(mistralrs_quant::UQFF_REPORT_JSON)
            .exists());
        assert_eq!(std::fs::read(final_parent.join("notes.txt"))?, b"keep me");
        Ok(())
    }

    #[test]
    fn gguf_imatrix_names_cover_native_weight_bindings() {
        assert_eq!(
            gguf_imatrix_names("output"),
            vec!["output.weight".to_string()]
        );
        assert_eq!(
            gguf_imatrix_names("model.layers.2.linear_attn.in_proj_qkvz")[0],
            "blk.2.ssm_in.weight"
        );
        assert_eq!(
            gguf_imatrix_names("model.layers.2.mlp.experts.gate_up_proj")[0],
            "blk.2.ffn_gate_up_exps.weight"
        );
        for (key, expected) in [
            ("model.layers.2.self_attn.qkv_proj", "blk.2.attn_qkv.weight"),
            (
                "model.language_model.layers.2.self_attn.gate_proj",
                "blk.2.attn_gate.weight",
            ),
            ("layers.2.attention.wq", "blk.2.attn_q.weight"),
            ("layers.2.attention.wk", "blk.2.attn_k.weight"),
            ("layers.2.attention.wv", "blk.2.attn_v.weight"),
            ("layers.2.attention.wo", "blk.2.attn_output.weight"),
            ("model.layers.2.mlp.gate_up_proj", "blk.2.ffn_gate.weight"),
            ("model.layers.2.mlp.c_fc", "blk.2.ffn_up.weight"),
            ("model.layers.2.mlp.c_proj", "blk.2.ffn_down.weight"),
            (
                "model.layers.2.self_attn.kv_a_proj_with_mqa",
                "blk.2.attn_kv_a_mqa.weight",
            ),
            ("model.layers.2.self_attn.k_b_proj", "blk.2.attn_k_b.weight"),
            ("model.layers.2.self_attn.v_b_proj", "blk.2.attn_v_b.weight"),
            (
                "model.layers.2.linear_attn.in_proj_qkv",
                "blk.2.attn_qkv.weight",
            ),
            (
                "model.layers.2.linear_attn.in_proj_z",
                "blk.2.attn_gate.weight",
            ),
            (
                "model.layers.2.linear_attn.in_proj_b",
                "blk.2.ssm_beta.weight",
            ),
            (
                "model.layers.2.linear_attn.out_proj",
                "blk.2.ssm_out.weight",
            ),
            ("model.layers.2.mamba.in_proj", "blk.2.ssm_in.weight"),
            (
                "model.layers.2.shared_mlp.input_linear",
                "blk.2.ffn_gate_shexp.weight",
            ),
            (
                "model.layers.2.block_sparse_moe.input_linear",
                "blk.2.ffn_gate_exps.weight",
            ),
            (
                "model.layers.2.mlp.experts.gate_up_proj",
                "blk.2.ffn_gate_up_exps.weight",
            ),
            (
                "model.layers.2.mlp.shared_expert.gate_up_proj",
                "blk.2.ffn_gate_shexp.weight",
            ),
            (
                "language_model.model.layers.2.feed_forward.shared_expert.gate_proj",
                "blk.2.ffn_gate_shexp.weight",
            ),
            (
                "model.language_model.layers.2.mlp.shared_expert.up_proj",
                "blk.2.ffn_up_shexp.weight",
            ),
            (
                "model.layers.2.mlp.shared_expert_gate",
                "blk.2.ffn_gate_inp_shexp.weight",
            ),
        ] {
            assert!(
                gguf_imatrix_names(key).iter().any(|name| name == expected),
                "{key} should map to {expected}"
            );
        }
    }

    #[test]
    fn gguf_imatrix_omits_individually_materialized_experts() {
        assert!(gguf_imatrix_names("model.layers.2.mlp.experts.3.gate_proj").is_empty());
    }

    #[test]
    fn tracked_module_weights_are_not_written_as_residuals() {
        let tracked = HashSet::from(["model.embed_tokens", "model.layers.0.self_attn.q_proj"]);
        assert!(is_tracked_module_tensor(
            "model.embed_tokens.weight",
            &tracked
        ));
        assert!(is_tracked_module_tensor(
            "model.layers.0.self_attn.q_proj.bias",
            &tracked
        ));
        assert!(!is_tracked_module_tensor("model.norm.weight", &tracked));
        assert!(!is_tracked_module_tensor(
            "model.embed_tokens.weight_scale",
            &tracked
        ));
    }

    #[test]
    fn uqff_moqe_requantizes_only_routed_experts() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut topology_override = tracked_unquant("model.layers.0.self_attn.k_proj", false)?;
        topology_override.ty = Some(IsqType::Q4_0);
        let layers = vec![
            tracked_unquant("lm_head", false)?,
            tracked_unquant("model.layers.0.mlp.gate", false)?,
            tracked_unquant("model.layers.0.mlp.shared_expert.down_proj", false)?,
            tracked_afq4("model.layers.0.mlp.shared_expert.up_proj")?,
            tracked_unquant("model.layers.0.self_attn.q_proj", false)?,
            tracked_unquant("model.layers.0.mlp.experts", true)?,
            topology_override,
        ];
        let predicates = vec![Regex::new(r"^model\.layers\.\d+\.mlp\.experts(?:\.|$)")?];

        let q4_path = dir.path().join("moqe-q4.uqff");
        let imatrix = HashMap::new();
        let alternate = write_uqff_type(UqffTypeWriteContext {
            ty: IsqType::Q4_0,
            serialized: &q4_path,
            display_path: &q4_path,
            layers: &layers,
            swap_runtime: false,
            imatrix: &imatrix,
            quantize_predicates: Some(&predicates),
            type_index: 1,
            type_count: 2,
        })?;
        assert_eq!(alternate.actual_counts.get("q4_0"), Some(&2));
        assert_eq!(alternate.actual_counts.get("afq4"), Some(&1));
        assert_eq!(alternate.actual_counts.get("unquant"), Some(&4));
        assert_eq!(alternate.fallback_count, 0);
        assert_eq!(
            layers
                .iter()
                .find(|module| module.key == "model.layers.0.mlp.experts")
                .expect("routed experts")
                .ct
                .resolve()?
                .name(),
            "unquant-linear"
        );

        let q8_path = dir.path().join("moqe-q8.uqff");
        let report = write_uqff_type(UqffTypeWriteContext {
            ty: IsqType::Q8_0,
            serialized: &q8_path,
            display_path: &q8_path,
            layers: &layers,
            swap_runtime: true,
            imatrix: &imatrix,
            quantize_predicates: Some(&predicates),
            type_index: 2,
            type_count: 2,
        })?;

        assert_eq!(report.actual_counts.get("q8_0"), Some(&1));
        assert_eq!(report.actual_counts.get("q4_0"), Some(&1));
        assert_eq!(report.actual_counts.get("afq4"), Some(&1));
        assert_eq!(report.actual_counts.get("unquant"), Some(&4));
        assert_eq!(report.fallback_count, 0);
        for detail in &report.layer_details {
            if detail.module == "model.layers.0.mlp.experts" {
                assert_eq!(detail.stored, "q8_0");
                assert_eq!(detail.resolved_target.as_deref(), Some("q8_0"));
            } else if detail.module == "model.layers.0.self_attn.k_proj" {
                assert_eq!(detail.stored, "q4_0");
                assert_eq!(detail.resolved_target.as_deref(), Some("q4_0"));
            } else if detail.module == "model.layers.0.mlp.shared_expert.up_proj" {
                assert_eq!(detail.stored, "afq4");
                assert!(detail.default_target.is_none());
                assert!(detail.resolved_target.is_none());
            } else {
                assert_eq!(detail.stored, "unquant", "{}", detail.module);
                assert!(detail.default_target.is_none(), "{}", detail.module);
                assert!(detail.resolved_target.is_none(), "{}", detail.module);
            }
        }
        for module in &layers {
            let expected = if module.key == "model.layers.0.mlp.shared_expert.up_proj" {
                "afq-layer"
            } else if module.key == "model.layers.0.mlp.experts"
                || module.key == "model.layers.0.self_attn.k_proj"
            {
                "gguf"
            } else {
                "unquant-linear"
            };
            assert_eq!(module.ct.resolve()?.name(), expected, "{}", module.key);
        }
        Ok(())
    }

    #[test]
    fn uqff_rejects_unsupported_stacked_targets_before_writing() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let layers = vec![
            tracked_unquant("model.layers.0.mlp.experts.gate", false)?,
            tracked_unquant("model.layers.0.mlp.experts.up", true)?,
        ];
        let predicates = vec![Regex::new(r"^model\.layers\.\d+\.mlp\.experts(?:\.|$)")?];

        let path = dir.path().join("moqe.uqff");
        let imatrix = HashMap::new();
        let error = write_uqff_type(UqffTypeWriteContext {
            ty: IsqType::F8Q8,
            serialized: &path,
            display_path: &path,
            layers: &layers,
            swap_runtime: false,
            imatrix: &imatrix,
            quantize_predicates: Some(&predicates),
            type_index: 1,
            type_count: 1,
        })
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("does not support stacked expert gather"));
        assert!(std::fs::read_dir(dir.path())?.all(|entry| entry
            .is_ok_and(|entry| entry.path().extension().is_none_or(|ext| ext != "uqff"))));
        Ok(())
    }

    #[test]
    fn test_resolve_uqff_shorthand_numeric_q8() {
        let files = vec!["q8_0-0.uqff".to_string(), "config.json".to_string()];
        assert_eq!(
            resolve_uqff_shorthand("8", &files),
            Some("q8_0-0.uqff".to_string())
        );
    }

    #[test]
    fn report_quant_resolves_custom_named_shards() {
        let files = vec![
            "release-part-a.uqff".to_string(),
            "release-part-b.uqff".to_string(),
        ];
        let report = custom_named_uqff_report();

        for quant in ["q8_0", "8"] {
            let output = resolve_uqff_report_output(quant, &files, &report)
                .unwrap()
                .unwrap();
            assert_eq!(output.quant, "q8_0");
            assert_eq!(
                resolve_uqff_input_files(quant, &files, Some(&report)).unwrap(),
                files
            );
        }
    }

    #[test]
    fn exact_reported_shard_resolves_complete_custom_group() {
        let files = vec![
            "release-part-a.uqff".to_string(),
            "release-part-b.uqff".to_string(),
            "legacy.uqff".to_string(),
        ];
        let report = custom_named_uqff_report();

        assert_eq!(
            resolve_uqff_input_files("release-part-b.uqff", &files, Some(&report)).unwrap(),
            vec![
                "release-part-a.uqff".to_string(),
                "release-part-b.uqff".to_string(),
            ]
        );
        assert_eq!(
            resolve_uqff_input_files("legacy.uqff", &files, Some(&report)).unwrap(),
            vec!["legacy.uqff".to_string()]
        );
    }

    #[test]
    fn report_resolution_rejects_invalid_identity() {
        let files = vec![
            "release-part-a.uqff".to_string(),
            "release-part-b.uqff".to_string(),
        ];

        let mut empty_outputs = custom_named_uqff_report();
        empty_outputs.outputs.clear();
        let error = resolve_uqff_input_files("q8_0", &files, Some(&empty_outputs))
            .unwrap_err()
            .to_string();
        assert!(error.contains("report has no outputs"));

        let mut empty_quant = custom_named_uqff_report();
        empty_quant.outputs[0].quant.clear();
        let error = resolve_uqff_input_files("q8_0", &files, Some(&empty_quant))
            .unwrap_err()
            .to_string();
        assert!(error.contains("output quant must not be empty"));

        let mut empty_shards = custom_named_uqff_report();
        empty_shards.outputs[0].shards.clear();
        let error = resolve_uqff_input_files("q8_0", &files, Some(&empty_shards))
            .unwrap_err()
            .to_string();
        assert!(error.contains("output `q8_0` has no shards"));

        let mut missing_shard = custom_named_uqff_report();
        missing_shard.outputs[0].shards = vec!["missing.uqff".to_string()];
        let error = resolve_uqff_input_files("q8_0", &files, Some(&missing_shard))
            .unwrap_err()
            .to_string();
        assert!(error.contains("references missing shard `missing.uqff`"));

        let mut duplicate_quant = custom_named_uqff_report();
        let mut second = duplicate_quant.outputs[0].clone();
        second.quant = "Q8_0".to_string();
        second.shards = vec!["release-part-a.uqff".to_string()];
        duplicate_quant.outputs.push(second);
        let error = resolve_uqff_input_files("q8_0", &files, Some(&duplicate_quant))
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate output quant `Q8_0`"));

        let mut overlapping = custom_named_uqff_report();
        let mut second = overlapping.outputs[0].clone();
        second.quant = "q4k".to_string();
        second.shards = vec!["release-part-b.uqff".to_string()];
        overlapping.outputs.push(second);
        let error = resolve_uqff_input_files("q8_0", &files, Some(&overlapping))
            .unwrap_err()
            .to_string();
        assert!(error.contains("shard `release-part-b.uqff` belongs to both `q8_0` and `q4k`"));

        let mut duplicated_shard = custom_named_uqff_report();
        duplicated_shard.outputs[0]
            .shards
            .push("release-part-a.uqff".to_string());
        let error = resolve_uqff_input_files("q8_0", &files, Some(&duplicated_shard))
            .unwrap_err()
            .to_string();
        assert!(error.contains("shard `release-part-a.uqff` belongs to both `q8_0` and `q8_0`"));
    }

    #[test]
    fn reportless_uqff_input_keeps_filename_inference() {
        let files = vec!["q8_0-1.uqff".to_string(), "q8_0-0.uqff".to_string()];

        assert_eq!(
            resolve_uqff_input_files("q8_0", &files, None).unwrap(),
            vec!["q8_0-0.uqff".to_string(), "q8_0-1.uqff".to_string()]
        );
        assert_eq!(
            resolve_uqff_input_files("q8_0-1.uqff", &files, None).unwrap(),
            vec!["q8_0-0.uqff".to_string(), "q8_0-1.uqff".to_string()]
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
