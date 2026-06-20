use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future::Future,
    ops::Range,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
};

use candle_core::{Error, Result};
use safetensors::tensor::{Dtype, Metadata};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::QuantizedSerdeType;

use super::{UQFF_VERSION_MAJOR, UQFF_VERSION_MINOR, UQFF_VERSION_PATCH};

pub const UQFF_REPORT_JSON: &str = "uqff_report.json";
const SAFETENSORS_HEADER_LEN_BYTES: usize = 8;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
const SMALL_METADATA_TENSOR_BYTES: usize = 4096;

#[derive(Clone, Debug, Default)]
pub struct QuantizationReport {
    issues: Arc<Mutex<Vec<QuantizationIssue>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuantizationIssue {
    pub module: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    pub shape: Vec<usize>,
    pub reason: String,
}

impl QuantizationReport {
    pub fn record_skip(
        &self,
        module: impl Into<String>,
        requested: Option<String>,
        shape: Vec<usize>,
        reason: impl Into<String>,
    ) {
        self.issues
            .lock()
            .expect("quantization report poisoned")
            .push(QuantizationIssue {
                module: module.into(),
                requested,
                shape,
                reason: reason.into(),
            });
    }

    pub fn issues(&self) -> Vec<QuantizationIssue> {
        self.issues
            .lock()
            .expect("quantization report poisoned")
            .clone()
    }

    pub fn issue_for(&self, module: &str) -> Option<QuantizationIssue> {
        self.issues()
            .into_iter()
            .find(|issue| issue.module == module)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UqffGeneratedBy {
    pub tool: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mistralrs_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_revision: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UqffReport {
    pub schema: u32,
    pub generated_by: UqffGeneratedBy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_id: Option<String>,
    pub uqff_version: String,
    pub outputs: Vec<UqffOutputReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UqffOutputReport {
    pub quant: String,
    pub shards: Vec<String>,
    pub layers: usize,
    pub actual_counts: BTreeMap<String, usize>,
    pub fallback_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<UqffFallbackReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_details: Vec<UqffLayerReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UqffLayerReport {
    pub module: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_target: Option<String>,
    pub stored: String,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UqffFallbackReport {
    pub module: String,
    pub from: String,
    pub to: String,
    pub shape: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone)]
pub struct UqffArtifactFile {
    name: String,
    read_range: Arc<dyn Fn(Range<u64>) -> RangeReadFuture + Send + Sync>,
}

type RangeReadFuture = Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'static>>;

impl std::fmt::Debug for UqffArtifactFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UqffArtifactFile")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl UqffArtifactFile {
    pub fn new<F, Fut>(name: impl Into<String>, read_range: F) -> Self
    where
        F: Fn(Range<u64>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<u8>>> + Send + 'static,
    {
        let read_range = Arc::new(move |range| Box::pin(read_range(range)) as RangeReadFuture);
        Self {
            name: name.into(),
            read_range,
        }
    }

    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        Self::new(name, move |range| {
            let path = path.clone();
            async move { read_local_range(&path, range).await }
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    async fn read_range(&self, range: Range<u64>) -> Result<Vec<u8>> {
        (self.read_range)(range).await
    }
}

#[derive(Clone, Debug)]
pub struct UqffArtifactGroup {
    pub quant: String,
    pub files: Vec<UqffArtifactFile>,
}

#[derive(Clone, Debug)]
pub struct UqffArtifacts {
    pub groups: Vec<UqffArtifactGroup>,
    pub existing_report: Option<UqffReport>,
}

#[derive(Clone, Debug, Default)]
pub struct UqffReportOptions {
    pub generated_by: UqffGeneratedBy,
    pub base_model: Option<String>,
    pub repo_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UqffVerifyOptions {
    pub strict: bool,
    pub allow_newer_minor: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UqffVerifyResult {
    pub ok: bool,
    pub report: UqffReport,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct UqffInspection {
    pub report: UqffReport,
    pub tensors: Vec<UqffTensorSummary>,
    pub metadata: Vec<UqffMetadataSummary>,
}

#[derive(Clone, Debug)]
pub struct UqffTensorSummary {
    pub group: String,
    pub shard: String,
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub size_bytes: usize,
    pub labels: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct UqffMetadataSummary {
    pub shard: String,
    pub key: String,
    pub value: String,
}

#[derive(Clone, Debug)]
struct TensorMeta {
    dtype: Dtype,
    shape: Vec<usize>,
    size_bytes: usize,
    data: Option<Vec<u8>>,
}

#[derive(Clone, Debug)]
struct GroupScan {
    quant: String,
    shards: Vec<String>,
    tensors: HashMap<String, TensorMeta>,
    tensor_sources: HashMap<String, String>,
    duplicate_names: Vec<String>,
    metadata: Vec<UqffMetadataSummary>,
}

pub async fn build_uqff_report(path: &Path, options: UqffReportOptions) -> Result<UqffReport> {
    let artifacts = UqffArtifacts {
        groups: resolve_uqff_groups(path)?,
        existing_report: None,
    };
    build_uqff_report_from_artifacts(&artifacts, options).await
}

pub async fn build_uqff_report_from_artifacts(
    artifacts: &UqffArtifacts,
    options: UqffReportOptions,
) -> Result<UqffReport> {
    let mut outputs = Vec::with_capacity(artifacts.groups.len());
    let mut version = None;
    for group in &artifacts.groups {
        let scan = scan_group(group).await?;
        if version.is_none() {
            version = read_version_string(&scan.tensors).ok();
        }
        outputs.push(build_output_report(&scan, None));
    }

    Ok(UqffReport {
        schema: 1,
        generated_by: options.generated_by,
        base_model: options.base_model,
        repo_id: options.repo_id,
        uqff_version: version.unwrap_or_else(current_uqff_version),
        outputs,
    })
}

pub fn write_uqff_report(path: &Path, report: &UqffReport) -> Result<PathBuf> {
    let output_path = report_path_for_input(path)?;
    let data = serde_json::to_vec_pretty(report)
        .map_err(|e| Error::Msg(format!("Failed to serialize UQFF report: {e}")))?;
    std::fs::write(&output_path, data).map_err(|e| Error::from(e).with_path(&output_path))?;
    Ok(output_path)
}

pub async fn inspect_uqff_path(path: &Path, options: UqffReportOptions) -> Result<UqffInspection> {
    let artifacts = UqffArtifacts {
        groups: resolve_uqff_groups(path)?,
        existing_report: read_existing_report(path)?,
    };
    inspect_uqff_artifacts(&artifacts, options).await
}

pub async fn inspect_uqff_artifacts(
    artifacts: &UqffArtifacts,
    options: UqffReportOptions,
) -> Result<UqffInspection> {
    let mut outputs = Vec::with_capacity(artifacts.groups.len());
    let mut tensor_summaries = Vec::new();
    let mut metadata = Vec::new();
    let mut version = None;

    for group in &artifacts.groups {
        let scan = scan_group(group).await?;
        if version.is_none() {
            version = read_version_string(&scan.tensors).ok();
        }
        let output = build_output_report(&scan, None);
        let label_map = labels_by_module(&output);
        for (name, tensor) in &scan.tensors {
            let module = name
                .strip_suffix(".weight")
                .or_else(|| name.strip_suffix(".weight.format"))
                .or_else(|| name.strip_suffix(".weight.dtype"))
                .or_else(|| name.strip_suffix(".weight.shape"))
                .or_else(|| name.strip_suffix(".weight.bits"))
                .or_else(|| name.strip_suffix(".weight.group_size"))
                .or_else(|| name.strip_suffix(".weight.scales"))
                .or_else(|| name.strip_suffix(".weight.biases"))
                .or_else(|| name.strip_suffix(".weight.zeros"))
                .unwrap_or(name);
            let shard = scan
                .tensor_sources
                .get(name)
                .cloned()
                .unwrap_or_else(|| scan.shards.first().cloned().unwrap_or_default());
            tensor_summaries.push(UqffTensorSummary {
                group: scan.quant.clone(),
                shard,
                name: name.clone(),
                dtype: format!("{:?}", tensor.dtype),
                shape: tensor.shape.clone(),
                size_bytes: tensor.size_bytes,
                labels: label_map.get(module).cloned().unwrap_or_default(),
            });
        }
        metadata.extend(scan.metadata.clone());
        outputs.push(output);
    }

    tensor_summaries.sort_by(|a, b| a.name.cmp(&b.name));
    metadata.sort_by(|a, b| a.key.cmp(&b.key));

    let scanned_report = UqffReport {
        schema: 1,
        generated_by: options.generated_by,
        base_model: options.base_model,
        repo_id: options.repo_id,
        uqff_version: version.unwrap_or_else(current_uqff_version),
        outputs,
    };
    let report = artifacts.existing_report.clone().unwrap_or(scanned_report);

    Ok(UqffInspection {
        report,
        tensors: tensor_summaries,
        metadata,
    })
}

pub async fn verify_uqff_path(path: &Path, options: UqffVerifyOptions) -> Result<UqffVerifyResult> {
    let artifacts = UqffArtifacts {
        groups: resolve_uqff_groups(path)?,
        existing_report: read_existing_report(path)?,
    };
    verify_uqff_artifacts(&artifacts, options).await
}

pub async fn verify_uqff_artifacts(
    artifacts: &UqffArtifacts,
    options: UqffVerifyOptions,
) -> Result<UqffVerifyResult> {
    let report_options = UqffReportOptions {
        generated_by: UqffGeneratedBy {
            tool: "mistralrs uqff verify".to_string(),
            mistralrs_version: None,
            git_revision: None,
        },
        base_model: None,
        repo_id: None,
    };
    let mut outputs = Vec::with_capacity(artifacts.groups.len());
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut version = None;

    for group in &artifacts.groups {
        let scan = match scan_group(group).await {
            Ok(scan) => scan,
            Err(err) => {
                errors.push(format!("{}: {err}", group.quant));
                continue;
            }
        };
        if !scan.duplicate_names.is_empty() {
            for name in &scan.duplicate_names {
                errors.push(format!("{}: duplicate tensor key `{name}`", scan.quant));
            }
        }
        validate_group(&scan, &mut warnings, &mut errors, &options);
        if version.is_none() {
            version = read_version_string(&scan.tensors).ok();
        }
        let output = build_output_report(&scan, None);
        if options.strict && output.fallback_count > 0 {
            errors.push(format!(
                "{}: strict mode rejects {} fallback layer{}",
                output.quant,
                output.fallback_count,
                if output.fallback_count == 1 { "" } else { "s" }
            ));
        }
        outputs.push(output);
    }

    let mut report = UqffReport {
        schema: 1,
        generated_by: report_options.generated_by,
        base_model: None,
        repo_id: None,
        uqff_version: version.unwrap_or_else(current_uqff_version),
        outputs,
    };

    match &artifacts.existing_report {
        Some(existing) => {
            validate_report_consistency(&existing, &report, &mut errors);
            report = existing.clone();
        }
        None => {
            if options.strict {
                errors.push(format!("{UQFF_REPORT_JSON} is missing"));
            }
        }
    }

    let ok = errors.is_empty();
    Ok(UqffVerifyResult {
        ok,
        report,
        warnings,
        errors,
    })
}

pub fn build_output_report_from_layers(
    quant: String,
    shards: Vec<String>,
    layers: Vec<UqffLayerReport>,
    issues: &QuantizationReport,
) -> UqffOutputReport {
    let mut actual_counts = BTreeMap::new();
    let mut fallbacks = Vec::new();
    for layer in &layers {
        *actual_counts.entry(layer.stored.clone()).or_insert(0) += 1;
        let issue = issues.issue_for(&layer.module);
        let resolved = layer
            .resolved_target
            .as_deref()
            .or(layer.default_target.as_deref())
            .unwrap_or(&quant);
        if issue.is_some() || is_fallback_storage(&layer.stored, resolved) {
            fallbacks.push(UqffFallbackReport {
                module: layer.module.clone(),
                from: resolved.to_string(),
                to: layer.stored.clone(),
                shape: issue
                    .as_ref()
                    .map(|issue| issue.shape.clone())
                    .unwrap_or_else(|| layer.shape.clone()),
                reason: issue.map(|issue| issue.reason),
            });
        }
    }
    UqffOutputReport {
        quant,
        shards,
        layers: layers.len(),
        actual_counts,
        fallback_count: fallbacks.len(),
        fallbacks,
        layer_details: layers,
    }
}

pub fn stored_type_from_tensors(
    tensors: &[super::UqffTensor],
    prefix: &str,
) -> Result<(String, Vec<usize>)> {
    let format_key = format!("{prefix}.weight.format");
    let format = tensors
        .iter()
        .find(|tensor| tensor.name() == format_key)
        .ok_or_else(|| Error::Msg(format!("Missing `{format_key}`")))?;
    let format = QuantizedSerdeType::try_from(format.scalar_u8()? as usize)?;
    let stored = stored_type_from_format_and_lookup(format, &|suffix| {
        let key = format!("{prefix}.{suffix}");
        tensors.iter().find(|tensor| tensor.name() == key)
    })?;
    let shape = tensor_shape_from_lookup(prefix, &|suffix| {
        let key = format!("{prefix}.{suffix}");
        tensors.iter().find(|tensor| tensor.name() == key)
    });
    Ok((stored, shape))
}

fn resolve_uqff_groups(path: &Path) -> Result<Vec<UqffArtifactGroup>> {
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    if path.is_dir() {
        for entry in std::fs::read_dir(path).map_err(|e| Error::from(e).with_path(path))? {
            let entry = entry.map_err(Error::from)?;
            let entry_path = entry.path();
            if entry_path.extension().and_then(|ext| ext.to_str()) == Some("uqff") {
                let quant = group_key_for_path(&entry_path);
                groups.entry(quant).or_default().push(entry_path);
            }
        }
    } else if path.is_file() {
        let quant = group_key_for_path(path);
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default();
        let discover_siblings = shard_index_from_stem(stem).is_some();
        if discover_siblings {
            for entry in std::fs::read_dir(parent).map_err(|e| Error::from(e).with_path(parent))? {
                let entry = entry.map_err(Error::from)?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|ext| ext.to_str()) == Some("uqff")
                    && group_key_for_path(&entry_path) == quant
                {
                    groups.entry(quant.clone()).or_default().push(entry_path);
                }
            }
        } else {
            groups.entry(quant).or_default().push(path.to_path_buf());
        }
    } else {
        candle_core::bail!("UQFF path `{}` does not exist.", path.display());
    }

    if groups.is_empty() {
        candle_core::bail!("No `.uqff` files found at `{}`.", path.display());
    }

    Ok(groups
        .into_iter()
        .map(|(quant, mut files)| {
            files.sort_by_key(|path| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .and_then(shard_index_from_stem)
                    .unwrap_or(0)
            });
            UqffArtifactGroup {
                quant,
                files: files.into_iter().map(UqffArtifactFile::from_path).collect(),
            }
        })
        .collect())
}

async fn scan_group(group: &UqffArtifactGroup) -> Result<GroupScan> {
    let mut tensors = HashMap::new();
    let mut tensor_sources = HashMap::new();
    let mut duplicate_names = Vec::new();
    let mut metadata = Vec::new();
    let mut seen = HashSet::new();

    for file in &group.files {
        let shard_name = file.name().to_string();
        let (data_offset, header) = read_safetensors_metadata(file).await?;

        if let Some(header_metadata) = header.metadata() {
            for (key, value) in header_metadata {
                metadata.push(UqffMetadataSummary {
                    shard: shard_name.clone(),
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        for (name, info) in header.tensors() {
            if !is_version_key(&name) && !seen.insert(name.clone()) {
                duplicate_names.push(name.clone());
            }
            let size_bytes = info.data_offsets.1 - info.data_offsets.0;
            let data = if size_bytes <= SMALL_METADATA_TENSOR_BYTES {
                let start = data_offset + info.data_offsets.0 as u64;
                let end = data_offset + info.data_offsets.1 as u64;
                Some(file.read_range(start..end).await?)
            } else {
                None
            };
            let meta = TensorMeta {
                dtype: info.dtype,
                shape: info.shape.clone(),
                size_bytes,
                data,
            };
            tensors.insert(name.clone(), meta);
            tensor_sources.insert(name, shard_name.clone());
        }
    }

    Ok(GroupScan {
        quant: group.quant.clone(),
        shards: group
            .files
            .iter()
            .map(|file| file.name().to_string())
            .collect(),
        tensors,
        tensor_sources,
        duplicate_names,
        metadata,
    })
}

async fn read_safetensors_metadata(file: &UqffArtifactFile) -> Result<(u64, Metadata)> {
    let len_bytes = file
        .read_range(0..SAFETENSORS_HEADER_LEN_BYTES as u64)
        .await?;
    let len_bytes: [u8; SAFETENSORS_HEADER_LEN_BYTES] = len_bytes
        .as_slice()
        .try_into()
        .map_err(|_| Error::Msg(format!("{}: safetensors header is too small", file.name())))?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_SAFETENSORS_HEADER_BYTES {
        candle_core::bail!(
            "{}: safetensors header is too large: {header_len} bytes",
            file.name()
        );
    }
    let header = file
        .read_range(
            SAFETENSORS_HEADER_LEN_BYTES as u64..SAFETENSORS_HEADER_LEN_BYTES as u64 + header_len,
        )
        .await?;
    let metadata = serde_json::from_slice::<Metadata>(&header)
        .map_err(|e| Error::Msg(format!("{}: invalid safetensors header: {e}", file.name())))?;
    Ok((SAFETENSORS_HEADER_LEN_BYTES as u64 + header_len, metadata))
}

async fn read_local_range(path: &Path, range: Range<u64>) -> Result<Vec<u8>> {
    if range.start > range.end {
        candle_core::bail!(
            "{}: invalid byte range {}..{}",
            path.display(),
            range.start,
            range.end
        );
    }
    let len = usize::try_from(range.end - range.start)
        .map_err(|_| Error::Msg(format!("{}: byte range is too large", path.display())))?;
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| Error::from(e).with_path(path))?;
    file.seek(std::io::SeekFrom::Start(range.start))
        .await
        .map_err(|e| Error::from(e).with_path(path))?;
    let mut data = vec![0u8; len];
    file.read_exact(&mut data)
        .await
        .map_err(|e| Error::from(e).with_path(path))?;
    Ok(data)
}

fn build_output_report(scan: &GroupScan, issues: Option<&QuantizationReport>) -> UqffOutputReport {
    let mut layers = Vec::new();
    let mut actual_counts = BTreeMap::new();
    let mut fallbacks = Vec::new();
    let mut prefixes = scan
        .tensors
        .keys()
        .filter_map(|name| name.strip_suffix(".weight.format"))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    prefixes.sort();

    for prefix in prefixes {
        let stored =
            stored_type_for_prefix(scan, &prefix).unwrap_or_else(|_| "unknown".to_string());
        let shape = tensor_shape_for_prefix(scan, &prefix);
        let layer = UqffLayerReport {
            module: prefix.clone(),
            default_target: Some(scan.quant.clone()),
            resolved_target: None,
            stored: stored.clone(),
            shape: shape.clone(),
        };
        *actual_counts.entry(stored.clone()).or_insert(0) += 1;
        let issue = issues.and_then(|issues| issues.issue_for(&prefix));
        if issue.is_some() || is_fallback_storage(&stored, &scan.quant) {
            fallbacks.push(UqffFallbackReport {
                module: prefix,
                from: scan.quant.clone(),
                to: stored,
                shape: issue
                    .as_ref()
                    .map(|issue| issue.shape.clone())
                    .unwrap_or(shape),
                reason: issue.map(|issue| issue.reason),
            });
        }
        layers.push(layer);
    }

    UqffOutputReport {
        quant: scan.quant.clone(),
        shards: scan.shards.clone(),
        layers: layers.len(),
        actual_counts,
        fallback_count: fallbacks.len(),
        fallbacks,
        layer_details: layers,
    }
}

fn validate_group(
    scan: &GroupScan,
    warnings: &mut Vec<String>,
    errors: &mut Vec<String>,
    options: &UqffVerifyOptions,
) {
    match read_version(&scan.tensors) {
        Ok((major, minor, _patch)) => {
            if major != UQFF_VERSION_MAJOR {
                errors.push(format!(
                    "{}: incompatible UQFF major version {major}; expected {}",
                    scan.quant, UQFF_VERSION_MAJOR
                ));
            } else if minor > UQFF_VERSION_MINOR && !options.allow_newer_minor {
                errors.push(format!(
                    "{}: UQFF minor version {minor} is newer than supported {}",
                    scan.quant, UQFF_VERSION_MINOR
                ));
            }
        }
        Err(err) => errors.push(format!("{}: {err}", scan.quant)),
    }

    if !has_producer_metadata(scan) {
        let msg = format!("{}: producer metadata is missing", scan.quant);
        if options.strict {
            errors.push(msg);
        } else {
            warnings.push(msg);
        }
    }

    validate_contiguous_shards(scan, errors);

    for name in scan.tensors.keys() {
        if let Some(prefix) = name.strip_suffix(".weight.format") {
            validate_layer(scan, prefix, errors);
        }
    }
}

fn validate_contiguous_shards(scan: &GroupScan, errors: &mut Vec<String>) {
    let mut indexes = scan
        .shards
        .iter()
        .filter_map(|shard| shard.strip_suffix(".uqff"))
        .filter_map(shard_index_from_stem)
        .collect::<Vec<_>>();
    if indexes.is_empty() {
        return;
    }
    indexes.sort_unstable();
    for (expected, actual) in indexes.iter().enumerate() {
        if expected as u64 != *actual {
            errors.push(format!(
                "{}: shard indexes are not contiguous; expected {expected}, saw {actual}",
                scan.quant
            ));
            return;
        }
    }
}

fn validate_layer(scan: &GroupScan, prefix: &str, errors: &mut Vec<String>) {
    let Some(format) = scan
        .tensors
        .get(&format!("{prefix}.weight.format"))
        .and_then(scalar_u8)
        .and_then(|value| QuantizedSerdeType::try_from(value as usize).ok())
    else {
        errors.push(format!("{prefix}: invalid or missing weight.format"));
        return;
    };
    for required in required_suffixes(format) {
        let key = format!("{prefix}.{required}");
        if !scan.tensors.contains_key(&key) {
            errors.push(format!("{prefix}: missing `{required}`"));
        }
    }
}

fn validate_report_consistency(
    existing: &UqffReport,
    actual: &UqffReport,
    errors: &mut Vec<String>,
) {
    let actual_outputs = actual
        .outputs
        .iter()
        .map(|output| (&output.quant, output))
        .collect::<HashMap<_, _>>();
    for output in &existing.outputs {
        let Some(actual_output) = actual_outputs.get(&output.quant) else {
            errors.push(format!(
                "{UQFF_REPORT_JSON}: output `{}` is not present in artifacts",
                output.quant
            ));
            continue;
        };
        if output.shards != actual_output.shards {
            errors.push(format!(
                "{UQFF_REPORT_JSON}: output `{}` shard list does not match artifacts",
                output.quant
            ));
        }
        if output.layers != actual_output.layers {
            errors.push(format!(
                "{UQFF_REPORT_JSON}: output `{}` layer count {} does not match artifact count {}",
                output.quant, output.layers, actual_output.layers
            ));
        }
        if output.actual_counts != actual_output.actual_counts {
            errors.push(format!(
                "{UQFF_REPORT_JSON}: output `{}` stored-format counts do not match artifacts",
                output.quant
            ));
        }
    }
}

fn read_existing_report(path: &Path) -> Result<Option<UqffReport>> {
    let report_path = report_path_for_input(path)?;
    if !report_path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(&report_path)
        .map_err(|e| Error::from(e).with_path(&report_path))?;
    serde_json::from_str(&data)
        .map(Some)
        .map_err(|e| Error::Msg(format!("{}: {e}", report_path.display())))
}

fn report_path_for_input(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        Ok(path.join(UQFF_REPORT_JSON))
    } else if path.is_file() {
        Ok(path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(UQFF_REPORT_JSON))
    } else {
        candle_core::bail!("UQFF path `{}` does not exist.", path.display());
    }
}

fn labels_by_module(output: &UqffOutputReport) -> HashMap<String, Vec<String>> {
    let mut labels = HashMap::new();
    for fallback in &output.fallbacks {
        labels
            .entry(fallback.module.clone())
            .or_insert_with(Vec::new)
            .push("fallback".to_string());
    }
    for layer in &output.layer_details {
        if layer
            .resolved_target
            .as_ref()
            .zip(layer.default_target.as_ref())
            .is_some_and(|(resolved, default)| resolved != default)
        {
            labels
                .entry(layer.module.clone())
                .or_insert_with(Vec::new)
                .push("topology".to_string());
        }
    }
    labels
}

fn required_suffixes(format: QuantizedSerdeType) -> &'static [&'static str] {
    match format {
        QuantizedSerdeType::Gguf => &["weight", "weight.dtype", "weight.shape"],
        QuantizedSerdeType::Unquant => &["weight"],
        QuantizedSerdeType::Hqq => &[
            "weight",
            "weight.scales",
            "weight.zeros",
            "weight.shape",
            "weight.bits",
            "weight.group_size",
            "weight.axis",
            "weight.optimization_steps",
            "weight.round_zeros",
            "weight.channel_wise",
        ],
        QuantizedSerdeType::Fp8 => &[
            "weight",
            "weight.dequant_w_scale",
            "weight.dequant_x_scale",
            "weight.quant_scale",
            "weight.dtype",
        ],
        QuantizedSerdeType::Afq => &[
            "weight",
            "weight.bits",
            "weight.group_size",
            "weight.scales",
            "weight.biases",
        ],
        QuantizedSerdeType::F8Q8 => &["weight", "weight.num_blocks", "weight.shape"],
        QuantizedSerdeType::Mxfp4 => &["weight", "weight.scales"],
    }
}

fn stored_type_for_prefix(scan: &GroupScan, prefix: &str) -> Result<String> {
    let format = scan
        .tensors
        .get(&format!("{prefix}.weight.format"))
        .and_then(scalar_u8)
        .ok_or_else(|| Error::Msg(format!("{prefix}: missing weight.format")))?;
    let format = QuantizedSerdeType::try_from(format as usize)?;
    stored_type_from_format_and_lookup(format, &|suffix| {
        scan.tensors.get(&format!("{prefix}.{suffix}"))
    })
}

fn stored_type_from_format_and_lookup<'a, T>(
    format: QuantizedSerdeType,
    lookup: &dyn Fn(&str) -> Option<&'a T>,
) -> Result<String>
where
    T: TensorLike,
{
    match format {
        QuantizedSerdeType::Gguf => lookup("weight.dtype")
            .and_then(TensorLike::scalar_u32)
            .map(gguf_dtype_label)
            .ok_or_else(|| Error::Msg("Missing GGUF dtype".to_string())),
        QuantizedSerdeType::Unquant => Ok("unquant".to_string()),
        QuantizedSerdeType::Hqq => lookup("weight.bits")
            .and_then(TensorLike::scalar_u8)
            .map(|bits| format!("hqq{bits}"))
            .ok_or_else(|| Error::Msg("Missing HQQ bits".to_string())),
        QuantizedSerdeType::Fp8 => Ok("fp8".to_string()),
        QuantizedSerdeType::Afq => lookup("weight.bits")
            .and_then(TensorLike::scalar_u8)
            .map(afq_bits_label)
            .ok_or_else(|| Error::Msg("Missing AFQ bits".to_string())),
        QuantizedSerdeType::F8Q8 => Ok("f8q8".to_string()),
        QuantizedSerdeType::Mxfp4 => Ok("mxfp4".to_string()),
    }
}

trait TensorLike {
    fn scalar_u8(&self) -> Option<u8>;
    fn scalar_u32(&self) -> Option<u32>;
    fn u32_values(&self) -> Option<Vec<usize>>;
    fn shape(&self) -> Vec<usize>;
}

impl TensorLike for TensorMeta {
    fn scalar_u8(&self) -> Option<u8> {
        scalar_u8(self)
    }

    fn scalar_u32(&self) -> Option<u32> {
        scalar_u32(self)
    }

    fn u32_values(&self) -> Option<Vec<usize>> {
        scalar_u32_vec(self)
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

impl TensorLike for super::UqffTensor {
    fn scalar_u8(&self) -> Option<u8> {
        self.scalar_u8().ok()
    }

    fn scalar_u32(&self) -> Option<u32> {
        self.scalar_u32().ok()
    }

    fn u32_values(&self) -> Option<Vec<usize>> {
        self.u32_values().ok()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

fn tensor_shape_for_prefix(scan: &GroupScan, prefix: &str) -> Vec<usize> {
    tensor_shape_from_lookup(prefix, &|suffix| {
        scan.tensors.get(&format!("{prefix}.{suffix}"))
    })
}

fn tensor_shape_from_lookup<'a, T>(
    prefix: &str,
    lookup: &dyn Fn(&str) -> Option<&'a T>,
) -> Vec<usize>
where
    T: TensorLike,
{
    lookup("weight.shape")
        .and_then(TensorLike::u32_values)
        .unwrap_or_else(|| {
            lookup("weight")
                .map(TensorLike::shape)
                .unwrap_or_else(|| vec![prefix.len()])
        })
}

fn scalar_u8(meta: &TensorMeta) -> Option<u8> {
    let data = meta.data.as_deref()?;
    if meta.dtype == Dtype::U8 && meta.shape.is_empty() && data.len() == 1 {
        Some(data[0])
    } else {
        None
    }
}

fn scalar_u32(meta: &TensorMeta) -> Option<u32> {
    let data = meta.data.as_deref()?;
    if meta.dtype == Dtype::U32 && meta.shape.is_empty() && data.len() == 4 {
        Some(u32::from_le_bytes(data.try_into().ok()?))
    } else {
        None
    }
}

fn scalar_u32_vec(meta: &TensorMeta) -> Option<Vec<usize>> {
    let data = meta.data.as_deref()?;
    if meta.dtype != Dtype::U32 || data.len() % 4 != 0 {
        return None;
    }
    Some(
        data.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk is 4 bytes")) as usize)
            .collect(),
    )
}

fn read_version_string(tensors: &HashMap<String, TensorMeta>) -> Result<String> {
    let (major, minor, patch) = read_version(tensors)?;
    Ok(format!("{major}.{minor}.{patch}"))
}

fn read_version(tensors: &HashMap<String, TensorMeta>) -> Result<(u32, u32, u32)> {
    let major = read_required_u32(tensors, super::UQFF_VERSION_MAJOR_KEY)?;
    let minor = read_required_u32(tensors, super::UQFF_VERSION_MINOR_KEY)?;
    let patch = read_required_u32(tensors, super::UQFF_VERSION_PATCH_KEY)?;
    Ok((major, minor, patch))
}

fn read_required_u32(tensors: &HashMap<String, TensorMeta>, key: &str) -> Result<u32> {
    tensors
        .get(key)
        .and_then(scalar_u32)
        .ok_or_else(|| Error::Msg(format!("missing UQFF version tensor `{key}`")))
}

fn has_producer_metadata(scan: &GroupScan) -> bool {
    scan.metadata
        .iter()
        .any(|metadata| metadata.key == "uqff.producer")
}

fn is_version_key(name: &str) -> bool {
    matches!(
        name,
        super::UQFF_VERSION_MAJOR_KEY
            | super::UQFF_VERSION_MINOR_KEY
            | super::UQFF_VERSION_PATCH_KEY
    )
}

fn current_uqff_version() -> String {
    format!(
        "{}.{}.{}",
        UQFF_VERSION_MAJOR, UQFF_VERSION_MINOR, UQFF_VERSION_PATCH
    )
}

fn group_key_for_path(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    if let Some((prefix, suffix)) = stem.rsplit_once('-') {
        if suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return prefix.to_string();
        }
    }
    stem.to_string()
}

fn shard_index_from_stem(stem: &str) -> Option<u64> {
    stem.rsplit_once('-')
        .and_then(|(_, suffix)| suffix.parse::<u64>().ok())
}

fn is_fallback_storage(stored: &str, target: &str) -> bool {
    stored != target && matches!(stored, "unquant" | "f32" | "f16" | "bf16")
}

fn gguf_dtype_label(dtype: u32) -> String {
    match dtype {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2k",
        11 => "q3k",
        12 => "q4k",
        13 => "q5k",
        14 => "q6k",
        15 => "q8k",
        30 => "bf16",
        _ => "unknown",
    }
    .to_string()
}

fn afq_bits_label(bits: u8) -> String {
    match bits {
        2 => "afq2",
        3 => "afq3",
        4 => "afq4",
        6 => "afq6",
        8 => "afq8",
        _ => "afq",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UqffTensor;

    fn write_synthetic_afq3(path: &Path) {
        let mut tensors = super::super::uqff_version_tensors();
        tensors.extend([
            UqffTensor::from_u8_scalar(
                "model.layers.0.weight.format",
                QuantizedSerdeType::Afq as u8,
            ),
            UqffTensor::from_u8_scalar("model.layers.0.weight.bits", 3),
            UqffTensor::from_u8_scalar("model.layers.0.weight.group_size", 64),
            UqffTensor::from_raw_u8("model.layers.0.weight", vec![0], vec![1]),
            UqffTensor::from_raw_u8("model.layers.0.weight.scales", vec![0], vec![1]),
            UqffTensor::from_raw_u8("model.layers.0.weight.biases", vec![0], vec![1]),
            UqffTensor::from_u8_scalar(
                "model.layers.1.weight.format",
                QuantizedSerdeType::Unquant as u8,
            ),
            UqffTensor::from_raw_u8("model.layers.1.weight", vec![0, 1, 2, 3], vec![2, 2]),
        ]);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            path,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn report_backfills_without_inferred_wording() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_afq3(&dir.path().join("afq3-0.uqff"));

        let report = build_uqff_report(
            dir.path(),
            UqffReportOptions {
                generated_by: UqffGeneratedBy {
                    tool: "test".to_string(),
                    mistralrs_version: Some("0.0.0".to_string()),
                    git_revision: None,
                },
                base_model: Some("base/model".to_string()),
                repo_id: Some("repo/model".to_string()),
            },
        )
        .await
        .unwrap();

        assert_eq!(report.outputs.len(), 1);
        assert_eq!(report.outputs[0].fallback_count, 1);
        assert_eq!(report.outputs[0].actual_counts.get("afq3"), Some(&1));
        assert_eq!(report.outputs[0].actual_counts.get("unquant"), Some(&1));

        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.contains("inferred"));
        assert!(!json.contains("candidate"));
        assert!(!json.contains(":null"));
    }

    #[tokio::test]
    async fn verify_strict_rejects_fallbacks() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_afq3(&dir.path().join("afq3-0.uqff"));

        let result = verify_uqff_path(
            dir.path(),
            UqffVerifyOptions {
                strict: true,
                allow_newer_minor: false,
            },
        )
        .await
        .unwrap();

        assert!(!result.ok);
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("strict mode rejects")));
    }

    #[test]
    fn quantization_report_records_fallback_reason() {
        let report = QuantizationReport::default();
        report.record_skip(
            "model.layers.0",
            Some("afq3".to_string()),
            vec![1152, 4304],
            "last dim is not divisible by group size 64",
        );
        let issue = report.issue_for("model.layers.0").unwrap();
        assert_eq!(issue.requested.as_deref(), Some("afq3"));
        assert_eq!(issue.shape, vec![1152, 4304]);
    }
}
