use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future::Future,
    ops::Range,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
};

use candle_core::{Error, Result};
use futures::future::{join_all, try_join_all};
use safetensors::tensor::{Dtype, Metadata};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::QuantizedSerdeType;

use super::{
    UqffHeaderMatch, UqffLayerHeaderView, UqffTensorHeader, UQFF_VERSION_MAJOR, UQFF_VERSION_MINOR,
    UQFF_VERSION_PATCH, UQFF_WEIGHT_FORMAT_SUFFIX,
};

pub const UQFF_REPORT_JSON: &str = "uqff_report.json";
const SAFETENSORS_HEADER_LEN_BYTES: usize = 8;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
const SMALL_METADATA_TENSOR_BYTES: usize = 4096;
const SCALAR_BATCH_MAX_GAP_BYTES: u64 = 64 * 1024;
const SCALAR_BATCH_MAX_BYTES: u64 = 1024 * 1024;
const UQFF_METADATA_VERSION: &str = "uqff.version";

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
    pub stored: Option<QuantizedSerdeType>,
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

#[derive(Clone, Copy, Debug)]
enum ScanMode {
    HeaderOnly,
    Verify,
}

#[derive(Clone, Debug)]
struct TensorDataRange {
    name: String,
    range: Range<u64>,
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
    let scans = scan_groups(&artifacts.groups, ScanMode::HeaderOnly).await?;
    for scan in scans {
        if version.is_none() {
            version = version_string_for_scan(&scan);
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
    let scans = scan_groups(&artifacts.groups, ScanMode::HeaderOnly).await?;

    for scan in scans {
        if version.is_none() {
            version = version_string_for_scan(&scan);
        }
        let headers = header_map_for_scan(&scan);
        let prefixes = layer_prefixes(&scan);
        let output = build_output_report(&scan, None);
        let label_map = labels_by_module(&output);
        let serde_type_map = serde_type_by_module(&prefixes, &headers);
        for (name, tensor) in &scan.tensors {
            let module = owning_layer_prefix(name, &prefixes);
            let shard = scan
                .tensor_sources
                .get(name)
                .cloned()
                .unwrap_or_else(|| scan.shards.first().cloned().unwrap_or_default());
            let mut labels = module
                .and_then(|module| label_map.get(module))
                .cloned()
                .unwrap_or_default();
            if is_invalid_uqff_tensor_key(name) {
                labels.push("invalid".to_string());
                labels.push("old-format".to_string());
            }
            tensor_summaries.push(UqffTensorSummary {
                group: scan.quant.clone(),
                shard,
                name: name.clone(),
                dtype: format!("{:?}", tensor.dtype),
                shape: tensor.shape.clone(),
                size_bytes: tensor.size_bytes,
                labels,
                stored: module.and_then(|module| serde_type_map.get(module).copied()),
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

    let scans = scan_groups_lossy(&artifacts.groups, ScanMode::Verify).await;
    for (group, scan) in artifacts.groups.iter().zip(scans) {
        let scan = match scan {
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
            version = version_string_for_scan(&scan);
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
            validate_report_consistency(existing, &report, &mut errors);
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
    let format = super::tensor_with_suffix(tensors, prefix, UQFF_WEIGHT_FORMAT_SUFFIX)
        .ok_or_else(|| Error::Msg(format!("Missing `{prefix}.{UQFF_WEIGHT_FORMAT_SUFFIX}`")))?;
    let format = QuantizedSerdeType::try_from(format.scalar_u8()? as usize)?;
    let stored = format.stored_label_from_uqff_tensors(tensors, prefix)?;
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

async fn scan_groups(groups: &[UqffArtifactGroup], mode: ScanMode) -> Result<Vec<GroupScan>> {
    try_join_all(groups.iter().map(|group| scan_group(group, mode))).await
}

async fn scan_groups_lossy(groups: &[UqffArtifactGroup], mode: ScanMode) -> Vec<Result<GroupScan>> {
    join_all(groups.iter().map(|group| scan_group(group, mode))).await
}

async fn scan_group(group: &UqffArtifactGroup, mode: ScanMode) -> Result<GroupScan> {
    let mut tensors = HashMap::new();
    let mut tensor_sources = HashMap::new();
    let mut duplicate_names = Vec::new();
    let mut metadata = Vec::new();
    let mut seen = HashSet::new();

    for file in &group.files {
        let shard_name = file.name().to_string();
        let (data_offset, header) = read_safetensors_metadata(file).await?;
        let mut data_ranges = Vec::new();

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
            if should_read_tensor_data(&name, size_bytes, mode) {
                let start = data_offset + info.data_offsets.0 as u64;
                let end = data_offset + info.data_offsets.1 as u64;
                data_ranges.push(TensorDataRange {
                    name: name.clone(),
                    range: start..end,
                });
            }
            let meta = TensorMeta {
                dtype: info.dtype,
                shape: info.shape.clone(),
                size_bytes,
                data: None,
            };
            tensors.insert(name.clone(), meta);
            tensor_sources.insert(name, shard_name.clone());
        }

        for (name, data) in read_tensor_data_ranges(file, data_ranges).await? {
            if let Some(meta) = tensors.get_mut(&name) {
                meta.data = Some(data);
            }
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

fn should_read_tensor_data(name: &str, size_bytes: usize, mode: ScanMode) -> bool {
    if size_bytes > SMALL_METADATA_TENSOR_BYTES {
        return false;
    }
    is_storage_discriminator_key(name) || matches!(mode, ScanMode::Verify) && is_version_key(name)
}

async fn read_tensor_data_ranges(
    file: &UqffArtifactFile,
    mut ranges: Vec<TensorDataRange>,
) -> Result<HashMap<String, Vec<u8>>> {
    ranges.sort_by_key(|entry| entry.range.start);
    let mut out = HashMap::new();
    let mut batch_start = None;
    let mut batch_end = 0u64;
    let mut batch = Vec::new();

    for entry in ranges {
        let Some(start) = batch_start else {
            batch_end = entry.range.end;
            batch_start = Some(entry.range.start);
            batch.push(entry);
            continue;
        };
        let next_end = batch_end.max(entry.range.end);
        let gap = entry.range.start.saturating_sub(batch_end);
        if gap <= SCALAR_BATCH_MAX_GAP_BYTES && next_end - start <= SCALAR_BATCH_MAX_BYTES {
            batch_end = next_end;
            batch.push(entry);
        } else {
            read_tensor_data_batch(file, start, batch_end, &batch, &mut out).await?;
            batch_end = entry.range.end;
            batch_start = Some(entry.range.start);
            batch.clear();
            batch.push(entry);
        }
    }

    if let Some(start) = batch_start {
        read_tensor_data_batch(file, start, batch_end, &batch, &mut out).await?;
    }

    Ok(out)
}

async fn read_tensor_data_batch(
    file: &UqffArtifactFile,
    start: u64,
    end: u64,
    batch: &[TensorDataRange],
    out: &mut HashMap<String, Vec<u8>>,
) -> Result<()> {
    let data = file.read_range(start..end).await?;
    for entry in batch {
        let rel_start = usize::try_from(entry.range.start - start)
            .map_err(|_| Error::Msg(format!("{}: byte range is too large", file.name())))?;
        let rel_end = usize::try_from(entry.range.end - start)
            .map_err(|_| Error::Msg(format!("{}: byte range is too large", file.name())))?;
        out.insert(entry.name.clone(), data[rel_start..rel_end].to_vec());
    }
    Ok(())
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
    let headers = header_map_for_scan(scan);
    let prefixes = layer_prefixes(scan);

    for prefix in prefixes {
        let stored = unique_layer_match(&prefix, &headers)
            .map(|matched| stored_label_for_scan(scan, &prefix, matched.serde_type))
            .unwrap_or_else(|| "unknown".to_string());
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
    validate_version_tensor_headers(scan, errors);
    match read_version_for_verify(scan) {
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
    validate_tensor_keys(scan, errors);

    let headers = header_map_for_scan(scan);
    for prefix in layer_prefixes(scan) {
        validate_layer(&prefix, &headers, errors);
    }
}

fn validate_version_tensor_headers(scan: &GroupScan, errors: &mut Vec<String>) {
    for key in [
        super::UQFF_VERSION_MAJOR_KEY,
        super::UQFF_VERSION_MINOR_KEY,
        super::UQFF_VERSION_PATCH_KEY,
    ] {
        validate_header_tensor(scan, key, Dtype::U32, &[], errors);
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

fn validate_tensor_keys(scan: &GroupScan, errors: &mut Vec<String>) {
    let mut invalid_by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in scan
        .tensors
        .keys()
        .filter(|name| is_invalid_uqff_tensor_key(name))
    {
        let shard = scan
            .tensor_sources
            .get(name)
            .cloned()
            .unwrap_or_else(|| scan.shards.first().cloned().unwrap_or_default());
        invalid_by_shard
            .entry(shard)
            .or_default()
            .push(name.clone());
    }
    for (shard, mut names) in invalid_by_shard {
        names.sort_by(|a, b| {
            a.parse::<usize>()
                .ok()
                .cmp(&b.parse::<usize>().ok())
                .then_with(|| a.cmp(b))
        });
        let examples = names.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
        errors.push(format!(
            "{}: `{shard}` contains {} invalid old-format tensor key{} with no namespace{}",
            scan.quant,
            names.len(),
            if names.len() == 1 { "" } else { "s" },
            if examples.is_empty() {
                String::new()
            } else {
                format!(" (examples: {examples})")
            }
        ));
    }
}

fn validate_layer(
    prefix: &str,
    headers: &HashMap<String, UqffTensorHeader>,
    errors: &mut Vec<String>,
) {
    let layer = UqffLayerHeaderView::new(prefix, headers);
    if !layer.scalar(UQFF_WEIGHT_FORMAT_SUFFIX, Dtype::U8) {
        errors.push(format!("{prefix}: invalid `weight.format` header"));
    }
    match layer_matches(prefix, headers).as_slice() {
        [] => errors.push(format!("{prefix}: unrecognized UQFF layer structure")),
        [_] => {}
        matches => {
            let labels = matches
                .iter()
                .map(|matched| format!("{:?}", matched.serde_type))
                .collect::<Vec<_>>()
                .join(", ");
            errors.push(format!(
                "{prefix}: ambiguous UQFF layer structure: {labels}"
            ));
        }
    }
}

fn validate_header_tensor(
    scan: &GroupScan,
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    errors: &mut Vec<String>,
) {
    let Some(tensor) = scan.tensors.get(key) else {
        errors.push(format!("{}: missing `{key}`", scan.quant));
        return;
    };
    if tensor.dtype != dtype || tensor.shape != shape {
        errors.push(format!("{}: invalid `{key}` header", scan.quant));
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

fn header_map_for_scan(scan: &GroupScan) -> HashMap<String, UqffTensorHeader> {
    scan.tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                UqffTensorHeader {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                },
            )
        })
        .collect()
}

fn layer_prefixes(scan: &GroupScan) -> Vec<String> {
    let suffix = format!(".{UQFF_WEIGHT_FORMAT_SUFFIX}");
    let mut prefixes = scan
        .tensors
        .keys()
        .filter_map(|name| name.strip_suffix(&suffix))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    prefixes.sort();
    prefixes
}

fn layer_matches(
    prefix: &str,
    headers: &HashMap<String, UqffTensorHeader>,
) -> Vec<UqffHeaderMatch> {
    let layer = UqffLayerHeaderView::new(prefix, headers);
    QuantizedSerdeType::ALL
        .into_iter()
        .filter_map(|ty| ty.inspect_uqff_header(&layer))
        .collect()
}

fn unique_layer_match(
    prefix: &str,
    headers: &HashMap<String, UqffTensorHeader>,
) -> Option<UqffHeaderMatch> {
    let mut matches = layer_matches(prefix, headers);
    if matches.len() == 1 {
        matches.pop()
    } else {
        None
    }
}

fn serde_type_by_module(
    prefixes: &[String],
    headers: &HashMap<String, UqffTensorHeader>,
) -> HashMap<String, QuantizedSerdeType> {
    prefixes
        .iter()
        .filter_map(|prefix| {
            unique_layer_match(prefix, headers).map(|matched| (prefix.clone(), matched.serde_type))
        })
        .collect()
}

fn owning_layer_prefix<'a>(name: &str, prefixes: &'a [String]) -> Option<&'a str> {
    prefixes
        .iter()
        .filter(|prefix| tensor_belongs_to_prefix(name, prefix))
        .max_by_key(|prefix| prefix.len())
        .map(String::as_str)
}

fn tensor_belongs_to_prefix(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .and_then(|suffix| suffix.strip_prefix('.'))
        .is_some_and(|suffix| {
            suffix == "weight" || suffix.starts_with("weight.") || suffix == "bias"
        })
}

trait TensorLike {
    fn u32_values(&self) -> Option<Vec<usize>>;
    fn shape(&self) -> Vec<usize>;
}

impl TensorLike for TensorMeta {
    fn u32_values(&self) -> Option<Vec<usize>> {
        scalar_u32_vec(self)
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

impl TensorLike for super::UqffTensor {
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

fn scalar_u32(meta: &TensorMeta) -> Option<u32> {
    let data = meta.data.as_deref()?;
    if meta.dtype == Dtype::U32 && meta.shape.is_empty() && data.len() == 4 {
        Some(u32::from_le_bytes(data.try_into().ok()?))
    } else {
        None
    }
}

fn scalar_u8(meta: &TensorMeta) -> Option<u8> {
    let data = meta.data.as_deref()?;
    if meta.dtype == Dtype::U8 && meta.shape.is_empty() && data.len() == 1 {
        data.first().copied()
    } else {
        None
    }
}

fn stored_label_for_scan(scan: &GroupScan, prefix: &str, serde_type: QuantizedSerdeType) -> String {
    let tensors = ["weight.bits", "weight.dtype"]
        .into_iter()
        .filter_map(|suffix| {
            let name = format!("{prefix}.{suffix}");
            let meta = scan.tensors.get(&name)?;
            match meta.dtype {
                Dtype::U8 => {
                    scalar_u8(meta).map(|value| super::UqffTensor::from_u8_scalar(name, value))
                }
                Dtype::U32 => {
                    scalar_u32(meta).map(|value| super::UqffTensor::from_u32_scalar(name, value))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    serde_type
        .stored_label_from_uqff_tensors(&tensors, prefix)
        .unwrap_or_else(|_| serde_type.stored_label(&scan.quant))
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

fn version_string_for_scan(scan: &GroupScan) -> Option<String> {
    version_string_from_metadata(&scan.metadata).or_else(|| read_version_string(&scan.tensors).ok())
}

fn version_string_from_metadata(metadata: &[UqffMetadataSummary]) -> Option<String> {
    metadata
        .iter()
        .find(|entry| entry.key == UQFF_METADATA_VERSION)
        .map(|entry| entry.value.clone())
}

fn version_from_metadata(metadata: &[UqffMetadataSummary]) -> Result<(u32, u32, u32)> {
    let Some(version) = version_string_from_metadata(metadata) else {
        candle_core::bail!(
            "missing UQFF version tensor `{}`",
            super::UQFF_VERSION_MAJOR_KEY
        );
    };
    parse_version_string(&version)
}

fn parse_version_string(version: &str) -> Result<(u32, u32, u32)> {
    let parts = version.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        candle_core::bail!("invalid UQFF version metadata `{version}`");
    }
    let major = parts[0]
        .parse::<u32>()
        .map_err(|_| Error::Msg(format!("invalid UQFF version metadata `{version}`")))?;
    let minor = parts[1]
        .parse::<u32>()
        .map_err(|_| Error::Msg(format!("invalid UQFF version metadata `{version}`")))?;
    let patch = parts[2]
        .parse::<u32>()
        .map_err(|_| Error::Msg(format!("invalid UQFF version metadata `{version}`")))?;
    Ok((major, minor, patch))
}

fn read_version(tensors: &HashMap<String, TensorMeta>) -> Result<(u32, u32, u32)> {
    let major = read_required_u32(tensors, super::UQFF_VERSION_MAJOR_KEY)?;
    let minor = read_required_u32(tensors, super::UQFF_VERSION_MINOR_KEY)?;
    let patch = read_required_u32(tensors, super::UQFF_VERSION_PATCH_KEY)?;
    Ok((major, minor, patch))
}

fn read_version_for_verify(scan: &GroupScan) -> Result<(u32, u32, u32)> {
    let has_version_tensors = [
        super::UQFF_VERSION_MAJOR_KEY,
        super::UQFF_VERSION_MINOR_KEY,
        super::UQFF_VERSION_PATCH_KEY,
    ]
    .iter()
    .all(|key| scan.tensors.contains_key(*key));
    if has_version_tensors {
        read_version(&scan.tensors)
    } else {
        version_from_metadata(&scan.metadata)
    }
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

fn is_storage_discriminator_key(name: &str) -> bool {
    name.ends_with(".weight.bits") || name.ends_with(".weight.dtype")
}

fn is_invalid_uqff_tensor_key(name: &str) -> bool {
    !is_version_key(name) && !name.contains('.')
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UqffTensor;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

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

    fn push_synthetic_afq_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, bits: u8) {
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Afq as u8,
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), bits),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.group_size"), 64),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0], vec![1]),
            UqffTensor::from_raw_u8(format!("{prefix}.weight.scales"), vec![0], vec![1]),
            UqffTensor::from_raw_u8(format!("{prefix}.weight.biases"), vec![0], vec![1]),
        ]);
    }

    fn push_synthetic_hqq_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, bits: u8) {
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Hqq as u8,
            ),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0], vec![1]),
            UqffTensor::from_raw_u8(format!("{prefix}.weight.scales"), vec![0], vec![1]),
            UqffTensor::from_raw_u8(format!("{prefix}.weight.zeros"), vec![0], vec![1]),
            UqffTensor::from_u32_vec(format!("{prefix}.weight.shape"), vec![1], vec![1]),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), bits),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.group_size"), 64),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.axis"), 0),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.optimization_steps"), 0),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.round_zeros"), 0),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.channel_wise"), 1),
        ]);
    }

    fn push_synthetic_gguf_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, dtype: u32) {
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Gguf as u8,
            ),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0], vec![1]),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.dtype"), dtype),
            UqffTensor::from_u32_vec(format!("{prefix}.weight.shape"), vec![1], vec![1]),
        ]);
    }

    fn write_synthetic_mixed_formats(dir: &Path) {
        let mut afq = super::super::uqff_version_tensors();
        push_synthetic_afq_layer(&mut afq, "model.layers.0", 4);
        push_synthetic_afq_layer(&mut afq, "model.layers.1", 6);
        safetensors::serialize_to_file(
            afq.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &dir.join("afq4-0.uqff"),
        )
        .unwrap();

        let mut hqq = super::super::uqff_version_tensors();
        push_synthetic_hqq_layer(&mut hqq, "model.layers.0", 4);
        push_synthetic_hqq_layer(&mut hqq, "model.layers.1", 8);
        safetensors::serialize_to_file(
            hqq.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &dir.join("hqq4-0.uqff"),
        )
        .unwrap();

        let mut gguf = super::super::uqff_version_tensors();
        push_synthetic_gguf_layer(&mut gguf, "model.layers.0", 12);
        push_synthetic_gguf_layer(&mut gguf, "model.layers.1", 14);
        safetensors::serialize_to_file(
            gguf.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &dir.join("q4k-0.uqff"),
        )
        .unwrap();
    }

    fn assert_stored_counts(report: &UqffReport, quant: &str, expected: [(&str, usize); 2]) {
        let output = report
            .outputs
            .iter()
            .find(|output| output.quant == quant)
            .unwrap();
        assert_eq!(output.actual_counts.len(), expected.len());
        for (stored, count) in expected {
            assert_eq!(output.actual_counts.get(stored), Some(&count));
        }
    }

    fn counted_artifact(path: &Path) -> (UqffArtifactFile, Arc<AtomicUsize>) {
        let data = Arc::new(std::fs::read(path).unwrap());
        let reads = Arc::new(AtomicUsize::new(0));
        let file = UqffArtifactFile::new("afq3-0.uqff", {
            let data = data.clone();
            let reads = reads.clone();
            move |range: Range<u64>| {
                let data = data.clone();
                let reads = reads.clone();
                async move {
                    reads.fetch_add(1, Ordering::Relaxed);
                    Ok(data[range.start as usize..range.end as usize].to_vec())
                }
            }
        });
        (file, reads)
    }

    fn counted_artifacts(file: UqffArtifactFile) -> UqffArtifacts {
        UqffArtifacts {
            groups: vec![UqffArtifactGroup {
                quant: "afq3".to_string(),
                files: vec![file],
            }],
            existing_report: None,
        }
    }

    #[tokio::test]
    async fn inspect_batches_discriminator_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("afq3-0.uqff");
        write_synthetic_afq3(&path);
        let (file, reads) = counted_artifact(&path);

        inspect_uqff_artifacts(&counted_artifacts(file), UqffReportOptions::default())
            .await
            .unwrap();

        assert_eq!(reads.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn report_batches_discriminator_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("afq3-0.uqff");
        write_synthetic_afq3(&path);
        let (file, reads) = counted_artifact(&path);

        build_uqff_report_from_artifacts(&counted_artifacts(file), UqffReportOptions::default())
            .await
            .unwrap();

        assert_eq!(reads.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn verify_batches_metadata_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("afq3-0.uqff");
        write_synthetic_afq3(&path);
        let (file, reads) = counted_artifact(&path);

        verify_uqff_artifacts(
            &counted_artifacts(file),
            UqffVerifyOptions {
                strict: false,
                allow_newer_minor: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(reads.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn mixed_storage_discriminators_drive_report_inspect_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_mixed_formats(dir.path());

        let report = build_uqff_report(dir.path(), UqffReportOptions::default())
            .await
            .unwrap();
        assert_stored_counts(&report, "afq4", [("afq4", 1), ("afq6", 1)]);
        assert_stored_counts(&report, "hqq4", [("hqq4", 1), ("hqq8", 1)]);
        assert_stored_counts(&report, "q4k", [("q4k", 1), ("q6k", 1)]);

        let inspection = inspect_uqff_path(dir.path(), UqffReportOptions::default())
            .await
            .unwrap();
        assert_eq!(inspection.report.outputs, report.outputs);

        let artifacts = UqffArtifacts {
            groups: resolve_uqff_groups(dir.path()).unwrap(),
            existing_report: Some(report),
        };
        let result = verify_uqff_artifacts(
            &artifacts,
            UqffVerifyOptions {
                strict: false,
                allow_newer_minor: false,
            },
        )
        .await
        .unwrap();
        assert!(result.ok, "{:?}", result.errors);
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

    #[tokio::test]
    async fn verify_rejects_old_numeric_sibling_shard() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_afq3(&dir.path().join("afq3-0.uqff"));
        let old_tensors = [
            UqffTensor::from_raw_u8("234", vec![0], vec![1]),
            UqffTensor::from_raw_u8("235", vec![0], vec![1]),
        ];
        safetensors::serialize_to_file(
            old_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &dir.path().join("afq3-1.uqff"),
        )
        .unwrap();

        let result = verify_uqff_path(
            dir.path(),
            UqffVerifyOptions {
                strict: false,
                allow_newer_minor: false,
            },
        )
        .await
        .unwrap();

        assert!(!result.ok);
        assert!(result
            .errors
            .iter()
            .any(|error| error.contains("invalid old-format tensor key")));
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
