use super::*;
use crate::{UqffReader, UqffTensor};
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
        Some(HashMap::from([
            ("uqff.producer".to_string(), "test".to_string()),
            ("uqff.version".to_string(), current_uqff_version()),
        ])),
        path,
    )
    .unwrap();
}

fn push_synthetic_afq_layer_with_format(
    tensors: &mut Vec<UqffTensor>,
    prefix: &str,
    bits: u8,
    format: u8,
) {
    tensors.extend([
        UqffTensor::from_u8_scalar(format!("{prefix}.weight.format"), format),
        UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), bits),
        UqffTensor::from_u8_scalar(format!("{prefix}.weight.group_size"), 64),
        UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0], vec![1]),
        UqffTensor::from_raw_u8(format!("{prefix}.weight.scales"), vec![0], vec![1]),
        UqffTensor::from_raw_u8(format!("{prefix}.weight.biases"), vec![0], vec![1]),
    ]);
}

fn push_synthetic_afq_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, bits: u8) {
    push_synthetic_afq_layer_with_format(tensors, prefix, bits, QuantizedSerdeType::Afq as u8);
}

fn write_synthetic_afq_shard(path: &Path, prefix: &str) {
    let mut tensors = super::super::uqff_version_tensors();
    push_synthetic_afq_layer(&mut tensors, prefix, 3);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        Some(HashMap::from([
            ("uqff.producer".to_string(), "test".to_string()),
            ("uqff.version".to_string(), current_uqff_version()),
        ])),
        path,
    )
    .unwrap();
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
    // One whole block, so the fixture describes a layer the loader can actually open:
    // `verify` rejects GGUF weights whose byte length disagrees with `weight.shape`.
    let block = crate::gguf::GgufMatMul::block_size_from_uqff_dtype(dtype).unwrap();
    let bytes = crate::gguf::uqff_gguf_weight_bytes(dtype, &[block]).unwrap();
    tensors.extend([
        UqffTensor::from_u8_scalar(
            format!("{prefix}.weight.format"),
            QuantizedSerdeType::Gguf as u8,
        ),
        UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0; bytes], vec![bytes]),
        UqffTensor::from_u32_scalar(format!("{prefix}.weight.dtype"), dtype),
        UqffTensor::from_u32_vec(
            format!("{prefix}.weight.shape"),
            vec![block as u32],
            vec![1],
        ),
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
async fn report_uses_logical_weight_shape_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4k-0.uqff");
    let mut tensors = super::super::uqff_version_tensors();
    tensors.extend([
        UqffTensor::from_u8_scalar(
            "model.layers.0.weight.format",
            QuantizedSerdeType::Gguf as u8,
        ),
        UqffTensor::from_raw_u8("model.layers.0.weight", vec![0], vec![1]),
        UqffTensor::from_u32_scalar("model.layers.0.weight.dtype", 12),
        UqffTensor::from_u32_vec("model.layers.0.weight.shape", vec![17, 23], vec![2]),
    ]);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &path,
    )
    .unwrap();

    let report = build_uqff_report(&path, UqffReportOptions::default())
        .await
        .unwrap();

    assert_eq!(report.outputs[0].layer_details[0].shape, vec![17, 23]);
}

#[tokio::test]
async fn verify_rejects_invalid_or_mismatched_format_payload_without_report() {
    for (format, expected) in [
        (
            QuantizedSerdeType::Unquant as u8,
            "tensor structure matches Afq",
        ),
        (u8::MAX, "invalid `weight.format` discriminator"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("afq3-0.uqff");
        let mut tensors = super::super::uqff_version_tensors();
        push_synthetic_afq_layer_with_format(&mut tensors, "model.layers.0", 3, format);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let result = verify_uqff_path(
            &path,
            UqffVerifyOptions {
                strict: false,
                allow_newer_minor: false,
            },
        )
        .await
        .unwrap();

        assert!(!result.ok);
        assert!(
            result.errors.iter().any(|error| error.contains(expected)),
            "{:?}",
            result.errors
        );
    }
}

#[tokio::test]
async fn verify_rejects_missing_format_payload_without_report() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("afq3-0.uqff");
    let mut tensors = super::super::uqff_version_tensors();
    tensors.extend([
        UqffTensor::from_u8_scalar("model.layers.0.weight.bits", 3),
        UqffTensor::from_u8_scalar("model.layers.0.weight.group_size", 64),
        UqffTensor::from_raw_u8("model.layers.0.weight", vec![0], vec![1]),
        UqffTensor::from_raw_u8("model.layers.0.weight.scales", vec![0], vec![1]),
        UqffTensor::from_raw_u8("model.layers.0.weight.biases", vec![0], vec![1]),
    ]);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &path,
    )
    .unwrap();

    let result = verify_uqff_path(
        &path,
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
        .any(|error| error.contains("invalid `weight.format` header")));
}

#[tokio::test]
async fn verify_rejects_gguf_weight_shorter_than_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4k-0.uqff");
    let mut tensors = super::super::uqff_version_tensors();
    tensors.extend([
        UqffTensor::from_u8_scalar(
            "model.layers.0.weight.format",
            QuantizedSerdeType::Gguf as u8,
        ),
        UqffTensor::from_raw_u8("model.layers.0.weight", vec![0; 16], vec![16]),
        UqffTensor::from_u32_scalar("model.layers.0.weight.dtype", 0),
        UqffTensor::from_u32_vec("model.layers.0.weight.shape", vec![4, 64], vec![2]),
    ]);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &path,
    )
    .unwrap();
    let result = verify_uqff_path(
        &path,
        UqffVerifyOptions {
            strict: false,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();
    assert!(!result.ok);
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.contains("GGUF `weight` holds 16 bytes")),
        "{:?}",
        result.errors
    );
}

#[tokio::test]
async fn verify_rejects_malformed_logical_shape_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q4k-0.uqff");
    let mut tensors = super::super::uqff_version_tensors();
    tensors.extend([
        UqffTensor::from_u8_scalar(
            "model.layers.0.weight.format",
            QuantizedSerdeType::Gguf as u8,
        ),
        UqffTensor::from_raw_u8("model.layers.0.weight", vec![0], vec![1]),
        UqffTensor::from_u32_scalar("model.layers.0.weight.dtype", 12),
        UqffTensor::from_u32_vec("model.layers.0.weight.shape", vec![17, 0], vec![2]),
    ]);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        None,
        &path,
    )
    .unwrap();

    let result = verify_uqff_path(
        &path,
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
        .any(|error| error.contains("invalid `weight.shape` payload")));
}

#[tokio::test]
async fn local_report_controls_custom_shards_and_exposes_missing_or_extra_files() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("afq3-0.uqff");
    let second = dir.path().join("afq3-1.uqff");
    write_synthetic_afq_shard(&first, "model.layers.0");
    write_synthetic_afq_shard(&second, "model.layers.1");
    let mut report = build_uqff_report(dir.path(), UqffReportOptions::default())
        .await
        .unwrap();
    let custom_first = dir.path().join("custom-1.uqff");
    let custom_second = dir.path().join("projection-final.uqff");
    std::fs::rename(first, &custom_first).unwrap();
    std::fs::rename(second, &custom_second).unwrap();
    report.outputs[0].shards = vec![
        "custom-1.uqff".to_string(),
        "projection-final.uqff".to_string(),
    ];
    write_uqff_report(dir.path(), &report).unwrap();

    let valid = verify_uqff_path(
        &custom_first,
        UqffVerifyOptions {
            strict: true,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();
    assert!(valid.ok, "{:?}", valid.errors);

    let inspection = inspect_uqff_path(&custom_first, UqffReportOptions::default())
        .await
        .unwrap();
    assert_eq!(inspection.report.outputs[0].quant, "afq3");
    assert!(inspection
        .tensors
        .iter()
        .all(|tensor| tensor.group == "afq3"));
    assert!(inspection
        .tensors
        .iter()
        .any(|tensor| tensor.name.starts_with("model.layers.0.")));
    assert!(inspection
        .tensors
        .iter()
        .any(|tensor| tensor.name.starts_with("model.layers.1.")));

    write_synthetic_afq_shard(&dir.path().join("surprise.uqff"), "model.layers.2");
    let extra = verify_uqff_path(
        dir.path(),
        UqffVerifyOptions {
            strict: true,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();
    assert!(!extra.ok);
    assert!(extra.errors.iter().any(|error| {
        error.contains("artifact output `surprise` is not present in the report")
    }));

    std::fs::rename(&custom_second, dir.path().join("projection-final.bak")).unwrap();
    let missing = verify_uqff_path(
        &custom_first,
        UqffVerifyOptions {
            strict: true,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();
    assert!(!missing.ok);
    assert!(missing
        .errors
        .iter()
        .any(|error| error.contains("shard list does not match artifacts")));
}

#[test]
fn report_consistency_rejects_empty_duplicate_overlapping_and_unreported_outputs() {
    let output = |quant: &str, shards: &[&str]| UqffOutputReport {
        quant: quant.to_string(),
        shards: shards.iter().map(|shard| (*shard).to_string()).collect(),
        layers: 0,
        actual_counts: BTreeMap::new(),
        fallback_count: 0,
        fallbacks: Vec::new(),
        layer_details: Vec::new(),
    };
    let existing = UqffReport {
        schema: 1,
        generated_by: UqffGeneratedBy::default(),
        base_model: None,
        repo_id: None,
        uqff_version: current_uqff_version(),
        outputs: vec![
            output("q4k", &[]),
            output("q4k", &["shared.uqff", "shared.uqff"]),
            output("q6k", &["shared.uqff"]),
        ],
    };
    let actual = UqffReport {
        outputs: vec![output("q8_0", &["artifact.uqff"])],
        ..existing.clone()
    };
    let mut errors = Vec::new();
    validate_report_consistency(&existing, &actual, &mut errors);
    validate_output_integrity(
        "artifacts",
        &[
            output("q8_0", &["artifact.uqff"]),
            output("q8_0", &["artifact.uqff"]),
        ],
        &mut errors,
    );
    validate_output_integrity(UQFF_REPORT_JSON, &[], &mut errors);

    for expected in [
        "outputs list is empty",
        "empty shard list",
        "duplicate output quant identifier `q4k`",
        "claims shard `shared.uqff` more than once",
        "both claim shard `shared.uqff`",
        "output `q4k` is not present in artifacts",
        "artifact output `q8_0` is not present in the report",
        "duplicate output quant identifier `q8_0`",
        "both claim shard `artifact.uqff`",
    ] {
        assert!(
            errors.iter().any(|error| error.contains(expected)),
            "missing {expected:?} in {errors:?}"
        );
    }
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
async fn verify_strict_accepts_reported_intentional_native_layers() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_afq3(&dir.path().join("afq3-0.uqff"));

    let mut report = build_uqff_report(dir.path(), UqffReportOptions::default())
        .await
        .unwrap();
    let output = &mut report.outputs[0];
    let native = output
        .layer_details
        .iter_mut()
        .find(|layer| layer.module == "model.layers.1")
        .unwrap();
    native.default_target = None;
    native.resolved_target = None;
    let native_module = native.module.clone();
    output
        .fallbacks
        .retain(|fallback| fallback.module != native_module);
    output.fallback_count = output.fallbacks.len();
    write_uqff_report(dir.path(), &report).unwrap();

    let result = verify_uqff_path(
        dir.path(),
        UqffVerifyOptions {
            strict: true,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();

    assert!(result.ok, "{:?}", result.errors);
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

#[tokio::test]
async fn verified_self_contained_shards_open_in_reader() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("afq3-0.uqff");
    let second = dir.path().join("afq3-1.uqff");
    write_synthetic_afq_shard(&first, "model.layers.0");
    write_synthetic_afq_shard(&second, "model.layers.1");

    let result = verify_uqff_path(
        dir.path(),
        UqffVerifyOptions {
            strict: false,
            allow_newer_minor: false,
        },
    )
    .await
    .unwrap();
    assert!(result.ok, "{:?}", result.errors);

    let reader = UqffReader::open(&[first, second]).unwrap();
    assert!(reader.contains("model.layers.0.weight"));
    assert!(reader.contains("model.layers.1.weight"));
}

#[tokio::test]
async fn verify_rejects_conflicting_version_copies() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("afq3-0.uqff");
    let second = dir.path().join("afq3-1.uqff");
    write_synthetic_afq_shard(&first, "model.layers.0");

    let mut tensors = vec![
        UqffTensor::from_u32_scalar(super::super::UQFF_VERSION_MAJOR_KEY, UQFF_VERSION_MAJOR + 1),
        UqffTensor::from_u32_scalar(super::super::UQFF_VERSION_MINOR_KEY, UQFF_VERSION_MINOR),
        UqffTensor::from_u32_scalar(super::super::UQFF_VERSION_PATCH_KEY, UQFF_VERSION_PATCH),
    ];
    push_synthetic_afq_layer(&mut tensors, "model.layers.1", 3);
    safetensors::serialize_to_file(
        tensors.iter().map(|tensor| (tensor.name(), tensor)),
        Some(HashMap::from([
            ("uqff.producer".to_string(), "test".to_string()),
            ("uqff.version".to_string(), current_uqff_version()),
        ])),
        &second,
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
        .any(|error| error.contains("conflicting `uqff.version.major`")));
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
