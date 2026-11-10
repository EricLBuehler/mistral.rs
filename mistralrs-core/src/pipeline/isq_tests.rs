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
        "compression_config": {"quant_method": "compressed-tensors"},
        "text_config": {
            "hidden_size": 128,
            "quantization_config": {"quant_method": "fp8"},
            "compression_config": {"quant_method": "compressed-tensors"}
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
    assert!(sanitized.get("compression_config").is_none());
    assert!(sanitized["text_config"]
        .get("quantization_config")
        .is_none());
    assert!(sanitized["text_config"].get("compression_config").is_none());
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
        video_preprocessor_config: None,
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
    assert!(std::fs::read_dir(dir.path())?
        .all(|entry| entry
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
