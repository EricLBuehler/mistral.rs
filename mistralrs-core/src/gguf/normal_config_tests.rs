use super::*;
use crate::models;
use serde::de::DeserializeOwned;
use std::collections::HashSet;
use strum::IntoEnumIterator;

const LAYER_COUNT: usize = 4;

fn insert_u32(
    metadata: &mut HashMap<String, GgufValue>,
    architecture: CanonicalGgufArchitecture,
    suffix: &str,
    value: u32,
) {
    metadata.insert(
        format!("{}.{suffix}", architecture.as_str()),
        GgufValue::U32(value),
    );
}

fn insert_f32(
    metadata: &mut HashMap<String, GgufValue>,
    architecture: CanonicalGgufArchitecture,
    suffix: &str,
    value: f32,
) {
    metadata.insert(
        format!("{}.{suffix}", architecture.as_str()),
        GgufValue::F32(value),
    );
}

fn insert_bool(
    metadata: &mut HashMap<String, GgufValue>,
    architecture: CanonicalGgufArchitecture,
    suffix: &str,
    value: bool,
) {
    metadata.insert(
        format!("{}.{suffix}", architecture.as_str()),
        GgufValue::Bool(value),
    );
}

fn insert_string(
    metadata: &mut HashMap<String, GgufValue>,
    architecture: CanonicalGgufArchitecture,
    suffix: &str,
    value: &str,
) {
    metadata.insert(
        format!("{}.{suffix}", architecture.as_str()),
        GgufValue::String(value.to_string()),
    );
}

fn insert_u32_array(
    metadata: &mut HashMap<String, GgufValue>,
    architecture: CanonicalGgufArchitecture,
    suffix: &str,
    values: &[u32],
) {
    metadata.insert(
        format!("{}.{suffix}", architecture.as_str()),
        GgufValue::Array(values.iter().copied().map(GgufValue::U32).collect()),
    );
}

fn is_moe(loader: &NormalLoaderType) -> bool {
    matches!(
        loader,
        NormalLoaderType::Mixtral
            | NormalLoaderType::Phi3_5MoE
            | NormalLoaderType::DeepSeekV2
            | NormalLoaderType::DeepSeekV3
            | NormalLoaderType::GLM4MoeLite
            | NormalLoaderType::GLM4Moe
            | NormalLoaderType::Qwen3Moe
            | NormalLoaderType::Flint
            | NormalLoaderType::GptOss
            | NormalLoaderType::HunYuanMoEV1
            | NormalLoaderType::Qwen3Next
            | NormalLoaderType::Lfm2Moe
    )
}

fn fixture(
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
) -> (HashMap<String, GgufValue>, Vec<String>) {
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String(architecture.as_str().to_string()),
    );
    insert_u32(&mut metadata, architecture, "vocab_size", 32_000);
    insert_u32(&mut metadata, architecture, "context_length", 4_096);
    insert_u32(&mut metadata, architecture, "embedding_length", 512);
    insert_u32(
        &mut metadata,
        architecture,
        "block_count",
        u32::try_from(LAYER_COUNT).expect("test layer count fits u32"),
    );
    insert_u32(&mut metadata, architecture, "feed_forward_length", 1_024);
    insert_u32(&mut metadata, architecture, "attention.head_count", 8);
    insert_u32(&mut metadata, architecture, "attention.head_count_kv", 4);
    insert_u32(&mut metadata, architecture, "attention.key_length", 64);
    insert_f32(
        &mut metadata,
        architecture,
        "attention.layer_norm_rms_epsilon",
        1e-5,
    );
    insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
    insert_f32(&mut metadata, architecture, "rope.freq_base", 10_000.0);

    let mut tensors = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "blk.0.attn_q.weight".to_string(),
        "blk.0.ffn_gate.weight".to_string(),
    ];

    if is_moe(loader) {
        insert_u32(&mut metadata, architecture, "expert_count", 8);
        insert_u32(&mut metadata, architecture, "expert_used_count", 2);
        insert_u32(
            &mut metadata,
            architecture,
            "expert_feed_forward_length",
            1_024,
        );
        insert_u32(&mut metadata, architecture, "expert_shared_count", 1);
        insert_u32(
            &mut metadata,
            architecture,
            "expert_shared_feed_forward_length",
            512,
        );
        insert_u32(&mut metadata, architecture, "expert_group_count", 1);
        insert_u32(&mut metadata, architecture, "expert_group_used_count", 1);
        insert_u32(&mut metadata, architecture, "leading_dense_block_count", 0);
        insert_f32(&mut metadata, architecture, "expert_weights_scale", 1.0);
        insert_bool(&mut metadata, architecture, "expert_weights_norm", true);
        insert_u32(
            &mut metadata,
            architecture,
            "expert_gating_func",
            if matches!(loader, NormalLoaderType::Lfm2Moe | NormalLoaderType::Flint) {
                2
            } else {
                1
            },
        );
        tensors.extend(
            [
                "blk.1.ffn_gate_inp.weight",
                "blk.1.ffn_gate_exps.weight",
                "blk.1.ffn_up_exps.weight",
                "blk.1.ffn_down_exps.weight",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }

    if matches!(
        loader,
        NormalLoaderType::DeepSeekV2 | NormalLoaderType::DeepSeekV3 | NormalLoaderType::GLM4MoeLite
    ) {
        insert_u32(&mut metadata, architecture, "rope.dimension_count", 32);
        insert_u32(&mut metadata, architecture, "attention.q_lora_rank", 128);
        insert_u32(&mut metadata, architecture, "attention.kv_lora_rank", 64);
        insert_u32(&mut metadata, architecture, "attention.key_length_mla", 64);
        insert_u32(
            &mut metadata,
            architecture,
            "attention.value_length_mla",
            32,
        );
        tensors.extend(
            ["blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight"]
                .into_iter()
                .map(str::to_string),
        );
    }

    if matches!(
        loader,
        NormalLoaderType::Qwen3Next
            | NormalLoaderType::Qwen3_5
            | NormalLoaderType::GraniteMoeHybrid
    ) {
        insert_u32(&mut metadata, architecture, "ssm.conv_kernel", 4);
        insert_u32(&mut metadata, architecture, "ssm.inner_size", 1_024);
        insert_u32(&mut metadata, architecture, "ssm.state_size", 64);
        insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 8);
        insert_u32(&mut metadata, architecture, "ssm.group_count", 4);
    }

    if matches!(
        loader,
        NormalLoaderType::Qwen3Next | NormalLoaderType::Qwen3_5
    ) {
        insert_u32(&mut metadata, architecture, "full_attention_interval", 4);
        tensors.extend(
            ["blk.1.ssm_a.weight", "blk.1.ssm_conv1d.weight"]
                .into_iter()
                .map(str::to_string),
        );
        if architecture == CanonicalGgufArchitecture::Qwen35Moe {
            tensors.extend(
                ["blk.1.ssm_alpha.weight", "blk.1.ssm_beta.weight"]
                    .into_iter()
                    .map(str::to_string),
            );
        }
        if architecture == CanonicalGgufArchitecture::Qwen35 {
            insert_u32_array(
                &mut metadata,
                architecture,
                "rope.dimension_sections",
                &[11, 11, 10, 0],
            );
            tensors.extend(
                ["blk.1.ssm_alpha.weight", "blk.1.ssm_beta.weight"]
                    .into_iter()
                    .map(str::to_string),
            );
        }
    }

    if matches!(loader, NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe) {
        tensors.retain(|tensor| tensor != "output_norm.weight");
        insert_u32_array(
            &mut metadata,
            architecture,
            "attention.head_count_kv",
            &[4, 0, 4, 0],
        );
        insert_u32(&mut metadata, architecture, "shortconv.l_cache", 3);
        tensors.extend(
            [
                "blk.1.shortconv.conv.weight",
                "blk.1.shortconv.in_proj.weight",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }

    if matches!(loader, NormalLoaderType::Lfm3) {
        insert_u32_array(
            &mut metadata,
            architecture,
            "attention.head_count_kv",
            &[0, 4, 4, 0],
        );
        metadata.insert(
            format!("{}.attention.sliding_window_pattern", architecture.as_str()),
            GgufValue::Array(
                [false, false, true, false]
                    .into_iter()
                    .map(GgufValue::Bool)
                    .collect(),
            ),
        );
        insert_u32(&mut metadata, architecture, "attention.sliding_window", 256);
        insert_u32(&mut metadata, architecture, "shortconv.l_cache", 7);
        insert_f32(&mut metadata, architecture, "rope.freq_base", 1_000_000.0);
        insert_f32(&mut metadata, architecture, "rope.freq_base_swa", 10_000.0);
        tensors.retain(|tensor| tensor != "blk.0.attn_q.weight");
        tensors.push("rope_freqs.weight".to_string());
        tensors.push("blk.1.attn_q.weight".to_string());
        for layer in 0..LAYER_COUNT {
            tensors.push(format!("blk.{layer}.post_attention_norm.weight"));
            tensors.push(format!("blk.{layer}.post_ffw_norm.weight"));
            if layer == 0 || layer == 3 {
                tensors.push(format!("blk.{layer}.shortconv.conv.weight"));
                tensors.push(format!("blk.{layer}.shortconv.in_proj.weight"));
            }
        }
    }

    if matches!(loader, NormalLoaderType::Gemma2 | NormalLoaderType::GptOss) {
        insert_u32(
            &mut metadata,
            architecture,
            "attention.sliding_window",
            1_024,
        );
    }

    if matches!(loader, NormalLoaderType::Flint) {
        insert_u32(&mut metadata, architecture, "attention.sliding_window", 513);
        for suffix in ["attention.sliding_window_pattern", "attention.rope_pattern"] {
            metadata.insert(
                format!("{}.{suffix}", architecture.as_str()),
                GgufValue::Array(
                    [false, true, true, false]
                        .into_iter()
                        .map(GgufValue::Bool)
                        .collect(),
                ),
            );
        }
        tensors.extend(
            [
                "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight",
                "blk.0.attn_sinks.weight",
                "blk.1.ffn_gate_shexp.weight",
                "blk.1.ffn_up_shexp.weight",
                "blk.1.ffn_down_shexp.weight",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }

    if matches!(loader, NormalLoaderType::HunYuanMoEV1) {
        tensors.push("blk.1.ffn_gate_shexp.weight".to_string());
    }

    (metadata, tensors)
}

fn default_fixture(
    loader: &NormalLoaderType,
) -> (
    CanonicalGgufArchitecture,
    HashMap<String, GgufValue>,
    Vec<String>,
) {
    let architecture = registry_adapter(loader).unwrap().architectures[0];
    let (metadata, tensors) = fixture(loader, architecture);
    (architecture, metadata, tensors)
}

fn assert_deserializes<T: DeserializeOwned>(loader: &NormalLoaderType, config: JsonValue) {
    if let Err(error) = serde_json::from_value::<T>(config.clone()) {
        panic!("synthesized `{loader}` config did not deserialize: {error}\n{config:#}");
    }
}

fn assert_native_config_deserializes(loader: &NormalLoaderType, config: JsonValue) {
    match loader {
        NormalLoaderType::Mistral => assert_deserializes::<models::mistral::Config>(loader, config),
        NormalLoaderType::Gemma => assert_deserializes::<models::gemma::Config>(loader, config),
        NormalLoaderType::Mixtral => assert_deserializes::<models::mixtral::Config>(loader, config),
        NormalLoaderType::Llama => assert_deserializes::<models::llama::Config>(loader, config),
        NormalLoaderType::Phi2 => assert_deserializes::<models::phi2::Config>(loader, config),
        NormalLoaderType::Phi3 => assert_deserializes::<models::phi3::Config>(loader, config),
        NormalLoaderType::Qwen2 => assert_deserializes::<models::qwen2::Config>(loader, config),
        NormalLoaderType::Gemma2 => assert_deserializes::<models::gemma2::Config>(loader, config),
        NormalLoaderType::Starcoder2 => {
            assert_deserializes::<models::starcoder2::Config>(loader, config)
        }
        NormalLoaderType::Phi3_5MoE => {
            assert_deserializes::<models::phi3_5_moe::Config>(loader, config)
        }
        NormalLoaderType::DeepSeekV2 => {
            assert_deserializes::<models::deepseek2::DeepSeekV2Config>(loader, config)
        }
        NormalLoaderType::DeepSeekV3 => {
            assert_deserializes::<models::deepseek3::DeepSeekV3Config>(loader, config)
        }
        NormalLoaderType::Qwen3 => assert_deserializes::<models::qwen3::Config>(loader, config),
        NormalLoaderType::GLM4 => assert_deserializes::<models::glm4::Config>(loader, config),
        NormalLoaderType::GLM4MoeLite => {
            assert_deserializes::<models::glm4_moe_lite::Glm4MoeLiteConfig>(loader, config)
        }
        NormalLoaderType::GLM4Moe => {
            assert_deserializes::<models::glm4_moe::Glm4MoeConfig>(loader, config)
        }
        NormalLoaderType::Qwen3Moe => {
            assert_deserializes::<models::qwen3_moe::Config>(loader, config)
        }
        NormalLoaderType::Flint => assert_deserializes::<models::flint::Config>(loader, config),
        NormalLoaderType::SmolLm3 => assert_deserializes::<models::smollm3::Config>(loader, config),
        NormalLoaderType::GraniteMoeHybrid => {
            assert_deserializes::<models::granite::Config>(loader, config)
        }
        NormalLoaderType::GptOss => assert_deserializes::<models::gpt_oss::Config>(loader, config),
        NormalLoaderType::HunYuanDenseV1 => {
            assert_deserializes::<models::hunyuan_v1_dense::Config>(loader, config)
        }
        NormalLoaderType::HunYuanMoEV1 => {
            assert_deserializes::<models::hunyuan_v1_moe::Config>(loader, config)
        }
        NormalLoaderType::Qwen3Next => {
            assert_deserializes::<models::qwen3_next::Config>(loader, config)
        }
        NormalLoaderType::Qwen3_5 => {
            assert_deserializes::<crate::vision_models::qwen3_5::TextConfig>(loader, config)
        }
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
            assert_deserializes::<models::lfm2::Config>(loader, config)
        }
        NormalLoaderType::Lfm3 => assert_deserializes::<models::lfm3::Config>(loader, config),
    }
}

#[test]
fn config_builder_registry_is_exhaustive() {
    assert_eq!(NORMAL_MODEL_ADAPTERS.len(), NORMAL_CONFIG_BUILDERS.len());

    let mut builders = HashSet::new();
    for builder in NORMAL_CONFIG_BUILDERS {
        assert!(builders.insert(builder.loader.to_string()));
        assert!(registry_adapter(&builder.loader).is_some());
    }
    for loader in NormalLoaderType::iter() {
        assert!(builder_for(&loader).is_some(), "{loader}");
        assert!(builders.contains(&loader.to_string()), "{loader}");
    }
}

#[test]
fn every_loader_synthesizes_a_native_config() {
    for loader in NORMAL_CONFIG_BUILDERS
        .iter()
        .map(|builder| builder.loader.clone())
    {
        let (_, metadata, tensors) = default_fixture(&loader);
        let config = synthesize_normal_config_value(&loader, &metadata, &tensors)
            .unwrap_or_else(|error| panic!("failed to synthesize `{loader}` fixture: {error}"));
        assert_eq!(
            config["architectures"],
            json!([loader.causal_lm_name()]),
            "synthesized `{loader}` config is not auto-loadable"
        );
        assert_eq!(config["model_type"], loader.model_type_name());
        assert_eq!(
            normal_loader_hint_from_external_config(&config.to_string()).unwrap(),
            Some(loader.clone())
        );
        assert_native_config_deserializes(&loader, config);
    }
}

#[test]
fn flint_published_metadata_preserves_native_architecture() {
    let loader = NormalLoaderType::Flint;
    let (architecture, mut metadata, mut tensors) = default_fixture(&loader);
    for (key, value) in [
        ("vocab_size", 150_144),
        ("context_length", 262_144),
        ("embedding_length", 2_048),
        ("block_count", 49),
        ("feed_forward_length", 6_144),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("expert_count", 256),
        ("expert_used_count", 6),
        ("expert_feed_forward_length", 512),
        ("expert_shared_feed_forward_length", 1_024),
        ("leading_dense_block_count", 2),
        ("expert_shared_count", 2),
    ] {
        insert_u32(&mut metadata, architecture, key, value);
    }
    insert_f32(
        &mut metadata,
        architecture,
        "attention.layer_norm_rms_epsilon",
        1e-6,
    );
    metadata.remove("flint.rope.dimension_count");
    let full_attention = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48];
    for key in ["attention.sliding_window_pattern", "attention.rope_pattern"] {
        metadata.insert(
            format!("flint.{key}"),
            GgufValue::Array(
                (0..49)
                    .map(|layer| GgufValue::Bool(!full_attention.contains(&layer)))
                    .collect(),
            ),
        );
    }
    for name in &mut tensors {
        *name = name.replace("blk.1.", "blk.2.");
    }
    tensors.extend(
        ["output.weight", "blk.1.ffn_gate.weight"]
            .into_iter()
            .map(str::to_string),
    );
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    let native: models::flint::Config = serde_json::from_value(config.clone()).unwrap();
    native.validate().unwrap();
    assert_eq!(native.vocab_size, 150_144);
    assert_eq!(native.hidden_size, 2_048);
    assert_eq!(native.intermediate_size, 6_144);
    assert_eq!(native.num_hidden_layers, 49);
    assert_eq!(native.num_attention_heads, 32);
    assert_eq!(native.num_key_value_heads, 4);
    assert_eq!(native.head_dim(), 128);
    assert_eq!(native.max_position_embeddings, 262_144);
    assert!((native.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert_eq!(native.rope_theta, 10_000.);
    assert_eq!(native.moe_intermediate_size, 512);
    assert_eq!(native.shared_expert_intermediate_size, 1_024);
    assert_eq!(native.num_experts, 256);
    assert_eq!(native.num_experts_per_tok, 6);
    assert_eq!(native.mlp_only_layers, vec![0, 1]);
    assert_eq!(native.no_rope_layer_indices, full_attention);
    assert!(native.has_attention_sinks());
    assert!(native.norm_topk_prob);
    assert!(!native.tie_word_embeddings);
    assert!(!native.attention_bias);
    assert_eq!(config["router_score_function"], "sigmoid");
    for layer in 0..49 {
        assert_eq!(
            native.sliding_window(layer),
            (!full_attention.contains(&layer)).then_some(513)
        );
        assert_eq!(native.is_moe_layer(layer), layer >= 2);
    }
}

#[test]
fn flint_schedules_and_routing_are_independent() {
    let loader = NormalLoaderType::Flint;
    let (architecture, mut metadata, mut tensors) = default_fixture(&loader);
    metadata.insert(
        "flint.attention.rope_pattern".to_string(),
        GgufValue::Array(
            [true, false, true, false]
                .into_iter()
                .map(GgufValue::Bool)
                .collect(),
        ),
    );
    insert_u32(&mut metadata, architecture, "expert_gating_func", 1);
    insert_bool(&mut metadata, architecture, "expert_weights_norm", false);
    tensors.push("blk.2.ffn_gate.weight".to_string());
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(
        config["layer_types"],
        json!([
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ])
    );
    assert_eq!(config["no_rope_layer_indices"], json!([1, 3]));
    assert_eq!(config["mlp_only_layers"], json!([0, 2]));
    assert_eq!(config["router_score_function"], "softmax");
    assert_eq!(config["norm_topk_prob"], false);
}

#[test]
fn flint_rejects_missing_or_incompatible_metadata() {
    let loader = NormalLoaderType::Flint;
    let (_, metadata, tensors) = default_fixture(&loader);
    for key in [
        "flint.attention.sliding_window_pattern",
        "flint.attention.rope_pattern",
    ] {
        let mut incomplete = metadata.clone();
        incomplete.remove(key);
        let error = synthesize_normal_config_value(&loader, &incomplete, &tensors).unwrap_err();
        assert!(error.to_string().contains(key));
        assert!(error.to_string().contains("original Hugging Face config"));
    }
    for (key, value) in [
        (
            "flint.attention.rope_pattern",
            GgufValue::Array(vec![GgufValue::Bool(true)]),
        ),
        (
            "flint.attention.rope_pattern",
            GgufValue::Array(vec![GgufValue::U32(1); LAYER_COUNT]),
        ),
        ("flint.attention.sliding_window", GgufValue::U32(0)),
        ("flint.expert_weights_scale", GgufValue::F32(1.25)),
        ("flint.attention.value_length", GgufValue::U32(32)),
        ("flint.rope.dimension_count", GgufValue::U32(32)),
    ] {
        let mut incompatible = metadata.clone();
        incompatible.insert(key.to_string(), value);
        let error = synthesize_normal_config_value(&loader, &incompatible, &tensors).unwrap_err();
        assert!(error.to_string().contains(key), "{error}");
    }
}

#[test]
fn qwen35moe_standalone_metadata_synthesizes_tiled_native_config() {
    let loader = NormalLoaderType::Qwen3Next;
    let architecture = CanonicalGgufArchitecture::Qwen35Moe;
    let (mut metadata, tensors) = fixture(&loader, architecture);
    metadata.remove(&format!("{}.feed_forward_length", architecture.as_str()));
    insert_u32(&mut metadata, architecture, "attention.key_length", 256);
    insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
    insert_u32(&mut metadata, architecture, "ssm.state_size", 128);
    insert_u32(&mut metadata, architecture, "ssm.group_count", 16);
    insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 32);
    insert_u32(&mut metadata, architecture, "ssm.inner_size", 4_096);

    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(config["intermediate_size"], 512);
    assert_eq!(config["head_dim"], 256);
    assert_eq!(config["partial_rotary_factor"], 0.25);
    assert_eq!(config[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
    assert_eq!(config["architectures"][0], "Qwen3NextForCausalLM");
    let native: models::qwen3_next::Config = serde_json::from_value(config.clone()).unwrap();
    assert_eq!(
        crate::gdn::GdnConfig::v_head_layout(&native),
        crate::gdn::GdnVHeadLayout::Tiled
    );
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn qwen35_standalone_metadata_synthesizes_dense_text_config() {
    let loader = NormalLoaderType::Qwen3_5;
    let architecture = CanonicalGgufArchitecture::Qwen35;
    let (mut metadata, tensors) = fixture(&loader, architecture);
    insert_u32(&mut metadata, architecture, "vocab_size", 248_320);
    insert_u32(&mut metadata, architecture, "context_length", 262_144);
    insert_u32(&mut metadata, architecture, "embedding_length", 2_560);
    insert_u32(&mut metadata, architecture, "feed_forward_length", 9_216);
    insert_u32(&mut metadata, architecture, "block_count", 32);
    insert_u32(&mut metadata, architecture, "attention.head_count", 16);
    insert_u32(&mut metadata, architecture, "attention.head_count_kv", 4);
    insert_u32(&mut metadata, architecture, "attention.key_length", 256);
    insert_u32(&mut metadata, architecture, "attention.value_length", 256);
    insert_f32(
        &mut metadata,
        architecture,
        "attention.layer_norm_rms_epsilon",
        1e-6,
    );
    insert_u32(&mut metadata, architecture, "rope.dimension_count", 64);
    insert_f32(&mut metadata, architecture, "rope.freq_base", 10_000_000.0);
    insert_u32(&mut metadata, architecture, "ssm.state_size", 128);
    insert_u32(&mut metadata, architecture, "ssm.group_count", 16);
    insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 32);
    insert_u32(&mut metadata, architecture, "ssm.inner_size", 4_096);

    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(config["vocab_size"], 248_320);
    assert_eq!(config["hidden_size"], 2_560);
    assert_eq!(config["intermediate_size"], 9_216);
    assert_eq!(config["num_hidden_layers"], 32);
    assert_eq!(config["head_dim"], 256);
    assert_eq!(config["rope_parameters"]["partial_rotary_factor"], 0.25);
    assert_eq!(
        config["rope_parameters"]["mrope_section"],
        json!([11, 11, 10])
    );
    assert_eq!(config[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
    let native: crate::vision_models::qwen3_5::TextConfig =
        serde_json::from_value(config.clone()).unwrap();
    assert_eq!(
        crate::gdn::GdnConfig::v_head_layout(&native),
        crate::gdn::GdnVHeadLayout::Tiled
    );
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn qwen35_rejects_invalid_hybrid_and_mrope_metadata() {
    let loader = NormalLoaderType::Qwen3_5;
    let architecture = CanonicalGgufArchitecture::Qwen35;
    let (mut metadata, tensors) = fixture(&loader, architecture);
    insert_u32(&mut metadata, architecture, "full_attention_interval", 0);
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("full attention interval"));

    insert_u32(&mut metadata, architecture, "full_attention_interval", 4);
    insert_u32_array(
        &mut metadata,
        architecture,
        "rope.dimension_sections",
        &[11, 0, 21, 0],
    );
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("three non-zero MRoPE sections"));

    insert_u32_array(
        &mut metadata,
        architecture,
        "rope.dimension_sections",
        &[11, 11, 10, 0],
    );
    insert_u32(&mut metadata, architecture, "ssm.group_count", 3);
    insert_u32(&mut metadata, architecture, "ssm.time_step_rank", 8);
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("incompatible GDN head counts"));
}

#[test]
fn standalone_moe_normalization_defaults_match_native_models() {
    for (loader, architecture) in [
        (
            NormalLoaderType::Qwen3Moe,
            CanonicalGgufArchitecture::Qwen3Moe,
        ),
        (
            NormalLoaderType::Qwen3Next,
            CanonicalGgufArchitecture::Qwen3Next,
        ),
        (
            NormalLoaderType::Qwen3Next,
            CanonicalGgufArchitecture::Qwen35Moe,
        ),
        (
            NormalLoaderType::GLM4Moe,
            CanonicalGgufArchitecture::Glm4Moe,
        ),
    ] {
        let (mut metadata, tensors) = fixture(&loader, architecture);
        metadata.remove(&format!("{}.expert_weights_norm", architecture.as_str()));

        let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
        assert_eq!(
            config["norm_topk_prob"], true,
            "{loader} with {architecture}"
        );
        assert_native_config_deserializes(&loader, config);
    }
}

#[test]
fn llama_linear_rope_and_baked_factor_gate_are_explicit() {
    let loader = NormalLoaderType::Llama;
    let (architecture, mut metadata, mut tensors) = default_fixture(&loader);
    insert_string(&mut metadata, architecture, "rope.scaling.type", "linear");
    insert_f32(&mut metadata, architecture, "rope.scaling.factor", 2.0);
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(config["rope_scaling"]["rope_type"], "linear");
    assert_eq!(config["rope_scaling"]["factor"], 2.0);
    assert_native_config_deserializes(&loader, config);

    tensors.push("rope_freqs.weight".to_string());
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert!(config["rope_scaling"].is_null());
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn qwen_sliding_policies_preserve_layer_semantics() {
    let loader = NormalLoaderType::Qwen2;
    let (architecture, mut metadata, tensors) = default_fixture(&loader);
    insert_u32(
        &mut metadata,
        architecture,
        "attention.sliding_window",
        1_024,
    );
    metadata.insert(
        format!("{}.attention.sliding_window_pattern", architecture.as_str()),
        GgufValue::Array(
            [true, false, true, false]
                .into_iter()
                .map(GgufValue::Bool)
                .collect(),
        ),
    );
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(
        config["layer_types"],
        json!([
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ])
    );
    assert_native_config_deserializes(&loader, config);

    let qwen3 = NormalLoaderType::Qwen3;
    let (architecture, mut metadata, tensors) = default_fixture(&qwen3);
    insert_u32(
        &mut metadata,
        architecture,
        "attention.sliding_window",
        1_024,
    );
    let error = synthesize_normal_config_value(&qwen3, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("layer policy"));
}

#[test]
fn qwen_moe_inventory_marks_dense_layers() {
    let loader = NormalLoaderType::Qwen3Moe;
    let (_, metadata, tensors) = default_fixture(&loader);
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(config["mlp_only_layers"], json!([0]));
    assert_eq!(config["num_experts"], 8);
    assert_eq!(config["num_experts_per_tok"], 2);
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn hybrid_ssm_and_shortconv_configs_preserve_layer_inventory() {
    let loader = NormalLoaderType::GraniteMoeHybrid;
    let architecture = CanonicalGgufArchitecture::GraniteHybrid;
    let (mut metadata, mut tensors) = fixture(&loader, architecture);
    insert_u32_array(
        &mut metadata,
        architecture,
        "attention.head_count_kv",
        &[4, 0, 4, 0],
    );
    insert_u32(&mut metadata, architecture, "expert_count", 8);
    insert_u32(&mut metadata, architecture, "expert_used_count", 2);
    tensors.extend(
        ["blk.1.ssm_a.weight", "blk.1.ssm_conv1d.weight"]
            .into_iter()
            .map(str::to_string),
    );
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(
        config["layer_types"],
        json!(["attention", "mamba", "attention", "mamba"])
    );
    assert_eq!(config["mamba_d_head"], 128);
    assert_eq!(config["mamba_expand"], 2);
    assert_native_config_deserializes(&loader, config);

    let loader = NormalLoaderType::Lfm2;
    let (_, metadata, tensors) = default_fixture(&loader);
    assert!(!tensors.iter().any(|tensor| tensor == "output_norm.weight"));
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(
        config["layer_types"],
        json!(["full_attention", "conv", "full_attention", "conv"])
    );
    assert_eq!(config["conv_L_cache"], 3);
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn lfm3_synthesizes_hybrid_layers_and_rope() {
    let loader = NormalLoaderType::Lfm3;
    let (_, metadata, tensors) = default_fixture(&loader);
    let config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    assert_eq!(
        config["layer_types"],
        json!(["conv", "full_attention", "sliding_attention", "conv"])
    );
    assert_eq!(config["num_key_value_heads"], 4);
    assert_eq!(config["head_dim"], 64);
    assert_eq!(config["conv_L_cache"], 7);
    assert_eq!(config["sliding_window"], 256);
    assert_eq!(config["use_sandwich_norm"], true);
    assert_eq!(config["tie_word_embeddings"], true);
    assert_eq!(
        config["rope_parameters"]["full_attention"]["rope_theta"],
        1_000_000.0
    );
    assert_eq!(
        config["rope_parameters"]["sliding_attention"]["rope_theta"],
        10_000.0
    );
    assert_eq!(
        config["rope_parameters"]["full_attention"]["partial_rotary_factor"],
        1.0
    );
    assert_native_config_deserializes(&loader, config);
}

#[test]
fn lfm3_rejects_inconsistent_layer_metadata() {
    let loader = NormalLoaderType::Lfm3;
    let (architecture, mut metadata, mut tensors) = default_fixture(&loader);
    insert_bool(&mut metadata, architecture, "attention.causal", false);
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("noncausal"));
    insert_bool(&mut metadata, architecture, "attention.causal", true);

    tensors.retain(|tensor| tensor != "blk.2.post_ffw_norm.weight");
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("sandwich norms"));
    tensors.push("blk.2.post_ffw_norm.weight".to_string());

    tensors.push("blk.0.shortconv.conv.bias".to_string());
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("convolution bias"));
    tensors.push("blk.3.shortconv.conv.bias".to_string());

    insert_u32(&mut metadata, architecture, "attention.value_length", 32);
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("key and value dimensions"));
    insert_u32(&mut metadata, architecture, "attention.value_length", 64);

    metadata.insert(
        "lfm3.attention.sliding_window_pattern".to_string(),
        GgufValue::Array(vec![GgufValue::Bool(true); LAYER_COUNT]),
    );
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("convolution layer 0"));
}

#[test]
fn lfm3_vl_external_config_unwraps_native_text_config() {
    let loader = NormalLoaderType::Lfm3;
    let (architecture, metadata, tensors) = default_fixture(&loader);
    let mut text_config = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap();
    text_config["rope_parameters"]["full_attention"]["partial_rotary_factor"] = json!(0.5);
    text_config["quantization_config"] = json!({"quant_method": "awq"});
    let raw = json!({
        "architectures": ["Lfm3VlForConditionalGeneration"],
        "model_type": "lfm3_vl",
        "text_config": text_config,
        "vision_config": {"sentinel": true},
    })
    .to_string();
    assert_eq!(
        normal_loader_hint_from_external_config(&raw).unwrap(),
        Some(loader.clone())
    );
    let normalized = normalize_external_normal_config(&loader, architecture, &raw).unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    text_config["quantization_config"] = JsonValue::Null;
    assert_eq!(value, text_config);
    assert_native_config_deserializes(&loader, value);
}

#[test]
fn missing_ssm_and_shortconv_semantics_are_actionable() {
    let loader = NormalLoaderType::Qwen3Next;
    let (architecture, mut metadata, tensors) = default_fixture(&loader);
    metadata.remove(&format!("{}.ssm.time_step_rank", architecture.as_str()));
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("ssm.time_step_rank"));
    assert!(error.to_string().contains("original Hugging Face config"));

    let loader = NormalLoaderType::Lfm2;
    let (architecture, mut metadata, tensors) = default_fixture(&loader);
    metadata.remove(&format!("{}.shortconv.l_cache", architecture.as_str()));
    let error = synthesize_normal_config_value(&loader, &metadata, &tensors).unwrap_err();
    assert!(error.to_string().contains("shortconv.l_cache"));
}

#[test]
fn external_flat_configs_are_preserved_for_every_loader() {
    let mut architectures = HashSet::new();
    for (index, loader) in NormalLoaderType::iter().enumerate() {
        let model_architecture = loader.causal_lm_name();
        assert!(architectures.insert(model_architecture));
        let raw_value = json!({
            "architectures": [model_architecture],
            "sentinel": index,
        });
        let raw = raw_value.to_string();
        assert_eq!(
            normal_loader_hint_from_external_config(&raw).unwrap(),
            Some(loader.clone())
        );
        let adapter = registry_adapter(&loader).unwrap();
        let gguf_architecture = adapter.architectures[0];
        let normalized: JsonValue = serde_json::from_str(
            &normalize_external_normal_config(&loader, gguf_architecture, &raw).unwrap(),
        )
        .unwrap();
        let mut expected = raw_value;
        expected["quantization_config"] = JsonValue::Null;
        expected["model_type"] = json!(loader.model_type_name());
        if matches!(
            gguf_architecture,
            CanonicalGgufArchitecture::Qwen35 | CanonicalGgufArchitecture::Qwen35Moe
        ) {
            expected[GDN_V_HEAD_LAYOUT_CONFIG_KEY] = json!("tiled");
        }
        assert_eq!(
            normalized, expected,
            "external config fields changed for native `{loader}`"
        );
    }
}

#[test]
fn mistral3_external_config_unwraps_native_text_config() {
    let raw = json!({
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": {
            "vocab_size": 32000,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "sliding_window": null,
            "head_dim": 64,
            "quantization_config": null
        },
        "vision_config": {
            "sentinel": true
        }
    })
    .to_string();
    assert_eq!(
        normal_loader_hint_from_external_config(&raw).unwrap(),
        Some(NormalLoaderType::Mistral)
    );
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Mistral,
        CanonicalGgufArchitecture::Mistral3,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert!(value.get("vision_config").is_none());
    assert_eq!(value["hidden_size"], 512);
    assert_deserializes::<models::mistral::Config>(&NormalLoaderType::Mistral, value);
}

#[test]
fn mistral3_external_config_uses_gguf_quantization() {
    let raw = json!({
        "architectures": ["Mistral3ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 512,
            "quantization_config": null
        },
        "quantization_config": {
            "quant_method": "sentinel"
        }
    })
    .to_string();
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Mistral,
        CanonicalGgufArchitecture::Mistral3,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert!(value["quantization_config"].is_null());
}

#[test]
fn ordinary_external_config_uses_gguf_quantization() {
    let raw = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "quantization_config": {
            "quant_method": "awq"
        }
    })
    .to_string();
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Qwen3,
        CanonicalGgufArchitecture::Qwen3,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert!(value["quantization_config"].is_null());
}

#[test]
fn lfm2_vl_external_config_unwraps_native_text_config() {
    let raw = json!({
        "architectures": ["Lfm2VlForConditionalGeneration"],
        "model_type": "lfm2_vl",
        "text_config": {
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2",
            "vocab_size": 32000,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 4096,
            "norm_eps": 0.00001,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "rope_type": "default"
            },
            "conv_L_cache": 3,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"],
            "quantization_config": null
        },
        "vision_config": {
            "sentinel": true
        }
    })
    .to_string();
    assert_eq!(
        normal_loader_hint_from_external_config(&raw).unwrap(),
        Some(NormalLoaderType::Lfm2)
    );
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Lfm2,
        CanonicalGgufArchitecture::Lfm2,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert!(value.get("vision_config").is_none());
    assert_native_config_deserializes(&NormalLoaderType::Lfm2, value);
}

#[test]
fn qwen35_moe_external_config_maps_to_qwen3_next() {
    let raw = json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "rms_norm_eps": 0.00001,
            "head_dim": 64,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 512,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "full_attention_interval": 4,
            "quantization_config": null
        },
        "vision_config": {
            "sentinel": true
        }
    })
    .to_string();
    assert_eq!(
        normal_loader_hint_from_external_config(&raw).unwrap(),
        Some(NormalLoaderType::Qwen3Next)
    );
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Qwen3Next,
        CanonicalGgufArchitecture::Qwen35Moe,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert_eq!(value["rope_theta"], 10000000.0);
    assert_eq!(value["partial_rotary_factor"], 0.25);
    assert_eq!(value["intermediate_size"], 512);
    assert_eq!(value[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
    assert_eq!(value["architectures"][0], "Qwen3NextForCausalLM");
    assert_native_config_deserializes(&NormalLoaderType::Qwen3Next, value);
}

#[test]
fn qwen35_external_config_unwraps_dense_text_config() {
    let raw = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 32000,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "rms_norm_eps": 0.00001,
            "head_dim": 64,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "mrope_section": [3, 3, 2],
                "partial_rotary_factor": 0.25
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "full_attention_interval": 4,
            "tie_word_embeddings": true
        },
        "vision_config": {
            "sentinel": true
        }
    })
    .to_string();
    assert_eq!(
        normal_loader_hint_from_external_config(&raw).unwrap(),
        Some(NormalLoaderType::Qwen3_5)
    );
    let normalized = normalize_external_normal_config(
        &NormalLoaderType::Qwen3_5,
        CanonicalGgufArchitecture::Qwen35,
        &raw,
    )
    .unwrap();
    let value: JsonValue = serde_json::from_str(&normalized).unwrap();
    assert!(value.get("vision_config").is_none());
    assert_eq!(value["tie_word_embeddings"], false);
    assert_eq!(value[GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
    assert_native_config_deserializes(&NormalLoaderType::Qwen3_5, value);
}

#[test]
fn external_config_tying_matches_the_gguf_output_tensor() {
    let tied = r#"{"tie_word_embeddings":true}"#;
    let untied = r#"{"tie_word_embeddings":false}"#;
    let no_output = vec!["token_embd.weight".to_string()];
    let with_output = vec!["token_embd.weight".to_string(), "output.weight".to_string()];

    validate_normal_config_tensor_inventory(tied, &no_output).unwrap();
    validate_normal_config_tensor_inventory(untied, &with_output).unwrap();
    assert!(validate_normal_config_tensor_inventory(tied, &with_output)
        .unwrap_err()
        .to_string()
        .contains("tie_word_embeddings"));
    assert!(validate_normal_config_tensor_inventory(untied, &no_output)
        .unwrap_err()
        .to_string()
        .contains("tie_word_embeddings"));
    validate_normal_config_tensor_inventory("{}", &with_output).unwrap();
}

#[test]
fn external_config_architecture_errors_are_actionable() {
    let mismatched = r#"{"architectures":["Qwen3ForCausalLM"]}"#;
    let error = normalize_external_normal_config(
        &NormalLoaderType::Llama,
        CanonicalGgufArchitecture::Llama,
        mismatched,
    )
    .unwrap_err();
    assert!(error.to_string().contains("qwen3"));
    assert!(error.to_string().contains("llama"));

    let unknown = r#"{"architectures":["FutureForCausalLM"]}"#;
    let error = normal_loader_hint_from_external_config(unknown).unwrap_err();
    assert!(error.to_string().contains("FutureForCausalLM"));
    assert!(error.to_string().contains("supported native text model"));

    let missing_text = r#"{
            "architectures": ["Mistral3ForConditionalGeneration"],
            "model_type": "mistral3"
        }"#;
    let error = normalize_external_normal_config(
        &NormalLoaderType::Mistral,
        CanonicalGgufArchitecture::Mistral3,
        missing_text,
    )
    .unwrap_err();
    assert!(error.to_string().contains("text_config"));
}
