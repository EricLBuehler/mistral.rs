use std::collections::HashMap;

use super::super::AutoDeviceMapQuantization;
use super::*;
use crate::{
    device_map::DummyDeviceMapper,
    matformer::{MatformerConfig, MatformerSliceConfig, Slice},
};
use mistralrs_quant::IsqType;

fn qwen_activation_config(architecture: &str) -> serde_json::Value {
    serde_json::json!({
        "architectures": [architecture],
        "text_config": {
            "head_dim": 8,
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "hidden_act": "silu",
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {"mrope_section": [1, 1, 2]},
            "rope_parameters": {"mrope_section": [1, 1, 2]},
            "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 16,
            "num_experts": 8,
            "num_experts_per_tok": 3,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4
        },
        "vision_config": {"out_hidden_size": 16, "deepstack_visual_indexes": [1, 2]},
        "image_token_id": 1,
        "video_token_id": 2,
        "vision_start_token_id": 3,
        "vision_end_token_id": 4,
        "tie_word_embeddings": false
    })
}

#[test]
fn qwen4_exp_topology_uses_the_checkpoint_namespace() -> Result<()> {
    use super::super::ResourceSource;
    use mistralrs_quant::{estimate_isq_weight_bytes, ShardedSafeTensors};

    const WIDTH: usize = 256;
    let mut config = qwen_activation_config("Qwen4ExpForConditionalGeneration");
    config["vision_config"] = serde_json::Value::Null;
    config["text_config"]["hidden_size"] = WIDTH.into();
    config["text_config"]["vocab_size"] = WIDTH.into();
    let config = config.to_string();
    let estimate = |quantization: &AutoDeviceMapQuantization<'_>| {
        Qwen4ExpLoader.non_mapped_size_in_bytes(&config, DType::BF16, 1, Some(quantization), None)
    };
    let base = estimate(&AutoDeviceMapQuantization::isq(Some(IsqType::Q4K), None))?;
    let extra = estimate_isq_weight_bytes(&[WIDTH, WIDTH], DType::BF16, IsqType::Q8_0)
        - estimate_isq_weight_bytes(&[WIDTH, WIDTH], DType::BF16, IsqType::Q6K);
    for root in ["model.language_model", "language_model.model"] {
        let weights = ShardedSafeTensors::wrap(
            HashMap::from([(
                format!("{root}.embed_tokens.weight"),
                Tensor::zeros((1, 1), DType::BF16, &Device::Cpu)?,
            )]),
            DType::BF16,
            Device::Cpu,
        );
        let checkpoint = ResourceSource {
            weights: Some(&weights),
            paths: &[],
        }
        .checkpoint_tensors()?;
        for pattern_root in ["model.language_model", "language_model.model"] {
            let topology = crate::Topology::from_str(&format!(
                "'/^{}\\.embed_tokens\\.weight$/':\n  isq: Q8_0\n",
                regex::escape(pattern_root),
            ))?;
            let quantization = AutoDeviceMapQuantization::isq(Some(IsqType::Q4K), Some(&topology))
                .with_checkpoint(&checkpoint);
            assert_eq!(
                estimate(&quantization)?,
                base + usize::from(root == pattern_root) * extra
            );
        }
    }
    Ok(())
}

#[test]
fn qwen_vision_memory_matches_auto_loader_and_honors_image_limits() -> Result<()> {
    for (architecture, loader) in [
        (
            "Qwen3VLForConditionalGeneration",
            &Qwen3VLLoader as &dyn DeviceMappedModelLoader,
        ),
        ("Qwen3VLMoeForConditionalGeneration", &Qwen3VLMoELoader),
        ("Qwen3_5ForConditionalGeneration", &Qwen3_5Loader),
        ("Qwen3_5MoeForConditionalGeneration", &Qwen3_5MoeLoader),
    ] {
        let config = qwen_activation_config(architecture).to_string();
        let params = AutoDeviceMapParams::Multimodal {
            max_seq_len: 4096,
            max_batch_size: 16,
            max_image_shape: (512, 512),
            max_num_images: 1,
        };
        let bytes = loader.non_mapped_activation_memory_bytes(
            &config,
            &params,
            DType::BF16,
            &Device::Cpu,
        )?;
        assert!(bytes > 0);
        assert_eq!(
            bytes,
            AutoMultimodalLoader.non_mapped_activation_memory_bytes(
                &config,
                &params,
                DType::BF16,
                &Device::Cpu
            )?
        );
        let no_images = AutoDeviceMapParams::Multimodal {
            max_seq_len: 4096,
            max_batch_size: 16,
            max_image_shape: (512, 512),
            max_num_images: 0,
        };
        assert_eq!(
            AutoMultimodalLoader.non_mapped_activation_memory_bytes(
                &config,
                &no_images,
                DType::BF16,
                &Device::Cpu
            )?,
            0
        );
    }
    Ok(())
}

#[test]
fn qwen_dense_activation_metadata_preserves_widths_and_deepstack() -> Result<()> {
    let config = qwen_activation_config("Qwen3VLForConditionalGeneration").to_string();
    for loader in [
        &Qwen3VLLoader as &dyn DeviceMappedModelLoader,
        &AutoMultimodalLoader,
    ] {
        let metadata = loader.activation_model_config(&config)?;
        assert_eq!(metadata.intermediate_size, 32);
        assert_eq!(metadata.vocab_size, 128);
        assert_eq!(metadata.extra_hidden_elements_per_token, 32);
        assert!(!metadata.custom_prefill_mask);
        assert_eq!(
            metadata.attention_features,
            Some(PrefixPrefillAttentionFeatures::default())
        );
    }
    Ok(())
}

#[test]
fn qwen_moe_activation_metadata_counts_active_shared_and_recurrent_work() -> Result<()> {
    let vl_config = qwen_activation_config("Qwen3VLMoeForConditionalGeneration").to_string();
    let vl = Qwen3VLMoELoader.activation_model_config(&vl_config)?;
    assert_eq!(vl.intermediate_size, 72);
    assert_eq!(vl.extra_hidden_elements_per_token, 32);
    let hybrid_config = qwen_activation_config("Qwen3_5MoeForConditionalGeneration").to_string();
    let hybrid = Qwen3_5MoeLoader.activation_model_config(&hybrid_config)?;
    assert_eq!(hybrid.intermediate_size, 88);
    assert_eq!(hybrid.extra_hidden_elements_per_token, 120);
    assert_eq!(hybrid.vocab_size, 128);
    Ok(())
}

#[test]
fn qwen4_exp_activation_metadata_counts_hybrid_residual_and_indexer_work() -> Result<()> {
    let mut config = qwen_activation_config("Qwen4ExpForConditionalGeneration");
    config["text_config"]["hc_count"] = 4.into();
    config["text_config"]["hc_lowrank"] = 8.into();
    config["text_config"]["indexer_n_heads"] = 2.into();
    config["text_config"]["indexer_kv_heads"] = 1.into();
    config["text_config"]["indexer_head_dim"] = 8.into();
    config["text_config"]["indexer_budget"] = 16.into();
    config["text_config"]["indexer_compress_ratio"] = 4.into();
    for loader in [
        &Qwen4ExpLoader as &dyn DeviceMappedModelLoader,
        &AutoMultimodalLoader,
    ] {
        let metadata = loader.activation_model_config(&config.to_string())?;
        assert_eq!(metadata.intermediate_size, 88);
        assert_eq!(metadata.vocab_size, 128);
        assert_eq!(metadata.extra_hidden_elements_per_token, 432);
        assert!(!metadata.custom_prefill_mask);
    }
    config["vision_config"] = serde_json::Value::Null;
    let metadata = Qwen4ExpLoader.activation_model_config(&config.to_string())?;
    assert_eq!(metadata.extra_hidden_elements_per_token, 400);
    assert_eq!(
        AutoMultimodalLoader.supports_paged_attention(&config.to_string()),
        cfg!(cuda_unix)
    );
    Ok(())
}

#[test]
fn auto_loader_preserves_checkpoint_residency_policy() -> Result<()> {
    for (architecture, loader, expected) in [
        (
            "Qwen3VLForConditionalGeneration",
            &Qwen3VLLoader as &dyn DeviceMappedModelLoader,
            true,
        ),
        (
            "Qwen4ExpForConditionalGeneration",
            &Qwen4ExpLoader as &dyn DeviceMappedModelLoader,
            false,
        ),
    ] {
        let config = qwen_activation_config(architecture).to_string();
        assert_eq!(
            loader.supports_checkpoint_weight_inventory(&config)?,
            expected
        );
        assert_eq!(
            AutoMultimodalLoader.supports_checkpoint_weight_inventory(&config)?,
            expected
        );
    }
    Ok(())
}

#[test]
fn auto_loader_preserves_resource_reservations_and_errors() -> Result<()> {
    let mut config = qwen_activation_config("Qwen4ExpForConditionalGeneration");
    config["text_config"]["ple_layer_ids"] = serde_json::json!([1]);
    config["text_config"]["ple_embed_dim"] = 32.into();
    config["text_config"]["ngram_size"] = 2.into();
    config["text_config"]["heads_per_ngram"] = 1.into();
    config["text_config"]["ngram_vocab_size_base"] = 17.into();
    config["text_config"]["eos_token_id"] = 2.into();
    let expected = Qwen4ExpLoader
        .layer_resources(&config.to_string(), super::super::ResourceSource::default())?;
    #[cfg(feature = "cuda")]
    assert!(
        matches!(expected.as_slice(), [resource] if resource.layer == 0 && resource.device_bytes > 0 && resource.host_backing.is_none())
    );
    #[cfg(not(feature = "cuda"))]
    assert!(expected.is_empty());
    assert_eq!(
        AutoMultimodalLoader
            .layer_resources(&config.to_string(), super::super::ResourceSource::default())?,
        expected
    );
    config["text_config"]["hidden_size"] = serde_json::Value::Null;
    assert!(Qwen4ExpLoader
        .layer_resources(&config.to_string(), super::super::ResourceSource::default())
        .is_err());
    assert!(AutoMultimodalLoader
        .layer_resources(&config.to_string(), super::super::ResourceSource::default())
        .is_err());
    Ok(())
}

fn qwen2_vl_activation_config() -> serde_json::Value {
    let mut config =
        qwen_activation_config("Qwen2VLForConditionalGeneration")["text_config"].clone();
    config["vision_config"] = serde_json::json!({
        "depth": 2,
        "embed_dim": 16,
        "hidden_size": 16,
        "mlp_ratio": 2.0,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    });
    config["image_token_id"] = 1.into();
    config["video_token_id"] = 2.into();
    config["tie_word_embeddings"] = false.into();
    config["use_sliding_window"] = true.into();
    config["sliding_window"] = 128.into();
    config
}

#[test]
fn qwen2_5_vl_still_image_activation_estimate_scales_with_batch() -> Result<()> {
    let config = qwen2_vl_activation_config().to_string();
    let params = |max_batch_size| AutoDeviceMapParams::Multimodal {
        max_seq_len: 16,
        max_batch_size,
        max_image_shape: (8, 8),
        max_num_images: 1,
    };
    let single_encoder = Qwen2_5VLLoader.non_mapped_max_act_size_elems(&config, &params(1))?;
    let single_mapped = Qwen2_5VLLoader.mapped_max_act_size_elems(&config, &params(1))?;
    assert_eq!(single_encoder, 2 * 16 * 16);
    assert_eq!(single_mapped, 2 * 32 * 32);
    assert_eq!(
        Qwen2_5VLLoader.non_mapped_max_act_size_elems(&config, &params(4))?,
        single_encoder * 4
    );
    assert_eq!(
        Qwen2_5VLLoader.mapped_max_act_size_elems(&config, &params(4))?,
        single_mapped * 4
    );
    Ok(())
}

#[test]
fn qwen2_vl_attention_fallback_requires_uniform_layers() -> Result<()> {
    let mut config = qwen2_vl_activation_config();
    for loader in [
        &Qwen2VLLoader as &dyn DeviceMappedModelLoader,
        &Qwen2_5VLLoader,
    ] {
        config.as_object_mut().unwrap().remove("layer_types");
        config["max_window_layers"] = 2.into();
        assert_eq!(
            loader
                .activation_model_config(&config.to_string())?
                .attention_features,
            None
        );
        config["max_window_layers"] = 0.into();
        assert!(
            loader
                .activation_model_config(&config.to_string())?
                .attention_features
                .unwrap()
                .has_sliding_window
        );
        config["layer_types"] = serde_json::json!([
            "full_attention",
            "full_attention",
            "full_attention",
            "full_attention"
        ]);
        assert_eq!(
            loader
                .activation_model_config(&config.to_string())?
                .attention_features,
            Some(PrefixPrefillAttentionFeatures::default())
        );
        config["layer_types"] = serde_json::json!([
            "full_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention"
        ]);
        assert_eq!(
            loader
                .activation_model_config(&config.to_string())?
                .attention_features,
            None
        );
    }
    Ok(())
}

#[test]
fn active_expert_activation_metadata_rejects_overflow() {
    let mut config = qwen_activation_config("Qwen3_5MoeForConditionalGeneration");
    config["text_config"]["moe_intermediate_size"] = usize::MAX.into();
    let error = Qwen3_5MoeLoader
        .activation_model_config(&config.to_string())
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("active expert activation size overflow"));
}

#[test]
fn muse_glimmer_activation_metadata_uses_typed_text_config() -> Result<()> {
    let mut config: serde_json::Value = serde_json::from_str(&muse_glimmer_test_config())?;
    config["text_config"]["intermediate_size"] = 48.into();
    config["text_config"]["vocab_size"] = 96.into();
    let metadata = MuseGlimmerLoader.activation_model_config(&config.to_string())?;
    assert_eq!(metadata.intermediate_size, 48);
    assert_eq!(metadata.vocab_size, 96);
    assert!(metadata.custom_prefill_mask);
    Ok(())
}

fn matches_any(regexes: &[Regex], name: &str) -> bool {
    regexes.iter().any(|regex| regex.is_match(name))
}

#[test]
fn qwen3_vl_family_reports_video_input() -> Result<()> {
    let expected_input = vec![
        SupportedModality::Text,
        SupportedModality::Vision,
        SupportedModality::Video,
    ];
    for (name, modalities) in [
        ("Qwen3VL", Qwen3VLLoader.modalities("")?),
        ("Qwen3VLMoE", Qwen3VLMoELoader.modalities("")?),
        ("Qwen3.5", Qwen3_5Loader.modalities("")?),
        ("Qwen3.5 MoE", Qwen3_5MoeLoader.modalities("")?),
    ] {
        assert_eq!(modalities.input, expected_input, "{name}");
        assert_eq!(modalities.output, vec![SupportedModality::Text], "{name}");
    }
    Ok(())
}

#[test]
fn auto_loader_reports_encoder_cache_capability() {
    for (architecture, expected) in [
        ("Phi3VForCausalLM", true),
        ("Idefics2ForConditionalGeneration", true),
        ("LlavaNextForConditionalGeneration", true),
        ("LlavaForConditionalGeneration", true),
        ("Lfm2VlForConditionalGeneration", false),
        ("MllamaForConditionalGeneration", true),
        ("Qwen2VLForConditionalGeneration", true),
        ("Idefics3ForConditionalGeneration", true),
        ("MiniCPMO", true),
        ("Phi4MMForCausalLM", true),
        ("Qwen2_5_VLForConditionalGeneration", true),
        ("Gemma3ForConditionalGeneration", false),
        ("Mistral3ForConditionalGeneration", true),
        ("Llama4ForConditionalGeneration", true),
        ("Gemma3nForConditionalGeneration", true),
        ("Qwen3VLForConditionalGeneration", true),
        ("Qwen3VLMoeForConditionalGeneration", true),
        ("Qwen3_5ForConditionalGeneration", true),
        ("Qwen3_5MoeForConditionalGeneration", true),
        ("VoxtralRealtimeForConditionalGeneration", false),
        ("Gemma4ForConditionalGeneration", true),
        ("MuseGlimmerForConditionalGeneration", true),
        ("DiffusionGemmaForBlockDiffusion", false),
    ] {
        let config = format!(r#"{{"architectures":["{architecture}"]}}"#);
        assert_eq!(
            AutoMultimodalLoader.supports_encoder_cache(&config),
            expected,
            "{architecture}"
        );
    }
}

#[test]
fn gemma3_reports_encoder_cache_only_with_vision_config() {
    let loader = Gemma3Loader;
    assert!(
        !loader.supports_encoder_cache(r#"{"architectures":["Gemma3ForConditionalGeneration"]}"#)
    );
    assert!(!loader.supports_encoder_cache(
        r#"{"architectures":["Gemma3ForConditionalGeneration"],"vision_config":null}"#
    ));
    assert!(loader.supports_encoder_cache(
        r#"{"architectures":["Gemma3ForConditionalGeneration"],"vision_config":{}}"#
    ));
}

fn assert_fused_moe_default_isq_predicates(
    loader_name: &str,
    loader: &dyn IsqModelLoader,
    prefixes: &[&str],
) -> Result<()> {
    let predicate_sets = [
        ("isq", loader.isq_layer_regexes("")?),
        ("immediate", loader.immediate_isq_predicates("")?),
    ];
    for (kind, predicates) in predicate_sets {
        for prefix in prefixes {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let key = format!("{prefix}.{projection}.weight");
                assert!(
                    matches_any(&predicates, &key),
                    "{loader_name} {kind} predicates did not match {key}"
                );
            }
        }
    }
    Ok(())
}

struct PromotedIsqCase {
    name: &'static str,
    architecture: &'static str,
    loader: Box<dyn IsqModelLoader>,
    accepted: Vec<&'static str>,
}

fn promoted_isq_cases() -> Vec<PromotedIsqCase> {
    vec![
        PromotedIsqCase {
            name: "phi3v",
            architecture: "Phi3VForCausalLM",
            loader: Box::new(Phi3VLoader),
            accepted: vec![
                "model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "idefics2",
            architecture: "Idefics2ForConditionalGeneration",
            loader: Box::new(Idefics2Loader),
            accepted: vec![
                "model.text_model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "llava_next",
            architecture: "LlavaNextForConditionalGeneration",
            loader: Box::new(LLaVANextLoader),
            accepted: vec![
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "llava",
            architecture: "LlavaForConditionalGeneration",
            loader: Box::new(LLaVALoader),
            accepted: vec![
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "lfm2vl",
            architecture: "Lfm2VlForConditionalGeneration",
            loader: Box::new(Lfm2VlLoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "vllama",
            architecture: "MllamaForConditionalGeneration",
            loader: Box::new(VLlamaLoader),
            accepted: vec![
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen2vl",
            architecture: "Qwen2VLForConditionalGeneration",
            loader: Box::new(Qwen2VLLoader),
            accepted: vec![
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "idefics3",
            architecture: "Idefics3ForConditionalGeneration",
            loader: Box::new(Idefics3Loader),
            accepted: vec![
                "model.text_model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "minicpmo",
            architecture: "MiniCPMO",
            loader: Box::new(MiniCpmOLoader),
            accepted: vec![
                "llm.model.embed_tokens.weight",
                "llm.lm_head.weight",
                "llm.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "phi4mm",
            architecture: "Phi4MMForCausalLM",
            loader: Box::new(Phi4MMLoader),
            accepted: vec![
                "model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen2_5vl",
            architecture: "Qwen2_5_VLForConditionalGeneration",
            loader: Box::new(Qwen2_5VLLoader),
            accepted: vec![
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "gemma3",
            architecture: "Gemma3ForConditionalGeneration",
            loader: Box::new(Gemma3Loader),
            accepted: vec![
                "model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "mistral3",
            architecture: "Mistral3ForConditionalGeneration",
            loader: Box::new(Mistral3Loader),
            accepted: vec![
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "llama4",
            architecture: "Llama4ForConditionalGeneration",
            loader: Box::new(VLlama4Loader),
            accepted: vec![
                "language_model.model.embed_tokens.weight",
                "language_model.lm_head.weight",
                "language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "gemma3n",
            architecture: "Gemma3nForConditionalGeneration",
            loader: Box::new(Gemma3nLoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "model.language_model.embed_tokens_per_layer.weight",
                "model.language_model.lm_head.weight",
                "model.language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen3vl",
            architecture: "Qwen3VLForConditionalGeneration",
            loader: Box::new(Qwen3VLLoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen3vlmoe",
            architecture: "Qwen3VLMoeForConditionalGeneration",
            loader: Box::new(Qwen3VLMoELoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen3_5",
            architecture: "Qwen3_5ForConditionalGeneration",
            loader: Box::new(Qwen3_5Loader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "qwen3_5moe",
            architecture: "Qwen3_5MoeForConditionalGeneration",
            loader: Box::new(Qwen3_5MoeLoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "language_model.model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "voxtral",
            architecture: "VoxtralRealtimeForConditionalGeneration",
            loader: Box::new(VoxtralLoader),
            accepted: vec![
                "mm_streams_embeddings.embedding_module.tok_embeddings.weight",
                "output.weight",
                "output.bias",
            ],
        },
        PromotedIsqCase {
            name: "gemma4",
            architecture: "Gemma4ForConditionalGeneration",
            loader: Box::new(Gemma4Loader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "model.language_model.embed_tokens_per_layer.weight",
                "model.language_model.lm_head.weight",
                "model.language_model.lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "muse_glimmer",
            architecture: "MuseGlimmerForConditionalGeneration",
            loader: Box::new(MuseGlimmerLoader),
            accepted: vec![
                "model.language_model.embed_tokens.weight",
                "lm_head.weight",
                "lm_head.bias",
            ],
        },
        PromotedIsqCase {
            name: "diffusiongemma",
            architecture: "DiffusionGemmaForBlockDiffusion",
            loader: Box::new(DiffusionGemmaLoader),
            accepted: vec![
                "model.decoder.embed_tokens.weight",
                "model.decoder.lm_head.weight",
                "model.decoder.lm_head.bias",
            ],
        },
    ]
}

fn assert_model_scoped_promoted_predicates(case: &PromotedIsqCase, predicates: &[Regex]) {
    for name in &case.accepted {
        assert!(
            matches_any(predicates, name),
            "{} did not promote {name}",
            case.name
        );
        for lookalike in [format!("vision_tower.{name}"), format!("{name}.shadow")] {
            assert!(
                !matches_any(predicates, &lookalike),
                "{} promoted lookalike {lookalike}",
                case.name
            );
        }
    }

    for lookalike in [
        "model.vision_tower.embed_tokens.weight",
        "model.vision_tower.embed_tokens_per_layer.weight",
        "model.vision_tower.lm_head.weight",
        "vision_model.embeddings.word_embeddings.weight",
        "transformer.wte.weight",
    ] {
        assert!(
            !matches_any(predicates, lookalike),
            "{} promoted lookalike {lookalike}",
            case.name
        );
    }
}

#[test]
fn multimodal_promoted_isq_predicates_are_model_scoped() -> Result<()> {
    for case in promoted_isq_cases() {
        let predicates = case.loader.promoted_isq_predicates("")?;
        assert_model_scoped_promoted_predicates(&case, &predicates);
    }

    let gemma4 = Gemma4Loader.promoted_isq_predicates("")?;
    assert!(matches_any(
        &gemma4,
        "model.language_model.embed_tokens_per_layer.weight"
    ));
    for name in [
        "model.language_model.per_layer_model_projection.weight",
        "model.language_model.layers.0.per_layer_projection.weight",
        "model.language_model.per_layer_projection_norm.weight",
    ] {
        assert!(!matches_any(&gemma4, name), "Gemma4 promoted {name}");
    }

    let gemma3n = Gemma3nLoader.promoted_isq_predicates("")?;
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
        "model.language_model.lm_head.weight",
    ] {
        assert!(matches_any(&gemma3n, name), "Gemma3n missed {name}");
    }

    Ok(())
}

#[test]
fn auto_multimodal_delegates_promoted_isq_predicates() -> Result<()> {
    let auto = AutoMultimodalLoader;
    for case in promoted_isq_cases() {
        let config = format!(r#"{{"architectures":["{}"]}}"#, case.architecture);
        let direct = case.loader.promoted_isq_predicates(&config)?;
        let delegated = auto.promoted_isq_predicates(&config)?;
        assert_eq!(
            delegated.iter().map(Regex::as_str).collect::<Vec<_>>(),
            direct.iter().map(Regex::as_str).collect::<Vec<_>>(),
            "{}",
            case.name
        );
        assert_model_scoped_promoted_predicates(&case, &delegated);
    }

    Ok(())
}

#[test]
fn voxtral_detection_is_realtime_only() {
    assert_eq!(
        MultimodalLoaderType::from_causal_lm_name("VoxtralRealtimeForConditionalGeneration")
            .unwrap(),
        MultimodalLoaderType::Voxtral
    );
    assert!(MultimodalLoaderType::from_causal_lm_name("VoxtralForConditionalGeneration").is_err());
}

#[test]
fn qwen3_5_moe_isq_matches_stacked_experts() -> Result<()> {
    let loader = Qwen3_5MoeLoader;
    let names = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj.weight",
        "model.language_model.layers.0.mlp.experts.down_proj.weight",
        "language_model.model.layers.0.mlp.experts.gate_up_proj.weight",
        "language_model.model.layers.0.mlp.experts.down_proj.weight",
    ];

    for regexes in [
        loader.immediate_isq_predicates("")?,
        loader.immediate_isq_predicates_moqe("")?,
    ] {
        for name in names {
            assert!(matches_any(&regexes, name), "{name} was not matched");
        }
    }

    Ok(())
}

#[test]
fn qwen3_multimodal_isq_accepts_both_text_namespaces() -> Result<()> {
    let cases: [(&dyn IsqModelLoader, &[&str]); 4] = [
        (
            &Qwen3VLLoader,
            &["self_attn.q_proj.weight", "mlp.gate_proj.weight"],
        ),
        (
            &Qwen3VLMoELoader,
            &[
                "self_attn.q_proj.weight",
                "mlp.experts.0.gate_proj.weight",
                "mlp.experts.gate_proj.weight",
            ],
        ),
        (
            &Qwen3_5Loader,
            &[
                "self_attn.q_proj.weight",
                "linear_attn.in_proj_qkv.weight",
                "mlp.gate_proj.weight",
            ],
        ),
        (
            &Qwen3_5MoeLoader,
            &[
                "self_attn.q_proj.weight",
                "linear_attn.in_proj_qkv.weight",
                "mlp.experts.0.gate_proj.weight",
                "mlp.experts.gate_up_proj.weight",
                "mlp.shared_expert.gate_proj.weight",
            ],
        ),
    ];

    for (loader, suffixes) in cases {
        for predicates in [
            loader.isq_layer_regexes("")?,
            loader.immediate_isq_predicates("")?,
        ] {
            for prefix in ["model.language_model", "language_model.model"] {
                for suffix in suffixes {
                    let name = format!("{prefix}.layers.0.{suffix}");
                    assert!(matches_any(&predicates, &name), "{name} was not matched");
                }
            }
        }
    }

    for loader in [
        &Qwen3VLMoELoader as &dyn IsqModelLoader,
        &Qwen3_5MoeLoader as &dyn IsqModelLoader,
    ] {
        for predicates in [
            loader.isq_layer_regexes_moqe("")?,
            loader.immediate_isq_predicates_moqe("")?,
        ] {
            for prefix in ["model.language_model", "language_model.model"] {
                for suffix in [
                    "mlp.experts.0.gate_proj.weight",
                    "mlp.experts.gate_proj.weight",
                ] {
                    let name = format!("{prefix}.layers.0.{suffix}");
                    assert!(matches_any(&predicates, &name), "{name} was not matched");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn multimodal_moe_loaders_match_canonical_fused_experts() -> Result<()> {
    assert_fused_moe_default_isq_predicates(
        "VLlama4Loader",
        &VLlama4Loader,
        &["language_model.model.layers.0.feed_forward.experts"],
    )?;
    assert_fused_moe_default_isq_predicates(
        "Qwen3VLMoELoader",
        &Qwen3VLMoELoader,
        &[
            "model.language_model.layers.0.mlp.experts",
            "language_model.model.layers.0.mlp.experts",
        ],
    )?;
    assert_fused_moe_default_isq_predicates(
        "Qwen3_5MoeLoader",
        &Qwen3_5MoeLoader,
        &[
            "model.language_model.layers.0.mlp.experts",
            "language_model.model.layers.0.mlp.experts",
        ],
    )?;
    assert_fused_moe_default_isq_predicates(
        "Gemma4Loader",
        &Gemma4Loader,
        &[
            "model.language_model.layers.0.moe",
            "model.language_model.layers.0.experts",
        ],
    )?;
    Ok(())
}

#[test]
fn unsupported_multimodal_moqe_predicates_are_empty() -> Result<()> {
    for (loader_name, loader) in [
        ("VLlama4Loader", &VLlama4Loader as &dyn IsqModelLoader),
        ("Gemma4Loader", &Gemma4Loader as &dyn IsqModelLoader),
        (
            "MuseGlimmerLoader",
            &MuseGlimmerLoader as &dyn IsqModelLoader,
        ),
    ] {
        assert!(
            loader.isq_layer_regexes_moqe("")?.is_empty(),
            "{loader_name} unexpectedly exposes MoQE predicates"
        );
        assert!(
            loader.immediate_isq_predicates_moqe("")?.is_empty(),
            "{loader_name} unexpectedly exposes immediate MoQE predicates"
        );
    }
    Ok(())
}

#[test]
fn llama4_isq_matches_shared_expert_projections() -> Result<()> {
    let loader = VLlama4Loader;
    let names = [
        "language_model.model.layers.0.feed_forward.shared_expert.gate_proj.weight",
        "language_model.model.layers.0.feed_forward.shared_expert.up_proj.weight",
        "language_model.model.layers.0.feed_forward.shared_expert.down_proj.weight",
    ];

    for predicates in [
        loader.isq_layer_regexes("")?,
        loader.immediate_isq_predicates("")?,
    ] {
        for name in names {
            assert!(matches_any(&predicates, name), "{name} was not matched");
        }
    }

    Ok(())
}

#[test]
fn qwen_multimodal_moqe_only_matches_routed_experts() -> Result<()> {
    let qwen35 = Qwen3_5MoeLoader;
    for predicates in [
        qwen35.isq_layer_regexes("")?,
        qwen35.immediate_isq_predicates("")?,
    ] {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let name =
                format!("model.language_model.layers.0.mlp.shared_expert.{projection}.weight");
            assert!(matches_any(&predicates, &name), "{name} was not matched");
        }
    }

    let accepted = [
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.0.mlp.experts.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.down_proj.weight",
    ];
    let rejected = [
        "lm_head.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.visual.blocks.0.mlp.linear_fc1.weight",
    ];

    for (loader_name, loader) in [
        ("Qwen3VLMoELoader", &Qwen3VLMoELoader as &dyn IsqModelLoader),
        ("Qwen3_5MoeLoader", &Qwen3_5MoeLoader as &dyn IsqModelLoader),
    ] {
        for (kind, predicates) in [
            ("moqe", loader.isq_layer_regexes_moqe("")?),
            ("immediate moqe", loader.immediate_isq_predicates_moqe("")?),
        ] {
            for name in accepted {
                assert!(
                    matches_any(&predicates, name),
                    "{loader_name} {kind} predicates did not match {name}"
                );
            }
            for name in rejected {
                assert!(
                    !matches_any(&predicates, name),
                    "{loader_name} {kind} predicates matched {name}"
                );
            }
        }
    }

    Ok(())
}

#[test]
fn mllama_enables_paged_attention_without_prefix_caching() {
    let loader = VLlamaLoader;
    assert!(loader.supports_paged_attention(""));
    assert!(!loader.supports_prefix_cacher(""));
}

#[test]
fn llama4_processor_uses_defaults_without_processor_config() {
    let loader = VLlama4Loader;
    let processor = loader.get_processor("", None, PreProcessorConfig::default(), None);

    assert!(!processor.get_special_tokens().is_empty());
}

#[test]
fn direct_gguf_adjacent_rope_overrides_multimodal_defaults() {
    let metadata = NormalLoadingMetadata {
        mapper: Box::new(DummyDeviceMapper {
            nm_device: Device::Cpu,
        }),
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(crate::utils::progress::new_multi_progress()),
        matformer_slicing_config: None,
        rope_pairing: Some(RopePairing::Adjacent),
    };

    for loader in [
        &Idefics3Loader as &dyn MultimodalModelLoader,
        &Mistral3Loader as &dyn MultimodalModelLoader,
    ] {
        assert!(loader.is_gptx(""));
        assert!(!loader.is_gptx_for("{}", &metadata).unwrap());
        assert!(!loader
            .is_gptx_for(
                r#"{"_mistralrs_qk_rope_layout":"adjacent"}"#,
                &NormalLoadingMetadata {
                    mapper: Box::new(DummyDeviceMapper {
                        nm_device: Device::Cpu,
                    }),
                    loading_isq: false,
                    real_device: Device::Cpu,
                    multi_progress: Arc::new(crate::utils::progress::new_multi_progress()),
                    matformer_slicing_config: None,
                    rope_pairing: None,
                },
            )
            .unwrap());
    }
}

#[test]
fn gemma3n_enables_paged_attention_and_prefix_caching() {
    let loader = Gemma3nLoader;
    assert!(loader.supports_paged_attention(""));
    assert!(loader.supports_prefix_cacher(""));
}

#[test]
fn gemma4_all_bidirectional_disables_incremental_cache() {
    let all = r#"{"text_config":{"use_bidirectional_attention":"all"}}"#;
    let vision = r#"{"text_config":{"use_bidirectional_attention":"vision"}}"#;

    for loader in [
        &Gemma4Loader as &dyn MultimodalModelLoader,
        &DiffusionGemmaLoader as &dyn MultimodalModelLoader,
    ] {
        assert!(!loader.supports_paged_attention(all));
        assert!(!loader.supports_prefix_cacher(all));
        assert!(loader.supports_paged_attention(vision));
        assert!(loader.supports_prefix_cacher(vision));
    }
}

fn muse_glimmer_test_config() -> String {
    serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "text_config": {},
        "vision_config": {},
        "image_token_id": 200092,
        "video_token_id": 200091
    })
    .to_string()
}

#[test]
fn muse_glimmer_loader_integrates_native_and_gguf_capabilities() -> Result<()> {
    assert_eq!(
        MultimodalLoaderType::from_causal_lm_name("MuseGlimmerForConditionalGeneration")
            .map_err(anyhow::Error::msg)?,
        MultimodalLoaderType::MuseGlimmer
    );
    assert_eq!(
        "muse_glimmer"
            .parse::<MultimodalLoaderType>()
            .map_err(anyhow::Error::msg)?,
        MultimodalLoaderType::MuseGlimmer
    );
    assert_eq!(
        MultimodalLoaderType::MuseGlimmer.to_string(),
        "muse_glimmer"
    );

    let loader = MuseGlimmerLoader;
    let native = muse_glimmer_test_config();
    assert_eq!(
        loader.modalities(&native)?.input,
        vec![
            SupportedModality::Text,
            SupportedModality::Vision,
            SupportedModality::Video
        ]
    );
    let mut gguf: serde_json::Value = serde_json::from_str(&native)?;
    gguf.as_object_mut().unwrap().insert(
        "_mistralrs_muse_glimmer_gguf_collapsed_temporal".to_string(),
        serde_json::Value::Bool(true),
    );
    assert_eq!(
        loader.modalities(&serde_json::to_string(&gguf)?)?.input,
        vec![SupportedModality::Text, SupportedModality::Vision]
    );
    assert!(loader.supports_paged_attention(&native));
    assert!(loader.supports_prefix_cacher(&native));
    Ok(())
}

#[test]
fn muse_glimmer_loader_covers_attention_gate_and_text_only_device_mapping() -> Result<()> {
    let loader = MuseGlimmerLoader;
    let config = muse_glimmer_test_config();
    for predicates in [
        loader.isq_layer_regexes(&config)?,
        loader.immediate_isq_predicates(&config)?,
    ] {
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.gate_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "lm_head.weight",
        ] {
            assert!(matches_any(&predicates, name), "{name} was not matched");
        }
        assert!(!matches_any(
            &predicates,
            "model.vision_tower.layers.0.attn.q_proj.weight"
        ));
    }

    let mapper = DummyDeviceMapper {
        nm_device: Device::Cpu,
    };
    let device_for = loader.get_device_for_tensor(&config, &mapper, false)?;
    assert!(matches!(
        device_for("model.language_model.layers.7.self_attn.q_proj.weight".to_string()),
        DeviceForLoadTensor::Idx(7)
    ));
    assert!(matches!(
        device_for("model.vision_tower.layers.7.attn.q_proj.weight".to_string()),
        DeviceForLoadTensor::Base
    ));
    Ok(())
}

#[test]
fn muse_glimmer_runtime_config_caps_context() -> Result<()> {
    let loader = MuseGlimmerLoader;
    let config = muse_glimmer_test_config();
    let capped = loader.runtime_config(&config, Some(8192))?;
    assert_eq!(
        serde_json::from_str::<MuseGlimmerConfig>(&capped)?
            .text_config
            .max_position_embeddings,
        8192
    );
    assert!(loader.runtime_config(&config, Some(0)).is_err());
    Ok(())
}

#[test]
fn muse_glimmer_estimator_keeps_vision_weights_dense() -> Result<()> {
    let config = serde_json::json!({
        "text_config": {
            "vocab_size": 32,
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "tie_word_embeddings": false
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 3,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4
        },
        "image_token_id": 30,
        "video_token_id": 31,
        "out_hidden_size": 32,
        "projector_hidden_size": 10
    })
    .to_string();
    let loader = MuseGlimmerLoader;
    let dtype = DType::BF16;
    let text_elements = 32 * 12 * 2 + 12;
    let vision_layer = 4 * 8usize.pow(2) + 2 * 8 * 16 + 16 + 9 * 8;
    let vision_elements = 8 * 2 * 3 * 2usize.pow(2)
        + 4 * 4 * 8
        + 4 * 8
        + 3 * vision_layer
        + 32 * 10
        + 10usize.pow(2)
        + 10 * 12;
    assert_eq!(
        loader.non_mapped_size_in_bytes(&config, dtype, 1, None, None)?,
        (text_elements + vision_elements) * dtype.size_in_bytes()
    );

    let isq = IsqType::Q4K;
    let pack_factor = isq.pack_factor(dtype);
    let promoted_pack_factor = isq.promote_for_sensitive_tensor().pack_factor(dtype);
    let quantization = AutoDeviceMapQuantization::isq(Some(isq), None);
    let quantized =
        loader.non_mapped_size_in_bytes(&config, dtype, pack_factor, Some(&quantization), None)?;
    let quantized_text = 32 * 12 * 2 / promoted_pack_factor + 12;
    assert_eq!(
        quantized,
        (quantized_text + vision_elements) * dtype.size_in_bytes()
    );
    Ok(())
}

#[test]
fn gemma4_uqff_keeps_legacy_dense_embeddings_out_of_dummy_layer_regexes() -> Result<()> {
    let loader = Gemma4Loader;
    let embeddings = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
    ];

    let isq_layers = loader.isq_layer_regexes("")?;
    for name in embeddings {
        assert!(!matches_any(&isq_layers, name), "{name} was matched");
    }

    let promoted = loader.promoted_isq_predicates("")?;
    for name in embeddings {
        assert!(matches_any(&promoted, name), "{name} was not promoted");
    }

    Ok(())
}

fn gemma4_estimator_config(tie_word_embeddings: bool) -> String {
    format!(
        r#"{{
                "architectures": ["Gemma4ForCausalLM"],
                "text_config": {{
                    "hidden_size": 12,
                    "intermediate_size": 24,
                    "num_hidden_layers": 2,
                    "sliding_window": 16,
                    "final_logit_softcapping": null,
                    "vocab_size": 24,
                    "tie_word_embeddings": {tie_word_embeddings},
                    "layer_types": ["sliding_attention", "full_attention"],
                    "hidden_size_per_layer_input": 6,
                    "vocab_size_per_layer_input": 18
                }}
            }}"#
    )
}

#[test]
fn gemma4_mapper_metadata_matches_runtime_layer_geometry() -> Result<()> {
    let mut config: serde_json::Value = serde_json::from_str(&gemma4_estimator_config(true))?;
    config["text_config"]["num_key_value_heads"] = 2.into();
    config["text_config"]["num_global_key_value_heads"] = 4.into();
    config["text_config"]["head_dim"] = 64.into();
    config["text_config"]["global_head_dim"] = 128.into();
    let mapped = Gemma4Loader.model_config(&config.to_string())?;
    assert_eq!(mapped.num_kv_heads_for_layer(0), 2);
    assert_eq!(mapped.num_kv_heads_for_layer(1), 4);
    assert_eq!(mapped.k_head_dim_for_layer(0), 64);
    assert_eq!(mapped.k_head_dim_for_layer(1), 128);
    assert_eq!(
        mapped.total_kv_cache_elements_per_token(),
        2 * 128 + 4 * 256
    );
    let config: Gemma4Config = serde_json::from_value(config)?;
    let sharded = crate::vision_models::gemma4::text::Gemma4ModelConfigLike::new(
        &config.text_config,
        &[2, 2],
    )?;
    assert_eq!(sharded.num_attn_heads(), mapped.num_attn_heads() / 2);
    assert_eq!(
        sharded.total_kv_cache_elements_per_token(),
        mapped.total_kv_cache_elements_per_token() / 2
    );
    Ok(())
}

#[test]
fn gemma4_runtime_config_caps_nested_and_flat_contexts() -> Result<()> {
    let loader = Gemma4Loader;
    let config = gemma4_estimator_config(true);

    assert!(matches!(
        loader.runtime_config(&config, None)?,
        Cow::Borrowed(_)
    ));
    assert!(loader.runtime_config(&config, Some(0)).is_err());

    let capped = loader.runtime_config(&config, Some(8192))?;
    let parsed: Gemma4Config = serde_json::from_str(&capped)?;
    assert_eq!(parsed.text_config.max_position_embeddings, 8192);
    assert!(matches!(
        loader.runtime_config(&capped, Some(16384))?,
        Cow::Borrowed(_)
    ));

    let mut flat_value: serde_json::Value = serde_json::from_str(&config)?;
    let text_config = flat_value
        .as_object_mut()
        .unwrap()
        .remove("text_config")
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    flat_value.as_object_mut().unwrap().extend(text_config);
    let flat = serde_json::to_string(&flat_value)?;
    let flat_capped = AutoMultimodalLoader.runtime_config(&flat, Some(8192))?;
    assert_eq!(
        serde_json::from_str::<Gemma4Config>(&flat_capped)?
            .text_config
            .max_position_embeddings,
        8192
    );

    Ok(())
}

#[test]
fn gemma4_estimator_promotes_tied_untied_and_ple_embeddings() -> Result<()> {
    let loader = Gemma4Loader;
    let dtype = DType::BF16;

    for default in [
        IsqType::AFQ4,
        IsqType::AFQ6,
        IsqType::Q4K,
        IsqType::Q5K,
        IsqType::Q6K,
    ] {
        let quantization = AutoDeviceMapQuantization::isq(Some(default), None);
        let pack_factor = default.pack_factor(dtype);
        let promoted_pack_factor = default.promote_for_sensitive_tensor().pack_factor(dtype);
        for tied in [true, false] {
            let config = gemma4_estimator_config(tied);
            let cfg: Gemma4Config = serde_json::from_str(&config)?;
            let tc = cfg.text_config;
            let ple_dim = tc.hidden_size_per_layer_input.unwrap();
            let ple_vocab = tc.vocab_size_per_layer_input.unwrap();
            let embedding_count = if tied { 1 } else { 2 };
            let expected_elements = embedding_count * tc.hidden_size * tc.vocab_size
                / promoted_pack_factor
                + ple_vocab * tc.num_hidden_layers * ple_dim / promoted_pack_factor
                + tc.hidden_size * tc.num_hidden_layers * ple_dim / pack_factor
                + tc.hidden_size
                + ple_dim;
            assert_eq!(
                loader.non_mapped_size_in_bytes(
                    &config,
                    dtype,
                    pack_factor,
                    Some(&quantization),
                    None,
                )?,
                expected_elements * dtype.size_in_bytes(),
                "{default} tied={tied}"
            );
        }
    }

    Ok(())
}

#[test]
fn gemma4_estimator_keeps_multimodal_weights_dense_for_isq() -> Result<()> {
    let loader = Gemma4Loader;
    let dtype = DType::BF16;
    let text_only = gemma4_estimator_config(true);
    let mut multimodal: serde_json::Value = serde_json::from_str(&text_only)?;
    let multimodal = {
        let object = multimodal.as_object_mut().unwrap();
        object.insert("vision_config".to_string(), serde_json::json!({}));
        object.insert("audio_config".to_string(), serde_json::json!({}));
        serde_json::to_string(&multimodal)?
    };

    let dense_text = loader.non_mapped_size_in_bytes(&text_only, dtype, 1, None, None)?;
    let dense_multimodal = loader.non_mapped_size_in_bytes(&multimodal, dtype, 1, None, None)?;
    let dense_multimodal_bytes = dense_multimodal - dense_text;
    assert!(dense_multimodal_bytes > 0);

    let isq = IsqType::Q4K;
    let pack_factor = isq.pack_factor(dtype);
    let quantization = AutoDeviceMapQuantization::isq(Some(isq), None);
    let isq_text = loader.non_mapped_size_in_bytes(
        &text_only,
        dtype,
        pack_factor,
        Some(&quantization),
        None,
    )?;
    let isq_multimodal = loader.non_mapped_size_in_bytes(
        &multimodal,
        dtype,
        pack_factor,
        Some(&quantization),
        None,
    )?;

    assert_eq!(isq_multimodal - isq_text, dense_multimodal_bytes);
    Ok(())
}

fn llama4_estimator_config() -> String {
    serde_json::json!({
        "text_config": {
            "hidden_act": "silu",
            "hidden_size": 12,
            "intermediate_size": 24,
            "vocab_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "rope_scaling": null,
            "tie_word_embeddings": true,
            "use_qk_norm": false,
            "moe_layers": [],
            "interleave_moe_layer_step": 1,
            "intermediate_size_mlp": 24,
            "num_local_experts": 1,
            "num_experts_per_tok": 1,
            "attention_chunk_size": 16
        },
        "vision_config": {
            "hidden_size": 8,
            "hidden_act": "gelu",
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "intermediate_size": 16,
            "vision_output_dim": 8,
            "image_size": 8,
            "patch_size": 4,
            "norm_eps": 1e-6,
            "pixel_shuffle_ratio": 1.0,
            "projector_input_dim": 8,
            "projector_output_dim": 12,
            "vision_feature_layer": -1,
            "rope_theta": 10000.0
        },
        "image_token_index": 0
    })
    .to_string()
}

#[test]
fn llama4_encoder_activation_estimate_scales_with_batch() -> Result<()> {
    let config = llama4_estimator_config();
    let params = |max_batch_size| AutoDeviceMapParams::Multimodal {
        max_seq_len: 16,
        max_batch_size,
        max_image_shape: (8, 8),
        max_num_images: 1,
    };
    let single = VLlama4Loader.non_mapped_max_act_size_elems(&config, &params(1))?;
    assert!(single > 0);
    assert_eq!(
        VLlama4Loader.non_mapped_max_act_size_elems(&config, &params(4))?,
        single * 4
    );
    Ok(())
}

#[test]
fn llama4_estimator_keeps_vision_weights_dense_for_isq() -> Result<()> {
    let config = llama4_estimator_config();
    let loader = VLlama4Loader;
    let dtype = DType::BF16;
    let dense = loader.non_mapped_size_in_bytes(&config, dtype, 1, None, None)?;
    let dense_text_elems = 12 * 24 + 12;
    let dense_vision = dense - dense_text_elems * dtype.size_in_bytes();

    let isq = IsqType::Q4K;
    let pack_factor = isq.pack_factor(dtype);
    let promoted_pack_factor = isq.promote_for_sensitive_tensor().pack_factor(dtype);
    let quantization = AutoDeviceMapQuantization::isq(Some(isq), None);
    let quantized =
        loader.non_mapped_size_in_bytes(&config, dtype, pack_factor, Some(&quantization), None)?;
    let quantized_text_elems = 12 * 24 / promoted_pack_factor + 12;

    assert_eq!(
        quantized,
        dense_vision + quantized_text_elems * dtype.size_in_bytes()
    );
    Ok(())
}

#[test]
fn lfm2vl_estimator_only_packs_selected_non_mapped_weights() -> Result<()> {
    let config = serde_json::json!({
        "text_config": {
            "hidden_size": 12,
            "vocab_size": 24,
            "tie_word_embeddings": true
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "num_patches": 4,
            "patch_size": 2
        },
        "projector_hidden_size": 10,
        "downsample_factor": 2,
        "projector_bias": true,
        "projector_use_layernorm": true
    })
    .to_string();
    let loader = Lfm2VlLoader;
    let dtype = DType::BF16;
    let dense = loader.non_mapped_size_in_bytes(&config, dtype, 1, None, None)?;

    let isq = IsqType::Q4K;
    let pack_factor = isq.pack_factor(dtype);
    let promoted_pack_factor = isq.promote_for_sensitive_tensor().pack_factor(dtype);
    let quantization = AutoDeviceMapQuantization::isq(Some(isq), None);
    let quantized =
        loader.non_mapped_size_in_bytes(&config, dtype, pack_factor, Some(&quantization), None)?;

    let embedding = 12 * 24;
    let projector_linears = (8 * 2usize.pow(2)) * 10 + 10 * 12;
    let expected_savings = embedding - embedding / promoted_pack_factor + projector_linears
        - projector_linears / pack_factor;
    assert_eq!(dense - quantized, expected_savings * dtype.size_in_bytes());
    Ok(())
}

#[test]
fn diffusion_gemma_estimator_keeps_vision_weights_dense_for_isq() -> Result<()> {
    let mut config: serde_json::Value = serde_json::from_str(&gemma4_estimator_config(true))?;
    let object = config.as_object_mut().unwrap();
    object.remove("architectures");
    object.insert("vision_config".to_string(), serde_json::json!({}));
    let config = serde_json::to_string(&config)?;
    let loader = DiffusionGemmaLoader;
    let dtype = DType::BF16;
    let dense = loader.non_mapped_size_in_bytes(&config, dtype, 1, None, None)?;

    let isq = IsqType::Q4K;
    let pack_factor = isq.pack_factor(dtype);
    let quantization = AutoDeviceMapQuantization::isq(Some(isq), None);
    let quantized =
        loader.non_mapped_size_in_bytes(&config, dtype, pack_factor, Some(&quantization), None)?;

    assert_eq!(quantized, dense);
    Ok(())
}

#[test]
fn gemma4_estimator_accounts_audio_as_f32() -> Result<()> {
    let loader = Gemma4Loader;
    let without_audio = gemma4_estimator_config(true);
    let mut with_audio: serde_json::Value = serde_json::from_str(&without_audio)?;
    with_audio
        .as_object_mut()
        .unwrap()
        .insert("audio_config".to_string(), serde_json::json!({}));
    let with_audio = serde_json::to_string(&with_audio)?;

    let bf16_audio_bytes =
        loader.non_mapped_size_in_bytes(&with_audio, DType::BF16, 1, None, None)?
            - loader.non_mapped_size_in_bytes(&without_audio, DType::BF16, 1, None, None)?;
    let f32_audio_bytes =
        loader.non_mapped_size_in_bytes(&with_audio, DType::F32, 1, None, None)?
            - loader.non_mapped_size_in_bytes(&without_audio, DType::F32, 1, None, None)?;

    assert_eq!(bf16_audio_bytes, f32_audio_bytes);
    Ok(())
}

fn gemma3n_estimator_config(ple_vocab_size: usize) -> String {
    format!(
        r#"{{
                "text_config": {{
                    "hidden_size": 12,
                    "intermediate_size": 24,
                    "num_hidden_layers": 12,
                    "num_kv_shared_layers": 0,
                    "vocab_size": 24,
                    "sliding_window": 16,
                    "tie_word_embeddings": true,
                    "rope_scaling": null,
                    "vocab_size_per_layer_input": {ple_vocab_size},
                    "hidden_size_per_layer_input": 6,
                    "altup_num_inputs": 2,
                    "layer_types": [
                        "sliding_attention", "full_attention", "sliding_attention",
                        "full_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "full_attention", "sliding_attention",
                        "full_attention", "sliding_attention", "full_attention"
                    ],
                    "altup_active_idx": 0,
                    "altup_coef_clip": null,
                    "laurel_rank": 4,
                    "altup_correct_scale": true,
                    "activation_sparsity_pattern": [
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ],
                    "final_logit_softcapping": null
                }},
                "vision_config": {{}},
                "audio_config": {{}},
                "audio_soft_tokens_per_image": 0
            }}"#
    )
}

#[test]
fn gemma3n_estimator_packs_default_ple_and_keeps_matformer_ple_dense() -> Result<()> {
    let loader = Gemma3nLoader;
    let dtype = DType::BF16;
    let matformer = MatformerSliceConfig::new(
        "slice".to_string(),
        Arc::new(MatformerConfig {
            slices: HashMap::from([(
                "slice".to_string(),
                Slice {
                    effective_params: 0.0,
                    ffn_hidden_dimensions: vec![24; 12],
                    layers_skipped: Some(vec![0]),
                },
            )]),
        }),
    );

    for default in [IsqType::Q4K, IsqType::Q5K, IsqType::Q6K] {
        let quantization = AutoDeviceMapQuantization::isq(Some(default), None);
        let pack_factor = default.pack_factor(dtype);
        let promoted_pack_factor = default.promote_for_sensitive_tensor().pack_factor(dtype);
        let base = loader.non_mapped_size_in_bytes(
            &gemma3n_estimator_config(18),
            dtype,
            pack_factor,
            Some(&quantization),
            None,
        )?;
        let expanded = loader.non_mapped_size_in_bytes(
            &gemma3n_estimator_config(19),
            dtype,
            pack_factor,
            Some(&quantization),
            None,
        )?;
        assert_eq!(
            expanded - base,
            12 * 6 * dtype.size_in_bytes() / promoted_pack_factor,
            "{default}"
        );

        let matformer_base = loader.non_mapped_size_in_bytes(
            &gemma3n_estimator_config(18),
            dtype,
            pack_factor,
            Some(&quantization),
            Some(&matformer),
        )?;
        let matformer_expanded = loader.non_mapped_size_in_bytes(
            &gemma3n_estimator_config(19),
            dtype,
            pack_factor,
            Some(&quantization),
            Some(&matformer),
        )?;
        assert_eq!(
            matformer_expanded - matformer_base,
            11 * 6 * dtype.size_in_bytes(),
            "{default} MatFormer"
        );
    }

    Ok(())
}

fn gemma3_text_loader_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Gemma3ForCausalLM"],
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "num_attention_heads": 4,
        "num_hidden_layers": 26,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "sliding_window": 512
    })
}

#[test]
fn gemma3_activation_metadata_preserves_structured_metal_masks() -> Result<()> {
    let text = gemma3_text_loader_config();
    let multimodal = serde_json::json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "text_config": text,
        "vision_config": {},
        "image_token_index": 7,
        "mm_tokens_per_image": 256
    });
    for (config, custom_prefill_mask) in [(text, false), (multimodal, true)] {
        for loader in [
            &Gemma3Loader as &dyn DeviceMappedModelLoader,
            &AutoMultimodalLoader,
        ] {
            let metadata = loader.activation_model_config(&config.to_string())?;
            assert_eq!(metadata.custom_prefill_mask, custom_prefill_mask);
            assert!(metadata.metal_prefill_mask_is_structured);
        }
    }
    Ok(())
}

#[test]
fn gemma3_reports_text_only_modalities_without_vision_config() -> Result<()> {
    let loader = Gemma3Loader;
    let text = gemma3_text_loader_config().to_string();
    assert_eq!(
        loader.modalities(&text)?.input,
        vec![SupportedModality::Text]
    );
    assert!(loader.non_mapped_sub_models_for_config(&text)?.is_none());

    let params = AutoDeviceMapParams::Multimodal {
        max_seq_len: 4096,
        max_batch_size: 2,
        max_image_shape: (1024, 1024),
        max_num_images: 1,
    };
    assert!(matches!(
        AutoMultimodalLoader.auto_device_map_params(&text, &params)?,
        AutoDeviceMapParams::Text {
            max_seq_len: 4096,
            max_batch_size: 2
        }
    ));
    assert!(AutoMultimodalLoader
        .non_mapped_sub_models_for_config(&text)?
        .is_none());

    let multimodal = serde_json::json!({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "text_config": gemma3_text_loader_config(),
        "vision_config": {},
        "image_token_index": 7,
        "mm_tokens_per_image": 256
    })
    .to_string();
    assert_eq!(
        loader.modalities(&multimodal)?.input,
        vec![SupportedModality::Text, SupportedModality::Vision]
    );
    let sub_models = loader
        .non_mapped_sub_models_for_config(&multimodal)?
        .expect("vision sub-model");
    assert!(matches!(sub_models.as_slice(), [NonMappedSubModel::Vision]));
    Ok(())
}

#[test]
fn gemma3_device_map_uses_explicit_head_dimension() -> Result<()> {
    let config = gemma3_text_loader_config().to_string();
    let model_config = Gemma3Loader.model_config(&config)?;
    let params = AutoDeviceMapParams::Text {
        max_seq_len: 4096,
        max_batch_size: 1,
    };

    assert_eq!(model_config.k_head_dim(), 256);
    assert_eq!(model_config.v_head_dim(), 256);
    assert!(Gemma3Loader.mapped_max_act_size_elems(&config, &params)? > 0);
    assert_eq!(
        Gemma3Loader.non_mapped_max_act_size_elems(&config, &params)?,
        0
    );
    Ok(())
}

#[test]
fn gemma3_immediate_isq_accepts_both_text_namespaces() -> Result<()> {
    let predicates = Gemma3Loader.immediate_isq_predicates("")?;
    for name in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "language_model.lm_head.weight",
    ] {
        assert!(
            predicates.iter().any(|predicate| predicate.is_match(name)),
            "missing immediate ISQ predicate for {name}"
        );
    }
    Ok(())
}
