use std::{borrow::Cow, collections::HashMap, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device};
use mistralrs_quant::ShardedVarBuilder;
use regex::Regex;

use super::{
    ActivationModelConfig, AutoDeviceMapParams, AutoDeviceMapQuantization, DeviceMappedModelLoader,
    Lfm3Loader, Lfm3VlLoader, MultimodalModel, MultimodalModelLoader, NonMappedSubModel,
    NormalLoadingMetadata,
};
use crate::{
    attention::ATTENTION_CHUNK_SIZE,
    device_map::DeviceMapper,
    matformer::MatformerSliceConfig,
    paged_attention::{AttentionImplementation, ModelConfigLike},
    pipeline::{
        isq::IsqModelLoader, Modalities, MultimodalPromptPrefixer, Processor, SupportedModality,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    vision_models::{
        lfm3_audio::{config::AudioConfig, Lfm3AudioProcessor},
        lfm3_omni::{memory::AudioMemoryEstimate, Lfm3OmniModel},
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

const LEGACY_PREFIXES: [(&str, &str); 11] = [
    ("lfm.model.", "model.language_model."),
    ("lfm.lm_head.", "lm_head."),
    ("encoder.downsample.", "model.audio_encoder.downsample."),
    ("encoder.lfm.lfm.", "model.audio_encoder.lfm."),
    ("mlp.model.0.", "model.audio_projector.layer_norm."),
    ("mlp.model.1.", "model.audio_projector.linear_1."),
    ("mlp.model.3.", "model.audio_projector.linear_2."),
    ("codebook_embedding.", "model.codebook_embedding."),
    ("depth_linear.", "model.depth_decoder.depth_linear."),
    ("depthformer.model.", "model.depth_decoder.model."),
    ("depth_embeddings.", "model.depth_decoder.depth_embeddings."),
];

pub struct Lfm3AudioLoader;
struct Lfm3AudioPrefixer;

impl MultimodalPromptPrefixer for Lfm3AudioPrefixer {}

fn canonical_name(name: &str) -> Option<String> {
    if name == "mask_embedding" {
        return Some("model.depth_decoder.mask_embedding".to_string());
    }
    LEGACY_PREFIXES.iter().find_map(|(legacy, canonical)| {
        name.strip_prefix(legacy)
            .map(|suffix| format!("{canonical}{suffix}"))
    })
}

fn remap_legacy_weights(vb: ShardedVarBuilder) -> ShardedVarBuilder {
    if vb.contains_tensor("model.language_model.embed_tokens.weight")
        || !vb.contains_tensor("lfm.model.embed_tokens.weight")
    {
        return vb;
    }
    let aliases: HashMap<_, _> = vb
        .tensor_names()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|name| canonical_name(&name).map(|canonical| (canonical, name)))
        .collect();
    vb.with_tensor_aliases(aliases)
}

fn text_config(config: &str) -> Result<String> {
    let cfg: AudioConfig = serde_json::from_str(config)?;
    let mut text = cfg.text_config;
    text.tie_word_embeddings = cfg.tie_word_embeddings;
    Ok(serde_json::to_string(&text)?)
}

fn text_layer_index(name: &str) -> Option<usize> {
    name.strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("lfm.model.layers."))?
        .split('.')
        .next()?
        .parse()
        .ok()
}

impl MultimodalModelLoader for Lfm3AudioLoader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        Lfm3VlLoader.runtime_config(config, max_model_len)
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(Lfm3OmniModel::new_audio(
            &serde_json::from_str(config)?,
            remap_legacy_weights(vb),
            self.is_gptx_for(config, &metadata)?,
            metadata,
            attention,
        )?))
    }

    fn is_gptx(&self, _config: &str) -> bool {
        true
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(serde_json::from_str::<AudioConfig>(config)?))
    }

    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Lfm3AudioProcessor::new())
    }

    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }

    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Lfm3AudioPrefixer)
    }

    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Audio],
            output: vec![SupportedModality::Text, SupportedModality::Audio],
        })
    }

    fn get_device_for_tensor(
        &self,
        _config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync>> {
        Ok(Arc::new(move |name| {
            if !loading_isq {
                if let Some(layer) = text_layer_index(&name) {
                    return DeviceForLoadTensor::Idx(layer);
                }
            }
            DeviceForLoadTensor::Base
        }))
    }
}

impl IsqModelLoader for Lfm3AudioLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^(model\.language_model|lfm\.model)\.embed_tokens\.weight$")?,
            Regex::new(r"^(lfm\.)?lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^(lfm\.)?lm_head\.(weight|bias)$")?,
            Regex::new(
                r"^(model\.language_model|lfm\.model)\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))\.(weight|bias)$",
            )?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Lfm3AudioLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        Lfm3Loader.activation_model_config(&text_config(config)?)
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: AudioConfig = serde_json::from_str(config)?;
        Ok(params.max_batch_size()
            * cfg.text_config.num_attention_heads
            * params.max_seq_len().min(ATTENTION_CHUNK_SIZE).pow(2))
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(
            self.non_mapped_activation_memory_bytes(config, params, DType::F32, &Device::Cpu)?
                / DType::F32.size_in_bytes(),
        )
    }

    fn non_mapped_activation_memory_bytes(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        dtype: DType,
        device: &Device,
    ) -> Result<usize> {
        let cfg: AudioConfig = serde_json::from_str(config)?;
        let audio = AudioMemoryEstimate::new(
            &cfg.speech.audio_encoder_config,
            cfg.text_config.hidden_size,
            dtype,
            device,
        )
        .estimate_batch(params.max_seq_len(), params.max_batch_size())?;
        let speech = cfg.speech.output_activation_memory(
            cfg.text_config.max_position_embeddings,
            params.max_batch_size(),
            cfg.text_config.hidden_size,
            dtype,
        );
        Ok(audio.combine_serial(speech)?.total_bytes()?)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: AudioConfig = serde_json::from_str(config)?;
        let text = &cfg.text_config;
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            cfg.tie_word_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let language = text.hidden_size * text.vocab_size / embed_pack
            + if cfg.tie_word_embeddings {
                0
            } else {
                text.hidden_size * text.vocab_size / head_pack
            }
            + text.hidden_size;
        Ok(language * dtype.size_in_bytes()
            + cfg.speech.weight_size_in_bytes(text.hidden_size, dtype))
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Lfm3Loader.layer_sizes_in_bytes(&text_config(config)?, dtype, weight_pack_factor, matformer)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        Lfm3Loader.num_layers(&text_config(config)?)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Lfm3Loader.model_config(&text_config(config)?)
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Audio])
    }

    fn checkpoint_layer_index(&self, _config: &str, name: &str) -> Option<usize> {
        text_layer_index(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const LOADED_PARAMETERS: usize = 2_013_919_106;
    const CODEC_WINDOW: usize = 1280;
    const SERVING_SEQUENCES: usize = 32;
    const GIB: usize = 1024 * 1024 * 1024;
    const SERVING_ACTIVATION_BUDGET: usize = 16 * GIB;
    const OMNI_ACTIVATION_BUDGET: usize = 10 * GIB;

    fn checkpoint_config() -> Result<String> {
        let mut encoder_layers = ["conv", "conv", "full_attention"].repeat(5);
        encoder_layers.extend(["full_attention", "full_attention"]);
        let mut config = json!({
            "lfm": {"head_dim":128},
            "encoder": {"dim_in":128,"dim_conv":256,"dim_out":768,"sliding_window":32,
                "lfm": {"hidden_size":768,"intermediate_size":2560,"num_hidden_layers":17,
                    "num_attention_heads":8,"num_key_value_heads":4,"norm_eps":1e-5,
                    "rope_parameters":{"rope_theta":1000000.0,"rope_type":"default"},
                    "conv_bias":false,"conv_L_cache":3,"block_multiple_of":256,
                    "block_ffn_dim_multiplier":1.0,"block_auto_adjust_ff_dim":true,
                    "layer_types":encoder_layers}},
            "depthformer":{"hidden_size":1024,"intermediate_size":2816,"num_hidden_layers":3,
                "num_attention_heads":32,"num_key_value_heads":8,"head_dim":32,"norm_eps":1e-5,
                "conv_bias":false,"conv_L_cache":7,"use_sandwich_norm":false,
                "layer_types":["full_attention","full_attention","full_attention"],
                "sliding_window":256,"rope_parameters":{"full_attention":{
                    "rope_theta":1000000.0,"rope_type":"default","partial_rotary_factor":1.0}}},
            "codec_config":{"encoder_dim":512,"stft_n_fft":1280,"stft_hop_length":320,
                "stft_win_length":CODEC_WINDOW,"stft_frame_stack":6,"quantizer_dim":256,
                "fsq_levels":[8,7,6,6],"n_q":12,"decoder_window":30,"sampling_rate":24000}
        });
        crate::vision_models::lfm3_audio::config::normalize_config(&mut config)?;
        Ok(serde_json::to_string(&config)?)
    }

    #[test]
    fn checkpoint_memory_and_runtime_context_match_audio_model() -> Result<()> {
        let config = checkpoint_config()?;
        let layers = Lfm3AudioLoader.layer_sizes_in_bytes(&config, DType::BF16, 1, None)?;
        let shared =
            Lfm3AudioLoader.non_mapped_size_in_bytes(&config, DType::BF16, 1, None, None)?;
        assert_eq!(
            layers.iter().sum::<usize>() + shared,
            LOADED_PARAMETERS * DType::BF16.size_in_bytes() + CODEC_WINDOW * 2
        );
        let runtime = Lfm3AudioLoader.runtime_config(&config, Some(4096))?;
        let cfg: AudioConfig = serde_json::from_str(&runtime)?;
        assert_eq!(cfg.text_config.max_position_embeddings, 4096);
        let model = Lfm3AudioLoader.model_config(&config)?;
        assert_eq!(model.total_kv_cache_elements_per_token(), 11 * 4 * 256);
        assert_eq!(model.sliding_window_for_layer(4), Some(256));
        Ok(())
    }

    #[test]
    fn serving_memory_covers_batching_and_full_output_context() -> Result<()> {
        let config = checkpoint_config()?;
        let estimate = |config: &str, batch| {
            Lfm3AudioLoader.non_mapped_activation_memory_bytes(
                config,
                &AutoDeviceMapParams::Text {
                    max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
                    max_batch_size: batch,
                },
                DType::BF16,
                &Device::Cpu,
            )
        };
        let single = estimate(&config, 1)?;
        let serving = estimate(&config, SERVING_SEQUENCES)?;
        assert!(serving > single);
        assert!(serving < single * 2);
        assert!(serving < SERVING_ACTIVATION_BUDGET);
        assert_eq!(Lfm3AudioLoader.model_config(&config)?.max_seq_len(), 128000);
        let shorter = Lfm3AudioLoader
            .runtime_config(&config, Some(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN))?;
        assert!(serving > estimate(&shorter, SERVING_SEQUENCES)? + GIB);
        let cfg: AudioConfig = serde_json::from_str(&config)?;
        let speech = |batch| {
            cfg.speech
                .output_activation_memory(
                    cfg.text_config.max_position_embeddings,
                    batch,
                    cfg.text_config.hidden_size,
                    DType::BF16,
                )
                .total_bytes()
        };
        let single = speech(1)?;
        let batched = speech(SERVING_SEQUENCES)?;
        assert!(batched > single);
        assert!(batched < single * 2);
        Ok(())
    }

    #[test]
    fn default_omni_serving_shares_workspace_between_modalities() -> Result<()> {
        use crate::pipeline::{Lfm3OmniLoader, Lfm3VlLoader};
        use crate::vision_models::lfm3_omni::Config;

        let mut config: serde_json::Value = serde_json::from_str(&checkpoint_config()?)?;
        let mut vision_layers = vec!["conv", "conv"];
        vision_layers.extend(["full_attention", "conv"].repeat(6));
        config.as_object_mut().unwrap().extend(
            json!({
                "vision_config": {
                    "hidden_size":1024,"intermediate_size":2560,"num_hidden_layers":14,
                    "num_attention_heads":16,"num_key_value_heads":8,"in_channels":3,
                    "patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
                    "conv_kernel_size":3,"norm_eps":1e-5,"layer_types":vision_layers,
                    "rope_parameters":{"rope_theta":1000000.0}
                },
                "image_token_id":125030,"video_token_id":125029,
                "projector_hidden_act":"gelu","projector_hidden_size":4096,
                "projector_bias":true,"downsample_factor":2
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        let config = serde_json::to_string(&config)?;
        let cfg: Config = serde_json::from_str(&config)?;
        let params = AutoDeviceMapParams::Multimodal {
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: SERVING_SEQUENCES,
            max_image_shape: (1024, 1024),
            max_num_images: 1,
        };
        let vision = Lfm3VlLoader.non_mapped_activation_memory_bytes(
            &config,
            &params,
            DType::BF16,
            &Device::Cpu,
        )?;
        let audio = AudioMemoryEstimate::new(
            &cfg.speech.audio_encoder_config,
            cfg.vision.text_config.hidden_size,
            DType::BF16,
            &Device::Cpu,
        )
        .estimate_batch(params.max_seq_len(), SERVING_SEQUENCES)?
        .total_bytes()?;
        let speech = cfg
            .output_activation_memory(
                cfg.vision.text_config.max_position_embeddings,
                SERVING_SEQUENCES,
                DType::BF16,
            )
            .total_bytes()?;
        let combined = Lfm3OmniLoader.non_mapped_activation_memory_bytes(
            &config,
            &params,
            DType::BF16,
            &Device::Cpu,
        )?;
        assert!(combined > vision.max(audio).max(speech));
        assert!(combined < vision + audio + speech);
        assert!(combined < OMNI_ACTIVATION_BUDGET);
        assert_eq!(Lfm3OmniLoader.model_config(&config)?.max_seq_len(), 128000);
        Ok(())
    }

    #[test]
    fn legacy_and_canonical_names_use_only_text_layers_for_isq_and_mapping() -> Result<()> {
        let regexes = Lfm3AudioLoader.isq_layer_regexes("{}")?;
        for prefix in ["lfm.model", "model.language_model"] {
            let name = format!("{prefix}.layers.2.self_attn.gate_proj.weight");
            assert!(regexes.iter().any(|regex| regex.is_match(&name)));
            assert_eq!(text_layer_index(&name), Some(2));
        }
        for name in [
            "encoder.lfm.lfm.layers.2.self_attn.q_proj.weight",
            "model.audio_encoder.lfm.layers.2.self_attn.q_proj.weight",
            "depthformer.model.layers.2.self_attn.q_proj.weight",
            "model.depth_decoder.model.layers.2.self_attn.q_proj.weight",
            "depth_embeddings.0.to_logits.weight",
            "decoder.lfm.layers.2.self_attn.q_proj.weight",
        ] {
            assert!(!regexes.iter().any(|regex| regex.is_match(name)), "{name}");
            assert_eq!(text_layer_index(name), None, "{name}");
        }
        assert_eq!(
            canonical_name("encoder.lfm.lfm.embedding_norm.weight").as_deref(),
            Some("model.audio_encoder.lfm.embedding_norm.weight")
        );
        assert_eq!(
            canonical_name("mlp.model.3.bias").as_deref(),
            Some("model.audio_projector.linear_2.bias")
        );
        assert_eq!(
            canonical_name("mask_embedding").as_deref(),
            Some("model.depth_decoder.mask_embedding")
        );
        assert_eq!(canonical_name("decoder.lfm.embedding_norm.weight"), None);
        Ok(())
    }

    #[test]
    fn legacy_weight_aliases_are_idempotent_and_leave_codec_names_intact() -> Result<()> {
        let vb = mistralrs_quant::ShardedSafeTensors::wrap(
            HashMap::from([
                (
                    "lfm.model.embed_tokens.weight".to_string(),
                    candle_core::Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?,
                ),
                (
                    "decoder.lin.bias".to_string(),
                    candle_core::Tensor::zeros(2, DType::F32, &Device::Cpu)?,
                ),
            ]),
            DType::F32,
            Device::Cpu,
        );
        let vb = remap_legacy_weights(remap_legacy_weights(vb));
        assert_eq!(
            vb.pp("model.language_model.embed_tokens")
                .get((2, 3), "weight")?
                .dims(),
            &[2, 3]
        );
        assert!(vb.contains_tensor("decoder.lin.bias"));
        assert_eq!(
            vb.tensor_names().unwrap(),
            [
                "decoder.lin.bias",
                "model.language_model.embed_tokens.weight"
            ]
        );
        Ok(())
    }
}
