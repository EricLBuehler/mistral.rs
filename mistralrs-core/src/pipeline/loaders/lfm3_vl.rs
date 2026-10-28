use std::{borrow::Cow, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device};
use mistralrs_quant::ShardedVarBuilder;
use regex::Regex;

use super::{
    lfm3::Lfm3Loader, ActivationModelConfig, AutoDeviceMapParams, AutoDeviceMapQuantization,
    DeviceMappedModelLoader, MultimodalModel, MultimodalModelLoader, NonMappedSubModel,
    NormalLoadingMetadata, NormalModelLoader,
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
        lfm3_vl::{
            config::LayerType, memory::EncoderMemoryEstimate, Config, Lfm3VlModel, Lfm3VlProcessor,
        },
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

const VIDEO_SAMPLING: crate::VideoFrameSampling = crate::VideoFrameSampling::FpsInclusive {
    fps: 2.0,
    min_frames: 4,
    max_frames: 768,
};

pub struct Lfm3VlLoader;
struct Lfm3VlPrefixer;

impl MultimodalPromptPrefixer for Lfm3VlPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_owned()
    }
}

fn text_config(config: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(config)?;
    Ok(serde_json::to_string(&value["text_config"])?)
}

impl MultimodalModelLoader for Lfm3VlLoader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        if max_model_len.is_none() {
            return Ok(Cow::Borrowed(config));
        }
        let mut value: serde_json::Value = serde_json::from_str(config)?;
        let text = serde_json::to_string(&value["text_config"])?;
        value["text_config"] =
            serde_json::from_str(&Lfm3Loader.runtime_config(&text, max_model_len)?)?;
        Ok(Cow::Owned(serde_json::to_string(&value)?))
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(Lfm3VlModel::new(
            &serde_json::from_str(config)?,
            vb,
            self.is_gptx_for(config, &metadata)?,
            metadata,
            attention,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(serde_json::from_str::<Config>(config)?))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Lfm3VlProcessor::new(
            &serde_json::from_str(model_config).expect("Invalid LFM3-VL config"),
            &preprocessor_config,
            max_edge,
        ))
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
        Arc::new(Lfm3VlPrefixer)
    }
    fn video_frame_sampling(&self, _config: &str) -> crate::VideoFrameSampling {
        VIDEO_SAMPLING
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Video,
            ],
            output: vec![SupportedModality::Text],
        })
    }
    fn get_device_for_tensor(
        &self,
        _config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
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

fn text_layer_index(name: &str) -> Option<usize> {
    name.strip_prefix("model.language_model.layers.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

impl IsqModelLoader for Lfm3VlLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.language_model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^lm_head\.(weight|bias)$")?,
            Regex::new(
                r"^model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)|conv\.(in_proj|out_proj)|feed_forward\.(w1|w2|w3))\.(weight|bias)$",
            )?,
            Regex::new(r"^model\.multi_modal_projector\.linear_[12]\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Lfm3VlLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        Lfm3Loader.activation_model_config(&text_config(config)?)
    }
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal device map parameters");
        };
        let cfg: Config = serde_json::from_str(config)?;
        Ok(max_batch_size
            * cfg.text_config.num_attention_heads
            * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2))
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
        let cfg: Config = serde_json::from_str(config)?;
        let vc = &cfg.vision_config;
        let AutoDeviceMapParams::Multimodal {
            max_batch_size,
            max_image_shape,
            max_num_images,
            ..
        } = params
        else {
            anyhow::bail!("Expected multimodal device map parameters");
        };
        let factor = vc.patch_size * vc.spatial_merge_size;
        let items = max_batch_size * max_num_images;
        Ok(EncoderMemoryEstimate::new(vc, dtype, device).estimate([[
            items,
            max_image_shape.0.div_ceil(factor) * vc.spatial_merge_size,
            max_image_shape.1.div_ceil(factor) * vc.spatial_merge_size,
        ]])?)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Config = serde_json::from_str(config)?;
        let tc = &cfg.text_config;
        let vc = &cfg.vision_config;
        let (embed_pack, head_pack) = super::language_model_pack_factors(
            quantization,
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            tc.tie_word_embeddings(),
            dtype,
            weight_pack_factor,
        )?;
        let text = tc.hidden_size * tc.vocab_size / embed_pack
            + if tc.tie_word_embeddings() {
                0
            } else {
                tc.hidden_size * tc.vocab_size / head_pack
            }
            + tc.hidden_size;
        let mut vision =
            vc.in_channels * vc.temporal_patch_size * vc.patch_size.pow(2) * vc.hidden_size
                + vc.hidden_size;
        let head_dim = vc.hidden_size / vc.num_attention_heads;
        for layer in &vc.layer_types {
            vision += 2 * vc.hidden_size + 3 * vc.hidden_size * vc.intermediate_size;
            vision += match layer {
                LayerType::Conv => {
                    4 * vc.hidden_size.pow(2) + vc.hidden_size * vc.conv_kernel_size.pow(2)
                }
                LayerType::Attention => {
                    2 * vc.hidden_size.pow(2)
                        + 2 * vc.hidden_size * vc.num_key_value_heads * head_dim
                        + 2 * head_dim
                }
            };
        }
        let projector_pack = |name| {
            quantization.map_or(Ok(weight_pack_factor), |q| {
                q.pack_factor_for_candidates(&[name], dtype, weight_pack_factor, false)
            })
        };
        let projector_weight =
            vc.hidden_size * cfg.downsample_factor.pow(2) * cfg.projector_hidden_size
                / projector_pack("model.multi_modal_projector.linear_1.weight")?
                + cfg.projector_hidden_size * tc.hidden_size
                    / projector_pack("model.multi_modal_projector.linear_2.weight")?;
        let projector = 2 * vc.hidden_size
            + projector_weight
            + if cfg.projector_bias {
                cfg.projector_hidden_size + tc.hidden_size
            } else {
                0
            };
        Ok((text + vision + projector) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Lfm3Loader.layer_sizes_in_bytes(
            &text_config(config)?,
            dtype,
            weight_pack_factor,
            matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Lfm3Loader.num_layers(&text_config(config)?)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Lfm3Loader.model_config(&text_config(config)?)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
    fn checkpoint_layer_index(&self, _config: &str, tensor_name: &str) -> Option<usize> {
        text_layer_index(tensor_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_layers_are_not_device_mapped_with_text_layers() {
        assert_eq!(
            text_layer_index("model.language_model.layers.3.conv.in_proj.weight"),
            Some(3)
        );
        assert_eq!(
            text_layer_index("model.vision_tower.layers.3.conv.in_proj.weight"),
            None
        );
    }

    #[test]
    fn quantization_selects_text_and_projector_weights_only() -> Result<()> {
        let patterns = Lfm3VlLoader.isq_layer_regexes("")?;
        for name in [
            "model.language_model.layers.3.conv.in_proj.weight",
            "model.language_model.layers.4.self_attn.q_proj.weight",
            "model.language_model.layers.4.self_attn.o_proj.weight",
            "model.language_model.layers.4.self_attn.gate_proj.weight",
            "model.multi_modal_projector.linear_1.weight",
        ] {
            assert!(patterns.iter().any(|pattern| pattern.is_match(name)));
        }
        for name in [
            "model.vision_tower.layers.3.conv.in_proj.weight",
            "model.language_model.layers.3.conv.conv.weight",
            "model.multi_modal_projector.layer_norm.weight",
        ] {
            assert!(!patterns.iter().any(|pattern| pattern.is_match(name)));
        }
        Ok(())
    }
}
