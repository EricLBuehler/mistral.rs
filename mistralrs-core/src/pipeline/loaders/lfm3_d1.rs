use std::{borrow::Cow, fmt::Debug, sync::Arc};

use anyhow::Result;
use candle_core::{DType, Device};
use mistralrs_quant::ShardedVarBuilder;
use regex::Regex;

use super::{
    ActivationModelConfig, AutoDeviceMapParams, AutoDeviceMapQuantization, DeviceMappedModelLoader,
    Lfm3VlLoader, MultimodalModel, MultimodalModelLoader, NonMappedSubModel, NormalLoadingMetadata,
};
use crate::{
    attention::ATTENTION_CHUNK_SIZE,
    device_map::DeviceMapper,
    matformer::MatformerSliceConfig,
    models::lfm3::DECISION_PROJECTION_CHUNK_SIZE,
    paged_attention::{AttentionImplementation, ModelConfigLike},
    pipeline::{
        isq::IsqModelLoader, Modalities, MultimodalPromptPrefixer, Processor, SupportedModality,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    vision_models::{
        lfm3_d1::{
            config::{Config, ModelConfig},
            DecisionProcessor, Lfm3D1Model,
        },
        lfm3_omni::memory::{ActivationMemory, AudioMemoryEstimate},
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

pub struct Lfm3D1Loader;

const HIDDEN_BUFFERS: usize = 4;
const MLP_BUFFERS: usize = 3;
const SCORE_BUFFERS: usize = 2;
const FLOAT32_ELEMENT_RATIO: usize = 2;

impl MultimodalModelLoader for Lfm3D1Loader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        let value: serde_json::Value = serde_json::from_str(config)?;
        anyhow::ensure!(
            matches!(value["model_type"].as_str(), Some("lfm3_vl" | "lfm3_omni")),
            "The d1 loader supports LFM3-VL and LFM3-Omni backbones"
        );
        Lfm3VlLoader.runtime_config(config, max_model_len)
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        metadata: NormalLoadingMetadata,
        attention: AttentionImplementation,
    ) -> Result<Box<dyn MultimodalModel + Send + Sync>> {
        Ok(Box::new(Lfm3D1Model::new(
            &serde_json::from_str(config)?,
            vb,
            self.is_gptx_for(config, &metadata)?,
            metadata,
            attention,
        )?))
    }

    fn is_gptx(&self, config: &str) -> bool {
        Lfm3VlLoader.is_gptx(config)
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
        Arc::new(DecisionProcessor::new(
            model_config,
            preprocessor_config,
            max_edge,
        ))
    }

    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        false
    }
    fn supports_encoder_cache(&self, _config: &str) -> bool {
        true
    }

    fn prefixer(&self, config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Lfm3VlLoader.prefixer(config)
    }

    fn video_frame_sampling(&self, config: &str) -> crate::VideoFrameSampling {
        Lfm3VlLoader.video_frame_sampling(config)
    }

    fn modalities(&self, config: &str) -> Result<Modalities> {
        let cfg: Config = serde_json::from_str(config)?;
        let mut modalities = Lfm3VlLoader.modalities(config)?;
        if cfg.audio_encoder_config.is_some() {
            modalities.input.push(SupportedModality::Audio);
        }
        modalities.output = vec![SupportedModality::Decision];
        Ok(modalities)
    }

    fn get_device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync>> {
        Lfm3VlLoader.get_device_for_tensor(config, mapper, loading_isq)
    }
}

impl IsqModelLoader for Lfm3D1Loader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Lfm3VlLoader.promoted_isq_predicates(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Lfm3VlLoader.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Lfm3VlLoader.immediate_isq_predicates(config)
    }
}

impl DeviceMappedModelLoader for Lfm3D1Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        Ok(ActivationModelConfig {
            custom_prefill_mask: true,
            ..Lfm3VlLoader.activation_model_config(config)?
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Config = serde_json::from_str(config)?;
        let text = &cfg.vision.text_config;
        let tokens = params.max_seq_len().min(text.max_position_embeddings);
        let pairs = tokens * tokens.min(ATTENTION_CHUNK_SIZE);
        let hidden = HIDDEN_BUFFERS * tokens * text.hidden_size;
        let attention = SCORE_BUFFERS * FLOAT32_ELEMENT_RATIO * text.num_attention_heads * pairs
            + (FLOAT32_ELEMENT_RATIO + 1) * pairs;
        let mlp = MLP_BUFFERS * tokens * text.intermediate_size;
        let projection = FLOAT32_ELEMENT_RATIO * DECISION_PROJECTION_CHUNK_SIZE * text.vocab_size;
        Ok(hidden + attention.max(mlp).max(projection))
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(
            self.non_mapped_activation_memory_bytes(config, params, DType::BF16, &Device::Cpu)?
                / DType::BF16.size_in_bytes(),
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
        let (peak, retained) =
            Lfm3VlLoader::encoder_activation_memory_bytes(config, params, dtype, device)?;
        let vision = ActivationMemory {
            retained_bytes: retained,
            transient_bytes: peak.saturating_sub(retained),
        };
        let Some(audio) = &cfg.audio_encoder_config else {
            return Ok(vision.total_bytes()?);
        };
        let audio =
            AudioMemoryEstimate::new(audio, cfg.vision.text_config.hidden_size, dtype, device)
                .estimate_batch(params.max_seq_len(), params.max_batch_size())?;
        Ok(vision.combine_serial(audio)?.total_bytes()?)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
        matformer: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Config = serde_json::from_str(config)?;
        let audio = cfg.audio_encoder_config.as_ref().map_or(0, |audio| {
            let input = audio.hidden_size;
            let hidden = cfg.vision.text_config.hidden_size;
            let projector = 2 * input + input * hidden + hidden * hidden + 2 * hidden;
            (audio.parameter_count() + projector) * dtype.size_in_bytes()
        });
        Ok(Lfm3VlLoader.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            matformer,
        )? + audio)
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Lfm3VlLoader.layer_sizes_in_bytes(config, dtype, weight_pack_factor, matformer)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        Lfm3VlLoader.num_layers(config)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Ok(Box::new(ModelConfig(Lfm3VlLoader.model_config(config)?)))
    }

    fn supports_checkpoint_weight_inventory(&self, _config: &str) -> Result<bool> {
        Ok(false)
    }

    fn non_mapped_sub_models_for_config(
        &self,
        config: &str,
    ) -> Result<Option<Vec<NonMappedSubModel>>> {
        let cfg: Config = serde_json::from_str(config)?;
        let mut models = vec![NonMappedSubModel::Vision];
        if cfg.audio_encoder_config.is_some() {
            models.push(NonMappedSubModel::Audio);
        }
        Ok(Some(models))
    }

    fn checkpoint_layer_index(&self, config: &str, name: &str) -> Option<usize> {
        Lfm3VlLoader.checkpoint_layer_index(config, name)
    }
}
