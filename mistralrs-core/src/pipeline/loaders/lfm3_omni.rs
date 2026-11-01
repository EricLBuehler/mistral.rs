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
    device_map::DeviceMapper,
    matformer::MatformerSliceConfig,
    paged_attention::{AttentionImplementation, ModelConfigLike},
    pipeline::{
        isq::IsqModelLoader, Modalities, MultimodalPromptPrefixer, Processor, SupportedModality,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    vision_models::{
        lfm3_omni::{
            memory::{ActivationMemory, AudioMemoryEstimate},
            Config, Lfm3OmniModel, Lfm3OmniProcessor,
        },
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
    },
};

const VIDEO_SAMPLING: crate::VideoFrameSampling = crate::VideoFrameSampling::FpsInclusive {
    fps: 2.0,
    min_frames: 4,
    max_frames: 384,
};

pub struct Lfm3OmniLoader;
struct Lfm3OmniPrefixer;

impl MultimodalPromptPrefixer for Lfm3OmniPrefixer {}

impl MultimodalModelLoader for Lfm3OmniLoader {
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
        Ok(Box::new(Lfm3OmniModel::new(
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
        let config: Config = serde_json::from_str(model_config).expect("Invalid LFM3-Omni config");
        Arc::new(Lfm3OmniProcessor::new(
            &config.vision,
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
        Arc::new(Lfm3OmniPrefixer)
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
                SupportedModality::Audio,
            ],
            output: vec![SupportedModality::Text, SupportedModality::Audio],
        })
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

impl IsqModelLoader for Lfm3OmniLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Lfm3VlLoader.promoted_isq_predicates(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Lfm3VlLoader.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Lfm3OmniLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        Lfm3VlLoader.activation_model_config(config)
    }
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Lfm3VlLoader.mapped_max_act_size_elems(config, params)
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
        let (vision_peak, vision_retained) =
            Lfm3VlLoader::encoder_activation_memory_bytes(config, params, dtype, device)?;
        let audio = AudioMemoryEstimate::new(
            &cfg.speech.audio_encoder_config,
            cfg.vision.text_config.hidden_size,
            dtype,
            device,
        )
        .estimate_batch(params.max_seq_len(), params.max_batch_size())?;
        let speech = cfg.output_activation_memory(
            cfg.vision.text_config.max_position_embeddings,
            params.max_batch_size(),
            dtype,
        );
        let vision = ActivationMemory {
            retained_bytes: vision_retained,
            transient_bytes: vision_peak.saturating_sub(vision_retained),
        };
        Ok(vision
            .combine_serial(audio)?
            .combine_serial(speech)?
            .total_bytes()?)
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
        Ok(Lfm3VlLoader.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            matformer,
        )? + cfg
            .speech
            .weight_size_in_bytes(cfg.vision.text_config.hidden_size, dtype))
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
        Lfm3VlLoader.model_config(config)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
    fn checkpoint_layer_index(&self, config: &str, name: &str) -> Option<usize> {
        Lfm3VlLoader.checkpoint_layer_index(config, name)
    }
}
