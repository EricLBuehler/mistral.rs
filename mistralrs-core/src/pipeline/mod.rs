mod amoe;
mod auto;
pub mod chat_template;
mod diffusion;
mod embedding;
mod ggml;
mod gguf;
pub(crate) mod hf;
mod inputs_processor;
mod isq;
pub(crate) mod llg;
mod loaders;
mod macros;
mod normal;
mod paths;
mod processing;
mod response;
mod sampling;
mod speculative;
mod speech;
mod vision;

pub use super::diffusion_models::DiffusionGenerationParams;
use crate::amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use crate::device_map::DeviceMapper;
use crate::paged_attention::{CacheConfig, CacheEngine, ModelConfigLike};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::PagedAttentionConfig;
pub use amoe::{AnyMoeLoader, AnyMoePipeline};
pub use auto::{AutoLoader, AutoLoaderBuilder};
use chat_template::ChatTemplate;
pub use diffusion::{DiffusionLoader, DiffusionLoaderBuilder};
pub use embedding::{EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingSpecificConfig};
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
use image::DynamicImage;
pub use inputs_processor::InputProcessorOutput;
pub(crate) use isq::IsqModelLoader;
pub use isq::{parse_isq_value, IsqModel, IsqOrganization, UQFF_MULTI_FILE_DELIMITER};
use llguidance::toktrie::TokEnv;
pub use loaders::{
    AdapterKind, AutoDeviceMapParams, AutoEmbeddingLoader, AutoNormalLoader, AutoVisionLoader,
    DeepSeekV2Loader, DeepSeekV3Loader, DeviceMappedModelLoader, DiffusionLoaderType,
    DiffusionModel, DiffusionModelLoader, EmbeddingGemmaLoader, EmbeddingLoaderType,
    EmbeddingModel, EmbeddingModelLoader, EmbeddingModelPaths, EmbeddingModule,
    EmbeddingModulePaths, EmbeddingModuleType, FluxLoader, GLM4Loader, GLM4MoeLiteLoader,
    GLM4MoeLoader, Gemma2Loader, Gemma3Loader, Gemma3nLoader, GemmaLoader, GptOssLoader,
    GraniteMoeHybridLoader, Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader,
    LlamaLoader, Loader, LocalModelPaths, MiniCpmOLoader, Mistral3Loader, MistralLoader,
    MixtralLoader, ModelKind, ModelPaths, NormalLoaderType, NormalLoadingMetadata, NormalModel,
    NormalModelLoader, Phi2Loader, Phi3Loader, Phi3VLoader, Phi3_5MoELoader, Phi4MMLoader,
    PrettyName, QuantizationKind, Qwen2Loader, Qwen2VLLoader, Qwen2_5VLLoader,
    Qwen3EmbeddingLoader, Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3VLLoader,
    Qwen3VLMoELoader, SmolLm3Loader, Starcoder2Loader, TokenSource, VLlama4Loader, VLlamaLoader,
    VisionLoaderType, VisionModel, VisionModelLoader, VoxtralLoader,
};
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_device_layers_for_loader(
    loader: &dyn loaders::DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &loaders::AutoDeviceMapParams,
    paged_attn_config: Option<&PagedAttentionConfig>,
) -> Result<crate::device_map::DeviceMapMetadata> {
    loaders::auto_device_map::get_device_layers(
        loader,
        config,
        num_layers,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        devices,
        dtype,
        params,
        paged_attn_config,
    )
}
use mistralrs_quant::IsqType;
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub(crate) use paths::{get_chat_template, get_model_paths, get_xlora_paths};
pub use paths::{AdapterPaths, LoraAdapterPaths};
pub(crate) use processing::{
    apply_chat_template, BasicProcessor, MessagesAction, Processor, ProcessorCreator,
};
use rand_isaac::Isaac64Rng;
pub use speculative::{SpeculativeConfig, SpeculativeLoader, SpeculativePipeline};
pub use speech::{SpeechLoader, SpeechPipeline};
use std::any::Any;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
pub use vision::{VisionLoader, VisionLoaderBuilder, VisionSpecificConfig};

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, Var};

use crate::sequence::Sequence;

pub use self::inputs_processor::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType,
};
use self::text_models_inputs_processor::PagedAttentionMeta;
pub use crate::kv_cache::{
    Cache, CacheManager, EitherCache, KvCache, LayerCaches, NormalCache, NormalCacheType,
};

#[derive(Clone, PartialEq, Eq)]
pub enum SupportedModality {
    Text,
    Audio,
    Vision,
    Embedding,
}

impl Debug for SupportedModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "📝 Text"),
            Self::Audio => write!(f, "🔊 Audio"),
            Self::Vision => write!(f, "🖼️ Vision"),
            Self::Embedding => write!(f, "🔢 Embedding"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Modalities {
    pub input: Vec<SupportedModality>,
    pub output: Vec<SupportedModality>,
}

pub struct GeneralMetadata {
    pub max_seq_len: usize,
    /// Only None if it doesn't make sense for the model
    pub llg_factory: Option<Arc<llguidance::ParserFactory>>,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub kind: ModelKind,
    // TODO: Replace is_xlora queries to check via kind instead:
    pub is_xlora: bool,
    pub activation_dtype: DType,
    pub sliding_window: Option<usize>,
    // PagedAttention stuff
    pub cache_config: Option<CacheConfig>,
    pub cache_engine: Option<CacheEngine>,
    pub model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>>,
    pub modalities: Modalities,
}

impl GeneralMetadata {
    pub fn tok_env(&self) -> Option<TokEnv> {
        self.llg_factory.as_ref().map(|f| f.tok_env().clone())
    }
}

pub enum CacheInstruction {
    In,
    Out,
    /// load_preallocated_cache means to load the preallocated cache, if applicable.
    Reset {
        load_preallocated_cache: bool,
        reset_non_granular: bool,
    },
    Nothing,
}

pub trait PreProcessingMixin: MetadataMixin {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    /// Only None if it doesnt make sense for the model
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>>;
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>>;
}

pub trait IsqPipelineMixin {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()>;
}

pub trait CacheManagerMixin {
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]);
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    );
    fn cache(&self) -> &EitherCache;
    fn do_preallocated_cache(&self) -> bool {
        matches!(self.cache(), EitherCache::Normal(_))
    }
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    /// Only None if it doesnt make sense for the model
    fn tokenizer(&self) -> Option<Arc<Tokenizer>>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    fn get_metadata(&self) -> Arc<GeneralMetadata>;
    fn device_mapper(&self) -> Option<&dyn DeviceMapper>;
}

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoePipelineMixin {
    /// Get vars for each gating layer
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        unreachable!()
    }
    fn amoe_finish_training(&mut self, _gate_model_id: Option<String>) -> candle_core::Result<()> {
        unreachable!()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        unreachable!()
    }
    fn amoe_supported(&self) -> bool {
        false
    }
    /// Per-layer cached outputs.
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        unreachable!()
    }
    /// Inject the MoE layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_create_layers(
        &mut self,
        _model_ids: Vec<String>,
        _token: &TokenSource,
        _revision: Option<String>,
        _match_regex: &str,
        _config: AnyMoeConfig,
        _dtype: DType,
        _dev: &Device,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _silent: bool,
        _gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        unreachable!()
    }
    /// Pre-train the gating layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_pre_train(
        &self,
        _inputs: AnyMoeTrainingInputs,
        (_prefix, _mlp): (String, String),
        _model_ids: Vec<String>,
        _token: TokenSource,
        _revision: Option<String>,
        _layers: Vec<usize>,
        _silent: bool,
    ) -> Result<Option<AnyMoeTrainingResult>, candle_core::Error> {
        unreachable!()
    }
}

/// Category of the model. This can also be used to extract model-category specific tools,
/// such as the vision model prompt prefixer.
#[derive(Clone)]
pub enum ModelCategory {
    Text,
    Vision {
        prefixer: Arc<dyn MultimodalPromptPrefixer>,
    },
    Diffusion,
    Audio,
    Speech,
    Embedding,
}

impl std::fmt::Debug for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Text => write!(f, "ModelCategory::Text"),
            ModelCategory::Vision { .. } => write!(f, "ModelCategory::Vision {{ prefixer: .. }}"),
            ModelCategory::Diffusion => write!(f, "ModelCategory::Diffusion"),
            ModelCategory::Audio => write!(f, "ModelCategory::Audio"),
            ModelCategory::Speech => write!(f, "ModelCategory::Speech"),
            ModelCategory::Embedding => write!(f, "ModelCategory::Embedding"),
        }
    }
}

impl PartialEq for ModelCategory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Text, Self::Text) => true,
            (Self::Vision { .. }, Self::Vision { .. }) => true,
            (Self::Audio, Self::Audio) => true,
            (Self::Speech, Self::Speech) => true,
            (Self::Diffusion, Self::Diffusion) => true,
            (Self::Embedding, Self::Embedding) => true,
            (
                Self::Text
                | Self::Vision { .. }
                | Self::Diffusion
                | Self::Audio
                | Self::Speech
                | Self::Embedding,
                _,
            ) => false,
        }
    }
}

/// Prepend a vision tag appropriate for the model to the prompt. Image indexing is assumed that start at 0.
pub trait MultimodalPromptPrefixer: Send + Sync {
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_image(&self, _image_indices: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_audio(&self, _audio_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

pub enum CacheBackendMetadata {
    DefaultInstructions {
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    },
    PagedAttention {
        metadata: PagedAttentionMeta,
    },
}

#[derive(Clone, Debug)]
pub enum ForwardInputsResult {
    RawLogits {
        logits: Tensor,
    },
    Embeddings {
        embeddings: Tensor,
    },
    CausalGeneration {
        logits: Tensor,
    },
    Image {
        images: Vec<DynamicImage>,
    },
    Speech {
        pcms: Vec<Arc<Vec<f32>>>,
        rates: Vec<usize>,
        channels: Vec<usize>,
    },
}

impl ForwardInputsResult {
    fn index_bs(&self, bs_idx: usize) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.i(bs_idx)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.i(bs_idx)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.i(bs_idx)?,
            }),
            Self::Image { images } => Ok(Self::Image {
                images: vec![images[bs_idx].clone()],
            }),
            Self::Speech {
                pcms,
                rates,
                channels,
            } => Ok(Self::Speech {
                pcms: vec![pcms[bs_idx].clone()],
                rates: vec![rates[bs_idx]],
                channels: vec![channels[bs_idx]],
            }),
        }
    }

    fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.to_device(device)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.to_device(device)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.to_device(device)?,
            }),
            Self::Image { .. } => Ok(self.clone()),
            Self::Speech { .. } => Ok(self.clone()),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct FileListCache {
    files: Vec<String>,
}

#[async_trait::async_trait]
pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + MetadataMixin
    + AnyMoePipelineMixin
{
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    /// Returns the total of model execution time.
    #[allow(clippy::too_many_arguments)]
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
    ) -> Result<Duration, candle_core::Error> {
        match backend_metadata {
            CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                let inputs_iter =
                    std::iter::once(self.get_processor().inputs_processor().process_inputs(
                        self.tokenizer(),
                        input_seqs,
                        is_prompt,
                        self.get_metadata().is_xlora,
                        &self.device(),
                        self.get_metadata().no_kv_cache,
                        None,
                        return_raw_logits,
                        self.get_input_processor_config(),
                        None,
                        self.device_mapper(),
                    ));

                let mut logits = vec![None; input_seqs.len()];
                let len_inputs = 1;
                let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
                let mut embedding_logits = vec![None; input_seqs.len()];

                let mut exec_duration = Duration::ZERO;
                for (i, inputs) in inputs_iter.into_iter().enumerate() {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = inputs.map_err(candle_core::Error::msg)?;
                    if i == 0 {
                        match pre_op {
                            CacheInstruction::In => self.clone_in_cache(input_seqs),
                            CacheInstruction::Nothing => (),
                            CacheInstruction::Reset {
                                load_preallocated_cache,
                                reset_non_granular,
                            } => self.set_none_cache(
                                input_seqs,
                                reset_non_granular,
                                false,
                                load_preallocated_cache,
                            ),
                            _ => unreachable!("Unreachable PRE cache op."),
                        }
                    }

                    let start = Instant::now();
                    let raw_logits = self.forward_inputs(inputs, return_raw_logits)?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                        if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                            raw_out_logits[seq_idx][i] =
                                Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
                        } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                            embedding_logits[seq_idx] =
                                Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
                        } else {
                            logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                        }
                    }
                }

                match post_op {
                    CacheInstruction::Out => self.clone_out_cache(input_seqs),
                    CacheInstruction::Nothing => (),
                    CacheInstruction::Reset {
                        load_preallocated_cache,
                        reset_non_granular,
                    } => self.set_none_cache(
                        input_seqs,
                        reset_non_granular,
                        false,
                        load_preallocated_cache,
                    ),
                    _ => unreachable!("Unreachable POST cache op."),
                }

                if raw_out_logits[0][0].is_some() {
                    let start = Instant::now();
                    response::send_raw_responses(
                        input_seqs,
                        raw_out_logits
                            .into_iter()
                            .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }
                if embedding_logits[0].is_some() {
                    let start = Instant::now();
                    response::send_embedding_responses(
                        input_seqs,
                        embedding_logits
                            .into_iter()
                            .map(|raw| {
                                raw.unwrap()
                                    .to_dtype(DType::F32)
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }

                let start = Instant::now();
                let logits_on_cpu = logits.len() > 1;
                let logits = logits
                    .into_iter()
                    .map(|l| {
                        let l = l.expect("Did not get any inputs. This is shocking.");
                        if logits_on_cpu {
                            l.to_device(&Device::Cpu)
                        } else {
                            Ok(l)
                        }
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;

                match &logits[0] {
                    ForwardInputsResult::RawLogits { .. }
                    | ForwardInputsResult::Embeddings { .. } => unreachable!(),
                    ForwardInputsResult::CausalGeneration { .. } => {
                        self.sample_causal_gen(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::CausalGeneration { logits } = r
                                    else {
                                        unreachable!(
                                            "All results must have same type, `CausalGeneration`"
                                        )
                                    };
                                    logits
                                })
                                .collect::<Vec<_>>(),
                            prefix_cacher,
                            disable_eos_stop,
                            rng,
                        )
                        .await?;
                    }
                    ForwardInputsResult::Image { .. } => {
                        response::send_image_responses(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::Image { images } = r
                                    else {
                                        unreachable!("All results must have same type, `Image`")
                                    };
                                    images
                                        .into_iter()
                                        .next()
                                        .expect("Must have at least 1 element.")
                                })
                                .collect::<Vec<_>>(),
                        )
                        .await?;
                    }
                    ForwardInputsResult::Speech { .. } => {
                        let rates = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { rates, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(rates.len(), 1, "Each sequence must have 1 PCM output.");
                                *rates.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let channels = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { channels, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(
                                    channels.len(),
                                    1,
                                    "Each sequence must have 1 PCM output."
                                );
                                *channels.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let pcms = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { pcms, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(pcms.len(), 1, "Each sequence must have 1 PCM output.");
                                pcms.into_iter().nth(0).unwrap()
                            })
                            .collect::<Vec<_>>();
                        response::send_speech_responses(input_seqs, &pcms, &rates, &channels)
                            .await?;
                    }
                }
                let end = Instant::now();
                exec_duration += end.duration_since(start);

                Ok(exec_duration)
            }
            CacheBackendMetadata::PagedAttention { metadata } => {
                let inputs_iter =
                    std::iter::once(self.get_processor().inputs_processor().process_inputs(
                        self.tokenizer(),
                        input_seqs,
                        is_prompt,
                        self.get_metadata().is_xlora,
                        &self.device(),
                        self.get_metadata().no_kv_cache,
                        None,
                        return_raw_logits,
                        self.get_input_processor_config(),
                        Some(metadata),
                        self.device_mapper(),
                    ));

                let mut logits = vec![None; input_seqs.len()];
                let len_inputs = 1;
                let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
                let mut embedding_logits = vec![None; input_seqs.len()];

                let mut exec_duration = Duration::ZERO;
                for (i, inputs) in inputs_iter.into_iter().enumerate() {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = inputs.map_err(candle_core::Error::msg)?;

                    let start = Instant::now();
                    let raw_logits = self.forward_inputs(inputs, return_raw_logits)?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                        if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                            raw_out_logits[seq_idx][i] =
                                Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
                        } else if let ForwardInputsResult::Embeddings { embeddings } = &raw_logits {
                            embedding_logits[seq_idx] =
                                Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
                        } else {
                            logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                        }
                    }
                }

                if raw_out_logits[0][0].is_some() {
                    let start = Instant::now();
                    response::send_raw_responses(
                        input_seqs,
                        raw_out_logits
                            .into_iter()
                            .map(|raw| raw.into_iter().flatten().collect::<Vec<_>>())
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }
                if embedding_logits[0].is_some() {
                    let start = Instant::now();
                    response::send_embedding_responses(
                        input_seqs,
                        embedding_logits
                            .into_iter()
                            .map(|raw| {
                                raw.unwrap()
                                    .to_dtype(DType::F32)
                                    .unwrap()
                                    .to_vec1::<f32>()
                                    .unwrap()
                            })
                            .collect(),
                    )
                    .await?;
                    let end = Instant::now();
                    exec_duration += end.duration_since(start);

                    return Ok(exec_duration);
                }

                let start = Instant::now();
                let logits_on_cpu = logits.len() > 1;
                let logits = logits
                    .into_iter()
                    .map(|l| {
                        let l = l.expect("Did not get any inputs. This is shocking.");
                        if logits_on_cpu {
                            l.to_device(&Device::Cpu)
                        } else {
                            Ok(l)
                        }
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;

                match &logits[0] {
                    ForwardInputsResult::RawLogits { .. }
                    | ForwardInputsResult::Embeddings { .. } => unreachable!(),
                    ForwardInputsResult::CausalGeneration { .. } => {
                        self.sample_causal_gen(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::CausalGeneration { logits } = r
                                    else {
                                        unreachable!("All results must have same type")
                                    };
                                    logits
                                })
                                .collect::<Vec<_>>(),
                            prefix_cacher,
                            disable_eos_stop,
                            rng,
                        )
                        .await?;
                    }
                    ForwardInputsResult::Image { .. } => {
                        response::send_image_responses(
                            input_seqs,
                            logits
                                .into_iter()
                                .map(|r| {
                                    #[allow(irrefutable_let_patterns)]
                                    let ForwardInputsResult::Image { images } = r
                                    else {
                                        unreachable!("All results must have same type, `Image`")
                                    };
                                    images
                                        .into_iter()
                                        .next()
                                        .expect("Must have at least 1 element.")
                                })
                                .collect::<Vec<_>>(),
                        )
                        .await?;
                    }
                    ForwardInputsResult::Speech { .. } => {
                        let rates = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { rates, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(rates.len(), 1, "Each sequence must have 1 PCM output.");
                                *rates.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let channels = logits
                            .iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { channels, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(
                                    channels.len(),
                                    1,
                                    "Each sequence must have 1 PCM output."
                                );
                                *channels.first().unwrap()
                            })
                            .collect::<Vec<_>>();
                        let pcms = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::Speech { pcms, .. } = r
                                else {
                                    unreachable!("All results must have same type, `Speech`")
                                };
                                assert_eq!(pcms.len(), 1, "Each sequence must have 1 PCM output.");
                                pcms.into_iter().nth(0).unwrap()
                            })
                            .collect::<Vec<_>>();
                        response::send_speech_responses(input_seqs, &pcms, &rates, &channels)
                            .await?;
                    }
                }
                let end = Instant::now();
                exec_duration += end.duration_since(start);

                Ok(exec_duration)
            }
        }
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;

    fn category(&self) -> ModelCategory;

    /// Return encoder cache hit/miss counters (hits, misses) if this pipeline has an encoder cache.
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        None
    }
}

pub(crate) fn extract_logits(
    logits: &Tensor,
    context_lens: Vec<(usize, usize)>,
) -> candle_core::Result<Tensor> {
    let mut toks = Vec::new();
    for (dim, (start, len)) in logits.chunk(logits.dims()[0], 0)?.iter().zip(context_lens) {
        toks.push(dim.narrow(1, start, len)?);
    }
    Tensor::cat(&toks, 0)
}

#[cfg(test)]
mod tests {
    use crate::MessageContent;
    use either::Either;
    use indexmap::IndexMap;
    use serde_json::Value;

    macro_rules! hashmap {
        (@single $($x:tt)*) => (());
        (@count $($rest:expr),*) => (<[()]>::len(&[$(hashmap!(@single $rest)),*]));

        ($($key:expr => $value:expr,)+) => { hashmap!($($key => $value),+) };
        ($($key:expr => $value:expr),*) => {
            {
                let _cap = hashmap!(@count $($key),*);
                let mut _map = ::indexmap::IndexMap::with_capacity(_cap);
                $(
                    let _ = _map.insert($key, Value::String($value));
                )*
                _map
            }
        };
    }

    #[cfg(test)]
    #[track_caller]
    fn test_with_inputs(
        templates: &[(bool, &str, &str, &str, &str)],
        expected_outputs: &[&str],
        inputs: Vec<IndexMap<String, MessageContent>>,
    ) {
        use crate::pipeline::chat_template::ChatTemplateValue;

        use super::chat_template::apply_chat_template_to;
        let mut failed = Vec::new();
        let n_templates = templates.len();
        for ((has_system, bos, eos, unk, template), expected) in
            templates.iter().zip(expected_outputs)
        {
            let output = match apply_chat_template_to(
                if !has_system {
                    inputs[1..].to_vec()
                } else {
                    inputs.clone()
                },
                true,
                None,
                None, // reasoning_effort
                &ChatTemplateValue(Either::Left(template.to_string())),
                Some(bos.to_string()),
                Some(eos.to_string()),
                Some(unk.to_string()),
                Vec::new(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    failed.push(format!("Failed with {e}."));
                    continue;
                }
            };
            if output != *expected {
                failed.push(format!(
                    "Expected: `{}` \n\nGot:      `{}`",
                    expected.replace('\n', "\\n"),
                    output.replace('\n', "\\n")
                ));
            }
        }
        if !failed.is_empty() {
            for (i, line) in failed.iter().enumerate() {
                println!("------------ Template {i} ------------");
                println!("{line}");
            }
            println!("------------------------");
            panic!("{}/{n_templates} chat templates failed.", failed.len());
        }
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> t=transformers.AutoTokenizer.from_pretrained(...)
    /// # If non-system prompt model
    /// >>> t.apply_chat_template([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// # If system prompt model
    /// >>> t.apply_chat_template([{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_chat_templates() {
        let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"),
            // mistralai/Mistral-7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // meta-llama/Llama-2-13b-chat-hf
            (true, "<s>", "</s>", "<unk>", "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"),
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // google/gemma-7b-it
            (false, "<bos>", "<eos>", "<unk>", "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"),
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
        let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s> [INST] Who are you [/INST]   I am an assistant   </s> [INST] Another question [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST] I am an assistant </s><s>[INST] Another question [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            // google/gemma-7b-it
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        ];
        let messages = [
            ["system", "You are a helpful assistant"],
            ["user", "Hello"],
            ["assistant", "Hi there"],
            ["user", "Who are you"],
            ["assistant", "   I am an assistant   "],
            ["user", "Another question"],
        ];
        let mut inputs = Vec::new();
        for [role, content] in messages {
            let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message.insert("role".to_string(), Either::Left(role.to_string()));
            message.insert("content".to_string(), Either::Left(content.to_string()));
            inputs.push(message);
        }
        test_with_inputs(&templates, &expected_outputs, inputs);
    }

    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> processor=transformers.AutoProcessor.from_pretrained(...)
    /// >>> processor.apply_chat_template([
    ///         {"role":"system","content":[{"type":"text", "text": "You are a helpful assistant"}]},
    ///         {"role":"user","content":[{"type":"image"}, {"type":"text", "text": "Hello, please describe the above."}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "Hi there"}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Who are you"}]},
    ///         {"role":"assistant","content":[{"type":"text", "text": "   I am an assistant   "}]},
    ///         {"role":"user","content":[{"type":"text", "text": "Another question"}]}
    ///     ], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_image_chat_templates() {
        let templates = [
            // HuggingFaceM4/idefics2-8b-chatty
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"),
        ];
        let expected_outputs = [
            // HuggingFaceM4/idefics2-8b-chatty
            "System: You are a helpful assistant<end_of_utterance>\nUser:<image>Hello, please describe the above.<end_of_utterance>\nAssistant: Hi there<end_of_utterance>\nUser:<image>This is me, who are you<end_of_utterance>\nAssistant:    I am an assistant   <end_of_utterance>\nUser:<image>Another question, what is this?<end_of_utterance>\nAssistant:",
        ];

        let mut inputs = Vec::new();

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("system".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "You are a helpful assistant".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Hello, please describe the above.".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "Hi there".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "This is me, who are you".to_string()
                },
            ]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("assistant".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![hashmap! {
                "type".to_string() => "text".to_string(),
                "text".to_string() => "   I am an assistant   ".to_string()
            }]),
        );
        inputs.push(message);

        let mut message: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
            IndexMap::new();
        message.insert("role".to_string(), Either::Left("user".to_string()));
        message.insert(
            "content".to_string(),
            Either::Right(vec![
                hashmap! {
                    "type".to_string() => "image".to_string()
                },
                hashmap! {
                    "type".to_string() => "text".to_string(),
                    "text".to_string() => "Another question, what is this?".to_string()
                },
            ]),
        );
        inputs.push(message);

        test_with_inputs(&templates, &expected_outputs, inputs);
    }
}
