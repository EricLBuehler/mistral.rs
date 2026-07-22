mod amoe;
mod auto;
pub mod chat_template;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_graph;
mod diffusion;
mod embedding;
mod ggml;
mod gguf;
pub(crate) mod hf;
mod inputs_processor;
mod isq;
mod isq_flow;
pub use isq_flow::CalibrationStatus;
pub(crate) mod llg;
mod loaders;
mod macros;
mod multimodal;
mod normal;
mod paths;
mod processing;
mod prompt_chunks;
mod response;
pub(crate) mod sampling;
mod speech;

pub use super::diffusion_models::DiffusionGenerationParams;
use crate::amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use crate::device_map::DeviceMapper;
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{CacheConfig, CacheEngine, ModelConfigLike};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::PagedAttentionConfig;
pub use amoe::{AnyMoeLoader, AnyMoePipeline};
pub use auto::{AutoLoader, AutoLoaderBuilder};
use chat_template::ChatTemplate;
pub use diffusion::{DiffusionLoader, DiffusionLoaderBuilder};
pub(crate) use embedding::EmbeddingLoadContext;
pub use embedding::{EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingSpecificConfig};
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
use image::DynamicImage;
pub use inputs_processor::InputProcessorOutput;
pub(crate) use isq::IsqModelLoader;
pub use isq::{
    expand_isq_value, expand_uqff_shards, parse_isq_value, parse_uqff_shard,
    resolve_uqff_shorthand, IsqModel, IsqOrganization, UqffWriteConfig, UQFF_MULTI_FILE_DELIMITER,
};
use llguidance::toktrie::TokEnv;
pub use loaders::{
    AdapterKind, AutoDeviceMapParams, AutoEmbeddingLoader, AutoMultimodalLoader, AutoNormalLoader,
    DeepSeekV2Loader, DeepSeekV3Loader, DeviceMappedModelLoader, DiffusionGemmaLoader,
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, EmbeddingGemmaLoader,
    EmbeddingLoaderType, EmbeddingModel, EmbeddingModelLoader, EmbeddingModelPaths,
    EmbeddingModule, EmbeddingModulePaths, EmbeddingModuleType, FluxLoader, GLM4Loader,
    GLM4MoeLiteLoader, GLM4MoeLoader, Gemma2Loader, Gemma3Loader, Gemma3nLoader, Gemma4Loader,
    GemmaLoader, GptOssLoader, GraniteMoeHybridLoader, HunYuanDenseV1Loader, HunYuanMoEV1Loader,
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Lfm2Loader, Lfm2VlLoader,
    LlamaLoader, Loader, LocalModelPaths, MiniCpmOLoader, Mistral3Loader, MistralLoader,
    MixtralLoader, ModelKind, ModelPaths, MultimodalLoaderType, MultimodalModel,
    MultimodalModelLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    Phi2Loader, Phi3Loader, Phi3VLoader, Phi3_5MoELoader, Phi4MMLoader, PrettyName,
    QuantizationKind, Qwen2Loader, Qwen2VLLoader, Qwen2_5VLLoader, Qwen3EmbeddingLoader,
    Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, SmolLm3Loader, Starcoder2Loader, TokenSource, VLlama4Loader, VLlamaLoader,
    VoxtralLoader,
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
pub use multimodal::{MultimodalLoader, MultimodalLoaderBuilder, MultimodalSpecificConfig};
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub(crate) use paths::{
    get_adapter_paths, get_chat_template, get_model_paths, AdapterPathOptions, XLoraPreload,
};
pub use paths::{AdapterPaths, ResolvedLoraAdapter};
pub(crate) use processing::{
    apply_chat_template, BasicProcessor, MessagesAction, Processor, ProcessorCreator,
};
use rand_isaac::Isaac64Rng;
pub use speech::{SpeechLoader, SpeechPipeline};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, DeviceLocation, IndexOp, Tensor, Var};

use crate::paged_attention::block_hash::{
    adapter_generation_key, compute_block_hashes, MultimodalAttentionPolicy,
};
use crate::sequence::Sequence;

use prompt_chunks::build_prompt_chunk_plan;

pub use self::inputs_processor::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType,
};
use self::text_models_inputs_processor::{
    FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
};

const DEFAULT_PAGED_PREFILL_CHUNK_SIZE: usize = 4096;

pub(crate) fn validate_lora_loader_config(
    adapters: Option<&[crate::LoraAdapterSpec]>,
    runtime_config: Option<crate::LoraRuntimeConfig>,
) -> anyhow::Result<()> {
    if let Some(runtime_config) = runtime_config {
        runtime_config.validate()?;
    }
    let Some(adapters) = adapters else {
        return Ok(());
    };
    let mut aliases = std::collections::HashSet::new();
    for adapter in adapters {
        let alias = adapter.alias.trim();
        if alias.is_empty() {
            anyhow::bail!("LoRA adapter alias must not be empty");
        }
        if alias.len() > crate::MAX_LORA_ALIAS_BYTES {
            anyhow::bail!(
                "LoRA adapter alias must not exceed {} bytes",
                crate::MAX_LORA_ALIAS_BYTES
            );
        }
        if adapter.source.trim().is_empty() {
            anyhow::bail!(
                "LoRA adapter source for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter.revision().is_empty() {
            anyhow::bail!(
                "LoRA adapter revision for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter
            .base_model_name
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            anyhow::bail!(
                "LoRA adapter `{}` has an empty base_model_name",
                adapter.alias
            );
        }
        if !aliases.insert(alias) {
            anyhow::bail!(
                "LoRA adapter alias `{}` is specified more than once",
                adapter.alias
            );
        }
    }
    if let Some(config) = runtime_config.filter(|config| aliases.len() > config.max_adapters) {
        anyhow::bail!(
            "LoRA adapter preload count {} exceeds the configured maximum {}",
            aliases.len(),
            config.max_adapters
        );
    }
    Ok(())
}

pub use crate::kv_cache::{
    Cache, CacheManager, EitherCache, HybridLayerCache, KvCache, LayerCaches, NormalCache,
    NormalCacheType,
};

pub(crate) type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

pub(crate) fn metadata_rope_positions<'a>(
    metadata: &'a PagedAttentionInputMetadata,
    device: &Device,
) -> Option<&'a Tensor> {
    metadata
        .rope_positions
        .as_ref()
        .and_then(|positions| positions.get(&device.location()))
}

#[allow(dead_code)]
pub(crate) enum ForwardCache<'a> {
    Normal(&'a mut [KvCache]),
    Paged {
        kv_cache: &'a [(Tensor, Tensor)],
        metadata: &'a PagedAttentionInputMetadata,
    },
    None,
}

#[allow(dead_code)]
impl<'a> ForwardCache<'a> {
    pub(crate) fn from_paged(
        metadata: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
    ) -> Self {
        match metadata {
            Some((kv_cache, metadata)) => Self::Paged { kv_cache, metadata },
            None => Self::None,
        }
    }

    pub(crate) fn paged_metadata(
        &self,
    ) -> Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)> {
        match self {
            Self::Paged { kv_cache, metadata } => Some((kv_cache.to_vec(), metadata)),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub(crate) fn paged_layer(
        &self,
        layer_idx: usize,
    ) -> Option<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        match self {
            Self::Paged { kv_cache, metadata } => Some((kv_cache[layer_idx].clone(), metadata)),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub(crate) fn rope_positions(&self, device: &Device) -> Option<&Tensor> {
        match self {
            Self::Paged { metadata, .. } => metadata_rope_positions(metadata, device),
            Self::Normal(_) | Self::None => None,
        }
    }

    pub(crate) fn is_final_prompt_chunk(&self) -> bool {
        match self {
            Self::Paged { metadata, .. } => metadata.is_final_prompt_chunk,
            Self::Normal(_) | Self::None => true,
        }
    }

    pub(crate) fn is_first_prompt_chunk(&self) -> bool {
        match self {
            Self::Paged { metadata, .. } => metadata.is_first_prompt_chunk,
            Self::Normal(_) | Self::None => true,
        }
    }

    pub(crate) fn normal_mut(&mut self) -> Option<&mut [KvCache]> {
        match self {
            Self::Normal(cache) => Some(cache),
            Self::Paged { .. } | Self::None => None,
        }
    }
}

#[allow(dead_code)]
pub(crate) enum ForwardPositions<'a> {
    Text { seqlen_offsets: &'a [usize] },
    Mrope { position_ids: &'a Tensor },
    None,
}

pub(crate) enum ForwardMaskCache<'a> {
    Normal(&'a [KvCache]),
    Paged(&'a [usize]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecurrentBatchKind {
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
pub(crate) struct RecurrentMetadata {
    batch_kind: RecurrentBatchKind,
    state_indices: Tensor,
    state_indices_host: Option<Vec<u32>>,
}

impl RecurrentMetadata {
    pub(crate) fn new(
        batch_kind: RecurrentBatchKind,
        state_indices: Tensor,
        state_indices_host: Option<Vec<u32>>,
    ) -> Self {
        Self {
            batch_kind,
            state_indices,
            state_indices_host,
        }
    }

    pub(crate) fn batch_kind(&self) -> RecurrentBatchKind {
        self.batch_kind
    }

    pub(crate) fn state_indices(&self) -> &Tensor {
        &self.state_indices
    }

    pub(crate) fn state_indices_host(&self) -> Option<&[u32]> {
        self.state_indices_host.as_deref()
    }
}

impl PastKvLenCache for ForwardMaskCache<'_> {
    fn get_past_kv_len(&self) -> candle_core::Result<usize> {
        match self {
            Self::Normal(cache) => Ok(cache
                .iter()
                .map(KvCache::current_seq_len)
                .max()
                .unwrap_or(0)),
            Self::Paged(offsets) => offsets.get_past_kv_len(),
        }
    }
}

pub(crate) struct ModelForwardContext<'a> {
    cache: ForwardCache<'a>,
    positions: ForwardPositions<'a>,
    rope_positions: HashMap<(DeviceLocation, usize), Tensor>,
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    flash_params: &'a FlashParams,
    recurrent_metadata: Option<RecurrentMetadata>,
    recurrent_batch_kind: Option<RecurrentBatchKind>,
    requires_full_prefill_queries: bool,
}

#[allow(dead_code)]
impl<'a> ModelForwardContext<'a> {
    pub(crate) fn new(
        seqlen_offsets: &'a [usize],
        context_lens: &'a [(usize, usize)],
        position_ids: &'a [usize],
        metadata: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
        flash_params: &'a FlashParams,
    ) -> Self {
        Self {
            cache: ForwardCache::from_paged(metadata),
            positions: ForwardPositions::Text { seqlen_offsets },
            rope_positions: HashMap::new(),
            context_lens,
            position_ids,
            flash_params,
            recurrent_metadata: None,
            recurrent_batch_kind: None,
            requires_full_prefill_queries: false,
        }
    }

    pub(crate) fn with_cache(
        cache: ForwardCache<'a>,
        seqlen_offsets: &'a [usize],
        context_lens: &'a [(usize, usize)],
        position_ids: &'a [usize],
        flash_params: &'a FlashParams,
    ) -> Self {
        Self {
            cache,
            positions: ForwardPositions::Text { seqlen_offsets },
            rope_positions: HashMap::new(),
            context_lens,
            position_ids,
            flash_params,
            recurrent_metadata: None,
            recurrent_batch_kind: None,
            requires_full_prefill_queries: false,
        }
    }

    pub(crate) fn with_recurrent_batch_kind(
        mut self,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> Self {
        self.recurrent_batch_kind = Some(recurrent_batch_kind);
        self
    }

    pub(crate) fn with_recurrent_metadata(
        mut self,
        recurrent_metadata: Option<RecurrentMetadata>,
    ) -> Self {
        if let Some(metadata) = recurrent_metadata.as_ref() {
            self.recurrent_batch_kind = Some(metadata.batch_kind());
        }
        self.recurrent_metadata = recurrent_metadata;
        self
    }

    pub(crate) fn require_full_prefill_queries(&mut self) {
        self.requires_full_prefill_queries = true;
    }

    pub(crate) fn requires_full_prefill_queries(&self) -> bool {
        self.requires_full_prefill_queries
    }

    pub(crate) fn cache(&self) -> &ForwardCache<'a> {
        &self.cache
    }

    pub(crate) fn cache_mut(&mut self) -> &mut ForwardCache<'a> {
        &mut self.cache
    }

    pub(crate) fn is_paged(&self) -> bool {
        matches!(self.cache, ForwardCache::Paged { .. })
    }

    pub(crate) fn seqlen_offsets(&self) -> &[usize] {
        match self.positions {
            ForwardPositions::Text { seqlen_offsets } => seqlen_offsets,
            ForwardPositions::Mrope { .. } | ForwardPositions::None => &[],
        }
    }

    pub(crate) fn context_lens(&self) -> &[(usize, usize)] {
        self.context_lens
    }

    pub(crate) fn context_lens_vec(&self) -> Vec<(usize, usize)> {
        self.context_lens.to_vec()
    }

    pub(crate) fn position_ids(&self) -> &[usize] {
        self.position_ids
    }

    pub(crate) fn position_ids_vec(&self) -> Vec<usize> {
        self.position_ids.to_vec()
    }

    pub(crate) fn flash_params(&self) -> &FlashParams {
        self.flash_params
    }

    pub(crate) fn recurrent_metadata(&self) -> Option<&RecurrentMetadata> {
        self.recurrent_metadata.as_ref()
    }

    pub(crate) fn recurrent_batch_kind(&self) -> Option<RecurrentBatchKind> {
        self.recurrent_batch_kind
    }

    pub(crate) fn prompt_chunk_attention_policy(&self) -> MultimodalAttentionPolicy {
        self.paged_input_metadata()
            .map_or(MultimodalAttentionPolicy::Causal, |metadata| {
                metadata.prompt_chunk_attention_policy
            })
    }

    pub(crate) fn paged_metadata(
        &self,
    ) -> Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)> {
        self.cache.paged_metadata()
    }

    pub(crate) fn paged_input_metadata(&self) -> Option<&PagedAttentionInputMetadata> {
        match &self.cache {
            ForwardCache::Paged { metadata, .. } => Some(*metadata),
            ForwardCache::Normal(_) | ForwardCache::None => None,
        }
    }

    pub(crate) fn paged_layer(
        &self,
        layer_idx: usize,
    ) -> Option<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        self.cache.paged_layer(layer_idx)
    }

    pub(crate) fn text_positions(
        &mut self,
        device: &Device,
        seq_len: usize,
    ) -> candle_core::Result<Option<&Tensor>> {
        if self.cache.rope_positions(device).is_some() {
            return Ok(self.cache.rope_positions(device));
        }
        let ForwardPositions::Text { seqlen_offsets } = self.positions else {
            return Ok(None);
        };
        let location = device.location();
        let key = (location, seq_len);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.rope_positions.entry(key) {
            entry.insert(text_positions_tensor(seqlen_offsets, seq_len, device)?);
        }
        Ok(self.rope_positions.get(&key))
    }

    pub(crate) fn text_positions_from_offsets(
        &self,
        seqlen_offsets: &[usize],
        seq_len: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        text_positions_tensor(seqlen_offsets, seq_len, device)
    }

    pub(crate) fn is_first_prompt_chunk(&self) -> bool {
        self.cache.is_first_prompt_chunk()
    }

    pub(crate) fn is_final_prompt_chunk(&self) -> bool {
        self.cache.is_final_prompt_chunk()
    }

    pub(crate) fn mask_cache<'b>(&'b self, normal_cache: &'b [KvCache]) -> ForwardMaskCache<'b> {
        match self.cache {
            ForwardCache::Paged { .. } => ForwardMaskCache::Paged(self.seqlen_offsets()),
            ForwardCache::Normal(_) | ForwardCache::None => ForwardMaskCache::Normal(normal_cache),
        }
    }

    pub(crate) fn logits(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
        LogitsSelection::from_context_lens(logits, self.context_lens, &[logits.device().clone()])?
            .select(logits)
    }
}

pub(crate) fn text_positions_tensor(
    seqlen_offsets: &[usize],
    seq_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut positions = Vec::with_capacity(seqlen_offsets.len() * seq_len);
    for offset in seqlen_offsets {
        for seq_idx in 0..seq_len {
            positions.push(u32::try_from(offset + seq_idx).map_err(candle_core::Error::wrap)?);
        }
    }
    Tensor::from_vec(positions, (seqlen_offsets.len() * seq_len,), device)
}

#[derive(Clone, Debug)]
pub(crate) enum LogitsSelection {
    Decode {
        start: usize,
        len: usize,
    },
    Indices {
        indices: DeviceTensorMap,
        batch: usize,
        len: usize,
    },
    All,
}

impl LogitsSelection {
    pub(crate) fn from_context_lens(
        source: &Tensor,
        context_lens: &[(usize, usize)],
        devices: &[Device],
    ) -> candle_core::Result<Self> {
        let dims = source.dims();
        if dims.len() < 2 {
            candle_core::bail!("logits selection source must have rank >= 2");
        }
        let batch = dims[0];
        let seq_len = dims[1];
        if context_lens.len() != batch {
            candle_core::bail!(
                "logits selection batch mismatch: {} spans for batch {batch}",
                context_lens.len()
            );
        }
        let Some((first_start, first_len)) = context_lens.first().copied() else {
            candle_core::bail!("logits selection requires at least one span");
        };
        for (start, len) in context_lens.iter().copied() {
            let end = start
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::msg("logits selection span overflow"))?;
            if end > seq_len {
                candle_core::bail!(
                    "logits selection span ({start}, {len}) exceeds sequence length {seq_len}"
                );
            }
        }
        if context_lens.iter().all(|span| *span == (0, seq_len)) {
            return Ok(Self::All);
        }
        if context_lens
            .iter()
            .all(|span| *span == (first_start, first_len))
        {
            return Ok(Self::Decode {
                start: first_start,
                len: first_len,
            });
        }

        if context_lens.iter().any(|(_, len)| *len != first_len) {
            candle_core::bail!("ragged logits selection spans are not supported");
        }

        let mut flat_indices = Vec::with_capacity(batch * first_len);
        for (batch_idx, (start, len)) in context_lens.iter().copied().enumerate() {
            let end = start + len;
            for pos in start..end {
                let idx = batch_idx
                    .checked_mul(seq_len)
                    .and_then(|idx| idx.checked_add(pos))
                    .ok_or_else(|| candle_core::Error::msg("logits selection index overflow"))?;
                flat_indices.push(u32::try_from(idx).map_err(candle_core::Error::wrap)?);
            }
        }

        let cpu_indices = Tensor::from_vec(flat_indices, (batch * first_len,), &Device::Cpu)?;
        let mut indices = HashMap::new();
        for device in devices {
            indices.insert(device.location(), cpu_indices.to_device(device)?);
        }
        Ok(Self::Indices {
            indices,
            batch,
            len: first_len,
        })
    }

    pub(crate) fn select(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::All => Ok(logits.clone()),
            Self::Decode { start, len } => {
                let seq_len = logits.dim(1)?;
                if *start == 0 && *len == seq_len {
                    Ok(logits.clone())
                } else {
                    logits.narrow(1, *start, *len)
                }
            }
            Self::Indices {
                indices,
                batch,
                len,
            } => {
                let (logits_batch, seq_len, hidden) = logits.dims3()?;
                if logits_batch != *batch {
                    candle_core::bail!(
                        "logits selection batch mismatch: logits batch {logits_batch}, selection batch {batch}"
                    );
                }
                let indices = indices
                    .get(&logits.device().location())
                    .ok_or_else(|| candle_core::Error::msg("missing logits selection indices"))?;
                let flat = logits.reshape((logits_batch * seq_len, hidden))?;
                flat.index_select(indices, 0)?
                    .reshape((*batch, *len, hidden))
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum SupportedModality {
    Text,
    Audio,
    Vision,
    Video,
    Embedding,
}

impl Debug for SupportedModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "📝 Text"),
            Self::Audio => write!(f, "🔊 Audio"),
            Self::Vision => write!(f, "🖼️ Vision"),
            Self::Video => write!(f, "🎬 Video"),
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
    // UQFF writes force the whole model onto CPU, so the pipeline is not servable afterwards.
    pub loaded_for_uqff_write: bool,
}

impl GeneralMetadata {
    pub fn tok_env(&self) -> Option<TokEnv> {
        self.llg_factory.as_ref().map(|f| f.tok_env().clone())
    }
}

#[derive(Clone, Copy)]
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

    /// Start collecting activation statistics from live traffic on every ISQ-tracked layer.
    fn begin_calibration(&mut self) -> Result<()> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    fn calibration_status(&self) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    /// Requantize with the collected statistics and swap the layers into the live model.
    fn apply_calibration(
        &mut self,
        _save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }
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
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    /// Only None if it doesnt make sense for the model
    fn tokenizer(&self) -> Option<Arc<Tokenizer>>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    /// Destroy decode graphs at teardown, while the engine thread's cuTile modules are still loaded.
    fn cleanup_cuda_graphs(&self) {}
    fn get_metadata(&self) -> Arc<GeneralMetadata>;
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        None
    }
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
/// such as the multimodal model prompt prefixer.
#[derive(Clone)]
pub enum ModelCategory {
    Text,
    Multimodal {
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
            ModelCategory::Multimodal { .. } => {
                write!(f, "ModelCategory::Multimodal {{ prefixer: .. }}")
            }
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
            (Self::Multimodal { .. }, Self::Multimodal { .. }) => true,
            (Self::Audio, Self::Audio) => true,
            (Self::Speech, Self::Speech) => true,
            (Self::Diffusion, Self::Diffusion) => true,
            (Self::Embedding, Self::Embedding) => true,
            (
                Self::Text
                | Self::Multimodal { .. }
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
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_video(&self, _video_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

#[derive(Clone)]
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
    BlockGeneration {
        token_blocks: Vec<Vec<u32>>,
        denoise_time: std::time::Duration,
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
            Self::BlockGeneration {
                token_blocks,
                denoise_time,
            } => Ok(Self::BlockGeneration {
                token_blocks: vec![token_blocks[bs_idx].clone()],
                denoise_time: *denoise_time,
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
            Self::BlockGeneration { .. } => Ok(self.clone()),
        }
    }
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
    fn adapter_runtime(&self) -> Option<Arc<crate::DynamicLoraRuntime>> {
        None
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    fn attach_speculative(
        &mut self,
        _config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support speculative decoding attachment.")
    }

    /// Append pre-sampled token blocks (block-diffusion canvases) to the sequences via the
    /// standard per-token finalize path. Overridden by pipelines whose models emit
    /// `ForwardInputsResult::BlockGeneration`.
    async fn sample_block_gen(
        &self,
        _input_seqs: &mut [&mut Sequence],
        _token_blocks: Vec<Vec<u32>>,
        _denoise_times: Vec<std::time::Duration>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support block generation.")
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        _input_seqs: &mut [&mut Sequence],
        _logits: &[Tensor],
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        _metadata: Option<PagedAttentionMeta>,
    ) -> Result<bool, candle_core::Error> {
        Ok(false)
    }

    fn snapshot_paged_recurrent_prefix(
        &mut self,
        seq: &Sequence,
        prefix_cacher: &mut PrefixCacheManagerV2,
        block_size: usize,
        cached_tokens: usize,
    ) -> Result<(), candle_core::Error> {
        if cached_tokens == 0
            || !cached_tokens.is_multiple_of(block_size)
            || !self.cache().is_hybrid()
        {
            return Ok(());
        }
        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return Ok(());
        };

        let snapshots = self.cache().hybrid().snapshot_recurrent_state(slot_idx)?;
        if snapshots.is_empty() {
            return Ok(());
        }
        let adapter_key = adapter_generation_key(seq.adapter_generation());
        let block_hashes = compute_block_hashes(
            seq.get_toks(),
            block_size,
            seq.mm_features(),
            adapter_key.as_slice(),
        );
        let n_blocks = cached_tokens / block_size;
        if block_hashes.len() >= n_blocks {
            prefix_cacher.add_paged_recurrent_prefix(block_hashes[..n_blocks].to_vec(), snapshots);
        }
        Ok(())
    }

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
                if !is_prompt && !return_raw_logits {
                    crate::speculative::driver::clear_staged_speculative_tokens(input_seqs);
                }

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
                        self.get_metadata().sliding_window,
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
                        let l = l.expect("missing forward result");
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
                        let logits = logits
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
                            .collect::<Vec<_>>();
                        if is_prompt
                            || return_raw_logits
                            || !self
                                .try_sample_speculative_causal_gen(
                                    input_seqs,
                                    &logits,
                                    prefix_cacher,
                                    disable_eos_stop,
                                    rng.clone(),
                                    None,
                                )
                                .await?
                        {
                            self.sample_causal_gen(
                                input_seqs,
                                logits,
                                prefix_cacher,
                                disable_eos_stop,
                                rng,
                            )
                            .await?;
                        }
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
                    ForwardInputsResult::BlockGeneration { .. } => {
                        let mut denoise_times = Vec::with_capacity(logits.len());
                        let token_blocks = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::BlockGeneration {
                                    token_blocks,
                                    denoise_time,
                                } = r
                                else {
                                    unreachable!(
                                        "All results must have same type, `BlockGeneration`"
                                    )
                                };
                                denoise_times.push(denoise_time);
                                token_blocks
                                    .into_iter()
                                    .next()
                                    .expect("Must have at least 1 element.")
                            })
                            .collect::<Vec<_>>();
                        self.sample_block_gen(
                            input_seqs,
                            token_blocks,
                            denoise_times,
                            prefix_cacher,
                            disable_eos_stop,
                        )
                        .await?;
                    }
                }
                let end = Instant::now();
                exec_duration += end.duration_since(start);

                Ok(exec_duration)
            }
            CacheBackendMetadata::PagedAttention { mut metadata } => {
                let block_size = metadata.block_size;
                let speculative_metadata = metadata.clone();
                // For hybrid models, build state_indices tensor from sequences'
                // recurrent_state_idx so recurrent layers are active during forward.
                // Paged attention manages KV caches separately, but recurrent state
                // pool access still needs the indices tensor to be set.
                if self.cache().is_hybrid() {
                    let mut hybrid_cache = self.cache().hybrid();
                    let recurrent_device = hybrid_cache.caches.iter().find_map(|c| {
                        if let HybridLayerCache::Recurrent(pool) = c {
                            Some(pool.device().clone())
                        } else {
                            None
                        }
                    });
                    if let Some(device) = recurrent_device {
                        #[allow(clippy::cast_possible_truncation)]
                        let indices: Vec<u32> = input_seqs
                            .iter()
                            .filter_map(|seq| seq.recurrent_state_idx().map(|idx| idx as u32))
                            .collect();
                        if indices.len() == input_seqs.len() {
                            if let Ok(si) =
                                Tensor::from_vec(indices.clone(), (input_seqs.len(),), &device)
                            {
                                hybrid_cache.set_state_indices_with_host(Some(si), Some(indices));
                            }
                        }
                    }
                }

                let chunk_size = if is_prompt
                    && !return_raw_logits
                    && !self.get_metadata().is_xlora
                    && self.device().is_cuda()
                {
                    Some(DEFAULT_PAGED_PREFILL_CHUNK_SIZE)
                } else {
                    None
                };
                if chunk_size.is_some() {
                    self.get_processor()
                        .inputs_processor()
                        .prepare_for_paged_prompt_planning(
                            self.tokenizer(),
                            input_seqs,
                            &self.device(),
                            self.get_input_processor_config(),
                            Some(&mut metadata),
                        )
                        .map_err(|e| candle_core::Error::msg(e.to_string()))?;
                    for seq in input_seqs.iter_mut() {
                        seq.clip_prefix_cache_len_for_non_causal_mm_features(metadata.block_size);
                    }
                }
                let has_deferred_multimodal_prompt = input_seqs.iter().any(|seq| {
                    (seq.has_images() || seq.has_audios() || seq.has_videos())
                        && seq.mm_features().is_empty()
                });
                let chunk_plans = (!has_deferred_multimodal_prompt)
                    .then(|| {
                        chunk_size.map(|chunk_size| {
                            input_seqs
                                .iter()
                                .map(|seq| {
                                    build_prompt_chunk_plan(
                                        seq.get_toks().len(),
                                        seq.prefix_cache_len(),
                                        chunk_size,
                                        seq.mm_features(),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .flatten();
                let should_chunk = chunk_plans
                    .as_ref()
                    .is_some_and(|plans| plans.iter().any(|plan| plan.len() > 1));
                let (logits, raw_out_logits, embedding_logits, mut exec_duration) = {
                    let inputs_iter = if let (Some(chunk_plans), true) = (chunk_plans, should_chunk)
                    {
                        let originals = input_seqs
                            .iter()
                            .map(|seq| (seq.get_toks().to_vec(), seq.prefix_cache_len()))
                            .collect::<Vec<_>>();
                        let mut plan_indices = vec![0usize; chunk_plans.len()];
                        let mut inputs = Vec::new();
                        while plan_indices
                            .iter()
                            .zip(chunk_plans.iter())
                            .any(|(plan_idx, plan)| *plan_idx < plan.len())
                        {
                            let attention_policy = plan_indices
                                .iter()
                                .zip(chunk_plans.iter())
                                .find_map(|(plan_idx, plan)| plan.get(*plan_idx))
                                .expect("at least one chunk plan is active")
                                .attention_policy;
                            let active_indices = plan_indices
                                .iter()
                                .zip(chunk_plans.iter())
                                .enumerate()
                                .filter_map(|(idx, (plan_idx, plan))| {
                                    plan.get(*plan_idx)
                                        .filter(|chunk| chunk.attention_policy == attention_policy)
                                        .map(|_| idx)
                                })
                                .collect::<Vec<_>>();

                            let mut recurrent_boundaries = Vec::new();
                            for &seq_idx in &active_indices {
                                let chunk = chunk_plans[seq_idx][plan_indices[seq_idx]];
                                let seq = &mut input_seqs[seq_idx];
                                seq.set_prefix_cache_len(chunk.start);
                                seq.set_prefill_toks(originals[seq_idx].0[..chunk.end].to_vec());
                                if chunk.end % block_size == 0 {
                                    recurrent_boundaries.push((seq_idx, chunk.end));
                                }
                            }

                            let mut chunk_metadata = metadata.clone();
                            chunk_metadata.prompt_chunk_attention_policy = attention_policy;
                            chunk_metadata.is_final_prompt_chunk = active_indices
                                .iter()
                                .all(|&idx| plan_indices[idx] + 1 == chunk_plans[idx].len());
                            let mut active_input_seqs = input_seqs
                                .iter_mut()
                                .enumerate()
                                .filter_map(|(idx, seq)| {
                                    active_indices.contains(&idx).then_some(&mut **seq)
                                })
                                .collect::<Vec<_>>();
                            chunk_metadata.set_noncausal_mm_context(active_input_seqs.as_slice());
                            let mut processed =
                                self.get_processor().inputs_processor().process_inputs(
                                    self.tokenizer(),
                                    active_input_seqs.as_mut_slice(),
                                    is_prompt,
                                    self.get_metadata().is_xlora,
                                    &self.device(),
                                    self.get_metadata().no_kv_cache,
                                    None,
                                    return_raw_logits,
                                    self.get_metadata().sliding_window,
                                    self.get_input_processor_config(),
                                    Some(chunk_metadata),
                                    self.device_mapper(),
                                );
                            drop(active_input_seqs);
                            if let Ok(processed) = &mut processed {
                                for seq_idx in &mut processed.seq_indices {
                                    *seq_idx = active_indices[*seq_idx];
                                }
                            }
                            inputs.push((processed, recurrent_boundaries));
                            for &seq_idx in &active_indices {
                                plan_indices[seq_idx] += 1;
                            }
                        }
                        for (seq, (tokens, prefix_len)) in
                            input_seqs.iter_mut().zip(originals.iter())
                        {
                            seq.set_prefix_cache_len(*prefix_len);
                            seq.set_prefill_toks(tokens.clone());
                        }
                        inputs
                    } else {
                        metadata.set_noncausal_mm_context(input_seqs);
                        vec![(
                            self.get_processor().inputs_processor().process_inputs(
                                self.tokenizer(),
                                input_seqs,
                                is_prompt,
                                self.get_metadata().is_xlora,
                                &self.device(),
                                self.get_metadata().no_kv_cache,
                                None,
                                return_raw_logits,
                                self.get_metadata().sliding_window,
                                self.get_input_processor_config(),
                                Some(metadata),
                                self.device_mapper(),
                            ),
                            Vec::new(),
                        )]
                    };

                    let mut logits = vec![None; input_seqs.len()];
                    let len_inputs = inputs_iter.len();
                    let mut raw_out_logits = vec![vec![None; len_inputs]; input_seqs.len()];
                    let mut embedding_logits = vec![None; input_seqs.len()];

                    let mut exec_duration = Duration::ZERO;
                    for (i, (inputs, recurrent_boundaries)) in inputs_iter.into_iter().enumerate() {
                        let InputProcessorOutput {
                            inputs,
                            seq_indices,
                        } = inputs.map_err(candle_core::Error::msg)?;

                        let start = Instant::now();
                        let raw_logits = self.forward_inputs(inputs, return_raw_logits)?;
                        let end = Instant::now();
                        exec_duration += end.duration_since(start);

                        for (seq_idx, end) in recurrent_boundaries {
                            self.snapshot_paged_recurrent_prefix(
                                &*input_seqs[seq_idx],
                                prefix_cacher,
                                block_size,
                                end,
                            )?;
                        }

                        for (logit_idx, seq_idx) in seq_indices.into_iter().enumerate() {
                            if let ForwardInputsResult::RawLogits { logits } = &raw_logits {
                                raw_out_logits[seq_idx][i] =
                                    Some(logits.i(logit_idx)?.to_device(&Device::Cpu)?);
                            } else if let ForwardInputsResult::Embeddings { embeddings } =
                                &raw_logits
                            {
                                embedding_logits[seq_idx] =
                                    Some(embeddings.i(logit_idx)?.to_device(&Device::Cpu)?);
                            } else {
                                logits[seq_idx] = Some(raw_logits.index_bs(logit_idx)?);
                            }
                        }
                    }
                    (logits, raw_out_logits, embedding_logits, exec_duration)
                };

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
                        let l = l.expect("missing forward result");
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
                        let logits = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::CausalGeneration { logits } = r
                                else {
                                    unreachable!("All results must have same type")
                                };
                                logits
                            })
                            .collect::<Vec<_>>();
                        if is_prompt
                            || return_raw_logits
                            || !self
                                .try_sample_speculative_causal_gen(
                                    input_seqs,
                                    &logits,
                                    prefix_cacher,
                                    disable_eos_stop,
                                    rng.clone(),
                                    Some(speculative_metadata),
                                )
                                .await?
                        {
                            self.sample_causal_gen(
                                input_seqs,
                                logits,
                                prefix_cacher,
                                disable_eos_stop,
                                rng,
                            )
                            .await?;
                        }
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
                    ForwardInputsResult::BlockGeneration { .. } => {
                        let mut denoise_times = Vec::with_capacity(logits.len());
                        let token_blocks = logits
                            .into_iter()
                            .map(|r| {
                                #[allow(irrefutable_let_patterns)]
                                let ForwardInputsResult::BlockGeneration {
                                    token_blocks,
                                    denoise_time,
                                } = r
                                else {
                                    unreachable!(
                                        "All results must have same type, `BlockGeneration`"
                                    )
                                };
                                denoise_times.push(denoise_time);
                                token_blocks
                                    .into_iter()
                                    .next()
                                    .expect("Must have at least 1 element.")
                            })
                            .collect::<Vec<_>>();
                        self.sample_block_gen(
                            input_seqs,
                            token_blocks,
                            denoise_times,
                            prefix_cacher,
                            disable_eos_stop,
                        )
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
    LogitsSelection::from_context_lens(logits, &context_lens, &[logits.device().clone()])?
        .select(logits)
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
