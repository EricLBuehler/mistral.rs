use super::cache_manager::{FullCacheManager, NormalCacheManager};
use super::llg::build_tok_env;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, QuantizationKind,
    TokenSource, XLoraPaths,
};
use super::{
    AdapterActivationMixin, AnyMoePipelineMixin, CacheManagerMixin, EitherCache,
    ForwardInputsResult, IsqPipelineMixin, MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::gguf::{
    get_gguf_chat_template, {convert_gguf_to_hf_tokenizer, GgufTokenizerConversion},
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::lora::Ordering;
use crate::paged_attention::{
    calculate_cache_config, AttentionImplementation, CacheEngine, ModelConfigLike,
};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkTok, GenerationConfig};
use crate::pipeline::get_chat_template;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::ChatTemplate;
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::model_config as ModelConfig;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths_gguf, DeviceMapMetadata, LocalModelPaths, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_qwen2::ModelWeights as QQwen2,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use either::Either;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
    Starcoder2(QStarcoder2),
    Qwen2(QQwen2),
}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub prompt_batchsize: Option<NonZeroUsize>,
    pub topology: Option<Topology>,
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` is the model ID where you can find a
    /// `tokenizer_config.json` file. If the `chat_template` is specified, then it will be treated as a
    /// path and used over remote files, removing all remote accesses.
    ///
    /// NOTE: Until v0.4.0, you should make sure to call `.with_no_kv_cache` if applicable.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            ..Default::default()
        }
    }

    // TODO(EricLBuehler): in 0.4.0 we can move this into the config
    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            kind,
            tgt_non_granular_index,
            config,
        }
    }
}

struct ContentConfig {
    hidden_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
}

#[allow(clippy::cast_possible_truncation)]
impl<'a, R: std::io::Seek + std::io::Read> From<&Content<'a, R>> for ContentConfig {
    fn from(value: &Content<'a, R>) -> Self {
        let metadata = value.get_metadata();
        let arch = metadata["general.architecture"].to_string().unwrap();
        Self {
            hidden_size: metadata[&format!("{arch}.embedding_length")]
                .to_u64()
                .unwrap() as usize,
            num_attn_heads: metadata[&format!("{arch}.attention.head_count")]
                .to_u64()
                .unwrap() as usize,
            num_kv_heads: metadata[&format!("{arch}.attention.head_count_kv")]
                .to_u64()
                .unwrap() as usize,
            num_layers: metadata[&format!("{arch}.block_count")].to_u64().unwrap() as usize,
        }
    }
}

impl ModelConfigLike for ContentConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
}

impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent
        );
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }

        // Otherwise, the device mapper will print it
        if mapper.is_dummy()
            && (self.config.topology.is_none()
                || self
                    .config
                    .topology
                    .as_ref()
                    .is_some_and(|t| t.is_dummy_device_map()))
        {
            info!(
                "Loading model `{}` on {}.",
                self.get_id(),
                device.device_pretty_repr()
            );
        } else if paged_attn_config.is_some() {
            warn!("Device mapping or device topology and PagedAttention are incompatible, disabling PagedAttention.");
            paged_attn_config = None;
        }

        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers = readers.iter_mut().collect::<Vec<_>>();

        let model = Content::from_readers(&mut readers)?;
        if !silent {
            model.print_metadata()?;
        }
        let arch = model.arch();

        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&model)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Handles the case where `self.chat_template` is either a file or a
        // chat_template string literal. As a file, `self.chat_template`
        // overrides the file from `paths`.
        let (chat_template_file, chat_template_literal) =
            match (paths.get_template_filename(), self.chat_template.clone()) {
                // If chat_template is file, it overrides.
                (_, Some(l)) if Path::new(&l).exists() => (Some(Path::new(&l).to_path_buf()), None),
                // Otherwise, use the template file + chat_template literal (if provided).
                (Some(f), l_opt) => (Some(Path::new(f).to_path_buf()), l_opt),
                (None, Some(l)) => (None, Some(l)),
                // Only load gguf chat template if there is nothing else
                (None, None) => (None, get_gguf_chat_template(&model)?),
            };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let paged_attn_config = if matches!(self.kind, ModelKind::GgufAdapter { .. }) {
            warn!("Adapter models do not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let model_config_metadata: ContentConfig = (&model).into();

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGUF(
                model,
                (device, mapper, self.config.topology.as_ref()).into(),
                if paged_attn_config.is_some() {
                    AttentionImplementation::PagedAttention
                } else {
                    AttentionImplementation::Eager
                },
            );

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::builder()
                .quant(quant)
                .and_adapter(adapter)
                .build()
        };

        // Config into model:
        let model = match self.kind {
            ModelKind::GgufQuantized { .. } => match arch {
                GGUFArchitecture::Llama => Model::Llama(QLlama::try_from(model_config)?),
                GGUFArchitecture::Phi2 => Model::Phi2(QPhi::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::Phi3(QPhi3::try_from(model_config)?),
                GGUFArchitecture::Starcoder2 => {
                    Model::Starcoder2(QStarcoder2::try_from(model_config)?)
                }
                GGUFArchitecture::Qwen2 => Model::Qwen2(QQwen2::try_from(model_config)?),
                a => bail!("Unsupported architecture `{a:?}` for GGUF"),
            },
            ModelKind::GgufAdapter { adapter, .. } => match arch {
                GGUFArchitecture::Llama => Model::XLoraLlama(XLoraQLlama::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::try_from(model_config)?),
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            let model_config: &dyn ModelConfigLike = &model_config_metadata;
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.mem_cpu,
                paged_attn_config.block_size,
                DType::F32,
                model_config,
                device,
            )?;
            let cache_engine = CacheEngine::new(model_config, &cache_config, DType::F32, device)?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let gen_conf: Option<GenerationConfig> = paths.get_gen_conf_filename().map(|f| {
            serde_json::from_str(&fs::read_to_string(f).unwrap())
                .expect("bos_token_id/eos_token_id missing in generation_config.json")
        });
        let mut chat_template = get_chat_template(
            paths,
            &paths
                .get_chat_template_json()
                .as_ref()
                .map(|x| x.to_string_lossy().to_string()),
            &chat_template_file.map(|p| p.to_string_lossy().to_string()),
            chat_template_literal,
        );

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::Phi2(ref p) => p.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::Phi3(ref p) => p.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
            Model::Starcoder2(ref p) => p.max_seq_len,
            Model::Qwen2(ref p) => p.max_seq_len,
        };
        let tok_env = build_tok_env(tokenizer.clone());
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.normal().0.len(),
            Model::Phi2(ref model) => model.cache.normal().0.len(),
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
            Model::Phi3(ref model) => model.cache.normal().0.len(),
            Model::XLoraPhi3(ref model) => model.cache.full().lock().len(),
            Model::Starcoder2(ref model) => model.cache.normal().0.len(),
            Model::Qwen2(ref model) => model.cache.normal().0.len(),
        };

        if chat_template.bos_token.is_none() && bos.is_some() {
            chat_template.bos_token = Some(BeginEndUnkTok(Either::Left(bos.unwrap())));
        }
        if chat_template.eos_token.is_none() && eos.is_some() {
            chat_template.eos_token = Some(BeginEndUnkTok(Either::Left(eos.unwrap())));
        }
        if chat_template.unk_token.is_none() && unk.is_some() {
            chat_template.unk_token = Some(BeginEndUnkTok(Either::Left(unk.unwrap())));
        }

        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self
                .model_id
                .clone()
                .unwrap_or(self.quantized_model_id.clone()),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                tok_env: Some(tok_env),
                has_no_kv_cache: self.no_kv_cache,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: DType::F32,
                sliding_window: None,
                cache_config,
                cache_engine,
                prompt_batchsize: self.config.prompt_batchsize,
                model_metadata: Some(Arc::new(model_config_metadata)),
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        } else {
            NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            );
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::Phi2(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
            Model::Phi3(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
            Model::Starcoder2(ref model) => &model.cache,
            Model::Qwen2(ref model) => &model.cache,
        }
    }
}

impl AdapterActivationMixin for GGUFPipeline {
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        let is_lora = self.metadata.kind.is_adapted_and(|a| a.is_lora());
        if !is_lora {
            anyhow::bail!("Activating adapters is only supported for models fine-tuned with LoRA.")
        }

        match self.model {
            Model::XLoraLlama(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            Model::XLoraPhi3(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            _ => unreachable!(),
        }
    }
}

impl MetadataMixin for GGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::Phi2(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::Phi3(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
            Model::Starcoder2(ref model) => model.device.clone(),
            Model::Qwen2(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
}

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
            position_ids: _, // NOTE(EricLBuehler): ignore, it is for phi3
            mut paged_attn_meta,
            flash_meta,
            flash_meta_full,
        } = *inputs.downcast().expect("Downcast failed.");
        let paged_attn_meta = match (
            self.get_metadata().cache_engine.as_ref(),
            &mut paged_attn_meta,
        ) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            (None, None) => None,
        };
        let logits = match self.model {
            Model::Llama(ref model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
                paged_attn_meta,
            )?,
            Model::Phi2(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
            }
            Model::XLoraLlama(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            Model::Phi3(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
            }
            Model::XLoraPhi3(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            Model::Starcoder2(ref model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                paged_attn_meta,
            )?,
            Model::Qwen2(ref model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
                paged_attn_meta,
            )?,
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

// TODO
impl AnyMoePipelineMixin for GGUFPipeline {}
