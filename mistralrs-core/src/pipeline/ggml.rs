use super::llg::build_llg_factory;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, QuantizationKind, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::device_map::DeviceMapper;
use crate::kv_cache::FullCacheManager;
use crate::lora::Ordering;
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::get_chat_template;
use crate::pipeline::inputs_processor::DEFAULT_PROMPT_CHUNK_SIZE;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::model_config as ModelConfig;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths, DeviceMapSetting, PagedAttentionConfig, Pipeline, Topology,
    TryIntoDType, DEBUG,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama, utils::tokens::get_token,
    xlora_models::XLoraQLlama,
};
use anyhow::Result;
use candle_core::quantized::ggml_file;
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::num::{NonZero, NonZeroUsize};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

enum Model {
    Llama(QLlama),
    XLoraLlama(Box<XLoraQLlama>),
}

pub struct GGMLPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
}

/// A loader for a GGML model.
pub struct GGMLLoader {
    model_id: String,
    config: GGMLSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    jinja_explicit: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Clone, Default)]
/// Config for a GGML loader.
pub struct GGMLSpecificConfig {
    pub gqa: usize,
    pub prompt_chunksize: Option<NonZeroUsize>,
    pub topology: Option<Topology>,
}

#[derive(Default)]
/// A builder for a GGML loader.
pub struct GGMLLoaderBuilder {
    model_id: Option<String>,
    config: GGMLSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
    jinja_explicit: Option<String>,
}

impl GGMLLoaderBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: GGMLSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Ggml,
        };

        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind,
            quantized_filename,
            quantized_model_id,
            no_kv_cache,
            jinja_explicit,
            ..Default::default()
        }
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
        self.kind = (AdapterKind::XLora, QuantizationKind::Ggml).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Ggml).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGMLLoader {
            model_id: self.model_id.unwrap(),
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filename: Some(self.quantized_filename),
            quantized_model_id: Some(self.quantized_model_id),
            jinja_explicit: self.jinja_explicit,
            lora_adapter_ids: None,
        })
    }
}

impl GGMLLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        config: GGMLSpecificConfig,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            id
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.as_ref().unwrap().base_model_id
            );
            xlora_order.as_ref().unwrap().base_model_id.clone()
        };
        Self {
            model_id,
            config,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            tokenizer_json,
            kind,
            tgt_non_granular_index,
            jinja_explicit,
            lora_adapter_ids: None,
        }
    }
}

impl Loader for GGMLLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGML model. This will not do anything."
            );
        }

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for diffusion models.")
        }

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for GGML models, disabling it.");

            paged_attn_config = None;
        }

        // Apply default prompt size here
        let prompt_chunksize = self
            .config
            .prompt_chunksize
            .unwrap_or(DEFAULT_PROMPT_CHUNK_SIZE.try_into().unwrap())
            .get();

        info!("Prompt chunk size is {prompt_chunksize}.",);

        info!(
            "Loading model `{}` on {}.",
            self.get_id(),
            device.device_pretty_repr()
        );

        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = ggml_file::Content::read(&mut file, device)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;

        info!("Model config: {:?}", model.hparams);

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            let mut tensors = Vec::new();
            for (name, t) in &model.tensors {
                tensors.push(format!(
                    "name = `{name}`, shape = {:?}, dtype = {:?}",
                    t.shape().clone(),
                    t.dtype(),
                ));
            }
            fs::write(
                "mistralrs_ggml_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!("Debug is enabled, wrote the names and information about each tensor to `mistralrs_ggml_tensors.txt`.");
        }

        let _ = if paged_attn_config.is_none() {
            warn!("GGML does not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());
        let internal_dtype = dtype.try_into_dtype(&[device]).unwrap();

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGML((model, self.config.gqa, internal_dtype).into());

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::new(quant, adapter)
        };

        // Config into model:
        // NOTE: No architecture to infer like GGUF, Llama model is implicitly matched
        let model = match self.kind {
            ModelKind::GgufQuantized { .. } => Model::Llama(QLlama::try_from(model_config)?),
            ModelKind::GgufAdapter { .. } => {
                Model::XLoraLlama(Box::new(XLoraQLlama::try_from(model_config)?))
            }
            _ => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths.get_gen_conf_filename().map(|f| {
            serde_json::from_str(&fs::read_to_string(f).unwrap())
                .expect("bos_token_id/eos_token_id missing in generation_config.json")
        });
        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            None,
        );

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
        };
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.normal().0.len(),
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
        };
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(GGMLPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                prompt_chunksize: Some(NonZero::new(prompt_chunksize).unwrap()),
                model_metadata: None,
            }),
        })))
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id,
            Some(vec![self.quantized_filename.as_ref().unwrap().clone()]),
            silent,
            false // Never loading UQFF
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

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(&self.model_id)
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGMLPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGMLPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGMLPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        FullCacheManager.clone_in_cache(self, seqs, false)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        FullCacheManager.clone_out_cache(self, seqs, false)
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,

        load_preallocated_cache: bool,
    ) {
        FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, load_preallocated_cache);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
        }
    }
}

impl MetadataMixin for GGMLPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
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
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for GGMLPipeline {
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
            context_lens,
            position_ids: _,    // NOTE(EricLBuehler): ignore, it is for phi3
            paged_attn_meta: _, // NOTE(EricLBuehler): ignore it for ggml
            flash_meta,         // NOTE(EricLBuehler): ignore it for ggml dequant into f32
            flash_meta_full,    // NOTE(EricLBuehler): ignore it for ggml dequant into f32
        } = *inputs.downcast().expect("Downcast failed.");
        let logits = match self.model {
            Model::Llama(ref model) => {
                model.forward(&input_ids, &seqlen_offsets, context_lens, None)?
            }
            Model::XLoraLlama(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
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
        prefix_cacher: &mut PrefixCacheManagerV2,
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
impl AnyMoePipelineMixin for GGMLPipeline {}
