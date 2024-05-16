use super::cache_manager::DefaultCacheManager;
use super::{
    get_model_paths, get_xlora_paths, CacheManager, GeneralMetadata, Loader, ModelInputs,
    ModelKind, ModelPaths, Pipeline, TokenSource, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::lora::Ordering;
use crate::pipeline::chat_template::calculate_eos_tokens;
use crate::pipeline::Cache;
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, load_preload_adapters};
use crate::xlora_models::NonGranularState;
use crate::{deserialize_chat_template, do_sample, get_mut_arcmutex, get_paths, DeviceMapMetadata};
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use candle_core::quantized::{
    gguf_file::{self, Value as GgufValue},
    GgmlDType,
};
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: GeneralMetadata,
}

pub struct GGUFLoader {
    model_id: String,
    config: GGUFSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
}

#[derive(Debug)]
enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
}

impl FromStr for GGUFArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "llama" => Ok(GGUFArchitecture::Llama),
            "mpt" => Ok(GGUFArchitecture::Mpt),
            "gptneox" => Ok(GGUFArchitecture::Gptneox),
            "gptj" => Ok(GGUFArchitecture::Gptj),
            "gpt2" => Ok(GGUFArchitecture::Gpt2),
            "bloom" => Ok(GGUFArchitecture::Bloom),
            "falcon" => Ok(GGUFArchitecture::Falcon),
            "mamba" => Ok(GGUFArchitecture::Mamba),
            "rwkv" => Ok(GGUFArchitecture::Rwkv),
            "phi2" => Ok(GGUFArchitecture::Phi2),
            "phi3" => Ok(GGUFArchitecture::Phi3),
            a => Err(format!("Unknown GGUF architecture `{a}`")),
        }
    }
}

#[derive(Clone, Copy, Default)]
/// A config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub repeat_last_n: usize,
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    config: GGUFSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
}

impl GGUFLoaderBuilder {
    pub fn new(
        config: GGUFSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
    ) -> Self {
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::QuantizedGGUF,
            quantized_filename,
            quantized_model_id,
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
        self.kind = ModelKind::XLoraGGUF;
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = ModelKind::LoraGGUF;
        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
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
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        config: GGUFSpecificConfig,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
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
        }
    }
}

fn parse_gguf_value(value: &GgufValue) -> String {
    match value {
        GgufValue::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::F32(x) => x.to_string(),
        GgufValue::F64(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::String(x) => x.to_string(),
        GgufValue::U8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::U64(x) => x.to_string(),
    }
}

impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id,
            self.quantized_filename,
            silent
        );
        self.load_model_from_path(&paths?, _dtype, device, silent, mapper, in_situ_quant)
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }
        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;
        let arch: GGUFArchitecture = model.metadata["general.architecture"]
            .to_string()
            .unwrap()
            .parse()
            .map_err(anyhow::Error::msg)?;

        info!("Model config:");
        let mut sorted_keys = model.metadata.keys().collect::<Vec<_>>();
        sorted_keys.sort();
        for name in sorted_keys {
            if !name.contains("tokenizer") {
                let value = parse_gguf_value(&model.metadata[name]);
                println!("{name}: {}", value);
            }
        }

        let mut is_lora = false;
        let model = match self.kind {
            ModelKind::QuantizedGGUF => match arch {
                GGUFArchitecture::Llama => {
                    Model::Llama(QLlama::from_gguf(model, &mut file, device, mapper)?)
                }
                GGUFArchitecture::Phi2 => {
                    Model::Phi2(QPhi::from_gguf(model, &mut file, device, mapper)?)
                }
                GGUFArchitecture::Phi3 => {
                    Model::Phi3(QPhi3::from_gguf(model, &mut file, device, mapper)?)
                }
                a => bail!("Unsupported architecture `{a:?}`"),
            },
            ModelKind::XLoraGGUF => {
                let vb = from_mmaped_safetensors(
                    vec![paths.get_classifier_path().as_ref().unwrap().to_path_buf()],
                    paths
                        .get_adapter_filenames()
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|(_, x)| (*x).to_owned())
                        .collect::<Vec<_>>(),
                    DType::F32,
                    device,
                    silent,
                )?;

                match arch {
                    GGUFArchitecture::Llama => Model::XLoraLlama(XLoraQLlama::from_gguf(
                        model,
                        &mut file,
                        device,
                        paths.get_adapter_configs().as_ref().unwrap(),
                        &vb,
                        paths.get_ordering().as_ref().unwrap(),
                        Some(paths.get_classifier_config().as_ref().unwrap().clone()),
                        mapper,
                        &load_preload_adapters(
                            paths.get_lora_preload_adapter_info(),
                            DType::F32,
                            device,
                            silent,
                        )?,
                    )?),
                    GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::from_gguf(
                        model,
                        &mut file,
                        device,
                        paths.get_adapter_configs().as_ref().unwrap(),
                        &vb,
                        paths.get_ordering().as_ref().unwrap(),
                        Some(paths.get_classifier_config().as_ref().unwrap().clone()),
                        mapper,
                        &load_preload_adapters(
                            paths.get_lora_preload_adapter_info(),
                            DType::F32,
                            device,
                            silent,
                        )?,
                    )?),
                    a => bail!("Unsupported architecture for GGUF X-LoRA `{a:?}`"),
                }
            }
            ModelKind::LoraGGUF => {
                is_lora = true;
                let vb = from_mmaped_safetensors(
                    vec![],
                    paths
                        .get_adapter_filenames()
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|(_, x)| (*x).to_owned())
                        .collect::<Vec<_>>(),
                    DType::F32,
                    device,
                    silent,
                )?;

                match arch {
                    GGUFArchitecture::Llama => Model::XLoraLlama(XLoraQLlama::from_gguf(
                        model,
                        &mut file,
                        device,
                        paths.get_adapter_configs().as_ref().unwrap(),
                        &vb,
                        paths.get_ordering().as_ref().unwrap(),
                        None,
                        mapper,
                        &load_preload_adapters(
                            paths.get_lora_preload_adapter_info(),
                            DType::F32,
                            device,
                            silent,
                        )?,
                    )?),
                    GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::from_gguf(
                        model,
                        &mut file,
                        device,
                        paths.get_adapter_configs().as_ref().unwrap(),
                        &vb,
                        paths.get_ordering().as_ref().unwrap(),
                        None,
                        mapper,
                        &load_preload_adapters(
                            paths.get_lora_preload_adapter_info(),
                            DType::F32,
                            device,
                            silent,
                        )?,
                    )?),
                    a => bail!("Unsupported architecture for GGUF LoRA `{a:?}`"),
                }
            }
            _ => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename())?;

        let (chat_template, gen_conf) = deserialize_chat_template!(paths, self);

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::Phi2(ref p) => p.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::Phi3(ref p) => p.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
        };
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let is_xlora = match &model {
            Model::Llama(_) | Model::Phi2(_) | Model::Phi3(_) => false,
            Model::XLoraLlama(_) | Model::XLoraPhi3(_) => !is_lora,
        };
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.lock().len(),
            Model::Phi2(ref model) => model.cache.lock().len(),
            Model::XLoraLlama(ref model) => model.cache.lock().len(),
            Model::Phi3(ref model) => model.cache.lock().len(),
            Model::XLoraPhi3(ref model) => model.cache.lock().len(),
        };
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tok_trie: tok_trie.clone(),
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
            metadata: GeneralMetadata {
                max_seq_len,
                repeat_last_n: self.config.repeat_last_n,
                tok_trie,
                has_no_kv_cache: self.no_kv_cache,
                is_xlora,
                num_hidden_layers,
                eos_tok: eos,
                is_lora,
            },
        })))
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

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn forward_inputs(
        &mut self,
        ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
            position_ids: _, // NOTE(EricLBuehler): ignore, it is for phi3
        }: ModelInputs,
    ) -> Result<Tensor, candle_core::Error> {
        match self.model {
            Model::Llama(ref mut model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
            ),
            Model::Phi2(ref mut model) => model.forward(&input_ids, &seqlen_offsets, context_lens),
            Model::XLoraLlama(ref mut model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
            ),
            Model::Phi3(ref mut model) => model.forward(&input_ids, &seqlen_offsets),
            Model::XLoraPhi3(ref mut model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
            ),
        }
    }
    async fn sample(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        do_sample!(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::Phi2(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::Phi3(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn re_isq_model(&mut self, _dtype: GgmlDType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
    fn get_metadata(&self) -> &GeneralMetadata {
        &self.metadata
    }
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
    }
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool) {
        DefaultCacheManager.set_none_cache(self, modify_draft_cache);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &Cache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::Phi2(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
            Model::Phi3(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
        }
    }
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        if !self.metadata.is_lora {
            anyhow::bail!("Cannot activate adapters non-LoRA models.")
        }
        match self.model {
            Model::Llama(_) => unreachable!(),
            Model::Phi2(_) => unreachable!(),
            Model::Phi3(_) => unreachable!(),
            Model::XLoraLlama(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            Model::XLoraPhi3(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
        }
    }
}
