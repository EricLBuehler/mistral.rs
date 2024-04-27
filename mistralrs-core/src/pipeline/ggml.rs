use super::{
    calculate_inputs_completion, calculate_inputs_prompt_batched, get_model_paths, get_xlora_paths,
    Loader, ModelInputs, ModelKind, ModelPaths, Pipeline, TokenSource, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::models::Cache;
use crate::pipeline::chat_template::calculate_eos_tokens;
use crate::pipeline::ChatTemplate;
use crate::utils::varbuilder_utils::from_mmaped_safetensors;
use crate::xlora_models::{NonGranularState, XLoraConfig};
use crate::{deserialize_chat_template, get_paths, DeviceMapMetadata};
use crate::{
    models::quantized_llama::ModelWeights as QLlama, sequence::Sequence, utils::tokens::get_token,
    xlora_models::XLoraModelWeights as XLoraQLlama,
};
use anyhow::Result;
use candle_core::quantized::{ggml_file, GgmlDType};
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_lora::{LoraConfig, Ordering};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use tracing::{info, warn};

enum Model {
    Llama(QLlama),
    XLoraLlama(XLoraQLlama),
}

pub struct MistralModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    template_filename: P,
    filenames: Vec<P>,
    xlora_adapter_filenames: Option<Vec<(String, P)>>,
    xlora_adapter_configs: Option<Vec<(String, LoraConfig)>>,
    classifier_path: Option<P>,
    classifier_config: Option<XLoraConfig>,
    xlora_ordering: Option<Ordering>,
}

impl ModelPaths for MistralModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>> {
        &self.xlora_adapter_filenames
    }
    fn get_adapter_configs(&self) -> &Option<Vec<(String, LoraConfig)>> {
        &self.xlora_adapter_configs
    }
    fn get_classifier_config(&self) -> &Option<XLoraConfig> {
        &self.classifier_config
    }
    fn get_classifier_path(&self) -> &Option<PathBuf> {
        &self.classifier_path
    }
    fn get_ordering(&self) -> &Option<Ordering> {
        &self.xlora_ordering
    }
    fn get_template_filename(&self) -> &PathBuf {
        &self.template_filename
    }
}

pub struct GGMLPipeline {
    model: Model,
    config: GGMLSpecificConfig,
    tokenizer: Arc<Tokenizer>,
    tok_trie: TokTrie,
    no_kv_cache: bool,
    chat_template: ChatTemplate,
    model_id: String,
    eos_tok: Vec<u32>,
    non_granular_state: Option<NonGranularState>,
    is_lora: bool,
}

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
}

#[derive(Clone, Copy, Default)]
/// Config for a GGML loader.
pub struct GGMLSpecificConfig {
    pub repeat_last_n: usize,
    pub gqa: usize,
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
}

impl GGMLLoaderBuilder {
    pub fn new(
        config: GGMLSpecificConfig,
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
            kind: ModelKind::QuantizedGGML,
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
        self.kind = ModelKind::XLoraGGML;
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = ModelKind::LoraGGML;
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
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

impl Loader for GGMLLoader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        silent: bool,
    ) -> Result<Box<dyn ModelPaths>> {
        get_paths!(
            MistralModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id,
            self.quantized_filename,
            silent
        )
    }

    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Box<Mutex<dyn Pipeline + Send + Sync>>> {
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }
        if !mapper.is_dummy() {
            warn!("GGML models do not support device mapping. Device mapping will not work. Please consider using a GGUF model.");
        }

        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = ggml_file::Content::read(&mut file, device)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;

        let mut is_lora = false;
        let model = match self.kind {
            ModelKind::QuantizedGGML => Model::Llama(QLlama::from_ggml(model, self.config.gqa)?),
            ModelKind::XLoraGGML => {
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

                Model::XLoraLlama(XLoraQLlama::from_ggml(
                    model,
                    self.config.gqa,
                    paths.get_adapter_configs().as_ref().unwrap(),
                    &vb,
                    paths.get_ordering().as_ref().unwrap(),
                    Some(paths.get_classifier_config().as_ref().unwrap().clone()),
                )?)
            }
            ModelKind::LoraGGML => {
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

                Model::XLoraLlama(XLoraQLlama::from_ggml(
                    model,
                    self.config.gqa,
                    paths.get_adapter_configs().as_ref().unwrap(),
                    &vb,
                    paths.get_ordering().as_ref().unwrap(),
                    None,
                )?)
            }
            _ => unreachable!(),
        };

        let tokenizer =
            Tokenizer::from_file(paths.get_tokenizer_filename()).map_err(anyhow::Error::msg)?;

        let chat_template: ChatTemplate = deserialize_chat_template!(paths, self);

        Ok(Box::new(Mutex::new(GGMLPipeline {
            model,
            config: self.config,
            eos_tok: calculate_eos_tokens(&chat_template, &tokenizer),
            tok_trie: build_tok_trie(tokenizer.clone()),
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template,
            model_id: self.model_id.clone(),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            is_lora,
        })))
    }

    fn get_id(&self) -> &str {
        self.xlora_model_id.as_deref().unwrap_or(&self.model_id)
    }

    fn get_kind(&self) -> ModelKind {
        self.kind
    }
}

impl Pipeline for GGMLPipeline {
    fn forward_prompt(&mut self, input_toks: &mut Sequence) -> Result<Tensor, candle_core::Error> {
        let device = self.device().clone();
        let chunk_size = self.get_prefill_chunk_size();

        let batches = calculate_inputs_prompt_batched(input_toks, &device, chunk_size).unwrap();
        let mut xs = None;
        for batch in batches {
            xs = Some(match self.model {
                Model::Llama(ref mut model) => model.forward(
                    &batch.input,
                    &batch.positions,
                    batch.positions_kernel,
                    batch.context_lens,
                ),
                Model::XLoraLlama(ref mut model) => model.forward(
                    &batch.input,
                    &batch.input,
                    &batch.positions,
                    &batch.positions,
                    batch.positions_kernel.clone(),
                    batch.positions_kernel,
                    self.no_kv_cache,
                    &self.non_granular_state,
                    batch.context_lens,
                ),
            });
        }
        let logits = xs.expect("No batches")?;
        logits.squeeze(0)
    }
    fn forward_completion(
        &mut self,
        input_toks: &[&mut Sequence],
    ) -> Result<Tensor, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
        } = calculate_inputs_completion(
            input_toks,
            self.is_xlora(),
            self.device(),
            self.no_kv_cache,
        )
        .unwrap();
        match self.model {
            Model::Llama(ref mut model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
            ),
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
        }
    }
    fn device(&self) -> &Device {
        match self.model {
            Model::Llama(ref model) => &model.device,
            Model::XLoraLlama(ref model) => &model.device,
        }
    }
    fn num_hidden_layers(&self) -> usize {
        self.cache().lock().len()
    }
    fn cache(&self) -> &Cache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
        }
    }
    fn get_repeat_last_n(&self) -> usize {
        self.config.repeat_last_n
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    fn eos_tok(&self) -> &[u32] {
        &self.eos_tok
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn get_max_seq_len(&self) -> usize {
        match &self.model {
            Model::Llama(model) => model.max_seq_len,
            Model::XLoraLlama(model) => model.max_seq_len,
        }
    }
    fn is_xlora(&self) -> bool {
        match &self.model {
            Model::Llama(_) => false,
            Model::XLoraLlama(_) => !self.is_lora,
        }
    }
    fn has_no_kv_cache(&self) -> bool {
        self.no_kv_cache
    }
    fn get_chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }
    fn get_non_granular_state(&self) -> &Option<NonGranularState> {
        &None
    }
    fn tok_trie(&self) -> &TokTrie {
        &self.tok_trie
    }
    fn re_isq_model(&mut self, _dtype: GgmlDType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGUF model. This will not do anything."
        )
    }
}
