use super::{
    get_completion_input, get_prompt_input, Conversation, Loader, ModelKind, ModelPaths, Pipeline,
    TokenSource,
};
use crate::models::{quantized_llama, Cache};
use crate::{deref_mut_refcell, deref_refcell};
use crate::{
    models::mixtral::{Config, Model as NormalModel},
    models::quantized_llama::ModelWeights as QModelWeights,
    sequence::Sequence,
    utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors},
};
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::Activation;
use candle_sampling::logits_processor::Logprobs;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_lora::{LoraConfig, Ordering};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::{rc::Rc, sync::Mutex};
use thiserror::Error;
use tokenizers::Tokenizer;

enum Model {
    Normal(NormalModel),
    Quantized(QModelWeights),
}

struct MixtralConversation;

impl Conversation for MixtralConversation {
    fn get_prompt(
        &self,
        messages: Vec<HashMap<String, String>>,
        _add_generation_prompt: bool,
    ) -> Result<String, String> {
        let bos_token = "<s>".to_string();
        let eos_token = "</s>".to_string();
        let loop_messages = if messages[0]["role"] == "system" {
            &messages[1..]
        } else {
            &messages[..]
        };
        let mut content = bos_token;
        for (i, message) in loop_messages.iter().enumerate() {
            if (message["role"] == "user") != (i % 2 == 0) {
                return Err(
                    "Conversation roles must alternate user/assistant/user/assistant/..."
                        .to_string(),
                );
            }

            content += &if message["role"] == "user" {
                format!("[INST] {} [/INST]", message["content"])
            } else if message["role"] == "system" {
                return Err("System role not supported for Mixtral".to_string());
            } else if message["role"] == "assistant" {
                format!("{}{} ", message["content"].trim(), eos_token)
            } else {
                unreachable!();
            };
        }
        Ok(content)
    }
}

pub struct MixtralModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    filenames: Vec<P>,
    xlora_adapter_filenames: Option<Vec<(String, P)>>,
    xlora_adapter_configs: Option<Vec<(String, LoraConfig)>>,
    classifier_path: Option<P>,
    classifier_config: Option<P>,
    xlora_ordering: Option<Ordering>,
}

impl ModelPaths for MixtralModelPaths<PathBuf> {
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
    fn get_classifier_config(&self) -> &Option<PathBuf> {
        &self.classifier_config
    }
    fn get_classifier_path(&self) -> &Option<PathBuf> {
        &self.classifier_path
    }
    fn get_ordering(&self) -> &Option<Ordering> {
        &self.xlora_ordering
    }
}

pub struct MixtralPipeline {
    model: Model,
    tokenizer: Tokenizer,
    config: MixtralSpecificConfig,
    no_kv_cache: bool,
}

pub struct MixtralLoader {
    model_id: String,
    config: MixtralSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
}

#[derive(Clone, Copy)]
pub struct MixtralSpecificConfig {
    pub use_flash_attn: bool,
    pub repeat_last_n: usize,
}

#[derive(Deserialize)]
pub struct BasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: usize,
    num_experts_per_tok: usize,
    num_local_experts: usize,
}

#[derive(Error, Debug)]
enum TokenizerError {
    #[error("`{0}`")]
    Error(String),
}

impl MixtralLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: String,
        config: MixtralSpecificConfig,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
    ) -> Self {
        Self {
            model_id,
            config,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            kind,
            xlora_order,
            no_kv_cache,
        }
    }
}

impl Loader for MixtralLoader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
    ) -> Result<Box<dyn ModelPaths>> {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(&token_source)?))
            .build()?;
        let revision = revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));

        let tokenizer_filename = api.get("tokenizer.json")?;

        let config_filename = api.get("config.json")?;

        let filenames = match &self.quantized_filename {
            Some(name) => {
                let qapi = ApiBuilder::new()
                    .with_progress(true)
                    .with_token(Some(get_token(&token_source)?))
                    .build()?;
                let qapi = qapi.repo(Repo::with_revision(
                    self.quantized_model_id.as_ref().unwrap().clone(),
                    RepoType::Model,
                    revision.clone(),
                ));
                vec![qapi.get(name).unwrap()]
            }
            None => {
                let mut filenames = vec![];
                for rfilename in api
                    .info()?
                    .siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .filter(|x| x.ends_with(".safetensors"))
                {
                    let filename = api.get(&rfilename)?;
                    filenames.push(filename);
                }
                filenames
            }
        };

        let (adapters_configs, adapters_safetensors, classifier_path, classifier_config, ordering) =
            if let Some(ref xlora_id) = self.xlora_model_id {
                let api = ApiBuilder::new()
                    .with_progress(true)
                    .with_token(Some(get_token(&token_source)?))
                    .build()?;
                let api = api.repo(Repo::with_revision(
                    xlora_id.clone(),
                    RepoType::Model,
                    revision,
                ));
                let xlora_classifier = &api
                    .info()?
                    .siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .filter(|x| x.contains("xlora_classifier.safetensors"))
                    .collect::<Vec<_>>()[0];
                let xlora_config = &api
                    .info()?
                    .siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .filter(|x| x.contains("xlora_config.json"))
                    .collect::<Vec<_>>()[0];
                let classifier_path = api.get(xlora_classifier)?;
                let config_path = api.get(xlora_config)?;

                let adapter_files = api
                    .info()?
                    .siblings
                    .iter()
                    .map(|x| x.rfilename.clone())
                    .filter(|x| x.contains("/adapter_"))
                    .map(|x| {
                        let mut split = x.split('/');
                        let pos = split.clone().count() - 2;
                        let name = split.nth(pos).unwrap().to_string();
                        (x, name)
                    })
                    .collect::<Vec<_>>();
                let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
                for (file, name) in adapter_files {
                    if let Some(paths) = adapters_paths.get_mut(&name) {
                        paths.push(api.get(&file)?);
                    } else {
                        adapters_paths.insert(name, vec![api.get(&file)?]);
                    }
                }
                let mut adapters_configs = Vec::new();
                let mut adapters_safetensors = Vec::new();
                for name in &self.xlora_order.as_ref().unwrap().adapters {
                    let paths = adapters_paths.get(name).unwrap();
                    for path in paths {
                        if path.extension().unwrap() == "safetensors" {
                            adapters_safetensors.push((name.clone(), path.to_owned()));
                        } else {
                            let conf = fs::read_to_string(path)?;
                            let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                            adapters_configs.push((name.clone(), lora_config));
                        }
                    }
                }
                (
                    Some(adapters_configs),
                    Some(adapters_safetensors),
                    Some(classifier_path),
                    Some(config_path),
                    self.xlora_order.clone(),
                )
            } else {
                (None, None, None, None, None)
            };

        Ok(Box::new(MixtralModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
            xlora_adapter_configs: adapters_configs,
            xlora_adapter_filenames: adapters_safetensors,
            classifier_path,
            classifier_config,
            xlora_ordering: ordering,
        }))
    }

    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<(
        Box<Mutex<dyn Pipeline>>,
        Arc<dyn Conversation + Send + Sync>,
    )> {
        let basic_config: BasicConfig =
            serde_json::from_slice(&std::fs::read(paths.get_config_filename())?)?;
        let config = Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            sliding_window: basic_config.sliding_window,
            use_flash_attn: self.config.use_flash_attn,
            num_experts_per_tok: basic_config.num_experts_per_tok,
            num_local_experts: basic_config.num_local_experts,
        };
        let default_dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        println!("Loading model on {device:?}...");
        let model = match self.kind {
            ModelKind::QuantizedGGUF => {
                let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
                let model = gguf_file::Content::read(&mut file)
                    .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;
                let model = QModelWeights::from_gguf(model, &mut file, device)?;
                Model::Quantized(model)
            }
            ModelKind::QuantizedGGML => unreachable!(),
            ModelKind::Normal => {
                let vb = from_mmaped_safetensors(
                    paths.get_weight_filenames().to_vec(),
                    Vec::new(),
                    dtype.unwrap_or(default_dtype),
                    device,
                    false,
                )?;

                let model = NormalModel::new(&config, vb)?;
                Model::Normal(model)
            }
            ModelKind::XLoraNormal => unreachable!(),
        };
        println!("Model loaded.");

        let tokenizer = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|e| TokenizerError::Error(e.to_string()))?;

        Ok((
            Box::new(Mutex::new(MixtralPipeline {
                model,
                tokenizer,
                config: self.config,
                no_kv_cache: self.no_kv_cache,
            })),
            Arc::new(MixtralConversation),
        ))
    }
}

impl Pipeline for MixtralPipeline {
    fn forward(&mut self, input_toks: Box<[Rc<RefCell<Sequence>>]>, is_prompt: bool) -> Tensor {
        let (input_ids, _input_ids_full, seqlen_offsets, _seqlen_offsets_full) =
            if self.is_xlora() && !is_prompt {
                let (input_ids_full, seqlen_offsets_full) =
                    get_prompt_input(&input_toks, self.device());
                let (input_ids, seqlen_offsets) =
                    get_completion_input(&input_toks, self.device(), self.no_kv_cache);
                (
                    input_ids,
                    Some(input_ids_full),
                    seqlen_offsets,
                    Some(seqlen_offsets_full),
                )
            } else if self.is_xlora() && is_prompt {
                let (input_ids_full, seqlen_offsets) = get_prompt_input(&input_toks, self.device());
                (
                    input_ids_full.clone(),
                    Some(input_ids_full),
                    seqlen_offsets.clone(),
                    Some(seqlen_offsets),
                )
            } else if is_prompt {
                let (input_ids, seqlen_offsets) = get_prompt_input(&input_toks, self.device());
                (input_ids, None, seqlen_offsets, None)
            } else {
                let (input_ids, seqlen_offsets) =
                    get_completion_input(&input_toks, self.device(), self.no_kv_cache);
                (input_ids, None, seqlen_offsets, None)
            };
        let result = match self.model {
            Model::Normal(ref mut model) => model.forward(&input_ids, &seqlen_offsets),
            Model::Quantized(ref mut model) => model.forward(&input_ids, &seqlen_offsets),
        };
        match result {
            Ok(v) => v,
            Err(e) => {
                panic!("Model failed with error `{e}`. Please raise an issue.");
            }
        }
    }
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn device(&self) -> &Device {
        match self.model {
            Model::Normal(ref model) => &model.device,
            Model::Quantized(ref model) => &model.device,
        }
    }
    fn num_hidden_layers(&self) -> usize {
        self.cache().lock().len()
    }
    fn cache(&self) -> &Cache {
        match self.model {
            Model::Normal(ref model) => &model.cache,
            Model::Quantized(ref model) => &model.cache,
        }
    }
    fn sample(&mut self, logits: Tensor, seq: Rc<RefCell<Sequence>>) -> Result<Logprobs> {
        let logits = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let start_at = deref_refcell!(seq)
            .get_toks()
            .len()
            .saturating_sub(self.config.repeat_last_n);
        let ctxt = deref_refcell!(seq).get_toks()[start_at..].to_vec();

        Ok(deref_mut_refcell!(seq)
            .logits_processor()
            .sample(&logits, Some(&ctxt))?)
    }
    fn tokenizer(&self) -> Tokenizer {
        self.tokenizer.clone()
    }
    fn eos_tok(&self) -> u32 {
        self.tokenizer
            .get_vocab(true)
            .get("</s>")
            .copied()
            .expect("Unable to extract `</s>` EOS token.")
    }
    fn name(&self) -> &'static str {
        "mixtral"
    }
    fn get_max_seq_len(&self) -> usize {
        match &self.model {
            Model::Normal(model) => model.max_seq_len,
            Model::Quantized(_) => quantized_llama::MAX_SEQ_LEN as usize,
        }
    }
    fn is_xlora(&self) -> bool {
        match &self.model {
            Model::Normal(_) | Model::Quantized(_) => false,
        }
    }
    fn has_no_kv_cache(&self) -> bool {
        self.no_kv_cache
    }
}
