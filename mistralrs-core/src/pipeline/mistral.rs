use super::{
    get_completion_input, get_prompt_input, get_xlora_paths, Conversation, Loader, ModelKind,
    ModelPaths, Pipeline, TokenSource, XLoraPaths,
};
use crate::models::{quantized_llama, Cache};
use crate::xlora_models::{XLoraConfig, XLoraMistral, XLoraModelWeights};
use crate::{deref_mut_refcell, deref_refcell};
use crate::{
    models::mistral::{Config, Model as NormalModel},
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
use std::path::PathBuf;
use std::sync::Arc;
use std::{rc::Rc, sync::Mutex};
use thiserror::Error;
use tokenizers::Tokenizer;

enum Model {
    Normal(NormalModel),
    Quantized(QModelWeights),
    XLoraQuantized(XLoraModelWeights),
    XLoraNormal(XLoraMistral),
}

struct MistralConversation;
struct ZephyrConversation;

impl Conversation for MistralConversation {
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
                return Err("System role not supported for Mistral".to_string());
            } else if message["role"] == "assistant" {
                format!("{}{} ", message["content"].trim(), eos_token)
            } else {
                unreachable!();
            };
        }
        Ok(content)
    }
}

impl Conversation for ZephyrConversation {
    fn get_prompt(
        &self,
        messages: Vec<HashMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let eos_token = "</s>".to_string();
        let mut content = "".to_string();
        for message in messages {
            if message["role"] == "user" {
                content += &format!("<|user|>\n{}{}", message["content"], eos_token);
            } else if message["role"] == "system" {
                content += &format!("<|system|>\n{}{}", message["content"], eos_token);
            } else if message["role"] == "assistant" {
                content += &format!("<|assistant|>\n{}{}", message["content"], eos_token);
            }
            content += "\n";
        }
        if add_generation_prompt {
            content += "<|assistant|>\n";
        }
        Ok(content)
    }
}

pub struct MistralModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
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
}

pub struct MistralPipeline {
    model: Model,
    tokenizer: Tokenizer,
    config: MistralSpecificConfig,
    no_kv_cache: bool,
}

pub struct MistralLoader {
    model_id: String,
    config: MistralSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
}

#[derive(Clone, Copy)]
pub struct MistralSpecificConfig {
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
}

#[derive(Error, Debug)]
enum TokenizerError {
    #[error("`{0}`")]
    Error(String),
}

impl MistralLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: String,
        config: MistralSpecificConfig,
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

impl Loader for MistralLoader {
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

        let XLoraPaths {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
        } = get_xlora_paths(
            &self.xlora_model_id,
            &token_source,
            revision.clone(),
            &self.xlora_order,
        )?;

        Ok(Box::new(MistralModelPaths {
            tokenizer_filename,
            config_filename,
            filenames,
            xlora_adapter_configs: adapter_configs,
            xlora_adapter_filenames: adapter_safetensors,
            classifier_path,
            classifier_config: xlora_config,
            xlora_ordering: xlora_order,
        }))
    }

    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<(
        Box<Mutex<dyn Pipeline + Send + Sync>>,
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
            ModelKind::XLoraNormal => {
                let mut safetensors_paths = paths.get_weight_filenames().iter().collect::<Vec<_>>();
                safetensors_paths.push(paths.get_classifier_path().as_ref().unwrap());
                let vb = from_mmaped_safetensors(
                    safetensors_paths
                        .iter()
                        .map(|x| (*x).to_owned())
                        .collect::<Vec<_>>(),
                    paths
                        .get_adapter_filenames()
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|(_, x)| (*x).to_owned())
                        .collect::<Vec<_>>(),
                    dtype.unwrap_or(default_dtype),
                    device,
                    false,
                )?;

                let model = XLoraMistral::new(
                    &config,
                    vb,
                    paths.get_adapter_configs().as_ref().unwrap(),
                    paths.get_classifier_config().as_ref().unwrap().clone(),
                    paths.get_ordering().as_ref().unwrap().clone(),
                )?;
                Model::XLoraNormal(model)
            }
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
                    dtype.unwrap_or(default_dtype),
                    device,
                    false,
                )?;

                let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
                let model = gguf_file::Content::read(&mut file)
                    .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;
                let model = XLoraModelWeights::from_gguf(
                    model,
                    &mut file,
                    device,
                    paths.get_adapter_configs().as_ref().unwrap(),
                    &vb,
                    paths.get_ordering().as_ref().unwrap(),
                    paths.get_classifier_config().as_ref().unwrap().clone(),
                )?;
                Model::XLoraQuantized(model)
            }
            ModelKind::XLoraGGML => unreachable!(),
        };
        println!("Model loaded.");

        let tokenizer = Tokenizer::from_file(paths.get_tokenizer_filename())
            .map_err(|e| TokenizerError::Error(e.to_string()))?;

        Ok((
            Box::new(Mutex::new(MistralPipeline {
                model,
                tokenizer,
                config: self.config,
                no_kv_cache: self.no_kv_cache,
            })),
            if self.model_id.contains("zephyr") {
                Arc::new(ZephyrConversation)
            } else {
                Arc::new(MistralConversation)
            },
        ))
    }
}

impl Pipeline for MistralPipeline {
    fn forward(&mut self, input_toks: Box<[Rc<RefCell<Sequence>>]>, is_prompt: bool) -> Tensor {
        let (input_ids, input_ids_full, seqlen_offsets, seqlen_offsets_full) =
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
            Model::XLoraNormal(ref mut model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap(),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap(),
                self.no_kv_cache,
            ),
            Model::XLoraQuantized(ref mut model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap(),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap(),
                self.no_kv_cache,
            ),
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
            Model::XLoraNormal(ref model) => &model.device,
            Model::XLoraQuantized(ref model) => &model.device,
        }
    }
    fn num_hidden_layers(&self) -> usize {
        self.cache().lock().len()
    }
    fn cache(&self) -> &Cache {
        match self.model {
            Model::Normal(ref model) => &model.cache,
            Model::Quantized(ref model) => &model.cache,
            Model::XLoraNormal(ref model) => &model.cache,
            Model::XLoraQuantized(ref model) => &model.cache,
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
        "mistral"
    }
    fn get_max_seq_len(&self) -> usize {
        match &self.model {
            Model::Normal(model) => model.max_seq_len,
            Model::Quantized(_) => quantized_llama::MAX_SEQ_LEN as usize,
            Model::XLoraNormal(model) => model.max_seq_len,
            Model::XLoraQuantized(_) => quantized_llama::MAX_SEQ_LEN as usize,
        }
    }
    fn is_xlora(&self) -> bool {
        match &self.model {
            Model::Normal(_) | Model::Quantized(_) => false,
            Model::XLoraNormal(_) | Model::XLoraQuantized(_) => true,
        }
    }
    fn has_no_kv_cache(&self) -> bool {
        self.no_kv_cache
    }
}
