mod gemma;
mod llama;
mod mistral;
mod mixtral;
mod phi2;
use crate::{get_bias_if_not_allowed, sampler::Logprobs, sequence::SequenceRecognizer};
use aici_abi::toktree::TokTrie;
use core::fmt;
use either::Either;
pub use gemma::{GemmaLoader, GemmaSpecificConfig, GEMMA_IS_GPTX};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use indexmap::IndexMap;
pub use llama::{LlamaLoader, LlamaSpecificConfig, LLAMA_IS_GPTX};
use minijinja::{context, Environment, ErrorKind};
pub use mistral::{MistralLoader, MistralSpecificConfig, MISTRAL_IS_GPTX};
use mistralrs_lora::{LoraConfig, Ordering};
pub use mixtral::{MixtralLoader, MixtralSpecificConfig, MIXTRAL_IS_GPTX};
pub use phi2::{Phi2Loader, Phi2SpecificConfig, PHI2_IS_GPTX};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    iter::repeat,
    path::PathBuf,
    str::FromStr,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{
    get_mut_arcmutex,
    models::Cache,
    sequence::Sequence,
    utils::tokens::get_token,
    xlora_models::{NonGranularState, XLoraConfig},
};

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &[PathBuf];
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
    fn get_template_filename(&self) -> &PathBuf;
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>>;
    fn get_adapter_configs(&self) -> &Option<Vec<(String, LoraConfig)>>;
    fn get_classifier_path(&self) -> &Option<PathBuf>;
    fn get_classifier_config(&self) -> &Option<XLoraConfig>;
    fn get_ordering(&self) -> &Option<Ordering>;
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatTemplate {
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    #[serde(with = "either::serde_untagged")]
    bos_token: Either<String, AddedTokensDecoder>,
    chat_template: Option<String>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    #[serde(with = "either::serde_untagged")]
    eos_token: Either<String, AddedTokensDecoder>,
    legacy: Option<bool>,
    model_max_length: f64,
    pad_token: Option<String>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: String,
    truncation_size: Option<String>,
    #[serde(with = "either::serde_untagged")]
    unk_token: Either<String, AddedTokensDecoder>,
    use_default_system_prompt: Option<bool>,
}

#[derive(Debug, Clone)]
pub enum TokenSource {
    Literal(String),
    EnvVar(String),
    Path(String),
    CacheToken,
    None,
}

impl FromStr for TokenSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        match parts[0] {
            "literal" => parts
                .get(1)
                .map(|&value| TokenSource::Literal(value.to_string()))
                .ok_or_else(|| "Expected a value for 'literal'".to_string()),
            "env" => Ok(TokenSource::EnvVar(
                parts
                    .get(1)
                    .unwrap_or(&"HUGGING_FACE_HUB_TOKEN")
                    .to_string(),
            )),
            "path" => parts
                .get(1)
                .map(|&value| TokenSource::Path(value.to_string()))
                .ok_or_else(|| "Expected a value for 'path'".to_string()),
            "cache" => Ok(TokenSource::CacheToken),
            "none" => Ok(TokenSource::None),
            _ => Err("Invalid token source format".to_string()),
        }
    }
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenSource::Literal(value) => write!(f, "literal:{}", value),
            TokenSource::EnvVar(value) => write!(f, "env:{}", value),
            TokenSource::Path(value) => write!(f, "path:{}", value),
            TokenSource::CacheToken => write!(f, "cache"),
            TokenSource::None => write!(f, "none"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum ModelKind {
    Normal,
    XLoraNormal,
    XLoraGGUF,
    XLoraGGML,
    QuantizedGGUF,
    QuantizedGGML,
    LoraGGUF,
    LoraGGML,
    LoraNormal,
}

impl AsRef<str> for ModelKind {
    fn as_ref(&self) -> &str {
        match self {
            ModelKind::Normal => "normal (no quant, no adapters)",
            ModelKind::QuantizedGGML => "quantized from ggml (no adapters)",
            ModelKind::QuantizedGGUF => "quantized from gguf (no adapters)",
            ModelKind::XLoraNormal => "x-lora (no quant)",
            ModelKind::XLoraGGML => "x-lora, quantized from ggml",
            ModelKind::XLoraGGUF => "x-lora, quantized from gguf",
            ModelKind::LoraGGUF => "lora, quantized from gguf",
            ModelKind::LoraGGML => "lora, quantized from ggml",
            ModelKind::LoraNormal => "lora (no quant)",
        }
    }
}

/// Encapsulate downloading and setting up the model. The `load_model` method is used to create the pipeline.
pub trait Loader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
    ) -> Result<Box<dyn ModelPaths>>;

    #[allow(clippy::type_complexity)]
    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<Box<Mutex<dyn Pipeline + Send + Sync>>>;

    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually F32).
    #[allow(clippy::type_complexity)]
    fn load_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<Box<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths = self.download_model(revision, token_source)?;
        self._setup_model(&*paths, dtype, device)
    }

    fn get_id(&self) -> &str;
    fn get_kind(&self) -> ModelKind;
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

fn apply_chat_template_to(
    messages: Vec<IndexMap<String, String>>,
    add_generation_prompt: bool,
    template: &str,
    bos_tok: &str,
    eos_tok: &str,
    unk_tok: &str,
) -> Result<String> {
    let mut env = Environment::new();
    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let template = template.replace(".strip()", "|trim");
    env.add_template("chat_template", template.as_str())?;
    env.add_function("raise_exception", raise_exception);
    let tmpl = env.get_template("chat_template").unwrap();
    Ok(tmpl.render(context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_tok,
        eos_token => eos_tok,
        unk_token => unk_tok,
    })?)
}

pub trait Pipeline: Send + Sync {
    fn forward(
        &mut self,
        input_toks: &[&mut Sequence],
        is_prompt: bool,
    ) -> Result<Tensor, candle_core::Error>;
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn device(&self) -> &Device;
    fn num_hidden_layers(&self) -> usize;
    fn cache(&self) -> &Cache;
    fn tokenizer(&self) -> Tokenizer;
    fn tok_trie(&self) -> Arc<TokTrie>;
    fn eos_tok(&self) -> u32;
    fn name(&self) -> String;
    fn get_max_seq_len(&self) -> usize;
    fn is_xlora(&self) -> bool;
    fn has_no_kv_cache(&self) -> bool;
    fn apply_chat_template(
        &self,
        messages: Vec<IndexMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let template = self.get_chat_template().chat_template.as_ref().unwrap();
        let bos_tok = match self.get_chat_template().bos_token {
            Either::Left(ref lit) => lit,
            Either::Right(ref added) => &added.content,
        };
        let eos_tok = match self.get_chat_template().eos_token {
            Either::Left(ref lit) => lit,
            Either::Right(ref added) => &added.content,
        };
        let unk_tok = match self.get_chat_template().unk_token {
            Either::Left(ref lit) => lit,
            Either::Right(ref added) => &added.content,
        };
        apply_chat_template_to(
            messages,
            add_generation_prompt,
            template,
            bos_tok,
            eos_tok,
            unk_tok,
        )
    }
    fn get_chat_template(&self) -> &ChatTemplate;
    fn get_non_granular_state(&self) -> &Option<NonGranularState>;
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.get_non_granular_state().as_ref() {
            *self.cache().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_repeat_last_n(&self) -> usize;
    fn sample(
        &mut self,
        logits: Tensor,
        seq: &mut Sequence,
        return_logprobs: bool,
    ) -> Result<Logprobs> {
        let logits = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let start_at = seq
            .get_toks()
            .len()
            .saturating_sub(self.get_repeat_last_n());
        let ctxt = seq.get_toks()[start_at..].to_vec();

        let first_lobprobs_response =
            seq.sampler()
                .sample(&logits, Some(&ctxt), return_logprobs)?;

        let bias_if_not_allowed = match &mut seq.recognizer {
            SequenceRecognizer::Regex(ref mut rx) => {
                get_bias_if_not_allowed!(self, rx.as_mut(), first_lobprobs_response.token)
            }
            SequenceRecognizer::Cfg(ref mut cfg) => {
                get_bias_if_not_allowed!(self, cfg.as_mut(), first_lobprobs_response.token)
            }
            SequenceRecognizer::None => None,
        };
        let second_logprobs_response = match bias_if_not_allowed {
            Some(token_set) => {
                let mut acc = vec![-f32::INFINITY; self.tok_trie().vocab_size()];
                token_set.apply_to(&mut acc);
                let new_logits = (logits + Tensor::from_slice(&acc, acc.len(), self.device())?)?;

                seq.sampler()
                    .sample(&new_logits, Some(&ctxt), return_logprobs)?
            }
            None => first_lobprobs_response,
        };

        match seq.recognizer {
            SequenceRecognizer::Regex(ref mut rx) => {
                self.tok_trie()
                    .append_token(rx.as_mut(), second_logprobs_response.token);
            }
            SequenceRecognizer::Cfg(ref mut cfg) => {
                self.tok_trie()
                    .append_token(cfg.as_mut(), second_logprobs_response.token);
            }
            SequenceRecognizer::None => {}
        }
        Ok(second_logprobs_response)
    }
}

struct InputMetadata {
    input: Tensor,
    positions: Vec<usize>,
    positions_kernel: Tensor, // [bs, seq len]
}

fn get_prompt_input(input_toks: &[&mut Sequence], device: &Device) -> Result<InputMetadata> {
    // NOTE(EricLBuehler): Unwrap reasoning: Get the maximum sequence length.
    let max_len = input_toks.iter().map(|seq| seq.len()).max().unwrap();
    let padding_tok = 0;
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    for seq in input_toks.iter() {
        let mut ctxt = seq.get_toks().to_vec();
        seqlen_offsets.push(0);

        ctxt.extend(repeat(padding_tok).take(max_len - ctxt.len()));

        // NOTE(EricLBuehler): Unwrap reasoning: The dimensions must match.
        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }

    let mut tmp = Vec::new();
    for pos in (0..seqs_tensors.len())
        .map(|_| (0..max_len).map(|x| x as i64).collect::<Vec<_>>())
        .collect::<Vec<_>>()
    {
        tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
    }
    let positions_kernel = Tensor::cat(&tmp, 0)?;
    // NOTE(EricLBuehler): Unwrap reasoning: Correct dimensions are provided.
    Ok(InputMetadata {
        input: Tensor::cat(&seqs_tensors, 0).unwrap(),
        positions: seqlen_offsets,
        positions_kernel,
    })
}

fn get_completion_input(
    input_toks: &[&mut Sequence],
    device: &Device,
    no_kv_cache: bool,
) -> Result<InputMetadata> {
    if no_kv_cache {
        return get_prompt_input(input_toks, device);
    }
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    for seq in input_toks.iter() {
        let start_pos = seq.get_toks().len().saturating_sub(1);
        let ctxt = seq.get_toks()[start_pos..].to_vec();
        seqlen_offsets.push(start_pos);

        // NOTE(EricLBuehler): Unwrap reasoning: The dimensions must match.
        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }
    // NOTE(EricLBuehler): Unwrap reasoning: Correct dimensions are provided.
    let mut tmp = Vec::new();
    for pos in (0..seqs_tensors.len())
        .map(|i| vec![*seqlen_offsets.get(i).unwrap() as i64])
        .collect::<Vec<_>>()
    {
        tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
    }
    let positions_kernel = Tensor::cat(&tmp, 0)?;
    Ok(InputMetadata {
        input: Tensor::cat(&seqs_tensors, 0).unwrap(),
        positions: seqlen_offsets,
        positions_kernel,
    })
}

struct ModelInputs {
    input_ids: Tensor,
    input_ids_full: Option<Tensor>,
    seqlen_offsets: Vec<usize>,
    seqlen_offsets_full: Option<Vec<usize>>,
    seqlen_offsets_kernel: Tensor,
    seqlen_offsets_kernel_full: Option<Tensor>,
}

fn calculate_inputs(
    input_toks: &[&mut Sequence],
    is_prompt: bool,
    is_xlora: bool,
    device: &Device,
    no_kv_cache: bool,
) -> Result<ModelInputs> {
    if is_xlora && !is_prompt {
        let InputMetadata {
            input: input_ids_full,
            positions: seqlen_offsets_full,
            positions_kernel: seqlen_offsets_kernel_full,
        } = get_prompt_input(input_toks, device)?;
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
        } = get_completion_input(input_toks, device, no_kv_cache)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: Some(input_ids_full),
            seqlen_offsets,
            seqlen_offsets_full: Some(seqlen_offsets_full),
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel_full),
        })
    } else if is_xlora && is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
        } = get_prompt_input(input_toks, device)?;
        Ok(ModelInputs {
            input_ids: input_ids.clone(),
            input_ids_full: Some(input_ids),
            seqlen_offsets: seqlen_offsets.clone(),
            seqlen_offsets_full: Some(seqlen_offsets),
            seqlen_offsets_kernel: seqlen_offsets_kernel.clone(),
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel),
        })
    } else if is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
        } = get_prompt_input(input_toks, device)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
        })
    } else {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
        } = get_completion_input(input_toks, device, no_kv_cache)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
        })
    }
}

struct XLoraPaths {
    adapter_configs: Option<Vec<(String, LoraConfig)>>,
    adapter_safetensors: Option<Vec<(String, PathBuf)>>,
    classifier_path: Option<PathBuf>,
    xlora_order: Option<Ordering>,
    xlora_config: Option<XLoraConfig>,
}

fn get_xlora_paths(
    xlora_model_id: &Option<String>,
    token_source: &TokenSource,
    revision: String,
    xlora_order: &Option<Ordering>,
) -> Result<XLoraPaths> {
    Ok(if let Some(ref xlora_id) = xlora_model_id {
        let api = ApiBuilder::new()
            .with_progress(true)
            .with_token(Some(get_token(token_source)?))
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
        let conf = fs::read_to_string(config_path)?;
        let xlora_config: XLoraConfig = serde_json::from_str(&conf)?;

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
        let adapter_order = if let Some(ref a) = xlora_config.adapters {
            a.clone()
        } else {
            if xlora_order.as_ref().unwrap().adapters.is_none() {
                return Err(anyhow::Error::msg(
                    "Must specify adapters in ordering.".to_string(),
                ));
            }
            xlora_order
                .as_ref()
                .unwrap()
                .adapters
                .as_ref()
                .unwrap()
                .clone()
        };
        for name in &adapter_order {
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
        XLoraPaths {
            adapter_configs: Some(adapters_configs),
            adapter_safetensors: Some(adapters_safetensors),
            classifier_path: Some(classifier_path),
            xlora_order: xlora_order.clone(),
            xlora_config: Some(xlora_config),
        }
    } else {
        XLoraPaths {
            adapter_configs: None,
            adapter_safetensors: None,
            classifier_path: None,
            xlora_order: None,
            xlora_config: None,
        }
    })
}

fn get_model_paths(
    revision: String,
    token_source: &TokenSource,
    quantized_model_id: &Option<String>,
    quantized_filename: &Option<String>,
    api: &ApiRepo,
) -> Result<Vec<PathBuf>> {
    match &quantized_filename {
        Some(name) => match quantized_model_id.as_ref().unwrap().as_str() {
            "" => Ok(vec![PathBuf::from_str(name).unwrap()]),
            id => {
                let qapi = ApiBuilder::new()
                    .with_progress(true)
                    .with_token(Some(get_token(token_source)?))
                    .build()?;
                let qapi = qapi.repo(Repo::with_revision(
                    id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                Ok(vec![qapi.get(name).unwrap()])
            }
        },
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
            Ok(filenames)
        }
    }
}

#[macro_export]
macro_rules! deserialize_chat_template {
    ($paths:expr, $this:ident) => {{
        use tracing::info;

        let template: ChatTemplate = serde_json::from_str(&fs::read_to_string(
            $paths.get_template_filename(),
        )?).unwrap();
        #[derive(Debug, serde::Deserialize)]
        struct SpecifiedTemplate {
            chat_template: String,
            bos_token: Option<String>,
            eos_token: Option<String>,
        }
        match template.chat_template {
            Some(_) => template,
            None => {
                info!("`tokenizer_config.json` does not contain a chat template, attempting to use specified JINJA chat template.");
                let mut deser: HashMap<String, Value> =
                    serde_json::from_str(&fs::read_to_string($paths.get_template_filename())?)
                        .unwrap();
                match $this.chat_template.clone() {
                    Some(t) => {
                        if t.ends_with(".json") {
                            info!("Loading specified loading chat template file at `{t}`.");
                            let templ: SpecifiedTemplate = serde_json::from_str(&fs::read_to_string(t.clone())?).unwrap();
                            deser.insert(
                                "chat_template".to_string(),
                                Value::String(templ.chat_template),
                            );
                            if templ.bos_token.is_some() {
                                deser.insert(
                                    "bos_token".to_string(),
                                    Value::String(templ.bos_token.unwrap()),
                                );
                            }
                            if templ.eos_token.is_some() {
                                deser.insert(
                                    "eos_token".to_string(),
                                    Value::String(templ.eos_token.unwrap()),
                                );
                            }
                            info!("Loaded chat template file.");
                        } else {
                            deser.insert(
                                "chat_template".to_string(),
                                Value::String(t),
                            );
                            info!("Loaded specified literal chat template.");
                        }
                    },
                    None => {
                        info!("No specified chat template, loading default chat template at `./default.json`.");
                        deser.insert(
                            "chat_template".to_string(),
                            Value::String(fs::read_to_string("./default.json")?),
                        );
                        info!("Default chat template loaded.");
                    }
                };
                let ser = serde_json::to_string_pretty(&deser).unwrap();
                serde_json::from_str(&ser).unwrap()
            }
        }
    }};
}

mod tests {
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
        use indexmap::IndexMap;

        use crate::pipeline::apply_chat_template_to;
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
            let mut message = IndexMap::new();
            message.insert("role".to_string(), role.to_string());
            message.insert("content".to_string(), content.to_string());
            inputs.push(message);
        }
        for ((i, (has_system, bos, eos, unk, template)), expected) in
            templates.into_iter().enumerate().zip(expected_outputs)
        {
            let output = apply_chat_template_to(
                if !has_system {
                    inputs[1..].to_vec()
                } else {
                    inputs.clone()
                },
                true,
                template,
                bos,
                eos,
                unk,
            )
            .unwrap_or_else(|_| panic!("Template number {i}"));
            assert_eq!(output, expected, "Template number {i}");
        }
    }
}
