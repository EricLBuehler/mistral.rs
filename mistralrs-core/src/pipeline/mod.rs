mod chat_template;
mod ggml;
mod gguf;
mod loaders;
mod macros;
mod normal;
use crate::aici::toktree::TokTrie;
use crate::DeviceMapMetadata;
use crate::{get_bias_if_not_allowed, sampler::Logprobs, sequence::SequenceRecognizer};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_nn::VarBuilder;
use chat_template::{apply_chat_template_to, ChatTemplate};
use core::fmt;
use either::Either;
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use indexmap::IndexMap;
pub use loaders::{
    GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType, Phi2Loader,
    Phi3Loader, Qwen2Loader,
};
use mistralrs_lora::{LoraConfig, Ordering};
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
use std::sync::Arc;
use std::{collections::HashMap, fs, iter::repeat, path::PathBuf, str::FromStr, sync::Mutex};
use tokenizers::Tokenizer;
use tqdm::Iter;
use tracing::{info, warn};

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

#[derive(Debug, Clone)]
/// The source of the HF token.
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

#[derive(Copy, Clone, Default)]
/// The kind of model to build.
pub enum ModelKind {
    #[default]
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

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use mistralrs_core::{Loader, TokenSource, DeviceMapMetadata};
/// use candle_core::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model(
///     None,
///     TokenSource::CacheToken,
///     None,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapMetadata::dummy(),
/// ).unwrap();
/// ```
pub trait Loader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        silent: bool,
    ) -> Result<Box<dyn ModelPaths>>;

    #[allow(clippy::type_complexity)]
    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Box<Mutex<dyn Pipeline + Send + Sync>>>;

    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually BF16).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Box<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths = self.download_model(revision, token_source, silent)?;
        self._setup_model(&*paths, dtype, device, silent, mapper, in_situ_quant)
    }

    fn get_id(&self) -> &str;
    fn get_kind(&self) -> ModelKind;
}

pub trait Pipeline: Send + Sync {
    fn forward(
        &mut self,
        input_seqs: &[&mut Sequence],
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
    fn tokenizer(&self) -> Arc<Tokenizer>;
    fn tok_trie(&self) -> &TokTrie;
    fn eos_tok(&self) -> &[u32];
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
        let bos_tok = if let Some(ref bos) = self.get_chat_template().bos_token {
            match bos.0 {
                Either::Left(ref lit) => Some(lit.to_string()),
                Either::Right(ref added) => Some(added.content.to_string()),
            }
        } else {
            None
        };
        let eos_tok = match self.get_chat_template().eos_token {
            Either::Left(ref lit) => lit,
            Either::Right(ref added) => &added.content,
        };
        let unk_tok = if let Some(ref unk) = self.get_chat_template().unk_token {
            match unk.0 {
                Either::Left(ref lit) => Some(lit.to_string()),
                Either::Right(ref added) => Some(added.content.to_string()),
            }
        } else {
            None
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
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let start_at = seq
            .get_toks()
            .len()
            .saturating_sub(self.get_repeat_last_n());
        let ctxt = seq.get_toks()[start_at..].to_vec();

        let first_lobprobs_response =
            seq.sampler()
                .sample(logits.clone(), Some(&ctxt), return_logprobs)?;

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
                    .sample(new_logits, Some(&ctxt), return_logprobs)?
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

pub trait ConfigMarker {}

pub trait NormalModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        mapper: DeviceMapMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        mapper: DeviceMapMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
}

pub trait NormalModel {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<usize>,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<usize>,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
    fn get_tensors(&mut self) -> Vec<&mut QMatMul>;
    /// Quantize the model in-situ.
    fn quantize(&mut self, dtype: GgmlDType) -> candle_core::Result<()> {
        let tensors = self.get_tensors();
        let total_tensors = tensors.len();
        let mut n_quantized = 0;
        info!("Applying in-situ quantization to {dtype:?}.");
        for tensor in tensors.into_iter().tqdm() {
            if let QMatMul::Tensor(t) = tensor {
                n_quantized += 1;
                *tensor = QMatMul::QTensor(Arc::new(QTensor::quantize(&*t, dtype)?));
            }
        }
        info!("Applied in-situ quantization into {dtype:?} to {n_quantized} tensors out of {total_tensors} total tensors.");
        Ok(())
    }
}

struct InputMetadata {
    input: Tensor,
    positions: Vec<usize>,
    positions_kernel: Tensor, // [bs, seq len]
    context_lens: Vec<usize>,
}

fn get_prompt_input(input_seqs: &[&mut Sequence], device: &Device) -> Result<InputMetadata> {
    let max_len = input_seqs
        .iter()
        .map(|seq| seq.len())
        .max()
        .expect("No sequences");
    let padding_tok = 0;
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    let mut context_lens = Vec::new();
    for seq in input_seqs.iter() {
        let mut ctxt = seq.get_toks().to_vec();
        seqlen_offsets.push(0);

        ctxt.extend(repeat(padding_tok).take(max_len.saturating_sub(ctxt.len())));
        context_lens.push(seq.len() - 1);

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
    Ok(InputMetadata {
        input: Tensor::cat(&seqs_tensors, 0).unwrap(),
        positions: seqlen_offsets,
        positions_kernel,
        context_lens,
    })
}

fn get_completion_input(
    input_seqs: &[&mut Sequence],
    device: &Device,
    no_kv_cache: bool,
) -> Result<InputMetadata> {
    if no_kv_cache {
        return get_prompt_input(input_seqs, device);
    }
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    let mut context_lens = Vec::new();
    for seq in input_seqs.iter() {
        let start_pos = seq.get_toks().len().saturating_sub(1);
        let ctxt = seq.get_toks()[start_pos..].to_vec();
        seqlen_offsets.push(start_pos);
        context_lens.push(0);

        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }
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
        context_lens,
    })
}

struct ModelInputs {
    input_ids: Tensor,
    input_ids_full: Option<Tensor>,
    seqlen_offsets: Vec<usize>,
    seqlen_offsets_full: Option<Vec<usize>>,
    seqlen_offsets_kernel: Tensor,
    seqlen_offsets_kernel_full: Option<Tensor>,
    context_lens: Vec<usize>,
}

fn calculate_inputs(
    input_seqs: &[&mut Sequence],
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
            context_lens: _,
        } = get_prompt_input(input_seqs, device)?;
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
        } = get_completion_input(input_seqs, device, no_kv_cache)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: Some(input_ids_full),
            seqlen_offsets,
            seqlen_offsets_full: Some(seqlen_offsets_full),
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel_full),
            context_lens,
        })
    } else if is_xlora && is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
        } = get_prompt_input(input_seqs, device)?;
        Ok(ModelInputs {
            input_ids: input_ids.clone(),
            input_ids_full: Some(input_ids),
            seqlen_offsets: seqlen_offsets.clone(),
            seqlen_offsets_full: Some(seqlen_offsets),
            seqlen_offsets_kernel: seqlen_offsets_kernel.clone(),
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel),
            context_lens,
        })
    } else if is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
        } = get_prompt_input(input_seqs, device)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
            context_lens,
        })
    } else {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
        } = get_completion_input(input_seqs, device, no_kv_cache)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
            context_lens,
        })
    }
}

pub fn extract_logits(logits: &Tensor, context_lens: Vec<usize>) -> candle_core::Result<Tensor> {
    let mut toks = Vec::new();
    for (dim, start) in logits.chunk(logits.dims()[0], 0)?.iter().zip(context_lens) {
        toks.push(dim.narrow(1, start, 1)?);
    }
    Tensor::cat(&toks, 0)
}

struct XLoraPaths {
    adapter_configs: Option<Vec<(String, LoraConfig)>>,
    adapter_safetensors: Option<Vec<(String, PathBuf)>>,
    classifier_path: Option<PathBuf>,
    xlora_order: Option<Ordering>,
    xlora_config: Option<XLoraConfig>,
}

fn get_xlora_paths(
    base_model_id: String,
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
            .collect::<Vec<_>>();
        if xlora_classifier.len() != 1 {
            warn!("Detected multiple X-LoRA classifiers: {xlora_classifier:?}");
            warn!("Selected classifier: `{}`", &xlora_classifier[0]);
        }
        let xlora_classifier = &xlora_classifier[0];
        let xlora_configs = &api
            .info()?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter(|x| x.contains("xlora_config.json"))
            .collect::<Vec<_>>();
        if xlora_configs.len() != 1 {
            warn!("Detected multiple X-LoRA configs: {xlora_configs:?}");
        }

        let classifier_path = api.get(xlora_classifier)?;

        let mut xlora_config: Option<XLoraConfig> = None;
        let mut last_err: Option<serde_json::Error> = None;
        for (i, config_path) in xlora_configs.iter().enumerate() {
            if xlora_configs.len() != 1 {
                warn!("Selecting config: `{}`", config_path);
            }
            let config_path = api.get(config_path)?;
            let conf = fs::read_to_string(config_path)?;
            let deser: Result<XLoraConfig, serde_json::Error> = serde_json::from_str(&conf);
            match deser {
                Ok(conf) => {
                    xlora_config = Some(conf);
                    break;
                }
                Err(e) => {
                    if i != xlora_configs.len() - 1 {
                        warn!("Config is broken with error `{e}`");
                    }
                    last_err = Some(e);
                }
            }
        }
        let xlora_config = xlora_config.unwrap_or_else(|| {
            panic!(
                "Unable to derserialize any configs. Last error: {}",
                last_err.unwrap()
            )
        });

        let adapter_files = api
            .info()?
            .siblings
            .iter()
            .map(|x| x.rfilename.clone())
            .filter_map(|name| {
                for adapter_name in xlora_order.as_ref().unwrap().adapters.as_ref().unwrap() {
                    if name.contains(adapter_name) {
                        return Some((name, adapter_name.clone()));
                    }
                }
                None
            })
            .collect::<Vec<_>>();
        if adapter_files.is_empty() {
            anyhow::bail!("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?")
        }
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
        for (i, name) in xlora_order
            .as_ref()
            .unwrap()
            .adapters
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
        {
            let paths = adapters_paths
                .get(name)
                .unwrap_or_else(|| panic!("Adapter {name} not found."));
            for path in paths {
                if path.extension().unwrap() == "safetensors" {
                    adapters_safetensors.push((name.clone(), path.to_owned()));
                } else {
                    let conf = fs::read_to_string(path)?;
                    let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                    adapters_configs.push(((i + 1).to_string(), lora_config));
                }
            }
        }

        if xlora_order
            .as_ref()
            .is_some_and(|order| order.base_model_id != xlora_config.base_model_id)
            || xlora_config.base_model_id != base_model_id
        {
            anyhow::bail!(
                "Adapter ordering file, adapter model config, and base model ID do not match: {}, {}, and {} respectively.",
                xlora_order.as_ref().unwrap().base_model_id,
                xlora_config.base_model_id,
                base_model_id
            );
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
                Some(bos.to_string()),
                eos,
                Some(unk.to_string()),
            )
            .unwrap_or_else(|_| panic!("Template number {i}"));
            assert_eq!(output, expected, "Template number {i}");
        }
    }
}
