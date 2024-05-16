mod cache_manager;
mod chat_template;
mod ggml;
mod gguf;
mod loaders;
mod macros;
mod normal;
mod sampling;
mod speculative;
use crate::aici::toktree::TokTrie;
use crate::device_map::DeviceMapper;
use crate::layers::set_use_matmul_via_f16;
use crate::prefix_cacher::PrefixCacheManager;
mod sampling_pipeline;
use crate::lora::{LoraConfig, Ordering};
use crate::{api_dir_list, api_get_file, DeviceMapMetadata};
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
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
pub use loaders::{
    GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType, Phi2Loader,
    Phi3Loader, Qwen2Loader,
};
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
use rand_isaac::Isaac64Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
pub use speculative::{SpeculativeConfig, SpeculativeLoader, SpeculativePipeline};
use std::fmt::{Debug, Display};
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::{collections::HashMap, fs, iter::repeat, path::PathBuf, str::FromStr};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{
    sequence::Sequence,
    utils::tokens::get_token,
    xlora_models::{NonGranularState, XLoraConfig},
};

pub use self::cache_manager::{Cache, CacheManager, LayerCaches};

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `SimpleModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths {
    /// Model weights files (multiple files supported).
    fn get_weight_filenames(&self) -> &[PathBuf];

    /// Retrieve the PretrainedConfig file.
    /// See: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/configuration#transformers.PretrainedConfig
    fn get_config_filename(&self) -> &PathBuf;

    /// A serialised `tokenizers.Tokenizer` HuggingFace object.
    /// See: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/tokenizer
    fn get_tokenizer_filename(&self) -> &PathBuf;

    /// Jinja format chat templating for chat completion.
    /// See: https://huggingface.co/docs/transformers/chat_templating
    fn get_template_filename(&self) -> &PathBuf;

    /// Optional adapter files. `(String, PathBuf)` is of the form `(id name, path)`.
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>>;

    /// Configuration of optional adapters. `(String, String)` is of the form `(id name, name)`.
    fn get_adapter_configs(&self) -> &Option<Vec<((String, String), LoraConfig)>>;

    /// Filepath for the XLORA classifier
    fn get_classifier_path(&self) -> &Option<PathBuf>;

    /// `XLoraConfig` for the XLORA classifier
    fn get_classifier_config(&self) -> &Option<XLoraConfig>;

    /// Return the defined ordering of adapters and layers within the model.
    fn get_ordering(&self) -> &Option<Ordering>;

    /// Filepath for general model configuration.
    fn get_gen_conf_filename(&self) -> Option<&PathBuf>;

    fn get_lora_preload_adapter_info(&self) -> &Option<HashMap<String, (PathBuf, LoraConfig)>>;
}

#[derive(Clone)]
pub struct LocalModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    template_filename: P,
    filenames: Vec<P>,
    xlora_adapter_filenames: Option<Vec<(String, P)>>,
    xlora_adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
    classifier_path: Option<P>,
    classifier_config: Option<XLoraConfig>,
    xlora_ordering: Option<Ordering>,
    gen_conf: Option<P>,
    lora_preload_adapter_info: Option<HashMap<String, (P, LoraConfig)>>,
}

impl<P> LocalModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        template_filename: P,
        filenames: Vec<P>,
        xlora_adapter_filenames: Option<Vec<(String, P)>>,
        xlora_adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
        classifier_path: Option<P>,
        classifier_config: Option<XLoraConfig>,
        xlora_ordering: Option<Ordering>,
        gen_conf: Option<P>,
        lora_preload_adapter_info: Option<HashMap<String, (P, LoraConfig)>>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            template_filename,
            filenames,
            xlora_adapter_filenames,
            xlora_adapter_configs,
            classifier_path,
            classifier_config,
            xlora_ordering,
            gen_conf,
            lora_preload_adapter_info,
        }
    }
}

impl ModelPaths for LocalModelPaths<PathBuf> {
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
    fn get_adapter_configs(&self) -> &Option<Vec<((String, String), LoraConfig)>> {
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
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        self.gen_conf.as_ref()
    }
    fn get_lora_preload_adapter_info(&self) -> &Option<HashMap<String, (PathBuf, LoraConfig)>> {
        &self.lora_preload_adapter_info
    }
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

#[derive(Clone, Default)]
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
    Speculative {
        target: Box<ModelKind>,
        draft: Box<ModelKind>,
    },
}

impl Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelKind::Normal => write!(f, "normal (no quant, no adapters)"),
            ModelKind::QuantizedGGML => write!(f, "quantized from ggml (no adapters)"),
            ModelKind::QuantizedGGUF => write!(f, "quantized from gguf (no adapters)"),
            ModelKind::XLoraNormal => write!(f, "x-lora (no quant)"),
            ModelKind::XLoraGGML => write!(f, "x-lora, quantized from ggml"),
            ModelKind::XLoraGGUF => write!(f, "x-lora, quantized from gguf"),
            ModelKind::LoraGGUF => write!(f, "lora, quantized from gguf"),
            ModelKind::LoraGGML => write!(f, "lora, quantized from ggml"),
            ModelKind::LoraNormal => write!(f, "lora (no quant)"),
            ModelKind::Speculative { target, draft } => {
                write!(f, "speculative: target: `{target}`, draft: `{draft}`")
            }
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
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     None,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapMetadata::dummy(),
///     None,
/// ).unwrap();
/// ```
pub trait Loader {
    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually BF16).
    /// If model is not found on HF, will attempt to resolve locally.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    fn get_id(&self) -> String;
    fn get_kind(&self) -> ModelKind;
}

#[derive(Clone)]
pub struct GeneralMetadata {
    pub max_seq_len: usize,
    pub repeat_last_n: usize,
    pub tok_trie: Arc<TokTrie>,
    pub has_no_kv_cache: bool,
    pub is_xlora: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub is_lora: bool,
}

pub enum AdapterInstruction {
    Activate(Vec<String>),
    None,
}

pub enum CacheInstruction {
    In(AdapterInstruction),
    Out,
    Reset {
        reset_non_granular: bool,
        adapter_inst: AdapterInstruction,
    },
    Nothing(AdapterInstruction),
}

#[async_trait::async_trait]
pub trait Pipeline: Send + Sync {
    fn forward_inputs(&mut self, inputs: ModelInputs) -> Result<Tensor, candle_core::Error>;
    /// This does forward pass of model followed by run.
    #[allow(clippy::too_many_arguments)]
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    ) -> Result<(), candle_core::Error> {
        let inputs = calculate_inputs(
            input_seqs,
            is_prompt,
            self.get_metadata().is_xlora,
            &self.device(),
            self.get_metadata().has_no_kv_cache,
            None,
        )
        .unwrap();

        match pre_op {
            CacheInstruction::In(adapter_inst) => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
                self.clone_in_cache(input_seqs, false)
            }
            CacheInstruction::Nothing(adapter_inst) => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
            }
            CacheInstruction::Reset {
                reset_non_granular,
                adapter_inst,
            } => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
                self.set_none_cache(reset_non_granular, false)
            }
            _ => unreachable!("Unreachable PRE cache op."),
        }

        let logits = self.forward_inputs(inputs)?;

        match post_op {
            CacheInstruction::Out => self.clone_out_cache(input_seqs, false),
            CacheInstruction::Nothing(_) => (),
            CacheInstruction::Reset {
                reset_non_granular,
                adapter_inst: _,
            } => self.set_none_cache(reset_non_granular, false),
            _ => unreachable!("Unreachable POST cache op."),
        }

        self.sample(input_seqs, logits, prefix_cacher, disable_eos_stop, rng)
            .await?;
        Ok(())
    }
    async fn sample(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn device(&self) -> Device;
    fn tokenizer(&self) -> Arc<Tokenizer>;
    fn name(&self) -> String;
    fn apply_chat_template(
        &self,
        messages: Vec<IndexMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let chat_template = self.get_chat_template();
        let template = chat_template.chat_template.as_ref().unwrap();
        let bos_tok = if let Some(ref bos) = self.get_chat_template().bos_token {
            match bos.0 {
                Either::Left(ref lit) => Some(lit.to_string()),
                Either::Right(ref added) => Some(added.content.to_string()),
            }
        } else {
            None
        };
        let eos_tok = match chat_template.eos_token {
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
    fn get_chat_template(&self) -> Arc<ChatTemplate>;
    fn reset_non_granular_state(&self);
    fn get_metadata(&self) -> &GeneralMetadata;
    fn re_isq_model(&mut self, dtype: GgmlDType) -> Result<()>;
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool);
    fn cache(&self) -> &Cache;
    /// Returns the number of activated adapters.
    fn activate_adapters(&mut self, adapters: Vec<String>) -> Result<usize>;
}
pub trait NormalModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        mapper: DeviceMapMetadata,
        loading_isq: bool,
        device: Device,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        mapper: DeviceMapMetadata,
        loading_isq: bool,
        device: Device,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
}

pub enum QuantizationBehaviour {
    Quantize(GgmlDType),
    Skip,
}

/// Return the fallback dtype for the given dtype.
fn get_fallback(dtype: GgmlDType) -> QuantizationBehaviour {
    // The normal `Q` quants are a bit more lenient than the `K` quants.
    // => Try to fallback to a similar `Q` quant.
    // If that's not possible, skip this tensor.
    match dtype {
        GgmlDType::Q2K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q3K => QuantizationBehaviour::Quantize(GgmlDType::Q4_0),
        GgmlDType::Q4K => QuantizationBehaviour::Quantize(GgmlDType::Q4_1),
        GgmlDType::Q5K => QuantizationBehaviour::Quantize(GgmlDType::Q5_0),
        GgmlDType::Q6K => QuantizationBehaviour::Quantize(GgmlDType::Q5_1),
        GgmlDType::Q8K => QuantizationBehaviour::Quantize(GgmlDType::Q8_1),
        _ => QuantizationBehaviour::Skip,
    }
}

/// Check if the tensor can be quantized with the given dtype.
fn can_quantize(tensor: &Tensor, dtype: GgmlDType) -> bool {
    let dims = tensor.shape().dims();
    // The tensor must not be empty and the last dimension must be a multiple of the block size.
    !(dims.is_empty() || (dims[dims.len() - 1] % dtype.block_size() != 0))
}

/// Check if we should quantize the tensor and if so, with which dtype.
fn get_quantization_behaviour(tensor: &Tensor, dtype: GgmlDType) -> QuantizationBehaviour {
    if dtype == GgmlDType::F32 {
        return QuantizationBehaviour::Skip;
    }

    if can_quantize(tensor, dtype) {
        return QuantizationBehaviour::Quantize(dtype);
    }
    let fallback = get_fallback(dtype);
    match fallback {
        QuantizationBehaviour::Skip => fallback,
        QuantizationBehaviour::Quantize(new_dtype) => get_quantization_behaviour(tensor, new_dtype),
    }
}

pub trait NormalModel {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
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
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper);
    /// Quantize the model in-situ.
    fn quantize(&mut self, dtype: GgmlDType, device: Device) -> candle_core::Result<()> {
        let (tensors, mapper) = self.get_tensors();
        let total_tensors = tensors.len();
        let n_quantized = AtomicUsize::new(0);
        info!(
            "Applying in-situ quantization into {dtype:?} to {total_tensors} tensors in parallel."
        );
        let bar = ProgressBar::new(total_tensors as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut devices = Vec::new();
        for (_, layer) in &tensors {
            let device = if let Some(layer) = layer {
                mapper.device_for(*layer, false).unwrap_or(&device)
            } else {
                &device
            };
            devices.push(device.clone());
        }

        tensors
            .into_par_iter()
            .zip(devices)
            .progress_with(bar)
            .for_each(|((tensor, _), device)| {
                if let QMatMul::Tensor(t) = tensor {
                    let t = t.to_device(&device).unwrap();
                    let quantization_behaviour = get_quantization_behaviour(&t, dtype);
                    *tensor =  match quantization_behaviour{
                        QuantizationBehaviour::Skip => {
                            let shape = t.shape();
                            warn!("Skipping quantization of tensor with shape {shape:?} as it is not quantizable.");
                            QMatMul::QTensor(Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap()))
                        },
                        QuantizationBehaviour::Quantize(dtype) => {
                            n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            QMatMul::QTensor(Arc::new(QTensor::quantize(&t, dtype).unwrap()))
                        }
                    }
                }
            });
        info!("Applied in-situ quantization into {dtype:?} to {n_quantized:?} tensors out of {total_tensors} total tensors.");
        Ok(())
    }
    fn activate_adapters(&mut self, _: Vec<String>) -> candle_core::Result<usize> {
        candle_core::bail!("Unable to activate adapters for model without adapters");
    }
}

struct InputMetadata {
    input: Tensor,
    positions: Vec<usize>,
    positions_kernel: Tensor,          // [bs, seq len]
    context_lens: Vec<(usize, usize)>, // (start index, len)
    position_ids: Vec<usize>,
}

fn get_prompt_input(
    input_seqs: &[&mut Sequence],
    device: &Device,
    last_n_context_len: Option<(usize, usize)>,
) -> Result<InputMetadata> {
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
    let mut position_ids = Vec::new();
    for seq in input_seqs.iter() {
        let mut ctxt = seq.get_toks().to_vec();
        let offset = if let Some((_, offset)) = last_n_context_len {
            offset
        } else {
            0
        };
        seqlen_offsets.push(offset);

        ctxt.extend(repeat(padding_tok).take(max_len.saturating_sub(ctxt.len())));
        context_lens.push((
            seq.len() - last_n_context_len.map(|(a, _)| a).unwrap_or(1),
            last_n_context_len.map(|(a, _)| a).unwrap_or(1),
        ));
        position_ids.push(seq.len());

        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }

    let mut tmp = Vec::new();
    if last_n_context_len.is_some() {
        for pos in (0..seqs_tensors.len())
            .map(|i| {
                (*seqlen_offsets.get(i).unwrap() as i64
                    ..*seqlen_offsets.get(i).unwrap() as i64 + max_len as i64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
        {
            tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
        }
    } else {
        for pos in (0..seqs_tensors.len())
            .map(|_| (0..max_len).map(|x| x as i64).collect::<Vec<_>>())
            .collect::<Vec<_>>()
        {
            tmp.push(Tensor::from_slice(&pos, pos.len(), device)?.unsqueeze(0)?);
        }
    }
    let positions_kernel = Tensor::cat(&tmp, 0)?;
    let input = Tensor::cat(&seqs_tensors, 0).unwrap();
    // Only use matmul via f16 if prompt and seqlen > 32
    if input.dim(1)? > 32 {
        set_use_matmul_via_f16(true);
    } else {
        set_use_matmul_via_f16(false);
    }
    Ok(InputMetadata {
        input,
        positions: seqlen_offsets,
        positions_kernel,
        context_lens,
        position_ids,
    })
}

fn get_completion_input(
    input_seqs: &[&mut Sequence],
    device: &Device,
    no_kv_cache: bool,
    last_n_context_len: Option<(usize, usize)>,
) -> Result<InputMetadata> {
    if no_kv_cache {
        return get_prompt_input(input_seqs, device, last_n_context_len);
    }
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    let mut context_lens = Vec::new();
    let mut position_ids = Vec::new();
    for seq in input_seqs.iter() {
        let start_pos = seq.get_toks().len().saturating_sub(1);
        let ctxt = seq.get_toks()[start_pos..].to_vec();
        seqlen_offsets.push(start_pos);
        context_lens.push((0, 1));
        position_ids.push(seq.len());

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
    set_use_matmul_via_f16(false);
    Ok(InputMetadata {
        input: Tensor::cat(&seqs_tensors, 0).unwrap(),
        positions: seqlen_offsets,
        positions_kernel,
        context_lens,
        position_ids,
    })
}

#[derive(Clone)]
pub struct ModelInputs {
    input_ids: Tensor,
    input_ids_full: Option<Tensor>,
    seqlen_offsets: Vec<usize>,
    seqlen_offsets_full: Option<Vec<usize>>,
    seqlen_offsets_kernel: Tensor,
    seqlen_offsets_kernel_full: Option<Tensor>,
    context_lens: Vec<(usize, usize)>,
    position_ids: Vec<usize>,
}

/// This will also enable matmul via f16 if prompt and the sequence length is greater than 32.
/// Otherwise, matmul via f16 is disabled
fn calculate_inputs(
    input_seqs: &[&mut Sequence],
    is_prompt: bool,
    is_xlora: bool,
    device: &Device,
    no_kv_cache: bool,
    last_n_context_len: Option<(usize, usize)>,
) -> Result<ModelInputs> {
    if is_xlora && !is_prompt {
        let InputMetadata {
            input: input_ids_full,
            positions: seqlen_offsets_full,
            positions_kernel: seqlen_offsets_kernel_full,
            context_lens: _,
            position_ids,
        } = get_prompt_input(input_seqs, device, last_n_context_len)?;
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
            position_ids: _,
        } = get_completion_input(input_seqs, device, no_kv_cache, last_n_context_len)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: Some(input_ids_full),
            seqlen_offsets,
            seqlen_offsets_full: Some(seqlen_offsets_full),
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel_full),
            context_lens,
            position_ids,
        })
    } else if is_xlora && is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
            position_ids,
        } = get_prompt_input(input_seqs, device, last_n_context_len)?;
        Ok(ModelInputs {
            input_ids: input_ids.clone(),
            input_ids_full: Some(input_ids),
            seqlen_offsets: seqlen_offsets.clone(),
            seqlen_offsets_full: Some(seqlen_offsets),
            seqlen_offsets_kernel: seqlen_offsets_kernel.clone(),
            seqlen_offsets_kernel_full: Some(seqlen_offsets_kernel),
            context_lens,
            position_ids,
        })
    } else if is_prompt {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
            position_ids,
        } = get_prompt_input(input_seqs, device, last_n_context_len)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
            context_lens,
            position_ids,
        })
    } else {
        let InputMetadata {
            input: input_ids,
            positions: seqlen_offsets,
            positions_kernel: seqlen_offsets_kernel,
            context_lens,
            position_ids,
        } = get_completion_input(input_seqs, device, no_kv_cache, last_n_context_len)?;
        Ok(ModelInputs {
            input_ids,
            input_ids_full: None,
            seqlen_offsets,
            seqlen_offsets_full: None,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full: None,
            context_lens,
            position_ids,
        })
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

struct XLoraPaths {
    adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
    adapter_safetensors: Option<Vec<(String, PathBuf)>>,
    classifier_path: Option<PathBuf>,
    xlora_order: Option<Ordering>,
    xlora_config: Option<XLoraConfig>,
    lora_preload_adapter_info: Option<HashMap<String, (PathBuf, LoraConfig)>>,
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
        let model_id = Path::new(&xlora_id);

        // Get the path for the xlora classifier
        let xlora_classifier = &api_dir_list!(api, model_id)
            .filter(|x| x.contains("xlora_classifier.safetensors"))
            .collect::<Vec<_>>();
        if xlora_classifier.len() > 1 {
            warn!("Detected multiple X-LoRA classifiers: {xlora_classifier:?}");
            warn!("Selected classifier: `{}`", &xlora_classifier[0]);
        }
        let xlora_classifier = xlora_classifier.first();

        let classifier_path =
            xlora_classifier.map(|xlora_classifier| api_get_file!(api, xlora_classifier, model_id));

        // Get the path for the xlora config by checking all for valid versions.
        // NOTE(EricLBuehler): Remove this functionality because all configs should be deserializable
        let xlora_configs = &api_dir_list!(api, model_id)
            .filter(|x| x.contains("xlora_config.json"))
            .collect::<Vec<_>>();
        if xlora_configs.len() > 1 {
            warn!("Detected multiple X-LoRA configs: {xlora_configs:?}");
        }

        let mut xlora_config: Option<XLoraConfig> = None;
        let mut last_err: Option<serde_json::Error> = None;
        for (i, config_path) in xlora_configs.iter().enumerate() {
            if xlora_configs.len() != 1 {
                warn!("Selecting config: `{}`", config_path);
            }
            let config_path = api_get_file!(api, config_path, model_id);
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
        let xlora_config = xlora_config.map(Some).unwrap_or_else(|| {
            if let Some(last_err) = last_err {
                panic!(
                    "Unable to derserialize any configs. Last error: {}",
                    last_err
                )
            } else {
                None
            }
        });

        // If there are adapters in the ordering file, get their names and remote paths
        let adapter_files = api_dir_list!(api, model_id)
            .filter_map(|name| {
                if let Some(ref adapters) = xlora_order.as_ref().unwrap().adapters {
                    for adapter_name in adapters {
                        if name.contains(adapter_name) {
                            return Some((name, adapter_name.clone()));
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>();
        if adapter_files.is_empty() && xlora_order.as_ref().unwrap().adapters.is_some() {
            anyhow::bail!("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?")
        }

        // Get the local paths for each adapter
        let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
        for (file, name) in adapter_files {
            if let Some(paths) = adapters_paths.get_mut(&name) {
                paths.push(api_get_file!(api, &file, model_id));
            } else {
                adapters_paths.insert(name, vec![api_get_file!(api, &file, model_id)]);
            }
        }

        // Sort local paths for the adapter configs and safetensors files
        let mut adapters_configs = Vec::new();
        let mut adapters_safetensors = Vec::new();
        if let Some(ref adapters) = xlora_order.as_ref().unwrap().adapters {
            for (i, name) in adapters.iter().enumerate() {
                let paths = adapters_paths
                    .get(name)
                    .unwrap_or_else(|| panic!("Adapter {name} not found."));
                for path in paths {
                    if path.extension().unwrap() == "safetensors" {
                        adapters_safetensors.push((name.clone(), path.to_owned()));
                    } else {
                        let conf = fs::read_to_string(path)?;
                        let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                        adapters_configs.push((((i + 1).to_string(), name.clone()), lora_config));
                    }
                }
            }
        }

        // Make sure they all match
        if xlora_order.as_ref().is_some_and(|order| {
            &order.base_model_id
                != xlora_config
                    .as_ref()
                    .map(|cfg| &cfg.base_model_id)
                    .unwrap_or(&base_model_id)
        }) || xlora_config
            .as_ref()
            .map(|cfg| &cfg.base_model_id)
            .unwrap_or(&base_model_id)
            != &base_model_id
        {
            anyhow::bail!(
                "Adapter ordering file, adapter model config, and base model ID do not match: {}, {}, and {} respectively.",
                xlora_order.as_ref().unwrap().base_model_id,
                xlora_config.map(|cfg| cfg.base_model_id).unwrap_or(base_model_id.clone()),
                base_model_id
            );
        }

        let lora_preload_adapter_info = if let Some(xlora_order) = xlora_order {
            // If preload adapters are specified, get their metadata like above
            if let Some(preload_adapters) = &xlora_order.preload_adapters {
                let mut output = HashMap::new();
                for adapter in preload_adapters {
                    // Get the names and remote paths of the files associated with this adapter
                    let adapter_files = api_dir_list!(api, &adapter.adapter_model_id)
                        .filter_map(|f| {
                            if f.contains(&adapter.name) {
                                Some((f, adapter.name.clone()))
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    if adapter_files.is_empty() {
                        anyhow::bail!("Adapter files are empty. Perhaps the ordering file adapters does not match the actual adapters?")
                    }
                    // Get local paths for this adapter
                    let mut adapters_paths: HashMap<String, Vec<PathBuf>> = HashMap::new();
                    for (file, name) in adapter_files {
                        if let Some(paths) = adapters_paths.get_mut(&name) {
                            paths.push(api_get_file!(api, &file, model_id));
                        } else {
                            adapters_paths.insert(name, vec![api_get_file!(api, &file, model_id)]);
                        }
                    }

                    let mut config = None;
                    let mut safetensor = None;

                    // Sort local paths for the adapter configs and safetensors files
                    let paths = adapters_paths
                        .get(&adapter.name)
                        .unwrap_or_else(|| panic!("Adapter {} not found.", adapter.name));
                    for path in paths {
                        if path.extension().unwrap() == "safetensors" {
                            safetensor = Some(path.to_owned());
                        } else {
                            let conf = fs::read_to_string(path)?;
                            let lora_config: LoraConfig = serde_json::from_str(&conf)?;
                            config = Some(lora_config);
                        }
                    }

                    let (config, safetensor) = (config.unwrap(), safetensor.unwrap());
                    output.insert(adapter.name.clone(), (safetensor, config));
                }
                Some(output)
            } else {
                None
            }
        } else {
            None
        };

        XLoraPaths {
            adapter_configs: Some(adapters_configs),
            adapter_safetensors: Some(adapters_safetensors),
            classifier_path,
            xlora_order: xlora_order.clone(),
            xlora_config,
            lora_preload_adapter_info,
        }
    } else {
        XLoraPaths {
            adapter_configs: None,
            adapter_safetensors: None,
            classifier_path: None,
            xlora_order: None,
            xlora_config: None,
            lora_preload_adapter_info: None,
        }
    })
}

fn get_model_paths(
    revision: String,
    token_source: &TokenSource,
    quantized_model_id: &Option<String>,
    quantized_filename: &Option<String>,
    api: &ApiRepo,
    model_id: &Path,
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
                let model_id = Path::new(&id);
                Ok(vec![api_get_file!(qapi, name, model_id)])
            }
        },
        None => {
            let mut filenames = vec![];
            for rfilename in api_dir_list!(api, model_id).filter(|x| x.ends_with(".safetensors")) {
                filenames.push(api_get_file!(api, &rfilename, model_id));
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
