mod gemma;
mod llama;
mod mistral;
mod mixtral;
use candle_sampling::logits_processor::Logprobs;
pub use gemma::{GemmaLoader, GemmaSpecificConfig};
pub use llama::{LlamaLoader, LlamaSpecificConfig};
pub use mistral::{MistralLoader, MistralSpecificConfig};
use mistralrs_lora::{LoraConfig, Ordering};
pub use mixtral::{MixtralLoader, MixtralSpecificConfig};
use std::{
    cell::RefCell,
    collections::HashMap,
    iter::repeat,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{deref_refcell, models::Cache, sequence::Sequence};

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &[PathBuf];
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>>;
    fn get_adapter_configs(&self) -> &Option<Vec<(String, LoraConfig)>>;
    fn get_classifier_path(&self) -> &Option<PathBuf>;
    fn get_classifier_config(&self) -> &Option<PathBuf>;
    fn get_ordering(&self) -> &Option<Ordering>;
}

pub enum TokenSource {
    EnvVar(String),
    Path(String),
    CacheToken,
}

pub enum ModelKind {
    Normal,
    XLoraNormal,
    QuantizedGGUF,
    XLoraGGUF,
    QuantizedGGML,
}

/// Define a method to implement specific conversation processors which take in the messages and return a prompt.
pub trait Conversation {
    fn get_prompt(
        &self,
        messages: Vec<HashMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String, String>;
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
    ) -> Result<(
        Box<Mutex<dyn Pipeline>>,
        Arc<dyn Conversation + Send + Sync>,
    )>;

    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually F32). TODO(EricLBuehler): refine
    #[allow(clippy::type_complexity)]
    fn load_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<(
        Box<Mutex<dyn Pipeline>>,
        Arc<dyn Conversation + Send + Sync>,
    )> {
        let paths = self.download_model(revision, token_source)?;
        self._setup_model(&*paths, dtype, device)
    }
}

pub trait Pipeline: Send + Sync {
    fn forward(&mut self, input_toks: Box<[Rc<RefCell<Sequence>>]>, is_prompt: bool) -> Tensor;
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>>;
    fn device(&self) -> &Device;
    fn num_hidden_layers(&self) -> usize;
    fn cache(&self) -> &Cache;
    fn sample(&mut self, logits: Tensor, seq: Rc<RefCell<Sequence>>) -> Result<Logprobs>;
    fn tokenizer(&self) -> Tokenizer;
    fn eos_tok(&self) -> u32;
    fn name(&self) -> &'static str;
    fn get_max_seq_len(&self) -> usize;
    fn is_xlora(&self) -> bool;
    fn has_no_kv_cache(&self) -> bool;
}

fn get_prompt_input(input_toks: &[Rc<RefCell<Sequence>>], device: &Device) -> (Tensor, Vec<usize>) {
    // NOTE(EricLBuehler): Unwrap reasoning: Get the maximum sequence length.
    let max_len = input_toks
        .iter()
        .map(|seq| deref_refcell!(seq).len())
        .max()
        .unwrap();
    let padding_tok = 0;
    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    for seq in input_toks.iter() {
        let mut ctxt = deref_refcell!(seq).get_toks().to_vec();
        seqlen_offsets.push(0);

        ctxt.extend(repeat(padding_tok).take(max_len - ctxt.len()));

        // NOTE(EricLBuehler): Unwrap reasoning: The dimensions must match.
        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }
    // NOTE(EricLBuehler): Unwrap reasoning: Correct dimensions are provided.
    (Tensor::cat(&seqs_tensors, 0).unwrap(), seqlen_offsets)
}

fn get_completion_input(
    input_toks: &[Rc<RefCell<Sequence>>],
    device: &Device,
    no_kv_cache: bool,
) -> (Tensor, Vec<usize>) {
    if no_kv_cache {
        return get_prompt_input(input_toks, device);
    }

    // Pad each sequence by the padding token to the max len.
    let mut seqs_tensors = Vec::new();
    let mut seqlen_offsets = Vec::new();
    for seq in input_toks.iter() {
        let start_pos = deref_refcell!(seq).get_toks().len().saturating_sub(1);
        let ctxt = deref_refcell!(seq).get_toks()[start_pos..].to_vec();
        seqlen_offsets.push(start_pos);

        // NOTE(EricLBuehler): Unwrap reasoning: The dimensions must match.
        seqs_tensors.push(Tensor::new(ctxt, device).unwrap().unsqueeze(0).unwrap());
    }
    // NOTE(EricLBuehler): Unwrap reasoning: Correct dimensions are provided.
    (Tensor::cat(&seqs_tensors, 0).unwrap(), seqlen_offsets)
}
