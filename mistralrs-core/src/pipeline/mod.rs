mod mistral;
use candle_sampling::logits_processor::Logprobs;
pub use mistral::{MistralLoader, MistralSpecificConfig};
use mistralrs_lora::LoraConfig;
use std::{
    cell::RefCell,
    collections::HashMap,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{models::Cache, sequence::Sequence};

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &[PathBuf];
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>>;
    fn get_adapter_configs(&self) -> &Option<Vec<(String, LoraConfig)>>;
    fn get_classifier_path(&self) -> &Option<PathBuf>;
    fn get_classifier_config(&self) -> &Option<PathBuf>;
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
    QuantizedGGML,
}

pub trait Conversation {
    fn get_prompt(
        &self,
        messages: Vec<HashMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String, String>;
}

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
}
