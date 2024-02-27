mod mistral;
pub use mistral::{MistralLoader, MistralSpecificConfig};
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

pub trait ModelPaths {
    fn get_weight_filenames(&self) -> &[PathBuf];
    fn get_config_filename(&self) -> &PathBuf;
    fn get_tokenizer_filename(&self) -> &PathBuf;
}

pub enum TokenSource {
    EnvVar(String),
    Path(String),
    CacheToken,
}

pub struct SimpleModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    filenames: Vec<P>,
}

impl ModelPaths for SimpleModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
}

pub trait Loader {
    fn download_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
    ) -> Result<Box<dyn ModelPaths>>;

    fn _setup_model(
        &self,
        paths: &dyn ModelPaths,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<Box<dyn Pipeline>>;

    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually F32). TODO(EricLBuehler): refine
    fn load_model(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
    ) -> Result<Box<dyn Pipeline>> {
        let paths = self.download_model(revision, token_source)?;
        self._setup_model(&*paths, dtype, device)
    }
}

pub trait Pipeline {
    fn forward(&self, input_ids: &Tensor) -> Tensor;
    fn tokenize_prompt(&self, prompt: String) -> Tensor;
}
