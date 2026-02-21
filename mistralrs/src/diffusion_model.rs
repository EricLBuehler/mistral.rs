use mistralrs_core::*;

use crate::model_builder_trait::{build_diffusion_pipeline, build_model_from_pipeline};
use crate::Model;

/// Configure a diffusion model (image generation) with the various parameters for loading, running, and other inference behaviors.
pub struct DiffusionModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,

    // Model running
    pub(crate) loader_type: DiffusionLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,

    // Other things
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
}

impl DiffusionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    pub fn new(model_id: impl ToString, loader_type: DiffusionLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            max_num_seqs: 32,
            with_logging: false,
        }
    }

    /// Load the model in a certain dtype.
    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Force usage of the CPU device. Do not use PagedAttention with this.
    pub fn with_force_cpu(mut self) -> Self {
        self.force_cpu = true;
        self
    }

    /// Source of the Hugging Face token.
    pub fn with_token_source(mut self, token_source: TokenSource) -> Self {
        self.token_source = token_source;
        self
    }

    /// Set the revision to use for a Hugging Face remote model.
    pub fn with_hf_revision(mut self, revision: impl ToString) -> Self {
        self.hf_revision = Some(revision.to_string());
        self
    }

    /// Set the maximum number of sequences which can be run at once.
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    /// Enable logging.
    pub fn with_logging(mut self) -> Self {
        self.with_logging = true;
        self
    }

    /// Load the diffusion model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_diffusion_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
