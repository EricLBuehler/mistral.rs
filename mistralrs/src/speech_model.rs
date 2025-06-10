use mistralrs_core::*;

use crate::{best_device, Model};

/// Configure a text model with the various parameters for loading, running, and other inference behaviors.
pub struct SpeechModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) dac_model_id: Option<String>,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) cfg: Option<SpeechGenerationConfig>,

    // Model running
    pub(crate) loader_type: SpeechLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,

    // Other things
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
}

impl SpeechModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    pub fn new(model_id: impl ToString, loader_type: SpeechLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            loader_type,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            max_num_seqs: 32,
            with_logging: false,
            cfg: None,
            dac_model_id: None,
        }
    }

    /// DAC Model ID to load from. If not provided, this is automatically downloaded from the default path for the model.
    /// This may be a HF hub repo or a local path.
    pub fn with_dac_model_id(mut self, dac_model_id: String) -> Self {
        self.dac_model_id = Some(dac_model_id);
        self
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

    pub async fn build(self) -> anyhow::Result<Model> {
        if self.with_logging {
            initialize_logging();
        }

        let loader = SpeechLoader {
            model_id: self.model_id,
            dac_model_id: self.dac_model_id,
            arch: self.loader_type,
            cfg: self.cfg,
        };

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.hf_revision,
            self.token_source,
            &self.dtype,
            &best_device(self.force_cpu)?,
            !self.with_logging,
            DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
            None,
            None,
        )?;

        let scheduler_method = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(self.max_num_seqs.try_into()?),
        };

        let runner = MistralRsBuilder::new(pipeline, scheduler_method, false, None);

        Ok(Model::new(runner.build().await))
    }
}
