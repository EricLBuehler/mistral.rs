use mistralrs_core::*;
use std::{num::NonZeroUsize, path::PathBuf};

use crate::{best_device, Model};

#[derive(Clone)]
/// Configure a vision model with the various parameters for loading, running, and other inference behaviors.
pub struct VisionModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) write_uqff: Option<PathBuf>,
    pub(crate) from_uqff: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) chat_template: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) max_edge: Option<u32>,

    // Model running
    pub(crate) use_flash_attn: bool,
    pub(crate) prompt_chunksize: Option<NonZeroUsize>,
    pub(crate) topology: Option<Topology>,
    pub(crate) loader_type: VisionLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqType>,

    // Other things
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
}

impl VisionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(model_id: impl ToString, loader_type: VisionLoaderType) -> Self {
        Self {
            model_id: model_id.to_string(),
            use_flash_attn: cfg!(feature = "flash-attn"),
            topology: None,
            write_uqff: None,
            from_uqff: None,
            prompt_chunksize: None,
            chat_template: None,
            tokenizer_json: None,
            max_edge: None,
            loader_type,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            isq: None,
            max_num_seqs: 32,
            with_logging: false,
            device_mapping: None,
            calibration_file: None,
            imatrix: None,
        }
    }

    /// Set the prompt batchsize to use for inference.
    pub fn with_prompt_chunksize(mut self, prompt_chunksize: NonZeroUsize) -> Self {
        self.prompt_chunksize = Some(prompt_chunksize);
        self
    }

    /// Set the model topology for use during loading. If there is an overlap, the topology type is used over the ISQ type.
    pub fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Literal Jinja chat template OR Path (ending in `.json`) to one.
    pub fn with_chat_template(mut self, chat_template: impl ToString) -> Self {
        self.chat_template = Some(chat_template.to_string());
        self
    }

    /// Path to a discrete `tokenizer.json` file.
    pub fn with_tokenizer_json(mut self, tokenizer_json: impl ToString) -> Self {
        self.tokenizer_json = Some(tokenizer_json.to_string());
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

    /// Use ISQ of a certain type. If there is an overlap, the topology type is used over the ISQ type.
    pub fn with_isq(mut self, isq: IsqType) -> Self {
        self.isq = Some(isq);
        self
    }

    /// Utilise this calibration_file file during ISQ
    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
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

    /// Provide metadata to initialize the device mapper.
    pub fn with_device_mapping(mut self, device_mapping: DeviceMapSetting) -> Self {
        self.device_mapping = Some(device_mapping);
        self
    }

    /// Path to read a UQFF file from.
    pub fn from_uqff(mut self, path: PathBuf) -> Self {
        self.from_uqff = Some(path);
        self
    }

    /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
    /// This is only supported on the Qwen2-VL and Idefics 2 models. Others handle this internally.
    pub fn from_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

    /// Path to write a UQFF file to.
    ///
    /// The parent (part of the path excluding the filename) will determine where any other files
    /// generated are written to. These can be used to load UQFF models standalone, and may include:
    /// - `residual.safetensors`
    /// - `tokenizer.json`
    /// - `config.json`
    /// - And others
    pub fn write_uqff(mut self, path: PathBuf) -> Self {
        self.write_uqff = Some(path);
        self
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        let config = VisionSpecificConfig {
            use_flash_attn: self.use_flash_attn,
            prompt_chunksize: self.prompt_chunksize,
            topology: self.topology,
            write_uqff: self.write_uqff,
            from_uqff: self.from_uqff,
            max_edge: self.max_edge,
            calibration_file: self.calibration_file,
            imatrix: self.imatrix,
        };

        if self.with_logging {
            initialize_logging();
        }

        let loader = VisionLoaderBuilder::new(
            config,
            self.chat_template,
            self.tokenizer_json,
            Some(self.model_id),
        )
        .build(self.loader_type);

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.hf_revision,
            self.token_source,
            &self.dtype,
            &best_device(self.force_cpu)?,
            !self.with_logging,
            self.device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_vision())),
            self.isq,
            None,
        )?;

        let scheduler_method = SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(self.max_num_seqs.try_into()?),
        };

        let runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(false)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(false);

        Ok(Model::new(runner.build()))
    }
}
