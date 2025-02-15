use mistralrs_core::*;
use std::{num::NonZeroUsize, path::PathBuf};

use crate::{best_device, Model};

#[derive(Clone)]
/// Configure a text model with the various parameters for loading, running, and other inference behaviors.
pub struct TextModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) write_uqff: Option<PathBuf>,
    pub(crate) from_uqff: Option<PathBuf>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) chat_template: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,

    // Model running
    pub(crate) use_flash_attn: bool,
    pub(crate) prompt_chunksize: Option<NonZeroUsize>,
    pub(crate) topology: Option<Topology>,
    pub(crate) organization: IsqOrganization,
    pub(crate) loader_type: Option<NormalLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqType>,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) no_kv_cache: bool,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
}

/// Builder for PagedAttention metadata.
pub struct PagedAttentionMetaBuilder {
    block_size: Option<usize>,
    mem_cpu: usize,
    mem_gpu: MemoryGpuConfig,
}

impl Default for PagedAttentionMetaBuilder {
    fn default() -> Self {
        Self {
            block_size: None,
            mem_cpu: 64,
            mem_gpu: MemoryGpuConfig::Utilization(0.9),
        }
    }
}

impl PagedAttentionMetaBuilder {
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    pub fn with_gpu_memory(mut self, mem_gpu: MemoryGpuConfig) -> Self {
        self.mem_gpu = mem_gpu;
        self
    }

    pub fn build(self) -> anyhow::Result<PagedAttentionConfig> {
        PagedAttentionConfig::new(self.block_size, self.mem_cpu, self.mem_gpu)
    }
}

impl TextModelBuilder {
    /// A few defaults are applied here:
    /// - MoQE ISQ organization
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            use_flash_attn: cfg!(feature = "flash-attn"),
            prompt_chunksize: None,
            topology: None,
            organization: IsqOrganization::Default,
            write_uqff: None,
            from_uqff: None,
            chat_template: None,
            tokenizer_json: None,
            loader_type: None,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 32,
            no_kv_cache: false,
            prefix_cache_n: Some(16),
            with_logging: false,
            device_mapping: None,
            imatrix: None,
            calibration_file: None,
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

    /// Organize ISQ to enable MoQE (Mixture of Quantized Experts, <https://arxiv.org/abs/2310.02410>)
    pub fn with_mixture_qexperts_isq(mut self) -> Self {
        self.organization = IsqOrganization::MoeExpertsOnly;
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

    /// Manually set the model loader type. Otherwise, it will attempt to automatically
    /// determine the loader type.
    pub fn with_loader_type(mut self, loader_type: NormalLoaderType) -> Self {
        self.loader_type = Some(loader_type);
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

    /// Utilise this imatrix file during ISQ. Incompatible with specifying a calibration file.
    pub fn with_imatrix(mut self, path: PathBuf) -> Self {
        self.imatrix = Some(path);
        self
    }

    /// Utilise this calibration file to collcet an imatrix. Incompatible with specifying a calibration file.
    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
        self
    }

    /// Enable PagedAttention. Configure PagedAttention with a [`PagedAttentionConfig`] object, which
    /// can be created with sensible values with a [`PagedAttentionMetaBuilder`].
    ///
    /// If PagedAttention is not supported (query with [`paged_attn_supported`]), this will do nothing.
    pub fn with_paged_attn(
        mut self,
        paged_attn_cfg: impl FnOnce() -> anyhow::Result<PagedAttentionConfig>,
    ) -> anyhow::Result<Self> {
        if paged_attn_supported() {
            self.paged_attn_cfg = Some(paged_attn_cfg()?);
        } else {
            self.paged_attn_cfg = None;
        }
        Ok(self)
    }

    /// Set the maximum number of sequences which can be run at once.
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    /// Disable KV cache. Trade performance for memory usage.
    pub fn with_no_kv_cache(mut self) -> Self {
        self.no_kv_cache = true;
        self
    }

    /// Set the number of sequences to hold in the prefix cache. Set to `None` to disable the prefix cacher.
    pub fn with_prefix_cache_n(mut self, n_seqs: Option<usize>) -> Self {
        self.prefix_cache_n = n_seqs;
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
        let config = NormalSpecificConfig {
            use_flash_attn: self.use_flash_attn,
            prompt_chunksize: self.prompt_chunksize,
            topology: self.topology,
            organization: self.organization,
            write_uqff: self.write_uqff,
            from_uqff: self.from_uqff,
            imatrix: self.imatrix,
            calibration_file: self.calibration_file,
        };

        if self.with_logging {
            initialize_logging();
        }

        let loader = NormalLoaderBuilder::new(
            config,
            self.chat_template,
            self.tokenizer_json,
            Some(self.model_id),
        )
        .with_no_kv_cache(self.no_kv_cache)
        .build(self.loader_type)?;

        // Load, into a Pipeline
        let pipeline = loader.load_model_from_hf(
            self.hf_revision,
            self.token_source,
            &self.dtype,
            &best_device(self.force_cpu)?,
            !self.with_logging,
            self.device_mapping
                .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
            self.isq,
            self.paged_attn_cfg,
        )?;

        let scheduler_method = match self.paged_attn_cfg {
            Some(_) => {
                let config = pipeline
                    .lock()
                    .await
                    .get_metadata()
                    .cache_config
                    .as_ref()
                    .unwrap()
                    .clone();

                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: self.max_num_seqs,
                    config,
                }
            }
            None => SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(self.max_num_seqs.try_into()?),
            },
        };

        let mut runner = MistralRsBuilder::new(pipeline, scheduler_method)
            .with_no_kv_cache(self.no_kv_cache)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(self.prefix_cache_n.is_none());

        if let Some(n) = self.prefix_cache_n {
            runner = runner.with_prefix_cache_n(n)
        }

        Ok(Model::new(runner.build()))
    }
}
