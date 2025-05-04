use mistralrs_core::*;
use std::{
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    path::PathBuf,
};

use crate::{best_device, Model};

#[derive(Clone)]
/// Configure a vision model with the various parameters for loading, running, and other inference behaviors.
pub struct VisionModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) write_uqff: Option<PathBuf>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) chat_template: Option<String>,
    pub(crate) jinja_explicit: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) max_edge: Option<u32>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) search_bert_model: Option<BertEmbeddingModel>,

    // Model running
    pub(crate) use_flash_attn: bool,
    pub(crate) prompt_chunksize: Option<NonZeroUsize>,
    pub(crate) topology: Option<Topology>,
    pub(crate) loader_type: VisionLoaderType,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqType>,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
}

impl VisionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    /// - By default, web searching compatible with the OpenAI `web_search_options` setting is disabled.
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
            jinja_explicit: None,
            throughput_logging: false,
            paged_attn_cfg: None,
            hf_cache_path: None,
            search_bert_model: None,
        }
    }

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This uses the BERT model specified or the default.
    pub fn with_search(mut self, search_bert_model: BertEmbeddingModel) -> Self {
        self.search_bert_model = Some(search_bert_model);
        self
    }

    /// Enable runner throughput logging.
    pub fn with_throughput_logging(mut self) -> Self {
        self.throughput_logging = true;
        self
    }

    /// Explicit JINJA chat template file (.jinja) to be used. If specified, this overrides all other chat templates.
    pub fn with_jinja_explicit(mut self, jinja_explicit: String) -> Self {
        self.jinja_explicit = Some(jinja_explicit);
        self
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
    pub fn from_uqff(mut self, path: Vec<PathBuf>) -> Self {
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

    /// Cache path for Hugging Face models downloaded locally
    pub fn from_hf_cache_pathf(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
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
            hf_cache_path: self.hf_cache_path,
        };

        if self.with_logging {
            initialize_logging();
        }

        let loader = VisionLoaderBuilder::new(
            config,
            self.chat_template,
            self.tokenizer_json,
            Some(self.model_id),
            self.jinja_explicit,
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

        let runner = MistralRsBuilder::new(
            pipeline,
            scheduler_method,
            self.throughput_logging,
            self.search_bert_model,
        )
        .with_no_kv_cache(false)
        .with_no_prefix_cache(false);

        Ok(Model::new(runner.build()))
    }
}

#[derive(Clone)]
/// Configure a UQFF text model with the various parameters for loading, running, and other inference behaviors.
/// This wraps and implements `DerefMut` for the VisionModelBuilder, so users should take care to not call UQFF-related methods.
pub struct UqffVisionModelBuilder(VisionModelBuilder);

impl UqffVisionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(
        model_id: impl ToString,
        loader_type: VisionLoaderType,
        uqff_file: Vec<PathBuf>,
    ) -> Self {
        let mut inner = VisionModelBuilder::new(model_id, loader_type);
        inner = inner.from_uqff(uqff_file);
        Self(inner)
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        self.0.build().await
    }

    /// This wraps the VisionModelBuilder, so users should take care to not call UQFF-related methods.
    pub fn into_inner(self) -> VisionModelBuilder {
        self.0
    }
}

impl Deref for UqffVisionModelBuilder {
    type Target = VisionModelBuilder;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UqffVisionModelBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<UqffVisionModelBuilder> for VisionModelBuilder {
    fn from(value: UqffVisionModelBuilder) -> Self {
        value.0
    }
}
