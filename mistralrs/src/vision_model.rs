use candle_core::Device;
use mistralrs_core::*;
use mistralrs_core::{SearchCallback, Tool, ToolCallback};
use std::collections::HashMap;
use std::{
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Arc,
};

use crate::model_builder_trait::{build_model_from_pipeline, build_vision_pipeline};
use crate::Model;

/// A tool callback with its associated Tool definition.
#[derive(Clone)]
pub struct ToolCallbackWithTool {
    pub callback: Arc<ToolCallback>,
    pub tool: Tool,
}

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
    pub(crate) search_embedding_model: Option<SearchEmbeddingModel>,
    pub(crate) search_callback: Option<Arc<SearchCallback>>,
    pub(crate) tool_callbacks: HashMap<String, Arc<ToolCallback>>,
    pub(crate) tool_callbacks_with_tools: HashMap<String, ToolCallbackWithTool>,
    pub(crate) device: Option<Device>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,

    // Model running
    pub(crate) topology: Option<Topology>,
    pub(crate) loader_type: Option<VisionLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqType>,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
}

impl VisionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    /// - By default, web searching compatible with the OpenAI `web_search_options` setting is disabled.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            topology: None,
            write_uqff: None,
            from_uqff: None,
            chat_template: None,
            tokenizer_json: None,
            max_edge: None,
            loader_type: None,
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
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            device: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            prefix_cache_n: None,
        }
    }

    /// Enable searching compatible with the OpenAI `web_search_options` setting. This loads the selected search embedding reranker (EmbeddingGemma by default).
    pub fn with_search(mut self, search_embedding_model: SearchEmbeddingModel) -> Self {
        self.search_embedding_model = Some(search_embedding_model);
        self
    }

    /// Override the search function used when `web_search_options` is enabled.
    pub fn with_search_callback(mut self, callback: Arc<SearchCallback>) -> Self {
        self.search_callback = Some(callback);
        self
    }

    pub fn with_tool_callback(
        mut self,
        name: impl Into<String>,
        callback: Arc<ToolCallback>,
    ) -> Self {
        self.tool_callbacks.insert(name.into(), callback);
        self
    }

    /// Register a callback with an associated Tool definition that will be automatically
    /// added to requests when tool callbacks are active.
    pub fn with_tool_callback_and_tool(
        mut self,
        name: impl Into<String>,
        callback: Arc<ToolCallback>,
        tool: Tool,
    ) -> Self {
        let name = name.into();
        self.tool_callbacks_with_tools
            .insert(name, ToolCallbackWithTool { callback, tool });
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

    /// Manually set the model loader type. Otherwise, it will attempt to automatically
    /// determine the loader type.
    pub fn with_loader_type(mut self, loader_type: VisionLoaderType) -> Self {
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

    /// Utilise this calibration_file file during ISQ
    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
        self
    }

    /// Enable PagedAttention. Configure PagedAttention with a [`PagedAttentionConfig`] object, which
    /// can be created with sensible values with a [`PagedAttentionMetaBuilder`](crate::PagedAttentionMetaBuilder).
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

    #[deprecated(
        note = "Use `UqffTextModelBuilder` to load a UQFF model instead of the generic `from_uqff`"
    )]
    /// Path to read a `.uqff` file from. Other necessary configuration files must be present at this location.
    ///
    /// For example, these include:
    /// - `residual.safetensors`
    /// - `tokenizer.json`
    /// - `config.json`
    /// - More depending on the model
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

    /// Path to write a `.uqff` file to and serialize the other necessary files.
    ///
    /// The parent (part of the path excluding the filename) will determine where any other files
    /// serialized are written to.
    ///
    /// For example, these include:
    /// - `residual.safetensors`
    /// - `tokenizer.json`
    /// - `config.json`
    /// - More depending on the model
    pub fn write_uqff(mut self, path: PathBuf) -> Self {
        self.write_uqff = Some(path);
        self
    }

    /// Cache path for Hugging Face models downloaded locally
    pub fn from_hf_cache_pathf(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    /// Set the main device to load this model onto. Automatic device mapping will be performed starting with this device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Path to a Matryoshka Transformer configuration CSV file.
    pub fn with_matformer_config_path(mut self, path: PathBuf) -> Self {
        self.matformer_config_path = Some(path);
        self
    }

    /// Name of the slice to use from the Matryoshka Transformer configuration.
    pub fn with_matformer_slice_name(mut self, name: String) -> Self {
        self.matformer_slice_name = Some(name);
        self
    }

    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_vision_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
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
    pub fn new(model_id: impl ToString, uqff_file: Vec<PathBuf>) -> Self {
        let mut inner = VisionModelBuilder::new(model_id);
        inner.from_uqff = Some(uqff_file);
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
