use candle_core::Device;
use mistralrs_core::*;
use mistralrs_core::{SearchCallback, Tool, ToolCallback};
use std::collections::HashMap;

use crate::model_builder_trait::{build_gguf_pipeline, build_model_from_pipeline};
use crate::Model;
use std::sync::Arc;

/// Configure a text GGUF model with the various parameters for loading, running, and other inference behaviors.
pub struct GgufModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) files: Vec<String>,
    pub(crate) tok_model_id: Option<String>,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) chat_template: Option<String>,
    pub(crate) jinja_explicit: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) search_embedding_model: Option<SearchEmbeddingModel>,
    pub(crate) search_callback: Option<Arc<SearchCallback>>,
    pub(crate) tool_callbacks: HashMap<String, Arc<ToolCallback>>,
    pub(crate) tool_callbacks_with_tools: HashMap<String, ToolCallbackWithTool>,
    pub(crate) device: Option<Device>,

    // Model running
    pub(crate) force_cpu: bool,
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) no_kv_cache: bool,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
}

impl GgufModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    /// - By default, web searching compatible with the OpenAI `web_search_options` setting is disabled.
    pub fn new(model_id: impl ToString, files: Vec<impl ToString>) -> Self {
        Self {
            model_id: model_id.to_string(),
            files: files.into_iter().map(|f| f.to_string()).collect::<Vec<_>>(),
            chat_template: None,
            tokenizer_json: None,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            paged_attn_cfg: None,
            max_num_seqs: 32,
            no_kv_cache: false,
            prefix_cache_n: Some(16),
            with_logging: false,
            topology: None,
            topology_path: None,
            tok_model_id: None,
            device_mapping: None,
            jinja_explicit: None,
            throughput_logging: false,
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            device: None,
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

    /// Register a callback for a specific tool name.
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

    /// Source the tokenizer and chat template from this model ID (must contain `tokenizer.json` and `tokenizer_config.json`).
    pub fn with_tok_model_id(mut self, tok_model_id: impl ToString) -> Self {
        self.tok_model_id = Some(tok_model_id.to_string());
        self
    }

    /// Set the model topology for use during loading. If there is an overlap, the topology type is used over the ISQ type.
    pub fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Set the model topology from a path. This preserves the path for unload/reload support.
    /// If there is an overlap, the topology type is used over the ISQ type.
    pub fn with_topology_from_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> anyhow::Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.topology = Some(Topology::from_path(&path)?);
        self.topology_path = Some(path_str);
        Ok(self)
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

    /// Enable PagedAttention. Configure PagedAttention with a [`PagedAttentionConfig`] object, which
    /// can be created with sensible values with a [`PagedAttentionMetaBuilder`].
    ///
    /// If PagedAttention is not supported (query with [`paged_attn_supported`]), this will do nothing.
    ///
    /// [`PagedAttentionMetaBuilder`]: crate::PagedAttentionMetaBuilder
    pub fn with_paged_attn(mut self, paged_attn_cfg: PagedAttentionConfig) -> Self {
        if paged_attn_supported() {
            self.paged_attn_cfg = Some(paged_attn_cfg);
        }
        self
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

    /// Set the main device to load this model onto. Automatic device mapping will be performed starting with this device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Load the GGUF model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_gguf_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
