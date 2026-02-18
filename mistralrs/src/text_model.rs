use candle_core::Device;
use mistralrs_core::*;
use mistralrs_core::{SearchCallback, Tool, ToolCallback};

use crate::{IsqBits, IsqSetting};
use std::collections::HashMap;
use std::{
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::Arc,
};

use crate::model_builder_trait::{build_model_from_pipeline, build_text_pipeline};
use crate::Model;

#[derive(Clone)]
/// Configure a text model with the various parameters for loading, running, and other inference behaviors.
pub struct TextModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) write_uqff: Option<PathBuf>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) chat_template: Option<String>,
    pub(crate) jinja_explicit: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) search_embedding_model: Option<SearchEmbeddingModel>,
    pub(crate) search_callback: Option<Arc<SearchCallback>>,
    pub(crate) tool_callbacks: HashMap<String, Arc<ToolCallback>>,
    pub(crate) tool_callbacks_with_tools: HashMap<String, ToolCallbackWithTool>,
    pub(crate) mcp_client_config: Option<McpClientConfig>,
    pub(crate) device: Option<Device>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,

    // Model running
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) organization: IsqOrganization,
    pub(crate) loader_type: Option<NormalLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqSetting>,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) no_kv_cache: bool,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
}

/// Builder for PagedAttention metadata.
///
/// # Example
///
/// ```no_run
/// # use mistralrs::*;
/// let config = PagedAttentionMetaBuilder::default()
///     .with_block_size(32)
///     .with_gpu_memory(MemoryGpuConfig::ContextSize(8192))
///     .build()
///     .unwrap();
/// ```
pub struct PagedAttentionMetaBuilder {
    block_size: Option<usize>,
    mem_gpu: MemoryGpuConfig,
    cache_type: PagedCacheType,
}

impl Default for PagedAttentionMetaBuilder {
    fn default() -> Self {
        Self {
            block_size: None,
            mem_gpu: MemoryGpuConfig::ContextSize(4096),
            cache_type: PagedCacheType::Auto,
        }
    }
}

impl PagedAttentionMetaBuilder {
    /// Set the block size for paged attention. If not specified, a default is chosen automatically.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = Some(block_size);
        self
    }

    /// Set the GPU memory configuration for the KV cache. Defaults to `MemoryGpuConfig::ContextSize(4096)`.
    pub fn with_gpu_memory(mut self, mem_gpu: MemoryGpuConfig) -> Self {
        self.mem_gpu = mem_gpu;
        self
    }

    /// Set the paged cache data type. Defaults to `PagedCacheType::Auto`.
    pub fn with_paged_cache_type(mut self, cache_type: PagedCacheType) -> Self {
        self.cache_type = cache_type;
        self
    }

    /// Build the [`PagedAttentionConfig`]. Returns an error if the configuration is invalid.
    pub fn build(self) -> anyhow::Result<PagedAttentionConfig> {
        PagedAttentionConfig::new(self.block_size, self.mem_gpu, self.cache_type)
    }
}

impl TextModelBuilder {
    /// A few defaults are applied here:
    /// - MoQE ISQ organization
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    /// - By default, web searching compatible with the OpenAI `web_search_options` setting is disabled.
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            topology: None,
            topology_path: None,
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
            jinja_explicit: None,
            throughput_logging: false,
            hf_cache_path: None,
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            tool_callbacks_with_tools: HashMap::new(),
            mcp_client_config: None,
            device: None,
            matformer_config_path: None,
            matformer_slice_name: None,
        }
    }

    // Shared methods from builder_macros.rs
    common_builder_methods!();

    /// Configure MCP client to connect to external MCP servers and automatically
    /// register their tools for use in automatic tool calling.
    pub fn with_mcp_client(mut self, config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(config);
        self
    }

    /// Manually set the model loader type. Otherwise, it will attempt to automatically
    /// determine the loader type.
    pub fn with_loader_type(mut self, loader_type: NormalLoaderType) -> Self {
        self.loader_type = Some(loader_type);
        self
    }

    /// Disable KV cache. Trade performance for memory usage.
    pub fn with_no_kv_cache(mut self) -> Self {
        self.no_kv_cache = true;
        self
    }

    #[deprecated(
        note = "Use `UqffTextModelBuilder` to load a UQFF model instead of the generic `from_uqff`"
    )]
    /// Path to read a `.uqff` file from. Other necessary configuration files must be present at this location.
    ///
    /// For sharded UQFF models, you only need to specify the first shard file
    /// (e.g., `q4k-0.uqff`). The remaining shards are auto-discovered.
    ///
    /// For example, required files include:
    /// - `residual.safetensors`
    /// - `tokenizer.json`
    /// - `config.json`
    /// - More depending on the model
    pub fn from_uqff(mut self, path: Vec<PathBuf>) -> Self {
        self.from_uqff = Some(path);
        self
    }

    /// Load the text model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_text_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[derive(Clone)]
/// Configure a UQFF text model with the various parameters for loading, running, and other inference behaviors.
/// This wraps and implements `DerefMut` for the TextModelBuilder, so users should take care to not call UQFF-related methods.
pub struct UqffTextModelBuilder(TextModelBuilder);

impl UqffTextModelBuilder {
    /// Create a UQFF text model builder. A few defaults are applied here:
    /// - MoQE ISQ organization
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    ///
    /// For sharded UQFF models, you only need to specify the first shard file
    /// (e.g., `q4k-0.uqff`). The remaining shards are auto-discovered from the
    /// same directory or Hugging Face repository.
    pub fn new(model_id: impl ToString, uqff_file: Vec<PathBuf>) -> Self {
        let mut inner = TextModelBuilder::new(model_id);
        inner.from_uqff = Some(uqff_file);
        Self(inner)
    }

    /// Load the UQFF text model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        self.0.build().await
    }

    /// Unwrap into the inner [`TextModelBuilder`]. Take care not to call UQFF-related methods on it.
    pub fn into_inner(self) -> TextModelBuilder {
        self.0
    }
}

impl Deref for UqffTextModelBuilder {
    type Target = TextModelBuilder;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UqffTextModelBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<UqffTextModelBuilder> for TextModelBuilder {
    fn from(value: UqffTextModelBuilder) -> Self {
        value.0
    }
}
