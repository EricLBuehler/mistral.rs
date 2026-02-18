use candle_core::Device;
use mistralrs_core::*;
use mistralrs_core::{SearchCallback, Tool, ToolCallback};

use crate::{IsqBits, IsqSetting};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::model_builder_trait::{build_auto_pipeline, build_model_from_pipeline};
use crate::Model;

#[derive(Clone)]
/// Configure a model with automatic detection of model type (text, vision, embedding, etc.).
///
/// This builder works like the CLI `run` command: it reads the model's `config.json` at build time
/// to determine whether it should be loaded as a text, vision, or embedding model.
///
/// Use this when you don't know (or don't care) whether a model ID corresponds to a text or
/// vision architecture. For example, `google/gemma-3-4b-it` is detected as vision,
/// while `Qwen/Qwen3-4B` is detected as text — both work seamlessly.
///
/// # Example
///
/// ```no_run
/// use mistralrs::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let model = ModelBuilder::new("Qwen/Qwen3-4B")
///         .with_auto_isq(IsqBits::Four)
///         .with_logging()
///         .build()
///         .await?;
///
///     let messages = TextMessages::new()
///         .add_message(TextMessageRole::User, "Hello!");
///     let response = model.send_chat_request(messages).await?;
///     println!("{}", response.choices[0].message.content.as_ref().unwrap());
///     Ok(())
/// }
/// ```
pub struct ModelBuilder {
    // Shared fields (see builder_macros.rs for the canonical list)
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
    pub(crate) device: Option<Device>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) organization: IsqOrganization,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqSetting>,
    pub(crate) throughput_logging: bool,
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,

    // Auto-model unique fields
    pub(crate) max_edge: Option<u32>,
    pub(crate) no_kv_cache: bool,
    pub(crate) mcp_client_config: Option<McpClientConfig>,
}

impl ModelBuilder {
    /// A few defaults are applied here:
    /// - MoQE ISQ organization
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Number of sequences to hold in prefix cache is 16.
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
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
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 32,
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
            device: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            // Unique fields
            max_edge: None,
            no_kv_cache: false,
            mcp_client_config: None,
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

    /// Disable KV cache. Trade performance for memory usage. Only applies to text models.
    pub fn with_no_kv_cache(mut self) -> Self {
        self.no_kv_cache = true;
        self
    }

    /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
    /// Only applies to vision models that support this (e.g., Qwen2-VL, Idefics 2).
    pub fn with_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

    /// Load the model (auto-detecting type) and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_auto_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}
