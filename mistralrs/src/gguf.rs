use candle_core::Device;
use mistralrs_core::*;
use mistralrs_core::{SearchCallback, Tool, ToolCallback};
use std::{collections::HashMap, path::PathBuf};

use crate::model_builder_trait::{build_gguf_pipeline, build_model_from_pipeline};
use crate::{IsqBits, IsqSetting, Model};
use std::sync::Arc;

#[derive(Clone)]
/// Configure a GGUF model with the various parameters for loading, running, and other inference behaviors.
pub struct GgufModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) files: Vec<String>,
    pub(crate) mmproj_files: Option<Vec<String>>,
    pub(crate) tok_model_id: Option<String>,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) chat_template: Option<String>,
    pub(crate) jinja_explicit: Option<String>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) matformer_config_path: Option<PathBuf>,
    pub(crate) matformer_slice_name: Option<String>,
    pub(crate) write_uqff: Option<UqffWriteConfig>,
    pub(crate) imatrix: Option<PathBuf>,
    pub(crate) calibration_file: Option<PathBuf>,
    pub(crate) search_embedding_model: Option<SearchEmbeddingModel>,
    pub(crate) search_callback: Option<Arc<SearchCallback>>,
    pub(crate) tool_callbacks: HashMap<String, ToolCallbackWithTool>,
    pub(crate) mcp_client_config: Option<McpClientConfig>,
    pub(crate) device: Option<Device>,

    // Model running
    pub(crate) force_cpu: bool,
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) organization: IsqOrganization,
    pub(crate) isq: Option<IsqSetting>,
    pub(crate) max_edge: Option<u32>,
    pub(crate) max_model_len: Option<usize>,
    pub(crate) dtype: ModelDType,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) paged_attn_cfg: Option<PagedAttentionConfig>,
    pub(crate) max_num_seqs: usize,
    pub(crate) no_kv_cache: bool,
    pub(crate) with_logging: bool,
    pub(crate) prefix_cache_n: Option<usize>,
    pub(crate) code_exec_config: Option<mistralrs_core::CodeExecutionConfig>,
    pub(crate) shell_config: Option<mistralrs_core::ShellConfig>,
    pub(crate) lora_adapters: Option<Vec<LoraAdapterSpec>>,
    pub(crate) lora_runtime_config: LoraRuntimeConfig,
    pub(crate) mtp_config: Option<MtpConfig>,
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
            mmproj_files: None,
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
            max_edge: None,
            max_model_len: None,
            dtype: ModelDType::Auto,
            tok_model_id: None,
            device_mapping: None,
            hf_cache_path: None,
            matformer_config_path: None,
            matformer_slice_name: None,
            write_uqff: None,
            imatrix: None,
            calibration_file: None,
            jinja_explicit: None,
            throughput_logging: false,
            search_embedding_model: None,
            search_callback: None,
            tool_callbacks: HashMap::new(),
            mcp_client_config: None,
            device: None,
            code_exec_config: None,
            shell_config: None,
            lora_adapters: None,
            lora_runtime_config: LoraRuntimeConfig::default(),
            mtp_config: None,
            organization: IsqOrganization::Default,
            isq: None,
        }
    }

    /// Enable the dynamic LoRA runtime without preloading an adapter.
    pub fn with_lora(mut self) -> Self {
        self.lora_adapters.get_or_insert_default();
        self
    }

    /// Preload a dynamic LoRA adapter under a request-facing alias.
    pub fn with_lora_adapter(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        self.lora_adapters
            .get_or_insert_default()
            .push(LoraAdapterSpec::new(alias, source));
        self
    }

    /// Preload a dynamic LoRA adapter at a specific Hugging Face revision.
    pub fn with_lora_adapter_revision(
        mut self,
        alias: impl Into<String>,
        source: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.lora_adapters
            .get_or_insert_default()
            .push(LoraAdapterSpec::new(alias, source).with_revision(revision));
        self
    }

    /// Preload several dynamic LoRA adapters.
    pub fn with_lora_adapters(
        mut self,
        adapters: impl IntoIterator<Item = LoraAdapterSpec>,
    ) -> Self {
        self.lora_adapters.get_or_insert_default().extend(adapters);
        self
    }

    /// Set dynamic LoRA residency and rank limits.
    pub fn with_lora_runtime_config(mut self, runtime_config: LoraRuntimeConfig) -> Self {
        self.lora_adapters.get_or_insert_default();
        self.lora_runtime_config = runtime_config;
        self
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
        let name = name.into();
        self.tool_callbacks.insert(
            name.clone(),
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(callback),
                tool: Tool {
                    tp: ToolType::Function,
                    function: Function {
                        description: None,
                        name,
                        parameters: None,
                        strict: None,
                    },
                },
            },
        );
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
        self.tool_callbacks.insert(
            name,
            ToolCallbackWithTool {
                callback: ToolCallbackKind::Text(callback),
                tool,
            },
        );
        self
    }

    /// Configure MCP servers whose tools should be available to the model.
    pub fn with_mcp_client(mut self, config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(config);
        self
    }

    /// Enable Python code execution.
    pub fn with_code_execution(mut self, config: mistralrs_core::CodeExecutionConfig) -> Self {
        self.code_exec_config = Some(config);
        self
    }

    /// Enable shell execution.
    pub fn with_shell_execution(mut self, config: mistralrs_core::ShellConfig) -> Self {
        self.shell_config = Some(config);
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

    /// Override the GGUF configuration, tokenizer, and chat template with assets from this model ID.
    pub fn with_tok_model_id(mut self, tok_model_id: impl ToString) -> Self {
        self.tok_model_id = Some(tok_model_id.to_string());
        self
    }

    /// Add GGUF multimodal projector files.
    pub fn with_mmproj_files(mut self, files: Vec<impl ToString>) -> Self {
        self.mmproj_files = Some(files.into_iter().map(|f| f.to_string()).collect());
        self
    }

    /// Automatically resize and pad images to this maximum edge length.
    pub fn with_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

    /// Cap a multimodal model's runtime context length.
    pub fn with_max_model_len(mut self, max_model_len: usize) -> Self {
        self.max_model_len = Some(max_model_len);
        self
    }

    /// Load the model in a certain dtype.
    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set the cache path for Hugging Face model assets.
    pub fn from_hf_cache_path(mut self, path: PathBuf) -> Self {
        self.hf_cache_path = Some(path);
        self
    }

    /// Set the Matryoshka Transformer configuration.
    pub fn with_matformer_config_path(mut self, path: PathBuf) -> Self {
        self.matformer_config_path = Some(path);
        self
    }

    /// Select a Matryoshka Transformer slice.
    pub fn with_matformer_slice_name(mut self, name: String) -> Self {
        self.matformer_slice_name = Some(name);
        self
    }

    /// Use ISQ of a certain type.
    pub fn with_isq(mut self, isq: IsqType) -> Self {
        self.isq = Some(IsqSetting::Specific(isq));
        self
    }

    /// Automatically select an ISQ type for the target platform and bit width.
    pub fn with_auto_isq(mut self, bits: IsqBits) -> Self {
        self.isq = Some(IsqSetting::Auto(bits));
        self
    }

    /// Organize ISQ to quantize only MoE experts.
    pub fn with_mixture_qexperts_isq(mut self) -> Self {
        self.organization = IsqOrganization::MoeExpertsOnly;
        self
    }

    /// Use an imatrix file while requantizing the GGUF weights.
    pub fn with_imatrix(mut self, path: PathBuf) -> Self {
        self.imatrix = Some(path);
        self
    }

    /// Generate an imatrix from a calibration file while requantizing the GGUF weights.
    pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
        self.calibration_file = Some(path);
        self
    }

    /// Write the loaded weights as UQFF.
    pub fn write_uqff(mut self, config: impl Into<UqffWriteConfig>) -> Self {
        self.write_uqff = Some(config.into());
        self
    }

    /// Attach an MTP assistant for speculative decoding.
    pub fn with_mtp_config(mut self, mtp_config: MtpConfig) -> Self {
        self.mtp_config = Some(mtp_config);
        self
    }

    /// Attach an MTP assistant by model id or path.
    pub fn with_mtp_model(mut self, model: impl Into<String>, n_predict: Option<usize>) -> Self {
        self.mtp_config = Some(MtpConfig::new(model, n_predict));
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

#[cfg(test)]
mod tests {
    use super::GgufModelBuilder;
    use mistralrs_core::{
        IsqOrganization, IsqType, LoraRuntimeConfig, McpClientConfig, ModelDType, MtpConfig,
    };
    use std::path::PathBuf;

    #[test]
    fn dynamic_lora_builder_distinguishes_disabled_and_empty_runtime() {
        let disabled = GgufModelBuilder::new("repo", vec!["model.gguf"]);
        assert!(disabled.lora_adapters.is_none());

        let enabled = GgufModelBuilder::new("repo", vec!["model.gguf"]).with_lora();
        assert_eq!(enabled.lora_adapters, Some(Vec::new()));
    }

    #[test]
    fn dynamic_lora_builder_preserves_adapters_and_limits() {
        let limits = LoraRuntimeConfig {
            max_adapters: 3,
            max_rank: 32,
            max_bytes: 1_024,
        };
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"])
            .with_lora_adapter_revision("code", "org/code", "revision")
            .with_lora_runtime_config(limits);
        let adapters = builder.lora_adapters.unwrap();

        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].alias, "code");
        assert_eq!(adapters[0].source, "org/code");
        assert_eq!(adapters[0].revision.as_deref(), Some("revision"));
        assert_eq!(builder.lora_runtime_config, limits);
    }

    #[test]
    fn multimodal_dynamic_lora_builder_preserves_projector_and_adapter() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"])
            .with_mmproj_files(vec!["mmproj.gguf"])
            .with_lora_adapter("vision-chat", "org/language-lora");

        assert_eq!(builder.mmproj_files, Some(vec!["mmproj.gguf".to_string()]));
        let adapters = builder.lora_adapters.unwrap();
        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].alias, "vision-chat");
        assert_eq!(adapters[0].source, "org/language-lora");
    }

    #[test]
    fn runtime_configuration_is_preserved() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"])
            .with_dtype(ModelDType::F16)
            .with_isq(IsqType::Q4K)
            .with_mixture_qexperts_isq()
            .with_imatrix(PathBuf::from("model.imatrix"))
            .with_max_model_len(8192)
            .from_hf_cache_path(PathBuf::from("hf-cache"))
            .with_matformer_config_path(PathBuf::from("matformer.csv"))
            .with_matformer_slice_name("small".to_string())
            .with_mtp_model("org/mtp", Some(3));

        assert_eq!(builder.dtype, ModelDType::F16);
        assert!(builder.isq.is_some());
        assert!(matches!(
            builder.organization,
            IsqOrganization::MoeExpertsOnly
        ));
        assert_eq!(builder.imatrix, Some(PathBuf::from("model.imatrix")));
        assert_eq!(builder.max_model_len, Some(8192));
        assert_eq!(builder.hf_cache_path, Some(PathBuf::from("hf-cache")));
        assert_eq!(
            builder.matformer_config_path,
            Some(PathBuf::from("matformer.csv"))
        );
        assert_eq!(builder.matformer_slice_name.as_deref(), Some("small"));
        let mtp = builder.mtp_config.unwrap();
        assert_eq!(mtp.model, "org/mtp");
        assert_eq!(mtp.n_predict, Some(3));
    }

    #[test]
    fn explicit_mtp_configuration_is_preserved() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"])
            .with_mtp_config(MtpConfig::new("local-mtp", None));

        assert_eq!(builder.mtp_config.unwrap().model, "local-mtp");
    }

    #[test]
    fn mcp_configuration_is_preserved() {
        let builder = GgufModelBuilder::new("repo", vec!["model.gguf"])
            .with_mcp_client(McpClientConfig::default());

        assert!(builder.mcp_client_config.is_some());
    }
}
