/// Shared builder methods for model builders.
///
/// All methods here are identical across ModelBuilder, TextModelBuilder, and VisionModelBuilder.
/// Invoke via `common_builder_methods!();` inside an `impl` block.
macro_rules! common_builder_methods {
    () => {
        /// Enable searching compatible with the OpenAI `web_search_options` setting.
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
            self.isq = Some(IsqSetting::Specific(isq));
            self
        }

        /// Automatically select the best ISQ quantization type for the given bit
        /// width based on the target platform.
        ///
        /// On Metal, this selects AFQ variants (e.g., AFQ4 for 4-bit).
        /// On CUDA and CPU, this selects Q*K variants (e.g., Q4K for 4-bit).
        ///
        /// The resolution happens at build time when the device is known.
        pub fn with_auto_isq(mut self, bits: IsqBits) -> Self {
            self.isq = Some(IsqSetting::Auto(bits));
            self
        }

        /// Utilise this imatrix file during ISQ. Incompatible with specifying a calibration file.
        pub fn with_imatrix(mut self, path: PathBuf) -> Self {
            self.imatrix = Some(path);
            self
        }

        /// Utilise this calibration file to collect an imatrix. Incompatible with specifying an imatrix file.
        pub fn with_calibration_file(mut self, path: PathBuf) -> Self {
            self.calibration_file = Some(path);
            self
        }

        /// Enable PagedAttention. If PagedAttention is not supported on this platform,
        /// the configuration is silently ignored.
        ///
        /// Configure with a [`PagedAttentionConfig`] object, which can be created with
        /// sensible defaults via [`crate::PagedAttentionMetaBuilder`]:
        ///
        /// ```no_run
        /// # use mistralrs::*;
        /// # fn example() -> anyhow::Result<()> {
        /// # let builder = ModelBuilder::new("model");
        /// let builder = builder.with_paged_attn(PagedAttentionMetaBuilder::default().build()?);
        /// # Ok(())
        /// # }
        /// ```
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

        /// Path to write a `.uqff` file to and serialize the other necessary files.
        pub fn write_uqff(mut self, path: PathBuf) -> Self {
            self.write_uqff = Some(path);
            self
        }

        /// Cache path for Hugging Face models downloaded locally
        pub fn from_hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
            self.hf_cache_path = Some(hf_cache_path);
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
    };
}
