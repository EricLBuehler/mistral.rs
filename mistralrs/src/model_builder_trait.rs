use mistralrs_core::{AddModelConfig, Pipeline, SchedulerConfig};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::Model;

/// Enum representing all possible model builders that can be used with MultiModelBuilder.
pub enum AnyModelBuilder {
    Text(crate::TextModelBuilder),
    Vision(crate::VisionModelBuilder),
    Gguf(crate::GgufModelBuilder),
    Diffusion(crate::DiffusionModelBuilder),
    Speech(crate::SpeechModelBuilder),
    Embedding(crate::EmbeddingModelBuilder),
}

impl AnyModelBuilder {
    /// Get the default model ID for this builder.
    pub fn model_id(&self) -> String {
        match self {
            AnyModelBuilder::Text(b) => b.model_id.clone(),
            AnyModelBuilder::Vision(b) => b.model_id.clone(),
            AnyModelBuilder::Gguf(b) => b.model_id.clone(),
            AnyModelBuilder::Diffusion(b) => b.model_id.clone(),
            AnyModelBuilder::Speech(b) => b.model_id.clone(),
            AnyModelBuilder::Embedding(b) => b.model_id.clone(),
        }
    }

    /// Build the pipeline and configuration for this model.
    pub async fn build_pipeline(
        self,
    ) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
        match self {
            AnyModelBuilder::Text(b) => build_text_pipeline(b).await,
            AnyModelBuilder::Vision(b) => build_vision_pipeline(b).await,
            AnyModelBuilder::Gguf(b) => build_gguf_pipeline(b).await,
            AnyModelBuilder::Diffusion(b) => build_diffusion_pipeline(b).await,
            AnyModelBuilder::Speech(b) => build_speech_pipeline(b).await,
            AnyModelBuilder::Embedding(b) => build_embedding_pipeline(b).await,
        }
    }
}

// Conversion implementations
impl From<crate::TextModelBuilder> for AnyModelBuilder {
    fn from(b: crate::TextModelBuilder) -> Self {
        AnyModelBuilder::Text(b)
    }
}

impl From<crate::VisionModelBuilder> for AnyModelBuilder {
    fn from(b: crate::VisionModelBuilder) -> Self {
        AnyModelBuilder::Vision(b)
    }
}

impl From<crate::GgufModelBuilder> for AnyModelBuilder {
    fn from(b: crate::GgufModelBuilder) -> Self {
        AnyModelBuilder::Gguf(b)
    }
}

impl From<crate::DiffusionModelBuilder> for AnyModelBuilder {
    fn from(b: crate::DiffusionModelBuilder) -> Self {
        AnyModelBuilder::Diffusion(b)
    }
}

impl From<crate::SpeechModelBuilder> for AnyModelBuilder {
    fn from(b: crate::SpeechModelBuilder) -> Self {
        AnyModelBuilder::Speech(b)
    }
}

impl From<crate::EmbeddingModelBuilder> for AnyModelBuilder {
    fn from(b: crate::EmbeddingModelBuilder) -> Self {
        AnyModelBuilder::Embedding(b)
    }
}

/// Builder for creating a Model with multiple models.
pub struct MultiModelBuilder {
    builders: Vec<AnyModelBuilder>,
    default_model_id: Option<String>,
}

impl Default for MultiModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModelBuilder {
    /// Create a new MultiModelBuilder.
    pub fn new() -> Self {
        Self {
            builders: Vec::new(),
            default_model_id: None,
        }
    }

    /// Add a model. The model ID will be the pipeline's model_id (e.g., "google/gemma-3-4b-it").
    pub fn add_model<B: Into<AnyModelBuilder>>(mut self, builder: B) -> Self {
        self.builders.push(builder.into());
        self
    }

    /// Set the default model by its model ID (e.g., "google/gemma-3-4b-it").
    pub fn with_default_model(mut self, model_id: impl ToString) -> Self {
        self.default_model_id = Some(model_id.to_string());
        self
    }

    /// Build the multi-model Model instance.
    pub async fn build(self) -> anyhow::Result<Model> {
        if self.builders.is_empty() {
            anyhow::bail!("MultiModelBuilder requires at least one model to be added");
        }

        // Build the first model to create the initial MistralRs instance
        let mut builders_iter = self.builders.into_iter();
        let first_builder = builders_iter.next().unwrap();

        let (pipeline, scheduler_config, add_model_config) = first_builder.build_pipeline().await?;

        // Create the MistralRsBuilder for the first model
        // The model ID will be taken from the pipeline's name
        let mut runner_builder = mistralrs_core::MistralRsBuilder::new(
            pipeline,
            scheduler_config,
            add_model_config.engine_config.throughput_logging_enabled,
            add_model_config.engine_config.search_embedding_model,
        );

        if let Some(cb) = add_model_config.engine_config.search_callback.clone() {
            runner_builder = runner_builder.with_search_callback(cb);
        }

        for (name, cb) in &add_model_config.engine_config.tool_callbacks {
            runner_builder = runner_builder.with_tool_callback(name.clone(), cb.clone());
        }

        for (name, callback_with_tool) in &add_model_config.engine_config.tool_callbacks_with_tools
        {
            runner_builder = runner_builder.with_tool_callback_and_tool(
                name.clone(),
                callback_with_tool.callback.clone(),
                callback_with_tool.tool.clone(),
            );
        }

        if let Some(mcp_config) = add_model_config.mcp_client_config.clone() {
            runner_builder = runner_builder.with_mcp_client(mcp_config);
        }

        if let Some(loader_config) = add_model_config.loader_config.clone() {
            runner_builder = runner_builder.with_loader_config(loader_config);
        }

        runner_builder = runner_builder
            .with_no_kv_cache(add_model_config.engine_config.no_kv_cache)
            .with_no_prefix_cache(add_model_config.engine_config.no_prefix_cache)
            .with_prefix_cache_n(add_model_config.engine_config.prefix_cache_n);

        let mistralrs = runner_builder.build().await;

        // Add remaining models using their pipeline names as IDs
        for builder in builders_iter {
            let model_id = builder.model_id();
            let (pipeline, scheduler_config, add_model_config) = builder.build_pipeline().await?;
            mistralrs
                .add_model(model_id, pipeline, scheduler_config, add_model_config)
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        // Set the default model if specified
        if let Some(default_id) = self.default_model_id {
            mistralrs
                .set_default_model_id(&default_id)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        // Otherwise, the first model is already the default (set by MistralRs::new)

        Ok(Model::new(mistralrs))
    }
}

// Pipeline building functions for each model type.
// These are public so individual builders can reuse them to avoid code duplication.

/// Create a Model from pipeline components.
/// This is the common code path used by all individual builder `build()` methods.
pub async fn build_model_from_pipeline(
    pipeline: Arc<Mutex<dyn mistralrs_core::Pipeline>>,
    scheduler_config: SchedulerConfig,
    add_model_config: AddModelConfig,
) -> Model {
    let mut runner_builder = mistralrs_core::MistralRsBuilder::new(
        pipeline,
        scheduler_config,
        add_model_config.engine_config.throughput_logging_enabled,
        add_model_config.engine_config.search_embedding_model,
    );

    if let Some(cb) = add_model_config.engine_config.search_callback.clone() {
        runner_builder = runner_builder.with_search_callback(cb);
    }

    for (name, cb) in &add_model_config.engine_config.tool_callbacks {
        runner_builder = runner_builder.with_tool_callback(name.clone(), cb.clone());
    }

    for (name, callback_with_tool) in &add_model_config.engine_config.tool_callbacks_with_tools {
        runner_builder = runner_builder.with_tool_callback_and_tool(
            name.clone(),
            callback_with_tool.callback.clone(),
            callback_with_tool.tool.clone(),
        );
    }

    if let Some(mcp_config) = add_model_config.mcp_client_config.clone() {
        runner_builder = runner_builder.with_mcp_client(mcp_config);
    }

    if let Some(loader_config) = add_model_config.loader_config.clone() {
        runner_builder = runner_builder.with_loader_config(loader_config);
    }

    runner_builder = runner_builder
        .with_no_kv_cache(add_model_config.engine_config.no_kv_cache)
        .with_no_prefix_cache(add_model_config.engine_config.no_prefix_cache)
        .with_prefix_cache_n(add_model_config.engine_config.prefix_cache_n);

    Model::new(runner_builder.build().await)
}

/// Build a text model pipeline from a TextModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_text_pipeline(
    builder: crate::TextModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    let config = NormalSpecificConfig {
        topology: builder.topology.clone(),
        organization: builder.organization,
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        imatrix: builder.imatrix.clone(),
        calibration_file: builder.calibration_file.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
    };

    if builder.with_logging {
        initialize_logging();
    }

    let loader = NormalLoaderBuilder::new(
        config,
        builder.chat_template.clone(),
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
        builder.no_kv_cache,
        builder.jinja_explicit.clone(),
    )
    .build(builder.loader_type.clone())?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &builder
            .device
            .clone()
            .unwrap_or(best_device(builder.force_cpu).unwrap()),
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        builder.isq,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = match builder.paged_attn_cfg {
        Some(_) => {
            let config = pipeline
                .lock()
                .await
                .get_metadata()
                .cache_config
                .as_ref()
                .cloned();

            if let Some(config) = config {
                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: builder.max_num_seqs,
                    config,
                }
            } else {
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
                }
            }
        }
        None => SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
        },
    };

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        search_embedding_model: builder.search_embedding_model,
        search_callback: builder.search_callback.clone(),
        tool_callbacks: builder.tool_callbacks.clone(),
        tool_callbacks_with_tools: builder
            .tool_callbacks_with_tools
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    mistralrs_core::ToolCallbackWithTool {
                        callback: v.callback.clone(),
                        tool: v.tool.clone(),
                    },
                )
            })
            .collect(),
        no_kv_cache: builder.no_kv_cache,
        no_prefix_cache: builder.prefix_cache_n.is_none(),
        prefix_cache_n: builder.prefix_cache_n.unwrap_or(16),
        disable_eos_stop: false,
    };

    // Create loader config for unload/reload support
    let device = builder
        .device
        .clone()
        .unwrap_or(best_device(builder.force_cpu).unwrap());
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = builder.from_uqff.as_ref().map(|paths| {
        paths
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(";")
    });

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Plain {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            organization: Some(builder.organization),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: from_uqff_str,
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: builder.isq,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a vision model pipeline from a VisionModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_vision_pipeline(
    builder: crate::VisionModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    let config = VisionSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        max_edge: builder.max_edge,
        calibration_file: builder.calibration_file.clone(),
        imatrix: builder.imatrix.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
    };

    if builder.with_logging {
        initialize_logging();
    }

    let loader = VisionLoaderBuilder::new(
        config,
        builder.chat_template.clone(),
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
        builder.jinja_explicit.clone(),
    )
    .build(builder.loader_type.clone());

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &builder
            .device
            .clone()
            .unwrap_or(best_device(builder.force_cpu).unwrap()),
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_vision())),
        builder.isq,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = match builder.paged_attn_cfg {
        Some(_) => {
            let config = pipeline
                .lock()
                .await
                .get_metadata()
                .cache_config
                .as_ref()
                .cloned();

            if let Some(config) = config {
                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: builder.max_num_seqs,
                    config,
                }
            } else {
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
                }
            }
        }
        None => SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
        },
    };

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        search_embedding_model: builder.search_embedding_model,
        search_callback: builder.search_callback.clone(),
        tool_callbacks: builder.tool_callbacks.clone(),
        tool_callbacks_with_tools: builder
            .tool_callbacks_with_tools
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    mistralrs_core::ToolCallbackWithTool {
                        callback: v.callback.clone(),
                        tool: v.tool.clone(),
                    },
                )
            })
            .collect(),
        no_kv_cache: false,
        no_prefix_cache: builder.prefix_cache_n.is_none(),
        prefix_cache_n: builder.prefix_cache_n.unwrap_or(16),
        disable_eos_stop: false,
    };

    // Create loader config for unload/reload support
    let device = builder
        .device
        .clone()
        .unwrap_or(best_device(builder.force_cpu).unwrap());
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_vision()));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = builder.from_uqff.as_ref().map(|paths| {
        paths
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(";")
    });

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::VisionPlain {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: from_uqff_str,
            max_edge: builder.max_edge,
            calibration_file: builder.calibration_file.clone(),
            imatrix: builder.imatrix.clone(),
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
            max_image_length: AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: builder.isq,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a GGUF model pipeline from a GgufModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_gguf_pipeline(
    builder: crate::GgufModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    let config = GGUFSpecificConfig {
        topology: builder.topology.clone(),
    };

    if builder.with_logging {
        initialize_logging();
    }

    let loader = GGUFLoaderBuilder::new(
        builder.chat_template.clone(),
        builder.tok_model_id.clone(),
        builder.model_id.clone(),
        builder.files.clone(),
        config,
        builder.no_kv_cache,
        builder.jinja_explicit.clone(),
    )
    .build();

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &ModelDType::Auto,
        &builder
            .device
            .clone()
            .unwrap_or(best_device(builder.force_cpu).unwrap()),
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        None,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = match builder.paged_attn_cfg {
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
                max_num_seqs: builder.max_num_seqs,
                config,
            }
        }
        None => SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
        },
    };

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        search_embedding_model: builder.search_embedding_model,
        search_callback: builder.search_callback.clone(),
        tool_callbacks: builder.tool_callbacks.clone(),
        tool_callbacks_with_tools: builder
            .tool_callbacks_with_tools
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    mistralrs_core::ToolCallbackWithTool {
                        callback: v.callback.clone(),
                        tool: v.tool.clone(),
                    },
                )
            })
            .collect(),
        no_kv_cache: builder.no_kv_cache,
        no_prefix_cache: builder.prefix_cache_n.is_none(),
        prefix_cache_n: builder.prefix_cache_n.unwrap_or(16),
        disable_eos_stop: false,
    };

    // Create loader config for unload/reload support
    let device = builder
        .device
        .clone()
        .unwrap_or(best_device(builder.force_cpu).unwrap());
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::GGUF {
            tok_model_id: builder.tok_model_id.clone(),
            quantized_model_id: builder.model_id.clone(),
            quantized_filename: builder.files.join(GGUF_MULTI_FILE_DELIMITER),
            dtype: ModelDType::Auto,
            topology: builder.topology_path.clone(),
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: ModelDType::Auto,
        device,
        device_map_setting,
        isq: None,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a diffusion model pipeline from a DiffusionModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_diffusion_pipeline(
    builder: crate::DiffusionModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    if builder.with_logging {
        initialize_logging();
    }

    let loader = DiffusionLoaderBuilder::new(Some(builder.model_id.clone()))
        .build(builder.loader_type.clone());

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &best_device(builder.force_cpu)?,
        !builder.with_logging,
        DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        None,
        None,
    )?;

    let scheduler_config = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
    };

    let engine_config = EngineConfig::default();

    // Create loader config for unload/reload support
    let device = best_device(builder.force_cpu)?;

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::DiffusionPlain {
            model_id: builder.model_id.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting: DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        isq: None,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a speech model pipeline from a SpeechModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_speech_pipeline(
    builder: crate::SpeechModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    if builder.with_logging {
        initialize_logging();
    }

    let loader = SpeechLoader {
        model_id: builder.model_id.clone(),
        dac_model_id: builder.dac_model_id.clone(),
        arch: builder.loader_type,
        cfg: builder.cfg,
    };

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &best_device(builder.force_cpu)?,
        !builder.with_logging,
        DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        None,
        None,
    )?;

    let scheduler_config = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
    };

    let engine_config = EngineConfig::default();

    // Create loader config for unload/reload support
    let device = best_device(builder.force_cpu)?;

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Speech {
            model_id: builder.model_id.clone(),
            dac_model_id: builder.dac_model_id.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting: DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        isq: None,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build an embedding model pipeline from an EmbeddingModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_embedding_pipeline(
    builder: crate::EmbeddingModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    let config = EmbeddingSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
    };

    if builder.with_logging {
        initialize_logging();
    }

    let loader = EmbeddingLoaderBuilder::new(
        config,
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
    )
    .build(builder.loader_type.clone());

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &builder
            .device
            .clone()
            .unwrap_or(best_device(builder.force_cpu).unwrap()),
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        builder.isq,
        None,
    )?;

    let scheduler_config = SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(builder.max_num_seqs.try_into()?),
    };

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        ..Default::default()
    };

    // Create loader config for unload/reload support
    let device = builder
        .device
        .clone()
        .unwrap_or(best_device(builder.force_cpu).unwrap());
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = builder.from_uqff.as_ref().map(|paths| {
        paths
            .iter()
            .map(|p| p.to_string_lossy())
            .collect::<Vec<_>>()
            .join(";")
    });

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Embedding {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: from_uqff_str,
            hf_cache_path: builder.hf_cache_path.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: builder.isq,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
    };

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}
