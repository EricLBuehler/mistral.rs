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
    builders: Vec<(String, AnyModelBuilder)>,
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

    /// Add a model with optional custom ID.
    /// If `model_id` is `None`, the builder's default model ID will be used.
    pub fn add_model<B: Into<AnyModelBuilder>>(
        mut self,
        builder: B,
        model_id: Option<String>,
    ) -> Self {
        let builder = builder.into();
        let id = model_id.unwrap_or_else(|| builder.model_id());
        self.builders.push((id, builder));
        self
    }

    /// Set the default model.
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
        let (first_model_id, first_builder) = builders_iter.next().unwrap();

        let (pipeline, scheduler_config, add_model_config) = first_builder.build_pipeline().await?;

        // Create the MistralRsBuilder for the first model
        let mut runner_builder = mistralrs_core::MistralRsBuilder::new(
            pipeline,
            scheduler_config,
            add_model_config.engine_config.throughput_logging_enabled,
            add_model_config
                .engine_config
                .search_embedding_model
                .clone(),
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

        if let Some(mcp_config) = add_model_config.mcp_client_config {
            runner_builder = runner_builder.with_mcp_client(mcp_config);
        }

        runner_builder = runner_builder
            .with_no_kv_cache(add_model_config.engine_config.no_kv_cache)
            .with_no_prefix_cache(add_model_config.engine_config.no_prefix_cache)
            .with_prefix_cache_n(add_model_config.engine_config.prefix_cache_n);

        let mistralrs = runner_builder.build().await;

        // Add remaining models
        for (model_id, builder) in builders_iter {
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
        } else {
            // Set the first model as default if not already set
            mistralrs
                .set_default_model_id(&first_model_id)
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        Ok(Model::new(mistralrs))
    }
}

// Pipeline building functions for each model type

async fn build_text_pipeline(
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
    .build(builder.loader_type)?;

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
        builder.paged_attn_cfg.clone(),
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
        search_embedding_model: builder.search_embedding_model.clone(),
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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: builder.mcp_client_config.clone(),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

async fn build_vision_pipeline(
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
    .build(builder.loader_type);

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
        builder.paged_attn_cfg.clone(),
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
        search_embedding_model: builder.search_embedding_model.clone(),
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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

async fn build_gguf_pipeline(
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
        builder.paged_attn_cfg.clone(),
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
        search_embedding_model: builder.search_embedding_model.clone(),
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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

async fn build_diffusion_pipeline(
    builder: crate::DiffusionModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use crate::best_device;
    use mistralrs_core::*;

    if builder.with_logging {
        initialize_logging();
    }

    let loader =
        DiffusionLoaderBuilder::new(Some(builder.model_id.clone())).build(builder.loader_type);

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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

async fn build_speech_pipeline(
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
        arch: builder.loader_type.clone(),
        cfg: builder.cfg.clone(),
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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

async fn build_embedding_pipeline(
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
    .build(builder.loader_type);

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

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}
