//! Multi-model builder and pipeline construction utilities.

use candle_core::Device;
use mistralrs_core::{
    AddModelConfig, DefaultSchedulerMethod, EngineConfig, IsqType, Pipeline, SchedulerConfig,
    SearchCallback, SearchEmbeddingModel, ToolCallbackWithTool,
};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

use crate::Model;

/// Enum representing all possible model builders that can be used with [`MultiModelBuilder`].
pub enum AnyModelBuilder {
    /// A text model builder.
    Text(crate::TextModelBuilder),
    /// A multimodal model builder.
    Multimodal(crate::MultimodalModelBuilder),
    /// An auto-detecting model builder.
    Auto(crate::ModelBuilder),
    /// A GGUF model builder.
    Gguf(crate::GgufModelBuilder),
    /// A diffusion (image generation) model builder.
    Diffusion(crate::DiffusionModelBuilder),
    #[cfg(feature = "audio")]
    /// A speech synthesis model builder.
    Speech(crate::SpeechModelBuilder),
    /// An embedding model builder.
    Embedding(crate::EmbeddingModelBuilder),
}

impl AnyModelBuilder {
    /// Get the default model ID for this builder.
    pub fn model_id(&self) -> String {
        match self {
            AnyModelBuilder::Text(b) => b.model_id.clone(),
            AnyModelBuilder::Multimodal(b) => b.model_id.clone(),
            AnyModelBuilder::Auto(b) => b.model_id.clone(),
            AnyModelBuilder::Gguf(b) => b.model_id.clone(),
            AnyModelBuilder::Diffusion(b) => b.model_id.clone(),
            #[cfg(feature = "audio")]
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
            AnyModelBuilder::Multimodal(b) => build_multimodal_pipeline(b).await,
            AnyModelBuilder::Auto(b) => build_auto_pipeline(b).await,
            AnyModelBuilder::Gguf(b) => build_gguf_pipeline(b).await,
            AnyModelBuilder::Diffusion(b) => build_diffusion_pipeline(b).await,
            #[cfg(feature = "audio")]
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

impl From<crate::MultimodalModelBuilder> for AnyModelBuilder {
    fn from(b: crate::MultimodalModelBuilder) -> Self {
        AnyModelBuilder::Multimodal(b)
    }
}

impl From<crate::ModelBuilder> for AnyModelBuilder {
    fn from(b: crate::ModelBuilder) -> Self {
        AnyModelBuilder::Auto(b)
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

#[cfg(feature = "audio")]
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

struct MultiModelEntry {
    builder: AnyModelBuilder,
    alias: Option<String>,
}

/// Builder for creating a Model with multiple models.
pub struct MultiModelBuilder {
    builders: Vec<MultiModelEntry>,
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

    /// Add a model. The model ID will be the pipeline name (e.g., "google/gemma-4-E4B-it").
    pub fn add_model<B: Into<AnyModelBuilder>>(mut self, builder: B) -> Self {
        self.builders.push(MultiModelEntry {
            builder: builder.into(),
            alias: None,
        });
        self
    }

    /// Add a model with a custom alias (nickname) used for API requests.
    pub fn add_model_with_alias<B: Into<AnyModelBuilder>>(
        mut self,
        alias: impl Into<String>,
        builder: B,
    ) -> Self {
        self.builders.push(MultiModelEntry {
            builder: builder.into(),
            alias: Some(alias.into()),
        });
        self
    }

    /// Set the default model by its model ID or alias.
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
        let first_entry = builders_iter.next().unwrap();

        let (pipeline, scheduler_config, add_model_config) =
            first_entry.builder.build_pipeline().await?;
        let pipeline_name = pipeline.lock().await.name();
        let primary_id = first_entry
            .alias
            .clone()
            .unwrap_or_else(|| pipeline_name.clone());

        // Create the MistralRsBuilder for the first model
        let mut runner_builder = mistralrs_core::MistralRsBuilder::new(
            pipeline,
            scheduler_config,
            add_model_config.engine_config.throughput_logging_enabled,
            add_model_config.engine_config.search_embedding_model,
        );
        if primary_id != pipeline_name {
            runner_builder = runner_builder.with_model_id(primary_id.clone());
        }

        if let Some(cb) = add_model_config.engine_config.search_callback.clone() {
            runner_builder = runner_builder.with_search_callback(cb);
        }

        for (name, callback_with_tool) in &add_model_config.engine_config.tool_callbacks {
            runner_builder = runner_builder.with_tool_callback_and_tool(
                name.clone(),
                callback_with_tool.callback.clone(),
                callback_with_tool.tool.clone(),
            );
        }

        #[cfg(feature = "mcp")]
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

        if let Some(alias) = first_entry.alias {
            if alias != pipeline_name {
                mistralrs
                    .register_model_alias(pipeline_name.clone(), &primary_id)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }

        // Add remaining models using their pipeline names as IDs (or aliases when provided)
        for entry in builders_iter {
            let (pipeline, scheduler_config, add_model_config) =
                entry.builder.build_pipeline().await?;
            let pipeline_name = pipeline.lock().await.name();
            let primary_id = entry.alias.clone().unwrap_or_else(|| pipeline_name.clone());
            mistralrs
                .add_model(
                    primary_id.clone(),
                    pipeline,
                    scheduler_config,
                    add_model_config,
                )
                .await
                .map_err(|e| anyhow::anyhow!(e))?;

            if let Some(alias) = entry.alias {
                if alias != pipeline_name {
                    mistralrs
                        .register_model_alias(pipeline_name.clone(), &primary_id)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
            }
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

pub(crate) fn maybe_initialize_logging(with_logging: bool) {
    if with_logging {
        mistralrs_core::initialize_logging();
    }
}

pub(crate) fn resolve_device(force_cpu: bool, device: Option<Device>) -> anyhow::Result<Device> {
    Ok(device
        .map(Ok)
        .unwrap_or_else(|| crate::best_device(force_cpu))?)
}

pub(crate) fn resolve_isq_type(
    isq: Option<&crate::IsqSetting>,
    device: &Device,
) -> anyhow::Result<Option<IsqType>> {
    isq.map(|setting| crate::resolve_isq(setting, device))
        .transpose()
}

pub(crate) fn default_scheduler_config(max_num_seqs: usize) -> anyhow::Result<SchedulerConfig> {
    Ok(SchedulerConfig::DefaultScheduler {
        method: DefaultSchedulerMethod::Fixed(max_num_seqs.try_into()?),
    })
}

pub(crate) async fn scheduler_config_from_pipeline<P>(
    pipeline: &Arc<Mutex<P>>,
    paged_attn_requested: bool,
    max_num_seqs: usize,
) -> anyhow::Result<SchedulerConfig>
where
    P: ?Sized + Pipeline,
{
    if paged_attn_requested {
        if let Some(config) = pipeline
            .lock()
            .await
            .get_metadata()
            .cache_config
            .as_ref()
            .cloned()
        {
            return Ok(SchedulerConfig::PagedAttentionMeta {
                max_num_seqs,
                config,
            });
        }
    }

    default_scheduler_config(max_num_seqs)
}

pub(crate) fn build_engine_config(
    throughput_logging_enabled: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
    search_callback: Option<Arc<SearchCallback>>,
    tool_callbacks: &HashMap<String, ToolCallbackWithTool>,
    no_kv_cache: bool,
    prefix_cache_n: Option<usize>,
) -> EngineConfig {
    EngineConfig {
        throughput_logging_enabled,
        search_embedding_model,
        search_callback,
        tool_callbacks: tool_callbacks.clone(),
        no_kv_cache,
        no_prefix_cache: prefix_cache_n.is_none(),
        prefix_cache_n: prefix_cache_n.unwrap_or(16),
        disable_eos_stop: false,
    }
}

pub(crate) fn join_path_list(paths: Option<&[PathBuf]>, delimiter: &str) -> Option<String> {
    paths.map(|paths| {
        paths
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(delimiter)
    })
}

pub(crate) async fn build_pipeline_from_text_loader(
    builder: crate::TextModelBuilder,
    loader: Box<dyn mistralrs_core::Loader>,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use mistralrs_core::*;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );
    let mcp_client_config = builder.mcp_client_config.clone();
    let device = resolve_device(builder.force_cpu, None)?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;
    let device_map_setting =
        builder
            .device_mapping
            .clone()
            .unwrap_or(mistralrs_core::DeviceMapSetting::Auto(
                mistralrs_core::AutoDeviceMapParams::default_text(),
            ));
    let paged_attn_requested = builder.paged_attn_cfg.is_some();

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision,
        builder.token_source,
        &builder.dtype,
        &device,
        !builder.with_logging,
        device_map_setting,
        isq_type,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config =
        scheduler_config_from_pipeline(&pipeline, paged_attn_requested, builder.max_num_seqs)
            .await?;

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config,
        loader_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

pub(crate) async fn build_pipeline_from_gguf_loader(
    builder: crate::GgufModelBuilder,
    loader: Box<dyn mistralrs_core::Loader>,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use mistralrs_core::*;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );
    let device = resolve_device(builder.force_cpu, None)?;
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));
    let paged_attn_requested = builder.paged_attn_cfg.is_some();

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision,
        builder.token_source,
        &ModelDType::Auto,
        &device,
        !builder.with_logging,
        device_map_setting,
        None,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config =
        scheduler_config_from_pipeline(&pipeline, paged_attn_requested, builder.max_num_seqs)
            .await?;

    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: None,
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

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

    for (name, callback_with_tool) in &add_model_config.engine_config.tool_callbacks {
        runner_builder = runner_builder.with_tool_callback_and_tool(
            name.clone(),
            callback_with_tool.callback.clone(),
            callback_with_tool.tool.clone(),
        );
    }

    #[cfg(feature = "mcp")]
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

    maybe_initialize_logging(builder.with_logging);

    let loader = NormalLoaderBuilder::new(
        config,
        builder.chat_template.clone(),
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
        builder.no_kv_cache,
        builder.jinja_explicit.clone(),
    )
    .build(builder.loader_type.clone())?;

    let device = resolve_device(builder.force_cpu, None)?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        isq_type,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = scheduler_config_from_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
    )
    .await?;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );

    // Create loader config for unload/reload support
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER);

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
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        #[cfg(feature = "mcp")]
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a multimodal model pipeline from a MultimodalModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_multimodal_pipeline(
    builder: crate::MultimodalModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use mistralrs_core::*;

    let config = MultimodalSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        max_edge: builder.max_edge,
        calibration_file: builder.calibration_file.clone(),
        imatrix: builder.imatrix.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
        organization: builder.organization,
    };

    maybe_initialize_logging(builder.with_logging);

    let loader = MultimodalLoaderBuilder::new(
        config,
        builder.chat_template.clone(),
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
        builder.jinja_explicit.clone(),
    )
    .build(builder.loader_type.clone());

    let device = resolve_device(builder.force_cpu, None)?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(
                AutoDeviceMapParams::default_multimodal(),
            )),
        isq_type,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = scheduler_config_from_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
    )
    .await?;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        false,
        builder.prefix_cache_n,
    );

    // Create loader config for unload/reload support
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(
            AutoDeviceMapParams::default_multimodal(),
        ));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER);

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::MultimodalPlain {
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
            organization: Some(builder.organization),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        #[cfg(feature = "mcp")]
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
    use mistralrs_core::*;

    let config = GGUFSpecificConfig {
        topology: builder.topology.clone(),
    };

    maybe_initialize_logging(builder.with_logging);

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

    let device = resolve_device(builder.force_cpu, None)?;
    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &ModelDType::Auto,
        &device,
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        None,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = scheduler_config_from_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
    )
    .await?;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );

    // Create loader config for unload/reload support
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
        #[cfg(feature = "mcp")]
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
    use mistralrs_core::*;

    maybe_initialize_logging(builder.with_logging);

    let loader = DiffusionLoaderBuilder::new(Some(builder.model_id.clone()))
        .build(builder.loader_type.clone());

    let device = resolve_device(builder.force_cpu, None)?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        None,
        None,
    )?;

    let scheduler_config = default_scheduler_config(builder.max_num_seqs)?;

    let engine_config = EngineConfig::default();

    // Create loader config for unload/reload support
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
        #[cfg(feature = "mcp")]
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

#[cfg(feature = "audio")]
/// Build a speech model pipeline from a SpeechModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_speech_pipeline(
    builder: crate::SpeechModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use mistralrs_core::*;

    maybe_initialize_logging(builder.with_logging);

    let loader = SpeechLoader {
        model_id: builder.model_id.clone(),
        dac_model_id: builder.dac_model_id.clone(),
        arch: builder.loader_type,
        cfg: builder.cfg,
    };

    let device = resolve_device(builder.force_cpu, None)?;
    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        None,
        None,
    )?;

    let scheduler_config = default_scheduler_config(builder.max_num_seqs)?;

    let engine_config = EngineConfig::default();

    // Create loader config for unload/reload support
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
        #[cfg(feature = "mcp")]
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
    use mistralrs_core::*;

    let config = EmbeddingSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
    };

    maybe_initialize_logging(builder.with_logging);

    let loader = EmbeddingLoaderBuilder::new(
        config,
        builder.tokenizer_json.clone(),
        Some(builder.model_id.clone()),
    )
    .build(builder.loader_type.clone());

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        isq_type,
        None,
    )?;

    let scheduler_config = default_scheduler_config(builder.max_num_seqs)?;

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        ..Default::default()
    };

    // Create loader config for unload/reload support
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER);

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
        isq: isq_type,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
    };

    let add_model_config = AddModelConfig {
        engine_config,
        #[cfg(feature = "mcp")]
        mcp_client_config: None,
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a model pipeline using auto-detection from a ModelBuilder.
/// This uses `AutoLoaderBuilder` to detect the model type (text, multimodal, embedding, etc.)
/// from the model's config.json, similar to the CLI `run` command.
pub async fn build_auto_pipeline(
    builder: crate::ModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use mistralrs_core::*;

    let normal_config = NormalSpecificConfig {
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

    let vision_config = MultimodalSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        max_edge: builder.max_edge,
        calibration_file: builder.calibration_file.clone(),
        imatrix: builder.imatrix.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
        organization: builder.organization,
    };

    let embedding_config = EmbeddingSpecificConfig {
        topology: builder.topology.clone(),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: builder.from_uqff.clone(),
        hf_cache_path: builder.hf_cache_path.clone(),
    };

    maybe_initialize_logging(builder.with_logging);

    let auto_builder = AutoLoaderBuilder::new(
        normal_config,
        vision_config,
        embedding_config,
        builder.chat_template.clone(),
        builder.tokenizer_json.clone(),
        builder.model_id.clone(),
        builder.no_kv_cache,
        builder.jinja_explicit.clone(),
    );
    let auto_builder = if let Some(ref path) = builder.hf_cache_path {
        auto_builder.hf_cache_path(path.clone())
    } else {
        auto_builder
    };
    let loader = auto_builder.build();

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;

    let pipeline = loader.load_model_from_hf(
        builder.hf_revision.clone(),
        builder.token_source.clone(),
        &builder.dtype,
        &device,
        !builder.with_logging,
        builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        isq_type,
        builder.paged_attn_cfg,
    )?;

    let scheduler_config = scheduler_config_from_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
    )
    .await?;

    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );

    // Convert from_uqff Vec<PathBuf> to semicolon-separated string if present
    let from_uqff_str = join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER);

    // Create loader config using ModelSelected::Run for auto-detection on reload
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Run {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            organization: Some(builder.organization),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: from_uqff_str,
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            max_edge: builder.max_edge,
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
    };

    let add_model_config = AddModelConfig {
        engine_config,
        #[cfg(feature = "mcp")]
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}
