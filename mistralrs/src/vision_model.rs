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

use crate::model_builder_trait::{build_model_from_pipeline, build_vision_pipeline};
use crate::Model;

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
    pub(crate) organization: IsqOrganization,

    // Model running
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) loader_type: Option<VisionLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqSetting>,
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
            topology_path: None,
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
            organization: IsqOrganization::Default,
            prefix_cache_n: None,
        }
    }

    // Shared methods from builder_macros.rs
    common_builder_methods!();

    /// Manually set the model loader type. Otherwise, it will attempt to automatically
    /// determine the loader type.
    pub fn with_loader_type(mut self, loader_type: VisionLoaderType) -> Self {
        self.loader_type = Some(loader_type);
        self
    }

    #[deprecated(
        note = "Use `UqffVisionModelBuilder` to load a UQFF model instead of the generic `from_uqff`"
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
    pub fn with_max_edge(mut self, max_edge: u32) -> Self {
        self.max_edge = Some(max_edge);
        self
    }

    /// Load the vision model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_vision_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[derive(Clone)]
/// Configure a UQFF vision model with the various parameters for loading, running, and other inference behaviors.
/// This wraps and implements `DerefMut` for the VisionModelBuilder, so users should take care to not call UQFF-related methods.
pub struct UqffVisionModelBuilder(VisionModelBuilder);

impl UqffVisionModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Maximum number of sequences running is 32
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    ///
    /// For sharded UQFF models, you only need to specify the first shard file
    /// (e.g., `q4k-0.uqff`). The remaining shards are auto-discovered from the
    /// same directory or Hugging Face repository.
    pub fn new(model_id: impl ToString, uqff_file: Vec<PathBuf>) -> Self {
        let mut inner = VisionModelBuilder::new(model_id);
        inner.from_uqff = Some(uqff_file);
        Self(inner)
    }

    /// Load the UQFF vision model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        self.0.build().await
    }

    /// Unwrap into the inner [`VisionModelBuilder`]. Take care not to call UQFF-related methods on it.
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
