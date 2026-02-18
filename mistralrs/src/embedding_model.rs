use candle_core::Device;
use mistralrs_core::*;

use crate::{IsqBits, IsqSetting};
use std::{
    ops::{Deref, DerefMut},
    path::PathBuf,
};

use crate::model_builder_trait::{build_embedding_pipeline, build_model_from_pipeline};
use crate::Model;

#[derive(Clone)]
/// Configure an embedding model with the various parameters for loading, running, and other inference behaviors.
pub struct EmbeddingModelBuilder {
    // Loading model
    pub(crate) model_id: String,
    pub(crate) token_source: TokenSource,
    pub(crate) hf_revision: Option<String>,
    pub(crate) write_uqff: Option<PathBuf>,
    pub(crate) from_uqff: Option<Vec<PathBuf>>,
    pub(crate) tokenizer_json: Option<String>,
    pub(crate) device_mapping: Option<DeviceMapSetting>,
    pub(crate) hf_cache_path: Option<PathBuf>,
    pub(crate) device: Option<Device>,

    // Model running
    pub(crate) topology: Option<Topology>,
    pub(crate) topology_path: Option<String>,
    pub(crate) loader_type: Option<EmbeddingLoaderType>,
    pub(crate) dtype: ModelDType,
    pub(crate) force_cpu: bool,
    pub(crate) isq: Option<IsqSetting>,
    pub(crate) throughput_logging: bool,

    // Other things
    pub(crate) max_num_seqs: usize,
    pub(crate) with_logging: bool,
}

impl EmbeddingModelBuilder {
    /// A few defaults are applied here:
    /// - Maximum number of sequences running is 32
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(model_id: impl ToString) -> Self {
        Self {
            model_id: model_id.to_string(),
            topology: None,
            topology_path: None,
            write_uqff: None,
            from_uqff: None,
            tokenizer_json: None,
            loader_type: None,
            dtype: ModelDType::Auto,
            force_cpu: false,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            isq: None,
            max_num_seqs: 32,
            with_logging: false,
            device_mapping: None,
            throughput_logging: false,
            hf_cache_path: None,
            device: None,
        }
    }

    /// Enable runner throughput logging.
    pub fn with_throughput_logging(mut self) -> Self {
        self.throughput_logging = true;
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

    /// Path to a discrete `tokenizer.json` file.
    pub fn with_tokenizer_json(mut self, tokenizer_json: impl ToString) -> Self {
        self.tokenizer_json = Some(tokenizer_json.to_string());
        self
    }

    /// Manually set the model loader type. Otherwise, it will attempt to automatically
    /// determine the loader type.
    pub fn with_loader_type(mut self, loader_type: EmbeddingLoaderType) -> Self {
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

    /// Set the maximum number of sequences which can be run at once.
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
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
        note = "Use `UqffEmbeddingModelBuilder` to load a UQFF model instead of the generic `from_uqff`"
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
    pub fn from_hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    /// Set the main device to load this model onto. Automatic device mapping will be performed starting with this device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Load the embedding model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        let (pipeline, scheduler_config, add_model_config) = build_embedding_pipeline(self).await?;
        Ok(build_model_from_pipeline(pipeline, scheduler_config, add_model_config).await)
    }
}

#[derive(Clone)]
/// Configure a UQFF embedding model with the various parameters for loading, running, and other inference behaviors.
/// This wraps and implements `DerefMut` for the UqffEmbeddingModelBuilder, so users should take care to not call UQFF-related methods.
pub struct UqffEmbeddingModelBuilder(EmbeddingModelBuilder);

impl UqffEmbeddingModelBuilder {
    /// A few defaults are applied here:
    /// - Token source is from the cache (.cache/huggingface/token)
    /// - Automatic device mapping with model defaults according to `AutoDeviceMapParams`
    pub fn new(model_id: impl ToString, uqff_file: Vec<PathBuf>) -> Self {
        let mut inner = EmbeddingModelBuilder::new(model_id);
        inner.from_uqff = Some(uqff_file);
        Self(inner)
    }

    /// Load the UQFF embedding model and return a ready-to-use [`Model`].
    pub async fn build(self) -> anyhow::Result<Model> {
        self.0.build().await
    }

    /// Unwrap into the inner [`EmbeddingModelBuilder`]. Take care not to call UQFF-related methods on it.
    pub fn into_inner(self) -> EmbeddingModelBuilder {
        self.0
    }
}

impl Deref for UqffEmbeddingModelBuilder {
    type Target = EmbeddingModelBuilder;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for UqffEmbeddingModelBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<UqffEmbeddingModelBuilder> for EmbeddingModelBuilder {
    fn from(value: UqffEmbeddingModelBuilder) -> Self {
        value.0
    }
}
