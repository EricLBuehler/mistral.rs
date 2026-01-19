//! Unified model builder API that simplifies model creation.
//!
//! This module provides a single `ModelBuilder` type that can create any type of model
//! (text, vision, embedding, diffusion, speech) with a consistent API.
//!
//! # Examples
//!
//! ```rust,ignore
//! use mistralrs::{ModelBuilder, IsqType, PagedAttentionMetaBuilder};
//!
//! // Simple text model with ISQ
//! let model = ModelBuilder::from_hf("microsoft/Phi-3.5-mini-instruct")
//!     .with_isq(IsqType::Q8_0)
//!     .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
//!     .build()
//!     .await?;
//!
//! // GGUF model
//! let model = ModelBuilder::from_gguf("TheBloke/Llama-2-7B-GGUF", vec!["llama-2-7b.Q4_K_M.gguf"])
//!     .with_logging()
//!     .build()
//!     .await?;
//!
//! // Vision model
//! let model = ModelBuilder::vision("llava-hf/llava-1.5-7b-hf")
//!     .with_isq(IsqType::Q4_0)
//!     .build()
//!     .await?;
//! ```

use candle_core::Device;
use mistralrs_core::*;
use std::path::PathBuf;

use crate::{best_device, Model};

/// Model source specification for the unified builder.
#[derive(Clone)]
pub enum ModelSource {
    /// Plain text model (safetensors/HF format)
    Plain {
        model_id: String,
        arch: Option<NormalLoaderType>,
    },
    /// Vision model
    Vision {
        model_id: String,
        arch: Option<VisionLoaderType>,
        max_edge: Option<u32>,
    },
    /// Embedding model
    Embedding {
        model_id: String,
        arch: Option<EmbeddingLoaderType>,
    },
    /// GGUF quantized model
    Gguf {
        model_id: String,
        files: Vec<String>,
        tok_model_id: Option<String>,
    },
    /// Diffusion model
    Diffusion {
        model_id: String,
        arch: DiffusionLoaderType,
    },
    /// Speech model
    Speech {
        model_id: String,
        arch: SpeechLoaderType,
        dac_model_id: Option<String>,
    },
}

/// Adapter configuration for LoRA/XLoRA.
#[derive(Clone, Default)]
pub enum AdapterConfig {
    #[default]
    None,
    /// LoRA adapter
    Lora {
        adapter_model_ids: Vec<String>,
    },
    /// XLoRA adapter
    XLora {
        xlora_model_id: String,
        order_file: PathBuf,
        tgt_non_granular_index: Option<usize>,
    },
}

/// Unified model builder that works with any model type.
///
/// This builder provides a consistent API for creating text, vision, embedding,
/// GGUF, diffusion, and speech models.
#[derive(Clone)]
pub struct ModelBuilder {
    source: ModelSource,
    adapter: AdapterConfig,

    // Common configuration
    token_source: TokenSource,
    hf_revision: Option<String>,
    dtype: ModelDType,
    device: Option<Device>,
    force_cpu: bool,
    device_mapping: Option<DeviceMapSetting>,
    isq: Option<IsqType>,
    paged_attn_cfg: Option<PagedAttentionConfig>,
    max_num_seqs: usize,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
    with_logging: bool,
    mcp_client_config: Option<McpClientConfig>,
    no_kv_cache: bool,
    prefix_cache_n: Option<usize>,
    topology: Option<Topology>,
    write_uqff: Option<PathBuf>,
    from_uqff: Option<Vec<PathBuf>>,
    imatrix: Option<PathBuf>,
    calibration_file: Option<PathBuf>,
    tokenizer_json: Option<String>,
    hf_cache_path: Option<PathBuf>,
    organization: IsqOrganization,
    throughput_logging: bool,
}

impl ModelBuilder {
    /// Create a new builder for a Hugging Face model (auto-detects type).
    ///
    /// This creates a plain text model by default. Use `.vision()`, `.embedding()`,
    /// etc. for other model types.
    pub fn from_hf(model_id: impl Into<String>) -> Self {
        Self {
            source: ModelSource::Plain {
                model_id: model_id.into(),
                arch: None,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    /// Create a builder for a GGUF quantized model.
    pub fn from_gguf(model_id: impl Into<String>, files: Vec<impl Into<String>>) -> Self {
        Self {
            source: ModelSource::Gguf {
                model_id: model_id.into(),
                files: files.into_iter().map(|f| f.into()).collect(),
                tok_model_id: None,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    /// Create a builder for a vision model.
    pub fn vision(model_id: impl Into<String>) -> Self {
        Self {
            source: ModelSource::Vision {
                model_id: model_id.into(),
                arch: None,
                max_edge: None,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    /// Create a builder for an embedding model.
    pub fn embedding(model_id: impl Into<String>) -> Self {
        Self {
            source: ModelSource::Embedding {
                model_id: model_id.into(),
                arch: None,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    /// Create a builder for a diffusion model.
    pub fn diffusion(model_id: impl Into<String>, arch: DiffusionLoaderType) -> Self {
        Self {
            source: ModelSource::Diffusion {
                model_id: model_id.into(),
                arch,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    /// Create a builder for a speech model.
    pub fn speech(model_id: impl Into<String>, arch: SpeechLoaderType) -> Self {
        Self {
            source: ModelSource::Speech {
                model_id: model_id.into(),
                arch,
                dac_model_id: None,
            },
            adapter: AdapterConfig::None,
            token_source: TokenSource::CacheToken,
            hf_revision: None,
            dtype: ModelDType::Auto,
            device: None,
            force_cpu: false,
            device_mapping: None,
            isq: None,
            paged_attn_cfg: None,
            max_num_seqs: 16,
            chat_template: None,
            jinja_explicit: None,
            with_logging: false,
            mcp_client_config: None,
            no_kv_cache: false,
            prefix_cache_n: None,
            topology: None,
            write_uqff: None,
            from_uqff: None,
            imatrix: None,
            calibration_file: None,
            tokenizer_json: None,
            hf_cache_path: None,
            organization: IsqOrganization::Default,
            throughput_logging: false,
        }
    }

    // Configuration methods

    /// Add LoRA adapters to the model.
    pub fn with_lora(mut self, adapter_model_ids: Vec<impl Into<String>>) -> Self {
        self.adapter = AdapterConfig::Lora {
            adapter_model_ids: adapter_model_ids.into_iter().map(|s| s.into()).collect(),
        };
        self
    }

    /// Add XLoRA adapter to the model.
    pub fn with_xlora(
        mut self,
        xlora_model_id: impl Into<String>,
        order_file: impl Into<PathBuf>,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.adapter = AdapterConfig::XLora {
            xlora_model_id: xlora_model_id.into(),
            order_file: order_file.into(),
            tgt_non_granular_index,
        };
        self
    }

    /// Set in-situ quantization type.
    pub fn with_isq(mut self, isq: IsqType) -> Self {
        self.isq = Some(isq);
        self
    }

    /// Set the model data type.
    pub fn with_dtype(mut self, dtype: ModelDType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Enable paged attention with the provided configuration.
    pub fn with_paged_attn(
        mut self,
        cfg: impl FnOnce() -> anyhow::Result<PagedAttentionConfig>,
    ) -> anyhow::Result<Self> {
        self.paged_attn_cfg = Some(cfg()?);
        Ok(self)
    }

    /// Set the device to run the model on.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Force CPU execution.
    pub fn with_cpu(mut self) -> Self {
        self.force_cpu = true;
        self
    }

    /// Enable logging.
    pub fn with_logging(mut self) -> Self {
        self.with_logging = true;
        self
    }

    /// Set a custom chat template.
    pub fn with_chat_template(mut self, template: impl Into<String>) -> Self {
        self.chat_template = Some(template.into());
        self
    }

    /// Set explicit Jinja template path.
    pub fn with_jinja_explicit(mut self, path: impl Into<String>) -> Self {
        self.jinja_explicit = Some(path.into());
        self
    }

    /// Set MCP client configuration.
    pub fn with_mcp_client(mut self, config: McpClientConfig) -> Self {
        self.mcp_client_config = Some(config);
        self
    }

    /// Set the token source.
    pub fn with_token_source(mut self, source: TokenSource) -> Self {
        self.token_source = source;
        self
    }

    /// Set the HF revision.
    pub fn with_hf_revision(mut self, revision: impl Into<String>) -> Self {
        self.hf_revision = Some(revision.into());
        self
    }

    /// Set the maximum number of sequences.
    pub fn with_max_num_seqs(mut self, max_num_seqs: usize) -> Self {
        self.max_num_seqs = max_num_seqs;
        self
    }

    /// Disable the KV cache.
    pub fn with_no_kv_cache(mut self) -> Self {
        self.no_kv_cache = true;
        self
    }

    /// Set the prefix cache size.
    pub fn with_prefix_cache_n(mut self, n: usize) -> Self {
        self.prefix_cache_n = Some(n);
        self
    }

    /// Set device mapping.
    pub fn with_device_mapping(mut self, mapping: DeviceMapSetting) -> Self {
        self.device_mapping = Some(mapping);
        self
    }

    /// Set topology from path.
    pub fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Set ISQ organization.
    pub fn with_isq_organization(mut self, organization: IsqOrganization) -> Self {
        self.organization = organization;
        self
    }

    /// Enable throughput logging.
    pub fn with_throughput_logging(mut self) -> Self {
        self.throughput_logging = true;
        self
    }

    /// Set UQFF write path.
    pub fn with_write_uqff(mut self, path: impl Into<PathBuf>) -> Self {
        self.write_uqff = Some(path.into());
        self
    }

    /// Set UQFF load paths.
    pub fn with_from_uqff(mut self, paths: Vec<impl Into<PathBuf>>) -> Self {
        self.from_uqff = Some(paths.into_iter().map(|p| p.into()).collect());
        self
    }

    /// Set imatrix path.
    pub fn with_imatrix(mut self, path: impl Into<PathBuf>) -> Self {
        self.imatrix = Some(path.into());
        self
    }

    /// Set calibration file path.
    pub fn with_calibration_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.calibration_file = Some(path.into());
        self
    }

    /// Set custom tokenizer JSON path.
    pub fn with_tokenizer_json(mut self, path: impl Into<String>) -> Self {
        self.tokenizer_json = Some(path.into());
        self
    }

    /// Set HF cache path.
    pub fn with_hf_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.hf_cache_path = Some(path.into());
        self
    }

    /// Build and load the model.
    pub async fn build(self) -> anyhow::Result<Model> {
        if self.with_logging {
            initialize_logging();
        }

        let device = self
            .device
            .unwrap_or_else(|| best_device(self.force_cpu).unwrap());

        // Build the loader based on source type
        let loader: Box<dyn Loader> = match &self.source {
            ModelSource::Plain { model_id, arch } => {
                let mut builder = NormalLoaderBuilder::new(
                    NormalSpecificConfig {
                        topology: self.topology.clone(),
                        organization: self.organization,
                        write_uqff: self.write_uqff.clone(),
                        from_uqff: self.from_uqff.clone(),
                        imatrix: self.imatrix.clone(),
                        calibration_file: self.calibration_file.clone(),
                        hf_cache_path: self.hf_cache_path.clone(),
                        matformer_config_path: None,
                        matformer_slice_name: None,
                    },
                    self.chat_template.clone(),
                    self.tokenizer_json.clone(),
                    Some(model_id.clone()),
                    self.no_kv_cache,
                    self.jinja_explicit.clone(),
                );

                // Apply adapter if configured
                match &self.adapter {
                    AdapterConfig::None => {}
                    AdapterConfig::Lora { adapter_model_ids } => {
                        builder = builder.with_lora(adapter_model_ids.clone());
                    }
                    AdapterConfig::XLora {
                        xlora_model_id,
                        order_file,
                        tgt_non_granular_index,
                    } => {
                        let order = serde_json::from_reader(std::fs::File::open(order_file)?)?;
                        builder = builder.with_xlora(
                            xlora_model_id.clone(),
                            order,
                            self.no_kv_cache,
                            *tgt_non_granular_index,
                        );
                    }
                }

                builder.build(arch.clone())?
            }
            ModelSource::Vision {
                model_id,
                arch,
                max_edge,
            } => VisionLoaderBuilder::new(
                VisionSpecificConfig {
                    topology: self.topology.clone(),
                    write_uqff: self.write_uqff.clone(),
                    from_uqff: self.from_uqff.clone(),
                    max_edge: *max_edge,
                    calibration_file: self.calibration_file.clone(),
                    imatrix: self.imatrix.clone(),
                    hf_cache_path: self.hf_cache_path.clone(),
                    matformer_config_path: None,
                    matformer_slice_name: None,
                },
                self.chat_template.clone(),
                self.tokenizer_json.clone(),
                Some(model_id.clone()),
                self.jinja_explicit.clone(),
            )
            .build(arch.clone()),
            ModelSource::Embedding { model_id, arch } => EmbeddingLoaderBuilder::new(
                EmbeddingSpecificConfig {
                    topology: self.topology.clone(),
                    write_uqff: self.write_uqff.clone(),
                    from_uqff: self.from_uqff.clone(),
                    hf_cache_path: self.hf_cache_path.clone(),
                },
                self.tokenizer_json.clone(),
                Some(model_id.clone()),
            )
            .build(arch.clone()),
            ModelSource::Gguf {
                model_id,
                files,
                tok_model_id,
            } => {
                let mut builder = GGUFLoaderBuilder::new(
                    self.chat_template.clone(),
                    tok_model_id.clone(),
                    model_id.clone(),
                    files.clone(),
                    GGUFSpecificConfig {
                        topology: self.topology.clone(),
                    },
                    self.no_kv_cache,
                    self.jinja_explicit.clone(),
                );

                // Apply adapter if configured
                match &self.adapter {
                    AdapterConfig::None => {}
                    AdapterConfig::Lora { adapter_model_ids } => {
                        // GGUF LoRA expects a single model ID
                        if let Some(first_id) = adapter_model_ids.first() {
                            let order =
                                serde_json::from_str("{}")?; // Placeholder - would need actual order
                            builder = builder.with_lora(first_id.clone(), order);
                        }
                    }
                    AdapterConfig::XLora {
                        xlora_model_id,
                        order_file,
                        tgt_non_granular_index,
                    } => {
                        let order = serde_json::from_reader(std::fs::File::open(order_file)?)?;
                        builder = builder.with_xlora(
                            xlora_model_id.clone(),
                            order,
                            self.no_kv_cache,
                            *tgt_non_granular_index,
                        );
                    }
                }

                builder.build()
            }
            ModelSource::Diffusion { model_id, arch } => {
                DiffusionLoaderBuilder::new(Some(model_id.clone())).build(arch.clone())
            }
            ModelSource::Speech {
                model_id,
                arch,
                dac_model_id,
            } => Box::new(SpeechLoader {
                model_id: model_id.clone(),
                dac_model_id: dac_model_id.clone(),
                arch: *arch,
                cfg: None,
            }),
        };

        // Set up device mapping
        let device_mapping = self
            .device_mapping
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

        // Load the model
        let pipeline = loader.load_model_from_hf(
            self.hf_revision.clone(),
            self.token_source.clone(),
            &self.dtype,
            &device,
            !self.with_logging,
            device_mapping,
            self.isq,
            self.paged_attn_cfg.clone(),
        )?;

        // Build the scheduler config
        let scheduler_config = if self.paged_attn_cfg.is_some() {
            if let Some(ref cache_config) = pipeline.blocking_lock().get_metadata().cache_config {
                SchedulerConfig::PagedAttentionMeta {
                    max_num_seqs: self.max_num_seqs,
                    config: cache_config.clone(),
                }
            } else {
                SchedulerConfig::DefaultScheduler {
                    method: DefaultSchedulerMethod::Fixed(
                        self.max_num_seqs.try_into().unwrap_or(
                            std::num::NonZeroUsize::new(16).expect("16 > 0"),
                        ),
                    ),
                }
            }
        } else {
            SchedulerConfig::DefaultScheduler {
                method: DefaultSchedulerMethod::Fixed(
                    self.max_num_seqs
                        .try_into()
                        .unwrap_or(std::num::NonZeroUsize::new(16).expect("16 > 0")),
                ),
            }
        };

        // Build MistralRs instance
        let mut builder = MistralRsBuilder::new(pipeline, scheduler_config, self.throughput_logging, None);

        if self.no_kv_cache {
            builder = builder.with_no_kv_cache(true);
        }

        if let Some(n) = self.prefix_cache_n {
            builder = builder.with_prefix_cache_n(n);
        }

        if let Some(mcp_config) = self.mcp_client_config {
            builder = builder.with_mcp_client(mcp_config);
        }

        let runner = builder.build().await;

        Ok(Model::new(runner))
    }
}
