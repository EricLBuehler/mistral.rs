use super::llg::build_llg_factory;
use super::{
    get_model_paths, text_models_inputs_processor::ModelInputs, AdapterKind, CacheManager,
    GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, QuantizationKind, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::WorkerTransferData;
use crate::gguf::{
    base_model::infer_hf_base_model_id,
    convert_gguf_metadata_to_hf_tokenizer,
    gemma3_bindings::build_gemma3_text_bindings,
    gemma3_config::{
        ensure_gemma3_vision_config, gemma3_text_uses_language_model_prefix,
        prepare_gemma3_text_config,
    },
    get_gguf_chat_template, get_gguf_chat_template_from_metadata,
    multimodal_bindings::build_gemma4_bindings,
    multimodal_vision_registry::resolve_native_multimodal_gguf,
    muse_glimmer_bindings::normalize_muse_glimmer_config,
    normal_bindings::build_normal_bindings,
    normal_config::{
        normal_loader_hint_from_external_config, normalize_external_normal_config,
        synthesize_normal_config, validate_normal_config_tensor_inventory,
    },
    normal_registry::{resolve_native_adapter, GgufDescriptor, RopePairing},
    qwen_multimodal_bindings::{
        build_qwen_multimodal_bindings, normalize_qwen_multimodal_config,
        qwen_multimodal_loader_type,
    },
    validate_external_gguf_tokenizer, GgufTokenizerConversion,
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::kv_cache::FullCacheManager;
use crate::lora::Ordering;
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::hf::{build_api, get_file, list_repo_files};
use crate::pipeline::loaders::{stamp_qk_rope_layout, DeviceMappedModelLoader};
use crate::pipeline::multimodal::{
    MultimodalLoaderBuilder, MultimodalSpecificConfig, PreparedMultimodalSource,
};
use crate::pipeline::normal::{NormalLoaderBuilder, NormalSpecificConfig, PreparedNormalSource};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::xlora_models::{XLoraQLlama, XLoraQPhi3};
use crate::{
    distributed, get_mut_arcmutex, get_paths_gguf, DeviceMapSetting, LocalModelPaths,
    LoraAdapterSpec, LoraRuntimeConfig, MultimodalLoaderType, PagedAttentionConfig, Pipeline,
    Topology, TryIntoDType, UqffWriteConfig, GLOBAL_HF_CACHE,
};
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use either::Either;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

const PROJECTOR_REQUIRED_ARCHITECTURES: &[&str] = &[
    "gemma3n",
    "gemma4",
    "llama4",
    "muse-glimmer",
    "qwen2vl",
    "qwen3vl",
    "qwen3vlmoe",
];

fn validate_native_dynamic_lora(
    dynamic_lora: Option<&DynamicLoraConfig>,
    rope_pairing: RopePairing,
    architecture: &str,
) -> Result<()> {
    if dynamic_lora.is_some() && rope_pairing == RopePairing::Adjacent {
        bail!(
            "dynamic LoRA is not supported for native GGUF architecture `{architecture}` because its Q/K tensors use converter-permuted adjacent RoPE order; load the original safetensors model or omit the LoRA adapter"
        );
    }
    Ok(())
}

fn validate_legacy_gguf_adapter_qk_layout(architecture: GGUFArchitecture) -> Result<()> {
    if matches!(
        architecture,
        GGUFArchitecture::Llama | GGUFArchitecture::Mistral3
    ) {
        bail!(
            "legacy LoRA and X-LoRA are not supported for GGUF `{architecture}` models because their Q/K tensors use converter-permuted adjacent RoPE order; load the original safetensors model or omit the adapter"
        );
    }
    Ok(())
}

fn preferred_hf_config(files: &[String]) -> Option<&'static str> {
    if files.iter().any(|file| file == "config.json") {
        Some("config.json")
    } else if files.iter().any(|file| file == "params.json") {
        Some("params.json")
    } else {
        None
    }
}

fn requires_multimodal_projector(architecture: &str) -> bool {
    PROJECTOR_REQUIRED_ARCHITECTURES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(architecture))
}

enum Model {
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
    generation_defaults: Option<crate::ModelGenerationDefaults>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: crate::pipeline::IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

impl GGUFSpecificConfig {
    fn multimodal_config(&self) -> MultimodalSpecificConfig {
        MultimodalSpecificConfig {
            topology: self.topology.clone(),
            organization: self.organization,
            write_uqff: self.write_uqff.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            max_edge: self.max_edge,
            max_model_len: self.max_model_len,
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
            ..Default::default()
        }
    }
}

struct NativeMultimodalLoadArgs<'a> {
    paths: &'a dyn ModelPaths,
    mmproj_paths: &'a [PathBuf],
    dtype: &'a dyn TryIntoDType,
    device: &'a Device,
    silent: bool,
    mapper: DeviceMapSetting,
    in_situ_quant: Option<IsqType>,
    paged_attn_config: Option<PagedAttentionConfig>,
}

fn prepare_native_multimodal_config(
    loader_type: &MultimodalLoaderType,
    config: &str,
) -> Result<String> {
    let config = super::isq::sanitize_quantized_weight_source_config(config)?;
    let config = normalize_qwen_multimodal_config(loader_type, &config)?;
    normalize_muse_glimmer_config(loader_type, &config)
}

struct NativeNormalLoadArgs<'a> {
    paths: &'a dyn ModelPaths,
    dtype: &'a dyn TryIntoDType,
    device: &'a Device,
    silent: bool,
    mapper: DeviceMapSetting,
    in_situ_quant: Option<IsqType>,
    paged_attn_config: Option<PagedAttentionConfig>,
}

#[derive(Clone)]
struct DynamicLoraConfig {
    adapters: Vec<LoraAdapterSpec>,
    runtime: LoraRuntimeConfig,
}

struct ResolvedGgufTokenizer {
    conversion: GgufTokenizerConversion,
    generation_config_compatible: bool,
}

#[derive(Clone, Copy)]
enum TokenizerFallback {
    Strict,
    Automatic,
    ModelAssets,
}

fn resolve_tokenizer_candidate(
    path: &Path,
    fallback: TokenizerFallback,
    external: Result<GgufTokenizerConversion>,
    embedded: impl FnOnce() -> Result<GgufTokenizerConversion>,
) -> Result<ResolvedGgufTokenizer> {
    match external {
        Ok(conversion) => Ok(ResolvedGgufTokenizer {
            conversion,
            generation_config_compatible: true,
        }),
        Err(error) if !matches!(fallback, TokenizerFallback::Strict) => {
            match fallback {
                TokenizerFallback::Automatic => warn!(
                    "Ignoring automatically discovered tokenizer `{}` because it does not match \
                     the GGUF vocabulary: {error}",
                    path.display()
                ),
                TokenizerFallback::ModelAssets => warn!(
                    "Tokenizer from model assets `{}` does not match the GGUF vocabulary; using \
                     the tokenizer embedded in the GGUF instead: {error}",
                    path.display()
                ),
                TokenizerFallback::Strict => unreachable!(),
            }
            Ok(ResolvedGgufTokenizer {
                conversion: embedded().with_context(|| {
                    format!(
                        "Tokenizer `{}` was rejected ({error}); GGUF tokenizer conversion also \
                         failed",
                        path.display()
                    )
                })?,
                generation_config_compatible: false,
            })
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "Tokenizer `{}` is incompatible with the GGUF model vocabulary",
                path.display()
            )
        }),
    }
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` optionally overrides embedded
    /// configuration and tokenizer assets. If `chat_template` is specified, it is treated as a path
    /// and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    pub fn with_mmproj_files(mut self, mmproj_filenames: Vec<String>) -> Self {
        self.mmproj_filenames = Some(mmproj_filenames);
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: String) -> Self {
        self.tokenizer_json = Some(tokenizer_json);
        self
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.dynamic_lora = None;
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_dynamic_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime: LoraRuntimeConfig,
    ) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();
        self.dynamic_lora = Some(DynamicLoraConfig { adapters, runtime });
        self.xlora_model_id = None;
        self.xlora_order = None;
        self.tgt_non_granular_index = None;
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            dynamic_lora: self.dynamic_lora,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            mmproj_filenames: self.mmproj_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            mmproj_filenames: None,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            tokenizer_json: None,
            dynamic_lora: None,
            kind,
            tgt_non_granular_index,
            config,
            jinja_explicit,
        }
    }

    fn dynamic_lora_adapters(&self) -> Option<&[LoraAdapterSpec]> {
        self.dynamic_lora
            .as_ref()
            .map(|config| config.adapters.as_slice())
    }

    fn resolve_tokenizer(
        &self,
        paths: &dyn ModelPaths,
        metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
    ) -> Result<ResolvedGgufTokenizer> {
        let (path, fallback) = if let Some(tokenizer_json) = self.tokenizer_json.as_ref() {
            (PathBuf::from(tokenizer_json), TokenizerFallback::Strict)
        } else if paths.get_tokenizer_filename().as_os_str().is_empty() {
            return Ok(ResolvedGgufTokenizer {
                conversion: convert_gguf_metadata_to_hf_tokenizer(metadata)?,
                generation_config_compatible: true,
            });
        } else {
            (
                paths.get_tokenizer_filename().clone(),
                if self.model_id.is_none() {
                    TokenizerFallback::Automatic
                } else {
                    TokenizerFallback::ModelAssets
                },
            )
        };

        let external = match get_tokenizer(&path, None) {
            Ok(tokenizer) => validate_external_gguf_tokenizer(tokenizer, metadata),
            Err(error) if matches!(fallback, TokenizerFallback::ModelAssets) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to load tokenizer from model assets `{}`",
                        path.display()
                    )
                })
            }
            Err(error) => Err(error),
        };
        resolve_tokenizer_candidate(&path, fallback, external, || {
            convert_gguf_metadata_to_hf_tokenizer(metadata)
        })
    }

    fn resolve_generation_config(
        &self,
        paths: &dyn ModelPaths,
        tokenizer: &ResolvedGgufTokenizer,
    ) -> Option<GenerationConfig> {
        let filename = paths.get_gen_conf_filename()?;
        if !tokenizer.generation_config_compatible {
            warn!(
                "Ignoring generation config `{}` because the external tokenizer was incompatible \
                 with the GGUF model",
                filename.display()
            );
            return None;
        }
        let config = match fs::read_to_string(filename)
            .with_context(|| format!("Failed to read `{}`", filename.display()))
            .and_then(|raw| {
                serde_json::from_str::<GenerationConfig>(&raw)
                    .context("Failed to parse generation_config.json")
            }) {
            Ok(config) => config,
            Err(error) => {
                warn!("Ignoring generation config: {error:#}");
                return None;
            }
        };
        if let Err(error) =
            config.validate_token_ids(tokenizer.conversion.tokenizer.get_vocab_size(true))
        {
            warn!("Ignoring generation config: {error}");
            return None;
        }
        Some(config)
    }

    fn load_native_normal(
        &self,
        args: NativeNormalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeNormalLoadArgs {
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        } = args;
        let archive = Arc::new(mistralrs_quant::GgufArchive::open(
            paths.get_weight_filenames(),
        )?);
        let architecture = match archive.metadata_value("general.architecture") {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => value.as_str(),
            Some(value) => {
                bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => bail!("GGUF metadata is missing `general.architecture`"),
        };
        if architecture.eq_ignore_ascii_case("gemma3") {
            return self.load_native_gemma3_text(
                archive,
                NativeNormalLoadArgs {
                    paths,
                    dtype,
                    device,
                    silent,
                    mapper,
                    in_situ_quant,
                    paged_attn_config,
                },
            );
        }
        let metadata_keys = archive
            .metadata()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let tensor_names = archive
            .tensors()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let metadata_string = |key| match archive.metadata_value(key) {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => Some(value.as_str()),
            _ => None,
        };
        if requires_multimodal_projector(architecture) {
            bail!(
                "GGUF architecture `{architecture}` is supported only as a multimodal model. Pass \
                 its companion projector with `--mmproj <file>`, or load from a GGUF repository \
                 that publishes one; text-only `{architecture}` checkpoints are not supported"
            );
        }
        let descriptor = GgufDescriptor::new(architecture, &metadata_keys, &tensor_names)?
            .with_model_identity(
                metadata_string("general.name"),
                metadata_string("general.basename"),
            );

        let external_config = if paths.get_config_filename().as_os_str().is_empty() {
            None
        } else {
            Some(fs::read_to_string(paths.get_config_filename())?)
        };
        let explicit_loader = external_config
            .as_deref()
            .map(normal_loader_hint_from_external_config)
            .transpose()?;
        let explicit_loader = explicit_loader.flatten();
        let resolved = resolve_native_adapter(&descriptor, explicit_loader)?;
        let rope_pairing =
            crate::gguf::normal_registry::schema_for(descriptor.architecture).rope_pairing;
        validate_native_dynamic_lora(
            self.dynamic_lora.as_ref(),
            rope_pairing,
            descriptor.architecture.as_str(),
        )?;
        debug!(
            "Loading GGUF architecture `{}` through native {:?} ({:?}, layouts {:?})",
            descriptor.architecture,
            resolved.adapter.loader,
            resolved.reason,
            resolved.adapter.layouts
        );
        let loader_type = resolved.adapter.loader.clone();
        let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
        let config = match external_config {
            Some(config) => {
                let config = normalize_external_normal_config(
                    &loader_type,
                    descriptor.architecture,
                    &config,
                )?;
                validate_normal_config_tensor_inventory(&config, &tensor_names)?;
                config
            }
            None => synthesize_normal_config(&loader_type, archive.metadata(), &tensor_names)?,
        };
        let config = stamp_qk_rope_layout(&config, rope_pairing)?;
        let bindings = build_normal_bindings(&archive, &loader_type, descriptor.architecture)?;
        let internal_dtype = dtype.try_into_dtype(&[device])?;
        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
            archive.clone(),
            &bindings,
            internal_dtype,
        )?);
        let weights = source.sharded_var_builder(Device::Cpu);

        let tokenizer = self.resolve_tokenizer(paths, archive.metadata())?;
        let generation_config = self.resolve_generation_config(paths, &tokenizer);
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template_from_metadata(archive.metadata())?
            } else {
                None
            };
        let source = PreparedNormalSource {
            config,
            weights,
            tokenizer: tokenizer.conversion.tokenizer,
            generation_config,
            chat_template: gguf_chat_template,
            bos_token: tokenizer.conversion.bos,
            eos_token: tokenizer.conversion.eos,
            unk_token: tokenizer.conversion.unk,
            source_weight_files: paths.get_weight_filenames().to_vec(),
            rope_pairing,
        };
        let mut loader = NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: self.config.topology.clone(),
                organization: self.config.organization,
                write_uqff: self.config.write_uqff.clone(),
                imatrix: self.config.imatrix.clone(),
                calibration_file: self.config.calibration_file.clone(),
                hf_cache_path: self.config.hf_cache_path.clone(),
                matformer_config_path: self.config.matformer_config_path.clone(),
                matformer_slice_name: self.config.matformer_slice_name.clone(),
                ..Default::default()
            },
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.no_kv_cache,
            self.jinja_explicit.clone(),
        );
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader = loader.build_with_source(loader_type, source, self.kind.clone())?;
        loader.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    fn load_native_gemma3_text(
        &self,
        archive: Arc<mistralrs_quant::GgufArchive>,
        args: NativeNormalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeNormalLoadArgs {
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        } = args;
        let external_config = if paths.get_config_filename().as_os_str().is_empty() {
            None
        } else {
            Some(fs::read_to_string(paths.get_config_filename())?)
        };
        let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
        let config = prepare_gemma3_text_config(
            external_config.as_deref(),
            archive.metadata(),
            &tensor_names,
        )?;
        let config = stamp_qk_rope_layout(&config, RopePairing::HalfSplit)?;
        let use_language_model_prefix = gemma3_text_uses_language_model_prefix(&config)?;
        let bindings = build_gemma3_text_bindings(&archive, use_language_model_prefix)?;
        let internal_dtype = dtype.try_into_dtype(&[device])?;
        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
            archive.clone(),
            &bindings,
            internal_dtype,
        )?);
        let weights = source.sharded_var_builder(Device::Cpu);
        let tokenizer = self.resolve_tokenizer(paths, archive.metadata())?;
        let generation_config = self.resolve_generation_config(paths, &tokenizer);
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template_from_metadata(archive.metadata())?
            } else {
                None
            };
        let source = PreparedMultimodalSource {
            config,
            weights,
            tokenizer: tokenizer.conversion.tokenizer,
            generation_config,
            chat_template: gguf_chat_template,
            bos_token: tokenizer.conversion.bos,
            eos_token: tokenizer.conversion.eos,
            unk_token: tokenizer.conversion.unk,
            processor_config: None,
            preprocessor_config: None,
            source_weight_files: paths.get_weight_filenames().to_vec(),
            rope_pairing: RopePairing::HalfSplit,
        };
        let mut loader = MultimodalLoaderBuilder::new(
            self.config.multimodal_config(),
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.jinja_explicit.clone(),
        );
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader =
            loader.build_with_source(MultimodalLoaderType::Gemma3, source, self.kind.clone());
        loader.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    fn load_native_multimodal(
        &self,
        args: NativeMultimodalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeMultimodalLoadArgs {
            paths,
            mmproj_paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        } = args;
        if !matches!(self.kind, ModelKind::GgufQuantized { .. }) && self.dynamic_lora.is_none() {
            bail!("multimodal GGUF does not support legacy GGUF adapters");
        }
        let mut archive = mistralrs_quant::GgufArchive::open(paths.get_weight_filenames())?;
        let architecture = match archive.metadata_value("general.architecture") {
            Some(candle_core::quantized::gguf_file::Value::String(value)) => value.clone(),
            Some(value) => {
                bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => bail!("GGUF metadata is missing `general.architecture`"),
        };
        let mmproj = mistralrs_quant::GgufArchive::open_components(mmproj_paths)?;
        archive.merge_components(mmproj)?;
        let archive = Arc::new(archive);

        let (loader_type, bindings, rope_pairing) = match architecture.as_str() {
            "gemma4" => (
                MultimodalLoaderType::Gemma4,
                build_gemma4_bindings(&archive)?,
                RopePairing::HalfSplit,
            ),
            "qwen2vl" | "qwen3vl" | "qwen3vlmoe" | "qwen35" | "qwen35moe" => (
                qwen_multimodal_loader_type(&archive)?,
                build_qwen_multimodal_bindings(&archive)?,
                RopePairing::HalfSplit,
            ),
            architecture => {
                let Some(resolved) = resolve_native_multimodal_gguf(&archive)? else {
                    bail!(
                        "multimodal GGUF architecture `{architecture}` is not supported by the native multimodal loader"
                    );
                };
                (
                    resolved.loader_type,
                    resolved.bindings,
                    resolved.rope_pairing,
                )
            }
        };
        validate_native_dynamic_lora(self.dynamic_lora.as_ref(), rope_pairing, &architecture)?;
        if paths.get_config_filename().as_os_str().is_empty() {
            bail!(
                "multimodal GGUF architecture `{architecture}` requires its original `config.json`; pass `--tok-model-id <original-model-id>`"
            );
        }
        let config = prepare_native_multimodal_config(
            &loader_type,
            &fs::read_to_string(paths.get_config_filename())?,
        )?;
        let config = stamp_qk_rope_layout(&config, rope_pairing)?;
        if architecture == "gemma3" {
            ensure_gemma3_vision_config(&config)?;
        }
        let internal_dtype = dtype.try_into_dtype(&[device])?;
        let source = Arc::new(mistralrs_quant::GgufWeightSource::new(
            archive.clone(),
            &bindings,
            internal_dtype,
        )?);
        let weights = source.sharded_var_builder(Device::Cpu);
        let tokenizer = self.resolve_tokenizer(paths, archive.metadata())?;
        let generation_config = self.resolve_generation_config(paths, &tokenizer);
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template_from_metadata(archive.metadata())?
            } else {
                None
            };
        let processor_config = paths
            .get_processor_config()
            .as_ref()
            .map(fs::read_to_string)
            .transpose()?;
        let preprocessor_config = paths
            .get_preprocessor_config()
            .as_ref()
            .map(fs::read_to_string)
            .transpose()?;
        let mut source_weight_files = paths.get_weight_filenames().to_vec();
        source_weight_files.extend_from_slice(mmproj_paths);
        let source = PreparedMultimodalSource {
            config,
            weights,
            tokenizer: tokenizer.conversion.tokenizer,
            generation_config,
            chat_template: gguf_chat_template,
            bos_token: tokenizer.conversion.bos,
            eos_token: tokenizer.conversion.eos,
            unk_token: tokenizer.conversion.unk,
            processor_config,
            preprocessor_config,
            source_weight_files,
            rope_pairing,
        };
        let mut loader = MultimodalLoaderBuilder::new(
            self.config.multimodal_config(),
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.jinja_explicit.clone(),
        );
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader = loader.build_with_source(loader_type, source, self.kind.clone());
        loader.load_model_from_path(
            paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    fn infer_multimodal_asset_paths(
        &self,
        paths: &dyn ModelPaths,
        mmproj_paths: &[PathBuf],
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<Option<LocalModelPaths<PathBuf>>> {
        if self.model_id.is_some() {
            return Ok(None);
        }

        if !paths.get_config_filename().as_os_str().is_empty()
            && paths.get_preprocessor_config().is_some()
            && paths.get_processor_config().is_some()
            && !paths.get_tokenizer_filename().as_os_str().is_empty()
        {
            return Ok(None);
        }

        let config_missing = paths.get_config_filename().as_os_str().is_empty();
        let model_archive = mistralrs_quant::GgufArchive::open(paths.get_weight_filenames())?;
        let projector_archives = mistralrs_quant::GgufArchive::open_components(mmproj_paths)?;
        let projector_labels = projector_archives
            .iter()
            .enumerate()
            .map(|(index, archive)| {
                format!(
                    "projector {} (`{}`)",
                    index + 1,
                    archive.shards()[0].path().display()
                )
            })
            .collect::<Vec<_>>();
        let inferred_model_id = infer_hf_base_model_id(
            std::iter::once(("model", model_archive.metadata())).chain(
                projector_labels
                    .iter()
                    .zip(&projector_archives)
                    .map(|(label, archive)| (label.as_str(), archive.metadata())),
            ),
        );
        let inferred_model_id = match inferred_model_id {
            Ok(inferred_model_id) => inferred_model_id,
            Err(error) if config_missing => {
                return Err(error).context(
                    "Cannot infer original model assets from GGUF base-model metadata; pass \
                     `--tok-model-id <original-model-id>` to override it",
                )
            }
            Err(error) => {
                warn!(
                    "Could not infer optional GGUF processor assets from base-model metadata: \
                     {error}"
                );
                return Ok(None);
            }
        };
        let Some(inferred_model_id) = inferred_model_id else {
            if config_missing {
                bail!(
                    "multimodal GGUF requires its original `config.json`, but the GGUF files do \
                     not identify one unambiguous Hugging Face base model; pass \
                     `--tok-model-id <original-model-id>`"
                );
            }
            return Ok(None);
        };

        let revision = "main";
        let api = match build_api(token_source, !silent) {
            Ok(api) => api,
            Err(error) if config_missing => return Err(error),
            Err(error) => {
                warn!("Could not prepare optional GGUF processor asset discovery: {error}");
                return Ok(None);
            }
        };
        let api = api.repo(Repo::with_revision(
            inferred_model_id.clone(),
            RepoType::Model,
            revision.to_string(),
        ));
        let model_id = Path::new(&inferred_model_id);
        let files = match list_repo_files(&api, model_id, true, revision) {
            Ok(files) => files,
            Err(error) if config_missing => return Err(error),
            Err(error) => {
                warn!(
                    "Could not discover optional GGUF processor assets from \
                     `{inferred_model_id}`: {error}"
                );
                return Ok(None);
            }
        };
        let get_optional = |filename: &str| -> Option<PathBuf> {
            if files.iter().any(|file| file == filename) {
                match get_file(&api, model_id, filename, revision) {
                    Ok(path) => Some(path),
                    Err(error) => {
                        warn!(
                            "Could not load optional GGUF processor asset `{filename}` from \
                             `{inferred_model_id}`: {error}"
                        );
                        None
                    }
                }
            } else {
                None
            }
        };

        let config_filename = if config_missing {
            let Some(config_name) = preferred_hf_config(&files) else {
                bail!(
                    "GGUF base model `{inferred_model_id}` has no `config.json` or `params.json`; \
                     pass `--tok-model-id <original-model-id>`"
                );
            };
            get_file(&api, model_id, config_name, revision)?
        } else {
            paths.get_config_filename().clone()
        };
        let preprocessor_config = match paths.get_preprocessor_config() {
            Some(filename) => Some(filename.clone()),
            None => get_optional("preprocessor_config.json"),
        };
        let video_preprocessor_config = paths
            .get_video_preprocessor_config()
            .cloned()
            .or_else(|| get_optional("video_preprocessor_config.json"));
        let processor_config = match paths.get_processor_config() {
            Some(filename) => Some(filename.clone()),
            None => get_optional("processor_config.json"),
        };
        let tokenizer_filename = if paths.get_tokenizer_filename().as_os_str().is_empty() {
            get_optional("tokenizer.json").unwrap_or_default()
        } else {
            paths.get_tokenizer_filename().clone()
        };
        let gen_conf = paths
            .get_gen_conf_filename()
            .cloned()
            .or_else(|| get_optional("generation_config.json"));

        info!("Using GGUF base-model assets from `{inferred_model_id}`.");
        Ok(Some(LocalModelPaths {
            tokenizer_filename,
            config_filename,
            template_filename: paths.get_template_filename().clone(),
            filenames: paths.get_weight_filenames().to_vec(),
            adapter_paths: paths.get_adapter_paths().clone(),
            gen_conf,
            preprocessor_config,
            video_preprocessor_config,
            processor_config,
            chat_template_json_filename: paths.get_chat_template_explicit().clone(),
        }))
    }
}

impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let cache = self
            .config
            .hf_cache_path
            .clone()
            .map(hf_hub::Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        if self.mmproj_filenames.as_ref().is_some_and(|filenames| {
            filenames.is_empty() || filenames.iter().any(|filename| filename.trim().is_empty())
        }) {
            bail!("multimodal GGUF requires at least one nonempty projector filename");
        }
        if self.mmproj_filenames.is_some() && self.kind.is_adapted() && self.dynamic_lora.is_none()
        {
            bail!("multimodal GGUF does not support legacy LoRA or X-LoRA adapters");
        }
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            Some(revision.clone()),
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent,
            true
        );
        let paths = paths?;
        if let Some(mmproj_filenames) = self.mmproj_filenames.as_ref() {
            let mmproj_paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
                LocalModelPaths,
                &token_source,
                Some(revision),
                self,
                self.quantized_model_id.clone(),
                mmproj_filenames.clone(),
                silent,
                false
            );
            let mmproj_paths = mmproj_paths?;
            let inferred_paths = self.infer_multimodal_asset_paths(
                paths.as_ref(),
                mmproj_paths.get_weight_filenames(),
                &token_source,
                silent,
            )?;
            let paths = inferred_paths
                .as_ref()
                .map(|paths| paths as &dyn ModelPaths)
                .unwrap_or(paths.as_ref());
            return self.load_native_multimodal(NativeMultimodalLoadArgs {
                paths,
                mmproj_paths: mmproj_paths.get_weight_filenames(),
                dtype,
                device,
                silent,
                mapper,
                in_situ_quant,
                paged_attn_config,
            });
        }

        self.load_model_from_path(
            paths.as_ref(),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        if matches!(self.kind, ModelKind::GgufQuantized { .. }) || self.dynamic_lora.is_some() {
            return self.load_native_normal(NativeNormalLoadArgs {
                paths,
                dtype,
                device,
                silent,
                mapper,
                in_situ_quant,
                paged_attn_config,
            });
        }
        if in_situ_quant.is_some()
            || self.config.write_uqff.is_some()
            || self.config.imatrix.is_some()
            || self.config.calibration_file.is_some()
        {
            bail!("ISQ conversion is only supported by the native GGUF loading path");
        }
        if paged_attn_config.is_some() {
            warn!("Adapter models do not currently support PagedAttention, running without");
        }

        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers = readers.iter_mut().collect::<Vec<_>>();
        let model = Content::from_readers(&mut readers)?;

        if !silent {
            model.print_metadata()?;
        }

        let arch = model.arch();
        validate_legacy_gguf_adapter_qk_layout(arch)?;

        // If auto, convert to Map
        let num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;

        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let devices = device_map::get_all_similar_devices(device)?;
            // Initial dtype
            let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

            let model = GgufDeviceMapLoaderInner {
                model: &model,
                arch,
            };

            let layer_sizes_in_bytes =
                model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let non_mapped_size_in_bytes =
                model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None, None)?;
            let total_model_size_in_bytes =
                layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

            let new = model.get_device_layers(
                "this is a dummy config!",
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                None,
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }

        let use_nccl = mistralrs_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { worker_rank, .. } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let pipeline_mapper = mapper.into_mapper(
            num_layers,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            num_layers,
            device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;

        let tokenizer = self.resolve_tokenizer(paths, model.get_metadata())?;
        let gen_conf = self.resolve_generation_config(paths, &tokenizer);
        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = tokenizer.conversion;

        // Only load gguf chat template if there is nothing else
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template(&model)?
            } else {
                None
            };

        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let model_config_metadata: ContentConfig = (&model).into();
        let internal_dtype = mapper.get_min_dtype(dtype)?;

        let quant = ModelConfig::ParamsGGUF(model, (device, mapper).into(), internal_dtype);
        let adapter = ModelConfig::Adapter::try_new(paths, device, silent, is_xlora)?;
        let model_config = ModelConfig::ModelParams::new(quant, Some(adapter));

        let model = match self.kind {
            ModelKind::GgufAdapter { adapter, .. } => match arch {
                GGUFArchitecture::Llama | GGUFArchitecture::Mistral3 => {
                    Model::XLoraLlama(XLoraQLlama::try_from(model_config)?)
                }
                GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::try_from(model_config)?),
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            gguf_chat_template,
        );

        let max_seq_len = match model {
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
        };
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model {
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
            Model::XLoraPhi3(ref model) => model.cache.full().lock().len(),
        };

        if chat_template.bos_token.is_none() {
            if let Some(v) = bos {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.eos_token.is_none() {
            if let Some(v) = eos {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.unk_token.is_none() {
            if let Some(v) = unk {
                chat_template.unk_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }

        let generation_defaults = gen_conf
            .as_ref()
            .and_then(GenerationConfig::generation_defaults);
        let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);
        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self.quantized_model_id.clone(),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: Some(Arc::new(model_config_metadata)),
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
                loaded_for_uqff_write: false,
            }),
            generation_defaults,
            mapper: pipeline_mapper,
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        FullCacheManager.clone_in_cache(self, seqs, false)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        FullCacheManager.clone_out_cache(self, seqs, false)
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
        FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::XLoraLlama(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
        }
    }
}

impl MetadataMixin for GGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        true
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            context_lens,
            position_ids: _, // NOTE(EricLBuehler): ignore, it is for phi3
            paged_attn_meta: _,
            flash_meta,
            flash_meta_full,
            recurrent_batch_kind: _,
            adapter_leases: _adapter_leases,
        } = *inputs.downcast().expect("Downcast failed.");
        let logits = match self.model {
            Model::XLoraLlama(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
            Model::XLoraPhi3(ref model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
        };
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for GGUFPipeline {}

#[cfg(test)]
mod tests {
    use super::{
        preferred_hf_config, prepare_native_multimodal_config, requires_multimodal_projector,
        resolve_tokenizer_candidate, validate_legacy_gguf_adapter_qk_layout,
        validate_native_dynamic_lora, DynamicLoraConfig, GGUFSpecificConfig,
        GgufTokenizerConversion, TokenizerFallback,
    };
    use crate::{
        gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY,
        gguf::{normal_registry::RopePairing, GGUFArchitecture},
        MultimodalLoaderType,
    };
    use std::path::{Path, PathBuf};
    use tokenizers::{models::bpe::BPE, Tokenizer};

    fn tokenizer_conversion(marker: &str) -> GgufTokenizerConversion {
        GgufTokenizerConversion {
            tokenizer: Tokenizer::new(BPE::default()),
            bos: None,
            eos: Some(marker.to_string()),
            unk: None,
        }
    }

    #[test]
    fn scopes_main_only_multimodal_architectures() {
        assert!(!requires_multimodal_projector("qwen35"));
        assert!(!requires_multimodal_projector("QWEN35"));
        assert!(!requires_multimodal_projector("gemma3"));
        assert!(requires_multimodal_projector("gemma3n"));
        assert!(requires_multimodal_projector("gemma4"));
        assert!(!requires_multimodal_projector("qwen35moe"));
        assert!(!requires_multimodal_projector("mistral3"));
        assert!(requires_multimodal_projector("muse-glimmer"));
    }

    #[test]
    fn native_dynamic_lora_rejects_adjacent_rope_gguf() {
        let dynamic_lora = DynamicLoraConfig {
            adapters: Vec::new(),
            runtime: Default::default(),
        };
        let error =
            validate_native_dynamic_lora(Some(&dynamic_lora), RopePairing::Adjacent, "llama")
                .unwrap_err();
        let error = error.to_string();
        assert!(error.contains("converter-permuted adjacent RoPE order"));
        assert!(error.contains("original safetensors model"));
        assert!(validate_native_dynamic_lora(
            Some(&dynamic_lora),
            RopePairing::HalfSplit,
            "qwen35",
        )
        .is_ok());
        assert!(validate_native_dynamic_lora(None, RopePairing::Adjacent, "llama").is_ok());
    }

    #[test]
    fn legacy_gguf_adapters_reject_adjacent_qk_layouts() {
        for architecture in [GGUFArchitecture::Llama, GGUFArchitecture::Mistral3] {
            let error = validate_legacy_gguf_adapter_qk_layout(architecture).unwrap_err();
            assert!(error
                .to_string()
                .contains("converter-permuted adjacent RoPE order"));
        }
        assert!(validate_legacy_gguf_adapter_qk_layout(GGUFArchitecture::Phi3).is_ok());
    }

    #[test]
    fn native_multimodal_configs_ignore_checkpoint_quantization_metadata() {
        let raw = r#"{
            "architectures":["ExampleForConditionalGeneration"],
            "quantization_config":{"quant_method":"awq"},
            "text_config":{
                "hidden_size":128,
                "quantization_config":{"quant_method":"fp8"},
                "rope_parameters":{"quantization_config":{"quant_method":"gptq"}}
            },
            "vision_config":{"hidden_size":64}
        }"#;
        for loader_type in [
            MultimodalLoaderType::Gemma3,
            MultimodalLoaderType::Gemma3n,
            MultimodalLoaderType::Gemma4,
            MultimodalLoaderType::Idefics3,
            MultimodalLoaderType::Llama4,
            MultimodalLoaderType::Lfm2Vl,
            MultimodalLoaderType::Mistral3,
            MultimodalLoaderType::MuseGlimmer,
            MultimodalLoaderType::Qwen2VL,
            MultimodalLoaderType::Qwen2_5VL,
            MultimodalLoaderType::Qwen3VL,
            MultimodalLoaderType::Qwen3VLMoE,
            MultimodalLoaderType::Qwen3_5,
            MultimodalLoaderType::Qwen3_5Moe,
        ] {
            let config = prepare_native_multimodal_config(&loader_type, raw).unwrap();
            let config: serde_json::Value = serde_json::from_str(&config).unwrap();
            assert!(config["quantization_config"].is_null(), "{loader_type:?}");
            assert!(
                config["text_config"]["quantization_config"].is_null(),
                "{loader_type:?}"
            );
            assert!(
                config["text_config"]["rope_parameters"]
                    .get("quantization_config")
                    .is_none(),
                "{loader_type:?}"
            );
            assert_eq!(
                config["architectures"][0],
                "ExampleForConditionalGeneration"
            );
            assert_eq!(config["vision_config"]["hidden_size"], 64);
            if matches!(
                loader_type,
                MultimodalLoaderType::Qwen3_5 | MultimodalLoaderType::Qwen3_5Moe
            ) {
                assert_eq!(config["text_config"][GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
            }
            if matches!(loader_type, MultimodalLoaderType::MuseGlimmer) {
                assert_eq!(
                    config["_mistralrs_muse_glimmer_gguf_collapsed_temporal"],
                    true
                );
            }
        }
    }

    #[test]
    fn automatic_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("automatic-tokenizer.json"),
            TokenizerFallback::Automatic,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn model_asset_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn compatible_model_asset_tokenizer_is_preserved() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Ok(tokenizer_conversion("external")),
            || Ok(tokenizer_conversion("unused")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("external"));
        assert!(resolved.generation_config_compatible);
    }

    #[test]
    fn explicit_tokenizer_file_mismatch_is_an_error() {
        let error = resolve_tokenizer_candidate(
            Path::new("explicit-tokenizer.json"),
            TokenizerFallback::Strict,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("unused")),
        )
        .err()
        .expect("explicit mismatch must fail");

        assert!(error.to_string().contains("incompatible"));
    }

    #[test]
    fn prefers_hugging_face_config_over_native_params() {
        let files = vec!["params.json".to_string(), "config.json".to_string()];
        assert_eq!(preferred_hf_config(&files), Some("config.json"));
        assert_eq!(
            preferred_hf_config(&["params.json".to_string()]),
            Some("params.json")
        );
        assert_eq!(preferred_hf_config(&[]), None);
    }

    #[test]
    fn prepared_multimodal_config_preserves_runtime_context_cap() {
        let config = GGUFSpecificConfig {
            organization: crate::pipeline::IsqOrganization::MoeExpertsOnly,
            imatrix: Some(PathBuf::from("model.imatrix")),
            max_edge: Some(1024),
            max_model_len: Some(8192),
            hf_cache_path: Some(PathBuf::from("hf-cache")),
            matformer_config_path: Some(PathBuf::from("matformer.csv")),
            matformer_slice_name: Some("small".to_string()),
            ..Default::default()
        }
        .multimodal_config();

        assert_eq!(config.max_edge, Some(1024));
        assert_eq!(config.max_model_len, Some(8192));
        assert!(matches!(
            config.organization,
            crate::pipeline::IsqOrganization::MoeExpertsOnly
        ));
        assert_eq!(config.imatrix, Some(PathBuf::from("model.imatrix")));
        assert_eq!(config.hf_cache_path, Some(PathBuf::from("hf-cache")));
        assert_eq!(
            config.matformer_config_path,
            Some(PathBuf::from("matformer.csv"))
        );
        assert_eq!(config.matformer_slice_name.as_deref(), Some("small"));
    }
}
