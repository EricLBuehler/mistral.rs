use crate::NormalLoaderType;
use std::{error::Error, fmt, str::FromStr};

pub(crate) const NORMAL_LOADER_TYPE_COUNT: usize = 26;
pub(crate) const CANONICAL_GGUF_ARCHITECTURE_COUNT: usize = 26;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CanonicalGgufArchitecture {
    Llama,
    Mistral3,
    Gemma,
    Gemma2,
    Phi2,
    Phi3,
    PhiMoe,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Qwen3Next,
    Qwen35,
    Qwen35Moe,
    Starcoder2,
    DeepSeek2,
    Glm4,
    Glm4Moe,
    SmolLm3,
    Granite,
    GraniteMoe,
    GraniteHybrid,
    GptOss,
    HunYuanDense,
    HunYuanMoe,
    Lfm2,
    Lfm2Moe,
}

impl CanonicalGgufArchitecture {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral3 => "mistral3",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
            Self::PhiMoe => "phimoe",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Next => "qwen3next",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Starcoder2 => "starcoder2",
            Self::DeepSeek2 => "deepseek2",
            Self::Glm4 => "glm4",
            Self::Glm4Moe => "glm4moe",
            Self::SmolLm3 => "smollm3",
            Self::Granite => "granite",
            Self::GraniteMoe => "granitemoe",
            Self::GraniteHybrid => "granitehybrid",
            Self::GptOss => "gpt-oss",
            Self::HunYuanDense => "hunyuan-dense",
            Self::HunYuanMoe => "hunyuan-moe",
            Self::Lfm2 => "lfm2",
            Self::Lfm2Moe => "lfm2moe",
        }
    }
}

impl fmt::Display for CanonicalGgufArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CanonicalGgufArchitecture {
    type Err = NormalGgufRegistryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "llama" => Ok(Self::Llama),
            "mistral3" => Ok(Self::Mistral3),
            "gemma" => Ok(Self::Gemma),
            "gemma2" => Ok(Self::Gemma2),
            "phi2" => Ok(Self::Phi2),
            "phi3" => Ok(Self::Phi3),
            "phimoe" => Ok(Self::PhiMoe),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" => Ok(Self::Qwen3),
            "qwen3moe" => Ok(Self::Qwen3Moe),
            "qwen3next" => Ok(Self::Qwen3Next),
            "qwen35" => Ok(Self::Qwen35),
            "qwen35moe" => Ok(Self::Qwen35Moe),
            "starcoder2" => Ok(Self::Starcoder2),
            "deepseek2" => Ok(Self::DeepSeek2),
            "glm4" => Ok(Self::Glm4),
            "glm4moe" => Ok(Self::Glm4Moe),
            "smollm3" => Ok(Self::SmolLm3),
            "granite" => Ok(Self::Granite),
            "granitemoe" => Ok(Self::GraniteMoe),
            "granitehybrid" => Ok(Self::GraniteHybrid),
            "gpt-oss" => Ok(Self::GptOss),
            "hunyuan-dense" => Ok(Self::HunYuanDense),
            "hunyuan-moe" => Ok(Self::HunYuanMoe),
            "lfm2" => Ok(Self::Lfm2),
            "lfm2moe" => Ok(Self::Lfm2Moe),
            _ => Err(NormalGgufRegistryError::UnknownArchitecture(
                value.to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RopePairing {
    Adjacent,
    HalfSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufLayout {
    Direct,
    ConverterPermutedQk,
    ShiftedRmsNorm,
    FusedQkvRowSlices,
    FusedGateUp,
    StackedExperts,
    SplitMlaKvB,
    PerLayerInventory,
    GraniteSsm,
    GraniteSplitGateUp,
    GptOssSeparatedMxfp4,
    Qwen3NextSplitQkvzGroupedBa,
    Qwen35SplitQkvzSplitBetaAlpha,
    SqueezedShortConv,
}

#[derive(Debug)]
pub(crate) struct GgufSchema {
    pub(crate) architecture: CanonicalGgufArchitecture,
    pub(crate) compatible_loaders: &'static [NormalLoaderType],
    pub(crate) rope_pairing: RopePairing,
    pub(crate) required_metadata: &'static [&'static str],
    pub(crate) required_tensors: &'static [&'static str],
    pub(crate) unsupported_metadata: &'static [&'static str],
}

#[derive(Debug)]
pub(crate) struct NativeModelAdapter {
    pub(crate) loader: NormalLoaderType,
    pub(crate) architectures: &'static [CanonicalGgufArchitecture],
    pub(crate) layouts: &'static [GgufLayout],
}

#[derive(Debug)]
pub(crate) struct GgufDescriptor<'a> {
    pub(crate) architecture: CanonicalGgufArchitecture,
    pub(crate) metadata_keys: &'a [&'a str],
    pub(crate) tensor_names: &'a [&'a str],
    general_name: Option<&'a str>,
    general_basename: Option<&'a str>,
}

impl<'a> GgufDescriptor<'a> {
    pub(crate) fn new(
        architecture: &str,
        metadata_keys: &'a [&'a str],
        tensor_names: &'a [&'a str],
    ) -> Result<Self, NormalGgufRegistryError> {
        Ok(Self {
            architecture: architecture.parse()?,
            metadata_keys,
            tensor_names,
            general_name: None,
            general_basename: None,
        })
    }

    pub(crate) fn with_model_identity(
        mut self,
        general_name: Option<&'a str>,
        general_basename: Option<&'a str>,
    ) -> Self {
        self.general_name = general_name;
        self.general_basename = general_basename;
        self
    }

    pub(crate) fn has_metadata(&self, pattern: &str) -> bool {
        self.metadata_keys
            .iter()
            .any(|key| metadata_key_matches(self.architecture, pattern, key))
    }

    pub(crate) fn has_tensor(&self, marker: &str) -> bool {
        self.tensor_names.iter().any(|name| name.contains(marker))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolutionReason {
    SingleCandidate,
    ExplicitOverride,
    ExpertInventory,
    ModelIdentity,
    TensorInventory,
    SlidingWindow,
    DenseLlamaDefault,
}

#[derive(Debug)]
pub(crate) struct ResolvedNativeModelAdapter {
    pub(crate) adapter: &'static NativeModelAdapter,
    pub(crate) reason: ResolutionReason,
}

#[derive(Debug)]
pub(crate) enum NormalGgufRegistryError {
    UnknownArchitecture(String),
    MissingMetadata {
        architecture: CanonicalGgufArchitecture,
        pattern: &'static str,
    },
    MissingMetadataAlternative {
        architecture: CanonicalGgufArchitecture,
        alternatives: &'static [&'static str],
    },
    MissingTensor {
        architecture: CanonicalGgufArchitecture,
        marker: &'static str,
    },
    UnsupportedMetadata {
        architecture: CanonicalGgufArchitecture,
        pattern: &'static str,
    },
    SchemaArchitectureMismatch {
        expected: CanonicalGgufArchitecture,
        actual: CanonicalGgufArchitecture,
    },
    ExplicitOverrideIncompatible {
        architecture: CanonicalGgufArchitecture,
        loader: NormalLoaderType,
    },
    AmbiguousArchitecture {
        architecture: CanonicalGgufArchitecture,
        candidates: &'static [NormalLoaderType],
    },
    MistralIdentityRequiresExternalConfig {
        architecture: CanonicalGgufArchitecture,
    },
    NoAdapter(CanonicalGgufArchitecture),
}

impl fmt::Display for NormalGgufRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownArchitecture(value) => {
                write!(f, "Unknown normal-model GGUF architecture `{value}`")
            }
            Self::MissingMetadata {
                architecture,
                pattern,
            } => write!(
                f,
                "GGUF architecture `{architecture}` is missing metadata `{pattern}`"
            ),
            Self::MissingMetadataAlternative {
                architecture,
                alternatives,
            } => write!(
                f,
                "GGUF architecture `{architecture}` is missing one of {alternatives:?}"
            ),
            Self::MissingTensor {
                architecture,
                marker,
            } => write!(
                f,
                "GGUF architecture `{architecture}` is missing a tensor matching `{marker}`"
            ),
            Self::UnsupportedMetadata {
                architecture,
                pattern,
            } => write!(
                f,
                "GGUF architecture `{architecture}` uses unsupported metadata `{pattern}`"
            ),
            Self::SchemaArchitectureMismatch { expected, actual } => write!(
                f,
                "GGUF schema `{expected}` cannot validate architecture `{actual}`"
            ),
            Self::ExplicitOverrideIncompatible {
                architecture,
                loader,
            } => write!(
                f,
                "Normal loader `{loader}` is incompatible with GGUF architecture `{architecture}`"
            ),
            Self::AmbiguousArchitecture {
                architecture,
                candidates,
            } => {
                write!(
                    f,
                    "GGUF architecture `{architecture}` requires an explicit normal loader override: "
                )?;
                for (index, candidate) in candidates.iter().enumerate() {
                    if index != 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{candidate}")?;
                }
                Ok(())
            }
            Self::MistralIdentityRequiresExternalConfig { architecture } => write!(
                f,
                "GGUF architecture `{architecture}` appears to be a Mistral model, but the file \
                 does not encode whether sliding-window attention is required; provide its \
                 original Hugging Face model with `--tok-model-id <model-id>`"
            ),
            Self::NoAdapter(architecture) => {
                write!(
                    f,
                    "GGUF architecture `{architecture}` has no normal-model adapter"
                )
            }
        }
    }
}

impl Error for NormalGgufRegistryError {}

pub(crate) const COMMON_METADATA_REQUIREMENTS: &[&str] = &[
    "general.architecture",
    "{arch}.context_length",
    "{arch}.embedding_length",
    "{arch}.block_count",
    "{arch}.attention.head_count",
];

pub(crate) const COMMON_METADATA_ALTERNATIVES: &[&[&str]] = &[
    &["{arch}.vocab_size", "tokenizer.ggml.tokens"],
    &[
        "{arch}.attention.layer_norm_rms_epsilon",
        "{arch}.attention.layer_norm_epsilon",
    ],
];

pub(crate) const COMMON_TENSOR_REQUIREMENTS: &[&str] = &["token_embd.weight"];

const NO_REQUIREMENTS: &[&str] = &[];
const MOE_METADATA: &[&str] = &["{arch}.expert_count", "{arch}.expert_used_count"];
const MLA_METADATA: &[&str] = &[
    "{arch}.expert_count",
    "{arch}.expert_used_count",
    "{arch}.attention.kv_lora_rank",
    "{arch}.attention.key_length_mla",
    "{arch}.attention.value_length_mla",
];
const GRANITE_HYBRID_METADATA: &[&str] = &[
    "{arch}.expert_count",
    "{arch}.expert_used_count",
    "{arch}.ssm.conv_kernel",
    "{arch}.ssm.inner_size",
    "{arch}.ssm.state_size",
];
const QWEN3_NEXT_METADATA: &[&str] = &[
    "{arch}.expert_count",
    "{arch}.expert_used_count",
    "{arch}.full_attention_interval",
    "{arch}.ssm.conv_kernel",
    "{arch}.ssm.inner_size",
    "{arch}.ssm.state_size",
];
const QWEN35_METADATA: &[&str] = &[
    "{arch}.full_attention_interval",
    "{arch}.rope.dimension_sections",
    "{arch}.ssm.conv_kernel",
    "{arch}.ssm.group_count",
    "{arch}.ssm.inner_size",
    "{arch}.ssm.state_size",
    "{arch}.ssm.time_step_rank",
];
const LFM2_METADATA: &[&str] = &["{arch}.shortconv.l_cache"];
const LFM2_MOE_METADATA: &[&str] = &[
    "{arch}.shortconv.l_cache",
    "{arch}.expert_count",
    "{arch}.expert_used_count",
];

const MOE_TENSORS: &[&str] = &[
    ".ffn_gate_inp.",
    ".ffn_gate_exps.",
    ".ffn_up_exps.",
    ".ffn_down_exps.",
];
const MLA_TENSORS: &[&str] = &[".attn_k_b.", ".attn_v_b."];
const SSM_TENSORS: &[&str] = &[".ssm_a", ".ssm_conv1d."];
const QWEN35_MOE_TENSORS: &[&str] = &[
    ".ffn_gate_inp.",
    ".ffn_gate_exps.",
    ".ffn_up_exps.",
    ".ffn_down_exps.",
    ".ssm_a",
    ".ssm_conv1d.",
    ".ssm_alpha.",
    ".ssm_beta.",
];
const QWEN35_TENSORS: &[&str] = &[".ssm_a", ".ssm_conv1d.", ".ssm_alpha.", ".ssm_beta."];
const SHORTCONV_TENSORS: &[&str] = &[".shortconv.conv.", ".shortconv.in_proj."];

const GLM_MROPE_UNSUPPORTED: &[&str] = &["{arch}.rope.dimension_sections"];

const LLAMA_LOADERS: &[NormalLoaderType] = &[
    NormalLoaderType::Llama,
    NormalLoaderType::Mistral,
    NormalLoaderType::Mixtral,
];
const MISTRAL_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Mistral];
const GEMMA_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Gemma];
const GEMMA2_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Gemma2];
const PHI2_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Phi2];
const PHI3_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Phi3];
const PHIMOE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Phi3_5MoE];
const QWEN2_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Qwen2];
const QWEN3_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Qwen3];
const QWEN3_MOE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Qwen3Moe];
const QWEN3_NEXT_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Qwen3Next];
const QWEN35_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Qwen3_5];
const STARCODER2_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Starcoder2];
const DEEPSEEK2_LOADERS: &[NormalLoaderType] = &[
    NormalLoaderType::DeepSeekV2,
    NormalLoaderType::DeepSeekV3,
    NormalLoaderType::GLM4MoeLite,
];
const GLM4_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::GLM4];
const GLM4_MOE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::GLM4Moe];
const SMOLLM3_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::SmolLm3];
const GRANITE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::GraniteMoeHybrid];
const GPT_OSS_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::GptOss];
const HUNYUAN_DENSE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::HunYuanDenseV1];
const HUNYUAN_MOE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::HunYuanMoEV1];
const LFM2_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Lfm2];
const LFM2_MOE_LOADERS: &[NormalLoaderType] = &[NormalLoaderType::Lfm2Moe];

pub(crate) const GGUF_SCHEMAS: &[GgufSchema; CANONICAL_GGUF_ARCHITECTURE_COUNT] = &[
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Llama,
        compatible_loaders: LLAMA_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Mistral3,
        compatible_loaders: MISTRAL_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Gemma,
        compatible_loaders: GEMMA_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Gemma2,
        compatible_loaders: GEMMA2_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Phi2,
        compatible_loaders: PHI2_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Phi3,
        compatible_loaders: PHI3_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::PhiMoe,
        compatible_loaders: PHIMOE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen2,
        compatible_loaders: QWEN2_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen3,
        compatible_loaders: QWEN3_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen3Moe,
        compatible_loaders: QWEN3_MOE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen3Next,
        compatible_loaders: QWEN3_NEXT_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: QWEN3_NEXT_METADATA,
        required_tensors: SSM_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen35,
        compatible_loaders: QWEN35_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: QWEN35_METADATA,
        required_tensors: QWEN35_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Qwen35Moe,
        compatible_loaders: QWEN3_NEXT_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: QWEN3_NEXT_METADATA,
        required_tensors: QWEN35_MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Starcoder2,
        compatible_loaders: STARCODER2_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::DeepSeek2,
        compatible_loaders: DEEPSEEK2_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: MLA_METADATA,
        required_tensors: MLA_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Glm4,
        compatible_loaders: GLM4_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: GLM_MROPE_UNSUPPORTED,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Glm4Moe,
        compatible_loaders: GLM4_MOE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: GLM_MROPE_UNSUPPORTED,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::SmolLm3,
        compatible_loaders: SMOLLM3_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Granite,
        compatible_loaders: GRANITE_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::GraniteMoe,
        compatible_loaders: GRANITE_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::GraniteHybrid,
        compatible_loaders: GRANITE_LOADERS,
        rope_pairing: RopePairing::Adjacent,
        required_metadata: GRANITE_HYBRID_METADATA,
        required_tensors: SSM_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::GptOss,
        compatible_loaders: GPT_OSS_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::HunYuanDense,
        compatible_loaders: HUNYUAN_DENSE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: NO_REQUIREMENTS,
        required_tensors: NO_REQUIREMENTS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::HunYuanMoe,
        compatible_loaders: HUNYUAN_MOE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: MOE_METADATA,
        required_tensors: MOE_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Lfm2,
        compatible_loaders: LFM2_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: LFM2_METADATA,
        required_tensors: SHORTCONV_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
    GgufSchema {
        architecture: CanonicalGgufArchitecture::Lfm2Moe,
        compatible_loaders: LFM2_MOE_LOADERS,
        rope_pairing: RopePairing::HalfSplit,
        required_metadata: LFM2_MOE_METADATA,
        required_tensors: SHORTCONV_TENSORS,
        unsupported_metadata: NO_REQUIREMENTS,
    },
];

pub(crate) const NORMAL_MODEL_ADAPTERS: &[NativeModelAdapter; NORMAL_LOADER_TYPE_COUNT] = &[
    NativeModelAdapter {
        loader: NormalLoaderType::Mistral,
        architectures: &[
            CanonicalGgufArchitecture::Llama,
            CanonicalGgufArchitecture::Mistral3,
        ],
        layouts: &[GgufLayout::Direct, GgufLayout::ConverterPermutedQk],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Gemma,
        architectures: &[CanonicalGgufArchitecture::Gemma],
        layouts: &[GgufLayout::Direct, GgufLayout::ShiftedRmsNorm],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Mixtral,
        architectures: &[CanonicalGgufArchitecture::Llama],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::ConverterPermutedQk,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Llama,
        architectures: &[CanonicalGgufArchitecture::Llama],
        layouts: &[GgufLayout::Direct, GgufLayout::ConverterPermutedQk],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Phi2,
        architectures: &[CanonicalGgufArchitecture::Phi2],
        layouts: &[GgufLayout::Direct, GgufLayout::FusedQkvRowSlices],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Phi3,
        architectures: &[CanonicalGgufArchitecture::Phi3],
        layouts: &[GgufLayout::Direct, GgufLayout::FusedGateUp],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Qwen2,
        architectures: &[CanonicalGgufArchitecture::Qwen2],
        layouts: &[GgufLayout::Direct, GgufLayout::PerLayerInventory],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Gemma2,
        architectures: &[CanonicalGgufArchitecture::Gemma2],
        layouts: &[GgufLayout::Direct, GgufLayout::ShiftedRmsNorm],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Starcoder2,
        architectures: &[CanonicalGgufArchitecture::Starcoder2],
        layouts: &[GgufLayout::Direct],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Phi3_5MoE,
        architectures: &[CanonicalGgufArchitecture::PhiMoe],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::FusedGateUp,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::DeepSeekV2,
        architectures: &[CanonicalGgufArchitecture::DeepSeek2],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::SplitMlaKvB,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::DeepSeekV3,
        architectures: &[CanonicalGgufArchitecture::DeepSeek2],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::SplitMlaKvB,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Qwen3,
        architectures: &[CanonicalGgufArchitecture::Qwen3],
        layouts: &[GgufLayout::Direct],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::GLM4,
        architectures: &[CanonicalGgufArchitecture::Glm4],
        layouts: &[GgufLayout::Direct, GgufLayout::FusedGateUp],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::GLM4MoeLite,
        architectures: &[CanonicalGgufArchitecture::DeepSeek2],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::SplitMlaKvB,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::GLM4Moe,
        architectures: &[CanonicalGgufArchitecture::Glm4Moe],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::FusedGateUp,
            GgufLayout::StackedExperts,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Qwen3Moe,
        architectures: &[CanonicalGgufArchitecture::Qwen3Moe],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::StackedExperts,
            GgufLayout::PerLayerInventory,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::SmolLm3,
        architectures: &[CanonicalGgufArchitecture::SmolLm3],
        layouts: &[GgufLayout::Direct, GgufLayout::ConverterPermutedQk],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::GraniteMoeHybrid,
        architectures: &[
            CanonicalGgufArchitecture::Granite,
            CanonicalGgufArchitecture::GraniteMoe,
            CanonicalGgufArchitecture::GraniteHybrid,
        ],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::ConverterPermutedQk,
            GgufLayout::StackedExperts,
            GgufLayout::PerLayerInventory,
            GgufLayout::GraniteSsm,
            GgufLayout::GraniteSplitGateUp,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::GptOss,
        architectures: &[CanonicalGgufArchitecture::GptOss],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::StackedExperts,
            GgufLayout::GptOssSeparatedMxfp4,
            GgufLayout::PerLayerInventory,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::HunYuanDenseV1,
        architectures: &[CanonicalGgufArchitecture::HunYuanDense],
        layouts: &[GgufLayout::Direct],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::HunYuanMoEV1,
        architectures: &[CanonicalGgufArchitecture::HunYuanMoe],
        layouts: &[GgufLayout::Direct, GgufLayout::StackedExperts],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Qwen3Next,
        architectures: &[
            CanonicalGgufArchitecture::Qwen3Next,
            CanonicalGgufArchitecture::Qwen35Moe,
        ],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::StackedExperts,
            GgufLayout::PerLayerInventory,
            GgufLayout::Qwen3NextSplitQkvzGroupedBa,
            GgufLayout::Qwen35SplitQkvzSplitBetaAlpha,
            GgufLayout::ShiftedRmsNorm,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Qwen3_5,
        architectures: &[CanonicalGgufArchitecture::Qwen35],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::PerLayerInventory,
            GgufLayout::Qwen35SplitQkvzSplitBetaAlpha,
            GgufLayout::ShiftedRmsNorm,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Lfm2,
        architectures: &[CanonicalGgufArchitecture::Lfm2],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::PerLayerInventory,
            GgufLayout::SqueezedShortConv,
        ],
    },
    NativeModelAdapter {
        loader: NormalLoaderType::Lfm2Moe,
        architectures: &[CanonicalGgufArchitecture::Lfm2Moe],
        layouts: &[
            GgufLayout::Direct,
            GgufLayout::StackedExperts,
            GgufLayout::PerLayerInventory,
            GgufLayout::SqueezedShortConv,
        ],
    },
];

impl GgufSchema {
    pub(crate) fn validate(
        &self,
        descriptor: &GgufDescriptor<'_>,
    ) -> Result<(), NormalGgufRegistryError> {
        if self.architecture != descriptor.architecture {
            return Err(NormalGgufRegistryError::SchemaArchitectureMismatch {
                expected: self.architecture,
                actual: descriptor.architecture,
            });
        }

        for pattern in COMMON_METADATA_REQUIREMENTS
            .iter()
            .chain(self.required_metadata)
        {
            if !descriptor.has_metadata(pattern) {
                return Err(NormalGgufRegistryError::MissingMetadata {
                    architecture: self.architecture,
                    pattern,
                });
            }
        }

        for alternatives in COMMON_METADATA_ALTERNATIVES {
            if !alternatives
                .iter()
                .any(|pattern| descriptor.has_metadata(pattern))
            {
                return Err(NormalGgufRegistryError::MissingMetadataAlternative {
                    architecture: self.architecture,
                    alternatives,
                });
            }
        }

        for marker in COMMON_TENSOR_REQUIREMENTS
            .iter()
            .chain(self.required_tensors)
        {
            if !descriptor.has_tensor(marker) {
                return Err(NormalGgufRegistryError::MissingTensor {
                    architecture: self.architecture,
                    marker,
                });
            }
        }

        for pattern in self.unsupported_metadata {
            if descriptor.has_metadata(pattern) {
                return Err(NormalGgufRegistryError::UnsupportedMetadata {
                    architecture: self.architecture,
                    pattern,
                });
            }
        }

        Ok(())
    }
}

pub(crate) fn schema_for(architecture: CanonicalGgufArchitecture) -> &'static GgufSchema {
    GGUF_SCHEMAS
        .iter()
        .find(|schema| schema.architecture == architecture)
        .expect("canonical GGUF architecture is missing its schema")
}

pub(crate) fn adapter_for(loader: &NormalLoaderType) -> Option<&'static NativeModelAdapter> {
    NORMAL_MODEL_ADAPTERS
        .iter()
        .find(|adapter| &adapter.loader == loader)
}

pub(crate) fn resolve_native_adapter(
    descriptor: &GgufDescriptor<'_>,
    explicit_override: Option<NormalLoaderType>,
) -> Result<ResolvedNativeModelAdapter, NormalGgufRegistryError> {
    let schema = schema_for(descriptor.architecture);
    schema.validate(descriptor)?;

    if let Some(loader) = explicit_override {
        let Some(adapter) = adapter_for(&loader) else {
            return Err(NormalGgufRegistryError::NoAdapter(descriptor.architecture));
        };
        if !adapter.architectures.contains(&descriptor.architecture) {
            return Err(NormalGgufRegistryError::ExplicitOverrideIncompatible {
                architecture: descriptor.architecture,
                loader,
            });
        }
        return Ok(ResolvedNativeModelAdapter {
            adapter,
            reason: ResolutionReason::ExplicitOverride,
        });
    }

    if descriptor.architecture == CanonicalGgufArchitecture::Llama {
        let (loader, reason) = if descriptor.has_metadata("{arch}.expert_count")
            || descriptor.has_tensor(".ffn_gate_exps.")
        {
            (NormalLoaderType::Mixtral, ResolutionReason::ExpertInventory)
        } else if descriptor.has_metadata("{arch}.attention.sliding_window") {
            (NormalLoaderType::Mistral, ResolutionReason::SlidingWindow)
        } else if llama_mistral_identity_hint(descriptor) {
            return Err(
                NormalGgufRegistryError::MistralIdentityRequiresExternalConfig {
                    architecture: descriptor.architecture,
                },
            );
        } else {
            (NormalLoaderType::Llama, ResolutionReason::DenseLlamaDefault)
        };
        return Ok(ResolvedNativeModelAdapter {
            adapter: adapter_for(&loader).expect("llama resolver references a missing adapter"),
            reason,
        });
    }

    if descriptor.architecture == CanonicalGgufArchitecture::DeepSeek2 {
        let (loader, reason) = if let Some(loader) = deepseek2_identity_hint(descriptor) {
            (loader, ResolutionReason::ModelIdentity)
        } else if !descriptor.has_tensor(".exp_probs_b") {
            (
                NormalLoaderType::DeepSeekV2,
                ResolutionReason::TensorInventory,
            )
        } else {
            return Err(NormalGgufRegistryError::AmbiguousArchitecture {
                architecture: descriptor.architecture,
                candidates: &[NormalLoaderType::DeepSeekV3, NormalLoaderType::GLM4MoeLite],
            });
        };
        return Ok(ResolvedNativeModelAdapter {
            adapter: adapter_for(&loader).expect("deepseek2 resolver references a missing adapter"),
            reason,
        });
    }

    if schema.compatible_loaders.len() != 1 {
        return Err(NormalGgufRegistryError::AmbiguousArchitecture {
            architecture: descriptor.architecture,
            candidates: schema.compatible_loaders,
        });
    }

    let adapter = adapter_for(&schema.compatible_loaders[0])
        .ok_or(NormalGgufRegistryError::NoAdapter(descriptor.architecture))?;
    Ok(ResolvedNativeModelAdapter {
        adapter,
        reason: ResolutionReason::SingleCandidate,
    })
}

fn llama_mistral_identity_hint(descriptor: &GgufDescriptor<'_>) -> bool {
    descriptor
        .general_basename
        .or(descriptor.general_name)
        .is_some_and(|identity| {
            identity
                .split(|character: char| !character.is_ascii_alphanumeric())
                .find(|component| !component.is_empty())
                .is_some_and(|component| component.eq_ignore_ascii_case("mistral"))
        })
}

fn deepseek2_identity_hint(descriptor: &GgufDescriptor<'_>) -> Option<NormalLoaderType> {
    descriptor
        .general_name
        .into_iter()
        .chain(descriptor.general_basename)
        .find_map(|identity| {
            let compact = identity
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .map(|character| character.to_ascii_lowercase())
                .collect::<String>();
            if compact.contains("glm4") {
                Some(NormalLoaderType::GLM4MoeLite)
            } else if compact.contains("deepseekv3") || compact.contains("deepseekr1") {
                Some(NormalLoaderType::DeepSeekV3)
            } else if compact.contains("deepseekv2")
                || compact.contains("deepseekcoderv2")
                || compact.contains("deepseekllmv2")
            {
                Some(NormalLoaderType::DeepSeekV2)
            } else {
                None
            }
        })
}

fn metadata_key_matches(architecture: CanonicalGgufArchitecture, pattern: &str, key: &str) -> bool {
    let Some(suffix) = pattern.strip_prefix("{arch}") else {
        return key == pattern;
    };
    key.strip_prefix(architecture.as_str()) == Some(suffix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use strum::IntoEnumIterator;

    fn expand_metadata(architecture: CanonicalGgufArchitecture, pattern: &str) -> String {
        pattern.replace("{arch}", architecture.as_str())
    }

    fn valid_fixture(
        schema: &GgufSchema,
        extra_metadata: &[&str],
        extra_tensors: &[&str],
    ) -> (Vec<String>, Vec<String>) {
        let mut metadata = COMMON_METADATA_REQUIREMENTS
            .iter()
            .chain(schema.required_metadata)
            .map(|pattern| expand_metadata(schema.architecture, pattern))
            .collect::<Vec<_>>();
        metadata.extend(
            COMMON_METADATA_ALTERNATIVES
                .iter()
                .map(|alternatives| expand_metadata(schema.architecture, alternatives[0])),
        );
        metadata.extend(
            extra_metadata
                .iter()
                .map(|pattern| expand_metadata(schema.architecture, pattern)),
        );

        let mut tensors = COMMON_TENSOR_REQUIREMENTS
            .iter()
            .chain(schema.required_tensors)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        tensors.extend(extra_tensors.iter().map(ToString::to_string));
        (metadata, tensors)
    }

    fn descriptor_from_fixture<'a>(
        schema: &GgufSchema,
        metadata: &'a [String],
        tensors: &'a [String],
    ) -> GgufDescriptor<'a> {
        let metadata = metadata.iter().map(String::as_str).collect::<Vec<_>>();
        let tensors = tensors.iter().map(String::as_str).collect::<Vec<_>>();
        GgufDescriptor::new(
            schema.architecture.as_str(),
            Box::leak(metadata.into_boxed_slice()),
            Box::leak(tensors.into_boxed_slice()),
        )
        .unwrap()
    }

    #[test]
    fn normal_loader_registry_is_exhaustive() {
        assert_eq!(NormalLoaderType::iter().count(), NORMAL_LOADER_TYPE_COUNT);
        assert_eq!(NORMAL_MODEL_ADAPTERS.len(), NORMAL_LOADER_TYPE_COUNT);

        let mut loaders = HashSet::new();
        for adapter in NORMAL_MODEL_ADAPTERS {
            assert!(loaders.insert(adapter.loader.to_string()));
        }
        for loader in NormalLoaderType::iter() {
            assert!(loaders.contains(&loader.to_string()), "{loader}");
        }
    }

    #[test]
    fn canonical_schema_registry_is_exhaustive_and_round_trips() {
        assert_eq!(GGUF_SCHEMAS.len(), CANONICAL_GGUF_ARCHITECTURE_COUNT);
        let mut architectures = HashSet::new();
        for schema in GGUF_SCHEMAS {
            assert!(architectures.insert(schema.architecture));
            assert_eq!(
                schema
                    .architecture
                    .as_str()
                    .parse::<CanonicalGgufArchitecture>()
                    .unwrap(),
                schema.architecture
            );
            assert_eq!(
                schema
                    .architecture
                    .as_str()
                    .to_ascii_uppercase()
                    .parse::<CanonicalGgufArchitecture>()
                    .unwrap(),
                schema.architecture
            );
        }
        assert!("qwen3_moe".parse::<CanonicalGgufArchitecture>().is_err());
    }

    #[test]
    fn schema_and_adapter_coverage_is_reciprocal() {
        for schema in GGUF_SCHEMAS {
            assert!(!schema.compatible_loaders.is_empty());
            for loader in schema.compatible_loaders {
                let adapter = adapter_for(loader).unwrap();
                assert!(adapter.architectures.contains(&schema.architecture));
            }
        }
        for adapter in NORMAL_MODEL_ADAPTERS {
            assert!(!adapter.architectures.is_empty());
            assert!(!adapter.layouts.is_empty());
            for architecture in adapter.architectures {
                assert!(schema_for(*architecture)
                    .compatible_loaders
                    .contains(&adapter.loader));
            }
        }
    }

    #[test]
    fn every_schema_accepts_its_minimal_valid_fixture() {
        for schema in GGUF_SCHEMAS {
            let (metadata, tensors) = valid_fixture(schema, &[], &[]);
            let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
            schema.validate(&descriptor).unwrap();
        }
    }

    #[test]
    fn rope_pairing_matches_converter_layouts() {
        let adjacent = [
            CanonicalGgufArchitecture::Llama,
            CanonicalGgufArchitecture::Mistral3,
            CanonicalGgufArchitecture::DeepSeek2,
            CanonicalGgufArchitecture::Glm4,
            CanonicalGgufArchitecture::SmolLm3,
            CanonicalGgufArchitecture::Granite,
            CanonicalGgufArchitecture::GraniteMoe,
            CanonicalGgufArchitecture::GraniteHybrid,
        ];
        for architecture in adjacent {
            assert_eq!(schema_for(architecture).rope_pairing, RopePairing::Adjacent);
        }
        for schema in GGUF_SCHEMAS {
            if !adjacent.contains(&schema.architecture) {
                assert_eq!(schema.rope_pairing, RopePairing::HalfSplit);
            }
        }
    }

    #[test]
    fn exceptional_layouts_are_declared() {
        let cases = [
            (NormalLoaderType::Gemma, GgufLayout::ShiftedRmsNorm),
            (NormalLoaderType::Gemma2, GgufLayout::ShiftedRmsNorm),
            (NormalLoaderType::DeepSeekV2, GgufLayout::SplitMlaKvB),
            (NormalLoaderType::DeepSeekV3, GgufLayout::SplitMlaKvB),
            (NormalLoaderType::GLM4MoeLite, GgufLayout::SplitMlaKvB),
            (
                NormalLoaderType::GraniteMoeHybrid,
                GgufLayout::GraniteSplitGateUp,
            ),
            (NormalLoaderType::GptOss, GgufLayout::GptOssSeparatedMxfp4),
            (
                NormalLoaderType::Qwen3Next,
                GgufLayout::Qwen3NextSplitQkvzGroupedBa,
            ),
            (
                NormalLoaderType::Qwen3Next,
                GgufLayout::Qwen35SplitQkvzSplitBetaAlpha,
            ),
            (
                NormalLoaderType::Qwen3_5,
                GgufLayout::Qwen35SplitQkvzSplitBetaAlpha,
            ),
        ];
        for (loader, layout) in cases {
            assert!(adapter_for(&loader).unwrap().layouts.contains(&layout));
        }
    }

    #[test]
    fn llama_inventory_resolution_is_deterministic() {
        let schema = schema_for(CanonicalGgufArchitecture::Llama);

        let (dense_metadata, dense_tensors) = valid_fixture(schema, &[], &[]);
        let descriptor = descriptor_from_fixture(schema, &dense_metadata, &dense_tensors);
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Llama);
        assert_eq!(resolved.reason, ResolutionReason::DenseLlamaDefault);

        let (metadata, tensors) = valid_fixture(schema, &["{arch}.attention.sliding_window"], &[]);
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Mistral);
        assert_eq!(resolved.reason, ResolutionReason::SlidingWindow);

        let (metadata, tensors) = valid_fixture(
            schema,
            &["{arch}.expert_count"],
            &["blk.0.ffn_gate_exps.weight"],
        );
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Mixtral);
        assert_eq!(resolved.reason, ResolutionReason::ExpertInventory);

        for (name, basename) in [
            (Some("Mistral 7B Instruct v0.3"), None),
            (Some("Mistral Small 24B Instruct 2501"), None),
            (None, Some("Mistral-Nemo")),
        ] {
            let descriptor = descriptor_from_fixture(schema, &dense_metadata, &dense_tensors)
                .with_model_identity(name, basename);
            let error = resolve_native_adapter(&descriptor, None).unwrap_err();
            assert!(matches!(
                error,
                NormalGgufRegistryError::MistralIdentityRequiresExternalConfig {
                    architecture: CanonicalGgufArchitecture::Llama
                }
            ));
            assert!(error.to_string().contains("--tok-model-id"));
        }

        for (name, basename) in [
            (Some("Llama 3.1 8B Instruct"), None),
            (Some("SmolLM2 1.7B Instruct"), None),
            (Some("Mixtral 8x7B Instruct v0.1"), None),
            (Some("Ministral 8B Instruct 2410"), None),
            (Some("Ministral 3 14B Instruct 2512"), None),
            (Some("OpenHermes 2.5 Mistral 7B"), Some("OpenHermes")),
            (Some("Mistral 7B Instruct"), Some("OpenHermes")),
        ] {
            let descriptor = descriptor_from_fixture(schema, &dense_metadata, &dense_tensors)
                .with_model_identity(name, basename);
            let resolved = resolve_native_adapter(&descriptor, None).unwrap();
            assert_eq!(resolved.adapter.loader, NormalLoaderType::Llama);
            assert_eq!(resolved.reason, ResolutionReason::DenseLlamaDefault);
        }

        let descriptor = descriptor_from_fixture(schema, &dense_metadata, &dense_tensors)
            .with_model_identity(Some("Mistral 7B Instruct v0.3"), Some("Mistral"));
        let resolved = resolve_native_adapter(&descriptor, Some(NormalLoaderType::Llama)).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Llama);
        assert_eq!(resolved.reason, ResolutionReason::ExplicitOverride);

        let resolved =
            resolve_native_adapter(&descriptor, Some(NormalLoaderType::Mistral)).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Mistral);
        assert_eq!(resolved.reason, ResolutionReason::ExplicitOverride);
    }

    #[test]
    fn deepseek2_resolves_identity_inventory_and_explicit_override() {
        let schema = schema_for(CanonicalGgufArchitecture::DeepSeek2);
        let (metadata, tensors) = valid_fixture(schema, &[], &[]);
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::DeepSeekV2);
        assert_eq!(resolved.reason, ResolutionReason::TensorInventory);

        let (metadata, tensors) = valid_fixture(schema, &[], &["blk.0.exp_probs_b"]);
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        let error = resolve_native_adapter(&descriptor, None).unwrap_err();
        assert!(matches!(
            error,
            NormalGgufRegistryError::AmbiguousArchitecture {
                candidates,
                ..
            } if candidates == [NormalLoaderType::DeepSeekV3, NormalLoaderType::GLM4MoeLite]
        ));

        for (name, basename, expected) in [
            (
                Some("DeepSeek-Coder-V2-Instruct"),
                None,
                NormalLoaderType::DeepSeekV2,
            ),
            (Some("DeepSeek-R1"), None, NormalLoaderType::DeepSeekV3),
            (None, Some("GLM-4.5-Air"), NormalLoaderType::GLM4MoeLite),
        ] {
            let descriptor = descriptor_from_fixture(schema, &metadata, &tensors)
                .with_model_identity(name, basename);
            let resolved = resolve_native_adapter(&descriptor, None).unwrap();
            assert_eq!(resolved.adapter.loader, expected);
            assert_eq!(resolved.reason, ResolutionReason::ModelIdentity);
        }

        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors)
            .with_model_identity(Some("GLM-4.5-Air"), None);
        let resolved =
            resolve_native_adapter(&descriptor, Some(NormalLoaderType::DeepSeekV3)).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::DeepSeekV3);
        assert_eq!(resolved.reason, ResolutionReason::ExplicitOverride);

        let error = resolve_native_adapter(&descriptor, Some(NormalLoaderType::Qwen3)).unwrap_err();
        assert!(matches!(
            error,
            NormalGgufRegistryError::ExplicitOverrideIncompatible { .. }
        ));
    }

    #[test]
    fn unsupported_semantics_fail_validation() {
        for architecture in [
            CanonicalGgufArchitecture::Glm4,
            CanonicalGgufArchitecture::Glm4Moe,
        ] {
            let schema = schema_for(architecture);
            let (metadata, tensors) =
                valid_fixture(schema, &["{arch}.rope.dimension_sections"], &[]);
            let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
            assert!(matches!(
                schema.validate(&descriptor),
                Err(NormalGgufRegistryError::UnsupportedMetadata { .. })
            ));
        }

        let schema = schema_for(CanonicalGgufArchitecture::Mistral3);
        let (metadata, tensors) =
            valid_fixture(schema, &["{arch}.attention.temperature_scale"], &[]);
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        assert!(schema.validate(&descriptor).is_ok());
    }

    #[test]
    fn missing_required_metadata_and_tensors_fail_validation() {
        let schema = schema_for(CanonicalGgufArchitecture::Qwen3Next);
        let (metadata, tensors) = valid_fixture(schema, &[], &[]);

        let metadata_without_ssm = metadata
            .iter()
            .filter(|key| !key.ends_with(".ssm.state_size"))
            .map(String::as_str)
            .collect::<Vec<_>>();
        let tensor_refs = tensors.iter().map(String::as_str).collect::<Vec<_>>();
        let descriptor = GgufDescriptor::new(
            schema.architecture.as_str(),
            &metadata_without_ssm,
            &tensor_refs,
        )
        .unwrap();
        assert!(matches!(
            schema.validate(&descriptor),
            Err(NormalGgufRegistryError::MissingMetadata { .. })
        ));

        let metadata_refs = metadata.iter().map(String::as_str).collect::<Vec<_>>();
        let tensors_without_ssm = tensors
            .iter()
            .filter(|name| !name.contains(".ssm_a"))
            .map(String::as_str)
            .collect::<Vec<_>>();
        let descriptor = GgufDescriptor::new(
            schema.architecture.as_str(),
            &metadata_refs,
            &tensors_without_ssm,
        )
        .unwrap();
        assert!(matches!(
            schema.validate(&descriptor),
            Err(NormalGgufRegistryError::MissingTensor { .. })
        ));
    }

    #[test]
    fn qwen35moe_metadata_and_inventory_resolve_to_qwen3_next() {
        let metadata = [
            "general.architecture",
            "qwen35moe.context_length",
            "qwen35moe.embedding_length",
            "qwen35moe.block_count",
            "qwen35moe.attention.head_count",
            "qwen35moe.vocab_size",
            "qwen35moe.attention.layer_norm_rms_epsilon",
            "qwen35moe.expert_count",
            "qwen35moe.expert_used_count",
            "qwen35moe.full_attention_interval",
            "qwen35moe.ssm.conv_kernel",
            "qwen35moe.ssm.inner_size",
            "qwen35moe.ssm.state_size",
        ];
        let tensors = [
            "token_embd.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight",
        ];
        let descriptor = GgufDescriptor::new("qwen35moe", &metadata, &tensors).unwrap();
        assert_eq!(
            descriptor.architecture,
            CanonicalGgufArchitecture::Qwen35Moe
        );
        schema_for(descriptor.architecture)
            .validate(&descriptor)
            .unwrap();
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Qwen3Next);
        assert_eq!(resolved.reason, ResolutionReason::SingleCandidate);
    }

    #[test]
    fn qwen35_metadata_and_inventory_resolve_to_dense_text() {
        let schema = schema_for(CanonicalGgufArchitecture::Qwen35);
        let (metadata, tensors) = valid_fixture(schema, &[], &[]);
        let descriptor = descriptor_from_fixture(schema, &metadata, &tensors);
        let resolved = resolve_native_adapter(&descriptor, None).unwrap();
        assert_eq!(resolved.adapter.loader, NormalLoaderType::Qwen3_5);
        assert_eq!(resolved.reason, ResolutionReason::SingleCandidate);
    }
}
