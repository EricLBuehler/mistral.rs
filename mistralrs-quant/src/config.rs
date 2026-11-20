use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fp8ActivationScheme {
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "quant_method", rename_all = "lowercase")]
pub enum QuantizedConfig {
    GptqAwq {
        bits: usize,
        group_size: usize,
        checkpoint_format: Option<String>,
        is_awq: bool,
    },
    Fp8 {
        weight_block_size: Option<Vec<usize>>,
        activation_scheme: Option<Fp8ActivationScheme>,
        fmt: Option<String>,
        modules_to_not_convert: Vec<String>,
    },
    #[serde(rename = "compressed-tensors")]
    CompressedTensors {
        #[serde(skip_serializing)]
        config: CheckpointQuantConfig,
        #[serde(flatten)]
        raw: serde_json::Map<String, serde_json::Value>,
    },
    #[serde(rename = "modelopt")]
    ModelOpt {
        #[serde(skip_serializing)]
        config: CheckpointQuantConfig,
        #[serde(flatten)]
        raw: serde_json::Map<String, serde_json::Value>,
    },
    Bitsandbytes {
        bnb_4bit_quant_type: Option<String>,
    },
    Afq {
        bits: usize,
        group_size: usize,
    },
    MXFP4 {},
}

// Common fields for all variants
#[derive(Deserialize)]
pub(crate) struct RawConfig {
    pub(crate) quant_method: Option<String>,
    pub(crate) bits: Option<usize>,
    pub(crate) group_size: Option<usize>,
    pub(crate) checkpoint_format: Option<String>,
    pub(crate) weight_block_size: Option<Vec<usize>>,
    pub(crate) activation_scheme: Option<Fp8ActivationScheme>,
    pub(crate) fmt: Option<String>,
    pub(crate) modules_to_not_convert: Option<Vec<String>>,
    pub(crate) ignored_layers: Option<Vec<String>>,
    pub(crate) bnb_4bit_quant_type: Option<String>,
}

pub(crate) fn raw_quantization_config(
    value: &serde_json::Value,
) -> std::result::Result<serde_json::Map<String, serde_json::Value>, String> {
    let mut raw = value
        .as_object()
        .cloned()
        .ok_or_else(|| "quantization config must be an object".to_string())?;
    raw.remove("quant_method");
    Ok(raw)
}

// Custom deserializer implementation
impl<'de> Deserialize<'de> for QuantizedConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let raw = RawConfig::deserialize(value.clone()).map_err(serde::de::Error::custom)?;

        match &raw.quant_method {
            Some(m) if m == "gptq" || m == "awq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::GptqAwq {
                    bits,
                    group_size,
                    checkpoint_format: raw.checkpoint_format,
                    is_awq: m == "awq",
                })
            }
            Some(m) if m == "fp8" => {
                // weight_block_size is optional - None means per-tensor quantization
                let mut modules_to_not_convert = raw.modules_to_not_convert.unwrap_or_default();
                for module in raw.ignored_layers.unwrap_or_default() {
                    if !modules_to_not_convert.contains(&module) {
                        modules_to_not_convert.push(module);
                    }
                }
                CheckpointQuantConfig::native(
                    raw.weight_block_size.as_deref(),
                    raw.activation_scheme,
                    raw.fmt.as_deref(),
                    &modules_to_not_convert,
                )
                .map_err(serde::de::Error::custom)?;
                Ok(QuantizedConfig::Fp8 {
                    weight_block_size: raw.weight_block_size,
                    activation_scheme: raw.activation_scheme,
                    fmt: raw.fmt,
                    modules_to_not_convert,
                })
            }
            Some(m) if m == "compressed-tensors" || m == "compressed_tensors" => {
                Ok(QuantizedConfig::CompressedTensors {
                    config: CheckpointQuantConfig::compressed_tensors(&value)
                        .map_err(serde::de::Error::custom)?,
                    raw: raw_quantization_config(&value).map_err(serde::de::Error::custom)?,
                })
            }
            Some(m) if m.to_ascii_lowercase().starts_with("modelopt") => {
                Ok(QuantizedConfig::ModelOpt {
                    config: CheckpointQuantConfig::model_opt(&value).map_err(serde::de::Error::custom)?,
                    raw: raw_quantization_config(&value).map_err(serde::de::Error::custom)?,
                })
            }
            Some(m) if m == "bitsandbytes" => Ok(QuantizedConfig::Bitsandbytes {
                bnb_4bit_quant_type: raw.bnb_4bit_quant_type,
            }),
            Some(m) if m == "afq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(m) if m == "mxfp4" => {
                Ok(QuantizedConfig::MXFP4 {  })
            }
            None if value.get("config_groups").is_some() => {
                Ok(QuantizedConfig::CompressedTensors {
                    config: CheckpointQuantConfig::compressed_tensors(&value)
                        .map_err(serde::de::Error::custom)?,
                    raw: raw_quantization_config(&value).map_err(serde::de::Error::custom)?,
                })
            }
            None => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(unknown_method) => {
                Err(serde::de::Error::custom(format!(
                    "Unknown quantization method: {unknown_method}. Expected one of: gptq, fp8, compressed-tensors, modelopt, bitsandbytes, afq, or not specified"
                )))
            },
        }
    }
}

impl QuantizedConfig {
    pub fn from_modelopt_config(config: &serde_json::Value) -> std::result::Result<Self, String> {
        Ok(Self::ModelOpt {
            config: CheckpointQuantConfig::model_opt(config)?,
            raw: raw_quantization_config(config)?,
        })
    }

    pub fn from_modelopt_configs(
        embedded: Option<&serde_json::Value>,
        external: &serde_json::Value,
    ) -> std::result::Result<Self, String> {
        let merged = CheckpointQuantConfig::model_opt_merged_value(embedded, external)?;
        Ok(Self::ModelOpt {
            config: CheckpointQuantConfig::model_opt(&merged)?,
            raw: raw_quantization_config(&merged)?,
        })
    }

    pub fn resolve_checkpoint(&self, prefix: &str) -> Result<Option<CheckpointLinearSpec>> {
        match self {
            Self::Fp8 {
                weight_block_size,
                activation_scheme,
                fmt,
                modules_to_not_convert,
            } => CheckpointQuantConfig::native(
                weight_block_size.as_deref(),
                *activation_scheme,
                fmt.as_deref(),
                modules_to_not_convert,
            )
            .map_err(candle_core::Error::msg)?
            .resolve_checked(prefix)
            .map_err(candle_core::Error::msg),
            Self::CompressedTensors { config, .. } | Self::ModelOpt { config, .. } => config
                .resolve_checked(prefix)
                .map_err(candle_core::Error::msg),
            _ => Ok(None),
        }
    }

    pub fn resolve_fp8(&self, prefix: &str) -> Result<Option<Fp8LinearSpec>> {
        match self.resolve_checkpoint(prefix)? {
            Some(CheckpointLinearSpec::Fp8(spec)) => Ok(Some(spec)),
            Some(CheckpointLinearSpec::Nvfp4(_)) => {
                candle_core::bail!("NVFP4 layer `{prefix}` requires the NVFP4 loader")
            }
            Some(CheckpointLinearSpec::Mxfp8(_)) => {
                candle_core::bail!("MXFP8 layer `{prefix}` requires the MXFP8 loader")
            }
            None => Ok(None),
        }
    }

    pub fn resolve_explicit_checkpoint(
        &self,
        prefix: &str,
    ) -> Result<Option<CheckpointLinearSpec>> {
        match self {
            Self::CompressedTensors { config, .. } | Self::ModelOpt { config, .. } => config
                .resolve_explicit_checked(prefix)
                .map_err(candle_core::Error::msg),
            _ => Ok(None),
        }
    }

    /// Explicit rules require quantized loading; global hints require scale tensors at this prefix.
    pub fn has_checkpoint_quantization(&self, vb: &ShardedVarBuilder) -> Result<bool> {
        let prefix = vb.prefix();
        Ok(self.resolve_explicit_checkpoint(&prefix)?.is_some()
            || self
                .resolve_checkpoint(&prefix)?
                .is_some_and(|spec| spec.has_weight_scales(vb)))
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::GptqAwq { .. } => "gptq",
            Self::Fp8 { .. } => "fp8",
            Self::CompressedTensors { .. } => "compressed-tensors",
            Self::ModelOpt { .. } => "modelopt",
            Self::Bitsandbytes { .. } => "bitsandbytes",
            Self::Afq { .. } => "afq",
            Self::MXFP4 { .. } => "mxfp4",
        }
    }

    pub fn get_bits_name(&self, vb: &ShardedVarBuilder) -> String {
        match self {
            Self::GptqAwq { bits, .. } => format!("{bits} bits"),
            Self::Fp8 { .. } => "8 bits".to_string(),
            Self::CompressedTensors { .. } | Self::ModelOpt { .. } => {
                match self.resolve_checkpoint(&vb.prefix()) {
                    Ok(Some(spec)) => format!("{} bits", spec.bits()),
                    _ => "unquantized".to_string(),
                }
            }
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            } => "4 bits".to_string(),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => "8 bits".to_string(),
            Self::Afq { bits, .. } => format!("{bits} bits"),
            Self::MXFP4 {} => format!("{} bits", mxfp4::N_BITS),
        }
    }

    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::GptqAwq { bits, .. } => match bits {
                2 => IsqType::Q2K.pack_factor(dtype),
                3 => IsqType::Q3K.pack_factor(dtype),
                4 => IsqType::Q4K.pack_factor(dtype),
                5 => IsqType::Q5K.pack_factor(dtype),
                6 => IsqType::Q6K.pack_factor(dtype),
                8 => IsqType::Q8_0.pack_factor(dtype),
                40 => IsqType::MXFP4.pack_factor(dtype),
                other => panic!("Unexpected bits in `pack_factor` {other}"),
            },
            Self::Fp8 { .. } => IsqType::F8E4M3.pack_factor(dtype),
            Self::CompressedTensors { .. } | Self::ModelOpt { .. } => 1,
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            } => IsqType::Q4K.pack_factor(dtype),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => 1,
            Self::Afq { bits, group_size } => afq_pack_factor(dtype, *bits, *group_size),
            Self::MXFP4 {} => IsqType::MXFP4.pack_factor(dtype),
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    GptqAwq {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        qzeros: Option<Tensor>,
        scales: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        workspace: Option<Tensor>,
        is_marlin: bool,
        is_awq: bool,
    },
    Gguf {
        q_weight: Arc<QTensor>,
        b: Option<Tensor>,
    },
    Unquantized(Linear),
    Hqq {
        tensor: Tensor,
        bits: HqqBits,
        group_size: NonZeroUsize,
        axis: HqqAxis,
        optimization_steps: Option<usize>,
        round_zeros: Option<bool>,
        channel_wise: Option<bool>,
        bias: Option<Tensor>,
    },
    Dummy,
    FP8 {
        lin: Linear,
        dtype: DType,
    },
    Bnb {
        weight: Tensor,
        bias: Option<Tensor>,
        params: BnbQuantParams,
        quant_ty: BnbQuantType,
    },
    BlockwiseFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        bias: Option<Tensor>,
        dequant_dtype: DType,
        weight_block_size: Vec<usize>,
        activation_scheme: Option<Fp8ActivationScheme>,
    },
    PerTensorFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        activation_mode: Fp8ActivationMode,
        activation_scale: Option<Tensor>,
        bias: Option<Tensor>,
        dequant_dtype: DType,
    },
    Afq {
        weight: Tensor,
        bias: Option<Tensor>,
        bits: AfqBits,
        group_size: AfqGroupSize,
    },
    MXFP4 {
        blocks: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    },
}
