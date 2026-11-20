use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::{cmp::Ordering, sync::Arc};

use crate::QuantMethod;
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Linear;

const FP8_BITS: u64 = 8;
const NVFP4_BITS: u64 = 4;
pub const NVFP4_BLOCK_SIZE: usize = 16;
const NVFP4_COMPRESSED_FORMAT: &str = "nvfp4-pack-quantized";
const MODEL_OPT_BLOCK_SIZE: usize = 128;
const FP8_WEIGHT_SCALE_ALIASES: &[&str] = &["weight_scale", "weight_scale_inv"];
const MODEL_OPT_LEGACY_VISION_EXCLUSIONS: &[&str] =
    &["vision_tower", "vision_model", "vit_large_projector"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointDialect {
    Native,
    CompressedTensors,
    ModelOpt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp8WeightScaleLayout {
    Tensor,
    Channel,
    Block([usize; 2]),
}

impl Fp8WeightScaleLayout {
    pub fn logical_shape(self, weight_shape: [usize; 2]) -> CandleResult<Vec<usize>> {
        let [out_dim, in_dim] = weight_shape;
        Ok(match self {
            Self::Tensor => Vec::new(),
            Self::Channel => vec![out_dim],
            Self::Block([rows, cols]) => {
                if rows == 0 || cols == 0 {
                    candle_core::bail!("FP8 block scale dimensions must be positive");
                }
                vec![out_dim.div_ceil(rows), in_dim.div_ceil(cols)]
            }
        })
    }

    pub fn normalize(self, scale: Tensor, weight_shape: [usize; 2]) -> CandleResult<Tensor> {
        let [out_dim, in_dim] = weight_shape;
        let accepted = match self {
            Self::Tensor => vec![Vec::new(), vec![1]],
            Self::Channel => vec![vec![out_dim], vec![out_dim, 1]],
            Self::Block([rows, cols]) => {
                if rows == 0 || cols == 0 {
                    candle_core::bail!("FP8 block scale dimensions must be positive");
                }
                let scale_rows = out_dim.div_ceil(rows);
                let scale_cols = in_dim.div_ceil(cols);
                vec![
                    vec![scale_rows, scale_cols],
                    vec![scale_rows, 1, scale_cols, 1],
                ]
            }
        };
        if !accepted.iter().any(|shape| scale.dims() == shape) {
            candle_core::bail!(
                "FP8 {:?} weight scale has shape {:?}, expected one of {:?}",
                self,
                scale.dims(),
                accepted
            );
        }
        scale.reshape(accepted[0].as_slice())?.to_dtype(DType::F32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp8ActivationMode {
    None,
    StaticTensor,
    DynamicToken,
    DynamicBlock(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Fp8ScaleNames {
    pub weight: &'static [&'static str],
    pub activation: &'static [&'static str],
}

impl Fp8ScaleNames {
    pub fn weight_name(&self, vb: &crate::ShardedVarBuilder) -> CandleResult<&'static str> {
        resolve_scale_name(vb, self.weight, "weight")?.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "missing FP8 weight scale at prefix `{}`; expected one of {}",
                vb.prefix(),
                self.weight.join(", ")
            ))
        })
    }

    pub fn activation_name(
        &self,
        vb: &crate::ShardedVarBuilder,
    ) -> CandleResult<Option<&'static str>> {
        resolve_scale_name(vb, self.activation, "activation")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Fp8LinearSpec {
    pub dialect: CheckpointDialect,
    pub weight_scale: Fp8WeightScaleLayout,
    pub activation: Fp8ActivationMode,
    pub scale_names: Fp8ScaleNames,
}

impl Fp8LinearSpec {
    fn new(
        dialect: CheckpointDialect,
        weight_scale: Fp8WeightScaleLayout,
        activation: Fp8ActivationMode,
    ) -> Self {
        let weight: &'static [&'static str] = match (dialect, weight_scale) {
            (CheckpointDialect::Native, Fp8WeightScaleLayout::Block(_)) => {
                &["weight_scale_inv", "weight_scale"]
            }
            (CheckpointDialect::Native, _)
            | (CheckpointDialect::CompressedTensors, _)
            | (CheckpointDialect::ModelOpt, _) => &["weight_scale", "weight_scale_inv"],
        };
        let scale_activation: &'static [&'static str] = match activation {
            Fp8ActivationMode::StaticTensor => match dialect {
                CheckpointDialect::Native => &["input_scale", "activation_scale"],
                CheckpointDialect::CompressedTensors | CheckpointDialect::ModelOpt => {
                    &["input_scale", "activation_scale"]
                }
            },
            Fp8ActivationMode::None
            | Fp8ActivationMode::DynamicToken
            | Fp8ActivationMode::DynamicBlock(_) => &[],
        };
        Self {
            dialect,
            weight_scale,
            activation,
            scale_names: Fp8ScaleNames {
                weight,
                activation: scale_activation,
            },
        }
    }

    pub fn normalize_activation_scale(&self, scale: Tensor) -> CandleResult<Tensor> {
        if self.activation != Fp8ActivationMode::StaticTensor {
            candle_core::bail!("FP8 activation scale is only stored for static tensor activation");
        }
        if !scale.dims().is_empty() && scale.dims() != [1] {
            candle_core::bail!(
                "FP8 static activation scale has shape {:?}, expected a scalar",
                scale.dims()
            );
        }
        scale.reshape(())?.to_dtype(DType::F32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Nvfp4ActivationMode {
    None,
    DynamicBlock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Nvfp4ScaleEncoding {
    Plain,
    Csf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Mxfp8LinearSpec {
    pub group_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleConvention {
    Dequantize,
    Quantize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Nvfp4ScaleNames {
    pub weight: &'static str,
    pub block_scale: &'static str,
    pub global_scale: &'static str,
    pub activation_scale: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Nvfp4LinearSpec {
    pub dialect: CheckpointDialect,
    pub activation: Nvfp4ActivationMode,
    pub scale_names: Nvfp4ScaleNames,
    pub global_scale: ScaleConvention,
    pub weight_scale_encoding: Nvfp4ScaleEncoding,
}

impl Nvfp4LinearSpec {
    fn model_opt(activation: Nvfp4ActivationMode) -> Self {
        Self {
            dialect: CheckpointDialect::ModelOpt,
            activation,
            scale_names: Nvfp4ScaleNames {
                weight: "weight",
                block_scale: "weight_scale",
                global_scale: "weight_scale_2",
                activation_scale: (activation == Nvfp4ActivationMode::DynamicBlock)
                    .then_some("input_scale"),
            },
            global_scale: ScaleConvention::Dequantize,
            weight_scale_encoding: Nvfp4ScaleEncoding::Plain,
        }
    }

    fn compressed_tensors(activation: Nvfp4ActivationMode) -> Self {
        Self {
            dialect: CheckpointDialect::CompressedTensors,
            activation,
            scale_names: Nvfp4ScaleNames {
                weight: "weight_packed",
                block_scale: "weight_scale",
                global_scale: "weight_global_scale",
                activation_scale: (activation == Nvfp4ActivationMode::DynamicBlock)
                    .then_some("input_global_scale"),
            },
            // compressed-tensors' ModelOpt converter reciprocates both global scales.
            global_scale: ScaleConvention::Quantize,
            weight_scale_encoding: Nvfp4ScaleEncoding::Plain,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointLinearSpec {
    Fp8(Fp8LinearSpec),
    Mxfp8(Mxfp8LinearSpec),
    Nvfp4(Nvfp4LinearSpec),
}

impl CheckpointLinearSpec {
    pub fn bits(self) -> usize {
        match self {
            Self::Nvfp4(_) => NVFP4_BITS as usize,
            Self::Fp8(_) | Self::Mxfp8(_) => FP8_BITS as usize,
        }
    }

    pub fn pack_factor(self, dtype: DType) -> usize {
        (dtype.size_in_bytes() * u8::BITS as usize / self.bits()).max(1)
    }

    /// Resident weight and scale bytes for a matrix or stack of matrices.
    pub fn size_in_bytes(self, shape: &[usize]) -> CandleResult<usize> {
        let matrices = shape[..shape.len() - 2].iter().product::<usize>();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let elements = rows * cols;
        let float_bytes = DType::F32.size_in_bytes();
        let matrix_bytes = match self {
            Self::Nvfp4(spec) => {
                elements.div_ceil(2)
                    + rows * cols.div_ceil(NVFP4_BLOCK_SIZE)
                    + rows * float_bytes
                    + usize::from(spec.activation == Nvfp4ActivationMode::DynamicBlock)
                        * float_bytes
            }
            Self::Mxfp8(spec) => elements + rows * cols.div_ceil(spec.group_size),
            Self::Fp8(spec) => {
                let scales = spec
                    .weight_scale
                    .logical_shape([rows, cols])?
                    .iter()
                    .product::<usize>();
                elements
                    + scales * float_bytes
                    + usize::from(spec.activation == Fp8ActivationMode::StaticTensor) * float_bytes
            }
        };
        Ok(matrices * matrix_bytes)
    }

    pub fn has_weight_scales(self, vb: &crate::ShardedVarBuilder) -> bool {
        match self {
            Self::Fp8(spec) => spec
                .scale_names
                .weight
                .iter()
                .any(|name| vb.contains_tensor(name)),
            Self::Nvfp4(spec) => crate::Nvfp4Layer::has_checkpoint_scales(vb, spec),
            Self::Mxfp8(_) => vb.contains_tensor("weight_scale"),
        }
    }
}

#[derive(Clone, Debug)]
struct CheckpointPattern(Arc<Regex>);

impl PartialEq for CheckpointPattern {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_str() == other.0.as_str()
    }
}

impl Eq for CheckpointPattern {}

impl Serialize for CheckpointPattern {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckpointTarget {
    Linear,
    Exact(String),
    Module(String),
    Contains(String),
    Regex(CheckpointPattern),
    Glob(CheckpointPattern),
}

impl CheckpointTarget {
    fn compressed_tensors(raw: &str) -> Result<Self, String> {
        if raw == "Linear" {
            return Ok(Self::Linear);
        }
        if let Some(pattern) = raw.strip_prefix("re:") {
            let compiled = Regex::new(pattern)
                .map_err(|err| format!("invalid compressed-tensors target regex `{raw}`: {err}"))?;
            return Ok(Self::Regex(CheckpointPattern(Arc::new(compiled))));
        }
        Ok(Self::Exact(raw.to_string()))
    }

    fn model_opt(raw: &str) -> Result<Self, String> {
        if raw.contains(['*', '?', '[']) {
            let pattern = glob_regex(raw)?;
            return Ok(Self::Glob(pattern));
        }
        Ok(Self::Contains(raw.to_string()))
    }

    fn matches(&self, prefix: &str) -> bool {
        match self {
            Self::Linear => true,
            Self::Exact(target) => target == prefix,
            Self::Module(target) => {
                target == prefix
                    || prefix
                        .strip_prefix(target)
                        .is_some_and(|suffix| suffix.starts_with('.'))
            }
            Self::Contains(target) => prefix.contains(target),
            Self::Regex(pattern) | Self::Glob(pattern) => pattern.0.is_match(prefix),
        }
    }

    fn priority(&self) -> usize {
        match self {
            Self::Exact(_) => 0,
            Self::Module(_) | Self::Contains(_) | Self::Regex(_) | Self::Glob(_) => 1,
            Self::Linear => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct CheckpointConfigRule {
    targets: Vec<CheckpointTarget>,
    resolution: CheckpointRuleResolution,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
enum CheckpointRuleResolution {
    Quantized(CheckpointLinearSpec),
    Unquantized,
    Unsupported(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckpointDirectResolution {
    NoMatch,
    Resolved(Option<CheckpointLinearSpec>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckpointQuantConfig {
    dialect: CheckpointDialect,
    rules: Vec<CheckpointConfigRule>,
    ignore: Vec<CheckpointTarget>,
}

impl CheckpointQuantConfig {
    pub fn dialect(&self) -> CheckpointDialect {
        self.dialect
    }

    pub fn resolve(&self, prefix: &str) -> Option<CheckpointLinearSpec> {
        self.resolve_checked(prefix).ok().flatten()
    }

    pub fn resolve_fp8(&self, prefix: &str) -> Result<Option<Fp8LinearSpec>, String> {
        match self.resolve_checked(prefix)? {
            Some(CheckpointLinearSpec::Fp8(spec)) => Ok(Some(spec)),
            Some(CheckpointLinearSpec::Nvfp4(_)) => Err(format!(
                "module `{prefix}` uses NVFP4, but an FP8 specification was requested"
            )),
            Some(CheckpointLinearSpec::Mxfp8(_)) => Err(format!(
                "module `{prefix}` uses MXFP8, but an FP8 specification was requested"
            )),
            None => Ok(None),
        }
    }

    pub fn resolve_checked(&self, prefix: &str) -> Result<Option<CheckpointLinearSpec>, String> {
        match self.resolve_state_checked(prefix, true)? {
            CheckpointDirectResolution::NoMatch => Ok(None),
            CheckpointDirectResolution::Resolved(spec) => Ok(spec),
        }
    }

    pub fn resolve_explicit_checked(
        &self,
        prefix: &str,
    ) -> Result<Option<CheckpointLinearSpec>, String> {
        match self.resolve_state_checked(prefix, false)? {
            CheckpointDirectResolution::NoMatch => Ok(None),
            CheckpointDirectResolution::Resolved(spec) => Ok(spec),
        }
    }

    fn resolve_state_checked(
        &self,
        prefix: &str,
        include_linear_default: bool,
    ) -> Result<CheckpointDirectResolution, String> {
        let prefixes = prefix_candidates(prefix);
        if self
            .ignore
            .iter()
            .any(|target| prefixes.iter().any(|prefix| target.matches(prefix)))
        {
            return Ok(CheckpointDirectResolution::Resolved(None));
        }
        let exact = self.resolve_rules_at_priority(prefix, &prefixes, 0)?;
        if exact != CheckpointDirectResolution::NoMatch {
            return Ok(exact);
        }
        if let Some(constituents) = fused_constituents(prefix) {
            let fused = self.resolve_fused_state(&constituents, include_linear_default)?;
            if fused != CheckpointDirectResolution::NoMatch {
                return Ok(fused);
            }
        }
        let priorities = if include_linear_default { 1..3 } else { 1..2 };
        for priority in priorities {
            let resolved = self.resolve_rules_at_priority(prefix, &prefixes, priority)?;
            if resolved != CheckpointDirectResolution::NoMatch {
                return Ok(resolved);
            }
        }
        Ok(CheckpointDirectResolution::NoMatch)
    }

    fn resolve_rules_at_priority(
        &self,
        prefix: &str,
        prefixes: &[String],
        priority: usize,
    ) -> Result<CheckpointDirectResolution, String> {
        for rule in &self.rules {
            if rule.targets.iter().any(|target| {
                target.priority() == priority
                    && prefixes.iter().any(|prefix| target.matches(prefix))
            }) {
                return match &rule.resolution {
                    CheckpointRuleResolution::Quantized(spec) => {
                        Ok(CheckpointDirectResolution::Resolved(Some(*spec)))
                    }
                    CheckpointRuleResolution::Unquantized => {
                        Ok(CheckpointDirectResolution::Resolved(None))
                    }
                    CheckpointRuleResolution::Unsupported(reason) => Err(format!(
                        "unsupported quantization scheme for `{prefix}`: {reason}"
                    )),
                };
            }
        }
        Ok(CheckpointDirectResolution::NoMatch)
    }

    fn resolve_fused_state<S: AsRef<str>>(
        &self,
        prefixes: &[S],
        include_linear_default: bool,
    ) -> Result<CheckpointDirectResolution, String> {
        let Some((first, rest)) = prefixes.split_first() else {
            return Ok(CheckpointDirectResolution::NoMatch);
        };
        let resolved = self.resolve_state_checked(first.as_ref(), include_linear_default)?;
        for prefix in rest {
            let next = self.resolve_state_checked(prefix.as_ref(), include_linear_default)?;
            if next != resolved {
                return Err(format!(
                    "fused quantized projection has different schemes for `{}` and `{}`",
                    first.as_ref(),
                    prefix.as_ref()
                ));
            }
        }
        Ok(resolved)
    }

    pub(crate) fn native(
        weight_block_size: Option<&[usize]>,
        activation_scheme: Option<crate::Fp8ActivationScheme>,
        format: Option<&str>,
        modules_to_not_convert: &[String],
    ) -> Result<Self, String> {
        validate_e4m3(format)?;
        let weight_scale = match weight_block_size {
            Some(size) => Fp8WeightScaleLayout::Block(block_size(size, "weight_block_size")?),
            None => Fp8WeightScaleLayout::Tensor,
        };
        let activation = match activation_scheme {
            Some(crate::Fp8ActivationScheme::Static) => Fp8ActivationMode::StaticTensor,
            Some(crate::Fp8ActivationScheme::Dynamic) => match weight_scale {
                Fp8WeightScaleLayout::Block([_, cols]) => Fp8ActivationMode::DynamicBlock(cols),
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => {
                    Fp8ActivationMode::DynamicToken
                }
            },
            None => Fp8ActivationMode::None,
        };
        Ok(Self {
            dialect: CheckpointDialect::Native,
            rules: vec![CheckpointConfigRule {
                targets: vec![CheckpointTarget::Linear],
                resolution: CheckpointRuleResolution::Quantized(CheckpointLinearSpec::Fp8(
                    Fp8LinearSpec::new(CheckpointDialect::Native, weight_scale, activation),
                )),
            }],
            ignore: modules_to_not_convert
                .iter()
                .map(|target| CheckpointTarget::Exact(target.clone()))
                .collect(),
        })
    }

    pub fn compressed_tensors(value: &Value) -> Result<Self, String> {
        let object = value.as_object().ok_or_else(|| {
            "compressed-tensors quantization config must be an object".to_string()
        })?;
        let groups = object
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                "compressed-tensors quantization config requires `config_groups`".to_string()
            })?;
        let mut rules = Vec::new();
        let mut has_quantized = false;
        let mut groups = groups.iter().collect::<Vec<_>>();
        groups.sort_by(|(left, _), (right, _)| natural_group_cmp(left, right));
        for (name, group) in groups {
            let group = group.as_object().ok_or_else(|| {
                format!("compressed-tensors config group `{name}` must be an object")
            })?;
            let format = group.get("format").or_else(|| object.get("format"));
            let resolution = compressed_tensors_spec(group, name, format.and_then(Value::as_str))?;
            has_quantized |= matches!(&resolution, CheckpointRuleResolution::Quantized(_));
            let targets = string_array(
                group.get("targets"),
                &format!("config_groups.{name}.targets"),
            )?
            .into_iter()
            .map(|target| CheckpointTarget::compressed_tensors(&target))
            .collect::<Result<Vec<_>, _>>()?;
            if targets.is_empty() {
                return Err(format!(
                    "compressed-tensors config group `{name}` requires at least one target"
                ));
            }
            rules.push(CheckpointConfigRule {
                targets,
                resolution,
            });
        }
        if !has_quantized {
            return Err(
                "compressed-tensors config contains no supported FP8 or NVFP4 groups".to_string(),
            );
        }
        let ignore = optional_string_array(object.get("ignore"), "ignore")?
            .into_iter()
            .map(|target| CheckpointTarget::compressed_tensors(&target))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dialect: CheckpointDialect::CompressedTensors,
            rules,
            ignore,
        })
    }

    pub fn model_opt(value: &Value) -> Result<Self, String> {
        let root = value
            .as_object()
            .ok_or_else(|| "ModelOpt quantization config must be an object".to_string())?;
        let config = root
            .get("quantization")
            .and_then(Value::as_object)
            .unwrap_or(root);
        let algorithm = config
            .get("quant_algo")
            .or_else(|| root.get("quant_algo"))
            .and_then(Value::as_str)
            .ok_or_else(|| "ModelOpt quantization config requires `quant_algo`".to_string())?
            .to_ascii_uppercase();
        let ignore_value = config.get("exclude_modules").or_else(|| root.get("ignore"));
        let mut ignore = optional_string_array(ignore_value, "exclude_modules or ignore")?
            .into_iter()
            .map(|target| CheckpointTarget::model_opt(&target))
            .collect::<Result<Vec<_>, _>>()?;
        if algorithm != "MIXED_PRECISION" {
            ignore.extend(
                MODEL_OPT_LEGACY_VISION_EXCLUSIONS
                    .iter()
                    .map(|target| CheckpointTarget::Contains((*target).to_string())),
            );
        }

        let rules = if algorithm == "MIXED_PRECISION" {
            let layers = config
                .get("quantized_layers")
                .or_else(|| root.get("quantized_layers"))
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    "ModelOpt MIXED_PRECISION config requires `quantized_layers`".to_string()
                })?;
            let mut rules = Vec::new();
            let mut has_quantized = false;
            for (prefix, layer) in layers {
                let layer = layer.as_object().ok_or_else(|| {
                    format!("ModelOpt quantized layer `{prefix}` must be an object")
                })?;
                let algorithm = layer
                    .get("quant_algo")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        format!("ModelOpt quantized layer `{prefix}` requires `quant_algo`")
                    })?
                    .to_ascii_uppercase();
                let target = if prefix.ends_with(".experts") {
                    CheckpointTarget::Module(prefix.clone())
                } else {
                    CheckpointTarget::Exact(prefix.clone())
                };
                match algorithm.as_str() {
                    "NONE" | "UNQUANTIZED" => rules.push(CheckpointConfigRule {
                        targets: vec![target],
                        resolution: CheckpointRuleResolution::Unquantized,
                    }),
                    _ => {
                        has_quantized = true;
                        rules.push(CheckpointConfigRule {
                            targets: vec![target],
                            resolution: CheckpointRuleResolution::Quantized(model_opt_spec(
                                &algorithm,
                                layer.get("group_size").or_else(|| config.get("group_size")),
                                layer
                                    .get("weight_scale_encoding")
                                    .or_else(|| config.get("weight_scale_encoding")),
                            )?),
                        });
                    }
                }
            }
            if !has_quantized {
                return Err(
                    "ModelOpt MIXED_PRECISION config contains no supported FP8 or NVFP4 layers"
                        .to_string(),
                );
            }
            rules
        } else {
            let spec = model_opt_spec(
                &algorithm,
                config.get("group_size"),
                config.get("weight_scale_encoding"),
            )?;
            vec![CheckpointConfigRule {
                targets: vec![CheckpointTarget::Linear],
                resolution: CheckpointRuleResolution::Quantized(spec),
            }]
        };
        Ok(Self {
            dialect: CheckpointDialect::ModelOpt,
            rules,
            ignore,
        })
    }

    pub fn model_opt_merged(embedded: Option<&Value>, external: &Value) -> Result<Self, String> {
        let merged = Self::model_opt_merged_value(embedded, external)?;
        Self::model_opt(&merged)
    }

    pub(crate) fn model_opt_merged_value(
        embedded: Option<&Value>,
        external: &Value,
    ) -> Result<Value, String> {
        let mut merged = model_opt_extra_fields(external)?;
        merged.extend(model_opt_fields(external)?);
        if let Some(embedded) = embedded {
            merged.extend(model_opt_extra_fields(embedded)?);
            merged.extend(model_opt_fields(embedded)?);
        }
        merged.insert(
            "quant_method".to_string(),
            Value::String("modelopt".to_string()),
        );
        Ok(Value::Object(merged))
    }
}

pub(crate) fn checkpoint_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &crate::QuantizedConfig,
    bias: bool,
    hints: crate::Shard,
    vb: crate::ShardedVarBuilder,
) -> CandleResult<Arc<dyn crate::QuantMethod>> {
    match config.resolve_checkpoint(&vb.prefix())? {
        Some(CheckpointLinearSpec::Nvfp4(spec)) => {
            return crate::nvfp4::Nvfp4Layer::linear_b(in_dim, out_dim, spec, bias, hints, vb);
        }
        Some(CheckpointLinearSpec::Mxfp8(_)) => {
            return Ok(Arc::new(crate::mxfp8::Mxfp8Linear::load(
                in_dim, out_dim, bias, hints, vb,
            )?));
        }
        _ => {}
    }
    let Some(spec) = config.resolve_fp8(&vb.prefix())? else {
        if matches!(config, crate::QuantizedConfig::Fp8 { .. })
            && resolve_scale_name(&vb, FP8_WEIGHT_SCALE_ALIASES, "weight")?.is_some()
        {
            candle_core::bail!(
                "FP8-excluded module `{}` unexpectedly has a weight scale",
                vb.prefix()
            );
        }
        return unquantized_linear_b(in_dim, out_dim, bias, hints, vb);
    };
    if !vb.contains_tensor("weight") {
        return crate::make_dummy_or_error("fp8_linear", &vb, &["weight"]);
    }
    let scale_name = match resolve_scale_name(&vb, spec.scale_names.weight, "weight")? {
        Some(name) => name,
        None if matches!(
            config,
            crate::QuantizedConfig::Fp8 {
                modules_to_not_convert,
                ..
            } if modules_to_not_convert.is_empty()
        ) =>
        {
            return unquantized_linear_b(in_dim, out_dim, bias, hints, vb);
        }
        None => {
            candle_core::bail!(
                "missing FP8 weight scale at prefix `{}`; expected one of {}",
                vb.prefix(),
                spec.scale_names.weight.join(", ")
            )
        }
    };
    let CheckpointFp8Tensors {
        weight,
        weight_scale,
        activation_scale,
    } = load_checkpoint_fp8_tensors([out_dim, in_dim], spec, scale_name, hints, &vb)?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };
    let dequant_dtype = bias
        .as_ref()
        .map(Tensor::dtype)
        .filter(|dtype| matches!(dtype, DType::F16 | DType::BF16 | DType::F32))
        .unwrap_or_else(|| checkpoint_fp8_dequant_dtype(vb.dtype()));

    match spec.activation {
        Fp8ActivationMode::DynamicBlock(_) => {
            let Fp8WeightScaleLayout::Block(block_size) = spec.weight_scale else {
                candle_core::bail!("blockwise FP8 activation requires blockwise weights");
            };
            Ok(Arc::new(crate::BlockwiseFP8Linear::new(
                crate::QuantMethodConfig::BlockwiseFP8 {
                    weight,
                    weight_scale_inv: weight_scale,
                    bias,
                    dequant_dtype,
                    weight_block_size: block_size.to_vec(),
                    activation_scheme: Some(crate::Fp8ActivationScheme::Dynamic),
                },
            )?))
        }
        Fp8ActivationMode::None => {
            crate::fp8_w8a16_linear(weight, weight_scale, spec.weight_scale, bias, dequant_dtype)
        }
        Fp8ActivationMode::StaticTensor | Fp8ActivationMode::DynamicToken => {
            crate::fp8_w8a8_linear(crate::Fp8W8A8LinearArgs {
                weight,
                weight_scale,
                weight_scale_layout: spec.weight_scale,
                activation_mode: spec.activation,
                activation_scale,
                bias,
                dequant_dtype,
            })
        }
    }
}

pub(crate) struct CheckpointFp8Tensors {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub activation_scale: Option<Tensor>,
}

pub(crate) fn checkpoint_fp8_dequant_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 | DType::F32 => dtype,
        _ => DType::BF16,
    }
}

pub(crate) fn checkpoint_fp8_weight_scale_name(
    vb: &crate::ShardedVarBuilder,
    spec: Fp8LinearSpec,
) -> CandleResult<Option<&'static str>> {
    resolve_scale_name(vb, spec.scale_names.weight, "weight")
}

pub(crate) fn load_checkpoint_fp8_tensors(
    weight_shape: [usize; 2],
    spec: Fp8LinearSpec,
    scale_name: &'static str,
    hints: crate::Shard,
    vb: &crate::ShardedVarBuilder,
) -> CandleResult<CheckpointFp8Tensors> {
    let scale_shape = match vb.tensor_shape(scale_name) {
        Some(shape) => shape.to_vec(),
        None => spec.weight_scale.logical_shape(weight_shape)?,
    };
    let fused_partitions = fused_constituents(&vb.prefix()).map(|parts| parts.len());
    let scale_hints = scale_shard(
        spec.weight_scale,
        weight_shape,
        &scale_shape,
        hints,
        fused_partitions,
    )?;
    let weight = vb.get_with_hints_dtype(
        (weight_shape[0], weight_shape[1]),
        "weight",
        hints,
        DType::F8E4M3,
    )?;
    let weight_scale = vb.get_with_hints_dtype(scale_shape, scale_name, scale_hints, DType::F32)?;
    let local_shape: [usize; 2] = weight.dims2().map(|(rows, cols)| [rows, cols])?;
    let weight_scale = spec.weight_scale.normalize(weight_scale, local_shape)?;
    let activation_scale = if spec.activation == Fp8ActivationMode::StaticTensor {
        let name = spec.scale_names.activation_name(vb)?.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "missing FP8 static activation scale at prefix `{}`; expected one of {}",
                vb.prefix(),
                spec.scale_names.activation.join(", ")
            ))
        })?;
        let shape = vb
            .tensor_shape(name)
            .map(<[usize]>::to_vec)
            .unwrap_or_default();
        let scale_hints = tensor_scale_shard(weight_shape, &shape, hints, fused_partitions)?;
        Some(spec.normalize_activation_scale(vb.get_with_hints_dtype(
            shape,
            name,
            scale_hints,
            DType::F32,
        )?)?)
    } else {
        None
    };
    Ok(CheckpointFp8Tensors {
        weight,
        weight_scale,
        activation_scale,
    })
}

fn unquantized_linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    hints: crate::Shard,
    vb: crate::ShardedVarBuilder,
) -> CandleResult<Arc<dyn crate::QuantMethod>> {
    if !vb.contains_tensor("weight") {
        return crate::make_dummy_or_error("fp8_excluded_linear", &vb, &["weight"]);
    }
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", hints)?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };
    Ok(Arc::new(crate::UnquantLinear::new(
        crate::QuantMethodConfig::Unquantized(Linear::new(weight, bias)),
    )?))
}

fn scale_shard(
    layout: Fp8WeightScaleLayout,
    weight_shape: [usize; 2],
    checkpoint_shape: &[usize],
    weight_shard: crate::Shard,
    fused_partitions: Option<usize>,
) -> CandleResult<crate::Shard> {
    match layout {
        Fp8WeightScaleLayout::Tensor => tensor_scale_shard(
            weight_shape,
            checkpoint_shape,
            weight_shard,
            fused_partitions,
        ),
        Fp8WeightScaleLayout::Channel => {
            let weight_dim = shard_dim(weight_shard);
            if weight_dim != 0 {
                return Ok(Default::default());
            }
            let axis = checkpoint_shape
                .iter()
                .position(|dim| *dim == weight_shape[0])
                .ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "FP8 channel scale shape {checkpoint_shape:?} does not contain output dimension {}",
                        weight_shape[0]
                    ))
                })?;
            remap_shard(weight_shard, axis)
        }
        Fp8WeightScaleLayout::Block(block_size) => {
            let canonical = crate::blockwise_fp8::scale_shard_from_weight_shard(
                weight_shape,
                block_size,
                weight_shard,
            )?;
            let axes = match checkpoint_shape {
                [_, _] => [0, 1],
                [_, 1, _, 1] => [0, 2],
                _ => candle_core::bail!(
                    "unsupported FP8 block scale checkpoint shape {checkpoint_shape:?}; expected rank 2 or ModelOpt rank 4"
                ),
            };
            remap_shard(canonical, axes[shard_dim(weight_shard)])
        }
    }
}

fn tensor_scale_shard(
    weight_shape: [usize; 2],
    checkpoint_shape: &[usize],
    weight_shard: crate::Shard,
    fused_partitions: Option<usize>,
) -> CandleResult<crate::Shard> {
    let [partitions] = checkpoint_shape else {
        return Ok(Default::default());
    };
    if *partitions <= 1 {
        return Ok(Default::default());
    }
    let expected = fused_partitions.ok_or_else(|| {
        candle_core::Error::msg(format!(
            "partitioned FP8 tensor scale {checkpoint_shape:?} requires a recognized fused projection"
        ))
    })?;
    if *partitions != expected {
        candle_core::bail!(
            "fused FP8 tensor scale has {partitions} partitions, expected {expected}"
        )
    }
    let partition = match weight_shard {
        crate::Shard::Simple {
            dim: 0,
            rank,
            world_size,
        } => {
            if world_size == 0 || rank >= world_size {
                candle_core::bail!(
                    "invalid fused FP8 weight shard rank {rank} for world size {world_size}"
                )
            }
            if world_size % expected != 0 {
                candle_core::bail!(
                    "fused FP8 tensor scale has {partitions} partitions but weight shard world size is {world_size}"
                )
            }
            rank / (world_size / expected)
        }
        crate::Shard::Offset {
            dim: 0,
            offset,
            len,
        } => {
            let rows = weight_shape[0] / *partitions;
            if rows == 0
                || !weight_shape[0].is_multiple_of(*partitions)
                || len != rows
                || !offset.is_multiple_of(rows)
            {
                candle_core::bail!(
                    "fused FP8 tensor scale partitions do not align with weight shard offset={offset} len={len}"
                )
            }
            offset / rows
        }
        _ => {
            candle_core::bail!("fused FP8 tensor scales require an output-sharded projection")
        }
    };
    if partition >= *partitions {
        candle_core::bail!(
            "fused FP8 tensor scale partition {partition} is outside {partitions} entries"
        )
    }
    Ok(crate::Shard::Offset {
        dim: 0,
        offset: partition,
        len: 1,
    })
}

fn shard_dim(shard: crate::Shard) -> usize {
    match shard {
        crate::Shard::Simple { dim, .. } | crate::Shard::Offset { dim, .. } => dim,
    }
}

fn remap_shard(shard: crate::Shard, dim: usize) -> CandleResult<crate::Shard> {
    match shard {
        crate::Shard::Simple {
            rank, world_size, ..
        } => Ok(crate::Shard::Simple {
            dim,
            rank,
            world_size,
        }),
        crate::Shard::Offset { offset, len, .. } => Ok(crate::Shard::Offset { dim, offset, len }),
    }
}

fn compressed_tensors_spec(
    group: &serde_json::Map<String, Value>,
    name: &str,
    format: Option<&str>,
) -> Result<CheckpointRuleResolution, String> {
    let weights = group
        .get("weights")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("compressed-tensors config group `{name}` requires `weights`"))?;
    if weights.get("type").and_then(Value::as_str) == Some("float")
        && weights.get("num_bits").and_then(Value::as_u64) == Some(NVFP4_BITS)
        && format == Some(NVFP4_COMPRESSED_FORMAT)
    {
        return compressed_tensors_nvfp4_spec(group, weights, name)
            .map(CheckpointLinearSpec::Nvfp4)
            .map(CheckpointRuleResolution::Quantized);
    }
    if weights.get("type").and_then(Value::as_str) != Some("float")
        || weights.get("num_bits").and_then(Value::as_u64) != Some(FP8_BITS)
    {
        let weight_type = weights
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unspecified");
        let bits = weights
            .get("num_bits")
            .and_then(Value::as_u64)
            .map_or_else(|| "unspecified".to_string(), |bits| bits.to_string());
        return Ok(CheckpointRuleResolution::Unsupported(format!(
            "compressed-tensors group `{name}` uses {bits}-bit `{weight_type}` weights"
        )));
    }
    if group
        .get("output_activations")
        .is_some_and(|value| !value.is_null())
    {
        return Err(format!(
            "compressed-tensors FP8 group `{name}` uses unsupported output activation quantization"
        ));
    }
    if weights.get("dynamic").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "compressed-tensors FP8 group `{name}` requires static weights"
        ));
    }
    if weights.get("symmetric").and_then(Value::as_bool) == Some(false) {
        return Err(format!(
            "compressed-tensors FP8 group `{name}` requires symmetric weights"
        ));
    }
    let weight_scale = match weights.get("strategy").and_then(Value::as_str) {
        Some("tensor") | None => Fp8WeightScaleLayout::Tensor,
        Some("channel") => Fp8WeightScaleLayout::Channel,
        Some("block") => {
            let size = weights
                .get("block_structure")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!(
                        "compressed-tensors FP8 block group `{name}` requires `block_structure`"
                    )
                })?;
            Fp8WeightScaleLayout::Block(block_size_values(size, "block_structure")?)
        }
        Some(strategy) => {
            return Err(format!(
                "unsupported compressed-tensors FP8 weight strategy `{strategy}` in group `{name}`"
            ))
        }
    };

    let activation = match group.get("input_activations") {
        None | Some(Value::Null) => Fp8ActivationMode::None,
        Some(value) => {
            let input = value.as_object().ok_or_else(|| {
                format!("compressed-tensors input activations in group `{name}` must be an object")
            })?;
            if input.get("type").and_then(Value::as_str) != Some("float")
                || input.get("num_bits").and_then(Value::as_u64) != Some(FP8_BITS)
            {
                return Err(format!(
                    "compressed-tensors FP8 group `{name}` requires 8-bit float activations"
                ));
            }
            if input.get("symmetric").and_then(Value::as_bool) == Some(false) {
                return Err(format!(
                    "compressed-tensors FP8 group `{name}` requires symmetric activations"
                ));
            }
            match (
                input.get("dynamic").and_then(Value::as_bool),
                input.get("strategy").and_then(Value::as_str),
            ) {
                (Some(false), Some("tensor") | None) => Fp8ActivationMode::StaticTensor,
                (Some(true), Some("token") | None) => Fp8ActivationMode::DynamicToken,
                (Some(true), Some("group")) => {
                    let size = input
                        .get("group_size")
                        .and_then(Value::as_u64)
                        .and_then(|size| usize::try_from(size).ok())
                        .filter(|size| *size > 0)
                        .ok_or_else(|| {
                            format!(
                                "compressed-tensors FP8 group `{name}` requires a positive activation group size"
                            )
                        })?;
                    Fp8ActivationMode::DynamicBlock(size)
                }
                (dynamic, strategy) => {
                    return Err(format!(
                        "unsupported compressed-tensors FP8 activation scheme in group `{name}`: dynamic={dynamic:?}, strategy={strategy:?}"
                    ))
                }
            }
        }
    };
    if activation == Fp8ActivationMode::DynamicToken {
        if let Fp8WeightScaleLayout::Block([_, cols]) = weight_scale {
            return Err(format!(
                "compressed-tensors FP8 block group `{name}` requires blockwise activations of size {cols}"
            ));
        }
    }
    if let (
        Fp8WeightScaleLayout::Block([_, cols]),
        Fp8ActivationMode::DynamicBlock(activation_cols),
    ) = (weight_scale, activation)
    {
        if cols != activation_cols {
            return Err(format!(
                "compressed-tensors FP8 group `{name}` has weight block width {cols} but activation block width {activation_cols}"
            ));
        }
    }
    if matches!(activation, Fp8ActivationMode::DynamicBlock(_))
        && !matches!(weight_scale, Fp8WeightScaleLayout::Block(_))
    {
        return Err(format!(
            "compressed-tensors FP8 group `{name}` requires blockwise weights for blockwise activations"
        ));
    }
    Ok(CheckpointRuleResolution::Quantized(
        CheckpointLinearSpec::Fp8(Fp8LinearSpec::new(
            CheckpointDialect::CompressedTensors,
            weight_scale,
            activation,
        )),
    ))
}

fn compressed_tensors_nvfp4_spec(
    group: &serde_json::Map<String, Value>,
    weights: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<Nvfp4LinearSpec, String> {
    validate_nvfp4_args(weights, name, "weight")?;
    if weights.get("dynamic").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "compressed-tensors NVFP4 group `{name}` requires static weights"
        ));
    }
    if group
        .get("output_activations")
        .is_some_and(|value| !value.is_null())
    {
        return Err(format!(
            "compressed-tensors NVFP4 group `{name}` uses unsupported output activation quantization"
        ));
    }
    let activation = match group.get("input_activations") {
        None | Some(Value::Null) => Nvfp4ActivationMode::None,
        Some(value) => {
            let input = value.as_object().ok_or_else(|| {
                format!("compressed-tensors input activations in group `{name}` must be an object")
            })?;
            validate_nvfp4_args(input, name, "activation")?;
            if input.get("dynamic").and_then(Value::as_str) != Some("local") {
                return Err(format!(
                    "compressed-tensors NVFP4 group `{name}` requires dynamic=local activations with a calibrated global scale"
                ));
            }
            Nvfp4ActivationMode::DynamicBlock
        }
    };
    Ok(Nvfp4LinearSpec::compressed_tensors(activation))
}

fn validate_nvfp4_args(
    args: &serde_json::Map<String, Value>,
    name: &str,
    kind: &str,
) -> Result<(), String> {
    if args.get("type").and_then(Value::as_str) != Some("float")
        || args.get("num_bits").and_then(Value::as_u64) != Some(NVFP4_BITS)
        || args.get("strategy").and_then(Value::as_str) != Some("tensor_group")
        || args.get("group_size").and_then(Value::as_u64) != Some(NVFP4_BLOCK_SIZE as u64)
    {
        return Err(format!(
            "compressed-tensors NVFP4 group `{name}` requires 4-bit float {kind}s with tensor_group strategy and group_size={NVFP4_BLOCK_SIZE}"
        ));
    }
    if args.get("symmetric").and_then(Value::as_bool) == Some(false) {
        return Err(format!(
            "compressed-tensors NVFP4 group `{name}` requires symmetric {kind}s"
        ));
    }
    if let Some(dtype) = args.get("scale_dtype").filter(|value| !value.is_null()) {
        let dtype = dtype
            .as_str()
            .ok_or_else(|| format!("NVFP4 {kind} scale_dtype must be a string"))?;
        validate_e4m3(Some(dtype)).map_err(|_| {
            format!(
            "compressed-tensors NVFP4 group `{name}` requires E4M3FN {kind} scales, got `{dtype}`"
        )
        })?;
    }
    if args.get("actorder").is_some_and(|value| !value.is_null()) {
        return Err(format!(
            "compressed-tensors NVFP4 group `{name}` does not support activation ordering"
        ));
    }
    Ok(())
}

fn model_opt_spec(
    algorithm: &str,
    group_size: Option<&Value>,
    weight_scale_encoding: Option<&Value>,
) -> Result<CheckpointLinearSpec, String> {
    let encoding = match weight_scale_encoding {
        None | Some(Value::Null) => Nvfp4ScaleEncoding::Plain,
        Some(Value::String(value)) if value == "csf" => Nvfp4ScaleEncoding::Csf,
        Some(Value::String(value)) if value == "plain" => Nvfp4ScaleEncoding::Plain,
        Some(value) => {
            return Err(format!(
                "unsupported ModelOpt weight_scale_encoding `{value}`"
            ))
        }
    };
    if encoding != Nvfp4ScaleEncoding::Plain && !matches!(algorithm, "NVFP4" | "W4A16_NVFP4") {
        return Err(format!(
            "ModelOpt {algorithm} does not support CSF weight scales"
        ));
    }
    if algorithm == "MXFP8" {
        if group_size
            .is_some_and(|value| value.as_u64() != Some(crate::mxfp8::MXFP8_BLOCK_SIZE as u64))
        {
            return Err(format!(
                "ModelOpt MXFP8 requires group_size={}",
                crate::mxfp8::MXFP8_BLOCK_SIZE
            ));
        }
        return Ok(CheckpointLinearSpec::Mxfp8(Mxfp8LinearSpec {
            group_size: crate::mxfp8::MXFP8_BLOCK_SIZE,
        }));
    }
    if matches!(algorithm, "NVFP4" | "W4A16_NVFP4") {
        if group_size.is_some_and(|value| value.as_u64() != Some(NVFP4_BLOCK_SIZE as u64)) {
            return Err(format!(
                "ModelOpt {algorithm} requires group_size={NVFP4_BLOCK_SIZE}"
            ));
        }
        let activation = if algorithm == "NVFP4" {
            Nvfp4ActivationMode::DynamicBlock
        } else {
            Nvfp4ActivationMode::None
        };
        let mut spec = Nvfp4LinearSpec::model_opt(activation);
        spec.weight_scale_encoding = encoding;
        return Ok(CheckpointLinearSpec::Nvfp4(spec));
    }
    let (weight_scale, activation) = match algorithm {
        "FP8" => (
            Fp8WeightScaleLayout::Tensor,
            Fp8ActivationMode::StaticTensor,
        ),
        "FP8_PER_CHANNEL_PER_TOKEN" => (
            Fp8WeightScaleLayout::Channel,
            Fp8ActivationMode::DynamicToken,
        ),
        "FP8_PB_WO" => (
            Fp8WeightScaleLayout::Block([MODEL_OPT_BLOCK_SIZE; 2]),
            Fp8ActivationMode::None,
        ),
        other => return Err(format!("unsupported ModelOpt mixed algorithm `{other}`")),
    };
    Ok(CheckpointLinearSpec::Fp8(Fp8LinearSpec::new(
        CheckpointDialect::ModelOpt,
        weight_scale,
        activation,
    )))
}

fn model_opt_fields(value: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let root = value
        .as_object()
        .ok_or_else(|| "ModelOpt quantization config must be an object".to_string())?;
    let mut fields = root
        .get("quantization")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for name in [
        "quant_algo",
        "kv_cache_quant_algo",
        "kv_cache_scheme",
        "group_size",
        "weight_scale_encoding",
        "quantized_layers",
    ] {
        if let Some(value) = root.get(name) {
            fields.insert(name.to_string(), value.clone());
        }
    }
    let ignore = root
        .get("ignore")
        .or_else(|| root.get("exclude_modules"))
        .cloned()
        .or_else(|| fields.get("ignore").cloned())
        .or_else(|| fields.get("exclude_modules").cloned());
    fields.remove("exclude_modules");
    if let Some(ignore) = ignore {
        fields.insert("ignore".to_string(), ignore);
    }
    Ok(fields)
}

fn model_opt_extra_fields(value: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let mut fields = value
        .as_object()
        .cloned()
        .ok_or_else(|| "ModelOpt quantization config must be an object".to_string())?;
    for name in [
        "quantization",
        "quant_method",
        "quant_algo",
        "kv_cache_quant_algo",
        "kv_cache_scheme",
        "group_size",
        "quantized_layers",
        "ignore",
        "exclude_modules",
    ] {
        fields.remove(name);
    }
    Ok(fields)
}

fn resolve_scale_name(
    vb: &crate::ShardedVarBuilder,
    candidates: &'static [&'static str],
    kind: &str,
) -> CandleResult<Option<&'static str>> {
    let matches = candidates
        .iter()
        .copied()
        .filter(|name| vb.contains_tensor(name))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [name] => Ok(Some(*name)),
        _ => candle_core::bail!(
            "ambiguous FP8 {kind} scales at prefix `{}`: {}",
            vb.prefix(),
            matches.join(", ")
        ),
    }
}

fn prefix_candidates(prefix: &str) -> Vec<String> {
    let mut candidates = vec![prefix.to_string()];
    if let Some(suffix) = prefix.strip_prefix("model.language_model.") {
        candidates.push(format!("model.{suffix}"));
        candidates.push(format!("language_model.model.{suffix}"));
    } else if let Some(suffix) = prefix.strip_prefix("language_model.model.") {
        candidates.push(format!("model.{suffix}"));
        candidates.push(format!("model.language_model.{suffix}"));
    } else if let Some(suffix) = prefix.strip_prefix("model.") {
        candidates.push(format!("model.language_model.{suffix}"));
        candidates.push(format!("language_model.model.{suffix}"));
    }
    if prefix.ends_with(".lm_head") {
        candidates.push("lm_head".to_string());
    }
    candidates
}

fn fused_constituents(prefix: &str) -> Option<Vec<String>> {
    for (fused, constituents) in [
        ("gate_up_proj", &["gate_proj", "up_proj"][..]),
        ("qkv_proj", &["q_proj", "k_proj", "v_proj"][..]),
    ] {
        if let Some(base) = prefix.strip_suffix(fused) {
            return Some(
                constituents
                    .iter()
                    .map(|constituent| format!("{base}{constituent}"))
                    .collect(),
            );
        }
    }
    None
}

pub(crate) fn validate_e4m3(format: Option<&str>) -> Result<(), String> {
    let Some(format) = format else {
        return Ok(());
    };
    match format.to_ascii_lowercase().as_str() {
        "e4m3" | "e4m3fn" | "float8_e4m3fn" | "torch.float8_e4m3fn" => Ok(()),
        _ => Err(format!(
            "unsupported native FP8 format `{format}`; expected E4M3FN"
        )),
    }
}

fn natural_group_cmp(left: &str, right: &str) -> Ordering {
    match (numeric_suffix(left), numeric_suffix(right)) {
        (Some((left_prefix, left_number)), Some((right_prefix, right_number)))
            if left_prefix == right_prefix =>
        {
            left_number.cmp(&right_number).then_with(|| left.cmp(right))
        }
        _ => left.cmp(right),
    }
}

fn numeric_suffix(value: &str) -> Option<(&str, u64)> {
    let prefix = value.trim_end_matches(|ch: char| ch.is_ascii_digit());
    if prefix.len() == value.len() {
        return None;
    }
    value[prefix.len()..]
        .parse()
        .ok()
        .map(|number| (prefix, number))
}

fn block_size(values: &[usize], field: &str) -> Result<[usize; 2], String> {
    let [rows, cols]: [usize; 2] = values
        .try_into()
        .map_err(|_| format!("`{field}` must contain two dimensions"))?;
    if rows == 0 || cols == 0 {
        return Err(format!("`{field}` dimensions must be positive"));
    }
    Ok([rows, cols])
}

fn block_size_values(values: &[Value], field: &str) -> Result<[usize; 2], String> {
    let parsed = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| format!("`{field}` dimensions must be positive integers"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    block_size(&parsed, field)
}

fn string_array(value: Option<&Value>, field: &str) -> Result<Vec<String>, String> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("`{field}` must be an array of strings"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("`{field}` must be an array of strings"))
        })
        .collect()
}

fn optional_string_array(value: Option<&Value>, field: &str) -> Result<Vec<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(value) => string_array(Some(value), field),
    }
}

fn glob_regex(glob: &str) -> Result<CheckpointPattern, String> {
    let mut regex = String::from("^");
    let mut chars = glob.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '[' => {
                regex.push('[');
                if chars.peek() == Some(&'!') {
                    chars.next();
                    regex.push('^');
                }
                let mut closed = false;
                for ch in chars.by_ref() {
                    regex.push(ch);
                    if ch == ']' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(format!(
                        "invalid ModelOpt glob `{glob}`: unclosed character class"
                    ));
                }
            }
            other => regex.push_str(&regex::escape(&other.to_string())),
        }
    }
    regex.push('$');
    let compiled =
        Regex::new(&regex).map_err(|err| format!("invalid ModelOpt glob `{glob}`: {err}"))?;
    Ok(CheckpointPattern(Arc::new(compiled)))
}

#[cfg(test)]
#[path = "fp8_config_tests.rs"]
mod tests;
