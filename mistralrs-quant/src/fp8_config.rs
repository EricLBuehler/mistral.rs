use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::{cmp::Ordering, sync::Arc};

use crate::QuantMethod;
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Linear;

const FP8_BITS: u64 = 8;
const MODEL_OPT_BLOCK_SIZE: usize = 128;
const FP8_WEIGHT_SCALE_ALIASES: &[&str] = &["weight_scale", "weight_scale_inv"];
const MODEL_OPT_LEGACY_VISION_EXCLUSIONS: &[&str] =
    &["vision_tower", "vision_model", "vit_large_projector"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fp8CheckpointDialect {
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
    pub dialect: Fp8CheckpointDialect,
    pub weight_scale: Fp8WeightScaleLayout,
    pub activation: Fp8ActivationMode,
    pub scale_names: Fp8ScaleNames,
}

impl Fp8LinearSpec {
    fn new(
        dialect: Fp8CheckpointDialect,
        weight_scale: Fp8WeightScaleLayout,
        activation: Fp8ActivationMode,
    ) -> Self {
        let weight: &'static [&'static str] = match (dialect, weight_scale) {
            (Fp8CheckpointDialect::Native, Fp8WeightScaleLayout::Block(_)) => {
                &["weight_scale_inv", "weight_scale"]
            }
            (Fp8CheckpointDialect::Native, _)
            | (Fp8CheckpointDialect::CompressedTensors, _)
            | (Fp8CheckpointDialect::ModelOpt, _) => &["weight_scale", "weight_scale_inv"],
        };
        let scale_activation: &'static [&'static str] = match activation {
            Fp8ActivationMode::StaticTensor => match dialect {
                Fp8CheckpointDialect::Native => &["input_scale", "activation_scale"],
                Fp8CheckpointDialect::CompressedTensors | Fp8CheckpointDialect::ModelOpt => {
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Fp8Target {
    Linear,
    Exact(String),
    Contains(String),
    Regex(String),
    Glob(String),
}

impl Fp8Target {
    fn compressed_tensors(raw: &str) -> Result<Self, String> {
        if raw == "Linear" {
            return Ok(Self::Linear);
        }
        if let Some(pattern) = raw.strip_prefix("re:") {
            Regex::new(pattern)
                .map_err(|err| format!("invalid compressed-tensors target regex `{raw}`: {err}"))?;
            return Ok(Self::Regex(pattern.to_string()));
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
            Self::Contains(target) => prefix.contains(target),
            Self::Regex(pattern) | Self::Glob(pattern) => Regex::new(pattern)
                .expect("FP8 target regex is validated during config parsing")
                .is_match(prefix),
        }
    }

    fn priority(&self) -> usize {
        match self {
            Self::Exact(_) => 0,
            Self::Contains(_) | Self::Regex(_) | Self::Glob(_) => 1,
            Self::Linear => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Fp8ConfigRule {
    targets: Vec<Fp8Target>,
    resolution: Fp8RuleResolution,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
enum Fp8RuleResolution {
    Fp8(Fp8LinearSpec),
    Unquantized,
    Unsupported(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fp8DirectResolution {
    NoMatch,
    Resolved(Option<Fp8LinearSpec>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Fp8Config {
    dialect: Fp8CheckpointDialect,
    rules: Vec<Fp8ConfigRule>,
    ignore: Vec<Fp8Target>,
}

impl Fp8Config {
    pub fn dialect(&self) -> Fp8CheckpointDialect {
        self.dialect
    }

    pub fn resolve(&self, prefix: &str) -> Option<Fp8LinearSpec> {
        self.resolve_checked(prefix).ok().flatten()
    }

    pub(crate) fn resolve_checked(&self, prefix: &str) -> Result<Option<Fp8LinearSpec>, String> {
        match self.resolve_state_checked(prefix)? {
            Fp8DirectResolution::NoMatch => Ok(None),
            Fp8DirectResolution::Resolved(spec) => Ok(spec),
        }
    }

    fn resolve_state_checked(&self, prefix: &str) -> Result<Fp8DirectResolution, String> {
        let prefixes = prefix_candidates(prefix);
        if self
            .ignore
            .iter()
            .any(|target| prefixes.iter().any(|prefix| target.matches(prefix)))
        {
            return Ok(Fp8DirectResolution::Resolved(None));
        }
        let exact = self.resolve_rules_at_priority(prefix, &prefixes, 0)?;
        if exact != Fp8DirectResolution::NoMatch {
            return Ok(exact);
        }
        if let Some(constituents) = fused_constituents(prefix) {
            let fused = self.resolve_fused_state(&constituents)?;
            if fused != Fp8DirectResolution::NoMatch {
                return Ok(fused);
            }
        }
        for priority in 1..3 {
            let resolved = self.resolve_rules_at_priority(prefix, &prefixes, priority)?;
            if resolved != Fp8DirectResolution::NoMatch {
                return Ok(resolved);
            }
        }
        Ok(Fp8DirectResolution::NoMatch)
    }

    fn resolve_rules_at_priority(
        &self,
        prefix: &str,
        prefixes: &[String],
        priority: usize,
    ) -> Result<Fp8DirectResolution, String> {
        for rule in &self.rules {
            if rule.targets.iter().any(|target| {
                target.priority() == priority
                    && prefixes.iter().any(|prefix| target.matches(prefix))
            }) {
                return match &rule.resolution {
                    Fp8RuleResolution::Fp8(spec) => Ok(Fp8DirectResolution::Resolved(Some(*spec))),
                    Fp8RuleResolution::Unquantized => Ok(Fp8DirectResolution::Resolved(None)),
                    Fp8RuleResolution::Unsupported(reason) => Err(format!(
                        "unsupported quantization scheme for `{prefix}`: {reason}"
                    )),
                };
            }
        }
        Ok(Fp8DirectResolution::NoMatch)
    }

    pub fn resolve_fused<S: AsRef<str>>(
        &self,
        prefixes: &[S],
    ) -> Result<Option<Fp8LinearSpec>, String> {
        match self.resolve_fused_state(prefixes)? {
            Fp8DirectResolution::NoMatch => Ok(None),
            Fp8DirectResolution::Resolved(spec) => Ok(spec),
        }
    }

    fn resolve_fused_state<S: AsRef<str>>(
        &self,
        prefixes: &[S],
    ) -> Result<Fp8DirectResolution, String> {
        let Some((first, rest)) = prefixes.split_first() else {
            return Ok(Fp8DirectResolution::NoMatch);
        };
        let resolved = self.resolve_state_checked(first.as_ref())?;
        for prefix in rest {
            let next = self.resolve_state_checked(prefix.as_ref())?;
            if next != resolved {
                return Err(format!(
                    "fused FP8 projection has different schemes for `{}` and `{}`",
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
            dialect: Fp8CheckpointDialect::Native,
            rules: vec![Fp8ConfigRule {
                targets: vec![Fp8Target::Linear],
                resolution: Fp8RuleResolution::Fp8(Fp8LinearSpec::new(
                    Fp8CheckpointDialect::Native,
                    weight_scale,
                    activation,
                )),
            }],
            ignore: modules_to_not_convert
                .iter()
                .map(|target| Fp8Target::Exact(target.clone()))
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
        let mut has_fp8 = false;
        let mut groups = groups.iter().collect::<Vec<_>>();
        groups.sort_by(|(left, _), (right, _)| natural_group_cmp(left, right));
        for (name, group) in groups {
            let group = group.as_object().ok_or_else(|| {
                format!("compressed-tensors config group `{name}` must be an object")
            })?;
            let resolution = compressed_tensors_spec(group, name)?;
            has_fp8 |= matches!(&resolution, Fp8RuleResolution::Fp8(_));
            let targets = string_array(
                group.get("targets"),
                &format!("config_groups.{name}.targets"),
            )?
            .into_iter()
            .map(|target| Fp8Target::compressed_tensors(&target))
            .collect::<Result<Vec<_>, _>>()?;
            if targets.is_empty() {
                return Err(format!(
                    "compressed-tensors config group `{name}` requires at least one target"
                ));
            }
            rules.push(Fp8ConfigRule {
                targets,
                resolution,
            });
        }
        if !has_fp8 {
            return Err("compressed-tensors config contains no supported FP8 groups".to_string());
        }
        let ignore = optional_string_array(object.get("ignore"), "ignore")?
            .into_iter()
            .map(|target| Fp8Target::compressed_tensors(&target))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            dialect: Fp8CheckpointDialect::CompressedTensors,
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
            .map(|target| Fp8Target::model_opt(&target))
            .collect::<Result<Vec<_>, _>>()?;
        ignore.extend(
            MODEL_OPT_LEGACY_VISION_EXCLUSIONS
                .iter()
                .map(|target| Fp8Target::Contains((*target).to_string())),
        );

        let rules = if algorithm == "MIXED_PRECISION" {
            let layers = config
                .get("quantized_layers")
                .or_else(|| root.get("quantized_layers"))
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    "ModelOpt MIXED_PRECISION config requires `quantized_layers`".to_string()
                })?;
            let mut rules = Vec::new();
            let mut has_fp8 = false;
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
                match algorithm.as_str() {
                    "NONE" | "UNQUANTIZED" => rules.push(Fp8ConfigRule {
                        targets: vec![Fp8Target::Exact(prefix.clone())],
                        resolution: Fp8RuleResolution::Unquantized,
                    }),
                    _ => {
                        has_fp8 = true;
                        rules.push(Fp8ConfigRule {
                            targets: vec![Fp8Target::Exact(prefix.clone())],
                            resolution: Fp8RuleResolution::Fp8(model_opt_spec(&algorithm)?),
                        });
                    }
                }
            }
            if !has_fp8 {
                return Err(
                    "ModelOpt MIXED_PRECISION config contains no supported FP8 layers".to_string(),
                );
            }
            rules
        } else {
            let spec = model_opt_spec(&algorithm)?;
            vec![Fp8ConfigRule {
                targets: vec![Fp8Target::Linear],
                resolution: Fp8RuleResolution::Fp8(spec),
            }]
        };
        Ok(Self {
            dialect: Fp8CheckpointDialect::ModelOpt,
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

pub(crate) fn fp8_checkpoint_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &crate::QuantizedConfig,
    bias: bool,
    hints: crate::Shard,
    vb: crate::ShardedVarBuilder,
) -> CandleResult<Arc<dyn crate::QuantMethod>> {
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
    let scale_shape = match vb.tensor_shape(scale_name) {
        Some(shape) => shape.to_vec(),
        None => spec.weight_scale.logical_shape([out_dim, in_dim])?,
    };
    let fused_partitions = fused_constituents(&vb.prefix()).map(|parts| parts.len());
    let scale_hints = scale_shard(
        spec.weight_scale,
        [out_dim, in_dim],
        &scale_shape,
        hints,
        fused_partitions,
    )?;
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale = vb.get_with_hints_dtype(scale_shape, scale_name, scale_hints, DType::F32)?;
    let weight_shape: [usize; 2] = weight.dims2().map(|(rows, cols)| [rows, cols])?;
    let weight_scale = spec.weight_scale.normalize(weight_scale, weight_shape)?;
    let activation_scale = if spec.activation == Fp8ActivationMode::StaticTensor {
        let name = spec.scale_names.activation_name(&vb)?.ok_or_else(|| {
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
        let scale_hints = tensor_scale_shard([out_dim, in_dim], &shape, hints, fused_partitions)?;
        Some(spec.normalize_activation_scale(vb.get_with_hints_dtype(
            shape,
            name,
            scale_hints,
            DType::F32,
        )?)?)
    } else {
        None
    };
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };
    let dequant_dtype = bias
        .as_ref()
        .map(Tensor::dtype)
        .filter(|dtype| matches!(dtype, DType::F16 | DType::BF16 | DType::F32))
        .unwrap_or_else(|| match vb.dtype() {
            DType::F16 | DType::BF16 | DType::F32 => vb.dtype(),
            _ => DType::BF16,
        });

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
) -> Result<Fp8RuleResolution, String> {
    let weights = group
        .get("weights")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("compressed-tensors config group `{name}` requires `weights`"))?;
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
        return Ok(Fp8RuleResolution::Unsupported(format!(
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
    Ok(Fp8RuleResolution::Fp8(Fp8LinearSpec::new(
        Fp8CheckpointDialect::CompressedTensors,
        weight_scale,
        activation,
    )))
}

fn model_opt_spec(algorithm: &str) -> Result<Fp8LinearSpec, String> {
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
    Ok(Fp8LinearSpec::new(
        Fp8CheckpointDialect::ModelOpt,
        weight_scale,
        activation,
    ))
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

fn glob_regex(glob: &str) -> Result<String, String> {
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
    Regex::new(&regex).map_err(|err| format!("invalid ModelOpt glob `{glob}`: {err}"))?;
    Ok(regex)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use serde_json::json;

    fn ct_args(strategy: &str) -> Value {
        json!({
            "dynamic": false,
            "num_bits": 8,
            "strategy": strategy,
            "symmetric": true,
            "type": "float"
        })
    }

    #[test]
    fn compressed_tensors_resolves_channel_dynamic_and_ignore() {
        let config = Fp8Config::compressed_tensors(&json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("channel"),
                    "input_activations": {
                        "dynamic": true,
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            },
            "ignore": ["lm_head"]
        }))
        .unwrap();

        let spec = config.resolve("model.layers.0.self_attn.q_proj").unwrap();
        assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Channel);
        assert_eq!(spec.activation, Fp8ActivationMode::DynamicToken);
        assert_eq!(spec.scale_names.weight[0], "weight_scale");
        assert_eq!(config.resolve("lm_head"), None);
    }

    #[test]
    fn compressed_tensors_resolves_block_and_regex_targets() {
        let config = Fp8Config::compressed_tensors(&json!({
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "format": "float-quantized",
                    "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
                    "weights": {
                        "block_structure": [128, 128],
                        "dynamic": false,
                        "num_bits": 8,
                        "strategy": "block",
                        "symmetric": true,
                        "type": "float"
                    },
                    "input_activations": {
                        "dynamic": true,
                        "group_size": 128,
                        "num_bits": 8,
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float"
                    }
                },
                "group_1": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float"}
                }
            }
        }))
        .unwrap();

        let spec = config.resolve("model.layers.2.self_attn.q_proj").unwrap();
        assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Block([128, 128]));
        assert_eq!(spec.activation, Fp8ActivationMode::DynamicBlock(128));
        assert_eq!(config.resolve("model.layers.2.mlp.gate_proj"), None);
    }

    #[test]
    fn compressed_tensors_resolves_w8a16() {
        let config = Fp8Config::compressed_tensors(&json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("tensor"),
                    "input_activations": null
                }
            }
        }))
        .unwrap();
        let spec = config.resolve("model.layers.0.mlp.down_proj").unwrap();
        assert_eq!(spec.activation, Fp8ActivationMode::None);
        assert_eq!(spec.weight_scale, Fp8WeightScaleLayout::Tensor);
    }

    #[test]
    fn compressed_tensors_exact_target_precedes_linear_class() {
        let config = Fp8Config::compressed_tensors(&json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("tensor")
                },
                "group_1": {
                    "targets": ["model.layers.0.self_attn.q_proj"],
                    "weights": ct_args("channel")
                }
            }
        }))
        .unwrap();
        assert_eq!(
            config
                .resolve("model.layers.0.self_attn.q_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Channel
        );
    }

    #[test]
    fn compressed_tensors_non_fp8_target_shadows_linear_fp8() {
        let config = Fp8Config::compressed_tensors(&json!({
            "format": "mixed-precision",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("channel")
                },
                "group_1": {
                    "targets": ["model.layers.0.mlp.down_proj"],
                    "output_activations": {
                        "dynamic": false,
                        "num_bits": 8,
                        "type": "int"
                    },
                    "weights": {
                        "dynamic": false,
                        "num_bits": 4,
                        "strategy": "group",
                        "symmetric": true,
                        "type": "int"
                    }
                }
            }
        }))
        .unwrap();
        assert_eq!(config.resolve("model.layers.0.mlp.down_proj"), None);
        assert!(config
            .resolve_checked("model.layers.0.mlp.down_proj")
            .unwrap_err()
            .contains("4-bit `int` weights"));
        assert!(config.resolve("model.layers.0.mlp.up_proj").is_some());
    }

    #[test]
    fn compressed_tensors_uses_natural_group_order() {
        let config = Fp8Config::compressed_tensors(&json!({
            "config_groups": {
                "group_10": {
                    "targets": ["re:.*q_proj$"],
                    "weights": ct_args("channel")
                },
                "group_2": {
                    "targets": ["re:.*q_proj$"],
                    "weights": ct_args("tensor")
                }
            }
        }))
        .unwrap();
        assert_eq!(
            config
                .resolve("model.layers.0.self_attn.q_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Tensor
        );
    }

    #[test]
    fn compressed_tensors_fused_projection_propagates_unsupported_constituent() {
        let config = Fp8Config::compressed_tensors(&json!({
            "config_groups": {
                "group_0": {
                    "targets": ["model.layers.0.mlp.gate_proj"],
                    "weights": ct_args("tensor")
                },
                "group_1": {
                    "targets": ["model.layers.0.mlp.up_proj"],
                    "weights": {
                        "dynamic": false,
                        "num_bits": 4,
                        "strategy": "tensor_group",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            }
        }))
        .unwrap();
        let error = config
            .resolve_checked("model.layers.0.mlp.gate_up_proj")
            .unwrap_err();
        assert!(error.contains("4-bit `float` weights"));
    }

    #[test]
    fn compressed_tensors_rejects_output_activation_quantization() {
        let error = Fp8Config::compressed_tensors(&json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("tensor"),
                    "output_activations": ct_args("tensor")
                }
            }
        }))
        .unwrap_err();
        assert!(error.contains("output activation"));
    }

    #[test]
    fn compressed_tensors_rejects_asymmetric_fp8_activations() {
        let error = Fp8Config::compressed_tensors(&json!({
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("tensor"),
                    "input_activations": {
                        "dynamic": true,
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": false,
                        "type": "float"
                    }
                }
            }
        }))
        .unwrap_err();
        assert!(error.contains("symmetric activations"));
    }

    #[test]
    fn compressed_tensors_rejects_block_activations_with_non_block_weights() {
        let error = Fp8Config::compressed_tensors(&json!({
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("channel"),
                    "input_activations": {
                        "dynamic": true,
                        "group_size": 128,
                        "num_bits": 8,
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            }
        }))
        .unwrap_err();
        assert!(error.contains("blockwise weights for blockwise activations"));
    }

    #[test]
    fn model_opt_resolves_all_fp8_algorithms() {
        for (algorithm, weight, activation) in [
            (
                "FP8",
                Fp8WeightScaleLayout::Tensor,
                Fp8ActivationMode::StaticTensor,
            ),
            (
                "FP8_PER_CHANNEL_PER_TOKEN",
                Fp8WeightScaleLayout::Channel,
                Fp8ActivationMode::DynamicToken,
            ),
            (
                "fp8_pb_wo",
                Fp8WeightScaleLayout::Block([128, 128]),
                Fp8ActivationMode::None,
            ),
        ] {
            let config = Fp8Config::model_opt(&json!({
                "quant_method": "modelopt",
                "quant_algo": algorithm
            }))
            .unwrap();
            let spec = config.resolve("model.layers.0.mlp.gate_proj").unwrap();
            assert_eq!(spec.weight_scale, weight);
            assert_eq!(spec.activation, activation);
            assert_eq!(spec.scale_names.weight[0], "weight_scale");
        }
    }

    #[test]
    fn model_opt_mixed_precision_resolves_per_layer() {
        let config = Fp8Config::model_opt(&json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "ignore": ["mtp*"],
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                "model.layers.0.self_attn.k_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                },
                "model.layers.0.mlp.gate_proj": {"quant_algo": "UNQUANTIZED"}
            }
        }))
        .unwrap();

        assert_eq!(
            config
                .resolve("model.layers.0.self_attn.q_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Tensor
        );
        assert_eq!(
            config
                .resolve("model.layers.0.self_attn.k_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Channel
        );
        assert_eq!(config.resolve("model.layers.0.mlp.gate_proj"), None);
        assert_eq!(config.resolve("mtp.layers.0.self_attn.q_proj"), None);
    }

    #[test]
    fn model_opt_mixed_precision_resolves_fused_projection_constituents() {
        let config = Fp8Config::model_opt(&json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8"},
                "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
                "model.layers.0.self_attn.q_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                },
                "model.layers.0.self_attn.k_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                },
                "model.layers.0.self_attn.v_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                },
                "model.layers.1.mlp.gate_proj": {"quant_algo": "FP8"},
                "model.layers.1.mlp.up_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                },
                "model.layers.2.mlp.gate_up_proj": {"quant_algo": "FP8_PB_WO"},
                "model.layers.2.mlp.gate_proj": {"quant_algo": "FP8"},
                "model.layers.2.mlp.up_proj": {
                    "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"
                }
            }
        }))
        .unwrap();

        assert_eq!(
            config
                .resolve_checked("model.layers.0.mlp.gate_up_proj")
                .unwrap()
                .unwrap()
                .activation,
            Fp8ActivationMode::StaticTensor
        );
        assert_eq!(
            config
                .resolve_checked("model.layers.0.self_attn.qkv_proj")
                .unwrap()
                .unwrap()
                .activation,
            Fp8ActivationMode::DynamicToken
        );
        assert!(config
            .resolve_checked("model.layers.1.mlp.gate_up_proj")
            .unwrap_err()
            .contains("different schemes"));
        assert_eq!(
            config
                .resolve_checked("model.layers.2.mlp.gate_up_proj")
                .unwrap()
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Block([128, 128])
        );
    }

    #[test]
    fn model_opt_mixed_precision_rejects_unsupported_algorithms() {
        let error = Fp8Config::model_opt(&json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"}
            }
        }))
        .unwrap_err();
        assert!(error.contains("NVFP4"));
    }

    #[test]
    fn model_opt_mixed_precision_requires_an_fp8_layer() {
        let error = Fp8Config::model_opt(&json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "UNQUANTIZED"}
            }
        }))
        .unwrap_err();
        assert!(error.contains("no supported FP8 layers"));
    }

    #[test]
    fn legacy_model_opt_quantization_object_is_supported() {
        let config = Fp8Config::model_opt(&json!({
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
                "exclude_modules": ["lm_head"]
            }
        }))
        .unwrap();
        assert_eq!(config.resolve("lm_head"), None);
        assert_eq!(
            config
                .resolve("model.layers.0.mlp.up_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Channel
        );
        for prefix in [
            "model.vision_tower.encoder.layers.0.mlp.fc1",
            "model.vision_model.encoder.layers.0.mlp.fc1",
            "model.vit_large_projector.linear_1",
        ] {
            assert_eq!(config.resolve(prefix), None);
        }
    }

    #[test]
    fn embedded_model_opt_fields_override_legacy_file() {
        let external = json!({
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "FP8",
                "exclude_modules": ["old_head"]
            }
        });
        let embedded = json!({
            "quant_method": "modelopt",
            "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            "ignore": ["lm_head"]
        });
        let config = Fp8Config::model_opt_merged(Some(&embedded), &external).unwrap();
        assert_eq!(config.resolve("lm_head"), None);
        assert_eq!(
            config
                .resolve("model.layers.0.mlp.up_proj")
                .unwrap()
                .weight_scale,
            Fp8WeightScaleLayout::Channel
        );
    }

    #[test]
    fn scale_layout_normalizes_singleton_checkpoint_dimensions() {
        let device = candle_core::Device::Cpu;
        let tensor = Tensor::zeros(1, DType::F16, &device).unwrap();
        let tensor = Fp8WeightScaleLayout::Tensor
            .normalize(tensor, [896, 896])
            .unwrap();
        assert!(tensor.dims().is_empty());
        assert_eq!(tensor.dtype(), DType::F32);

        let channel = Tensor::zeros((896, 1), DType::BF16, &device).unwrap();
        let channel = Fp8WeightScaleLayout::Channel
            .normalize(channel, [896, 896])
            .unwrap();
        assert_eq!(channel.dims(), [896]);
        assert_eq!(channel.dtype(), DType::F32);

        let blocks = Tensor::zeros((7, 1, 7, 1), DType::F32, &device).unwrap();
        let blocks = Fp8WeightScaleLayout::Block([128, 128])
            .normalize(blocks, [896, 896])
            .unwrap();
        assert_eq!(blocks.dims(), [7, 7]);

        for degenerate in [
            Tensor::zeros((1, 1), DType::F32, &device).unwrap(),
            Tensor::zeros((1, 1, 1, 1), DType::F32, &device).unwrap(),
        ] {
            assert_eq!(
                Fp8WeightScaleLayout::Block([128, 128])
                    .normalize(degenerate, [64, 64])
                    .unwrap()
                    .dims(),
                [1, 1]
            );
        }

        assert!(Fp8WeightScaleLayout::Tensor
            .normalize(Tensor::zeros((1, 1), DType::F32, &device).unwrap(), [1, 1])
            .is_err());
        assert!(Fp8WeightScaleLayout::Channel
            .normalize(
                Tensor::zeros((1, 896, 1), DType::F32, &device).unwrap(),
                [896, 896]
            )
            .is_err());
        assert!(Fp8WeightScaleLayout::Block([128, 128])
            .normalize(
                Tensor::zeros((1, 7, 7, 1), DType::F32, &device).unwrap(),
                [896, 896]
            )
            .is_err());
        assert!(Fp8WeightScaleLayout::Block([0, 128])
            .logical_shape([896, 896])
            .is_err());

        let spec = Fp8LinearSpec::new(
            Fp8CheckpointDialect::ModelOpt,
            Fp8WeightScaleLayout::Tensor,
            Fp8ActivationMode::StaticTensor,
        );
        assert!(spec
            .normalize_activation_scale(Tensor::zeros(1, DType::F16, &device).unwrap())
            .unwrap()
            .dims()
            .is_empty());
        assert!(spec
            .normalize_activation_scale(Tensor::zeros((1, 1), DType::F32, &device).unwrap())
            .is_err());
    }

    #[test]
    fn native_config_canonicalizes_language_model_prefixes() {
        let config = Fp8Config::native(
            None,
            None,
            Some("e4m3"),
            &["model.language_model.embed_tokens".to_string()],
        )
        .unwrap();
        assert_eq!(config.resolve("model.embed_tokens"), None);
        assert!(config.resolve("model.layers.0.mlp.up_proj").is_some());
        assert!(Fp8Config::native(None, None, Some("e5m2"), &[]).is_err());
    }

    #[test]
    fn duplicate_scale_aliases_are_rejected() {
        let scale = Tensor::zeros((), DType::F32, &candle_core::Device::Cpu).unwrap();
        let vb = crate::ShardedSafeTensors::wrap(
            HashMap::from([
                ("layer.weight_scale".to_string(), scale.clone()),
                ("layer.weight_scale_inv".to_string(), scale),
            ]),
            DType::BF16,
            candle_core::Device::Cpu,
        )
        .pp("layer");
        let config = Fp8Config::native(None, None, None, &[]).unwrap();
        let spec = config.resolve("layer").unwrap();
        assert!(spec.scale_names.weight_name(&vb).is_err());
    }

    #[test]
    fn native_missing_scale_falls_back_only_without_an_exclusion_policy() -> CandleResult<()> {
        let device = candle_core::Device::Cpu;
        let vb = crate::ShardedSafeTensors::wrap(
            HashMap::from([(
                "layer.weight".to_string(),
                Tensor::ones((2, 2), DType::F32, &device)?,
            )]),
            DType::F32,
            device,
        )
        .pp("layer");
        let permissive = crate::QuantizedConfig::Fp8 {
            weight_block_size: None,
            activation_scheme: None,
            fmt: None,
            modules_to_not_convert: Vec::new(),
        };
        let layer =
            fp8_checkpoint_linear_b(2, 2, &permissive, false, Default::default(), vb.clone())?;
        assert_eq!(layer.dequantize_w()?.dtype(), DType::F32);

        let explicit = crate::QuantizedConfig::Fp8 {
            weight_block_size: None,
            activation_scheme: None,
            fmt: None,
            modules_to_not_convert: vec!["lm_head".to_string()],
        };
        let error = fp8_checkpoint_linear_b(2, 2, &explicit, false, Default::default(), vb)
            .expect_err("a non-excluded native FP8 module must have a scale");
        assert!(error.to_string().contains("missing FP8 weight scale"));
        Ok(())
    }

    #[test]
    fn native_excluded_module_rejects_a_stray_scale() -> CandleResult<()> {
        let device = candle_core::Device::Cpu;
        let vb = crate::ShardedSafeTensors::wrap(
            HashMap::from([
                (
                    "layer.weight".to_string(),
                    Tensor::ones((2, 2), DType::F32, &device)?,
                ),
                (
                    "layer.weight_scale".to_string(),
                    Tensor::ones((), DType::F32, &device)?,
                ),
            ]),
            DType::F32,
            device,
        )
        .pp("layer");
        let config = crate::QuantizedConfig::Fp8 {
            weight_block_size: None,
            activation_scheme: None,
            fmt: None,
            modules_to_not_convert: vec!["layer".to_string()],
        };
        let error = fp8_checkpoint_linear_b(2, 2, &config, false, Default::default(), vb)
            .expect_err("an excluded native module must not retain a scale");
        assert!(error
            .to_string()
            .contains("unexpectedly has a weight scale"));
        Ok(())
    }

    #[test]
    fn compressed_tensors_target_requires_a_scale() -> CandleResult<()> {
        let device = candle_core::Device::Cpu;
        let vb = crate::ShardedSafeTensors::wrap(
            HashMap::from([(
                "layer.weight".to_string(),
                Tensor::ones((2, 2), DType::F32, &device)?,
            )]),
            DType::F32,
            device,
        )
        .pp("layer");
        let config: crate::QuantizedConfig = serde_json::from_value(json!({
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": ct_args("tensor")
                }
            }
        }))
        .unwrap();
        let error = fp8_checkpoint_linear_b(2, 2, &config, false, Default::default(), vb)
            .expect_err("a compressed-tensors FP8 target must have a scale");
        assert!(error.to_string().contains("missing FP8 weight scale"));
        Ok(())
    }

    #[test]
    fn model_opt_fused_tensor_scales_follow_projection_chunks() -> CandleResult<()> {
        use float8::F8E4M3;

        const N: usize = 8;
        const K: usize = 4;

        let device = candle_core::Device::Cpu;
        let weights = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); N * K], (N, K), &device)?;
        let vb = crate::ShardedSafeTensors::wrap(
            HashMap::from([
                (
                    "model.layers.0.mlp.gate_up_proj.weight".to_string(),
                    weights,
                ),
                (
                    "model.layers.0.mlp.gate_up_proj.weight_scale".to_string(),
                    Tensor::from_vec(vec![2f32, 3.0], 2, &device)?,
                ),
                (
                    "model.layers.0.mlp.gate_up_proj.input_scale".to_string(),
                    Tensor::from_vec(vec![0.5f32, 0.75], 2, &device)?,
                ),
            ]),
            DType::BF16,
            device,
        )
        .pp("model.layers.0.mlp.gate_up_proj");
        let config: crate::QuantizedConfig = serde_json::from_value(json!({
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        }))
        .unwrap();

        for (rank, expected) in [(0, 2f32), (1, 2f32), (2, 3f32), (3, 3f32)] {
            let layer = fp8_checkpoint_linear_b(
                K,
                N,
                &config,
                false,
                crate::Shard::Simple {
                    dim: 0,
                    rank,
                    world_size: 4,
                },
                vb.clone(),
            )?;
            let weight = layer
                .dequantize_w()?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            assert_eq!(weight.len(), N / 4);
            assert!(weight.iter().flatten().all(|value| *value == expected));
        }
        Ok(())
    }

    #[test]
    fn fused_tensor_scale_shards_require_exact_aligned_partitions() -> CandleResult<()> {
        for rank in 0..6 {
            assert_eq!(
                tensor_scale_shard(
                    [12, 4],
                    &[3],
                    crate::Shard::Simple {
                        dim: 0,
                        rank,
                        world_size: 6,
                    },
                    Some(3),
                )?,
                crate::Shard::Offset {
                    dim: 0,
                    offset: rank / 2,
                    len: 1,
                }
            );
        }
        assert_eq!(
            tensor_scale_shard(
                [12, 4],
                &[3],
                crate::Shard::Offset {
                    dim: 0,
                    offset: 8,
                    len: 4,
                },
                Some(3),
            )?,
            crate::Shard::Offset {
                dim: 0,
                offset: 2,
                len: 1,
            }
        );
        assert!(tensor_scale_shard([12, 4], &[2], Default::default(), Some(3)).is_err());
        assert!(tensor_scale_shard([12, 4], &[3], Default::default(), Some(3)).is_err());
        assert!(tensor_scale_shard([12, 4], &[3], Default::default(), None).is_err());
        assert!(tensor_scale_shard(
            [12, 4],
            &[3],
            crate::Shard::Simple {
                dim: 0,
                rank: 0,
                world_size: 0,
            },
            Some(3),
        )
        .is_err());
        assert!(tensor_scale_shard(
            [12, 4],
            &[3],
            crate::Shard::Offset {
                dim: 0,
                offset: 2,
                len: 4,
            },
            Some(3),
        )
        .is_err());
        Ok(())
    }
}
