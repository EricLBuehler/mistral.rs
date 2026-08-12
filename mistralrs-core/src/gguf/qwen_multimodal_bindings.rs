use std::collections::{BTreeSet, HashMap};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::{gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY, pipeline::MultimodalLoaderType};

const GENERAL_ARCHITECTURE: &str = "general.architecture";
const PROJECTOR_TYPE: &str = "clip.projector_type";
const VISION_PROJECTOR_TYPE: &str = "clip.vision.projector_type";
const DEEPSTACK_LAYERS: &str = "clip.vision.is_deepstack_layers";

const QWEN2VL_PROJECTOR: &str = "qwen2vl_merger";
const QWEN25VL_PROJECTOR: &str = "qwen2.5vl_merger";
const QWEN3VL_PROJECTOR: &str = "qwen3vl_merger";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenMultimodalFamily {
    Qwen2Vl,
    Qwen25Vl,
    Qwen3Vl,
    Qwen3VlMoe,
    Qwen35,
    Qwen35Moe,
}

impl QwenMultimodalFamily {
    fn loader_type(self) -> MultimodalLoaderType {
        match self {
            Self::Qwen2Vl => MultimodalLoaderType::Qwen2VL,
            Self::Qwen25Vl => MultimodalLoaderType::Qwen2_5VL,
            Self::Qwen3Vl => MultimodalLoaderType::Qwen3VL,
            Self::Qwen3VlMoe => MultimodalLoaderType::Qwen3VLMoE,
            Self::Qwen35 => MultimodalLoaderType::Qwen3_5,
            Self::Qwen35Moe => MultimodalLoaderType::Qwen3_5Moe,
        }
    }

    fn uses_qwen3_vision(self) -> bool {
        matches!(
            self,
            Self::Qwen3Vl | Self::Qwen3VlMoe | Self::Qwen35 | Self::Qwen35Moe
        )
    }

    fn uses_language_model_prefix(self) -> bool {
        !matches!(self, Self::Qwen2Vl | Self::Qwen25Vl)
    }

    fn has_gemma_norm_offsets(self) -> bool {
        matches!(self, Self::Qwen35 | Self::Qwen35Moe)
    }
}

struct TensorInventory {
    shapes: HashMap<String, Vec<usize>>,
}

impl TensorInventory {
    fn from_archive(archive: &GgufArchive) -> Self {
        Self {
            shapes: archive
                .tensors()
                .iter()
                .map(|(name, info)| (name.clone(), info.shape().to_vec()))
                .collect(),
        }
    }

    #[cfg(test)]
    fn new(tensors: impl IntoIterator<Item = (String, Vec<usize>)>) -> Self {
        Self {
            shapes: tensors.into_iter().collect(),
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }

    fn shape(&self, name: &str) -> Result<&[usize]> {
        self.shapes
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("cannot find GGUF tensor `{name}`"))
    }

    fn layer_indices(&self, prefix: &str) -> BTreeSet<usize> {
        self.shapes
            .keys()
            .filter_map(|name| name.strip_prefix(prefix)?.split_once('.')?.0.parse().ok())
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct GdnMetadata {
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

impl GdnMetadata {
    fn value_per_key(self) -> Result<usize> {
        if self.key_heads == 0
            || self.value_heads == 0
            || !self.value_heads.is_multiple_of(self.key_heads)
        {
            bail!(
                "Qwen3.5 GDN has {} key heads and {} value heads",
                self.key_heads,
                self.value_heads
            );
        }
        Ok(self.value_heads / self.key_heads)
    }

    fn key_dim(self) -> Result<usize> {
        self.key_heads
            .checked_mul(self.key_head_dim)
            .context("Qwen3.5 GDN key dimension overflow")
    }

    fn value_dim(self) -> Result<usize> {
        self.value_heads
            .checked_mul(self.value_head_dim)
            .context("Qwen3.5 GDN value dimension overflow")
    }
}

pub(crate) fn qwen_multimodal_loader_type(archive: &GgufArchive) -> Result<MultimodalLoaderType> {
    qwen_family(archive).map(QwenMultimodalFamily::loader_type)
}

pub(crate) fn build_qwen_multimodal_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let family = qwen_family(archive)?;
    let inventory = TensorInventory::from_archive(archive);
    let deepstack_layers = metadata_bool_indices(archive, DEEPSTACK_LAYERS)?;
    let gdn = if family.has_gemma_norm_offsets() {
        Some(read_gdn_metadata(archive)?)
    } else {
        None
    };
    build_qwen_multimodal_bindings_from_inventory(
        &inventory,
        family,
        deepstack_layers.as_deref(),
        gdn,
    )
}

pub(crate) fn normalize_qwen_multimodal_config(
    loader_type: &MultimodalLoaderType,
    config: &str,
) -> Result<String> {
    if !matches!(
        loader_type,
        MultimodalLoaderType::Qwen3_5 | MultimodalLoaderType::Qwen3_5Moe
    ) {
        return Ok(config.to_string());
    }
    let mut config: serde_json::Value =
        serde_json::from_str(config).context("Qwen3.5 multimodal config is not valid JSON")?;
    let config = config
        .as_object_mut()
        .context("Qwen3.5 multimodal config requires a JSON object")?;
    config.insert("quantization_config".to_string(), serde_json::Value::Null);
    let text_config = config
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
        .context("Qwen3.5 multimodal config requires an object-valued `text_config`")?;
    text_config.insert("quantization_config".to_string(), serde_json::Value::Null);
    text_config.insert(
        GDN_V_HEAD_LAYOUT_CONFIG_KEY.to_string(),
        serde_json::Value::String("tiled".to_string()),
    );
    serde_json::to_string(&config).context("Failed to serialize Qwen3.5 multimodal config")
}

pub(crate) fn build_qwen35_text_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let architecture = metadata_string(archive, GENERAL_ARCHITECTURE)?
        .context("GGUF metadata is missing `general.architecture`")?;
    if architecture != "qwen35" {
        bail!("Qwen3.5 text binding does not support `{architecture}`");
    }
    let inventory = TensorInventory::from_archive(archive);
    let mut bindings = GgufBindingMap::new();
    bind_text(
        &inventory,
        QwenMultimodalFamily::Qwen35,
        Some(read_gdn_metadata(archive)?),
        &mut bindings,
    )?;
    Ok(bindings)
}

fn qwen_family(archive: &GgufArchive) -> Result<QwenMultimodalFamily> {
    let architecture = metadata_string(archive, GENERAL_ARCHITECTURE)?
        .context("GGUF metadata is missing `general.architecture`")?;
    let projector = projector_type(archive)?;
    qwen_family_from_names(architecture, projector)
}

fn qwen_family_from_names(
    architecture: &str,
    projector: Option<&str>,
) -> Result<QwenMultimodalFamily> {
    let family = match architecture {
        "qwen2vl" => match projector {
            Some(QWEN2VL_PROJECTOR) => QwenMultimodalFamily::Qwen2Vl,
            Some(QWEN25VL_PROJECTOR) => QwenMultimodalFamily::Qwen25Vl,
            Some(other) => bail!("unsupported Qwen 2 VL projector `{other}`"),
            None => bail!(
                "`qwen2vl` requires projector metadata to distinguish Qwen 2 VL from Qwen 2.5 VL"
            ),
        },
        "qwen3vl" => {
            require_projector(projector, QWEN3VL_PROJECTOR)?;
            QwenMultimodalFamily::Qwen3Vl
        }
        "qwen3vlmoe" => {
            require_projector(projector, QWEN3VL_PROJECTOR)?;
            QwenMultimodalFamily::Qwen3VlMoe
        }
        "qwen35" => {
            require_projector(projector, QWEN3VL_PROJECTOR)?;
            QwenMultimodalFamily::Qwen35
        }
        "qwen35moe" => {
            require_projector(projector, QWEN3VL_PROJECTOR)?;
            QwenMultimodalFamily::Qwen35Moe
        }
        other => bail!("unsupported Qwen multimodal GGUF architecture `{other}`"),
    };
    Ok(family)
}

fn require_projector(projector: Option<&str>, expected: &str) -> Result<()> {
    match projector {
        Some(projector) if projector == expected => Ok(()),
        Some(projector) => {
            bail!("expected Qwen vision projector `{expected}`, found `{projector}`")
        }
        None => bail!("Qwen multimodal GGUF requires vision projector metadata `{expected}`"),
    }
}

fn projector_type(archive: &GgufArchive) -> Result<Option<&str>> {
    let standalone = metadata_string(archive, PROJECTOR_TYPE)?;
    let mixed = metadata_string(archive, VISION_PROJECTOR_TYPE)?;
    match (standalone, mixed) {
        (Some(left), Some(right)) if left != right => bail!(
            "GGUF projector metadata conflicts: `{PROJECTOR_TYPE}` is `{left}`, `{VISION_PROJECTOR_TYPE}` is `{right}`"
        ),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<Option<&'a str>> {
    match archive.metadata_value(key) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => Ok(None),
    }
}

fn metadata_bool_indices(archive: &GgufArchive, key: &str) -> Result<Option<Vec<usize>>> {
    let Some(value) = archive.metadata_value(key) else {
        return Ok(None);
    };
    let Value::Array(values) = value else {
        bail!("GGUF metadata `{key}` must be a boolean array");
    };
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| match value {
            Value::Bool(true) => Some(Ok(index)),
            Value::Bool(false) => None,
            _ => Some(Err(anyhow::anyhow!(
                "GGUF metadata `{key}` must be a boolean array"
            ))),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn metadata_usize(archive: &GgufArchive, key: &str) -> Result<usize> {
    let value = archive
        .metadata_value(key)
        .with_context(|| format!("GGUF metadata is missing `{key}`"))?;
    let value = match value {
        Value::U8(value) => *value as u64,
        Value::U16(value) => *value as u64,
        Value::U32(value) => *value as u64,
        Value::U64(value) => *value,
        Value::I8(value) if *value >= 0 => *value as u64,
        Value::I16(value) if *value >= 0 => *value as u64,
        Value::I32(value) if *value >= 0 => *value as u64,
        Value::I64(value) if *value >= 0 => *value as u64,
        _ => bail!("GGUF metadata `{key}` must be a nonnegative integer"),
    };
    usize::try_from(value).with_context(|| format!("GGUF metadata `{key}` does not fit usize"))
}

fn read_gdn_metadata(archive: &GgufArchive) -> Result<GdnMetadata> {
    let architecture = metadata_string(archive, GENERAL_ARCHITECTURE)?
        .context("GGUF metadata is missing `general.architecture`")?;
    let prefix = architecture;
    let key_heads = metadata_usize(archive, &format!("{prefix}.ssm.group_count"))?;
    let value_heads = metadata_usize(archive, &format!("{prefix}.ssm.time_step_rank"))?;
    let key_head_dim = metadata_usize(archive, &format!("{prefix}.ssm.state_size"))?;
    let value_dim = metadata_usize(archive, &format!("{prefix}.ssm.inner_size"))?;
    if value_heads == 0 || !value_dim.is_multiple_of(value_heads) {
        bail!("Qwen3.5 GDN inner size {value_dim} is not divisible by {value_heads} value heads");
    }
    let metadata = GdnMetadata {
        key_heads,
        value_heads,
        key_head_dim,
        value_head_dim: value_dim / value_heads,
    };
    metadata.value_per_key()?;
    Ok(metadata)
}

fn build_qwen_multimodal_bindings_from_inventory(
    inventory: &TensorInventory,
    family: QwenMultimodalFamily,
    deepstack_layers: Option<&[usize]>,
    gdn: Option<GdnMetadata>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_text(inventory, family, gdn, &mut bindings)?;
    if family.uses_qwen3_vision() {
        bind_qwen3_vision(inventory, deepstack_layers, &mut bindings)?;
    } else {
        bind_qwen2_vision(inventory, family, &mut bindings)?;
    }
    Ok(bindings)
}

fn bind_text(
    inventory: &TensorInventory,
    family: QwenMultimodalFamily,
    gdn: Option<GdnMetadata>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let model = if family.uses_language_model_prefix() {
        "model.language_model"
    } else {
        "model"
    };
    bind(
        inventory,
        bindings,
        format!("{model}.embed_tokens.weight"),
        "token_embd.weight",
    );
    bind_text_norm(
        inventory,
        bindings,
        family,
        format!("{model}.norm.weight"),
        "output_norm.weight",
    );
    bind(inventory, bindings, "lm_head.weight", "output.weight");

    for layer in inventory.layer_indices("blk.") {
        let native = format!("{model}.layers.{layer}");
        let source = format!("blk.{layer}");
        for suffix in ["weight", "bias"] {
            for (target, role) in [
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_output"),
                ("mlp.gate_proj", "ffn_gate"),
                ("mlp.up_proj", "ffn_up"),
                ("mlp.down_proj", "ffn_down"),
                ("mlp.gate", "ffn_gate_inp"),
            ] {
                bind(
                    inventory,
                    bindings,
                    format!("{native}.{target}.{suffix}"),
                    format!("{source}.{role}.{suffix}"),
                );
            }
        }
        for (target, role) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            (
                "post_attention_layernorm.weight",
                if family.has_gemma_norm_offsets() {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                },
            ),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ] {
            bind_text_norm(
                inventory,
                bindings,
                family,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
        bind_experts(inventory, &native, &source, bindings);
        bind_shared_expert(inventory, &native, &source, bindings)?;
        if let Some(gdn) = gdn {
            bind_gdn(inventory, &native, &source, gdn, bindings)?;
        }
    }
    Ok(())
}

fn bind_experts(
    inventory: &TensorInventory,
    native: &str,
    source: &str,
    bindings: &mut GgufBindingMap,
) {
    for (projection, role) in [
        ("gate", "ffn_gate_exps"),
        ("up", "ffn_up_exps"),
        ("down", "ffn_down_exps"),
    ] {
        bind(
            inventory,
            bindings,
            format!("{native}.mlp.experts.{projection}_proj.weight"),
            format!("{source}.{role}.weight"),
        );
    }
}

fn bind_shared_expert(
    inventory: &TensorInventory,
    native: &str,
    source: &str,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    for (projection, role) in [
        ("gate", "ffn_gate_shexp"),
        ("up", "ffn_up_shexp"),
        ("down", "ffn_down_shexp"),
    ] {
        bind(
            inventory,
            bindings,
            format!("{native}.mlp.shared_expert.{projection}_proj.weight"),
            format!("{source}.{role}.weight"),
        );
    }
    let source_gate = format!("{source}.ffn_gate_inp_shexp.weight");
    if inventory.contains(&source_gate) {
        let shape = inventory.shape(&source_gate)?;
        let binding = match shape {
            [hidden] => GgufTensorBinding::tensor(&source_gate).reshape(vec![1, *hidden]),
            [1, _] => GgufTensorBinding::tensor(&source_gate),
            _ => bail!(
                "Qwen3.5 shared expert gate `{source_gate}` must have shape [hidden] or [1, hidden]"
            ),
        };
        bindings.insert(format!("{native}.mlp.shared_expert_gate.weight"), binding);
    }
    Ok(())
}

fn bind_gdn(
    inventory: &TensorInventory,
    native: &str,
    source: &str,
    metadata: GdnMetadata,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let native = format!("{native}.linear_attn");
    let qkv = format!("{source}.attn_qkv.weight");
    if inventory.contains(&qkv) {
        let shape = inventory.shape(&qkv)?;
        let &[rows, _] = shape else {
            bail!("Qwen3.5 GDN tensor `{qkv}` must be rank 2");
        };
        let value_dim = metadata.value_dim()?;
        let expected_rows = metadata
            .key_dim()?
            .checked_mul(2)
            .and_then(|qk| qk.checked_add(value_dim))
            .context("Qwen3.5 GDN projection dimension overflow")?;
        if rows != expected_rows {
            bail!("Qwen3.5 GDN tensor `{qkv}` has incompatible shape {shape:?}");
        }
        bindings.insert(
            format!("{native}.in_proj_qkv.weight"),
            GgufTensorBinding::tensor(&qkv),
        );
    }
    for (target, role) in [
        ("in_proj_z.weight", "attn_gate.weight"),
        ("in_proj_b.weight", "ssm_beta.weight"),
        ("in_proj_a.weight", "ssm_alpha.weight"),
        ("dt_bias", "ssm_dt.bias"),
        ("out_proj.weight", "ssm_out.weight"),
    ] {
        bind(
            inventory,
            bindings,
            format!("{native}.{target}"),
            format!("{source}.{role}"),
        );
    }
    let a = format!("{source}.ssm_a");
    if inventory.contains(&a) {
        bindings.insert(
            format!("{native}.A_log"),
            GgufTensorBinding::tensor(&a).affine(-1.0, 0.0).log(),
        );
    }
    let conv = format!("{source}.ssm_conv1d.weight");
    if inventory.contains(&conv) {
        let shape = inventory.shape(&conv)?;
        let &[channels, kernel] = shape else {
            bail!("Qwen3.5 GDN convolution `{conv}` must be rank 2");
        };
        let value_dim = metadata.value_dim()?;
        let expected_channels = metadata
            .key_dim()?
            .checked_mul(2)
            .and_then(|qk| qk.checked_add(value_dim))
            .context("Qwen3.5 GDN convolution dimension overflow")?;
        if channels != expected_channels {
            bail!("Qwen3.5 GDN convolution `{conv}` has incompatible shape {shape:?}");
        }
        bindings.insert(
            format!("{native}.conv1d.weight"),
            GgufTensorBinding::tensor(&conv).reshape(vec![channels, 1, kernel]),
        );
    }
    bind(
        inventory,
        bindings,
        format!("{native}.norm.weight"),
        format!("{source}.ssm_norm.weight"),
    );
    Ok(())
}

fn bind_qwen2_vision(
    inventory: &TensorInventory,
    family: QwenMultimodalFamily,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    bind_patch_embedding(inventory, "visual", bindings)?;
    for layer in inventory.layer_indices("v.blk.") {
        let native = format!("visual.blocks.{layer}");
        let source = format!("v.blk.{layer}");
        bind_fused_qkv(inventory, &native, &source, bindings);
        for suffix in ["weight", "bias"] {
            bind(
                inventory,
                bindings,
                format!("{native}.attn.proj.{suffix}"),
                format!("{source}.attn_out.{suffix}"),
            );
            bind(
                inventory,
                bindings,
                format!("{native}.norm1.{suffix}"),
                format!("{source}.ln1.{suffix}"),
            );
            bind(
                inventory,
                bindings,
                format!("{native}.norm2.{suffix}"),
                format!("{source}.ln2.{suffix}"),
            );
            match family {
                QwenMultimodalFamily::Qwen2Vl => {
                    bind(
                        inventory,
                        bindings,
                        format!("{native}.mlp.fc1.{suffix}"),
                        format!("{source}.ffn_up.{suffix}"),
                    );
                    bind(
                        inventory,
                        bindings,
                        format!("{native}.mlp.fc2.{suffix}"),
                        format!("{source}.ffn_down.{suffix}"),
                    );
                }
                QwenMultimodalFamily::Qwen25Vl => {
                    for projection in ["gate", "up", "down"] {
                        bind(
                            inventory,
                            bindings,
                            format!("{native}.mlp.{projection}_proj.{suffix}"),
                            format!("{source}.ffn_{projection}.{suffix}"),
                        );
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    bind_merger(
        inventory,
        "visual.merger",
        "ln_q",
        "mlp.0",
        "mlp.2",
        bindings,
    );
    Ok(())
}

fn bind_qwen3_vision(
    inventory: &TensorInventory,
    deepstack_layers: Option<&[usize]>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    bind_patch_embedding(inventory, "model.visual", bindings)?;
    bind(
        inventory,
        bindings,
        "model.visual.pos_embed.weight",
        "v.position_embd.weight",
    );
    for layer in inventory.layer_indices("v.blk.") {
        let native = format!("model.visual.blocks.{layer}");
        let source = format!("v.blk.{layer}");
        bind_fused_qkv(inventory, &native, &source, bindings);
        for suffix in ["weight", "bias"] {
            for (target, role) in [
                ("attn.proj", "attn_out"),
                ("norm1", "ln1"),
                ("norm2", "ln2"),
                ("mlp.linear_fc1", "ffn_up"),
                ("mlp.linear_fc2", "ffn_down"),
            ] {
                bind(
                    inventory,
                    bindings,
                    format!("{native}.{target}.{suffix}"),
                    format!("{source}.{role}.{suffix}"),
                );
            }
        }
    }
    bind_merger(
        inventory,
        "model.visual.merger",
        "norm",
        "linear_fc1",
        "linear_fc2",
        bindings,
    );

    let layers = deepstack_layers.map(ToOwned::to_owned).unwrap_or_else(|| {
        inventory
            .layer_indices("v.deepstack.")
            .into_iter()
            .collect()
    });
    for (merger, layer) in layers.into_iter().enumerate() {
        for suffix in ["weight", "bias"] {
            for (target, role) in [
                ("norm", "norm"),
                ("linear_fc1", "fc1"),
                ("linear_fc2", "fc2"),
            ] {
                bind(
                    inventory,
                    bindings,
                    format!("model.visual.deepstack_merger_list.{merger}.{target}.{suffix}"),
                    format!("v.deepstack.{layer}.{role}.{suffix}"),
                );
            }
        }
    }
    Ok(())
}

fn bind_patch_embedding(
    inventory: &TensorInventory,
    native: &str,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let first = "v.patch_embd.weight";
    let second = "v.patch_embd.weight.1";
    if inventory.contains(first) {
        let shape = inventory.shape(first)?;
        let binding = if inventory.contains(second) {
            if inventory.shape(second)? != shape {
                bail!("Qwen vision temporal patch tensors have different shapes");
            }
            GgufTensorBinding::stack(
                vec![
                    GgufTensorBinding::tensor(first),
                    GgufTensorBinding::tensor(second),
                ],
                2,
            )
        } else if shape.len() == 5 {
            GgufTensorBinding::tensor(first)
        } else {
            bail!("Qwen vision patch tensor `{second}` is missing");
        };
        bindings.insert(format!("{native}.patch_embed.proj.weight"), binding);
    }
    bind(
        inventory,
        bindings,
        format!("{native}.patch_embed.proj.bias"),
        "v.patch_embd.bias",
    );
    Ok(())
}

fn bind_fused_qkv(
    inventory: &TensorInventory,
    native: &str,
    source: &str,
    bindings: &mut GgufBindingMap,
) {
    for suffix in ["weight", "bias"] {
        let fused = format!("{source}.attn_qkv.{suffix}");
        if inventory.contains(&fused) {
            bindings.insert(
                format!("{native}.attn.qkv.{suffix}"),
                GgufTensorBinding::tensor(fused),
            );
            continue;
        }
        let inputs = ["q", "k", "v"]
            .into_iter()
            .map(|projection| format!("{source}.attn_{projection}.{suffix}"))
            .collect::<Vec<_>>();
        if inputs.iter().all(|name| inventory.contains(name)) {
            bindings.insert(
                format!("{native}.attn.qkv.{suffix}"),
                GgufTensorBinding::concat(
                    inputs.into_iter().map(GgufTensorBinding::tensor).collect(),
                    0,
                ),
            );
        }
    }
}

fn bind_merger(
    inventory: &TensorInventory,
    native: &str,
    norm: &str,
    first: &str,
    second: &str,
    bindings: &mut GgufBindingMap,
) {
    for suffix in ["weight", "bias"] {
        bind(
            inventory,
            bindings,
            format!("{native}.{norm}.{suffix}"),
            format!("v.post_ln.{suffix}"),
        );
        bind(
            inventory,
            bindings,
            format!("{native}.{first}.{suffix}"),
            format!("mm.0.{suffix}"),
        );
        bind(
            inventory,
            bindings,
            format!("{native}.{second}.{suffix}"),
            format!("mm.2.{suffix}"),
        );
    }
}

fn bind_text_norm(
    inventory: &TensorInventory,
    bindings: &mut GgufBindingMap,
    family: QwenMultimodalFamily,
    native: impl Into<String>,
    source: impl Into<String>,
) {
    let source = source.into();
    if inventory.contains(&source) {
        let binding = GgufTensorBinding::tensor(&source);
        bindings.insert(
            native,
            if family.has_gemma_norm_offsets() {
                binding.affine(1.0, -1.0)
            } else {
                binding
            },
        );
    }
}

fn bind(
    inventory: &TensorInventory,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
) {
    let source = source.into();
    if inventory.contains(&source) {
        bindings.insert(native, GgufTensorBinding::tensor(source));
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Arc};

    use candle_core::{
        quantized::{gguf_file, GgmlDType, QTensor},
        DType, Device, Tensor,
    };
    use mistralrs_quant::{
        ColumnParallelLayer, Comm, GgufWeightSource, Id, QuantizedConfig, QuantizedWeightSource,
        Shard,
    };
    use tempfile::NamedTempFile;

    use super::*;

    const TINY_HIDDEN_SIZE: usize = 256;
    const TINY_KEY_HEADS: u32 = 1;
    const TINY_VALUE_HEADS: u32 = 2;
    const TINY_KEY_HEAD_DIM: u32 = 1;
    const TINY_VALUE_DIM: u32 = 256;
    const TINY_QKV_ROWS: usize = (2 * TINY_KEY_HEADS * TINY_KEY_HEAD_DIM + TINY_VALUE_DIM) as usize;

    fn inventory(tensors: &[(&str, &[usize])]) -> TensorInventory {
        TensorInventory::new(
            tensors
                .iter()
                .map(|(name, shape)| ((*name).to_string(), shape.to_vec())),
        )
    }

    fn names(bindings: &GgufBindingMap) -> BTreeSet<String> {
        bindings.iter().map(|(name, _)| name.to_string()).collect()
    }

    fn tiny_qwen35_multimodal_archive(
        architecture: &str,
    ) -> Result<(NamedTempFile, Arc<GgufArchive>)> {
        let qkv = q4k_ones(TINY_QKV_ROWS, TINY_HIDDEN_SIZE)?;
        let gate = q4k_ones(TINY_VALUE_DIM as usize, TINY_HIDDEN_SIZE)?;
        let beta = q4k_ones(TINY_VALUE_HEADS as usize, TINY_HIDDEN_SIZE)?;
        let alpha = q4k_ones(TINY_VALUE_HEADS as usize, TINY_HIDDEN_SIZE)?;
        let output = q4k_ones(TINY_HIDDEN_SIZE, TINY_VALUE_DIM as usize)?;
        let metadata = [
            (
                GENERAL_ARCHITECTURE.to_string(),
                Value::String(architecture.to_string()),
            ),
            (
                PROJECTOR_TYPE.to_string(),
                Value::String(QWEN3VL_PROJECTOR.to_string()),
            ),
            (
                format!("{architecture}.ssm.group_count"),
                Value::U32(TINY_KEY_HEADS),
            ),
            (
                format!("{architecture}.ssm.time_step_rank"),
                Value::U32(TINY_VALUE_HEADS),
            ),
            (
                format!("{architecture}.ssm.state_size"),
                Value::U32(TINY_KEY_HEAD_DIM),
            ),
            (
                format!("{architecture}.ssm.inner_size"),
                Value::U32(TINY_VALUE_DIM),
            ),
        ];
        let metadata = metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect::<Vec<_>>();

        let mut file = NamedTempFile::new()?;
        gguf_file::write(
            file.as_file_mut(),
            &metadata,
            &[
                ("blk.0.attn_qkv.weight", &qkv),
                ("blk.0.attn_gate.weight", &gate),
                ("blk.0.ssm_beta.weight", &beta),
                ("blk.0.ssm_alpha.weight", &alpha),
                ("blk.0.ssm_out.weight", &output),
            ],
        )?;
        file.as_file_mut().flush()?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        Ok((file, archive))
    }

    fn q4k_ones(rows: usize, cols: usize) -> Result<QTensor> {
        let weight = Tensor::ones((rows, cols), DType::F32, &Device::Cpu)?;
        QTensor::quantize(&weight, GgmlDType::Q4K).map_err(Into::into)
    }

    #[test]
    fn architecture_and_projector_select_exact_loaders() -> Result<()> {
        let cases = [
            (
                "qwen2vl",
                Some(QWEN2VL_PROJECTOR),
                MultimodalLoaderType::Qwen2VL,
            ),
            (
                "qwen2vl",
                Some(QWEN25VL_PROJECTOR),
                MultimodalLoaderType::Qwen2_5VL,
            ),
            (
                "qwen3vl",
                Some(QWEN3VL_PROJECTOR),
                MultimodalLoaderType::Qwen3VL,
            ),
            (
                "qwen3vlmoe",
                Some(QWEN3VL_PROJECTOR),
                MultimodalLoaderType::Qwen3VLMoE,
            ),
            (
                "qwen35",
                Some(QWEN3VL_PROJECTOR),
                MultimodalLoaderType::Qwen3_5,
            ),
            (
                "qwen35moe",
                Some(QWEN3VL_PROJECTOR),
                MultimodalLoaderType::Qwen3_5Moe,
            ),
        ];
        for (architecture, projector, expected) in cases {
            assert_eq!(
                qwen_family_from_names(architecture, projector)?.loader_type(),
                expected
            );
        }
        assert!(qwen_family_from_names("qwen2vl", None).is_err());
        assert!(qwen_family_from_names("qwen3vl", None).is_err());
        assert!(qwen_family_from_names("qwen3vl", Some(QWEN2VL_PROJECTOR)).is_err());
        Ok(())
    }

    #[test]
    fn qwen2vl_inventory_is_exhaustive() -> Result<()> {
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("output.weight", &[32, 8]),
            ("blk.0.attn_norm.weight", &[8]),
            ("blk.0.ffn_norm.weight", &[8]),
            ("blk.0.attn_q.weight", &[8, 8]),
            ("blk.0.attn_q.bias", &[8]),
            ("blk.0.attn_k.weight", &[8, 8]),
            ("blk.0.attn_k.bias", &[8]),
            ("blk.0.attn_v.weight", &[8, 8]),
            ("blk.0.attn_v.bias", &[8]),
            ("blk.0.attn_output.weight", &[8, 8]),
            ("blk.0.ffn_gate.weight", &[16, 8]),
            ("blk.0.ffn_up.weight", &[16, 8]),
            ("blk.0.ffn_down.weight", &[8, 16]),
            ("v.patch_embd.weight", &[8, 3, 2, 2]),
            ("v.patch_embd.weight.1", &[8, 3, 2, 2]),
            ("v.blk.0.attn_q.weight", &[8, 8]),
            ("v.blk.0.attn_k.weight", &[8, 8]),
            ("v.blk.0.attn_v.weight", &[8, 8]),
            ("v.blk.0.attn_q.bias", &[8]),
            ("v.blk.0.attn_k.bias", &[8]),
            ("v.blk.0.attn_v.bias", &[8]),
            ("v.blk.0.attn_out.weight", &[8, 8]),
            ("v.blk.0.attn_out.bias", &[8]),
            ("v.blk.0.ln1.weight", &[8]),
            ("v.blk.0.ln1.bias", &[8]),
            ("v.blk.0.ln2.weight", &[8]),
            ("v.blk.0.ln2.bias", &[8]),
            ("v.blk.0.ffn_up.weight", &[16, 8]),
            ("v.blk.0.ffn_up.bias", &[16]),
            ("v.blk.0.ffn_down.weight", &[8, 16]),
            ("v.blk.0.ffn_down.bias", &[8]),
            ("v.post_ln.weight", &[8]),
            ("v.post_ln.bias", &[8]),
            ("mm.0.weight", &[16, 8]),
            ("mm.0.bias", &[16]),
            ("mm.2.weight", &[8, 16]),
            ("mm.2.bias", &[8]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen2Vl,
            None,
            None,
        )?;
        let expected = [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.norm.weight",
            "visual.blocks.0.attn.proj.bias",
            "visual.blocks.0.attn.proj.weight",
            "visual.blocks.0.attn.qkv.bias",
            "visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.mlp.fc1.bias",
            "visual.blocks.0.mlp.fc1.weight",
            "visual.blocks.0.mlp.fc2.bias",
            "visual.blocks.0.mlp.fc2.weight",
            "visual.blocks.0.norm1.bias",
            "visual.blocks.0.norm1.weight",
            "visual.blocks.0.norm2.bias",
            "visual.blocks.0.norm2.weight",
            "visual.merger.ln_q.bias",
            "visual.merger.ln_q.weight",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.0.weight",
            "visual.merger.mlp.2.bias",
            "visual.merger.mlp.2.weight",
            "visual.patch_embed.proj.weight",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        assert_eq!(names(&bindings), expected);
        assert!(matches!(
            bindings.get("visual.blocks.0.attn.qkv.weight"),
            Some(GgufTensorBinding::Concat { dim: 0, .. })
        ));
        assert!(matches!(
            bindings.get("visual.patch_embed.proj.weight"),
            Some(GgufTensorBinding::Stack { dim: 2, .. })
        ));
        Ok(())
    }

    #[test]
    fn qwen25vl_uses_gated_vision_mlp() -> Result<()> {
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("v.patch_embd.weight", &[8, 3, 2, 2]),
            ("v.patch_embd.weight.1", &[8, 3, 2, 2]),
            ("v.blk.0.ffn_gate.weight", &[16, 8]),
            ("v.blk.0.ffn_up.weight", &[16, 8]),
            ("v.blk.0.ffn_down.weight", &[8, 16]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen25Vl,
            None,
            None,
        )?;
        for projection in ["gate", "up", "down"] {
            assert!(bindings
                .get(&format!("visual.blocks.0.mlp.{projection}_proj.weight"))
                .is_some());
        }
        assert!(bindings.get("visual.blocks.0.mlp.fc1.weight").is_none());
        Ok(())
    }

    #[test]
    fn qwen3vl_maps_deepstack_absolute_layers_to_merger_slots() -> Result<()> {
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("v.patch_embd.weight", &[8, 3, 2, 2]),
            ("v.patch_embd.weight.1", &[8, 3, 2, 2]),
            ("v.patch_embd.bias", &[8]),
            ("v.position_embd.weight", &[16, 8]),
            ("v.blk.0.attn_qkv.weight", &[24, 8]),
            ("v.blk.0.attn_qkv.bias", &[24]),
            ("v.blk.0.attn_out.weight", &[8, 8]),
            ("v.blk.0.ln1.weight", &[8]),
            ("v.blk.0.ln2.weight", &[8]),
            ("v.blk.0.ffn_up.weight", &[16, 8]),
            ("v.blk.0.ffn_down.weight", &[8, 16]),
            ("v.post_ln.weight", &[8]),
            ("mm.0.weight", &[16, 8]),
            ("mm.2.weight", &[8, 16]),
            ("v.deepstack.5.norm.weight", &[8]),
            ("v.deepstack.5.fc1.weight", &[16, 8]),
            ("v.deepstack.5.fc2.weight", &[8, 16]),
            ("v.deepstack.17.norm.weight", &[8]),
            ("v.deepstack.17.fc1.weight", &[16, 8]),
            ("v.deepstack.17.fc2.weight", &[8, 16]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen3Vl,
            Some(&[5, 17]),
            None,
        )?;
        assert!(matches!(
            bindings.get("model.visual.blocks.0.attn.qkv.weight"),
            Some(GgufTensorBinding::Tensor(source)) if source == "v.blk.0.attn_qkv.weight"
        ));
        assert!(bindings
            .get("model.visual.deepstack_merger_list.0.norm.weight")
            .is_some());
        assert!(bindings
            .get("model.visual.deepstack_merger_list.1.linear_fc2.weight")
            .is_some());
        assert!(bindings
            .get("model.visual.deepstack_merger_list.5.norm.weight")
            .is_none());
        Ok(())
    }

    #[test]
    fn moe_experts_bind_directly_for_fast_weight_source() -> Result<()> {
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("blk.0.ffn_gate_inp.weight", &[4, 8]),
            ("blk.0.ffn_gate_exps.weight", &[4, 16, 8]),
            ("blk.0.ffn_up_exps.weight", &[4, 16, 8]),
            ("blk.0.ffn_down_exps.weight", &[4, 8, 16]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen3VlMoe,
            None,
            None,
        )?;
        for (projection, source) in [
            ("gate", "blk.0.ffn_gate_exps.weight"),
            ("up", "blk.0.ffn_up_exps.weight"),
            ("down", "blk.0.ffn_down_exps.weight"),
        ] {
            assert!(matches!(
                bindings.get(&format!(
                    "model.language_model.layers.0.mlp.experts.{projection}_proj.weight"
                )),
                Some(GgufTensorBinding::Tensor(actual)) if actual == source
            ));
        }
        Ok(())
    }

    #[test]
    fn qwen35_moe_preserves_tiled_gdn_and_inverts_remaining_transforms() -> Result<()> {
        let metadata = GdnMetadata {
            key_heads: 2,
            value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
        };
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("blk.0.attn_norm.weight", &[8]),
            ("blk.0.post_attention_norm.weight", &[8]),
            ("blk.0.attn_qkv.weight", &[16, 8]),
            ("blk.0.attn_gate.weight", &[8, 8]),
            ("blk.0.ssm_beta.weight", &[4, 8]),
            ("blk.0.ssm_alpha.weight", &[4, 8]),
            ("blk.0.ssm_dt.bias", &[4]),
            ("blk.0.ssm_a", &[4]),
            ("blk.0.ssm_conv1d.weight", &[16, 3]),
            ("blk.0.ssm_norm.weight", &[2]),
            ("blk.0.ssm_out.weight", &[8, 8]),
            ("blk.0.ffn_gate_exps.weight", &[4, 2, 8]),
            ("blk.0.ffn_up_exps.weight", &[4, 2, 8]),
            ("blk.0.ffn_down_exps.weight", &[4, 8, 2]),
            ("blk.0.ffn_gate_inp.weight", &[4, 8]),
            ("blk.0.ffn_gate_shexp.weight", &[2, 8]),
            ("blk.0.ffn_up_shexp.weight", &[2, 8]),
            ("blk.0.ffn_down_shexp.weight", &[8, 2]),
            ("blk.0.ffn_gate_inp_shexp.weight", &[8]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen35Moe,
            None,
            Some(metadata),
        )?;
        assert!(matches!(
            bindings.get("model.language_model.norm.weight"),
            Some(GgufTensorBinding::Affine {
                mul,
                add,
                ..
            }) if *mul == 1.0 && *add == -1.0
        ));
        assert!(matches!(
            bindings.get("model.language_model.layers.0.linear_attn.A_log"),
            Some(GgufTensorBinding::Log { .. })
        ));
        assert!(matches!(
            bindings.get("model.language_model.layers.0.linear_attn.conv1d.weight"),
            Some(GgufTensorBinding::Reshape { dims, .. }) if dims == &[16, 1, 3]
        ));
        assert!(matches!(
            bindings.get(
                "model.language_model.layers.0.mlp.shared_expert_gate.weight"
            ),
            Some(GgufTensorBinding::Reshape { dims, .. }) if dims == &[1, 8]
        ));
        assert!(matches!(
            bindings.get(
                "model.language_model.layers.0.mlp.experts.gate_proj.weight"
            ),
            Some(GgufTensorBinding::Tensor(source))
                if source == "blk.0.ffn_gate_exps.weight"
        ));
        for (target, source) in [
            ("in_proj_qkv.weight", "blk.0.attn_qkv.weight"),
            ("in_proj_z.weight", "blk.0.attn_gate.weight"),
            ("in_proj_b.weight", "blk.0.ssm_beta.weight"),
            ("in_proj_a.weight", "blk.0.ssm_alpha.weight"),
            ("dt_bias", "blk.0.ssm_dt.bias"),
            ("out_proj.weight", "blk.0.ssm_out.weight"),
        ] {
            assert!(matches!(
                bindings.get(&format!(
                    "model.language_model.layers.0.linear_attn.{target}"
                )),
                Some(GgufTensorBinding::Tensor(actual)) if actual == source
            ));
        }
        Ok(())
    }

    #[test]
    fn qwen35_text_preserves_tiled_quantized_gdn_weights() -> Result<()> {
        let metadata = GdnMetadata {
            key_heads: 2,
            value_heads: 4,
            key_head_dim: 2,
            value_head_dim: 2,
        };
        let inventory = inventory(&[
            ("token_embd.weight", &[32, 8]),
            ("output_norm.weight", &[8]),
            ("blk.0.attn_qkv.weight", &[16, 8]),
            ("blk.0.attn_gate.weight", &[8, 8]),
            ("blk.0.ssm_beta.weight", &[4, 8]),
            ("blk.0.ssm_alpha.weight", &[4, 8]),
            ("blk.0.ssm_dt.bias", &[4]),
            ("blk.0.ssm_a", &[4]),
            ("blk.0.ssm_conv1d.weight", &[16, 3]),
            ("blk.0.ssm_norm.weight", &[2]),
            ("blk.0.ssm_out.weight", &[8, 8]),
        ]);
        let bindings = build_qwen_multimodal_bindings_from_inventory(
            &inventory,
            QwenMultimodalFamily::Qwen35,
            None,
            Some(metadata),
        )?;
        for (target, source) in [
            ("in_proj_qkv.weight", "blk.0.attn_qkv.weight"),
            ("in_proj_z.weight", "blk.0.attn_gate.weight"),
            ("in_proj_b.weight", "blk.0.ssm_beta.weight"),
            ("in_proj_a.weight", "blk.0.ssm_alpha.weight"),
            ("dt_bias", "blk.0.ssm_dt.bias"),
            ("out_proj.weight", "blk.0.ssm_out.weight"),
        ] {
            assert!(matches!(
                bindings.get(&format!(
                    "model.language_model.layers.0.linear_attn.{target}"
                )),
                Some(GgufTensorBinding::Tensor(actual)) if actual == source
            ));
        }
        assert!(matches!(
            bindings.get("model.language_model.layers.0.linear_attn.A_log"),
            Some(GgufTensorBinding::Log { .. })
        ));
        assert!(matches!(
            bindings.get("model.language_model.layers.0.linear_attn.conv1d.weight"),
            Some(GgufTensorBinding::Reshape { dims, .. }) if dims == &[16, 1, 3]
        ));
        Ok(())
    }

    #[test]
    fn qwen35_multimodal_q4k_gdn_stays_packed() -> Result<()> {
        for architecture in ["qwen35", "qwen35moe"] {
            let (_file, archive) = tiny_qwen35_multimodal_archive(architecture)?;
            let bindings = build_qwen_multimodal_bindings(&archive)?;
            let source = GgufWeightSource::new(archive, &bindings, DType::F32)?;
            for (projection, expected_shape) in [
                ("in_proj_qkv", [TINY_QKV_ROWS, TINY_HIDDEN_SIZE]),
                ("in_proj_z", [TINY_VALUE_DIM as usize, TINY_HIDDEN_SIZE]),
                ("in_proj_b", [TINY_VALUE_HEADS as usize, TINY_HIDDEN_SIZE]),
                ("in_proj_a", [TINY_VALUE_HEADS as usize, TINY_HIDDEN_SIZE]),
                ("out_proj", [TINY_HIDDEN_SIZE, TINY_VALUE_DIM as usize]),
            ] {
                let layer = source
                    .load_linear(
                        &format!("model.language_model.layers.0.linear_attn.{projection}"),
                        &Device::Cpu,
                        Shard::default(),
                    )?
                    .unwrap();

                assert_eq!(layer.name(), "gguf", "{architecture} {projection}");
                let qweight = layer
                    .get_qtensor()
                    .unwrap_or_else(|| panic!("{architecture} {projection} did not stay packed"));
                assert_eq!(
                    qweight.dtype(),
                    GgmlDType::Q4K,
                    "{architecture} {projection}"
                );
                assert_eq!(
                    qweight.shape().dims(),
                    expected_shape,
                    "{architecture} {projection}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn qwen35_multimodal_config_selects_tiled_gdn() -> Result<()> {
        for loader_type in [
            MultimodalLoaderType::Qwen3_5,
            MultimodalLoaderType::Qwen3_5Moe,
        ] {
            let config = normalize_qwen_multimodal_config(
                &loader_type,
                r#"{"text_config":{"_mistralrs_gdn_v_head_layout":"grouped"}}"#,
            )?;
            let config: serde_json::Value = serde_json::from_str(&config)?;
            assert_eq!(config["text_config"][GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
        }
        Ok(())
    }

    #[test]
    fn qwen35_multimodal_config_uses_gguf_weights_over_hf_quantization_metadata() -> Result<()> {
        let quantization_config = r#"{"quant_method":"awq","bits":4,"group_size":128}"#;
        for (architecture, loader_type) in [
            ("qwen35", MultimodalLoaderType::Qwen3_5),
            ("qwen35moe", MultimodalLoaderType::Qwen3_5Moe),
        ] {
            let config = normalize_qwen_multimodal_config(
                &loader_type,
                &format!(
                    r#"{{"quantization_config":{quantization_config},"text_config":{{"quantization_config":{quantization_config}}}}}"#
                ),
            )?;
            let config: serde_json::Value = serde_json::from_str(&config)?;
            let top_level: Option<QuantizedConfig> =
                serde_json::from_value(config["quantization_config"].clone())?;
            let text: Option<QuantizedConfig> =
                serde_json::from_value(config["text_config"]["quantization_config"].clone())?;
            assert!(top_level.is_none(), "{architecture}");
            assert!(text.is_none(), "{architecture}");

            let (_file, archive) = tiny_qwen35_multimodal_archive(architecture)?;
            let bindings = build_qwen_multimodal_bindings(&archive)?;
            let source = Arc::new(GgufWeightSource::new(archive, &bindings, DType::F32)?);
            let vb = source
                .sharded_var_builder(Device::Cpu)
                .pp("model")
                .pp("language_model")
                .pp("layers")
                .pp(0)
                .pp("linear_attn")
                .pp("in_proj_qkv");
            let comm = Arc::new(Comm::from_device(Id::new(), &Device::Cpu, 0, 1)?);
            let layer = ColumnParallelLayer::new(
                TINY_HIDDEN_SIZE,
                TINY_QKV_ROWS,
                &top_level.or(text),
                false,
                &comm,
                vb,
            )?;
            assert_eq!(layer.name(), "gguf", "{architecture}");
        }
        Ok(())
    }

    #[test]
    fn qwen_multimodal_config_normalization_is_scoped() -> Result<()> {
        let config = r#"{"text_config":{}}"#;
        assert_eq!(
            normalize_qwen_multimodal_config(&MultimodalLoaderType::Qwen3VL, config)?,
            config
        );
        assert!(normalize_qwen_multimodal_config(
            &MultimodalLoaderType::Qwen3_5,
            r#"{"text_config":null}"#,
        )
        .is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen3.5 main and mmproj GGUF paths"]
    fn local_qwen35moe_archives_have_complete_native_bindings() -> Result<()> {
        let path = std::env::var("MISTRALRS_QWEN35_GGUF")
            .map_err(|_| anyhow::anyhow!("MISTRALRS_QWEN35_GGUF is not set"))?;
        let mmproj_path = std::env::var("MISTRALRS_QWEN35_MMPROJ_GGUF")
            .map_err(|_| anyhow::anyhow!("MISTRALRS_QWEN35_MMPROJ_GGUF is not set"))?;
        let mut archive = GgufArchive::open_file(path)?;
        archive.merge_component(GgufArchive::open_file(mmproj_path)?)?;
        let bindings = build_qwen_multimodal_bindings(&archive)?;
        assert_eq!(
            qwen_multimodal_loader_type(&archive)?,
            MultimodalLoaderType::Qwen3_5Moe
        );
        for target in [
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.experts.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.up_proj.weight",
            "model.language_model.layers.0.mlp.experts.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "model.language_model.layers.3.self_attn.v_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
            "model.language_model.layers.3.self_attn.q_norm.weight",
            "model.language_model.layers.3.self_attn.k_norm.weight",
        ] {
            assert!(bindings.get(target).is_some(), "{target}");
        }
        Ok(())
    }
}
