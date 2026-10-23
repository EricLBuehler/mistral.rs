use std::collections::{BTreeSet, HashMap};

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

pub(super) const PROJECTOR_TYPE: &str = "clip.projector_type";
pub(super) const VISION_PROJECTOR_TYPE: &str = "clip.vision.projector_type";

pub(super) struct TensorInventory<'a> {
    shapes: HashMap<&'a str, &'a [usize]>,
}

impl<'a> TensorInventory<'a> {
    pub(super) fn from_archive(archive: &'a GgufArchive) -> Self {
        Self::new(
            archive
                .tensors()
                .iter()
                .map(|(name, info)| (name.as_str(), info.shape())),
        )
    }

    pub(super) fn new(tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>) -> Self {
        Self {
            shapes: tensors.into_iter().collect(),
        }
    }

    pub(super) fn contains(&self, name: &str) -> bool {
        self.shapes.contains_key(name)
    }

    pub(super) fn shape(&self, name: &str) -> Result<&[usize]> {
        self.shapes
            .get(name)
            .copied()
            .with_context(|| format!("cannot find GGUF tensor `{name}`"))
    }

    pub(super) fn layer_indices(&self, prefix: &str) -> BTreeSet<usize> {
        self.shapes
            .keys()
            .filter_map(|name| {
                name.strip_prefix(prefix)?
                    .split_once('.')?
                    .0
                    .parse::<usize>()
                    .ok()
            })
            .collect()
    }

    pub(super) fn require_layers(&self, prefix: &str, family: &str) -> Result<BTreeSet<usize>> {
        let layers = self.layer_indices(prefix);
        let Some(last) = layers.last().copied() else {
            bail!("{family} GGUF has no `{prefix}<layer>` tensors");
        };
        if layers.len() != last + 1 {
            bail!("{family} GGUF has a non-contiguous `{prefix}<layer>` tensor inventory");
        }
        Ok(layers)
    }
}

pub(super) fn validate_architecture(archive: &GgufArchive, expected: &str) -> Result<()> {
    let architecture = required_metadata_string(archive, "general.architecture")?;
    if architecture != expected {
        bail!("expected `{expected}` GGUF architecture, found `{architecture}`");
    }
    Ok(())
}

pub(super) fn validate_projector(archive: &GgufArchive, expected: &str) -> Result<()> {
    let projector = projector_type(archive)?
        .with_context(|| "GGUF vision projector metadata is required".to_string())?;
    if projector != expected {
        bail!("expected `{expected}` vision projector, found `{projector}`");
    }
    Ok(())
}

pub(super) fn projector_type(archive: &GgufArchive) -> Result<Option<&str>> {
    let standalone = metadata_string(archive, PROJECTOR_TYPE)?;
    let mixed = metadata_string(archive, VISION_PROJECTOR_TYPE)?;
    match (standalone, mixed) {
        (Some(left), Some(right)) if left != right => bail!(
            "GGUF projector metadata conflicts: `{PROJECTOR_TYPE}` is `{left}`, \
             `{VISION_PROJECTOR_TYPE}` is `{right}`"
        ),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(super) fn metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<Option<&'a str>> {
    match archive.metadata_value(key) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => Ok(None),
    }
}

fn required_metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<&'a str> {
    metadata_string(archive, key)?.with_context(|| format!("GGUF metadata `{key}` is required"))
}

pub(super) fn metadata_usize(archive: &GgufArchive, key: &str) -> Result<usize> {
    let value = archive
        .metadata_value(key)
        .with_context(|| format!("GGUF metadata `{key}` is required"))?;
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

pub(super) fn bind(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
) {
    let source = source.into();
    if inventory.contains(&source) {
        bindings.insert(native, GgufTensorBinding::tensor(source));
    }
}

pub(super) fn bind_required(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
) -> Result<()> {
    let source = source.into();
    inventory.shape(&source)?;
    bindings.insert(native, GgufTensorBinding::tensor(source));
    Ok(())
}

pub(super) fn bind_required_with(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
    make_binding: impl FnOnce(String, &[usize]) -> Result<GgufTensorBinding>,
) -> Result<()> {
    let source = source.into();
    let shape = inventory.shape(&source)?;
    bindings.insert(native, make_binding(source, shape)?);
    Ok(())
}

pub(super) fn bind_required_linear(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        format!("{native}.weight"),
        format!("{source}.weight"),
    )?;
    if inventory.contains(&format!("{source}.bias")) {
        bind_required(
            inventory,
            bindings,
            format!("{native}.bias"),
            format!("{source}.bias"),
        )?;
    }
    Ok(())
}

pub(super) fn bind_llama_text(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    model_prefix: &str,
    lm_head: &str,
    family: &str,
) -> Result<()> {
    bind_required(
        inventory,
        bindings,
        format!("{model_prefix}.embed_tokens.weight"),
        "token_embd.weight",
    )?;
    bind_required(
        inventory,
        bindings,
        format!("{model_prefix}.norm.weight"),
        "output_norm.weight",
    )?;
    bind(
        inventory,
        bindings,
        format!("{lm_head}.weight"),
        "output.weight",
    );
    bind(
        inventory,
        bindings,
        format!("{model_prefix}.rope_freqs.weight"),
        "rope_freqs.weight",
    );

    for layer in inventory.require_layers("blk.", family)? {
        let native = format!("{model_prefix}.layers.{layer}");
        let source = format!("blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_output"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
            )?;
        }
        bind_required(
            inventory,
            bindings,
            format!("{native}.input_layernorm.weight"),
            format!("{source}.attn_norm.weight"),
        )?;
        bind_required(
            inventory,
            bindings,
            format!("{native}.post_attention_layernorm.weight"),
            format!("{source}.ffn_norm.weight"),
        )?;
    }
    Ok(())
}

pub(super) fn inverse_llama_permute(
    source: String,
    shape: &[usize],
    heads: usize,
) -> Result<GgufTensorBinding> {
    if shape.is_empty() || heads == 0 || !shape[0].is_multiple_of(heads * 2) {
        bail!(
            "GGUF tensor `{source}` with shape {shape:?} cannot be inverse-permuted across {heads} heads"
        );
    }
    let pair_width = shape[0] / heads / 2;
    let mut reshaped = vec![heads, pair_width, 2];
    reshaped.extend_from_slice(&shape[1..]);
    let mut permutation = vec![0, 2, 1];
    permutation.extend(3..reshaped.len());
    Ok(GgufTensorBinding::tensor(source)
        .reshape(reshaped)
        .permute(permutation)
        .reshape(shape.to_vec()))
}

#[cfg(test)]
pub(super) fn binding_sources(bindings: &GgufBindingMap) -> BTreeSet<String> {
    fn collect(binding: &GgufTensorBinding, sources: &mut BTreeSet<String>) {
        match binding {
            GgufTensorBinding::Tensor(name)
            | GgufTensorBinding::Mxfp4Blocks(name)
            | GgufTensorBinding::Mxfp4Scales(name) => {
                sources.insert(name.clone());
            }
            GgufTensorBinding::Slice { input, .. }
            | GgufTensorBinding::Transpose { input, .. }
            | GgufTensorBinding::Permute { input, .. }
            | GgufTensorBinding::Reshape { input, .. }
            | GgufTensorBinding::Affine { input, .. }
            | GgufTensorBinding::Log { input }
            | GgufTensorBinding::InverseSoftplus { input }
            | GgufTensorBinding::Cast { input, .. } => collect(input, sources),
            GgufTensorBinding::Concat { inputs, .. }
            | GgufTensorBinding::Stack { inputs, .. }
            | GgufTensorBinding::Interleave { inputs, .. } => {
                for input in inputs {
                    collect(input, sources);
                }
            }
        }
    }

    let mut sources = BTreeSet::new();
    for (_, binding) in bindings.iter() {
        collect(binding, &mut sources);
    }
    sources
}
