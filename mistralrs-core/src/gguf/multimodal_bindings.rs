use anyhow::{bail, Result};
use candle_core::{quantized::gguf_file::Value, DType};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use std::collections::{BTreeSet, HashMap};

const VISION_PROJECTOR_TYPE: &str = "clip.vision.projector_type";
const AUDIO_PROJECTOR_TYPE: &str = "clip.audio.projector_type";
const CLIP_BOUNDS: [&str; 4] = ["input_min", "input_max", "output_min", "output_max"];

struct TensorInventory<'a> {
    shapes: HashMap<&'a str, &'a [usize]>,
}

impl<'a> TensorInventory<'a> {
    fn new(tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>) -> Self {
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
            .copied()
            .ok_or_else(|| anyhow::anyhow!("cannot find GGUF tensor `{name}`"))
    }

    fn layer_indices(&self, prefix: &str) -> BTreeSet<usize> {
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
}

pub(crate) fn build_gemma4_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    if let Some(architecture) = metadata_string(archive, "general.architecture")? {
        if architecture != "gemma4" {
            bail!("expected Gemma 4 GGUF architecture, found `{architecture}`");
        }
    }
    build_gemma4_bindings_from_inventory(
        archive
            .tensors()
            .iter()
            .map(|(name, info)| (name.as_str(), info.shape())),
        metadata_string(archive, VISION_PROJECTOR_TYPE)?,
        metadata_string(archive, AUDIO_PROJECTOR_TYPE)?,
    )
}

fn build_gemma4_bindings_from_inventory<'a>(
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    vision_projector: Option<&str>,
    audio_projector: Option<&str>,
) -> Result<GgufBindingMap> {
    let inventory = TensorInventory::new(tensors);
    let mut bindings = GgufBindingMap::new();
    bind_text(&inventory, &mut bindings)?;
    match vision_projector {
        Some("gemma4v") => bind_vision(&inventory, &mut bindings)?,
        Some("gemma4uv") => bind_unified_vision(&inventory, &mut bindings)?,
        Some(projector) => bail!("unsupported Gemma 4 vision projector `{projector}`"),
        None => {}
    }
    match audio_projector {
        Some("gemma4a") => bind_audio(&inventory, &mut bindings)?,
        Some("gemma4ua") => bind_unified_audio(&inventory, &mut bindings),
        Some(projector) => bail!("unsupported Gemma 4 audio projector `{projector}`"),
        None => {}
    }
    Ok(bindings)
}

fn metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<Option<&'a str>> {
    match archive.metadata_value(key) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => bail!("GGUF metadata `{key}` must be a string"),
        None => Ok(None),
    }
}

fn bind_text(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind(
        inventory,
        bindings,
        "model.language_model.embed_tokens.weight",
        "token_embd.weight",
    );
    bind(
        inventory,
        bindings,
        "model.language_model.norm.weight",
        "output_norm.weight",
    );
    bind(
        inventory,
        bindings,
        "model.language_model.lm_head.weight",
        "output.weight",
    );
    bind(
        inventory,
        bindings,
        "model.language_model.embed_tokens_per_layer.weight",
        "per_layer_token_embd.weight",
    );
    bind(
        inventory,
        bindings,
        "model.language_model.per_layer_model_projection.weight",
        "per_layer_model_proj.weight",
    );
    bind(
        inventory,
        bindings,
        "model.language_model.per_layer_projection_norm.weight",
        "per_layer_proj_norm.weight",
    );

    for layer in inventory.layer_indices("blk.") {
        let native = format!("model.language_model.layers.{layer}");
        let source = format!("blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj.weight", "attn_q.weight"),
            ("self_attn.k_proj.weight", "attn_k.weight"),
            ("self_attn.v_proj.weight", "attn_v.weight"),
            ("self_attn.o_proj.weight", "attn_output.weight"),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("mlp.gate_proj.weight", "ffn_gate.weight"),
            ("mlp.up_proj.weight", "ffn_up.weight"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
            ("input_layernorm.weight", "attn_norm.weight"),
            (
                "post_attention_layernorm.weight",
                "post_attention_norm.weight",
            ),
            ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
            ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
            ("per_layer_input_gate.weight", "inp_gate.weight"),
            ("per_layer_projection.weight", "proj.weight"),
            ("post_per_layer_input_norm.weight", "post_norm.weight"),
            ("layer_scalar", "layer_output_scale.weight"),
            (
                "pre_feedforward_layernorm_2.weight",
                "pre_ffw_norm_2.weight",
            ),
            (
                "post_feedforward_layernorm_1.weight",
                "post_ffw_norm_1.weight",
            ),
            (
                "post_feedforward_layernorm_2.weight",
                "post_ffw_norm_2.weight",
            ),
            ("router.proj.weight", "ffn_gate_inp.weight"),
            ("router.scale", "ffn_gate_inp.scale"),
            ("router.per_expert_scale", "ffn_down_exps.scale"),
        ] {
            bind(
                inventory,
                bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
        bind_experts(inventory, bindings, &native, &source)?;
    }
    Ok(())
}

fn bind_experts(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    let fused = format!("{source}.ffn_gate_up_exps.weight");
    let gate = format!("{source}.ffn_gate_exps.weight");
    let up = format!("{source}.ffn_up_exps.weight");
    let down = format!("{source}.ffn_down_exps.weight");
    let native_experts = format!("{native}.experts");

    if inventory.contains(&fused) {
        let shape = inventory.shape(&fused)?;
        if shape.len() != 3 || !shape[1].is_multiple_of(2) {
            bail!(
                "Gemma 4 fused expert tensor `{fused}` must have shape [experts, 2 * intermediate, hidden]"
            );
        }
        let intermediate = shape[1] / 2;
        let fused_binding = GgufTensorBinding::tensor(&fused);
        bindings.insert(
            format!("{native_experts}.gate_up_proj"),
            fused_binding.clone(),
        );
        bindings.insert(
            format!("{native_experts}.gate_proj.weight"),
            fused_binding.clone().slice(1, 0, intermediate),
        );
        bindings.insert(
            format!("{native_experts}.up_proj.weight"),
            fused_binding.slice(1, intermediate, intermediate),
        );
    } else if inventory.contains(&gate) && inventory.contains(&up) {
        bindings.insert(
            format!("{native_experts}.gate_up_proj"),
            GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor(&gate),
                    GgufTensorBinding::tensor(&up),
                ],
                1,
            ),
        );
        bindings.insert(
            format!("{native_experts}.gate_proj.weight"),
            GgufTensorBinding::tensor(&gate),
        );
        bindings.insert(
            format!("{native_experts}.up_proj.weight"),
            GgufTensorBinding::tensor(&up),
        );
    }

    if inventory.contains(&down) {
        let binding = GgufTensorBinding::tensor(&down);
        bindings.insert(format!("{native_experts}.down_proj"), binding.clone());
        bindings.insert(format!("{native_experts}.down_proj.weight"), binding);
    }
    Ok(())
}

fn bind_vision(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind(
        inventory,
        bindings,
        "model.embed_vision.embedding_projection.weight",
        "mm.input_projection.weight",
    );
    bind_nonunified_patch_embedding(inventory, bindings)?;
    bind(
        inventory,
        bindings,
        "model.vision_tower.patch_embedder.position_embedding_table",
        "v.position_embd.weight",
    );
    bind(
        inventory,
        bindings,
        "model.vision_tower.std_bias",
        "v.std_bias",
    );
    bind(
        inventory,
        bindings,
        "model.vision_tower.std_scale",
        "v.std_scale",
    );

    for layer in inventory.layer_indices("v.blk.") {
        let native = format!("model.vision_tower.encoder.layers.{layer}");
        let source = format!("v.blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_out"),
            ("mlp.gate_proj", "ffn_gate"),
            ("mlp.up_proj", "ffn_up"),
            ("mlp.down_proj", "ffn_down"),
        ] {
            bind_clippable(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
            );
        }
        for (target, role) in [
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("input_layernorm.weight", "ln1.weight"),
            ("post_attention_layernorm.weight", "attn_post_norm.weight"),
            ("pre_feedforward_layernorm.weight", "ln2.weight"),
            ("post_feedforward_layernorm.weight", "ffn_post_norm.weight"),
            ("layer_scalar", "out_scale.weight"),
        ] {
            bind(
                inventory,
                bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
    }
    Ok(())
}

fn bind_nonunified_patch_embedding(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let source = "v.patch_embd.weight";
    if !inventory.contains(source) {
        return Ok(());
    }
    let shape = inventory.shape(source)?;
    let &[output, channels, patch_h, patch_w] = shape else {
        bail!("Gemma 4 vision patch tensor `{source}` must be rank 4");
    };
    if channels != 3 || patch_h != patch_w {
        bail!("Gemma 4 vision patch tensor `{source}` must have shape [output, 3, patch, patch]");
    }
    bindings.insert(
        "model.vision_tower.patch_embedder.input_proj.weight",
        GgufTensorBinding::tensor(source)
            .permute(vec![0, 2, 3, 1])
            .reshape(vec![output, channels * patch_h * patch_w]),
    );
    Ok(())
}

fn bind_audio(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) -> Result<()> {
    bind_cast(
        inventory,
        bindings,
        "model.embed_audio.embedding_projection.weight",
        "mm.a.input_projection.weight",
        DType::F32,
    );
    for layer in 0..2 {
        bind(
            inventory,
            bindings,
            format!("model.audio_tower.subsample_conv_projection.layer{layer}.conv.weight"),
            format!("a.conv1d.{layer}.weight"),
        );
        bind(
            inventory,
            bindings,
            format!("model.audio_tower.subsample_conv_projection.layer{layer}.norm.weight"),
            format!("a.conv1d.{layer}.norm.weight"),
        );
    }
    bind_cast(
        inventory,
        bindings,
        "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
        "a.input_projection.weight",
        DType::F32,
    );
    bind_cast(
        inventory,
        bindings,
        "model.audio_tower.output_proj.weight",
        "a.pre_encode.out.weight",
        DType::F32,
    );
    bind_cast(
        inventory,
        bindings,
        "model.audio_tower.output_proj.bias",
        "a.pre_encode.out.bias",
        DType::F32,
    );

    for layer in inventory.layer_indices("a.blk.") {
        let native = format!("model.audio_tower.layers.{layer}");
        let source = format!("a.blk.{layer}");
        for (target, role) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.post", "attn_out"),
            ("feed_forward1.ffw_layer_1", "ffn_up"),
            ("feed_forward1.ffw_layer_2", "ffn_down"),
            ("feed_forward2.ffw_layer_1", "ffn_up_1"),
            ("feed_forward2.ffw_layer_2", "ffn_down_1"),
            ("lconv1d.linear_start", "conv_pw1"),
            ("lconv1d.linear_end", "conv_pw2"),
        ] {
            bind_clippable_cast(
                inventory,
                bindings,
                &format!("{native}.{target}"),
                &format!("{source}.{role}"),
                DType::F32,
            );
        }
        bind_cast(
            inventory,
            bindings,
            format!("{native}.self_attn.relative_k_proj.weight"),
            format!("{source}.attn_k_rel.weight"),
            DType::F32,
        );
        for (target, role) in [
            ("norm_pre_attn.weight", "attn_pre_norm.weight"),
            ("norm_post_attn.weight", "attn_post_norm.weight"),
            ("norm_out.weight", "ln2.weight"),
            ("feed_forward1.pre_layer_norm.weight", "ffn_norm.weight"),
            (
                "feed_forward1.post_layer_norm.weight",
                "ffn_post_norm.weight",
            ),
            ("feed_forward2.pre_layer_norm.weight", "ffn_norm_1.weight"),
            (
                "feed_forward2.post_layer_norm.weight",
                "ffn_post_norm_1.weight",
            ),
            ("lconv1d.pre_layer_norm.weight", "conv_norm.weight"),
            ("lconv1d.conv_norm.weight", "norm_conv.weight"),
        ] {
            bind(
                inventory,
                bindings,
                format!("{native}.{target}"),
                format!("{source}.{role}"),
            );
        }
        bind_inverse_softplus(
            inventory,
            bindings,
            format!("{native}.self_attn.per_dim_scale"),
            format!("{source}.per_dim_scale.weight"),
        );
        bind_inverse_softplus(
            inventory,
            bindings,
            format!("{native}.self_attn.per_dim_key_scale"),
            format!("{source}.per_dim_k_scale.weight"),
        );
        bind_audio_depthwise(inventory, bindings, &native, &source)?;
    }
    Ok(())
}

fn bind_audio_depthwise(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) -> Result<()> {
    let source = format!("{source}.conv_dw.weight");
    if !inventory.contains(&source) {
        return Ok(());
    }
    let shape = inventory.shape(&source)?;
    let &[output, kernel] = shape else {
        bail!("Gemma 4 audio depthwise tensor `{source}` must be rank 2");
    };
    bindings.insert(
        format!("{native}.lconv1d.depthwise_conv1d.weight"),
        GgufTensorBinding::tensor(source).reshape(vec![output, 1, kernel]),
    );
    Ok(())
}

fn bind_unified_vision(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    bind(
        inventory,
        bindings,
        "model.embed_vision.embedding_projection.weight",
        "mm.input_projection.weight",
    );
    bind_unified_patch_weight(inventory, bindings)?;
    bind(
        inventory,
        bindings,
        "model.vision_embedder.patch_dense.bias",
        "v.patch_embd.bias",
    );
    bind_unified_patch_norm_1(inventory, bindings, "weight")?;
    bind_unified_patch_norm_1(inventory, bindings, "bias")?;
    for suffix in ["weight", "bias"] {
        bind(
            inventory,
            bindings,
            format!("model.vision_embedder.patch_ln2.{suffix}"),
            format!("v.patch_norm.2.{suffix}"),
        );
        bind(
            inventory,
            bindings,
            format!("model.vision_embedder.pos_norm.{suffix}"),
            format!("v.patch_norm.3.{suffix}"),
        );
    }
    if inventory.contains("v.position_embd.weight") {
        bindings.insert(
            "model.vision_embedder.pos_embedding",
            GgufTensorBinding::tensor("v.position_embd.weight").permute(vec![1, 0, 2]),
        );
    }
    Ok(())
}

fn bind_unified_patch_weight(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let source = "v.patch_embd.weight";
    if !inventory.contains(source) {
        return Ok(());
    }
    let shape = inventory.shape(source)?;
    let &[output, input] = shape else {
        bail!("Gemma 4 unified vision patch tensor `{source}` must be rank 2");
    };
    let patch = patch_size(input, source)?;
    bindings.insert(
        "model.vision_embedder.patch_dense.weight",
        GgufTensorBinding::tensor(source)
            .reshape(vec![output, 3, patch, patch])
            .permute(vec![0, 2, 3, 1])
            .reshape(vec![output, input]),
    );
    Ok(())
}

fn bind_unified_patch_norm_1(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    suffix: &str,
) -> Result<()> {
    let source = format!("v.patch_norm.1.{suffix}");
    if !inventory.contains(&source) {
        return Ok(());
    }
    let shape = inventory.shape(&source)?;
    let &[input] = shape else {
        bail!("Gemma 4 unified vision patch norm tensor `{source}` must be rank 1");
    };
    let patch = patch_size(input, &source)?;
    bindings.insert(
        format!("model.vision_embedder.patch_ln1.{suffix}"),
        GgufTensorBinding::tensor(source)
            .reshape(vec![3, patch, patch])
            .permute(vec![1, 2, 0])
            .reshape(vec![input]),
    );
    Ok(())
}

fn bind_unified_audio(inventory: &TensorInventory<'_>, bindings: &mut GgufBindingMap) {
    bind_cast(
        inventory,
        bindings,
        "model.embed_audio.embedding_projection.weight",
        "mm.a.input_projection.weight",
        DType::F32,
    );
}

fn patch_size(input: usize, source: &str) -> Result<usize> {
    if !input.is_multiple_of(3) {
        bail!("Gemma 4 vision tensor `{source}` input dimension must be divisible by 3");
    }
    let area = input / 3;
    let patch = area.isqrt();
    if patch * patch != area {
        bail!("Gemma 4 vision tensor `{source}` input dimension is not 3 * patch * patch");
    }
    Ok(patch)
}

fn bind_clippable(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
) {
    bind(
        inventory,
        bindings,
        format!("{native}.linear.weight"),
        format!("{source}.weight"),
    );
    for suffix in CLIP_BOUNDS {
        let source_name = format!("{source}.{suffix}");
        if inventory.contains(&source_name) {
            bindings.insert(
                format!("{native}.{suffix}"),
                GgufTensorBinding::tensor(source_name).reshape(vec![]),
            );
        }
    }
}

fn bind_clippable_cast(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: &str,
    source: &str,
    dtype: DType,
) {
    bind_cast(
        inventory,
        bindings,
        format!("{native}.linear.weight"),
        format!("{source}.weight"),
        dtype,
    );
    for suffix in CLIP_BOUNDS {
        let source_name = format!("{source}.{suffix}");
        if inventory.contains(&source_name) {
            bindings.insert(
                format!("{native}.{suffix}"),
                GgufTensorBinding::tensor(source_name).reshape(vec![]),
            );
        }
    }
}

fn bind_inverse_softplus(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
) {
    let source = source.into();
    if inventory.contains(&source) {
        bindings.insert(
            native,
            GgufTensorBinding::tensor(source)
                .cast(DType::F32)
                .inverse_softplus(),
        );
    }
}

fn bind(
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

fn bind_cast(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native: impl Into<String>,
    source: impl Into<String>,
    dtype: DType,
) {
    let source = source.into();
    if inventory.contains(&source) {
        bindings.insert(native, GgufTensorBinding::tensor(source).cast(dtype));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn add(shapes: &mut HashMap<String, Vec<usize>>, name: impl Into<String>, shape: &[usize]) {
        shapes.insert(name.into(), shape.to_vec());
    }

    fn build_fixture(
        shapes: &HashMap<String, Vec<usize>>,
        vision_projector: Option<&str>,
        audio_projector: Option<&str>,
    ) -> Result<GgufBindingMap> {
        build_gemma4_bindings_from_inventory(
            shapes
                .iter()
                .map(|(name, shape)| (name.as_str(), shape.as_slice())),
            vision_projector,
            audio_projector,
        )
    }

    fn e4b_source_inventory() -> HashMap<String, Vec<usize>> {
        let mut shapes = HashMap::new();
        for name in [
            "token_embd.weight",
            "output_norm.weight",
            "per_layer_model_proj.weight",
            "per_layer_proj_norm.weight",
            "per_layer_token_embd.weight",
            "rope_freqs.weight",
        ] {
            add(&mut shapes, name, &[1]);
        }
        for layer in 0..42 {
            for role in [
                "attn_k.weight",
                "attn_k_norm.weight",
                "attn_norm.weight",
                "attn_output.weight",
                "attn_q.weight",
                "attn_q_norm.weight",
                "attn_v.weight",
                "ffn_down.weight",
                "ffn_gate.weight",
                "ffn_norm.weight",
                "ffn_up.weight",
                "inp_gate.weight",
                "layer_output_scale.weight",
                "post_attention_norm.weight",
                "post_ffw_norm.weight",
                "post_norm.weight",
                "proj.weight",
            ] {
                add(&mut shapes, format!("blk.{layer}.{role}"), &[1]);
            }
        }

        add(&mut shapes, "mm.input_projection.weight", &[1, 1]);
        add(&mut shapes, "v.patch_embd.weight", &[768, 3, 16, 16]);
        add(&mut shapes, "v.position_embd.weight", &[2, 10240, 768]);
        for layer in 0..16 {
            for role in [
                "attn_k", "attn_out", "attn_q", "attn_v", "ffn_down", "ffn_gate", "ffn_up",
            ] {
                add(&mut shapes, format!("v.blk.{layer}.{role}.weight"), &[1, 1]);
                for suffix in CLIP_BOUNDS {
                    add(&mut shapes, format!("v.blk.{layer}.{role}.{suffix}"), &[1]);
                }
            }
            for role in [
                "attn_k_norm.weight",
                "attn_post_norm.weight",
                "attn_q_norm.weight",
                "ffn_post_norm.weight",
                "ln1.weight",
                "ln2.weight",
            ] {
                add(&mut shapes, format!("v.blk.{layer}.{role}"), &[1]);
            }
        }

        for name in [
            "mm.a.input_projection.weight",
            "a.conv1d.0.norm.weight",
            "a.conv1d.0.weight",
            "a.conv1d.1.norm.weight",
            "a.conv1d.1.weight",
            "a.input_projection.weight",
            "a.pre_encode.out.bias",
            "a.pre_encode.out.weight",
        ] {
            add(&mut shapes, name, &[1, 1]);
        }
        for layer in 0..12 {
            for role in [
                "attn_k",
                "attn_out",
                "attn_q",
                "attn_v",
                "conv_pw1",
                "conv_pw2",
                "ffn_down",
                "ffn_down_1",
                "ffn_up",
                "ffn_up_1",
            ] {
                add(&mut shapes, format!("a.blk.{layer}.{role}.weight"), &[1, 1]);
                for suffix in CLIP_BOUNDS {
                    add(&mut shapes, format!("a.blk.{layer}.{role}.{suffix}"), &[1]);
                }
            }
            for role in [
                "attn_k_rel.weight",
                "attn_post_norm.weight",
                "attn_pre_norm.weight",
                "conv_norm.weight",
                "ffn_norm.weight",
                "ffn_norm_1.weight",
                "ffn_post_norm.weight",
                "ffn_post_norm_1.weight",
                "ln2.weight",
                "norm_conv.weight",
                "per_dim_scale.weight",
            ] {
                add(&mut shapes, format!("a.blk.{layer}.{role}"), &[1]);
            }
            add(
                &mut shapes,
                format!("a.blk.{layer}.conv_dw.weight"),
                &[1024, 5],
            );
        }
        shapes
    }

    fn e4b_native_inventory() -> HashSet<String> {
        let mut names = HashSet::new();
        for name in [
            "model.language_model.embed_tokens.weight",
            "model.language_model.embed_tokens_per_layer.weight",
            "model.language_model.norm.weight",
            "model.language_model.per_layer_model_projection.weight",
            "model.language_model.per_layer_projection_norm.weight",
            "model.embed_vision.embedding_projection.weight",
            "model.vision_tower.patch_embedder.input_proj.weight",
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.embed_audio.embedding_projection.weight",
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight",
            "model.audio_tower.subsample_conv_projection.layer0.norm.weight",
            "model.audio_tower.subsample_conv_projection.layer1.conv.weight",
            "model.audio_tower.subsample_conv_projection.layer1.norm.weight",
            "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
            "model.audio_tower.output_proj.weight",
            "model.audio_tower.output_proj.bias",
        ] {
            names.insert(name.to_string());
        }
        for layer in 0..42 {
            for role in [
                "input_layernorm.weight",
                "layer_scalar",
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "per_layer_input_gate.weight",
                "per_layer_projection.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "post_per_layer_input_norm.weight",
                "pre_feedforward_layernorm.weight",
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
            ] {
                names.insert(format!("model.language_model.layers.{layer}.{role}"));
            }
        }
        for layer in 0..16 {
            for role in [
                "self_attn.k_proj",
                "self_attn.o_proj",
                "self_attn.q_proj",
                "self_attn.v_proj",
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
            ] {
                names.insert(format!(
                    "model.vision_tower.encoder.layers.{layer}.{role}.linear.weight"
                ));
                for suffix in CLIP_BOUNDS {
                    names.insert(format!(
                        "model.vision_tower.encoder.layers.{layer}.{role}.{suffix}"
                    ));
                }
            }
            for role in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "self_attn.k_norm.weight",
                "self_attn.q_norm.weight",
            ] {
                names.insert(format!("model.vision_tower.encoder.layers.{layer}.{role}"));
            }
        }
        for layer in 0..12 {
            for role in [
                "self_attn.k_proj",
                "self_attn.post",
                "self_attn.q_proj",
                "self_attn.v_proj",
                "feed_forward1.ffw_layer_1",
                "feed_forward1.ffw_layer_2",
                "feed_forward2.ffw_layer_1",
                "feed_forward2.ffw_layer_2",
                "lconv1d.linear_start",
                "lconv1d.linear_end",
            ] {
                names.insert(format!(
                    "model.audio_tower.layers.{layer}.{role}.linear.weight"
                ));
                for suffix in CLIP_BOUNDS {
                    names.insert(format!("model.audio_tower.layers.{layer}.{role}.{suffix}"));
                }
            }
            for role in [
                "feed_forward1.post_layer_norm.weight",
                "feed_forward1.pre_layer_norm.weight",
                "feed_forward2.post_layer_norm.weight",
                "feed_forward2.pre_layer_norm.weight",
                "lconv1d.conv_norm.weight",
                "lconv1d.depthwise_conv1d.weight",
                "lconv1d.pre_layer_norm.weight",
                "norm_out.weight",
                "norm_post_attn.weight",
                "norm_pre_attn.weight",
                "self_attn.per_dim_scale",
                "self_attn.relative_k_proj.weight",
            ] {
                names.insert(format!("model.audio_tower.layers.{layer}.{role}"));
            }
        }
        names
    }

    fn collect_sources(binding: &GgufTensorBinding, sources: &mut HashSet<String>) {
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
            | GgufTensorBinding::Cast { input, .. } => collect_sources(input, sources),
            GgufTensorBinding::Concat { inputs, .. }
            | GgufTensorBinding::Stack { inputs, .. }
            | GgufTensorBinding::Interleave { inputs, .. } => {
                for input in inputs {
                    collect_sources(input, sources);
                }
            }
        }
    }

    #[test]
    fn e4b_main_and_mmproj_inventory_is_exhaustive() -> Result<()> {
        let sources = e4b_source_inventory();
        assert_eq!(sources.len(), 2131);
        let bindings = build_fixture(&sources, Some("gemma4v"), Some("gemma4a"))?;
        let actual = bindings
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<HashSet<_>>();
        let expected = e4b_native_inventory();
        assert_eq!(expected.len(), 2130);
        assert_eq!(actual, expected);

        let mut referenced = HashSet::new();
        for (_, binding) in bindings.iter() {
            collect_sources(binding, &mut referenced);
        }
        let mut expected_sources = sources.keys().cloned().collect::<HashSet<_>>();
        expected_sources.remove("rope_freqs.weight");
        assert_eq!(referenced, expected_sources);

        assert_eq!(
            bindings.get("model.vision_tower.patch_embedder.input_proj.weight"),
            Some(
                &GgufTensorBinding::tensor("v.patch_embd.weight")
                    .permute(vec![0, 2, 3, 1])
                    .reshape(vec![768, 768])
            )
        );
        assert_eq!(
            bindings.get("model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"),
            Some(&GgufTensorBinding::tensor("a.blk.0.conv_dw.weight").reshape(vec![1024, 1, 5]))
        );
        assert_eq!(
            bindings.get("model.audio_tower.layers.0.self_attn.per_dim_scale"),
            Some(
                &GgufTensorBinding::tensor("a.blk.0.per_dim_scale.weight")
                    .cast(DType::F32)
                    .inverse_softplus()
            )
        );
        for (native, source) in [
            (
                "model.embed_audio.embedding_projection.weight",
                "mm.a.input_projection.weight",
            ),
            (
                "model.audio_tower.subsample_conv_projection.input_proj_linear.weight",
                "a.input_projection.weight",
            ),
            (
                "model.audio_tower.output_proj.weight",
                "a.pre_encode.out.weight",
            ),
            (
                "model.audio_tower.output_proj.bias",
                "a.pre_encode.out.bias",
            ),
            (
                "model.audio_tower.layers.0.self_attn.q_proj.linear.weight",
                "a.blk.0.attn_q.weight",
            ),
            (
                "model.audio_tower.layers.0.self_attn.relative_k_proj.weight",
                "a.blk.0.attn_k_rel.weight",
            ),
        ] {
            assert_eq!(
                bindings.get(native),
                Some(&GgufTensorBinding::tensor(source).cast(DType::F32))
            );
        }
        assert_eq!(
            bindings.get("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_min"),
            Some(&GgufTensorBinding::tensor("v.blk.0.attn_q.input_min").reshape(vec![]))
        );
        Ok(())
    }

    #[test]
    fn moe_inventory_exposes_fast_backend_weights() -> Result<()> {
        let mut shapes = HashMap::new();
        add(
            &mut shapes,
            "blk.0.ffn_gate_up_exps.weight",
            &[128, 1408, 2816],
        );
        add(&mut shapes, "blk.0.ffn_down_exps.weight", &[128, 2816, 704]);
        for name in [
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_inp.scale",
            "blk.0.ffn_down_exps.scale",
            "blk.0.pre_ffw_norm_2.weight",
            "blk.0.post_ffw_norm_1.weight",
            "blk.0.post_ffw_norm_2.weight",
        ] {
            add(&mut shapes, name, &[1]);
        }
        let bindings = build_fixture(&shapes, None, None)?;
        let prefix = "model.language_model.layers.0";
        assert_eq!(
            bindings.get(&format!("{prefix}.experts.gate_up_proj")),
            Some(&GgufTensorBinding::tensor("blk.0.ffn_gate_up_exps.weight"))
        );
        assert_eq!(
            bindings.get(&format!("{prefix}.experts.gate_proj.weight")),
            Some(&GgufTensorBinding::tensor("blk.0.ffn_gate_up_exps.weight").slice(1, 0, 704))
        );
        assert_eq!(
            bindings.get(&format!("{prefix}.experts.up_proj.weight")),
            Some(&GgufTensorBinding::tensor("blk.0.ffn_gate_up_exps.weight").slice(1, 704, 704))
        );
        assert_eq!(
            bindings.get(&format!("{prefix}.experts.down_proj.weight")),
            Some(&GgufTensorBinding::tensor("blk.0.ffn_down_exps.weight"))
        );
        for name in [
            "router.proj.weight",
            "router.scale",
            "router.per_expert_scale",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ] {
            assert!(bindings.get(&format!("{prefix}.{name}")).is_some());
        }
        Ok(())
    }

    #[test]
    fn separate_expert_inventory_builds_fused_native_alias() -> Result<()> {
        let mut shapes = HashMap::new();
        for name in [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ] {
            add(&mut shapes, name, &[8, 16, 32]);
        }
        let bindings = build_fixture(&shapes, None, None)?;
        assert_eq!(
            bindings.get("model.language_model.layers.0.experts.gate_up_proj"),
            Some(&GgufTensorBinding::concat(
                vec![
                    GgufTensorBinding::tensor("blk.0.ffn_gate_exps.weight"),
                    GgufTensorBinding::tensor("blk.0.ffn_up_exps.weight"),
                ],
                1
            ))
        );
        assert!(bindings
            .get("model.language_model.layers.0.experts.gate_proj.weight")
            .is_some());
        assert!(bindings
            .get("model.language_model.layers.0.experts.up_proj.weight")
            .is_some());
        assert!(bindings
            .get("model.language_model.layers.0.experts.down_proj.weight")
            .is_some());
        Ok(())
    }

    #[test]
    fn unified_mmproj_inventory_is_exhaustive() -> Result<()> {
        let mut shapes = HashMap::new();
        add(&mut shapes, "mm.a.input_projection.weight", &[3840, 640]);
        add(&mut shapes, "mm.input_projection.weight", &[3840, 3840]);
        add(&mut shapes, "v.patch_embd.bias", &[3840]);
        add(&mut shapes, "v.patch_embd.weight", &[3840, 6912]);
        add(&mut shapes, "v.position_embd.weight", &[2, 1120, 3840]);
        for suffix in ["weight", "bias"] {
            add(&mut shapes, format!("v.patch_norm.1.{suffix}"), &[6912]);
            add(&mut shapes, format!("v.patch_norm.2.{suffix}"), &[3840]);
            add(&mut shapes, format!("v.patch_norm.3.{suffix}"), &[3840]);
        }
        assert_eq!(shapes.len(), 11);
        let bindings = build_fixture(&shapes, Some("gemma4uv"), Some("gemma4ua"))?;
        let expected = [
            "model.embed_audio.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "model.vision_embedder.patch_dense.bias",
            "model.vision_embedder.patch_dense.weight",
            "model.vision_embedder.patch_ln1.bias",
            "model.vision_embedder.patch_ln1.weight",
            "model.vision_embedder.patch_ln2.bias",
            "model.vision_embedder.patch_ln2.weight",
            "model.vision_embedder.pos_embedding",
            "model.vision_embedder.pos_norm.bias",
            "model.vision_embedder.pos_norm.weight",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<HashSet<_>>();
        let actual = bindings
            .iter()
            .map(|(name, _)| name.to_string())
            .collect::<HashSet<_>>();
        assert_eq!(actual, expected);
        assert_eq!(
            bindings.get("model.embed_audio.embedding_projection.weight"),
            Some(&GgufTensorBinding::tensor("mm.a.input_projection.weight").cast(DType::F32))
        );
        assert_eq!(
            bindings.get("model.vision_embedder.patch_dense.weight"),
            Some(
                &GgufTensorBinding::tensor("v.patch_embd.weight")
                    .reshape(vec![3840, 3, 48, 48])
                    .permute(vec![0, 2, 3, 1])
                    .reshape(vec![3840, 6912])
            )
        );
        assert_eq!(
            bindings.get("model.vision_embedder.patch_ln1.weight"),
            Some(
                &GgufTensorBinding::tensor("v.patch_norm.1.weight")
                    .reshape(vec![3, 48, 48])
                    .permute(vec![1, 2, 0])
                    .reshape(vec![6912])
            )
        );
        assert_eq!(
            bindings.get("model.vision_embedder.pos_embedding"),
            Some(&GgufTensorBinding::tensor("v.position_embd.weight").permute(vec![1, 0, 2]))
        );
        Ok(())
    }
}
