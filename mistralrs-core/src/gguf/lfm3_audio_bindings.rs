use anyhow::{bail, Result};
use candle_core::DType;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

use crate::NormalLoaderType;

use super::{
    lfm3_vl_bindings::{build_lfm3_vl_bindings, rebase_lfm3_text_bindings},
    multimodal_binding_utils::{
        bind, bind_required, bind_required_linear, bind_required_with, metadata_string,
        validate_architecture, TensorInventory,
    },
    normal_bindings::build_normal_bindings,
    normal_registry::CanonicalGgufArchitecture,
};

const FAMILY: &str = "LFM3 audio";
const DEPTH_EMBEDDINGS: &str = "a.gen.code.embd.weight";
const DEPTH_NORMS: &str = "a.gen.code.head_norm.weight";

pub(crate) fn build_lfm3_audio_bindings(
    archive: &GgufArchive,
    dtype: DType,
) -> Result<GgufBindingMap> {
    validate_architecture(archive, "lfm3")?;
    let text = build_normal_bindings(
        archive,
        &NormalLoaderType::Lfm3,
        CanonicalGgufArchitecture::Lfm3,
    )?;
    bind_audio(archive, rebase_lfm3_text_bindings(text), dtype)
}

pub(crate) fn build_lfm3_omni_bindings(
    archive: &GgufArchive,
    dtype: DType,
) -> Result<GgufBindingMap> {
    bind_audio(archive, build_lfm3_vl_bindings(archive, dtype)?, dtype)
}

pub(super) fn build_lfm3_audio_input_bindings(
    archive: &GgufArchive,
    mut base: GgufBindingMap,
    dtype: DType,
) -> Result<GgufBindingMap> {
    if metadata_string(archive, "clip.audio.projector_type")? != Some("lfm3a") {
        bail!("{FAMILY} GGUF requires `clip.audio.projector_type` to be `lfm3a`");
    }
    let inventory = TensorInventory::from_archive(archive);
    let mut bindings = GgufBindingMap::new();
    let downsample = "model.audio_encoder.downsample";
    for (target, index) in [
        ("conv_in", 0),
        ("conv_mid", 2),
        ("lin_mid", 3),
        ("conv_end", 5),
        ("lin_end", 6),
    ] {
        bind_required(
            &inventory,
            &mut bindings,
            format!("{downsample}.{target}.weight"),
            format!("a.conv1d.{index}.weight"),
        )?;
        bind_required_with(
            &inventory,
            &mut bindings,
            format!("{downsample}.{target}.bias"),
            format!("a.conv1d.{index}.bias"),
            |source, shape| {
                let [channels, 1, 1] = shape else {
                    bail!("{FAMILY} GGUF `{source}` must have shape [channels, 1, 1]");
                };
                Ok(GgufTensorBinding::tensor(source).reshape(vec![*channels]))
            },
        )?;
    }
    bind_required_with(
        &inventory,
        &mut bindings,
        format!("{downsample}.lin_out.weight"),
        "a.pre_encode.out.weight",
        |source, shape| {
            let [output, input] = shape else {
                bail!("{FAMILY} GGUF `{source}` must have rank 2");
            };
            Ok(GgufTensorBinding::tensor(source).reshape(vec![*output, *input, 1]))
        },
    )?;
    bind_required(
        &inventory,
        &mut bindings,
        format!("{downsample}.lin_out.bias"),
        "a.pre_encode.out.bias",
    )?;
    for (target, source) in [
        ("model.audio_encoder.lfm.embedding_norm", "a.post_ln"),
        ("model.audio_projector.layer_norm", "mm.a.mlp.0"),
        ("model.audio_projector.linear_1", "mm.a.mlp.1"),
        ("model.audio_projector.linear_2", "mm.a.mlp.3"),
    ] {
        bind_required_linear(&inventory, &mut bindings, target, source)?;
    }
    bind_stack(
        &inventory,
        &mut bindings,
        "model.audio_encoder.lfm",
        "a",
        Stack::Encoder,
    )?;
    // Audio modules load dense tensors even when their GGUF sources are quantized.
    for (native, binding) in bindings.iter() {
        base.insert(native, binding.clone().cast(dtype));
    }
    Ok(base)
}

fn bind_audio(archive: &GgufArchive, base: GgufBindingMap, dtype: DType) -> Result<GgufBindingMap> {
    let mut base = build_lfm3_audio_input_bindings(archive, base, dtype)?;
    if metadata_string(archive, "clip.gen.audio.projector_type")? != Some("lfm3a_gen") {
        bail!("{FAMILY} GGUF requires `clip.gen.audio.projector_type` to be `lfm3a_gen`");
    }
    let inventory = TensorInventory::from_archive(archive);
    let mut bindings = GgufBindingMap::new();
    for (target, source) in [
        ("model.depth_decoder.depth_linear", "a.gen.code.proj_in"),
        ("model.codebook_embedding", "a.gen.code.out_embd"),
        (
            "model.depth_decoder.model.embedding_norm",
            "a.gen.code.output_norm",
        ),
        ("quantizer.project_out", "a.gen.wav.quant.first.out_proj"),
        ("from_quantizer", "a.gen.wav.tfm.in_proj"),
        ("decoder.lin", "a.gen.wav.tfm.out_proj"),
        ("decoder.lfm.embedding_norm", "a.gen.wav.tfm.output_norm"),
    ] {
        bind_required_linear(&inventory, &mut bindings, target, source)?;
    }
    bind_required(
        &inventory,
        &mut bindings,
        "model.depth_decoder.mask_embedding",
        "a.gen.code.mask_embd",
    )?;
    let [codebooks, vocab, hidden] = inventory.shape(DEPTH_EMBEDDINGS)? else {
        bail!("{FAMILY} GGUF `{DEPTH_EMBEDDINGS}` must have rank 3");
    };
    if inventory.shape(DEPTH_NORMS)? != [*codebooks, *hidden] {
        bail!("{FAMILY} GGUF depth embedding and normalization shapes disagree");
    }
    for index in 0..*codebooks {
        let root = format!("model.depth_decoder.depth_embeddings.{index}");
        bindings.insert(
            format!("{root}.embedding.weight"),
            GgufTensorBinding::tensor(DEPTH_EMBEDDINGS)
                .slice(0, index, 1)
                .reshape(vec![*vocab, *hidden]),
        );
        bindings.insert(
            format!("{root}.embedding_norm.weight"),
            GgufTensorBinding::tensor(DEPTH_NORMS)
                .slice(0, index, 1)
                .reshape(vec![*hidden]),
        );
    }
    for (native, source, kind) in [
        ("model.depth_decoder.model", "a.gen.code", Stack::Depth),
        ("decoder.lfm", "a.gen.wav.tfm", Stack::Codec),
    ] {
        bind_stack(&inventory, &mut bindings, native, source, kind)?;
    }
    // Speech modules load dense tensors even when their GGUF sources are quantized.
    for (native, binding) in bindings.iter() {
        base.insert(native, binding.clone().cast(dtype));
    }
    Ok(base)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stack {
    Encoder,
    Depth,
    Codec,
}

fn bind_stack(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    native_root: &str,
    source_root: &str,
    kind: Stack,
) -> Result<()> {
    for layer in inventory.require_layers(&format!("{source_root}.blk."), FAMILY)? {
        let native = format!("{native_root}.layers.{layer}");
        let source = format!("{source_root}.blk.{layer}");
        if inventory.contains(&format!("{source}.conv_dw.weight")) {
            for (target, role) in [("conv.in_proj", "conv_pw1"), ("conv.out_proj", "conv_pw2")] {
                bind_required_linear(
                    inventory,
                    bindings,
                    &format!("{native}.{target}"),
                    &format!("{source}.{role}"),
                )?;
            }
            bind_required_with(
                inventory,
                bindings,
                format!("{native}.conv.conv.weight"),
                format!("{source}.conv_dw.weight"),
                |source, shape| {
                    let [channels, width] = shape else {
                        bail!("{FAMILY} GGUF `{source}` must have rank 2");
                    };
                    Ok(GgufTensorBinding::tensor(source).reshape(vec![*channels, 1, *width]))
                },
            )?;
            bind(
                inventory,
                bindings,
                format!("{native}.conv.conv.bias"),
                format!("{source}.conv_dw.bias"),
            );
        } else {
            let (output, q_norm, k_norm) = if kind == Stack::Encoder {
                ("out_proj", "q_layernorm", "k_layernorm")
            } else {
                ("o_proj", "q_norm", "k_norm")
            };
            for (target, role) in [
                ("q_proj", "attn_q"),
                ("k_proj", "attn_k"),
                ("v_proj", "attn_v"),
                (output, "attn_out"),
                (q_norm, "attn_q_norm"),
                (k_norm, "attn_k_norm"),
            ] {
                bind_required_linear(
                    inventory,
                    bindings,
                    &format!("{native}.self_attn.{target}"),
                    &format!("{source}.{role}"),
                )?;
            }
            if kind != Stack::Encoder {
                bind_required(
                    inventory,
                    bindings,
                    format!("{native}.self_attn.gate_proj.weight"),
                    format!("{source}.attn_gate.weight"),
                )?;
            }
        }
        for (target, role) in [
            ("operator_norm", "ln1"),
            ("ffn_norm", "ln2"),
            ("feed_forward.w1", "ffn_gate"),
            ("feed_forward.w3", "ffn_up"),
            ("feed_forward.w2", "ffn_down"),
        ] {
            bind_required(
                inventory,
                bindings,
                format!("{native}.{target}.weight"),
                format!("{source}.{role}.weight"),
            )?;
        }
        if kind == Stack::Codec {
            for (target, role) in [
                ("post_operator_norm", "attn_post_norm"),
                ("post_ffn_norm", "ffn_post_norm"),
            ] {
                bind_required(
                    inventory,
                    bindings,
                    format!("{native}.{target}.weight"),
                    format!("{source}.{role}.weight"),
                )?;
            }
        }
    }
    Ok(())
}
