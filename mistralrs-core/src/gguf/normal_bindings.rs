use super::normal_registry::CanonicalGgufArchitecture;
use crate::NormalLoaderType;
use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file::Value, DType};
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

pub(crate) fn build_normal_bindings(
    archive: &GgufArchive,
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
) -> Result<GgufBindingMap> {
    if matches!(loader, NormalLoaderType::Qwen3_5) {
        return super::qwen_multimodal_bindings::build_qwen35_text_bindings(archive);
    }
    let mut bindings = GgufBindingMap::new();
    let block_count = native_block_count(archive, architecture)?;
    bind_root_tensors(archive, loader, architecture, &mut bindings);
    for name in archive.tensors().keys() {
        let Some((layer, role, suffix)) = parse_block_tensor(name) else {
            continue;
        };
        if layer >= block_count {
            continue;
        }
        bind_block_tensor(
            archive,
            loader,
            architecture,
            layer,
            role,
            suffix,
            name,
            &mut bindings,
        )?;
    }
    bind_composite_tensors(archive, loader, architecture, block_count, &mut bindings)?;
    Ok(bindings)
}

fn bind_root_tensors(
    archive: &GgufArchive,
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
    bindings: &mut GgufBindingMap,
) {
    bind(
        archive,
        bindings,
        "model.embed_tokens.weight",
        "token_embd.weight",
    );
    bind(
        archive,
        bindings,
        "model.rope_freqs.weight",
        "rope_freqs.weight",
    );
    bind(
        archive,
        bindings,
        "model.rope_factors_long.weight",
        "rope_factors_long.weight",
    );
    bind(
        archive,
        bindings,
        "model.rope_factors_short.weight",
        "rope_factors_short.weight",
    );
    bind(archive, bindings, "lm_head.weight", "output.weight");
    bind(archive, bindings, "lm_head.bias", "output.bias");
    match loader {
        NormalLoaderType::Phi2 => {
            bind(
                archive,
                bindings,
                "model.final_layernorm.weight",
                "output_norm.weight",
            );
            bind(
                archive,
                bindings,
                "model.final_layernorm.bias",
                "output_norm.bias",
            );
        }
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => bind(
            archive,
            bindings,
            "model.embedding_norm.weight",
            "token_embd_norm.weight",
        ),
        _ if archive.contains_tensor("output_norm.weight") => {
            bindings.insert(
                "model.norm.weight",
                norm_binding(
                    architecture,
                    "output_norm.weight",
                    GgufTensorBinding::tensor("output_norm.weight"),
                ),
            );
            bind(archive, bindings, "model.norm.bias", "output_norm.bias");
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_block_tensor(
    archive: &GgufArchive,
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
    layer: usize,
    role: &str,
    suffix: &str,
    source: &str,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let p = format!("model.layers.{layer}");
    let gpt_oss_mxfp4_expert = matches!(loader, NormalLoaderType::GptOss)
        && matches!(role, "ffn_gate_exps" | "ffn_up_exps" | "ffn_down_exps")
        && archive.tensor_info(source)?.dtype().raw() == 39;
    if role == "ffn_gate_up_exps" {
        if matches!(loader, NormalLoaderType::GraniteMoeHybrid) {
            bindings.insert(
                format!("{p}.block_sparse_moe.input_linear.{suffix}"),
                GgufTensorBinding::tensor(source),
            );
        } else {
            bind_fused_expert_gate_up(archive, loader, &p, suffix, source, bindings)?;
        }
        return Ok(());
    }
    let target = match role {
        "attn_norm" => Some(match loader {
            NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
                format!("{p}.operator_norm.{suffix}")
            }
            _ => format!("{p}.input_layernorm.{suffix}"),
        }),
        "ffn_norm" => Some(match loader {
            NormalLoaderType::Gemma2 => format!("{p}.pre_feedforward_layernorm.{suffix}"),
            NormalLoaderType::GLM4 => format!("{p}.post_attention_layernorm.{suffix}"),
            NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
                format!("{p}.ffn_norm.{suffix}")
            }
            _ => format!("{p}.post_attention_layernorm.{suffix}"),
        }),
        "post_attention_norm" => Some(match loader {
            NormalLoaderType::GLM4 => format!("{p}.post_self_attn_layernorm.{suffix}"),
            _ => format!("{p}.post_attention_layernorm.{suffix}"),
        }),
        "post_ffw_norm" => Some(match loader {
            NormalLoaderType::Gemma2 => format!("{p}.post_feedforward_layernorm.{suffix}"),
            NormalLoaderType::GLM4 => format!("{p}.post_mlp_layernorm.{suffix}"),
            _ => return Ok(()),
        }),
        "attn_q" if !matches!(loader, NormalLoaderType::Phi3) => {
            Some(format!("{p}.self_attn.q_proj.{suffix}"))
        }
        "attn_k" if !matches!(loader, NormalLoaderType::Phi3) => {
            Some(format!("{p}.self_attn.k_proj.{suffix}"))
        }
        "attn_v" if !matches!(loader, NormalLoaderType::Phi3) => {
            Some(format!("{p}.self_attn.v_proj.{suffix}"))
        }
        "attn_qkv" if matches!(loader, NormalLoaderType::Phi3) => {
            Some(format!("{p}.self_attn.qkv_proj.{suffix}"))
        }
        "attn_output" => Some(format!(
            "{p}.self_attn.{}.{suffix}",
            if matches!(loader, NormalLoaderType::Phi2) {
                "dense"
            } else if matches!(loader, NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe) {
                "out_proj"
            } else {
                "o_proj"
            }
        )),
        "attn_q_norm" => Some(format!("{p}.self_attn.{}.{suffix}", q_norm_name(loader))),
        "attn_k_norm" => Some(format!("{p}.self_attn.{}.{suffix}", k_norm_name(loader))),
        "attn_sinks" => Some(format!("{p}.self_attn.sinks")),
        "ffn_gate" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            dense_mlp_target(loader, &p, "gate", suffix)
        }
        "ffn_up" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            dense_mlp_target(loader, &p, "up", suffix)
        }
        "ffn_down" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            dense_mlp_target(loader, &p, "down", suffix)
        }
        "ffn_gate_inp" => Some(router_target(loader, &p, suffix)),
        "ffn_gate_exps"
            if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) && !gpt_oss_mxfp4_expert =>
        {
            Some(expert_target(loader, &p, "gate", suffix))
        }
        "ffn_up_exps"
            if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) && !gpt_oss_mxfp4_expert =>
        {
            Some(expert_target(loader, &p, "up", suffix))
        }
        "ffn_down_exps"
            if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) && !gpt_oss_mxfp4_expert =>
        {
            Some(expert_target(loader, &p, "down", suffix))
        }
        "ffn_gate_shexp" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            Some(shared_expert_target(loader, &p, "gate", suffix))
        }
        "ffn_up_shexp" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            Some(shared_expert_target(loader, &p, "up", suffix))
        }
        "ffn_down_shexp" if !matches!(loader, NormalLoaderType::GraniteMoeHybrid) => {
            Some(shared_expert_target(loader, &p, "down", suffix))
        }
        "ffn_gate_inp_shexp" => Some(format!("{p}.mlp.shared_expert_gate.{suffix}")),
        "exp_probs_b" => Some(match loader {
            NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
                format!("{p}.feed_forward.expert_bias")
            }
            _ => format!("{p}.mlp.gate.e_score_correction_bias"),
        }),
        "attn_q_a" => Some(format!("{p}.self_attn.q_a_proj.{suffix}")),
        "attn_kv_b" => Some(format!("{p}.self_attn.kv_b_proj.{suffix}")),
        "attn_q_b" => Some(format!("{p}.self_attn.q_b_proj.{suffix}")),
        "attn_q_a_norm" => Some(format!("{p}.self_attn.q_a_layernorm.{suffix}")),
        "attn_kv_a_mqa" => Some(format!("{p}.self_attn.kv_a_proj_with_mqa.{suffix}")),
        "attn_kv_a_norm" => Some(format!("{p}.self_attn.kv_a_layernorm.{suffix}")),
        "shortconv.in_proj" => Some(format!("{p}.conv.in_proj.{suffix}")),
        "shortconv.out_proj" => Some(format!("{p}.conv.out_proj.{suffix}")),
        "shortconv.conv" => Some(format!("{p}.conv.conv.{suffix}")),
        "ssm_in" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.in_proj.{suffix}"))
        }
        "ssm_conv1d" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.conv1d.{suffix}"))
        }
        "ssm_dt" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.dt_bias"))
        }
        "ssm_a" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.A_log"))
        }
        "ssm_d" if !matches!(loader, NormalLoaderType::Qwen3Next) => Some(format!("{p}.mamba.D")),
        "ssm_norm" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.norm.{suffix}"))
        }
        "ssm_out" if !matches!(loader, NormalLoaderType::Qwen3Next) => {
            Some(format!("{p}.mamba.out_proj.{suffix}"))
        }
        _ => None,
    };
    if let Some(target) = target {
        let mut binding = GgufTensorBinding::tensor(source);
        if role.contains("norm") || target.contains("layernorm") {
            binding = norm_binding(architecture, source, binding);
        }
        if role == "ffn_gate_inp" && matches!(loader, NormalLoaderType::HunYuanMoEV1) {
            binding = binding.cast(DType::F32);
        }
        if role == "ffn_gate_inp_shexp" && matches!(loader, NormalLoaderType::Qwen3Next) {
            let shape = archive.tensor_info(source)?.shape();
            if shape.len() == 1 {
                binding = binding.reshape(vec![1, shape[0]]);
            }
        }
        if role == "shortconv.conv" && suffix == "weight" {
            let shape = archive.tensor_info(source)?.shape();
            if shape.len() == 2 {
                binding = binding.reshape(vec![shape[0], 1, shape[1]]);
            }
        }
        if matches!(role, "ssm_a" | "ssm_d" | "ssm_norm" | "ssm_conv1d")
            && matches!(
                architecture,
                CanonicalGgufArchitecture::Granite
                    | CanonicalGgufArchitecture::GraniteMoe
                    | CanonicalGgufArchitecture::GraniteHybrid
            )
        {
            binding = granite_ssm_binding(archive, role, source, binding)?;
        }
        bindings.insert(target, binding);
    }
    Ok(())
}

fn bind_composite_tensors(
    archive: &GgufArchive,
    loader: &NormalLoaderType,
    architecture: CanonicalGgufArchitecture,
    block_count: usize,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    for layer in 0..block_count {
        match loader {
            NormalLoaderType::Phi2 | NormalLoaderType::Phi3_5MoE => {
                bind_split_qkv(archive, architecture, layer, bindings)?;
            }
            NormalLoaderType::GLM4 => {
                bind_split_qkv(archive, architecture, layer, bindings)?;
                bind_fused_gate_up(archive, layer, bindings);
            }
            NormalLoaderType::Phi3 => {
                bind_phi3_qkv(archive, layer, bindings);
                bind_fused_gate_up(archive, layer, bindings);
            }
            NormalLoaderType::DeepSeekV2
            | NormalLoaderType::DeepSeekV3
            | NormalLoaderType::GLM4MoeLite => {
                bind_deepseek_kv_b(archive, layer, bindings)?;
            }
            NormalLoaderType::Qwen3Next => bind_qwen3_next(archive, architecture, layer, bindings)?,
            NormalLoaderType::GptOss => bind_gpt_oss(archive, layer, bindings)?,
            NormalLoaderType::GraniteMoeHybrid => bind_granite(archive, layer, bindings),
            _ => {}
        }
    }
    Ok(())
}

fn bind_split_qkv(
    archive: &GgufArchive,
    architecture: CanonicalGgufArchitecture,
    layer: usize,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    for suffix in ["weight", "bias"] {
        let source = format!("blk.{layer}.attn_qkv.{suffix}");
        if !archive.contains_tensor(&source) {
            continue;
        }
        let shape = archive.tensor_info(&source)?.shape();
        let output = *shape.first().context("fused QKV tensor has scalar shape")?;
        let prefix = architecture.as_str();
        let hidden = metadata_usize(archive, &format!("{prefix}.embedding_length"))?;
        let head_count = metadata_usize(archive, &format!("{prefix}.attention.head_count"))?;
        let head_count_kv =
            metadata_optional_usize(archive, &format!("{prefix}.attention.head_count_kv"))?
                .unwrap_or(head_count);
        let default_head_dim = fused_qkv_default_head_dim(hidden, head_count)?;
        let key_length =
            metadata_optional_usize(archive, &format!("{prefix}.attention.key_length"))?
                .unwrap_or(default_head_dim);
        let value_length =
            metadata_optional_usize(archive, &format!("{prefix}.attention.value_length"))?
                .unwrap_or(key_length);
        if value_length != key_length {
            bail!(
                "native `{architecture}` loading requires equal key/value head lengths, got {key_length} and {value_length}"
            );
        }
        if matches!(
            architecture,
            CanonicalGgufArchitecture::Phi2 | CanonicalGgufArchitecture::PhiMoe
        ) && key_length != default_head_dim
        {
            bail!(
                "native `{architecture}` loading requires attention key length {default_head_dim}, got {key_length}"
            );
        }
        let q_rows = head_count
            .checked_mul(key_length)
            .context("fused QKV query size overflow")?;
        let k_rows = head_count_kv
            .checked_mul(key_length)
            .context("fused QKV key size overflow")?;
        let v_rows = head_count_kv
            .checked_mul(value_length)
            .context("fused QKV value size overflow")?;
        let expected = q_rows
            .checked_add(k_rows)
            .and_then(|rows| rows.checked_add(v_rows))
            .context("fused QKV output size overflow")?;
        if output != expected {
            bail!(
                "fused QKV tensor `{source}` has {output} rows, expected {q_rows} query + {k_rows} key + {v_rows} value rows"
            );
        }
        let p = format!("model.layers.{layer}.self_attn");
        for (name, start, len) in [
            ("q_proj", 0, q_rows),
            ("k_proj", q_rows, k_rows),
            ("v_proj", q_rows + k_rows, v_rows),
        ] {
            bindings.insert(
                format!("{p}.{name}.{suffix}"),
                GgufTensorBinding::tensor(&source).slice(0, start, len),
            );
        }
    }
    Ok(())
}

fn fused_qkv_default_head_dim(hidden: usize, head_count: usize) -> Result<usize> {
    if head_count == 0 || !hidden.is_multiple_of(head_count) {
        bail!(
            "fused QKV metadata has embedding length {hidden} and attention head count {head_count}"
        );
    }
    Ok(hidden / head_count)
}

fn bind_phi3_qkv(archive: &GgufArchive, layer: usize, bindings: &mut GgufBindingMap) {
    for suffix in ["weight", "bias"] {
        let fused = format!("blk.{layer}.attn_qkv.{suffix}");
        if archive.contains_tensor(&fused) {
            continue;
        }
        let inputs = ["attn_q", "attn_k", "attn_v"]
            .into_iter()
            .map(|role| format!("blk.{layer}.{role}.{suffix}"))
            .collect::<Vec<_>>();
        if inputs.iter().all(|name| archive.contains_tensor(name)) {
            bindings.insert(
                format!("model.layers.{layer}.self_attn.qkv_proj.{suffix}"),
                GgufTensorBinding::concat(
                    inputs.into_iter().map(GgufTensorBinding::tensor).collect(),
                    0,
                ),
            );
        }
    }
}

fn bind_fused_gate_up(archive: &GgufArchive, layer: usize, bindings: &mut GgufBindingMap) {
    for suffix in ["weight", "bias"] {
        let gate = format!("blk.{layer}.ffn_gate.{suffix}");
        let up = format!("blk.{layer}.ffn_up.{suffix}");
        let target = format!("model.layers.{layer}.mlp.gate_up_proj.{suffix}");
        if archive.contains_tensor(&gate) && archive.contains_tensor(&up) {
            bindings.insert(
                target,
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor(gate),
                        GgufTensorBinding::tensor(up),
                    ],
                    0,
                ),
            );
        } else if archive.contains_tensor(&up) {
            bindings.insert(target, GgufTensorBinding::tensor(up));
        }
    }
}

fn bind_fused_expert_gate_up(
    archive: &GgufArchive,
    loader: &NormalLoaderType,
    prefix: &str,
    suffix: &str,
    source: &str,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let shape = archive.tensor_info(source)?.shape();
    let expected_rank = if suffix == "bias" { 2 } else { 3 };
    if shape.len() != expected_rank || !shape[1].is_multiple_of(2) {
        bail!("fused expert gate/up tensor `{source}` has invalid shape {shape:?}");
    }
    let intermediate = shape[1] / 2;
    for (projection, start) in [("gate", 0), ("up", intermediate)] {
        bindings.insert(
            expert_target(loader, prefix, projection, suffix),
            GgufTensorBinding::tensor(source).slice(1, start, intermediate),
        );
    }
    Ok(())
}

fn bind_deepseek_kv_b(
    archive: &GgufArchive,
    layer: usize,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let key = format!("blk.{layer}.attn_k_b.weight");
    let value = format!("blk.{layer}.attn_v_b.weight");
    if !archive.contains_tensor(&key) || !archive.contains_tensor(&value) {
        return Ok(());
    }
    let ks = archive.tensor_info(&key)?.shape();
    let vs = archive.tensor_info(&value)?.shape();
    if ks.len() != 3 || vs.len() != 3 || ks[0] != vs[0] || ks[1] != vs[2] {
        bail!("DeepSeek layer {layer} has incompatible split MLA KV-B tensors");
    }
    bindings.insert(
        format!("model.layers.{layer}.self_attn.k_b_proj.weight"),
        GgufTensorBinding::tensor(key),
    );
    bindings.insert(
        format!("model.layers.{layer}.self_attn.v_b_proj.weight"),
        GgufTensorBinding::tensor(value),
    );
    Ok(())
}

fn bind_qwen3_next(
    archive: &GgufArchive,
    architecture: CanonicalGgufArchitecture,
    layer: usize,
    bindings: &mut GgufBindingMap,
) -> Result<()> {
    let p = format!("model.layers.{layer}.linear_attn");
    let projections = match architecture {
        CanonicalGgufArchitecture::Qwen3Next => &[
            ("in_proj_qkv.weight", "attn_qkv.weight"),
            ("in_proj_z.weight", "attn_gate.weight"),
            ("in_proj_ba.weight", "ssm_ba.weight"),
        ][..],
        CanonicalGgufArchitecture::Qwen35Moe => &[
            ("in_proj_qkv.weight", "attn_qkv.weight"),
            ("in_proj_z.weight", "attn_gate.weight"),
            ("in_proj_b.weight", "ssm_beta.weight"),
            ("in_proj_a.weight", "ssm_alpha.weight"),
        ][..],
        _ => bail!("native Qwen3Next binding does not support `{architecture}`"),
    };
    for (native, canonical) in projections.iter().copied().chain([
        ("dt_bias", "ssm_dt.bias"),
        ("norm.weight", "ssm_norm.weight"),
        ("out_proj.weight", "ssm_out.weight"),
    ]) {
        bind_named(
            archive,
            bindings,
            format!("{p}.{native}"),
            format!("blk.{layer}.{canonical}"),
        );
    }
    let conv = format!("blk.{layer}.ssm_conv1d.weight");
    if archive.contains_tensor(&conv) {
        let shape = archive.tensor_info(&conv)?.shape();
        let binding = if shape.len() == 2 {
            GgufTensorBinding::tensor(&conv).reshape(vec![shape[0], 1, shape[1]])
        } else {
            GgufTensorBinding::tensor(&conv)
        };
        bindings.insert(format!("{p}.conv1d.weight"), binding);
    }
    let a = format!("blk.{layer}.ssm_a");
    if archive.contains_tensor(&a) {
        bindings.insert(
            format!("{p}.A_log"),
            GgufTensorBinding::tensor(a).affine(-1.0, 0.0).log(),
        );
    }
    Ok(())
}

fn bind_gpt_oss(archive: &GgufArchive, layer: usize, bindings: &mut GgufBindingMap) -> Result<()> {
    let p = format!("model.layers.{layer}.mlp");
    for suffix in ["weight", "bias"] {
        bind_named(
            archive,
            bindings,
            format!("{p}.router.{suffix}"),
            format!("blk.{layer}.ffn_gate_inp.{suffix}"),
        );
    }
    for (native, canonical) in [
        ("down_proj", "ffn_down_exps"),
        ("gate_proj", "ffn_gate_exps"),
        ("up_proj", "ffn_up_exps"),
    ] {
        for suffix in ["weight", "bias"] {
            let source = format!("blk.{layer}.{canonical}.{suffix}");
            if archive.contains_tensor(&source) && archive.tensor_info(&source)?.dtype().raw() != 39
            {
                bindings.insert(
                    format!("{p}.experts.{native}.{suffix}"),
                    GgufTensorBinding::tensor(source),
                );
            }
        }
    }
    let gate = format!("blk.{layer}.ffn_gate_exps.weight");
    let up = format!("blk.{layer}.ffn_up_exps.weight");
    let down = format!("blk.{layer}.ffn_down_exps.weight");
    if archive.contains_tensor(&gate)
        && archive.contains_tensor(&up)
        && archive.tensor_info(&gate)?.dtype().raw() == 39
        && archive.tensor_info(&up)?.dtype().raw() == 39
    {
        let shape = archive.tensor_info(&gate)?.shape();
        let up_shape = archive.tensor_info(&up)?.shape();
        if shape.len() != 3 || up_shape != shape {
            bail!("GPT-OSS layer {layer} has incompatible gate/up expert tensors");
        }
        let output = shape[1]
            .checked_mul(2)
            .context("GPT-OSS gate/up output size overflow")?;
        bindings.insert(
            format!("{p}.experts.gate_up_proj_blocks"),
            GgufTensorBinding::interleave(
                vec![
                    GgufTensorBinding::mxfp4_blocks(&gate),
                    GgufTensorBinding::mxfp4_blocks(&up),
                ],
                1,
            )
            .reshape(vec![shape[0], output, shape[2] / 32, 16]),
        );
        bindings.insert(
            format!("{p}.experts.gate_up_proj_scales"),
            GgufTensorBinding::interleave(
                vec![
                    GgufTensorBinding::mxfp4_scales(&gate),
                    GgufTensorBinding::mxfp4_scales(&up),
                ],
                1,
            ),
        );
        let gate_bias = format!("blk.{layer}.ffn_gate_exps.bias");
        let up_bias = format!("blk.{layer}.ffn_up_exps.bias");
        if archive.contains_tensor(&gate_bias) && archive.contains_tensor(&up_bias) {
            bindings.insert(
                format!("{p}.experts.gate_up_proj_bias"),
                GgufTensorBinding::interleave(
                    vec![
                        GgufTensorBinding::tensor(gate_bias),
                        GgufTensorBinding::tensor(up_bias),
                    ],
                    1,
                ),
            );
        }
    }
    if archive.contains_tensor(&down) && archive.tensor_info(&down)?.dtype().raw() == 39 {
        let shape = archive.tensor_info(&down)?.shape();
        if shape.len() != 3 {
            bail!(
                "GPT-OSS layer {layer} has a rank-{} down expert tensor",
                shape.len()
            );
        }
        bindings.insert(
            format!("{p}.experts.down_proj_blocks"),
            GgufTensorBinding::mxfp4_blocks(&down).reshape(vec![
                shape[0],
                shape[1],
                shape[2] / 32,
                16,
            ]),
        );
        bindings.insert(
            format!("{p}.experts.down_proj_scales"),
            GgufTensorBinding::mxfp4_scales(&down),
        );
        let down_bias = format!("blk.{layer}.ffn_down_exps.bias");
        if archive.contains_tensor(&down_bias) {
            bindings.insert(
                format!("{p}.experts.down_proj_bias"),
                GgufTensorBinding::tensor(down_bias),
            );
        }
    }
    Ok(())
}

fn bind_granite(archive: &GgufArchive, layer: usize, bindings: &mut GgufBindingMap) {
    for suffix in ["weight", "bias"] {
        let shared_gate = format!("blk.{layer}.ffn_gate_shexp.{suffix}");
        let shared_up = format!("blk.{layer}.ffn_up_shexp.{suffix}");
        let dense_gate = format!("blk.{layer}.ffn_gate.{suffix}");
        let dense_up = format!("blk.{layer}.ffn_up.{suffix}");
        let (gate, up) =
            if archive.contains_tensor(&shared_gate) && archive.contains_tensor(&shared_up) {
                (shared_gate, shared_up)
            } else {
                (dense_gate, dense_up)
            };
        if archive.contains_tensor(&gate) && archive.contains_tensor(&up) {
            bindings.insert(
                format!("model.layers.{layer}.shared_mlp.input_linear.{suffix}"),
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor(gate),
                        GgufTensorBinding::tensor(up),
                    ],
                    0,
                ),
            );
        }
        bind_named(
            archive,
            bindings,
            format!("model.layers.{layer}.shared_mlp.output_linear.{suffix}"),
            if archive.contains_tensor(&format!("blk.{layer}.ffn_down_shexp.{suffix}")) {
                format!("blk.{layer}.ffn_down_shexp.{suffix}")
            } else {
                format!("blk.{layer}.ffn_down.{suffix}")
            },
        );

        let expert_gate = format!("blk.{layer}.ffn_gate_exps.{suffix}");
        let expert_up = format!("blk.{layer}.ffn_up_exps.{suffix}");
        if archive.contains_tensor(&expert_gate) && archive.contains_tensor(&expert_up) {
            bindings.insert(
                format!("model.layers.{layer}.block_sparse_moe.input_linear.{suffix}"),
                GgufTensorBinding::concat(
                    vec![
                        GgufTensorBinding::tensor(expert_gate),
                        GgufTensorBinding::tensor(expert_up),
                    ],
                    1,
                ),
            );
        }
        bind_named(
            archive,
            bindings,
            format!("model.layers.{layer}.block_sparse_moe.output_linear.{suffix}"),
            format!("blk.{layer}.ffn_down_exps.{suffix}"),
        );
    }
}

fn dense_mlp_target(
    loader: &NormalLoaderType,
    p: &str,
    projection: &str,
    suffix: &str,
) -> Option<String> {
    let name = match loader {
        NormalLoaderType::Phi2 => match projection {
            "up" => "fc1",
            "down" => "fc2",
            _ => return None,
        },
        NormalLoaderType::Starcoder2 => match projection {
            "up" => "c_fc",
            "down" => "c_proj",
            _ => return None,
        },
        NormalLoaderType::Phi3 | NormalLoaderType::GLM4 => match projection {
            "up" => "gate_up_proj",
            "down" => "down_proj",
            _ => return None,
        },
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => match projection {
            "gate" => "w1",
            "up" => "w3",
            "down" => "w2",
            _ => unreachable!(),
        },
        _ => match projection {
            "gate" => "gate_proj",
            "up" => "up_proj",
            "down" => "down_proj",
            _ => unreachable!(),
        },
    };
    let module = if matches!(loader, NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe) {
        "feed_forward"
    } else {
        "mlp"
    };
    Some(format!("{p}.{module}.{name}.{suffix}"))
}

fn router_target(loader: &NormalLoaderType, p: &str, suffix: &str) -> String {
    match loader {
        NormalLoaderType::Phi3_5MoE | NormalLoaderType::Mixtral => {
            format!("{p}.block_sparse_moe.gate.{suffix}")
        }
        NormalLoaderType::GraniteMoeHybrid => {
            format!("{p}.block_sparse_moe.router.layer.{suffix}")
        }
        NormalLoaderType::HunYuanMoEV1 => format!("{p}.mlp.gate.wg.{suffix}"),
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => {
            format!("{p}.feed_forward.gate.{suffix}")
        }
        NormalLoaderType::GptOss => format!("{p}.mlp.router.{suffix}"),
        _ => format!("{p}.mlp.gate.{suffix}"),
    }
}

fn expert_target(loader: &NormalLoaderType, p: &str, projection: &str, suffix: &str) -> String {
    let module = match loader {
        NormalLoaderType::Phi3_5MoE | NormalLoaderType::Mixtral => "block_sparse_moe",
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe => "feed_forward",
        _ => "mlp",
    };
    format!("{p}.{module}.experts.{projection}_proj.{suffix}")
}

fn shared_expert_target(
    loader: &NormalLoaderType,
    p: &str,
    projection: &str,
    suffix: &str,
) -> String {
    let module = match loader {
        NormalLoaderType::GLM4Moe | NormalLoaderType::GLM4MoeLite => "shared_experts",
        NormalLoaderType::HunYuanMoEV1 => "shared_mlp",
        NormalLoaderType::Qwen3Next => "shared_expert",
        _ => "shared_experts",
    };
    format!("{p}.mlp.{module}.{projection}_proj.{suffix}")
}

fn q_norm_name(loader: &NormalLoaderType) -> &'static str {
    match loader {
        NormalLoaderType::HunYuanDenseV1 | NormalLoaderType::HunYuanMoEV1 => "query_layernorm",
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe | NormalLoaderType::Phi2 => {
            "q_layernorm"
        }
        _ => "q_norm",
    }
}

fn k_norm_name(loader: &NormalLoaderType) -> &'static str {
    match loader {
        NormalLoaderType::HunYuanDenseV1 | NormalLoaderType::HunYuanMoEV1 => "key_layernorm",
        NormalLoaderType::Lfm2 | NormalLoaderType::Lfm2Moe | NormalLoaderType::Phi2 => {
            "k_layernorm"
        }
        _ => "k_norm",
    }
}

fn norm_binding(
    architecture: CanonicalGgufArchitecture,
    source: &str,
    binding: GgufTensorBinding,
) -> GgufTensorBinding {
    if source.ends_with(".weight")
        && matches!(
            architecture,
            CanonicalGgufArchitecture::Gemma
                | CanonicalGgufArchitecture::Gemma2
                | CanonicalGgufArchitecture::Qwen3Next
                | CanonicalGgufArchitecture::Qwen35Moe
        )
    {
        binding.affine(1.0, -1.0)
    } else {
        binding
    }
}

fn granite_ssm_binding(
    archive: &GgufArchive,
    role: &str,
    source: &str,
    mut binding: GgufTensorBinding,
) -> Result<GgufTensorBinding> {
    let shape = archive.tensor_info(source)?.shape();
    match role {
        "ssm_conv1d" if shape.len() == 2 => {
            binding = binding.reshape(vec![shape[0], 1, shape[1]]);
        }
        "ssm_a" => {
            binding = binding
                .reshape(vec![shape.iter().product()])
                .affine(-1.0, 0.0)
                .log();
        }
        "ssm_d" | "ssm_norm" => binding = binding.reshape(vec![shape.iter().product()]),
        _ => {}
    }
    Ok(binding)
}

fn parse_block_tensor(name: &str) -> Option<(usize, &str, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let (role, suffix) = match rest.rsplit_once('.') {
        Some((role, suffix)) if matches!(suffix, "weight" | "bias") => (role, suffix),
        _ => (rest, ""),
    };
    Some((layer.parse().ok()?, role, suffix))
}

fn metadata_usize(archive: &GgufArchive, key: &str) -> Result<usize> {
    let value = archive
        .metadata_value(key)
        .with_context(|| format!("GGUF metadata is missing `{key}`"))?;
    metadata_value_usize(value, key)
}

fn metadata_optional_usize(archive: &GgufArchive, key: &str) -> Result<Option<usize>> {
    archive
        .metadata_value(key)
        .map(|value| metadata_value_usize(value, key))
        .transpose()
}

fn metadata_value_usize(value: &Value, key: &str) -> Result<usize> {
    let value = match value {
        Value::U8(v) => *v as u64,
        Value::U16(v) => *v as u64,
        Value::U32(v) => *v as u64,
        Value::U64(v) => *v,
        Value::I8(v) if *v >= 0 => *v as u64,
        Value::I16(v) if *v >= 0 => *v as u64,
        Value::I32(v) if *v >= 0 => *v as u64,
        Value::I64(v) if *v >= 0 => *v as u64,
        _ => bail!("GGUF metadata `{key}` is not a nonnegative integer"),
    };
    usize::try_from(value).with_context(|| format!("GGUF metadata `{key}` does not fit usize"))
}

fn native_block_count(
    archive: &GgufArchive,
    architecture: CanonicalGgufArchitecture,
) -> Result<usize> {
    let prefix = architecture.as_str();
    let block_count = metadata_usize(archive, &format!("{prefix}.block_count"))?;
    let nextn_predict_layers =
        metadata_optional_usize(archive, &format!("{prefix}.nextn_predict_layers"))?.unwrap_or(0);
    subtract_mtp_layers(block_count, nextn_predict_layers)
}

fn subtract_mtp_layers(block_count: usize, nextn_predict_layers: usize) -> Result<usize> {
    block_count.checked_sub(nextn_predict_layers).with_context(|| {
        format!(
            "GGUF next-token prediction layer count {nextn_predict_layers} exceeds block count {block_count}"
        )
    })
}

fn bind(archive: &GgufArchive, bindings: &mut GgufBindingMap, native: &str, canonical: &str) {
    if archive.contains_tensor(canonical) {
        bindings.insert(native, GgufTensorBinding::tensor(canonical));
    }
}

fn bind_named(
    archive: &GgufArchive,
    bindings: &mut GgufBindingMap,
    native: String,
    canonical: String,
) {
    if archive.contains_tensor(&canonical) {
        bindings.insert(native, GgufTensorBinding::tensor(canonical));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_block_tensor_names() {
        assert_eq!(
            parse_block_tensor("blk.17.shortconv.in_proj.weight"),
            Some((17, "shortconv.in_proj", "weight"))
        );
        assert_eq!(
            parse_block_tensor("blk.2.ffn_gate_up_exps.weight"),
            Some((2, "ffn_gate_up_exps", "weight"))
        );
        assert_eq!(parse_block_tensor("token_embd.weight"), None);
    }

    #[test]
    fn moe_targets_match_native_prefixes() {
        let prefix = "model.layers.3";
        assert_eq!(
            expert_target(&NormalLoaderType::Mixtral, prefix, "gate", "weight"),
            "model.layers.3.block_sparse_moe.experts.gate_proj.weight"
        );
        assert_eq!(
            expert_target(&NormalLoaderType::Lfm2Moe, prefix, "up", "weight"),
            "model.layers.3.feed_forward.experts.up_proj.weight"
        );
        assert_eq!(
            router_target(&NormalLoaderType::GraniteMoeHybrid, prefix, "weight"),
            "model.layers.3.block_sparse_moe.router.layer.weight"
        );
    }

    #[test]
    fn mtp_tail_layers_are_excluded_from_native_block_count() {
        assert_eq!(subtract_mtp_layers(41, 1).unwrap(), 40);
        assert!(subtract_mtp_layers(1, 2).is_err());
    }

    #[test]
    fn fused_qkv_default_head_dimension_is_exact() {
        assert_eq!(fused_qkv_default_head_dim(4096, 32).unwrap(), 128);
        assert!(fused_qkv_default_head_dim(4096, 0).is_err());
        assert!(fused_qkv_default_head_dim(4097, 32).is_err());
    }
}
