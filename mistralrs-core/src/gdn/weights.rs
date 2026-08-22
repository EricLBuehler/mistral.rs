use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{
    Comm, LoraLinearSpec, QuantMethod, ReplicatedLayer, RowParallelLayer, Shard, ShardedVarBuilder,
};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

use super::config::{GdnConfig, GdnDims, GdnVHeadLayout};
use super::norm::RmsNormGated;
use super::projection::GdnInputProjection;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnInputProjectionKind {
    Grouped,
    Split,
    SplitQkvzGroupedBa,
}

struct SplitGdnLoraMaps {
    qkv: Arc<[usize]>,
    value: Arc<[usize]>,
    heads: Arc<[usize]>,
}

fn tiled_v_runtime_to_canonical(dims: &GdnDims, head_dim: usize) -> Result<Vec<usize>> {
    let expected_heads = dims
        .num_k_heads
        .checked_mul(dims.v_per_group)
        .ok_or_else(|| candle_core::Error::msg("GDN value head count overflow"))?;
    if dims.num_k_heads == 0 || dims.num_v_heads != expected_heads {
        candle_core::bail!(
            "GDN has incompatible tiled head counts: {} key heads, {} value heads, {} values per key",
            dims.num_k_heads,
            dims.num_v_heads,
            dims.v_per_group
        );
    }
    let feature_count = dims
        .num_v_heads
        .checked_mul(head_dim)
        .ok_or_else(|| candle_core::Error::msg("GDN value feature count overflow"))?;
    let mut runtime_to_canonical = Vec::with_capacity(feature_count);
    for runtime_head in 0..dims.num_v_heads {
        let key_head = runtime_head % dims.num_k_heads;
        let within_group = runtime_head / dims.num_k_heads;
        let canonical_head = key_head
            .checked_mul(dims.v_per_group)
            .and_then(|head| head.checked_add(within_group))
            .ok_or_else(|| candle_core::Error::msg("GDN value head index overflow"))?;
        let canonical_start = canonical_head
            .checked_mul(head_dim)
            .ok_or_else(|| candle_core::Error::msg("GDN value feature index overflow"))?;
        for feature in 0..head_dim {
            runtime_to_canonical.push(canonical_start + feature);
        }
    }
    Ok(runtime_to_canonical)
}

fn split_gdn_lora_maps(dims: &GdnDims) -> Result<Option<SplitGdnLoraMaps>> {
    if dims.v_head_layout == GdnVHeadLayout::Grouped {
        return Ok(None);
    }
    let value: Arc<[usize]> = tiled_v_runtime_to_canonical(dims, dims.head_v_dim)?.into();
    let heads: Arc<[usize]> = tiled_v_runtime_to_canonical(dims, 1)?.into();
    let qk_dim = dims
        .key_dim
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::msg("GDN QK feature count overflow"))?;
    let mut qkv = (0..qk_dim).collect::<Vec<_>>();
    qkv.reserve(value.len());
    for &canonical in value.iter() {
        qkv.push(
            qk_dim
                .checked_add(canonical)
                .ok_or_else(|| candle_core::Error::msg("GDN QKV feature index overflow"))?,
        );
    }
    if qkv.len() != dims.conv_dim {
        candle_core::bail!(
            "GDN tiled QKV map has length {}, expected {}",
            qkv.len(),
            dims.conv_dim
        );
    }
    Ok(Some(SplitGdnLoraMaps {
        qkv: qkv.into(),
        value,
        heads,
    }))
}

fn validate_projection_layout(
    layout: GdnVHeadLayout,
    input_projection_kind: GdnInputProjectionKind,
) -> Result<()> {
    if layout == GdnVHeadLayout::Tiled && input_projection_kind != GdnInputProjectionKind::Split {
        candle_core::bail!(
            "tiled GDN value-head layout requires split QKV/Z/B/A projections, got {input_projection_kind:?}"
        );
    }
    Ok(())
}

fn replicated_output_lora_spec(
    in_features: usize,
    out_features: usize,
    runtime_to_canonical: Option<Arc<[usize]>>,
) -> Result<LoraLinearSpec> {
    let spec = LoraLinearSpec::replicated(in_features, out_features);
    match runtime_to_canonical {
        Some(runtime_to_canonical) => spec.with_output_runtime_to_canonical(runtime_to_canonical),
        None => Ok(spec),
    }
}

fn row_input_lora_spec(
    in_features: usize,
    out_features: usize,
    comm: &Comm,
    runtime_to_canonical: Option<Arc<[usize]>>,
) -> Result<LoraLinearSpec> {
    let spec = LoraLinearSpec::row(
        in_features,
        out_features,
        Shard::Simple {
            dim: 1,
            rank: comm.rank(),
            world_size: comm.world_size(),
        },
    );
    match runtime_to_canonical {
        Some(runtime_to_canonical) => spec.with_input_runtime_to_canonical(runtime_to_canonical),
        None => Ok(spec),
    }
}

pub struct GdnWeights {
    pub input_proj: GdnInputProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
}

pub struct GdnWeightLoadCtx<'a> {
    pub cfg: &'a dyn GdnConfig,
    pub dims: &'a GdnDims,
    pub mapper: &'a dyn DeviceMapper,
    pub layer_idx: usize,
    pub loading_isq: bool,
    pub comm: &'a Arc<Comm>,
    pub input_projection_kind: GdnInputProjectionKind,
}

impl GdnWeights {
    pub fn load(vb: ShardedVarBuilder, ctx: GdnWeightLoadCtx<'_>) -> Result<Self> {
        let GdnWeightLoadCtx {
            cfg,
            dims,
            mapper,
            layer_idx,
            loading_isq,
            comm,
            input_projection_kind,
        } = ctx;
        validate_projection_layout(dims.v_head_layout, input_projection_kind)?;
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };
        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);
        let split_lora_maps = if vb_la.lora_registry().is_some() {
            split_gdn_lora_maps(dims)?
        } else {
            None
        };

        let input_proj = match input_projection_kind {
            GdnInputProjectionKind::Grouped => GdnInputProjection::Grouped {
                in_proj_qkvz: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.qkvz_out_dim(),
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_qkvz"),
                )?,
                in_proj_ba: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.ba_out_dim(),
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_ba"),
                )?,
            },
            GdnInputProjectionKind::Split => {
                let qkv_spec = replicated_output_lora_spec(
                    dims.hidden_size,
                    dims.conv_dim,
                    split_lora_maps.as_ref().map(|maps| maps.qkv.clone()),
                )?;
                let z_spec = replicated_output_lora_spec(
                    dims.hidden_size,
                    dims.value_dim,
                    split_lora_maps.as_ref().map(|maps| maps.value.clone()),
                )?;
                let b_spec = replicated_output_lora_spec(
                    dims.hidden_size,
                    dims.num_v_heads,
                    split_lora_maps.as_ref().map(|maps| maps.heads.clone()),
                )?;
                let a_spec = replicated_output_lora_spec(
                    dims.hidden_size,
                    dims.num_v_heads,
                    split_lora_maps.as_ref().map(|maps| maps.heads.clone()),
                )?;
                let packed_qkv_z = ReplicatedLayer::new_packed(
                    &[qkv_spec.clone(), z_spec.clone()],
                    &["in_proj_qkv", "in_proj_z"],
                    cfg.quantization_config(),
                    false,
                    vb_la.clone(),
                )?;
                let (in_proj_qkv, in_proj_z, merged_qkv_z) = match &packed_qkv_z {
                    Some(group) => (
                        group.constituents[0].clone(),
                        group.constituents[1].clone(),
                        Some(crate::ops::MergedDenseProjection::from_packed(group)),
                    ),
                    None => (
                        ReplicatedLayer::new_with_lora_spec(
                            qkv_spec,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_qkv"),
                        )?,
                        ReplicatedLayer::new_with_lora_spec(
                            z_spec,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_z"),
                        )?,
                        None,
                    ),
                };
                let packed_b_a = ReplicatedLayer::new_packed(
                    &[b_spec.clone(), a_spec.clone()],
                    &["in_proj_b", "in_proj_a"],
                    cfg.quantization_config(),
                    false,
                    vb_la.clone(),
                )?;
                let (in_proj_b, in_proj_a, merged_b_a) = match &packed_b_a {
                    Some(group) => (
                        group.constituents[0].clone(),
                        group.constituents[1].clone(),
                        Some(crate::ops::MergedDenseProjection::from_packed(group)),
                    ),
                    None => (
                        ReplicatedLayer::new_with_lora_spec(
                            b_spec,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_b"),
                        )?,
                        ReplicatedLayer::new_with_lora_spec(
                            a_spec,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_a"),
                        )?,
                        None,
                    ),
                };
                GdnInputProjection::Split {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                    merged_qkv_z,
                    merged_b_a,
                }
            }
            GdnInputProjectionKind::SplitQkvzGroupedBa => {
                let qkv_spec = LoraLinearSpec::replicated(dims.hidden_size, dims.conv_dim);
                let z_spec = LoraLinearSpec::replicated(dims.hidden_size, dims.value_dim);
                let packed_qkv_z = ReplicatedLayer::new_packed(
                    &[qkv_spec, z_spec],
                    &["in_proj_qkv", "in_proj_z"],
                    cfg.quantization_config(),
                    false,
                    vb_la.clone(),
                )?;
                let (in_proj_qkv, in_proj_z, merged_qkv_z) = match &packed_qkv_z {
                    Some(group) => (
                        group.constituents[0].clone(),
                        group.constituents[1].clone(),
                        Some(crate::ops::MergedDenseProjection::from_packed(group)),
                    ),
                    None => (
                        ReplicatedLayer::new(
                            dims.hidden_size,
                            dims.conv_dim,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_qkv"),
                        )?,
                        ReplicatedLayer::new(
                            dims.hidden_size,
                            dims.value_dim,
                            cfg.quantization_config(),
                            false,
                            vb_la.pp("in_proj_z"),
                        )?,
                        None,
                    ),
                };
                GdnInputProjection::SplitQkvzGroupedBa {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_ba: ReplicatedLayer::new(
                        dims.hidden_size,
                        dims.ba_out_dim(),
                        cfg.quantization_config(),
                        false,
                        vb_la.pp("in_proj_ba"),
                    )?,
                    merged_qkv_z,
                }
            }
        };
        let conv1d_weight = move_to_target(
            vb_la.get((dims.conv_dim, 1, dims.conv_kernel_size), "conv1d.weight")?,
            isq_target_device.as_ref(),
        )?;
        // The recurrence consumes these in f32 every step; A_log is f32 in the checkpoint anyway
        let vb_f32 = vb_la.clone().set_dtype(DType::F32);
        let dt_bias = move_to_target(
            vb_f32.get(dims.num_v_heads, "dt_bias")?,
            isq_target_device.as_ref(),
        )?;
        let a_log = move_to_target(
            vb_f32.get(dims.num_v_heads, "A_log")?,
            isq_target_device.as_ref(),
        )?;

        let norm = RmsNormGated::new(
            dims.head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;
        let out_spec = row_input_lora_spec(
            dims.value_dim,
            dims.hidden_size,
            comm,
            split_lora_maps.as_ref().map(|maps| maps.value.clone()),
        )?;
        let out_proj = RowParallelLayer::new_with_lora_spec(
            out_spec,
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        Ok(Self {
            input_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
        })
    }
}

fn move_to_target(tensor: Tensor, target_device: Option<&Device>) -> Result<Tensor> {
    if let Some(target_device) = target_device {
        tensor.to_device(target_device)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(layout: GdnVHeadLayout) -> GdnDims {
        GdnDims {
            hidden_size: 8,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 2,
            head_v_dim: 2,
            conv_kernel_size: 3,
            key_dim: 4,
            value_dim: 8,
            conv_dim: 16,
            v_per_group: 2,
            v_head_layout: layout,
        }
    }

    #[test]
    fn tiled_split_lora_maps_runtime_features_to_grouped_adapter_features() -> Result<()> {
        let maps =
            split_gdn_lora_maps(&dims(GdnVHeadLayout::Tiled))?.expect("tiled layout has LoRA maps");
        assert_eq!(&*maps.heads, &[0, 2, 1, 3]);
        assert_eq!(&*maps.value, &[0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(
            &*maps.qkv,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        assert!(split_gdn_lora_maps(&dims(GdnVHeadLayout::Grouped))?.is_none());
        Ok(())
    }

    #[test]
    fn tiled_layout_rejects_non_split_projection_shapes() {
        assert!(
            validate_projection_layout(GdnVHeadLayout::Tiled, GdnInputProjectionKind::Split)
                .is_ok()
        );
        for kind in [
            GdnInputProjectionKind::Grouped,
            GdnInputProjectionKind::SplitQkvzGroupedBa,
        ] {
            let error = validate_projection_layout(GdnVHeadLayout::Tiled, kind)
                .unwrap_err()
                .to_string();
            assert!(error.contains("requires split QKV/Z/B/A"), "{error}");
        }
        for kind in [
            GdnInputProjectionKind::Grouped,
            GdnInputProjectionKind::Split,
            GdnInputProjectionKind::SplitQkvzGroupedBa,
        ] {
            assert!(validate_projection_layout(GdnVHeadLayout::Grouped, kind).is_ok());
        }
    }
}
