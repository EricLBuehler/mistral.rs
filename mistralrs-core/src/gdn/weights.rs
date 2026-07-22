use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{Comm, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;
use super::projection::GdnInputProjection;

pub enum GdnInputProjectionKind {
    Grouped,
    Split,
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
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };
        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

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
            GdnInputProjectionKind::Split => GdnInputProjection::Split {
                in_proj_qkv: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.conv_dim,
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_qkv"),
                )?,
                in_proj_z: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.value_dim,
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_z"),
                )?,
                in_proj_b: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.num_v_heads,
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_b"),
                )?,
                in_proj_a: ReplicatedLayer::new(
                    dims.hidden_size,
                    dims.num_v_heads,
                    cfg.quantization_config(),
                    false,
                    vb_la.pp("in_proj_a"),
                )?,
            },
        };
        let conv1d_weight = move_to_target(
            vb_la.get((dims.conv_dim, 1, dims.conv_kernel_size), "conv1d.weight")?,
            isq_target_device.as_ref(),
        )?;
        let dt_bias = move_to_target(
            vb_la.get(dims.num_v_heads, "dt_bias")?,
            isq_target_device.as_ref(),
        )?;
        let a_log = move_to_target(
            vb_la.get(dims.num_v_heads, "A_log")?,
            isq_target_device.as_ref(),
        )?;

        let norm = RmsNormGated::new(
            dims.head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;
        let out_proj = RowParallelLayer::new(
            dims.value_dim,
            dims.hidden_size,
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
