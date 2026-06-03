use candle_core::{Device, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{Comm, QuantMethod, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;

pub enum GdnWeightMode {
    MergedOnly,
    MergedWithFallback,
}

pub struct GdnWeights {
    pub in_proj_qkvz: Linear,
    pub in_proj_ba: Linear,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
}

impl GdnWeights {
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &dyn GdnConfig,
        dims: &GdnDims,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<Comm>,
        weight_mode: GdnWeightMode,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };
        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkvz_w = move_to_target(
            load_qkvz(&vb_la, dims, &weight_mode)?,
            isq_target_device.as_ref(),
        )?;
        let ba_w = move_to_target(
            load_ba(&vb_la, dims, &weight_mode)?,
            isq_target_device.as_ref(),
        )?;
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
            in_proj_qkvz: Linear::new(qkvz_w, None),
            in_proj_ba: Linear::new(ba_w, None),
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
        })
    }
}

fn load_qkvz(vb: &ShardedVarBuilder, dims: &GdnDims, mode: &GdnWeightMode) -> Result<Tensor> {
    match mode {
        GdnWeightMode::MergedOnly => vb.get(
            (dims.qkvz_out_dim(), dims.hidden_size),
            "in_proj_qkvz.weight",
        ),
        GdnWeightMode::MergedWithFallback if vb.contains_tensor("in_proj_qkvz.weight") => vb.get(
            (dims.qkvz_out_dim(), dims.hidden_size),
            "in_proj_qkvz.weight",
        ),
        GdnWeightMode::MergedWithFallback => load_split_qkvz(vb, dims),
    }
}

fn load_split_qkvz(vb: &ShardedVarBuilder, dims: &GdnDims) -> Result<Tensor> {
    let qkv_w = vb.get(
        (dims.key_dim * 2 + dims.value_dim, dims.hidden_size),
        "in_proj_qkv.weight",
    )?;
    let z_w = vb.get((dims.value_dim, dims.hidden_size), "in_proj_z.weight")?;
    let q_w = qkv_w.narrow(0, 0, dims.key_dim)?;
    let k_w = qkv_w.narrow(0, dims.key_dim, dims.key_dim)?;
    let v_w = qkv_w.narrow(0, dims.key_dim * 2, dims.value_dim)?;
    let q_grouped = q_w.reshape((dims.num_k_heads, dims.head_k_dim, dims.hidden_size))?;
    let k_grouped = k_w.reshape((dims.num_k_heads, dims.head_k_dim, dims.hidden_size))?;
    let v_grouped = v_w.reshape((
        dims.num_k_heads,
        dims.v_per_group * dims.head_v_dim,
        dims.hidden_size,
    ))?;
    let z_grouped = z_w.reshape((
        dims.num_k_heads,
        dims.v_per_group * dims.head_v_dim,
        dims.hidden_size,
    ))?;
    Tensor::cat(&[q_grouped, k_grouped, v_grouped, z_grouped], 1)?
        .reshape((dims.qkvz_out_dim(), dims.hidden_size))
}

fn load_ba(vb: &ShardedVarBuilder, dims: &GdnDims, mode: &GdnWeightMode) -> Result<Tensor> {
    match mode {
        GdnWeightMode::MergedOnly => {
            vb.get((dims.ba_out_dim(), dims.hidden_size), "in_proj_ba.weight")
        }
        GdnWeightMode::MergedWithFallback if vb.contains_tensor("in_proj_ba.weight") => {
            vb.get((dims.ba_out_dim(), dims.hidden_size), "in_proj_ba.weight")
        }
        GdnWeightMode::MergedWithFallback => load_split_ba(vb, dims),
    }
}

fn load_split_ba(vb: &ShardedVarBuilder, dims: &GdnDims) -> Result<Tensor> {
    let b_w = vb.get((dims.num_v_heads, dims.hidden_size), "in_proj_b.weight")?;
    let a_w = vb.get((dims.num_v_heads, dims.hidden_size), "in_proj_a.weight")?;
    let b_grouped = b_w.reshape((dims.num_k_heads, dims.v_per_group, dims.hidden_size))?;
    let a_grouped = a_w.reshape((dims.num_k_heads, dims.v_per_group, dims.hidden_size))?;
    Tensor::cat(&[b_grouped, a_grouped], 1)?.reshape((dims.ba_out_dim(), dims.hidden_size))
}

fn move_to_target(tensor: Tensor, target_device: Option<&Device>) -> Result<Tensor> {
    if let Some(target_device) = target_device {
        tensor.to_device(target_device)
    } else {
        Ok(tensor)
    }
}
