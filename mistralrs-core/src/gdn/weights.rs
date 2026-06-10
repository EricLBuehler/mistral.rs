use candle_core::{Device, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{
    AfqLayer, Comm, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use std::sync::Arc;

use crate::device_map::DeviceMapper;

use super::config::{GdnConfig, GdnDims};
use super::norm::RmsNormGated;

pub enum GdnWeightMode {
    MergedOnly,
    MergedWithFallback,
}

// MLX-AFQ checkpoints (e.g. mlx-community/Qwen3.6-*-4bit) ship the four
// in_proj_{qkv,z,b,a} weights as separate AFQ-packed tensors with their own
// scales+biases. We can't dequantise-and-concat at load time (the activated
// memory cost would be huge), so we keep them split and combine on the
// activation side in `forward`.
pub enum GdnInProj {
    Merged(Arc<dyn QuantMethod>),
    SplitAfq {
        qkv: Arc<dyn QuantMethod>,
        z: Arc<dyn QuantMethod>,
        b: Arc<dyn QuantMethod>,
        a: Arc<dyn QuantMethod>,
    },
}

pub struct GdnWeights {
    pub in_proj: GdnInProj,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
}

fn is_afq(cfg: &Option<QuantizedConfig>) -> bool {
    matches!(cfg, Some(QuantizedConfig::Afq { .. }))
}

impl GdnInProj {
    // Activation-side projection. Merged path delegates to the single combined
    // matmul. SplitAfq runs four separate AFQ matmuls and re-creates the same
    // per-head interleaved layout that the merged-weight path would have
    // produced, so `GdnProjection::from_packed` downstream can unpack it
    // unchanged.
    pub fn forward(
        &self,
        x: &candle_core::Tensor,
        dims: &GdnDims,
    ) -> candle_core::Result<candle_core::Tensor> {
        use candle_core::D;
        match self {
            GdnInProj::Merged(inner) => inner.forward(x),
            GdnInProj::SplitAfq { qkv, z, b, a } => {
                // MLX-style activation path: each AfqLayer matmul produces
                // a flat per-type output, then we split / per-head reshape /
                // concat on the activation side. Mathematically matches the
                // Python `mlx_lm/models/qwen3_5.py` GatedDeltaNet code:
                //
                //   qkv = self.in_proj_qkv(x)         # (B, S, key_dim*2 + value_dim) flat
                //   z   = self.in_proj_z(x)           # (B, S, value_dim) flat
                //   ...
                //   q, k, v = split(qkv, [key_dim, 2*key_dim])
                //            each then reshape(B, S, num_*_heads, *_head_dim)
                //
                // After concat-along-last we re-flatten to the per-head
                // interleaved layout `GdnProjection::from_packed` expects.
                let qkv_out = qkv.forward(x)?; // (..., key_dim*2 + value_dim)
                let z_out = z.forward(x)?; // (..., value_dim)
                let b_out = b.forward(x)?; // (..., num_v_heads)
                let a_out = a.forward(x)?; // (..., num_v_heads)

                let last = qkv_out.dims().len() - 1;
                let q_flat = qkv_out.narrow(last, 0, dims.key_dim)?;
                let k_flat = qkv_out.narrow(last, dims.key_dim, dims.key_dim)?;
                let v_flat = qkv_out.narrow(last, 2 * dims.key_dim, dims.value_dim)?;

                let lead: Vec<usize> = x.dims().iter().take(x.dims().len() - 1).copied().collect();
                let q_shape: Vec<usize> = lead
                    .iter()
                    .copied()
                    .chain([dims.num_k_heads, dims.head_k_dim])
                    .collect();
                let v_shape: Vec<usize> = lead
                    .iter()
                    .copied()
                    .chain([dims.num_k_heads, dims.v_per_group * dims.head_v_dim])
                    .collect();
                let ba_shape: Vec<usize> = lead
                    .iter()
                    .copied()
                    .chain([dims.num_k_heads, dims.v_per_group])
                    .collect();

                let q_g = q_flat.reshape(q_shape.as_slice())?;
                let k_g = k_flat.reshape(q_shape.as_slice())?;
                let v_g = v_flat.reshape(v_shape.as_slice())?;
                let z_g = z_out.reshape(v_shape.as_slice())?;
                let b_g = b_out.reshape(ba_shape.as_slice())?;
                let a_g = a_out.reshape(ba_shape.as_slice())?;

                let qkvz = candle_core::Tensor::cat(&[q_g, k_g, v_g, z_g], D::Minus1)?;
                let ba = candle_core::Tensor::cat(&[b_g, a_g], D::Minus1)?;

                let qkvz_flat_shape: Vec<usize> =
                    lead.iter().copied().chain([dims.qkvz_out_dim()]).collect();
                let ba_flat_shape: Vec<usize> =
                    lead.iter().copied().chain([dims.ba_out_dim()]).collect();
                let qkvz = qkvz.contiguous()?.reshape(qkvz_flat_shape)?;
                let ba = ba.contiguous()?.reshape(ba_flat_shape)?;
                candle_core::Tensor::cat(&[qkvz, ba], D::Minus1)
            }
        }
    }

    // Dequantised weights for the rare callers that need them
    // (currently `GatedDeltaNet::residual_input_projection_tensors`). For the
    // SplitAfq path this is best-effort: callers should usually use `forward`.
    pub fn dequantize_concat(
        &self,
        dims: &GdnDims,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        match self {
            GdnInProj::Merged(inner) => {
                let w = inner.dequantize_w()?;
                let qkvz = w.narrow(0, 0, dims.qkvz_out_dim())?;
                let ba = w.narrow(0, dims.qkvz_out_dim(), dims.ba_out_dim())?;
                Ok((qkvz, ba))
            }
            GdnInProj::SplitAfq { qkv, z, b, a } => {
                // Mirror exactly what load_split_qkvz does: split the flat
                // [q | k | v] qkv weight into q, k, v slices first, THEN
                // per-head reshape each piece. Same for b/a.
                let qkv_w = qkv.dequantize_w()?; // (key_dim*2 + value_dim, hidden_size)
                let z_w = z.dequantize_w()?; // (value_dim, hidden_size)
                let b_w = b.dequantize_w()?; // (num_v_heads, hidden_size)
                let a_w = a.dequantize_w()?; // (num_v_heads, hidden_size)
                let q_w = qkv_w.narrow(0, 0, dims.key_dim)?;
                let k_w = qkv_w.narrow(0, dims.key_dim, dims.key_dim)?;
                let v_w = qkv_w.narrow(0, dims.key_dim * 2, dims.value_dim)?;
                let q_g = q_w.reshape((dims.num_k_heads, dims.head_k_dim, dims.hidden_size))?;
                let k_g = k_w.reshape((dims.num_k_heads, dims.head_k_dim, dims.hidden_size))?;
                let v_g = v_w.reshape((
                    dims.num_k_heads,
                    dims.v_per_group * dims.head_v_dim,
                    dims.hidden_size,
                ))?;
                let z_g = z_w.reshape((
                    dims.num_k_heads,
                    dims.v_per_group * dims.head_v_dim,
                    dims.hidden_size,
                ))?;
                let b_g = b_w.reshape((dims.num_k_heads, dims.v_per_group, dims.hidden_size))?;
                let a_g = a_w.reshape((dims.num_k_heads, dims.v_per_group, dims.hidden_size))?;
                let qkvz = candle_core::Tensor::cat(&[q_g, k_g, v_g, z_g], 1)?
                    .reshape((dims.qkvz_out_dim(), dims.hidden_size))?;
                let ba = candle_core::Tensor::cat(&[b_g, a_g], 1)?
                    .reshape((dims.ba_out_dim(), dims.hidden_size))?;
                Ok((qkvz, ba))
            }
        }
    }
}

pub struct GdnWeightLoadCtx<'a> {
    pub cfg: &'a dyn GdnConfig,
    pub dims: &'a GdnDims,
    pub mapper: &'a dyn DeviceMapper,
    pub layer_idx: usize,
    pub loading_isq: bool,
    pub comm: &'a Arc<Comm>,
    pub weight_mode: GdnWeightMode,
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
            weight_mode,
        } = ctx;
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };
        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qc = cfg.quantization_config();
        let in_proj = if is_afq(qc) {
            // MLX AFQ split layout: four separate AfqLayer's. Activation-side
            // combine happens in GdnInProj::forward.
            let qkv = AfqLayer::afq_linear_b(
                dims.hidden_size,
                dims.key_dim * 2 + dims.value_dim,
                qc.as_ref().unwrap(),
                false,
                vb_la.pp("in_proj_qkv"),
            )?;
            let z = AfqLayer::afq_linear_b(
                dims.hidden_size,
                dims.value_dim,
                qc.as_ref().unwrap(),
                false,
                vb_la.pp("in_proj_z"),
            )?;
            let b = AfqLayer::afq_linear_b(
                dims.hidden_size,
                dims.num_v_heads,
                qc.as_ref().unwrap(),
                false,
                vb_la.pp("in_proj_b"),
            )?;
            let a = AfqLayer::afq_linear_b(
                dims.hidden_size,
                dims.num_v_heads,
                qc.as_ref().unwrap(),
                false,
                vb_la.pp("in_proj_a"),
            )?;
            GdnInProj::SplitAfq { qkv, z, b, a }
        } else {
            let qkvz_w = move_to_target(
                load_qkvz(&vb_la, dims, &weight_mode)?,
                isq_target_device.as_ref(),
            )?;
            let ba_w = move_to_target(
                load_ba(&vb_la, dims, &weight_mode)?,
                isq_target_device.as_ref(),
            )?;
            let in_proj_w = Tensor::cat(&[qkvz_w, ba_w], 0)?;
            GdnInProj::Merged(ReplicatedLayer::from_linear(
                Linear::new(in_proj_w, None),
                vb_la.pp("in_proj"),
            )?)
        };
        // MLX-AFQ ships conv1d.weight transposed: (out, kernel, in=1) instead
        // of candle's (out, in=1, kernel). Try the native layout first; fall
        // back to a permuted load for MLX checkpoints.
        let conv1d_weight = match vb_la
            .get((dims.conv_dim, 1, dims.conv_kernel_size), "conv1d.weight")
        {
            Ok(t) => move_to_target(t, isq_target_device.as_ref())?,
            Err(_) => {
                let raw = vb_la.get((dims.conv_dim, dims.conv_kernel_size, 1), "conv1d.weight")?;
                move_to_target(
                    raw.permute((0, 2, 1))?.contiguous()?,
                    isq_target_device.as_ref(),
                )?
            }
        };
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
            in_proj,
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
