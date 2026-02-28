#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared GatedDeltaNet (linear attention) layer used by Qwen3Next, Qwen3.5, and Qwen3.5-MoE.
//!
//! This module provides the core DeltaNet recurrence and causal conv1d primitives that are common
//! across all Qwen3 hybrid-attention models.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder};
use std::sync::Arc;

use crate::device_map::DeviceMapper;
use crate::layers::MatMul;

// ====================== DeltaNet Config Trait ======================

/// Trait to abstract DeltaNet-relevant config fields from model-specific Config structs.
pub trait DeltaNetConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn quantization_config(&self) -> &Option<QuantizedConfig>;

    /// Total key dimension = num_key_heads * key_head_dim
    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }

    /// Total value dimension = num_value_heads * value_head_dim
    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }

    /// Conv dim for GDN = key_dim * 2 + value_dim (q, k, v before split)
    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

// ====================== RMSNorm Gated (for GDN output) ======================

/// RMSNorm with gating: `rms_norm(x) * weight * silu(gate)`
pub struct RmsNormGated {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNormGated {
    pub fn new(
        size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
    }
}

// ====================== GDN layer cache ======================

#[derive(Debug)]
pub struct GdnLayerCache {
    /// Conv state: (batch, conv_dim, kernel_size)
    pub conv_state: Tensor,
    /// Recurrent state: (batch, num_v_heads, head_k_dim, head_v_dim)
    pub recurrent_state: Tensor,
    pub seqlen_offset: usize,
}

impl GdnLayerCache {
    pub fn new(cfg: &dyn DeltaNetConfig, dtype: DType, device: &Device) -> Result<Self> {
        let conv_dim = cfg.linear_conv_dim();
        let conv_state = Tensor::zeros((1, conv_dim, cfg.linear_conv_kernel_dim()), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (
                1,
                cfg.linear_num_value_heads(),
                cfg.linear_key_head_dim(),
                cfg.linear_value_head_dim(),
            ),
            dtype,
            device,
        )?;
        Ok(Self {
            conv_state,
            recurrent_state,
            seqlen_offset: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            seqlen_offset: self.seqlen_offset,
        }
    }
}

// ====================== GDN math functions ======================

pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

/// Recurrent gated delta rule (CPU/Metal fallback).
///
/// q, k: (batch, seq, num_v_heads, head_k_dim)
/// v:    (batch, seq, num_v_heads, head_v_dim)
/// g:    (batch, seq, num_v_heads)
/// beta: (batch, seq, num_v_heads)
/// state: (batch, num_v_heads, head_k_dim, head_v_dim)
///
/// Returns: (batch, seq, num_v_heads, head_v_dim)
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();
    let k_head_dim = q.dim(D::Minus1)?;
    let scale = 1.0 / (k_head_dim as f64).sqrt();

    let q = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        let k_exp = k_t.unsqueeze(D::Minus1)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;

        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;

        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    let out = Tensor::stack(&outputs, 2)?;
    out.transpose(1, 2)?.contiguous()?.to_dtype(dtype)
}

// ====================== GDN Projection variants ======================

/// Projection strategy for GDN input. Qwen3Next and Qwen3.5 differ in how they pack weights.
#[allow(dead_code)]
pub enum GdnProjection {
    /// Qwen3Next: fused in_proj_qkvz (key_dim*2 + value_dim*2) + in_proj_ba (num_v_heads*2)
    FusedQkvzBa {
        in_proj_qkvz: Linear,
        in_proj_ba: Linear,
    },
    /// Qwen3.5: split in_proj_qkv (key_dim*2 + value_dim) + in_proj_z (value_dim) + in_proj_b (num_v_heads) + in_proj_a (num_v_heads)
    SplitQkvZa {
        in_proj_qkv: Linear,
        in_proj_z: Linear,
        in_proj_b: Linear,
        in_proj_a: Linear,
    },
}

// ====================== Gated Delta Net layer ======================

/// Projected outputs from the GDN input projections.
/// z is 4D (batch, seq, num_v_heads, head_v_dim), others are flat for conv.
struct GdnProjected {
    /// (batch, seq, key_dim)
    q: Tensor,
    /// (batch, seq, key_dim)
    k: Tensor,
    /// (batch, seq, value_dim)
    v_flat: Tensor,
    /// (batch, seq, num_v_heads, head_v_dim) — gating signal for norm
    z: Tensor,
    /// (batch, seq, num_v_heads)
    b: Tensor,
    /// (batch, seq, num_v_heads)
    a: Tensor,
}

pub struct GatedDeltaNet {
    pub projection: GdnProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: Arc<dyn QuantMethod>,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl GatedDeltaNet {
    /// Load GDN layer with fused Qwen3Next projection (in_proj_qkvz + in_proj_ba).
    #[allow(dead_code)]
    pub fn load_qwen3next(
        vb: ShardedVarBuilder,
        cfg: &dyn DeltaNetConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };

        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim();

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkvz_out = key_dim * 2 + value_dim * 2;
        let mut qkvz_w = vb_la.get((qkvz_out, cfg.hidden_size()), "in_proj_qkvz.weight")?;
        let mut ba_w = vb_la.get((num_v_heads * 2, cfg.hidden_size()), "in_proj_ba.weight")?;

        let conv_dim = key_dim * 2 + value_dim;
        let mut conv1d_weight = vb_la.get((conv_dim, 1, conv_kernel_size), "conv1d.weight")?;
        let mut dt_bias = vb_la.get(num_v_heads, "dt_bias")?;
        let mut a_log = vb_la.get(num_v_heads, "A_log")?;

        if let Some(ref target_dev) = isq_target_device {
            qkvz_w = qkvz_w.to_device(target_dev)?;
            ba_w = ba_w.to_device(target_dev)?;
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            cfg.hidden_size(),
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        Ok(Self {
            projection: GdnProjection::FusedQkvzBa {
                in_proj_qkvz: Linear::new(qkvz_w, None),
                in_proj_ba: Linear::new(ba_w, None),
            },
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        })
    }

    /// Load GDN layer with split Qwen3.5 projection (in_proj_qkv + in_proj_z + in_proj_b + in_proj_a).
    pub fn load_qwen3_5(
        vb: ShardedVarBuilder,
        cfg: &dyn DeltaNetConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let isq_target_device = if loading_isq {
            mapper.device_for(layer_idx, false).cloned()
        } else {
            None
        };

        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim();

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkv_out = key_dim * 2 + value_dim;
        let mut qkv_w = vb_la.get((qkv_out, cfg.hidden_size()), "in_proj_qkv.weight")?;
        let mut z_w = vb_la.get((value_dim, cfg.hidden_size()), "in_proj_z.weight")?;
        let mut b_w = vb_la.get((num_v_heads, cfg.hidden_size()), "in_proj_b.weight")?;
        let mut a_w = vb_la.get((num_v_heads, cfg.hidden_size()), "in_proj_a.weight")?;

        let conv_dim = key_dim * 2 + value_dim;
        let mut conv1d_weight = vb_la.get((conv_dim, 1, conv_kernel_size), "conv1d.weight")?;
        let mut dt_bias = vb_la.get(num_v_heads, "dt_bias")?;
        let mut a_log = vb_la.get(num_v_heads, "A_log")?;

        if let Some(ref target_dev) = isq_target_device {
            qkv_w = qkv_w.to_device(target_dev)?;
            z_w = z_w.to_device(target_dev)?;
            b_w = b_w.to_device(target_dev)?;
            a_w = a_w.to_device(target_dev)?;
            conv1d_weight = conv1d_weight.to_device(target_dev)?;
            dt_bias = dt_bias.to_device(target_dev)?;
            a_log = a_log.to_device(target_dev)?;
        }

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps(),
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            cfg.hidden_size(),
            cfg.quantization_config(),
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        Ok(Self {
            projection: GdnProjection::SplitQkvZa {
                in_proj_qkv: Linear::new(qkv_w, None),
                in_proj_z: Linear::new(z_w, None),
                in_proj_b: Linear::new(b_w, None),
                in_proj_a: Linear::new(a_w, None),
            },
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
        })
    }

    /// Project inputs and unpack into (q, k, v_flat, z, b, a) based on projection variant.
    fn project_inputs(&self, x: &Tensor) -> Result<GdnProjected> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let v_per_group = self.num_v_heads / self.num_k_heads;

        match &self.projection {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = in_proj_qkvz.forward(x)?;
                let mixed_ba = in_proj_ba.forward(x)?;

                let group_size_qkvz = 2 * self.head_k_dim + 2 * v_per_group * self.head_v_dim;
                let mixed_qkvz =
                    mixed_qkvz.reshape((batch_size, seq_len, self.num_k_heads, group_size_qkvz))?;

                let group_size_ba = 2 * v_per_group;
                let mixed_ba =
                    mixed_ba.reshape((batch_size, seq_len, self.num_k_heads, group_size_ba))?;

                let mut offset = 0;
                let q = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let k = mixed_qkvz.narrow(D::Minus1, offset, self.head_k_dim)?;
                offset += self.head_k_dim;
                let v = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;
                offset += v_per_group * self.head_v_dim;
                let z = mixed_qkvz.narrow(D::Minus1, offset, v_per_group * self.head_v_dim)?;

                let b = mixed_ba.narrow(D::Minus1, 0, v_per_group)?;
                let a = mixed_ba.narrow(D::Minus1, v_per_group, v_per_group)?;

                // Reshape to per-head
                let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
                let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
                let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
                let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

                let q = q.reshape((batch_size, seq_len, self.key_dim))?;
                let k = k.reshape((batch_size, seq_len, self.key_dim))?;
                let v_flat = v.reshape((batch_size, seq_len, self.value_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                })
            }
            GdnProjection::SplitQkvZa {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = in_proj_qkv.forward(x)?;
                let z_full = in_proj_z.forward(x)?;
                let b = in_proj_b.forward(x)?;
                let a = in_proj_a.forward(x)?;

                let q = proj_qkv.narrow(D::Minus1, 0, self.key_dim)?;
                let k = proj_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
                let v_flat = proj_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

                let z = z_full.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

                Ok(GdnProjected {
                    q,
                    k,
                    v_flat,
                    z,
                    b,
                    a,
                })
            }
        }
    }

    /// Run the full GDN forward pass.
    pub fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();
        let v_per_group = self.num_v_heads / self.num_k_heads;

        // 1. Project input
        let projected = self.project_inputs(x)?;
        let GdnProjected {
            q,
            k,
            v_flat,
            z,
            b,
            a,
        } = projected;

        // 2. Concatenate q, k, v for conv1d: (batch, seq, conv_dim)
        let mixed_qkv = Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?;

        // 3. Apply causal conv1d (includes silu activation)
        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        // 4. Split back after conv and reshape to per-head
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        // 5. Compute beta and g (3D: batch, seq, num_v_heads)
        let (beta, g) = self.compute_gating(&b, &a, dtype)?;

        // 6. If num_v_heads > num_k_heads, repeat_interleave q and k
        let (q, k) = if v_per_group > 1 {
            let q = q
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            let k = k
                .unsqueeze(3)?
                .repeat((1, 1, 1, v_per_group, 1))?
                .reshape((batch_size, seq_len, self.num_v_heads, self.head_k_dim))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. L2-normalize q and k
        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        // 8. Apply recurrence
        let y = self.apply_recurrence(&q, &k, &v, &g, &beta, batch_size, seq_len, dtype, cache)?;

        cache.seqlen_offset += seq_len;

        // 9. Apply RMSNormGated: flatten to 2D, apply norm with z as gate, reshape back
        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        // 10. Output projection
        let original_dtype = x.dtype();
        let mut y_proj = y;
        if let Some(t) = self.out_proj.quantized_act_type() {
            y_proj = y_proj.to_dtype(t)?;
        }
        let mut res = MatMul.qmethod_matmul(&y_proj, &*self.out_proj)?;
        if self.out_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    /// Compute beta (sigmoid of b) and g (gating decay from a, A_log, dt_bias).
    fn compute_gating(&self, b: &Tensor, a: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            if b.device().is_cuda() {
                let b_flat = b.contiguous()?.flatten_all()?;
                let a_flat = a.contiguous()?.flatten_all()?;
                let a_log_f32 = self.a_log.to_dtype(DType::F32)?.contiguous()?;
                let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32)?.contiguous()?;
                let (beta_flat, g_flat) = crate::cuda::gdn::fused_gdn_gating_cuda(
                    &b_flat,
                    &a_flat,
                    &a_log_f32,
                    &dt_bias_f32,
                )?;
                let shape = b.shape();
                return Ok((beta_flat.reshape(shape)?, g_flat.reshape(shape)?));
            }
        }
        let beta = candle_nn::ops::sigmoid(b)?;
        let a_f = a.to_dtype(DType::F32)?;
        let dt_bias_expanded = self
            .dt_bias
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let g = self
            .a_log
            .to_dtype(DType::F32)?
            .exp()?
            .neg()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
            .to_dtype(dtype)?;
        Ok((beta, g))
    }

    /// Apply recurrence (CUDA or CPU/Metal fallback).
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn apply_recurrence(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        batch_size: usize,
        seq_len: usize,
        dtype: DType,
        cache: &mut GdnLayerCache,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if q.device().is_cuda() {
                let num_heads = self.num_v_heads;
                let k_head = self.head_k_dim;
                let v_head = self.head_v_dim;
                let scale = 1.0 / (k_head as f64).sqrt();

                let q_bh = (q.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)? * scale)?
                    .reshape((batch_size * num_heads, seq_len, k_head))?
                    .contiguous()?;
                let k_bh = k
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(DType::F32)?
                    .reshape((batch_size * num_heads, seq_len, k_head))?
                    .contiguous()?;
                let v_bh = v
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(DType::F32)?
                    .reshape((batch_size * num_heads, seq_len, v_head))?
                    .contiguous()?;
                let g_bh = g
                    .to_dtype(DType::F32)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((batch_size * num_heads, seq_len))?
                    .contiguous()?;
                let beta_bh = beta
                    .to_dtype(DType::F32)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .reshape((batch_size * num_heads, seq_len))?
                    .contiguous()?;

                let mut state_flat = cache
                    .recurrent_state
                    .to_dtype(DType::F32)?
                    .reshape((batch_size * num_heads, k_head, v_head))?
                    .contiguous()?;

                let out_bh = crate::cuda::gdn::gated_delta_rule_recurrence_cuda(
                    &q_bh,
                    &k_bh,
                    &v_bh,
                    &g_bh,
                    &beta_bh,
                    &mut state_flat,
                )?;

                cache.recurrent_state = state_flat
                    .reshape((batch_size, num_heads, k_head, v_head))?
                    .to_dtype(cache.recurrent_state.dtype())?;

                return out_bh
                    .reshape((batch_size, num_heads, seq_len, v_head))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(dtype);
            }
        }

        gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
    }

    /// Single-step causal conv1d update for decode.
    fn causal_conv1d_update(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (_batch, seq_len, _conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let conv_state = cache.conv_state.contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &conv_state,
                self.conv_kernel_size,
                true,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let state_len = cache.conv_state.dim(2)?;
        let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
        let new_len = hidden_new.dim(2)?;
        cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

        let weight = self
            .conv1d_weight
            .squeeze(1)?
            .to_dtype(hidden_new.dtype())?;
        let mut conv_outputs = Vec::with_capacity(seq_len);
        let total_len = hidden_new.dim(2)?;
        for i in (total_len - seq_len)..total_len {
            let window =
                hidden_new.narrow(2, i + 1 - self.conv_kernel_size, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = candle_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }

    /// Full sequence causal conv1d for prefill.
    fn causal_conv1d_full(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let x_t = x.transpose(1, 2)?.contiguous()?;

        #[cfg(feature = "cuda")]
        if x_t.device().is_cuda() {
            let weight = self
                .conv1d_weight
                .squeeze(1)?
                .to_dtype(x_t.dtype())?
                .contiguous()?;
            let (output, new_conv_state) = crate::cuda::gdn::causal_conv1d_cuda(
                &x_t,
                &weight,
                &cache.conv_state,
                self.conv_kernel_size,
                false,
            )?;
            cache.conv_state = new_conv_state;
            return output.transpose(1, 2);
        }

        let pad_width = self.conv_kernel_size.saturating_sub(seq_len);
        cache.conv_state = if pad_width > 0 {
            let zeros =
                Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[zeros, x_t.clone()], 2)?
        } else {
            x_t.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)?
        };

        let padded_t = Tensor::cat(
            &[
                Tensor::zeros(
                    (batch_size, conv_dim, self.conv_kernel_size - 1),
                    x_t.dtype(),
                    x_t.device(),
                )?,
                x_t,
            ],
            2,
        )?;

        let weight = self.conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;

        let mut conv_outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let window = padded_t.narrow(2, i, self.conv_kernel_size)?;
            let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
            conv_outputs.push(out);
        }
        let out = Tensor::stack(&conv_outputs, 2)?;
        let out = candle_nn::ops::silu(&out)?;
        out.transpose(1, 2)
    }
}
