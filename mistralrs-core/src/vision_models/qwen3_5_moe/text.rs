#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, Linear};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::{LayerType, TextConfig};
use crate::{
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{self, Qwen3VLRotaryEmbedding, RmsNorm, Sdpa},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

// ====================== RMSNorm Gated (for GDN output) ======================

/// RMSNorm with gating: `rms_norm(x) * weight * silu(gate)`
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn new(
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

    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
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
struct GdnLayerCache {
    conv_state: Tensor,
    recurrent_state: Tensor,
    seqlen_offset: usize,
}

impl GdnLayerCache {
    fn new(cfg: &TextConfig, dtype: DType, device: &Device) -> Result<Self> {
        let conv_dim = cfg.linear_conv_dim();
        let conv_state = Tensor::zeros((1, conv_dim, cfg.linear_conv_kernel_dim), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (
                1,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
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

    fn reset(&mut self) -> Result<()> {
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

fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

fn gated_delta_rule_recurrence(
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

// ====================== Gated Delta Net layer ======================

struct GatedDeltaNet {
    in_proj_qkvz: Linear,
    in_proj_ba: Linear,
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: RmsNormGated,
    out_proj: Arc<dyn QuantMethod>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_kernel_size: usize,
    key_dim: usize,
    value_dim: usize,
}

impl GatedDeltaNet {
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
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

        let num_k_heads = cfg.linear_num_key_heads;
        let num_v_heads = cfg.linear_num_value_heads;
        let head_k_dim = cfg.linear_key_head_dim;
        let head_v_dim = cfg.linear_value_head_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_kernel_size = cfg.linear_conv_kernel_dim;

        let vb_la = mapper.set_device(layer_idx, vb.pp("linear_attn"), loading_isq);

        let qkvz_out = key_dim * 2 + value_dim * 2;
        let mut qkvz_w = vb_la.get((qkvz_out, cfg.hidden_size), "in_proj_qkvz.weight")?;
        let mut ba_w = vb_la.get((num_v_heads * 2, cfg.hidden_size), "in_proj_ba.weight")?;

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

        let in_proj_qkvz = Linear::new(qkvz_w, None);
        let in_proj_ba = Linear::new(ba_w, None);

        let norm = RmsNormGated::new(
            head_v_dim,
            cfg.rms_norm_eps,
            vb_la.pp("norm"),
            isq_target_device.as_ref(),
        )?;

        let out_proj = RowParallelLayer::new(
            value_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_la.pp("out_proj"),
        )?;

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
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

    fn forward(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden) = x.dims3()?;
        let dtype = x.dtype();

        let mixed_qkvz = self.in_proj_qkvz.forward(x)?;
        let mixed_ba = self.in_proj_ba.forward(x)?;

        let v_per_group = self.num_v_heads / self.num_k_heads;
        let group_size_qkvz = 2 * self.head_k_dim + 2 * v_per_group * self.head_v_dim;
        let mixed_qkvz =
            mixed_qkvz.reshape((batch_size, seq_len, self.num_k_heads, group_size_qkvz))?;

        let group_size_ba = 2 * v_per_group;
        let mixed_ba = mixed_ba.reshape((batch_size, seq_len, self.num_k_heads, group_size_ba))?;

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

        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let z = z.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let b = b.reshape((batch_size, seq_len, self.num_v_heads))?;
        let a = a.reshape((batch_size, seq_len, self.num_v_heads))?;

        let q = q.reshape((batch_size, seq_len, self.key_dim))?;
        let k = k.reshape((batch_size, seq_len, self.key_dim))?;
        let v_flat = v.reshape((batch_size, seq_len, self.value_dim))?;

        let mixed_qkv = Tensor::cat(&[&q, &k, &v_flat], D::Minus1)?;

        let mixed_qkv = if cache.seqlen_offset > 0 && seq_len == 1 {
            self.causal_conv1d_update(&mixed_qkv, cache)?
        } else {
            self.causal_conv1d_full(&mixed_qkv, cache)?
        };

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let (beta, g) = {
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
                    (beta_flat.reshape(shape)?, g_flat.reshape(shape)?)
                } else {
                    let beta = candle_nn::ops::sigmoid(&b)?;
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
                    (beta, g)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let beta = candle_nn::ops::sigmoid(&b)?;
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
                (beta, g)
            }
        };

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

        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        let y = {
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

                    out_bh
                        .reshape((batch_size, num_heads, seq_len, v_head))?
                        .transpose(1, 2)?
                        .contiguous()?
                        .to_dtype(dtype)?
                } else {
                    gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut cache.recurrent_state)?
            }
        };

        cache.seqlen_offset += seq_len;

        let z_shape = z.shape().clone();
        let y = y.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let y = self.norm.forward(&y, &z)?;
        let y = y.reshape(z_shape)?;
        let y = y.reshape((batch_size, seq_len, self.value_dim))?;

        let original_dtype = x.dtype();
        let mut y_proj = y;
        if let Some(t) = self.out_proj.quantized_act_type() {
            y_proj = y_proj.to_dtype(t)?;
        }
        let mut res = mistralrs_quant::MatMul.qmethod_matmul(&y_proj, &*self.out_proj)?;
        if self.out_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

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

// ====================== Full Attention layer with MRoPE ======================

#[allow(dead_code)]
struct FullAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    rot_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let vb_sa = mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // q_proj outputs num_heads * head_dim * 2 (doubled for gate)
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(num_kv_heads, head_dim, comm);
        let k_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;

        let vb_sa_norms = mapper.set_device(layer_idx, vb.pp("self_attn"), false);
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb_sa_norms.pp("k_norm"))?;

        let rot_dim = cfg.rot_dim();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            rot_dim,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(num_kv_heads, num_heads, comm),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            x = x.to_dtype(t)?;
        }
        let mut q_gate = mistralrs_quant::MatMul.qmethod_matmul(&x, &*self.q_proj)?;
        let mut k = mistralrs_quant::MatMul.qmethod_matmul(&x, &*self.k_proj)?;
        let mut v = mistralrs_quant::MatMul.qmethod_matmul(&x, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q_gate = q_gate.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        // Split q_gate into q and gate
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        // Reshape to (batch, heads, seq, head_dim)
        let (mut q, mut k, v) = if seq_len != 1 {
            let q = q.transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, seq_len, self.head_dim))?;
            (q, k, v)
        };

        // Apply QK norm
        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        // Apply partial MRoPE: split into rotated and pass-through portions
        if self.rot_dim < self.head_dim {
            let mut q_rot = q.narrow(D::Minus1, 0, self.rot_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;
            let mut k_rot = k.narrow(D::Minus1, 0, self.rot_dim)?;
            let k_pass = k.narrow(D::Minus1, self.rot_dim, self.head_dim - self.rot_dim)?;

            self.rotary_emb.forward(cos_sin, &mut q_rot, &mut k_rot)?;
            q = Tensor::cat(&[q_rot, q_pass], D::Minus1)?;
            k = Tensor::cat(&[k_rot, k_pass], D::Minus1)?;
        } else {
            self.rotary_emb.forward(cos_sin, &mut q, &mut k)?;
        }

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Standard attention
        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (cache_k, cache_v) = kv_cache.append(&k, &v)?;
                Sdpa.run_attention(
                    &q,
                    &cache_k.contiguous()?,
                    &cache_v.contiguous()?,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            y = y.to_dtype(t)?;
        }
        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        // Apply output gate: y = y * sigmoid(gate)
        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        y = y.broadcast_mul(&gate)?;

        let mut res = mistralrs_quant::MatMul.qmethod_matmul(&y, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

// ====================== MoE ======================

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: crate::layers::Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quant_config: &Option<mistralrs_quant::QuantizedConfig>,
        act_fn: crate::layers::Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = ColumnParallelLayer::new(
            hidden_size,
            intermediate_size,
            quant_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = RowParallelLayer::new(
            intermediate_size,
            hidden_size,
            quant_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let gate = mistralrs_quant::MatMul.qmethod_matmul(&xs, &*self.gate_proj)?;
        let up = mistralrs_quant::MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let activated = crate::ops::mul_and_act(&gate, &up, self.act_fn)?;
        let mut res = mistralrs_quant::MatMul.qmethod_matmul(&activated, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
    }
}

struct SparseMoeBlock {
    gate: Linear,
    experts: MoEExperts,
    shared_expert: Mlp,
    shared_expert_gate: Linear,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            vb.pp("gate").set_device(layer_device.clone()),
        )?;

        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_expert = Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        let mut seg_w = vb
            .pp("shared_expert_gate")
            .get((1, cfg.hidden_size), "weight")?;
        if loading_isq {
            seg_w = seg_w.to_device(&layer_device)?;
        }
        let shared_expert_gate = Linear::new(seg_w, None);

        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        let topk_ids = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;

        let mut topk_weights = routing_weights.gather(&topk_ids, D::Minus1)?;

        if self.norm_topk_prob {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?;
        }

        let mut y = self.experts.forward(xs, topk_weights, &topk_ids)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        let shared_out = self.shared_expert.forward(xs)?;
        let shared_gate = candle_nn::ops::sigmoid(
            &self
                .shared_expert_gate
                .forward(&xs.reshape(((), hidden_dim))?)?,
        )?;
        let shared_gate = shared_gate.reshape((b_size, seq_len, 1))?;
        let shared_out = shared_out.broadcast_mul(&shared_gate)?;

        y + shared_out
    }

    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        let mut layers = self.experts.get_isq_layers();
        layers.extend(self.shared_expert.get_isq_layers());
        layers
    }
}

// ====================== Decoder Layer ======================

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    moe: SparseMoeBlock,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_attention(
        &self,
        x: &Tensor,
        attention_mask: Option<&Tensor>,
        cos_sin: &(Tensor, Tensor),
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let attn = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => attn,
            _ => candle_core::bail!("Expected full attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let attn_out = attn.forward(
            &x,
            attention_mask,
            cos_sin,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let x = (attn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.moe.forward(&normed)?;
        ffn_out + residual
    }

    fn forward_linear(&self, x: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let gdn = match &self.layer_impl {
            LayerImpl::LinearAttention(gdn) => gdn,
            _ => candle_core::bail!("Expected linear attention layer"),
        };
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let gdn_out = gdn.forward(&x, cache)?;
        let x = (gdn_out + residual)?;
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let ffn_out = self.moe.forward(&normed)?;
        ffn_out + residual
    }
}

// ====================== Local hybrid cache ======================

enum LocalLayerCache {
    Attention,
    LinearAttention(GdnLayerCache),
}

struct LocalHybridCache {
    caches: Vec<LocalLayerCache>,
}

impl LocalHybridCache {
    fn new(
        layer_types: &[LayerType],
        cfg: &TextConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(layer_types.len());
        for lt in layer_types {
            match lt {
                LayerType::FullAttention => caches.push(LocalLayerCache::Attention),
                LayerType::LinearAttention => {
                    caches.push(LocalLayerCache::LinearAttention(GdnLayerCache::new(
                        cfg, dtype, device,
                    )?));
                }
            }
        }
        Ok(Self { caches })
    }
}

// ====================== Text Model ======================

pub struct Qwen3_5MoeTextModel {
    embed_tokens: Embedding,
    pub(super) norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    layer_types: Vec<LayerType>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    local_cache: Arc<Mutex<LocalHybridCache>>,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
}

impl Qwen3_5MoeTextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let vb_m = if vb.contains_tensor("language_model.model.embed_tokens.weight") {
            vb.pp("language_model").pp("model")
        } else {
            vb.pp("model").pp("language_model")
        };

        let embed_tokens = layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let layer_types = cfg.layer_types();

        // Create MRoPE embeddings (one per device, using rot_dim not head_dim)
        let rot_dim = cfg.rot_dim();
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if *layer_type != LayerType::FullAttention {
                continue;
            }
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.entry(device.location()).or_insert_with(|| {
                Arc::new(
                    Qwen3VLRotaryEmbedding::new(
                        cfg.rope_theta as f32,
                        rot_dim,
                        device,
                        cfg.rope_scaling.mrope_section.clone(),
                    )
                    .expect("Failed to create rotary embedding"),
                )
            });
        }

        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;

            let layer_impl = match layer_types[layer_idx] {
                LayerType::FullAttention => {
                    let device = mapper
                        .device_for(layer_idx, false)
                        .unwrap_or(&normal_loading_metadata.real_device);
                    let rotary_emb = ropes
                        .get(&device.location())
                        .expect("No RoPE for device location!")
                        .clone();
                    let paged_attn = match &attention_mechanism {
                        AttentionImplementation::Eager => None,
                        AttentionImplementation::PagedAttention => {
                            Some(PagedAttention::new(cfg.head_dim, device, None)?)
                        }
                    };
                    LayerImpl::FullAttention(FullAttention::load(
                        vb_l.pp(layer_idx),
                        cfg,
                        &*mapper,
                        layer_idx,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                        paged_attn,
                        &comm,
                    )?)
                }
                LayerType::LinearAttention => LayerImpl::LinearAttention(GatedDeltaNet::load(
                    vb_l.pp(layer_idx),
                    cfg,
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    &comm,
                )?),
            };

            let input_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb_l.pp(layer_idx).pp("input_layernorm"), false),
            )?;
            let post_attention_layernorm = RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("post_attention_layernorm"),
                    false,
                ),
            )?;

            let moe = SparseMoeBlock::new(
                cfg,
                mapper.set_device(
                    layer_idx,
                    vb_l.pp(layer_idx).pp("mlp"),
                    normal_loading_metadata.loading_isq,
                ),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )?;

            Ok(DecoderLayer {
                layer_impl,
                input_layernorm,
                post_attention_layernorm,
                moe,
            })
        })?;

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !tie {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };

        let local_cache = Arc::new(Mutex::new(LocalHybridCache::new(
            &layer_types,
            cfg,
            &normal_loading_metadata.real_device,
            vb_m.dtype(),
        )?));

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            layer_types: layer_types.clone(),
            lm_head,
            local_cache,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            dtype: vb.dtype(),
            mapper,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let mut local_cache = self.local_cache.lock().expect("Lock failed");

        // Compute MRoPE cos/sin using first full-attention layer's rotary embedding
        let cos_sin = {
            let first_attn_idx = self
                .layer_types
                .iter()
                .position(|lt| *lt == LayerType::FullAttention)
                .expect("No full attention layer found");
            match &self.layers[first_attn_idx].layer_impl {
                LayerImpl::FullAttention(attn) => {
                    attn.rotary_emb.compute_cos_sin(position_ids, xs.dtype())?
                }
                _ => unreachable!(),
            }
        };

        let attention_mask = DeviceMappedMask::new(attention_mask.cloned(), &*self.mapper)?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;

            match &self.layer_types[i] {
                LayerType::FullAttention => {
                    xs = layer.forward_attention(
                        &xs,
                        attention_mask.as_ref().map(|m| m.get(xs.device())),
                        &cos_sin,
                        &mut cache[i],
                        metadata
                            .as_ref()
                            .map(|(kv_cache, meta)| (kv_cache[i].clone(), *meta)),
                        flash_params,
                    )?;
                }
                LayerType::LinearAttention => {
                    let gdn_cache = match &mut local_cache.caches[i] {
                        LocalLayerCache::LinearAttention(c) => c,
                        _ => candle_core::bail!("Expected GDN cache for linear attention layer"),
                    };
                    if seqlen_offsets[0] == 0 {
                        gdn_cache.reset()?;
                    }
                    xs = layer.forward_linear(&xs, gdn_cache)?;
                }
            }

            // Integrate DeepStack visual features when provided
            if let (Some(visual_pos_masks), Some(deepstack)) =
                (visual_pos_masks, deepstack_visual_embeds)
            {
                if i < deepstack.len() {
                    xs = self.deepstack_process(xs, visual_pos_masks, &deepstack[i])?;
                }
            }
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let mut xs = extract_logits(&xs, context_lens)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        self.lm_head.forward(&xs)
    }

    fn deepstack_process(
        &self,
        hidden_states: Tensor,
        visual_pos_masks: &Tensor,
        visual_embeds: &Tensor,
    ) -> Result<Tensor> {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        let visual_embeds = visual_embeds.to_device(device)?.to_dtype(dtype)?;

        let (batch, seq, hidden) = hidden_states.dims3()?;
        let total = batch * seq;
        let hidden_flat = hidden_states.reshape((total, hidden))?;

        let mask_flat: Vec<f32> = visual_pos_masks
            .to_device(device)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let indices: Vec<u32> = mask_flat
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.0)
            .map(|(i, _)| i as u32)
            .collect();

        if indices.is_empty() {
            return Ok(hidden_states);
        }
        if indices.len() != visual_embeds.dim(0)? {
            candle_core::bail!(
                "Mismatch between DeepStack visual embeds ({}) and mask positions ({})",
                visual_embeds.dim(0)?,
                indices.len()
            );
        }

        let idx = Tensor::from_vec(indices, (visual_embeds.dim(0)?,), device)?;
        let idx_expanded = idx.unsqueeze(1)?.repeat((1, hidden))?;
        let result = hidden_flat.scatter_add(&idx_expanded, &visual_embeds, 0)?;
        result.reshape((batch, seq, hidden))
    }
}

impl IsqModel for Qwen3_5MoeTextModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match &mut layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    tensors.push((&mut attn.q_proj, Some(i)));
                    tensors.push((&mut attn.k_proj, Some(i)));
                    tensors.push((&mut attn.v_proj, Some(i)));
                    tensors.push((&mut attn.o_proj, Some(i)));
                }
                LayerImpl::LinearAttention(gdn) => {
                    tensors.push((&mut gdn.out_proj, Some(i)));
                }
            }
            for l in layer.moe.get_isq_layers() {
                tensors.push((l, Some(i)));
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_lm = uvb.pp("model").pp("language_model");
        uvb_lm.pp("embed_tokens").add(&self.embed_tokens);
        uvb_lm.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_lm.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);

            match &layer.layer_impl {
                LayerImpl::FullAttention(attn) => {
                    uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
                    uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
                }
                LayerImpl::LinearAttention(gdn) => {
                    uvb_l
                        .pp("linear_attn")
                        .pp("in_proj_qkvz")
                        .add_tensor("weight", gdn.in_proj_qkvz.weight().clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("in_proj_ba")
                        .add_tensor("weight", gdn.in_proj_ba.weight().clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("dt_bias", gdn.dt_bias.clone());
                    uvb_l
                        .pp("linear_attn")
                        .add_tensor("A_log", gdn.a_log.clone());
                    uvb_l
                        .pp("linear_attn")
                        .pp("norm")
                        .add_tensor("weight", gdn.norm.weight.clone());
                }
            }

            uvb_l
                .pp("mlp")
                .pp("gate")
                .add_tensor("weight", layer.moe.gate.weight().clone());
            uvb_l
                .pp("mlp")
                .pp("shared_expert_gate")
                .add_tensor("weight", layer.moe.shared_expert_gate.weight().clone());
        }

        uvb.to_safetensors()
    }
}
