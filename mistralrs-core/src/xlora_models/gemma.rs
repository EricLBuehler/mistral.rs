#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{iter::zip, sync::Arc};

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use mistralrs_lora::{linear_b as linear, LinearLayerLike, LoraConfig, Ordering};

use crate::models::{gemma::Config, Cache};

use super::{classifier::XLoraClassifier, XLoraConfig};

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offsets: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let mut q_embeds = Vec::with_capacity(b_sz);
        let mut k_embeds = Vec::with_capacity(b_sz);
        for (b, seqlen_offset) in zip(0..b_sz, seqlen_offsets) {
            let q_b = q.i(b)?.unsqueeze(0)?;
            let k_b = k.i(b)?.unsqueeze(0)?;
            let cos = self.cos.narrow(0, *seqlen_offset, seq_len)?;
            let sin = self.sin.narrow(0, *seqlen_offset, seq_len)?;
            let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
            let sin = sin.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, seq_len, dim)
            let q_embed = (q_b.broadcast_mul(&cos)? + rotate_half(&q_b)?.broadcast_mul(&sin))?;
            let k_embed = (k_b.broadcast_mul(&cos)? + rotate_half(&k_b)?.broadcast_mul(&sin))?;
            q_embeds.push(q_embed);
            k_embeds.push(k_embed);
        }
        Ok((Tensor::cat(&q_embeds, 0)?, Tensor::cat(&k_embeds, 0)?))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    up_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    down_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("gate_proj"),
            lora_config,
            count,
            ord,
        )?;
        let up_proj = linear(
            hidden_sz,
            intermediate_sz,
            false,
            vb.pp("up_proj"),
            lora_config,
            count,
            ord,
        )?;
        let down_proj = linear(
            intermediate_sz,
            hidden_sz,
            false,
            vb.pp("down_proj"),
            lora_config,
            count,
            ord,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor, scalings: Tensor, global_scaling_weight: f64) -> Result<Tensor> {
        let lhs = self
            .gate_proj
            .lora_forward(xs, scalings.clone(), global_scaling_weight)?
            .apply(&self.act_fn)?;
        let rhs = self
            .up_proj
            .lora_forward(xs, scalings.clone(), global_scaling_weight)?;
        self.down_proj
            .lora_forward(&(lhs * rhs)?, scalings, global_scaling_weight)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = linear(
            hidden_sz,
            num_heads * head_dim,
            bias,
            vb.pp("q_proj"),
            lora_config,
            count,
            ord,
        )?;
        let k_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            vb.pp("k_proj"),
            lora_config,
            count,
            ord,
        )?;
        let v_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            vb.pp("v_proj"),
            lora_config,
            count,
            ord,
        )?;
        let o_proj = linear(
            num_heads * head_dim,
            hidden_sz,
            bias,
            vb.pp("o_proj"),
            lora_config,
            count,
            ord,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
        })
    }

    fn repeat_kv(&self, xs: Tensor) -> Result<Tensor> {
        let n_rep = self.num_kv_groups;
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
            xs.unsqueeze(2)?
                .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
                .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self
            .q_proj
            .lora_forward(xs, scalings.clone(), global_scaling_weight)?;
        let key_states = self
            .k_proj
            .lora_forward(xs, scalings.clone(), global_scaling_weight)?;
        let value_states = self
            .v_proj
            .lora_forward(xs, scalings.clone(), global_scaling_weight)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offsets)?;

        let (key_states, value_states) = match &*kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = candle_nn::ops::kvconcat(&prev_k, &key_states, 2)?;
                let value_states = candle_nn::ops::kvconcat(&prev_v, &value_states, 2)?;
                (key_states, value_states)
            }
        };
        *kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = self.repeat_kv(key_states)?.contiguous()?;
        let value_states = self.repeat_kv(value_states)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        self.o_proj.lora_forward(
            &attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?,
            scalings.clone(),
            global_scaling_weight,
        )
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let self_attn =
            Attention::new(rotary_emb, cfg, vb.pp("self_attn"), lora_config, count, ord)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), lora_config, count, ord)?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.mlp.forward(
            &xs.apply(&self.post_attention_layernorm)?,
            scalings.clone(),
            global_scaling_weight,
        )?;
        residual + xs
    }
}

pub struct XLoraModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    dtype: DType,
    hidden_size: usize,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    xlora_classifier: XLoraClassifier,
}

impl XLoraModel {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        xlora_config: XLoraConfig,
        xlora_ordering: Ordering,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let mut count = 0;
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                lora_config,
                &mut count,
                &xlora_ordering,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = candle_nn::Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            hidden_size: cfg.hidden_size,
            cache: Cache::new(cfg.num_hidden_layers, true),
            max_seq_len: default_max_position_embeddings(),
            xlora_classifier: XLoraClassifier::new(
                xlora_config,
                count,
                lora_config.len(),
                vb,
                false,
            )?,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    fn calculate_past_kv_len(
        &self,
        seq_len: usize,
        kv_cache_1: &Option<(Tensor, Tensor)>,
    ) -> Result<usize> {
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        if k_cache_1.dims()[0] <= seq_len {
            Ok(0)
        } else {
            let indexed = k_cache_1.i(seq_len)?;
            let dims = indexed.dims();
            Ok(dims[dims.len() - 2])
        }
    }

    fn inner_forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        if seqlen_offsets.len() > b_size {
            candle_core::bail!("Expected seqlen offsets have length equal to batch size.")
        }

        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.cache.xlora_lock().len() {
                    new_cache.push(None);
                }

                *self.cache.xlora_lock() = new_cache.clone();
            }
            self.cache.xlora_lock()
        } else {
            self.cache.lock()
        };
        let past_key_values_length =
            self.calculate_past_kv_len(seq_len, cache.first().as_ref().unwrap())?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask =
                self.prepare_decoder_attention_mask(b_size, seq_len, past_key_values_length)?;
            Some(mask)
        };
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offsets,
                cache.get_mut(i).unwrap(),
                scalings.clone(),
                self.xlora_classifier.get_global_scaling_weight(),
            )?
        }
        xs.apply(&self.norm)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len_full) = input_ids_full.dims2()?;
        let (_, seq_len) = input_ids.dims2()?;

        let dummy_scalings = self.xlora_classifier.get_dummy_scalings(
            b_size,
            seq_len,
            input_ids.device(),
            self.dtype,
        )?;
        // Using X-LoRA cache here
        let hidden_states = if no_kv_cache {
            let res = self.inner_forward(
                input_ids_full,
                seqlen_offsets_full,
                dummy_scalings,
                true,
                no_kv_cache,
            )?;

            let mut new_cache = Vec::new();
            for _ in 0..self.cache.xlora_lock().len() {
                new_cache.push(Some((
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                )));
            }
            *self.cache.lock() = new_cache.clone();

            res
        } else {
            self.inner_forward(
                input_ids,
                seqlen_offsets,
                dummy_scalings,
                false,
                no_kv_cache,
            )?
        };

        let scalings = self.xlora_classifier.forward(hidden_states)?;

        if no_kv_cache {
            self.inner_forward(
                input_ids_full,
                seqlen_offsets_full,
                scalings,
                true,
                no_kv_cache,
            )?
            .apply(&self.lm_head)?
            .narrow(1, seq_len_full - 1, 1)
        } else {
            // is_full_pass=true is ok because no_kv_cache=false
            self.inner_forward(input_ids, seqlen_offsets, scalings, true, no_kv_cache)?
                .apply(&self.lm_head)?
                .narrow(1, seq_len - 1, 1)
        }
    }
}
