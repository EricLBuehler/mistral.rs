#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use mistralrs_lora::{linear_no_bias, LinearLayerLike, LoraConfig};
use std::{iter::zip, sync::Arc};

use crate::models::{mistral::Config, Cache};

use super::{classifier::XLoraClassifier, config::XLoraConfig};

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
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
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
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
    act_fn: Activation,
}

impl MLP {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("gate_proj"),
            lora_config,
            count,
        )?;
        let up_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("up_proj"),
            lora_config,
            count,
        )?;
        let down_proj = linear_no_bias(
            intermediate_sz,
            hidden_sz,
            vb.pp("down_proj"),
            lora_config,
            count,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor, scalings: Tensor) -> Result<Tensor> {
        let lhs = self
            .gate_proj
            .lora_forward(xs, scalings.clone())?
            .apply(&self.act_fn)?;
        let rhs = self.up_proj.lora_forward(xs, scalings.clone())?;
        self.down_proj.lora_forward(&(lhs * rhs)?, scalings)
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
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
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(
            hidden_sz,
            num_heads * head_dim,
            vb.pp("q_proj"),
            lora_config,
            count,
        )?;
        let k_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            lora_config,
            count,
        )?;
        let v_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            lora_config,
            count,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
            vb.pp("o_proj"),
            lora_config,
            count,
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
            hidden_size: hidden_sz,
            rotary_emb,
            use_flash_attn: cfg.use_flash_attn,
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
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.lora_forward(xs, scalings.clone())?;
        let key_states = self.k_proj.lora_forward(xs, scalings.clone())?;
        let value_states = self.v_proj.lora_forward(xs, scalings.clone())?;

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
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        *kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = self.repeat_kv(key_states)?;
        let value_states = self.repeat_kv(value_states)?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
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
            &attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.hidden_size))?,
            scalings,
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
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), lora_config, count)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), lora_config, count)?;
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
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            scalings.clone(),
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?, scalings)?;
        residual + xs
    }
}

#[derive(Debug)]
pub struct XLoraModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    sliding_window: usize,
    dtype: DType,
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
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cache: Cache::new(cfg.num_hidden_layers, true),
            max_seq_len: cfg.max_position_embeddings,
            xlora_classifier: XLoraClassifier::new(xlora_config, count, lora_config.len(), vb)?,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
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
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Tensor,
        is_full_pass: bool,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        if seqlen_offsets.len() > b_size {
            candle_core::bail!("Expected seqlen offsets have length equal to batch size.")
        }

        let mut cache = if is_full_pass {
            let mut new_cache = Vec::new();
            for _ in 0..self.cache.xlora_lock().len() {
                new_cache.push(None);
            }

            *self.cache.xlora_lock() = new_cache.clone();
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
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offsets,
                cache.get_mut(i).unwrap(),
                scalings.clone(),
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
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let dummy_scalings = self.xlora_classifier.get_dummy_scalings(
            b_size,
            seq_len,
            input_ids.device(),
            self.dtype,
        )?;
      
        let scalings = dummy_scalings;
        dbg!(input_ids_full);
        dbg!(seqlen_offsets_full);
        // Using normal cache here
        let o = self
            .inner_forward(input_ids_full, seqlen_offsets_full, scalings, true)?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)?;
        
        let mut new_cache = Vec::new();
        for _ in 0..self.cache.xlora_lock().len() {
            new_cache.push(Some((
                Tensor::new(&[0i64], &self.device)?,
                Tensor::new(&[0i64], &self.device)?,
            )));
        }
        *self.cache.lock() = new_cache.clone();
        return Ok(o);



        let (b_size, seq_len) = input_ids.dims2()?;
        let dummy_scalings = self.xlora_classifier.get_dummy_scalings(
            b_size,
            seq_len,
            input_ids.device(),
            self.dtype,
        )?;
        // Using X-LoRA cache here
        //let hidden_states = self.inner_forward(input_ids, seqlen_offsets, dummy_scalings, false)?;

        let hidden_states = self.inner_forward(&input_ids_full.clone(), seqlen_offsets_full, dummy_scalings.clone(), true)?;

        let scalings = self.xlora_classifier.forward(hidden_states)?;

        // Using no cache here
        /*let o=self.inner_forward(input_ids_full, seqlen_offsets_full, scalings, true)?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)?;*/
        let o=self.inner_forward(input_ids_full, seqlen_offsets_full, dummy_scalings, true)?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)?;

        let mut new_cache = Vec::new();
        for _ in 0..self.cache.xlora_lock().len() {
            new_cache.push(Some((
                Tensor::new(&[0i64], &self.device)?,
                Tensor::new(&[0i64], &self.device)?,
            )));
        }
        *self.cache.xlora_lock() = new_cache.clone();
        *self.cache.lock() = new_cache.clone();

        Ok(o)
    }
}
