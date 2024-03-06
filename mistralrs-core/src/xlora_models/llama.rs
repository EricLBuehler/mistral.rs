#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use mistralrs_lora::{linear_no_bias as linear, LinearLayerLike, LoraConfig, Ordering};
use std::{collections::HashMap, sync::Arc};

use crate::models::{
    self,
    llama::{Config, MAX_SEQ_LEN},
    LayerCaches,
};

use super::{classifier::XLoraClassifier, XLoraConfig};

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
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

impl CausalSelfAttention {
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        cache: &Cache,
    ) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, hidden_size) = x.dims4()?;
        let mut ropes = Vec::new();
        for (b, offset) in seqlen_offsets.iter().enumerate() {
            let cos = cache.cos.narrow(0, *offset, seq_len)?;
            let sin = cache.sin.narrow(0, *offset, seq_len)?;
            let cos = cos.broadcast_as((1, 1, seq_len, hidden_size))?;
            let sin = sin.broadcast_as((1, 1, seq_len, hidden_size))?;
            let x_b = x.i(b)?.unsqueeze(0)?;
            let x1 = x_b.narrow(D::Minus1, 0, hidden_size / 2)?;
            let x2 = x_b.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
            let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
            let rope = (x_b.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
            ropes.push(rope);
        }
        Tensor::cat(&ropes, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        cache: &mut Cache,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self
            .q_proj
            .lora_forward(x, scalings.clone(), global_scaling_weight)?;
        let k = self
            .k_proj
            .lora_forward(x, scalings.clone(), global_scaling_weight)?;
        let v = self
            .v_proj
            .lora_forward(x, scalings.clone(), global_scaling_weight)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, seqlen_offsets, cache)?;
        let mut k = self.apply_rotary_emb(&k, seqlen_offsets, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &kv_cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            kv_cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self
            .o_proj
            .lora_forward(&y, scalings.clone(), global_scaling_weight)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"), lora_config, count, ord)?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"), lora_config, count, ord)?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"), lora_config, count, ord)?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"), lora_config, count, ord)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Arc<dyn LinearLayerLike + Send + Sync>,
    c_fc2: Arc<dyn LinearLayerLike + Send + Sync>,
    c_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor, scalings: Tensor, global_scaling_weight: f64) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
        )?)? * self
            .c_fc2
            .lora_forward(x, scalings.clone(), global_scaling_weight)?)?;
        self.c_proj
            .lora_forward(&x, scalings.clone(), global_scaling_weight)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"), lora_config, count, ord)?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"), lora_config, count, ord)?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"), lora_config, count, ord)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        cache: &mut Cache,
        scalings: Tensor,
        global_scaling_weight: f64,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            seqlen_offsets,
            block_idx,
            kv_cache,
            cache,
            scalings.clone(),
            global_scaling_weight,
        )? + residual)?;
        let residual = &x;
        let x = (self
            .mlp
            .forward(&self.rms_2.forward(&x)?, scalings, global_scaling_weight)?
            + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg, lora_config, count, ord)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, lora_config, count, ord)?;
        let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct XLoraLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: candle_nn::Linear,
    pub kv_cache: models::Cache,
    pub device: Device,
    cache: Cache,
    xlora_classifier: XLoraClassifier,
    dtype: DType,
}

impl XLoraLlama {
    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(x)?;
        let mut cache = if is_full_pass {
            if no_kv_cache {
                let mut new_cache = Vec::new();
                for _ in 0..self.kv_cache.xlora_lock().len() {
                    new_cache.push(None);
                }

                *self.kv_cache.xlora_lock() = new_cache.clone();
            }
            self.kv_cache.xlora_lock()
        } else {
            self.kv_cache.lock()
        };
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                seqlen_offsets,
                block_idx,
                &mut cache,
                &mut self.cache,
                scalings.clone(),
                self.xlora_classifier.get_global_scaling_weight(),
            )?;
        }
        self.ln_f.forward(&x)
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
            for _ in 0..self.kv_cache.xlora_lock().len() {
                new_cache.push(Some((
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                    Tensor::zeros((1,), DType::U8, &Device::Cpu)?,
                )));
            }
            *self.kv_cache.lock() = new_cache.clone();

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
            .i((.., seq_len_full - 1, ..))?
        } else {
            // is_full_pass=true is ok because no_kv_cache=false
            self.inner_forward(input_ids, seqlen_offsets, scalings, true, no_kv_cache)?
                .apply(&self.lm_head)?
                .i((.., seq_len - 1, ..))?
        }
        .to_dtype(DType::F32)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        cfg: &Config,
        dtype: DType,
        device: &Device,
        lora_config: &Vec<(String, LoraConfig)>,
        xlora_config: XLoraConfig,
        xlora_ordering: Ordering,
        no_kv_cache: bool,
    ) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let mut count = 0;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cfg,
                    lora_config,
                    &mut count,
                    &xlora_ordering,
                )
                .unwrap()
            })
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cache: Cache::new(!no_kv_cache, dtype, cfg, device)?,
            kv_cache: models::Cache::new(cfg.num_hidden_layers, false),
            device: device.clone(),
            xlora_classifier: XLoraClassifier::new(xlora_config, count, lora_config.len(), vb)?,
            dtype,
        })
    }
}
