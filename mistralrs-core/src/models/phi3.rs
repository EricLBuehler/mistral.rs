#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py
use candle_core::{quantized::QMatMul, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, VarBuilder};
use either::Either;
use std::{collections::HashMap, sync::Arc};

use crate::{
    device_map::DeviceMapper,
    layers::{PhiRotaryEmbedding, RmsNorm},
    pipeline::{extract_logits, NormalModel},
    DeviceMapMetadata,
};

use super::{flash_attn, repeat_kv, Cache};

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<HashMap<String, Either<Vec<f32>, String>>>,
    pub max_position_embeddings: usize,
    pub use_flash_attn: bool,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(rotary_emb: Arc<PhiRotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = linear_no_bias(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj: QMatMul::Tensor(qkv_proj.weight().clone()),
            o_proj: QMatMul::Tensor(o_proj.weight().clone()),
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut qkv = self.qkv_proj.forward(&xs)?;
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            qkv = qkv.to_dtype(original_dtype)?;
        }
        let query_pos = self.num_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, query_pos)?;
        let k = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let v = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let (k, v, attn_mask) = match kv_cache.clone() {
            None => (k, v, attention_mask.cloned()),
            Some((mut prev_k, mut prev_v)) => {
                let mut mask = attention_mask.cloned();
                if let Some(sliding_window) = self.sliding_window {
                    let kv_seq_len = prev_k.dim(2)?;
                    if kv_seq_len > sliding_window {
                        prev_k = prev_k.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        prev_v = prev_v.narrow(
                            2,
                            kv_seq_len - (sliding_window - 1),
                            sliding_window - 1,
                        )?;
                        if let Some(ref mut mask) = mask {
                            let mask_len = mask.dim(1)?;
                            *mask = mask.narrow(
                                1,
                                mask_len - (sliding_window - 1),
                                sliding_window - 1,
                            )?;
                            *mask = Tensor::cat(
                                &[&*mask, &mask.narrow(1, mask_len - 1, 1)?.ones_like()?],
                                D::Minus1,
                            )?;
                        }
                    }
                }
                let k = Tensor::cat(&[prev_k, k], 2)?;
                let v = Tensor::cat(&[prev_v, v], 2)?;
                (k, v, mask)
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mut attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attn_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(&mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)?;
        if matches!(self.qkv_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: candle_nn::Activation,
    i_size: usize,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = linear_no_bias(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(i_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj: QMatMul::Tensor(gate_up_proj.weight().clone()),
            down_proj: QMatMul::Tensor(down_proj.weight().clone()),
            act_fn: cfg.hidden_act,
            i_size,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.gate_up_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let up_states = xs.apply(&self.gate_up_proj)?;
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        let mut res = up_states.apply(&self.down_proj)?;
        if matches!(self.gate_up_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<PhiRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
        )?;
        let mlp = Mlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
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
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    dtype: DType,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        _is_gptx: bool,
        mapper: DeviceMapMetadata,
        loading_isq: bool,
        real_device: Device,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let mapper = mapper.into_mapper(cfg.num_hidden_layers, &real_device)?;
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let rotary_emb = Arc::new(PhiRotaryEmbedding::new(
                vb.dtype(),
                cfg,
                mapper.device_for(layer_idx, false).unwrap_or(&real_device),
            )?);
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                loading_isq,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), loading_isq),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            device: real_device,
            dtype: vb.dtype(),
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            sliding_window: cfg.sliding_window,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        // Sliding window mask
        let sliding_window = sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
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

    fn calculate_past_kv_len(&mut self, seq_len: usize) -> Result<usize> {
        let cache = self.cache.lock();
        let kv_cache_1 = cache.first().unwrap();
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

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let past_key_values_length = self.calculate_past_kv_len(seq_len)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(
                b_size,
                seq_len,
                past_key_values_length,
                self.sliding_window,
            )?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut cache = self.cache.lock();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if matches!(self.lm_head, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        extract_logits(&xs.apply(&self.lm_head)?, context_lens)
    }
}

impl NormalModel for Model {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }
    fn xlora_forward(
        &mut self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _start_offsets_kernel: Tensor,
        _start_offsets_kernel_full: Tensor,
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &Cache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.qkv_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.gate_up_proj, Some(i)));
            tensors.push((&mut layer.mlp.down_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }
}
