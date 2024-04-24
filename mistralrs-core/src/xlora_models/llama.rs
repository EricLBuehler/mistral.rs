#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, RotaryEmbedding, VarBuilder};
use mistralrs_lora::{linear_no_bias as linear, LinearLayerLike, LoraConfig, Ordering};
use std::{collections::HashMap, sync::Arc};

use crate::{
    device_map::DeviceMapper,
    models::{
        self, flash_attn,
        llama::{Config, MAX_SEQ_LEN},
        repeat_kv, LayerCaches, RmsNorm,
    },
    pipeline::{extract_logits, NormalModel},
};

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    device: Device,
}

impl Cache {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            masks: HashMap::new(),
            device: device.clone(),
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
struct CausalSelfAttention {
    q_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    k_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    v_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    o_proj: Arc<dyn LinearLayerLike + Send + Sync>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        cache: &mut Cache,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let k = self.k_proj.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let v = self.v_proj.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;

        let mut q = q.reshape((b_sz * seq_len, self.num_attention_heads, self.head_dim))?;
        let mut k = k.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_dim))?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary_emb
            .forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 {
            q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        if let Some((cache_k, cache_v)) = &kv_cache[block_idx] {
            k = candle_nn::ops::kvconcat(cache_k, &k, 2)?.contiguous()?;
            v = candle_nn::ops::kvconcat(cache_v, &v, 2)?.contiguous()?;
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
        kv_cache[block_idx] = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;

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
        let y = self.o_proj.lora_forward(
            &y,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        Ok(y)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        is_gptx: bool,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"), lora_config, count, ord)?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"), lora_config, count, ord)?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"), lora_config, count, ord)?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"), lora_config, count, ord)?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta,
            head_dim,
            MAX_SEQ_LEN,
            vb.device(),
            is_gptx,
            vb.dtype(),
        )?);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            rotary_emb,
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
}

impl Mlp {
    fn forward(
        &self,
        x: &Tensor,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?)? * self.c_fc2.lora_forward(
            x,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?)?;
        self.c_proj
            .lora_forward(&x, scalings.clone(), global_scaling_weight, is_scaling_pass)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"), lora_config, count, ord)?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"), lora_config, count, ord)?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"), lora_config, count, ord)?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut LayerCaches,
        cache: &mut Cache,
        scalings: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            seqlen_offsets,
            start_offsets_kernel,
            block_idx,
            kv_cache,
            cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(
            &self.rms_2.forward(&x)?,
            scalings,
            global_scaling_weight,
            is_scaling_pass,
        )? + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        lora_config: &[(String, LoraConfig)],
        count: &mut usize,
        ord: &Ordering,
        is_gptx: bool,
    ) -> Result<Self> {
        let attn =
            CausalSelfAttention::load(vb.pp("self_attn"), cfg, lora_config, count, ord, is_gptx)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg, lora_config, count, ord)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
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
    xlora_classifier: Option<XLoraClassifier>,
    dtype: DType,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl XLoraLlama {
    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        x: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Option<Tensor>,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
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
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                seqlen_offsets,
                start_offsets_kernel.clone(),
                block_idx,
                &mut cache,
                &mut self.cache,
                scalings.clone(),
                self.xlora_classifier
                    .as_ref()
                    .map(|classifier| classifier.get_global_scaling_weight())
                    .unwrap_or(1.0),
                is_scaling_pass,
            )?;
        }
        self.ln_f.forward(&x)?.to_dtype(DType::F32)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        if self.xlora_classifier.is_some() {
            let scalings = self.get_scalings(
                input_ids,
                input_ids_full,
                seqlen_offsets,
                seqlen_offsets_full,
                &start_offsets_kernel,
                &start_offsets_kernel_full,
                no_kv_cache,
                non_granular_state,
            )?;

            if no_kv_cache {
                extract_logits(
                    &self
                        .inner_forward(
                            input_ids_full,
                            seqlen_offsets_full,
                            start_offsets_kernel_full,
                            Some(scalings),
                            true,
                            no_kv_cache,
                            None,
                        )?
                        .contiguous()?
                        .apply(&self.lm_head)?,
                    context_lens,
                )
            } else {
                // is_full_pass=true is ok because no_kv_cache=false
                extract_logits(
                    &self
                        .inner_forward(
                            input_ids,
                            seqlen_offsets,
                            start_offsets_kernel,
                            Some(scalings),
                            true,
                            no_kv_cache,
                            None,
                        )?
                        .contiguous()?
                        .apply(&self.lm_head)?,
                    context_lens,
                )
            }
        } else {
            extract_logits(
                &self
                    .inner_forward(
                        input_ids,
                        seqlen_offsets,
                        start_offsets_kernel,
                        None,
                        false,
                        no_kv_cache,
                        None,
                    )?
                    .contiguous()?
                    .apply(&self.lm_head)?,
                context_lens,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &[(String, LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        is_gptx: bool,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
    ) -> Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let mut count = 0;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    mapper.set_device(i, vb.pp(&format!("model.layers.{i}"))),
                    cfg,
                    lora_config,
                    &mut count,
                    &xlora_ordering,
                    is_gptx,
                )
                .expect("Failed to load block.")
            })
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cache: Cache::new(device)?,
            kv_cache: models::Cache::new(cfg.num_hidden_layers, true),
            device: device.clone(),
            xlora_classifier: xlora_config.map(|xlora_config| {
                XLoraClassifier::new(xlora_config, count, lora_config.len(), vb, false).unwrap()
            }),
            dtype,
            mapper,
        })
    }
}

impl NormalModel for XLoraLlama {
    fn forward(
        &mut self,
        _input_ids: &Tensor,
        _seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        _context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        unreachable!()
    }
    fn xlora_forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        context_lens: Vec<usize>,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            start_offsets_kernel,
            start_offsets_kernel_full,
            no_kv_cache,
            non_granular_state,
            context_lens,
        )
    }
    fn cache(&self) -> &super::Cache {
        &self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        true
    }
    fn max_seq_len(&self) -> usize {
        MAX_SEQ_LEN
    }
}

impl ScalingsMaker for XLoraLlama {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn get_cache(&self) -> &models::Cache {
        &self.kv_cache
    }
    fn get_classifier(&self) -> &XLoraClassifier {
        self.xlora_classifier.as_ref().unwrap()
    }
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            Some(scalings),
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
        )
    }
}
