#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Mixtral Model
/// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
/// https://mistral.ai/news/mixtral-of-experts/
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, RotaryEmbedding, VarBuilder};
use mistralrs_lora::{linear_no_bias, LinearLayerLike, LoraConfig, Ordering};
use std::sync::Arc;

use crate::{
    models::{mixtral::Config, Cache},
    pipeline::MIXTRAL_IS_GPTX,
};

use super::{classifier::XLoraClassifier, NonGranularState, ScalingsMaker, XLoraConfig};

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
        ord: &Ordering,
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
            ord,
        )?;
        let k_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            lora_config,
            count,
            ord,
        )?;
        let v_proj = linear_no_bias(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            lora_config,
            count,
            ord,
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let key_states = self.k_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let value_states = self.v_proj.lora_forward(
            xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;

        let mut query_states =
            query_states.reshape((b_sz * q_len, self.num_heads, self.head_dim))?;
        let mut key_states =
            key_states.reshape((b_sz * q_len, self.num_kv_heads, self.head_dim))?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.rotary_emb.forward(
            seqlen_offsets,
            &start_offsets_kernel,
            &mut query_states,
            &mut key_states,
        )?;

        if query_states.rank() == 3 {
            query_states = query_states
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            key_states = key_states
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        let (key_states, value_states) = match &*kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = candle_nn::ops::kvconcat(prev_k, &key_states, 2)?;
                let value_states = candle_nn::ops::kvconcat(prev_v, &value_states, 2)?;
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
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )
    }
}

#[derive(Debug, Clone)]
struct BlockSparseTop2MLP {
    w1: Arc<dyn LinearLayerLike + Send + Sync>,
    w2: Arc<dyn LinearLayerLike + Send + Sync>,
    w3: Arc<dyn LinearLayerLike + Send + Sync>,
    act_fn: Activation,
}

impl BlockSparseTop2MLP {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let w1 = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("w1"),
            lora_config,
            count,
            ord,
        )?;
        let w2 = linear_no_bias(
            intermediate_sz,
            hidden_sz,
            vb.pp("w2"),
            lora_config,
            count,
            ord,
        )?;
        let w3 = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("w3"),
            lora_config,
            count,
            ord,
        )?;
        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        scalings: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let lhs = self
            .w1
            .lora_forward(xs, scalings.clone(), global_scaling_weight, is_scaling_pass)?
            .apply(&self.act_fn)?;
        let rhs =
            self.w3
                .lora_forward(xs, scalings.clone(), global_scaling_weight, is_scaling_pass)?;
        self.w2.lora_forward(
            &(lhs * rhs)?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )
    }
}

#[derive(Debug, Clone)]
struct SparseMoeBlock {
    gate: Arc<dyn LinearLayerLike + Send + Sync>,
    experts: Vec<BlockSparseTop2MLP>,
    num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        lora_config: &Vec<(String, LoraConfig)>,
        count: &mut usize,
        ord: &Ordering,
    ) -> Result<Self> {
        let gate = linear_no_bias(
            cfg.hidden_size,
            cfg.num_local_experts,
            vb.pp("gate"),
            lora_config,
            count,
            ord,
        )?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        let vb = vb.pp("experts");
        for idx in 0..cfg.num_local_experts {
            let expert = BlockSparseTop2MLP::new(cfg, vb.pp(idx), lora_config, count, ord)?;
            experts.push(expert)
        }
        Ok(SparseMoeBlock {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        scalings: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let router_logits = self.gate.lora_forward(
            &xs,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // In order to extract topk, we extract the data from the tensor and manipulate it
        // directly. Maybe we will want to use some custom ops instead at some point.
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        // routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        // top_x contains the row indexes to evaluate for each expert.
        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_rws = vec![vec![]; self.experts.len()];
        for (row_idx, rw) in routing_weights.iter().enumerate() {
            let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                sum_routing_weights += routing_weight;
                top_x[expert_idx].push(row_idx as u32);
            }
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                selected_rws[expert_idx].push(routing_weight / sum_routing_weights)
            }
        }

        // routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        // expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let top_x = &top_x[expert_idx];
            if top_x.is_empty() {
                continue;
            }
            let top_x = Tensor::new(top_x.as_slice(), xs.device())?;
            let selected_rws =
                Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?.reshape(((), 1))?;
            // Index the correct hidden states and compute the expert hidden state for
            // the current expert. We need to make sure to multiply the output hidden
            // states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden_dim))?;
            // current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])
            let current_hidden_states = expert_layer.forward(
                &current_state,
                scalings.clone(),
                global_scaling_weight,
                is_scaling_pass,
            )?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws)?;
            ys = ys.index_add(&top_x, &current_hidden_states, 0)?;
        }

        let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
        Ok(ys)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    block_sparse_moe: SparseMoeBlock,
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
        let block_sparse_moe =
            SparseMoeBlock::new(cfg, vb.pp("block_sparse_moe"), lora_config, count, ord)?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        scalings: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.block_sparse_moe.forward(
            &xs.apply(&self.post_attention_layernorm)?,
            scalings.clone(),
            global_scaling_weight,
            is_scaling_pass,
        )?;
        residual + xs
    }
}

pub struct XLoraModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    sliding_window: usize,
    pub device: Device,
    pub cache: Cache,
    dtype: DType,
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
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings,
            vb_m.device(),
            MIXTRAL_IS_GPTX,
            vb.dtype(),
        )?);
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
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
            xlora_classifier: XLoraClassifier::new(
                xlora_config,
                count,
                lora_config.len(),
                vb,
                false,
            )?,
        })
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

    #[allow(clippy::too_many_arguments)]
    fn inner_forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        scalings: Tensor,
        is_full_pass: bool,
        no_kv_cache: bool,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let past_key_values_length = self.calculate_past_kv_len(seq_len)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask =
                self.prepare_decoder_attention_mask(b_size, seq_len, past_key_values_length)?;
            Some(mask)
        };
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
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                cache.get_mut(i).unwrap(),
                scalings.clone(),
                self.xlora_classifier.get_global_scaling_weight(),
                is_scaling_pass,
            )?
        }
        xs.apply(&self.norm)
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
    ) -> Result<Tensor> {
        let (_b_size, seq_len_full) = input_ids_full.dims2()?;
        let (_, seq_len) = input_ids.dims2()?;

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
            self.inner_forward(
                input_ids_full,
                seqlen_offsets_full,
                start_offsets_kernel_full,
                scalings,
                true,
                no_kv_cache,
                None,
            )?
            .apply(&self.lm_head)?
            .narrow(1, seq_len_full - 1, 1)
        } else {
            // is_full_pass=true is ok because no_kv_cache=false
            self.inner_forward(
                input_ids,
                seqlen_offsets,
                start_offsets_kernel,
                scalings,
                true,
                no_kv_cache,
                None,
            )?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)
        }
    }
}

impl ScalingsMaker for XLoraModel {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn get_cache(&self) -> &Cache {
        &self.cache
    }
    fn get_classifier(&self) -> &XLoraClassifier {
        &self.xlora_classifier
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
            scalings,
            is_full_pass,
            no_kv_cache,
            is_scaling_pass,
        )
    }
}
