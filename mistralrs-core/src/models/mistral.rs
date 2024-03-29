#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    layer_norm::RmsNormNonQuantized, linear_no_bias, rms_norm_non_quant, Activation, Linear,
    RotaryEmbedding, VarBuilder,
};
use std::sync::Arc;

use crate::{
    graph::{ComputationGraph, NodeOperator, Ready},
    pipeline::MISTRAL_IS_GPTX,
};

use super::Cache;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) sliding_window: usize,
    pub(crate) use_flash_attn: bool,
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm<RmsNormNonQuantized>,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm_non_quant(size, eps, vb)?;
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
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
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
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
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
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

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
            b_sz,
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
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
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
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
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
    graph: ComputationGraph<Ready>,
    lm_head: Linear,
    sliding_window: usize,
    dtype: DType,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut graph = ComputationGraph::empty();
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings,
            vb.device(),
            MISTRAL_IS_GPTX,
            vb.dtype(),
        )?;
        let norm = rms_norm_non_quant(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        let mut input_xs = graph.add_op(NodeOperator::Embedding { op: embed_tokens });
        let _ = graph.add_op(NodeOperator::StartModel { inp: input_xs });
        let vb_l = vb_m.pp("layers");
        let mut decoder_out = 0;
        for layer_idx in 0..cfg.num_hidden_layers {
            let vb = vb_l.pp(layer_idx);
            let input_layernorm =
                rms_norm_non_quant(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
            let post_attention_layernorm = rms_norm_non_quant(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?;

            let residual = input_xs;
            let xs = graph.add_op(NodeOperator::RmsNorm {
                op: input_layernorm,
                from: residual,
            });
            // Self attn
            let attn_output = {
                let vb = vb.pp("self_attn");
                let hidden_sz = cfg.hidden_size;
                let num_heads = cfg.num_attention_heads;
                let num_kv_heads = cfg.num_key_value_heads;
                let num_kv_groups = num_heads / num_kv_heads;
                let head_dim = hidden_sz / num_heads;
                let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
                let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
                let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
                let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

                // Run q,k,v proj
                let query_states = graph.add_op(NodeOperator::Linear {
                    op: q_proj,
                    from: xs,
                });
                let key_states = graph.add_op(NodeOperator::Linear {
                    op: k_proj,
                    from: xs,
                });
                let _value_states = graph.add_op(NodeOperator::Linear {
                    op: v_proj,
                    from: xs,
                });
                let value_states = graph.add_op(NodeOperator::Transpose12);

                // Reshape in prep for rmsnorm
                let query_states = graph.add_op(NodeOperator::ReshapeRms {
                    from: query_states,
                    num_heads,
                    head_dim,
                });
                let key_states = graph.add_op(NodeOperator::ReshapeRms {
                    from: key_states,
                    num_heads: num_kv_heads,
                    head_dim,
                });
                let _ = graph.add_op(NodeOperator::RoPE {
                    op: rotary_emb.clone(),
                    q: query_states,
                    k: key_states,
                });

                // If the rank is 3
                let _query_states = graph.add_op(NodeOperator::ReshapeAttn { from: query_states });
                let _query_states = graph.add_op(NodeOperator::Transpose12);
                let query_states = graph.add_op(NodeOperator::Contiguous);
                let _key_states = graph.add_op(NodeOperator::ReshapeAttn { from: query_states });
                let _key_states = graph.add_op(NodeOperator::Transpose12);
                let key_states = graph.add_op(NodeOperator::Contiguous);

                // Update KV cache
                let _ = graph.add_op(NodeOperator::UpdateKVCache {
                    k: key_states,
                    v: value_states,
                    layer_idx,
                });

                // Repeat kv cache, supporting GQA
                let query_states = graph.add_op(NodeOperator::RepeatKV {
                    num_kv_groups,
                    from: query_states,
                });
                let key_states = graph.add_op(NodeOperator::RepeatKV {
                    num_kv_groups,
                    from: key_states,
                });

                let scale = 1f64 / f64::sqrt(head_dim as f64);
                let key_states = graph.add_op(NodeOperator::Transpose23 { from: key_states });
                let _attn_weights = graph.add_op(NodeOperator::Matmul {
                    l: query_states,
                    r: key_states,
                });
                let attn_weights = graph.add_op(NodeOperator::Scale { factor: scale });
                let attn_weights =
                    graph.add_op(NodeOperator::ApplyAttentionMask { from: attn_weights });
                let attn_weights = graph.add_op(NodeOperator::Softmax { from: attn_weights });
                let _attn_weights = graph.add_op(NodeOperator::Matmul {
                    l: attn_weights,
                    r: value_states,
                });

                let _attn_output = graph.add_op(NodeOperator::Transpose12);
                let attn_output = graph.add_op(NodeOperator::ReshapeAttnOutput {
                    hidden_size: hidden_sz,
                });
                graph.add_op(NodeOperator::Linear {
                    op: o_proj,
                    from: attn_output,
                })
            };
            let residual = graph.add_op(NodeOperator::Add {
                l: attn_output,
                r: residual,
            });
            let xs = graph.add_op(NodeOperator::RmsNorm {
                op: post_attention_layernorm,
                from: residual,
            });
            // MLP
            let xs = {
                let vb = vb.pp("mlp");
                let hidden_sz = cfg.hidden_size;
                let intermediate_sz = cfg.intermediate_size;
                let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
                let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
                let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;

                let _gate = graph.add_op(NodeOperator::Linear {
                    op: gate_proj,
                    from: xs,
                });
                let lhs = graph.add_op(NodeOperator::Activation { op: cfg.hidden_act });
                let rhs = graph.add_op(NodeOperator::Linear {
                    op: up_proj,
                    from: xs,
                });
                let pre_down = graph.add_op(NodeOperator::Mul { l: lhs, r: rhs });
                graph.add_op(NodeOperator::Linear {
                    op: down_proj,
                    from: pre_down,
                })
            };
            decoder_out = graph.add_op(NodeOperator::Add { l: residual, r: xs });

            input_xs = decoder_out;
        }
        let _ = graph.add_op(NodeOperator::RmsNorm {
            op: norm,
            from: decoder_out,
        });

        Ok(Self {
            graph: graph.finalize_graph(),
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
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
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        if seqlen_offsets.len() > b_size {
            candle_core::bail!("Expected seqlen offsets have length equal to batch size.")
        }

        let past_key_values_length = self.calculate_past_kv_len(seq_len)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask =
                self.prepare_decoder_attention_mask(b_size, seq_len, past_key_values_length)?;
            Some(mask)
        };
        let cache = self.cache.lock();
        self.graph
            .execute(
                input_ids,
                attention_mask,
                cache,
                seqlen_offsets,
                start_offsets_kernel,
            )?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)
    }
}
