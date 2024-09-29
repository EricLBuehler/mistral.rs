#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Activation, Embedding, Linear, Module, VarBuilder};

use crate::{
    attention::SdpaParams,
    layers::{repeat_kv, CausalMasker, Llama3RotaryEmbedding, MatMul, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{extract_logits, Cache, NormalLoadingMetadata},
};

use super::config::MLlamaTextConfig;

struct MLlamaTextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl MLlamaTextMlp {
    fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down_proj.forward(
            &self
                .act
                .forward(&self.gate_proj.forward(xs)?)?
                .broadcast_mul(&self.up_proj.forward(xs)?)?,
        )
    }
}

struct MLlamaTextSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    sdpa_params: SdpaParams,
    rope: Arc<Llama3RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MLlamaTextSelfAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: VarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                use_flash_attn: false,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
            },
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let mut q = self.q_proj.forward(hidden_states)?;
        let mut k = self.k_proj.forward(hidden_states)?;
        let mut v = self.v_proj.forward(hidden_states)?;

        q = q.reshape((bs * q_len, self.num_heads, self.head_dim))?;
        k = k.reshape((bs * q_len, self.num_kv_heads, self.head_dim))?;
        v = v
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.rope
            .forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, bs)?;

        if q.rank() == 3 {
            q = q
                .reshape((bs, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

        let attn_output = Sdpa
            .run_attention(&q, &k, &v, attention_mask, None, &self.sdpa_params)?
            .transpose(1, 2)?
            .reshape((bs, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct MLlamaSelfAttentionDecoderLayer {
    attn: MLlamaTextSelfAttention,
    mlp: MLlamaTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MLlamaSelfAttentionDecoderLayer {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: VarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
    ) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let attn = MLlamaTextSelfAttention::new(cfg, vb.pp("self_attn"), rope)?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states = self.attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
        )?;
        hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;

        residual + hidden_states
    }
}

struct MLlamaTextCrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MLlamaTextCrossAttention {
    fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(cfg.head_dim(), cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(cfg.head_dim(), cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let mut q = self.q_proj.forward(hidden_states)?;
        q = q
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        let (k, v) = if let Some(cross_attn_states) = cross_attn_states {
            let mut k = self.k_proj.forward(cross_attn_states)?;
            k = k
                .reshape((bs, (), self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            k = self.k_norm.forward(&k)?;

            let mut v = self.v_proj.forward(cross_attn_states)?;
            v = v
                .reshape((bs, (), self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;

            k = repeat_kv(k.clone(), self.num_heads / self.num_kv_heads)?.contiguous()?;
            v = repeat_kv(v.clone(), self.num_heads / self.num_kv_heads)?.contiguous()?;

            (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;
            (k, v)
        } else if let Some((k_cache, v_cache)) = kv_cache {
            (k_cache.clone(), v_cache.clone())
        } else {
            candle_core::bail!("Cross attn cannot find k,v cache or cross attn hidden states!")
        };

        let attn_output = {
            let att = MatMul.matmul_affine_div(
                &q.contiguous()?,
                &k.t()?.contiguous()?,
                (self.head_dim as f64).sqrt(),
            )?;

            let att = match attention_mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            MatMul
                .matmul(&att, &v.contiguous()?)?
                .transpose(1, 2)?
                .reshape((bs, q_len, ()))?
        };

        self.o_proj.forward(&attn_output)
    }
}

struct MLlamaCrossAttentionDecoderLayer {
    attn: MLlamaTextCrossAttention,
    attn_gate: Tensor,
    mlp: MLlamaTextMlp,
    mlp_gate: Tensor,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MLlamaCrossAttentionDecoderLayer {
    fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let attn = MLlamaTextCrossAttention::new(cfg, vb.pp("cross_attn"))?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            // TODO: pre tanh?
            attn_gate: vb.get((1,), "cross_attn_attn_gate")?,
            // TODO: pre tanh?
            mlp_gate: vb.get((1,), "cross_attn_mlp_gate")?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        full_text_row_masked_out_mask: Option<&Tensor>,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states =
            self.attn
                .forward(&hidden_states, cross_attn_states, attention_mask, kv_cache)?;
        hidden_states = (residual + hidden_states.broadcast_mul(&self.attn_gate.tanh()?)?)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;
        if let Some(full_text_row_masked_out_mask) = full_text_row_masked_out_mask {
            hidden_states = full_text_row_masked_out_mask
                .to_dtype(hidden_states.dtype())?
                .i((.., 0))?
                .broadcast_mul(&hidden_states)?;
        }

        residual + hidden_states.broadcast_mul(&self.mlp_gate.tanh()?)?
    }
}

enum MLlamaDecoderLayer {
    CrossAttn(MLlamaCrossAttentionDecoderLayer),
    SelfAttn(MLlamaSelfAttentionDecoderLayer),
}

pub(super) struct MLlamaTextModel {
    embed_tokens: Embedding,
    lm_head: Linear,
    norm: RmsNorm,
    layers: Vec<MLlamaDecoderLayer>,
    pub(crate) cfg: ModelConfigMetadata,
    pub(crate) self_attn_cache: Cache,
    pub(crate) device: Device,
    pub(crate) max_position_embeddings: usize,
}

impl MLlamaTextModel {
    pub(super) fn new(
        cfg: &MLlamaTextConfig,
        vb: VarBuilder,
        is_gptx: bool,
        _normal_loading_metadata: &NormalLoadingMetadata,
        _attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        let embed_tokens = embedding(
            cfg.vocab_size + 8,
            cfg.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let vb = vb.pp("model");

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let rope = Arc::new(Llama3RotaryEmbedding::new_mllama3(
            vb.dtype(),
            cfg,
            vb.device(),
            is_gptx,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            if cfg.cross_attention_layers.contains(&i) {
                layers.push(MLlamaDecoderLayer::CrossAttn(
                    MLlamaCrossAttentionDecoderLayer::new(cfg, vb.pp(format!("layers.{i}")))?,
                ))
            } else {
                layers.push(MLlamaDecoderLayer::SelfAttn(
                    MLlamaSelfAttentionDecoderLayer::new(
                        cfg,
                        vb.pp(format!("layers.{i}")),
                        rope.clone(),
                    )?,
                ))
            }
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                head_dim: None,
            },
            self_attn_cache: Cache::new(cfg.num_hidden_layers, false),
            device: vb.device().clone(),
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        input_ids: &Tensor,
        cross_attn_states: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
        full_text_row_masked_out_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mut self_cache = self.self_attn_cache.lock();
        let self_mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            hidden_states.dtype(),
            self.cfg.num_attn_heads,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                MLlamaDecoderLayer::SelfAttn(attn) => {
                    hidden_states = attn.forward(
                        &hidden_states,
                        self_mask.as_ref(),
                        seqlen_offsets,
                        start_offsets_kernel.clone(),
                        &mut self_cache[i],
                    )?;
                }
                MLlamaDecoderLayer::CrossAttn(attn) => {
                    // For text-only path we should skip cross attention layers.
                    // Let's check if the layer is cross attention layer and if we have cross attention states
                    // or cached cross attention states.
                    if cross_attn_states.is_none() {
                        continue;
                    }
                    hidden_states = attn.forward(
                        &hidden_states,
                        cross_attn_states,
                        cross_attention_mask,
                        full_text_row_masked_out_mask,
                        &mut self_cache[i],
                    )?;
                }
            }
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        hidden_states = self
            .lm_head
            .forward(&extract_logits(&hidden_states, context_lens)?)?;

        Ok(hidden_states)
    }
}
