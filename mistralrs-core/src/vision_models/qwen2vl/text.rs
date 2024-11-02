use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use crate::{
    attention::SdpaParams,
    dummy_paged_attention::ModelConfigMetadata,
    layers::{Activation, FusedBiasLinear, Qwen2VLRotaryEmbedding, Sdpa},
    paged_attention::{AttentionImplementation, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, NormalLoadingMetadata,
    },
    utils::progress::NiceProgressBar,
};

use super::config::Config;

struct Qwen2VLRmsNorm {
    w: Tensor,
    eps: f64,
}

impl Qwen2VLRmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w: vb.get((size,), "weight")?,
            eps,
        })
    }
}

impl Module for Qwen2VLRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let initial_type = xs.dtype();
        let mut xs = xs.to_dtype(DType::F32)?;
        let var = xs.powf(2.)?.mean_keepdim(D::Minus1)?;
        xs = xs.broadcast_mul(&(&var + self.eps)?.recip()?.sqrt()?)?;
        xs.to_dtype(initial_type)?.broadcast_mul(&self.w)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: FusedBiasLinear,
    k_proj: FusedBiasLinear,
    v_proj: FusedBiasLinear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj =
            candle_nn::linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?.try_into()?;
        let k_proj =
            candle_nn::linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?.try_into()?;
        let v_proj =
            candle_nn::linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?.try_into()?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                use_flash_attn: false,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (mut q, mut k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        self.rotary_emb.forward(position_ids, &mut q, &mut k)?;

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    None,
                )?
            }
            None => {
                let (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }
}

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: Qwen2VLRmsNorm,
    post_attention_layernorm: Qwen2VLRmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<Qwen2VLRotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), paged_attn)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            Qwen2VLRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen2VLRmsNorm::new(
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

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            position_ids,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Qwen2VLTextModel {
    embed_tokens: Embedding,
    norm: Qwen2VLRmsNorm,
    layers: Vec<DecoderLayer>,
    pub lm_head: Linear,
    pub(super) cache: Cache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) max_seq_len: usize,
}

impl Qwen2VLTextModel {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let mut ropes = HashMap::new();
        for _layer_idx in 0..cfg.num_hidden_layers {
            let device = &normal_loading_metadata.real_device;
            ropes.insert(
                device.location(),
                Arc::new(Qwen2VLRotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    device,
                    cfg.rope_scaling.mrope_section.clone(),
                )?),
            );
        }

        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = &normal_loading_metadata.real_device;
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                    cfg.num_attention_heads,
                    head_dim,
                    (1.0 / (head_dim as f64).sqrt()) as f32,
                    Some(cfg.num_key_value_heads),
                    cfg.sliding_window,
                    device,
                    None,
                )?),
            };
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), paged_attn)?;
            layers.push(layer)
        }
        let norm = Qwen2VLRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                head_dim: None,
            },
            device: vb.device().clone(),
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        context_lens: Vec<(usize, usize)>,
        mut metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut cache = self.cache.lock();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                position_ids,
                &mut cache[i],
                metadata
                    .as_mut()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), &mut **metadata)),
                flash_params,
            )?;
        }
        let xs = xs.apply(&self.norm)?;
        extract_logits(&self.lm_head.forward(&xs)?, context_lens)
    }
}
