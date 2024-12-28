use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Linear, Module, VarBuilder};
use mistralrs_quant::QuantMethod;
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{
        Activation, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling,
        DeepSeekV2RotaryEmbedding, RmsNorm,
    },
    layers_masker::PastKvLenCache,
    ops::{SplitOp, TopKLastDimOp, TopKOutput},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, EitherCache, IsqModel, NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
    utils::progress::NiceProgressBar,
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    num_attention_heads: usize,
    n_shared_experts: Option<usize>,
    n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    first_k_dense_replace: usize,
    // k dense layers
    #[serde(default = "norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    tie_word_embeddings: bool,
    rope_theta: f32,
    rope_scaling: Option<DeepSeekV2RopeScaling>,
    attention_bias: bool,
    q_lora_rank: Option<usize>,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    qk_nope_head_dim: usize,
    n_group: Option<usize>,
    topk_group: Option<usize>,
}

enum QProj {
    Plain(Linear),
    Lora { a: Linear, norm: RmsNorm, b: Linear },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    softmax_scale: f32,
    cfg: DeepSeekV2Config,
    q_head_dim: usize,
}

impl Attention {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = candle_nn::linear_b(
                    cfg.hidden_size,
                    lora_rank,
                    cfg.attention_bias,
                    vb.pp("q_a_proj"),
                )?;
                let norm = RmsNorm::new(lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let b = candle_nn::linear_no_bias(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    vb.pp("q_b_proj"),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                vb.pp("q_proj"),
            )?),
        };

        let kv_a_proj_with_mqa = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm =
            RmsNorm::new(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = candle_nn::linear_no_bias(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;

        let o_proj = candle_nn::linear_b(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
        if let Some(DeepSeekV2RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = cfg.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            softmax_scale,
            cfg: cfg.clone(),
            q_head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        // kv_cache: &mut KvCache,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.cfg.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let q_pe = q_split[1].clone();

        let mut compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;
        let mut kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
        kv = kv
            .reshape((
                bs,
                seq_len,
                self.cfg.num_attention_heads,
                self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
            ))?
            .transpose(1, 2)?;

        let kv_split = kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let mut v = kv_split[1].clone();

        let (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offsets)?;

        let mut q = Tensor::zeros(
            (bs, self.cfg.num_attention_heads, seq_len, self.q_head_dim),
            q_pe.dtype(),
            q_pe.device(),
        )?;
        q = q.slice_assign(&[&.., &.., &.., &(..self.cfg.qk_nope_head_dim)], &q_nope)?;
        q = q.slice_assign(&[&.., &.., &.., &(self.cfg.qk_nope_head_dim..)], &q_pe)?;

        let mut k = Tensor::zeros(
            (bs, self.cfg.num_attention_heads, seq_len, self.q_head_dim),
            k_pe.dtype(),
            k_pe.device(),
        )?;
        k = k.slice_assign(&[&.., &.., &.., &(..self.cfg.qk_nope_head_dim)], &k_nope)?;
        let k_pe = k_pe.repeat((1, k.dim(1)?, 1, 1))?;
        k = k.slice_assign(&[&.., &.., &.., &(self.cfg.qk_nope_head_dim..)], &k_pe)?;

        // (k, v) = kv_cache.append(&k, &v)?;
        (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

        let mut attn_out = {
            let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale as f64)?;
            attn_weights = match attention_mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };
            attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_out = if attention_mask.is_some() {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        self.o_proj.forward(&attn_out)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    act: Activation,
}

impl Mlp {
    fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        Ok(Self {
            gate: candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up: candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down: candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&self.act)?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&(&lhs * &rhs)?)
    }
}

struct MoeGate {
    weight: Tensor,
    cfg: DeepSeekV2Config,
    top_k: usize,
}

impl MoeGate {
    fn new(cfg: &DeepSeekV2Config, vb: VarBuilder, n_routed_experts: usize) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k: cfg.num_experts_per_tok.unwrap(),
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_bs, _seq_len, h) = xs.dims3()?;
        // Compute gating score
        let xs = xs.reshape(((), h))?;
        let logits = xs.broadcast_matmul(&self.weight.t()?)?;
        let scores = match self.cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
        };

        // Select top-k experts
        let (mut topk_weight, topk_idx) = match self.cfg.topk_method {
            TopkMethod::Greedy => {
                let TopKOutput { values, indices } = scores.topk(self.top_k)?;
                (values, indices)
            }
            TopkMethod::GroupLimitedGreedy => {
                candle_core::bail!("GroupLimitedGreedy is not yet implemented!")
            }
        };

        if self.top_k > 1 && self.cfg.norm_topk_prob {
            let denmoninator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = (topk_weight / denmoninator)?;
        } else {
            topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;
        }
        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: Vec<Mlp>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
}

impl Moe {
    fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let mut experts = Vec::with_capacity(n_routed_experts);
        for i in 0..n_routed_experts {
            let vb_e = vb.pp("experts").pp(i);
            experts.push(Mlp::new(cfg, vb_e, None, Some(cfg.moe_intermediate_size))?);
        }
        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, vb.pp("gate"), n_routed_experts)?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut cnts: Tensor = Tensor::zeros(
            (topk_ids.dim(0)?, self.experts.len()),
            DType::F32,
            topk_ids.device(),
        )?;
        cnts = cnts
            .scatter_add(topk_ids, &topk_ids.ones_like()?.to_dtype(DType::F32)?, 1)?
            .to_dtype(topk_ids.dtype())?;
        let tokens_per_expert = cnts.sum(0)?.to_vec1::<u32>()?;
        let idxs = topk_ids.flatten_all()?.arg_sort_last_dim(true)?;
        let sorted_tokens = xs.index_select(&(&idxs / topk_ids.dim(1)? as f64)?, 0)?;

        let mut outputs = Vec::with_capacity(tokens_per_expert.len());
        let mut start_idx = 0;
        for (i, num_tokens) in tokens_per_expert.into_iter().enumerate() {
            let end_idx = start_idx + num_tokens;
            if num_tokens == 0 {
                continue;
            }
            let expert = &self.experts[i];
            let tokens_for_this_expert = sorted_tokens.i(start_idx as usize..end_idx as usize)?;
            let expert_out = expert.forward(&tokens_for_this_expert)?;
            outputs.push(expert_out);
            start_idx = end_idx;
        }

        let outs = Tensor::cat(&outputs, 0)?;

        let new_x = outs.zeros_like()?;
        new_x.index_add(&idxs, &outs, 0)?;

        let hole = new_x.elem_count() / topk_ids.elem_count();

        new_x
            .reshape([topk_ids.dims().to_vec(), vec![hole]].concat())?
            .broadcast_mul(&topk_weight.unsqueeze(D::Minus1)?)?
            .sum(1)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let orig_shape = xs.shape();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let xs = xs.reshape(((), xs.dim(D::Minus1)?))?;

        let mut y = self
            .moe_infer(&xs, &topk_idx, &topk_weight)?
            .reshape(orig_shape)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

enum MoeOrMlp {
    Moe(Moe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let moe_or_mlp = if cfg.n_routed_experts.is_some()
            && layer_idx >= cfg.first_k_dense_replace
            && layer_idx % cfg.moe_layer_freq == 0
        {
            MoeOrMlp::Moe(Moe::new(
                cfg,
                vb.pp("mlp"),
                cfg.n_shared_experts,
                cfg.n_routed_experts.unwrap(),
            )?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None)?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        // kv_cache: &mut KvCache,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .attn
            .forward(&xs, attention_mask, seqlen_offsets, kv_cache)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct DeepSeekV2 {
    lm_head: Linear,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
}

impl DeepSeekV2 {
    pub fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let mut ropes = HashMap::new();
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        for _i in 0..cfg.num_hidden_layers {
            let device = &normal_loading_metadata.real_device;
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &rope_cfg,
                    vb.dtype(),
                    device,
                )?),
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = &normal_loading_metadata.real_device;
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), layer_idx)?;
            layers.push(layer)
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            // cache: EitherCache::Normal(NormalCache::new(
            //     cfg.num_hidden_layers,
            //     cfg.max_position_embeddings,
            // )),
            cache: EitherCache::Full(Cache::new(cfg.num_hidden_layers, false)),
            device: normal_loading_metadata.real_device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_attention_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                head_dim: Some(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        // let cache = &mut self.cache.normal().0;
        let mut cache = self.cache.full().lock();
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
            )?;
        }
        let xs = xs.apply(&self.norm)?;
        extract_logits(&self.lm_head.forward(&xs)?, context_lens)
    }
}

impl IsqModel for DeepSeekV2 {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        todo!()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        todo!()
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        todo!()
    }
}

impl NormalModel for DeepSeekV2 {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(input_ids, seqlen_offsets, context_lens)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _start_offsets_kernel: Tensor,
        _start_offsets_kernel_full: Tensor,
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
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
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for DeepSeekV2 {}
