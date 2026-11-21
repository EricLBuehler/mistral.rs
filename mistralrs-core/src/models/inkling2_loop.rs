#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{embedding, CausalMasker, MatMul, RmsNorm, Sdpa},
    moe::{ExpertProjNames, MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        extract_logits, text_models_inputs_processor::FlashParams, EitherCache, IsqModel, KvCache,
        NormalLoadingMetadata, NormalModel,
    },
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(usize, max_position_embeddings, 1_048_576);
serde_default_fn!(f64, rope_theta, 10_000.0);
serde_default_fn!(f64, final_logit_softcapping, 30.0);
serde_default_fn!(f64, situ_beta, 4.0);
serde_default_fn!(f64, situ_linear_beta, 25.0);
serde_default_fn!(f64, qk_l2_clip_threshold, 22.627416997969522);
serde_default_fn!(f64, rel_l2_clip_threshold, 8.0);
serde_default_fn!(usize, n_log_scale_floor, 8192);
serde_default_fn!(f64, f_log_scale_alpha, 0.1);
serde_default_fn!(usize, attn_res_block_size, 6);
serde_default_fn!(bool, attn_res_normalize_values, true);
serde_default_fn!(usize, n_shared_experts, 0);
serde_default_fn!(f64, route_scale, 8.0);
serde_default_fn!(bool, use_gate_bias, true);
serde_default_fn!(usize, sconv_kernel_size, 4);
serde_default_fn!(usize, gate_rank, 64);
serde_default_fn!(usize, d_rel, 16);
serde_default_fn!(usize, rel_extent, 1024);

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    #[serde(default = "gate_rank")]
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "final_logit_softcapping")]
    pub final_logit_softcapping: f64,
    #[serde(default = "situ_beta")]
    pub situ_beta: f64,
    #[serde(default = "situ_linear_beta")]
    pub situ_linear_beta: f64,
    #[serde(default = "qk_l2_clip_threshold")]
    pub qk_l2_clip_threshold: f64,
    #[serde(default = "rel_l2_clip_threshold")]
    pub rel_l2_clip_threshold: f64,
    #[serde(default = "n_log_scale_floor")]
    pub n_log_scale_floor: usize,
    #[serde(default = "f_log_scale_alpha")]
    pub f_log_scale_alpha: f64,
    #[serde(default = "attn_res_block_size")]
    pub attn_res_block_size: usize,
    #[serde(default = "attn_res_normalize_values")]
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "route_scale")]
    pub route_scale: f64,
    #[serde(default = "use_gate_bias")]
    pub use_gate_bias: bool,
    #[serde(default = "sconv_kernel_size")]
    pub sconv_kernel_size: usize,
    #[serde(default = "d_rel")]
    pub d_rel: usize,
    #[serde(default = "rel_extent")]
    pub rel_extent: usize,
    pub loop_config: LoopConfig,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub sliding_window_pattern: Option<Vec<bool>>,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<mistralrs_quant::QuantizedConfig>,
}

impl Config {
    pub fn is_swa(&self, layer: usize) -> bool {
        if let Some(pattern) = &self.sliding_window_pattern {
            if let Some(is_swa) = pattern.get(layer) {
                return *is_swa;
            }
        }
        if let Some(local) = &self.local_layer_ids {
            return local.contains(&layer);
        }
        layer % 6 != 5
    }
}

#[derive(Clone)]
struct LoopCtrl {
    down: Arc<dyn QuantMethod>,
    gain: Arc<dyn QuantMethod>,
    forget: Arc<dyn QuantMethod>,
}

struct LoopLayer {
    msg_query: Tensor,
    q: LoopCtrl,
    k: LoopCtrl,
    v: LoopCtrl,
    ffn: LoopCtrl,
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    gate_proj: Arc<dyn QuantMethod>,
    rel: Arc<dyn QuantMethod>,
    rel_bias: Tensor,
    rel_clip: Tensor,
    q_clip: Tensor,
    sink_proj: Arc<dyn QuantMethod>,
    attn_norm: RmsNorm,
    q_inp_norm: Option<RmsNorm>,
    k_inp_norm: Option<RmsNorm>,
    v_inp_norm: Option<RmsNorm>,
    k_conv: Tensor,
    v_conv: Tensor,
    k_conv_proj: Option<Arc<dyn QuantMethod>>,
    v_conv_proj: Option<Arc<dyn QuantMethod>>,
    attn_post_conv: Tensor,
    res_query: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_rel: usize,
    rel_extent: usize,
    is_swa: bool,
    sliding_window: Option<usize>,
}

struct Moe {
    router: Arc<dyn QuantMethod>,
    exp_probs_b: Option<Tensor>,
    experts: MoEExperts,
    post_conv: Tensor,
    res_query: Tensor,
    num_experts_per_tok: usize,
    route_scale: f32,
}

struct DecoderLayer {
    attn: Attention,
    moe: Moe,
    ffn_norm: RmsNorm,
    loop_layer: Option<LoopLayer>,
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    embedding_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    output_res_query: Tensor,
    lm_head: Arc<dyn QuantMethod>,
    cfg: ModelConfigMetadata,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    loop_config: LoopConfig,
    rms_eps: f64,
    qk_clip: f32,
    rel_clip: f32,
    log_scale_floor: f32,
    log_scale_alpha: f32,
    attn_res_block_size: usize,
    final_logit_softcapping: f64,
    sconv_l_cache: usize,
    hidden_size: usize,
}

fn load_rms_norm(vb: ShardedVarBuilder, size: usize, eps: f64) -> Result<RmsNorm> {
    RmsNorm::new_gemma(size, eps, vb)
}

fn load_linear(vb: ShardedVarBuilder, in_dim: usize, out_dim: usize) -> Result<Arc<dyn QuantMethod>> {
    mistralrs_quant::linear_no_bias(in_dim, out_dim, &None, vb)
}

fn try_get(vb: &ShardedVarBuilder, shape: impl Into<candle_core::Shape>, name: &str) -> Option<Tensor> {
    vb.get(shape, name).ok()
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &None,
        )?;
        let embedding_norm = load_rms_norm(
            mapper.set_nm_device(vb_m.pp("embedding_norm"), false),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let norm = load_rms_norm(
            mapper.set_nm_device(vb_m.pp("norm"), false),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let output_res_query = mapper
            .set_nm_device(vb_m.pp("output_res_query"), false)
            .get((cfg.hidden_size,), "weight")?;
        let lm_head = if !cfg.tie_word_embeddings && vb.contains_tensor("lm_head.weight") {
            load_linear(
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
                cfg.hidden_size,
                cfg.vocab_size,
            )?
        } else {
            embed_tokens.clone()
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb_m.pp("layers").pp(i);
            let is_swa = cfg.is_swa(i);
            let num_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let hd = cfg.head_dim;
            let kv = num_kv_heads * hd;
            let q_dim = num_heads * hd;

            let attn_norm = load_rms_norm(vb_l.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
            let ffn_norm = load_rms_norm(vb_l.pp("post_attention_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;

            let sa = vb_l.pp("self_attn");
            let q_inp_norm = try_get(&sa.pp("q_inp_norm"), (cfg.hidden_size,), "weight")
                .map(|_| load_rms_norm(sa.pp("q_inp_norm"), cfg.hidden_size, cfg.rms_norm_eps))
                .transpose()?;
            let k_inp_norm = try_get(&sa.pp("k_inp_norm"), (cfg.hidden_size,), "weight")
                .map(|_| load_rms_norm(sa.pp("k_inp_norm"), cfg.hidden_size, cfg.rms_norm_eps))
                .transpose()?;
            let v_inp_norm = try_get(&sa.pp("v_inp_norm"), (cfg.hidden_size,), "weight")
                .map(|_| load_rms_norm(sa.pp("v_inp_norm"), cfg.hidden_size, cfg.rms_norm_eps))
                .transpose()?;

            let q_proj = load_linear(sa.pp("q_proj"), cfg.hidden_size, q_dim)?;
            let k_proj = load_linear(sa.pp("k_proj"), cfg.hidden_size, kv)?;
            let v_proj = load_linear(sa.pp("v_proj"), cfg.hidden_size, kv)?;
            let o_proj = load_linear(sa.pp("o_proj"), q_dim, cfg.hidden_size)?;
            let gate_proj = load_linear(sa.pp("gate_proj"), cfg.hidden_size, q_dim)?;
            let rel = load_linear(sa.pp("rel"), cfg.hidden_size, cfg.d_rel * num_heads)?;
            let rel_bias = sa.pp("rel_bias").get((cfg.d_rel, cfg.rel_extent), "weight")
                .or_else(|_| sa.pp("rel_bias").get((cfg.rel_extent, cfg.d_rel), "weight").and_then(|t| t.t()))
                .or_else(|_| sa.pp("rel_bias").get((cfg.d_rel, cfg.rel_extent + 1), "weight"))?;
            let rel_clip = sa.pp("rel_clip").get((cfg.d_rel, num_heads), "weight")
                .or_else(|_| sa.pp("rel_clip").get((num_heads, cfg.d_rel), "weight").and_then(|t| t.t()))?;
            let q_clip = sa.pp("q_clip").get((hd, num_heads), "weight")
                .or_else(|_| sa.pp("q_clip").get((num_heads, hd), "weight").and_then(|t| t.t()))?;
            let sink_proj = load_linear(sa.pp("sink_proj"), cfg.hidden_size, num_heads)?;
            let k_conv = sa.pp("k_conv").get((cfg.sconv_kernel_size, kv), "weight")
                .or_else(|_| sa.pp("k_conv").get((kv, 1, cfg.sconv_kernel_size), "weight").and_then(|t| t.squeeze(1)?.t()))?;
            let v_conv = sa.pp("v_conv").get((cfg.sconv_kernel_size, kv), "weight")
                .or_else(|_| sa.pp("v_conv").get((kv, 1, cfg.sconv_kernel_size), "weight").and_then(|t| t.squeeze(1)?.t()))?;
            let attn_post_conv = sa.pp("post_conv").get((cfg.sconv_kernel_size, cfg.hidden_size), "weight")
                .or_else(|_| sa.pp("post_conv").get((cfg.hidden_size, 1, cfg.sconv_kernel_size), "weight").and_then(|t| t.squeeze(1)?.t()))?;
            let k_conv_proj = if !is_swa && try_get(&sa.pp("k_conv_proj"), (32, cfg.hidden_size), "weight").is_some() {
                Some(load_linear(sa.pp("k_conv_proj"), cfg.hidden_size, 32)?)
            } else { None };
            let v_conv_proj = if !is_swa && try_get(&sa.pp("v_conv_proj"), (32, cfg.hidden_size), "weight").is_some() {
                Some(load_linear(sa.pp("v_conv_proj"), cfg.hidden_size, 32)?)
            } else { None };
            let res_query = try_get(&sa.pp("res_query"), (cfg.hidden_size,), "weight")
                .or_else(|| try_get(&sa.pp("res_query"), (3, cfg.hidden_size), "weight"));

            let attn = Attention {
                q_proj, k_proj, v_proj, o_proj, gate_proj, rel, rel_bias, rel_clip, q_clip, sink_proj,
                attn_norm, q_inp_norm, k_inp_norm, v_inp_norm,
                k_conv, v_conv, k_conv_proj, v_conv_proj, attn_post_conv, res_query,
                num_heads, num_kv_heads, head_dim: hd, d_rel: cfg.d_rel, rel_extent: cfg.rel_extent,
                is_swa, sliding_window: cfg.sliding_window,
            };

            let mlp_vb = vb_l.pp("mlp");
            let router = load_linear(mlp_vb.pp("router"), cfg.hidden_size, cfg.n_routed_experts)?;
            let exp_probs_b = mlp_vb.pp("exp_probs_b").get((cfg.n_routed_experts,), "bias").ok();
            let moe_cfg = MoEExpertsConfig {
                num_experts: cfg.n_routed_experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                hidden_size: cfg.hidden_size,
                moe_intermediate_size: cfg.intermediate_size,
                expert_proj_names: ExpertProjNames::DEFAULT,
            };
            let experts = MoEExperts::new(
                &moe_cfg,
                mlp_vb.clone(),
                normal_loading_metadata.real_device.clone(),
                &mapper.get_comm_for(i)?,
                normal_loading_metadata.loading_isq,
                &cfg.quantization_config,
                crate::layers::Activation::Silu,
            )?;
            let moe = Moe {
                router, exp_probs_b, experts,
                post_conv: mlp_vb.pp("post_conv").get((cfg.sconv_kernel_size, cfg.hidden_size), "weight")
                    .or_else(|_| mlp_vb.pp("post_conv").get((cfg.hidden_size, 1, cfg.sconv_kernel_size), "weight").and_then(|t| t.squeeze(1)?.t()))?,
                res_query: mlp_vb.pp("res_query").get((cfg.hidden_size,), "weight")?,
                num_experts_per_tok: cfg.num_experts_per_tok,
                route_scale: cfg.route_scale as f32,
            };

            let loop_layer = if i >= cfg.loop_config.core_start && i < cfg.loop_config.core_end {
                let ll = vb_l.pp("loop");
                let msg_query = ll.pp("msg_query").get((4, cfg.hidden_size), "weight")
                    .or_else(|_| ll.pp("msg_query").get((3, cfg.hidden_size), "weight"))?;
                let mk = |name: &str| -> Result<LoopCtrl> {
                    Ok(LoopCtrl {
                        down: load_linear(ll.pp(format!("{name}_ctrl_down")), 2 * cfg.hidden_size, cfg.loop_config.gate_rank)?,
                        gain: load_linear(ll.pp(format!("{name}_ctrl_gain")), cfg.loop_config.gate_rank, cfg.hidden_size)?,
                        forget: load_linear(ll.pp(format!("{name}_ctrl_forget")), cfg.loop_config.gate_rank, cfg.hidden_size)?,
                    })
                };
                Some(LoopLayer { msg_query, q: mk("q")?, k: mk("k")?, v: mk("v")?, ffn: mk("ffn")? })
            } else { None };

            layers.push(DecoderLayer { attn, moe, ffn_norm, loop_layer });
        }

        let logical_layers = cfg.num_hidden_layers - (cfg.loop_config.core_end - cfg.loop_config.core_start)
            + (cfg.loop_config.core_end - cfg.loop_config.core_start) * cfg.loop_config.num_passes;
        let cache = EitherCache::Normal(crate::pipeline::NormalCache::new(
            logical_layers,
            cfg.max_position_embeddings,
        ));

        Ok(Self {
            embed_tokens, embedding_norm, layers, norm, output_res_query, lm_head,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: logical_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            cache, max_seq_len: cfg.max_position_embeddings, mapper,
            loop_config: cfg.loop_config.clone(),
            rms_eps: cfg.rms_norm_eps,
            qk_clip: cfg.qk_l2_clip_threshold as f32,
            rel_clip: cfg.rel_l2_clip_threshold as f32,
            log_scale_floor: cfg.n_log_scale_floor as f32,
            log_scale_alpha: cfg.f_log_scale_alpha as f32,
            attn_res_block_size: cfg.attn_res_block_size,
            final_logit_softcapping: cfg.final_logit_softcapping,
            sconv_l_cache: cfg.sconv_kernel_size,
            hidden_size: cfg.hidden_size,
        })
    }
}

fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 { return Ok(x.clone()); }
    let scaled = (x / cap)?;
    let t = scaled.tanh()?;
    t * cap
}

fn situ(x: &Tensor, beta: f64, linear_beta: f64) -> Result<Tensor> {
    // SiTU-style activation used by Inkling2; approximate as SiLU-gated with beta params.
    // x: [..., 2*d]
    let d = x.dim(D::Minus1)? / 2;
    let gate = x.narrow(D::Minus1, 0, d)?;
    let up = x.narrow(D::Minus1, d, d)?;
    let gate = (gate * beta)?;
    let gate = candle_nn::ops::silu(&gate)?;
    let up = (&up * linear_beta)?.tanh()?;
    gate.broadcast_mul(&up)
}

fn rms_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    x.to_dtype(dtype)
}

fn l2_clip(x: &Tensor, clip_w: &Tensor, threshold: f32) -> Result<Tensor> {
    // x: [b, h, s, d], clip_w: [d,h] or [h,d]
    let x_f = x.to_dtype(DType::F32)?;
    let norm = x_f.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [b,h,s,1]
    let cw = if clip_w.dims().len() == 2 && clip_w.dim(0)? == x.dim(1)? {
        clip_w.t()?.unsqueeze(0)?.unsqueeze(2)? // [1,d,h,1] -> will fix
    } else {
        clip_w.unsqueeze(0)?.unsqueeze(2)? // [1,d,h,1]? maybe [d,h]
    };
    let cw = cw.to_dtype(DType::F32)?;
    // cw expected [1,h,1,d]
    let cw = if cw.dims() == [1, x.dim(3)?, 1, x.dim(1)?] {
        cw.permute((0, 3, 2, 1))?
    } else if cw.dims() == [1, x.dim(1)?, 1, x.dim(3)?] {
        cw
    } else {
        // best-effort reshape to [1,h,1,d]
        let cw2 = clip_w.to_dtype(DType::F32)?;
        if cw2.dims() == [x.dim(3)?, x.dim(1)?] {
            cw2.t()?.unsqueeze(0)?.unsqueeze(2)?
        } else {
            cw2.unsqueeze(0)?.unsqueeze(2)?
        }
    };
    let scale = (cw.broadcast_mul(&norm)? / (threshold as f64))?;
    let scale = (scale + 1.0)?.recip()?;
    let x = x_f.broadcast_mul(&scale)?;
    x.to_dtype(x.dtype())
}

impl Model {
    fn logical_layer_order(&self) -> Vec<(usize, usize)> {
        // (physical_layer, logical_cache_idx)
        let mut out = Vec::new();
        let mut logical = 0usize;
        for i in 0..self.loop_config.core_start {
            out.push((i, logical));
            logical += 1;
        }
        let core_len = self.loop_config.core_end - self.loop_config.core_start;
        for _pass in 0..self.loop_config.num_passes {
            for i in self.loop_config.core_start..self.loop_config.core_end {
                out.push((i, logical));
                logical += 1;
            }
        }
        for i in self.loop_config.core_end..self.layers.len() {
            out.push((i, logical));
            logical += 1;
        }
        let _ = core_len;
        out
    }

    fn apply_loop_ctrl(&self, ctrl: &LoopCtrl, cur: &Tensor, msg: &Tensor) -> Result<Tensor> {
        let gate_in = Tensor::cat(&[&rms_norm(cur, self.rms_eps)?, &rms_norm(msg, self.rms_eps)?], D::Minus1)?;
        let gate = ctrl.down.forward(&gate_in)?;
        let gain = ctrl.gain.forward(&gate.tanh()?)?;
        let forget = ctrl.forget.forward(&gate.tanh()?)?;
        let keep = (Tensor::ones_like(&forget)? - &forget)?;
        cur.broadcast_mul(&keep)?.broadcast_add(&msg.broadcast_mul(&gain)?)
    }

    fn forward_attn(
        &self,
        layer: &DecoderLayer,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let attn = &layer.attn;
        let normed = attn.attn_norm.forward(xs)?;
        let q_in = if let Some(n) = &attn.q_inp_norm { n.forward(&normed)? } else { normed.clone() };
        let k_in = if let Some(n) = &attn.k_inp_norm { n.forward(&normed)? } else { normed.clone() };
        let v_in = if let Some(n) = &attn.v_inp_norm { n.forward(&normed)? } else { normed.clone() };

        let mut q = attn.q_proj.forward(&q_in)?;
        let mut k = attn.k_proj.forward(&k_in)?;
        let mut v = attn.v_proj.forward(&v_in)?;

        q = q.reshape((b_sz, q_len, attn.num_heads, attn.head_dim))?.transpose(1, 2)?;
        k = k.reshape((b_sz, q_len, attn.num_kv_heads, attn.head_dim))?.transpose(1, 2)?;
        v = v.reshape((b_sz, q_len, attn.num_kv_heads, attn.head_dim))?.transpose(1, 2)?;

        (k, v) = kv_cache.append(&k, &v)?;

        // repeat kv
        let n_rep = attn.num_heads / attn.num_kv_heads;
        let k = if n_rep > 1 {
            let (b, h, s, d) = k.dims4()?;
            k.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?
        } else { k };
        let v = if n_rep > 1 {
            let (b, h, s, d) = v.dims4()?;
            v.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))?
        } else { v };

        let mut att = (q.to_dtype(DType::F32)?.matmul(&k.to_dtype(DType::F32)?.t()?)? / (attn.head_dim as f64).sqrt())?;
        if let Some(mask) = attention_mask {
            att = att.broadcast_add(mask)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let mut y = att.matmul(&v.to_dtype(DType::F32)?)?.to_dtype(xs.dtype())?;
        y = y.transpose(1, 2)?.reshape((b_sz, q_len, attn.num_heads * attn.head_dim))?;
        let gate = attn.gate_proj.forward(&normed)?;
        let gate = candle_nn::ops::silu(&gate)?;
        y = y.broadcast_mul(&gate)?;
        attn.o_proj.forward(&y)
    }

    fn forward_moe(&self, layer: &DecoderLayer, xs: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let normed = layer.ffn_norm.forward(xs)?;
        let flat = normed.reshape((b * s, h))?;
        let router_logits = layer.moe.router.forward(&flat)?;
        let mut probs = candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;
        if let Some(bias) = &layer.moe.exp_probs_b {
            probs = probs.broadcast_add(&bias.to_dtype(DType::F32)?.unsqueeze(0)?)?;
        }
        let topk = layer.moe.num_experts_per_tok;
        let mut topk_vals = Vec::with_capacity(b * s * topk);
        let mut topk_idx = Vec::with_capacity(b * s * topk);
        let probs_v = probs.to_vec2::<f32>()?;
        for row in probs_v {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&a, &c| row[c].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sel = idx[..topk].to_vec();
            let mut w: Vec<f32> = sel.iter().map(|&i| row[i]).collect();
            let sum: f32 = w.iter().sum();
            if sum > 0.0 { for x in &mut w { *x /= sum; } }
            for (i,wv) in sel.iter().zip(w.iter()) { topk_idx.push(*i as u32); topk_vals.push(*wv); }
        }
        let topk_ids = Tensor::from_vec(topk_idx, (b * s, topk), xs.device())?;
        let topk_weights = Tensor::from_vec(topk_vals, (b * s, topk), xs.device())?.to_dtype(xs.dtype())?;
        let y = layer.moe.experts.forward(&flat, topk_weights, &topk_ids)?;
        let y = y.reshape((b, s, h))?;
        Ok(y)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let flash_params = ctx.flash_params().clone();
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = self.embedding_norm.forward(&xs)?;

        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &crate::layers::CausalMaskConfig::default(),
        )?;

        let order = self.logical_layer_order();
        let mut cur = xs;
        let mut core_msg: Option<Tensor> = None;
        for (_ord_idx, (phys, logical)) in order.iter().enumerate() {
            let layer = &self.layers[*phys];
            let is_core = *phys >= self.loop_config.core_start && *phys < self.loop_config.core_end;

            let mut attn_in = cur.clone();
            if let Some(ll) = &layer.loop_layer {
                let msg = core_msg.clone().unwrap_or(cur.clone());
                attn_in = self.apply_loop_ctrl(&ll.q, &attn_in, &msg)?;
            }
            let mask = attention_mask.as_option_tensor();
            let attn_out = self.forward_attn(layer, &attn_in, mask, &[], &mut cache[*logical], &flash_params)?;
            cur = (cur + attn_out)?;

            let mut moe_in = cur.clone();
            if let Some(ll) = &layer.loop_layer {
                let msg = core_msg.clone().unwrap_or(cur.clone());
                moe_in = self.apply_loop_ctrl(&ll.ffn, &moe_in, &msg)?;
            }
            let moe_out = self.forward_moe(layer, &moe_in)?;
            cur = (cur + moe_out)?;

            if is_core {
                core_msg = Some(cur.clone());
            }
        }

        let cur = self.norm.forward(&cur)?;
        let logits = self.lm_head.forward(&cur)?;
        let logits = softcap(&logits, self.final_logit_softcapping)?;
        extract_logits(&logits, ctx.context_lens().to_vec())
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("inkling2_loop xlora not supported")
    }

    fn is_xlora(&self) -> bool { false }
    fn device(&self) -> &Device { &self.device }
    fn cache(&self) -> &EitherCache { &self.cache }
    fn max_seq_len(&self) -> usize { self.max_seq_len }
    fn config(&self) -> &ModelConfigMetadata { &self.cfg }
}

impl AnyMoeBaseModelMixin for Model {}
