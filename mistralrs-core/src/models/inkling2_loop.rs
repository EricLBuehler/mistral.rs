#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
static DIST_CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize, usize), Tensor>>> = OnceLock::new();
#[allow(clippy::type_complexity)]
fn dist_cache_mut() -> &'static Mutex<HashMap<(usize, usize, usize, usize), Tensor>> {
    DIST_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}
// removed duplicate Device import

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{embedding, RmsNorm},
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
    attn_norm: Option<RmsNorm>,
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
    conv_k: Mutex<Tensor>,
    conv_v: Mutex<Tensor>,
    conv_attn_post: Mutex<Tensor>,
    conv_ffn_post: Mutex<Tensor>,
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
    dist_cache: Mutex<std::collections::HashMap<(usize, usize, usize, usize), Tensor>>,
}

fn load_rms_norm(vb: ShardedVarBuilder, size: usize, eps: f64) -> Result<RmsNorm> {
    RmsNorm::new_gemma(size, eps, vb)
}

fn load_linear(
    vb: ShardedVarBuilder,
    in_dim: usize,
    out_dim: usize,
) -> Result<Arc<dyn QuantMethod>> {
    mistralrs_quant::linear_no_bias(in_dim, out_dim, &None, vb)
}

fn try_get(
    vb: &ShardedVarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> Option<Tensor> {
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

            let ffn_norm = load_rms_norm(
                vb_l.pp("post_attention_layernorm"),
                cfg.hidden_size,
                cfg.rms_norm_eps,
            )?;
            let attn_norm =
                if try_get(&vb_l.pp("input_layernorm"), (cfg.hidden_size,), "weight").is_some() {
                    Some(load_rms_norm(
                        vb_l.pp("input_layernorm"),
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                    )?)
                } else {
                    None
                };

            let sa = vb_l.pp("self_attn");
            let q_inp_norm =
                if try_get(&sa.pp("q_inp_norm"), (cfg.hidden_size,), "weight").is_some() {
                    Some(load_rms_norm(
                        sa.pp("q_inp_norm"),
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                    )?)
                } else {
                    None
                };
            let k_inp_norm =
                if try_get(&sa.pp("k_inp_norm"), (cfg.hidden_size,), "weight").is_some() {
                    Some(load_rms_norm(
                        sa.pp("k_inp_norm"),
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                    )?)
                } else {
                    None
                };
            let v_inp_norm =
                if try_get(&sa.pp("v_inp_norm"), (cfg.hidden_size,), "weight").is_some() {
                    Some(load_rms_norm(
                        sa.pp("v_inp_norm"),
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                    )?)
                } else {
                    None
                };

            let q_proj = load_linear(sa.pp("q_proj"), cfg.hidden_size, q_dim)?;
            let k_proj = load_linear(sa.pp("k_proj"), cfg.hidden_size, kv)?;
            let v_proj = load_linear(sa.pp("v_proj"), cfg.hidden_size, kv)?;
            let o_proj = load_linear(sa.pp("o_proj"), q_dim, cfg.hidden_size)?;
            let gate_proj = load_linear(sa.pp("gate_proj"), cfg.hidden_size, q_dim)?;
            let rel = load_linear(sa.pp("rel"), cfg.hidden_size, cfg.d_rel * num_heads)?;
            let rel_bias = sa
                .pp("rel_bias")
                .get((cfg.d_rel, cfg.rel_extent), "weight")
                .or_else(|_| {
                    sa.pp("rel_bias")
                        .get((cfg.rel_extent, cfg.d_rel), "weight")
                        .and_then(|t| t.t())
                })
                .or_else(|_| {
                    sa.pp("rel_bias")
                        .get((cfg.d_rel, cfg.rel_extent + 1), "weight")
                })?;
            let rel_clip = sa
                .pp("rel_clip")
                .get((cfg.d_rel, num_heads), "weight")
                .or_else(|_| {
                    sa.pp("rel_clip")
                        .get((num_heads, cfg.d_rel), "weight")
                        .and_then(|t| t.t())
                })?;
            let q_clip = sa
                .pp("q_clip")
                .get((hd, num_heads), "weight")
                .or_else(|_| {
                    sa.pp("q_clip")
                        .get((num_heads, hd), "weight")
                        .and_then(|t| t.t())
                })?;
            let sink_proj = load_linear(sa.pp("sink_proj"), cfg.hidden_size, num_heads)?;
            let k_conv = sa
                .pp("k_conv")
                .get((kv, cfg.sconv_kernel_size), "weight")
                .or_else(|_| {
                    sa.pp("k_conv")
                        .get((cfg.sconv_kernel_size, kv), "weight")
                        .and_then(|t| t.t())
                })?;
            let v_conv = sa
                .pp("v_conv")
                .get((kv, cfg.sconv_kernel_size), "weight")
                .or_else(|_| {
                    sa.pp("v_conv")
                        .get((cfg.sconv_kernel_size, kv), "weight")
                        .and_then(|t| t.t())
                })?;
            let attn_post_conv = sa
                .pp("post_conv")
                .get((cfg.hidden_size, cfg.sconv_kernel_size), "weight")
                .or_else(|_| {
                    sa.pp("post_conv")
                        .get((cfg.sconv_kernel_size, cfg.hidden_size), "weight")
                        .and_then(|t| t.t())
                })?;
            let k_conv_proj = if !is_swa
                && try_get(&sa.pp("k_conv_proj"), (32, cfg.hidden_size), "weight").is_some()
            {
                Some(load_linear(sa.pp("k_conv_proj"), cfg.hidden_size, 32)?)
            } else {
                None
            };
            let v_conv_proj = if !is_swa
                && try_get(&sa.pp("v_conv_proj"), (32, cfg.hidden_size), "weight").is_some()
            {
                Some(load_linear(sa.pp("v_conv_proj"), cfg.hidden_size, 32)?)
            } else {
                None
            };
            let res_query = try_get(&sa.pp("res_query"), (cfg.hidden_size,), "weight")
                .or_else(|| try_get(&sa.pp("res_query"), (3, cfg.hidden_size), "weight"));

            let attn = Attention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                rel,
                rel_bias,
                rel_clip,
                q_clip,
                sink_proj,
                attn_norm,
                q_inp_norm,
                k_inp_norm,
                v_inp_norm,
                k_conv,
                v_conv,
                k_conv_proj,
                v_conv_proj,
                attn_post_conv,
                res_query,
                num_heads,
                num_kv_heads,
                head_dim: hd,
                d_rel: cfg.d_rel,
                rel_extent: cfg.rel_extent,
                is_swa,
                sliding_window: cfg.sliding_window,
            };

            let mlp_vb = vb_l.pp("mlp");
            let router = load_linear(mlp_vb.pp("router"), cfg.hidden_size, cfg.n_routed_experts)?;
            let exp_probs_b = mlp_vb
                .pp("exp_probs_b")
                .get((cfg.n_routed_experts,), "bias")
                .ok();
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
                router,
                exp_probs_b,
                experts,
                post_conv: mlp_vb
                    .pp("post_conv")
                    .get((cfg.hidden_size, cfg.sconv_kernel_size), "weight")
                    .or_else(|_| {
                        mlp_vb
                            .pp("post_conv")
                            .get((cfg.sconv_kernel_size, cfg.hidden_size), "weight")
                            .and_then(|t| t.t())
                    })?,
                res_query: mlp_vb.pp("res_query").get((cfg.hidden_size,), "weight")?,
                num_experts_per_tok: cfg.num_experts_per_tok,
                route_scale: cfg.route_scale as f32,
            };

            let loop_layer = if i >= cfg.loop_config.core_start && i < cfg.loop_config.core_end {
                let ll = vb_l.pp("loop");
                let msg_query = ll
                    .pp("msg_query")
                    .get((4, cfg.hidden_size), "weight")
                    .or_else(|_| ll.pp("msg_query").get((3, cfg.hidden_size), "weight"))?;
                let mk = |name: &str| -> Result<LoopCtrl> {
                    Ok(LoopCtrl {
                        down: load_linear(
                            ll.pp(format!("{name}_ctrl_down")),
                            2 * cfg.hidden_size,
                            cfg.loop_config.gate_rank,
                        )?,
                        gain: load_linear(
                            ll.pp(format!("{name}_ctrl_gain")),
                            cfg.loop_config.gate_rank,
                            cfg.hidden_size,
                        )?,
                        forget: load_linear(
                            ll.pp(format!("{name}_ctrl_forget")),
                            cfg.loop_config.gate_rank,
                            cfg.hidden_size,
                        )?,
                    })
                };
                Some(LoopLayer {
                    msg_query,
                    q: mk("q")?,
                    k: mk("k")?,
                    v: mk("v")?,
                    ffn: mk("ffn")?,
                })
            } else {
                None
            };

            let dev = vb_m.device();
            let sconv = cfg.sconv_kernel_size.saturating_sub(1);
            let zeros_kv = Tensor::zeros((1, kv, sconv), DType::F32, dev)?;
            let zeros_h = Tensor::zeros((1, cfg.hidden_size, sconv), DType::F32, dev)?;
            layers.push(DecoderLayer {
                attn,
                moe,
                ffn_norm,
                loop_layer,
                conv_k: Mutex::new(zeros_kv.clone()),
                conv_v: Mutex::new(zeros_kv),
                conv_attn_post: Mutex::new(zeros_h.clone()),
                conv_ffn_post: Mutex::new(zeros_h),
            });
        }

        let logical_layers = cfg.num_hidden_layers
            - (cfg.loop_config.core_end - cfg.loop_config.core_start)
            + (cfg.loop_config.core_end - cfg.loop_config.core_start) * cfg.loop_config.num_passes;
        let cache = EitherCache::Normal(crate::pipeline::NormalCache::new(
            logical_layers,
            cfg.max_position_embeddings,
        ));

        Ok(Self {
            embed_tokens,
            embedding_norm,
            layers,
            norm,
            output_res_query,
            lm_head,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: logical_layers * 2,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            cache,
            max_seq_len: cfg.max_position_embeddings,
            mapper,
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
            dist_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }
}

fn softcap(x: &Tensor, cap: f64) -> Result<Tensor> {
    if cap <= 0.0 {
        return Ok(x.clone());
    }
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
    fn attn_res_mix(&self, blocks: &[Tensor], cur: &Tensor, query: &Tensor) -> Result<Tensor> {
        if blocks.is_empty() {
            return Ok(cur.clone());
        }
        let q = query.to_dtype(DType::F32)?;
        let q = if q.rank() == 1 { q.unsqueeze(0)? } else { q };
        let n = blocks.len();
        let qn = n + 1;
        let q = if q.dim(0)? >= qn {
            q.narrow(0, 0, qn)?
        } else {
            let pad = Tensor::zeros((qn - q.dim(0)?, q.dim(1)?), q.dtype(), q.device())?;
            Tensor::cat(&[&q, &pad], 0)?
        };
        let cur_f = cur.to_dtype(DType::F32)?;
        let mut stack = Vec::with_capacity(n + 1);
        for b in blocks {
            stack.push(b.to_dtype(DType::F32)?);
        }
        stack.push(cur_f.clone());
        let stacked = Tensor::stack(&stack, 2)?; // [b,s,n+1,h]
        let (_qn, hq) = q.dims2()?;
        let logits = stacked
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&q.reshape((1, 1, _qn, hq, 1))?)?
            .sum(D::Minus2)?
            .squeeze(D::Minus1)?; // [b,s,n+1]
        let scale = 1.0f64 / (self.hidden_size as f64).sqrt();
        let logits = (logits * scale)?;
        let w = candle_nn::ops::softmax_last_dim(&logits)?;
        let out = stacked
            .broadcast_mul(&w.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?;
        out.to_dtype(cur.dtype())
    }

    fn apply_loop_ctrl(&self, ctrl: &LoopCtrl, cur: &Tensor, msg: &Tensor) -> Result<Tensor> {
        let gate_in = Tensor::cat(
            &[&rms_norm(cur, self.rms_eps)?, &rms_norm(msg, self.rms_eps)?],
            D::Minus1,
        )?;
        let gate = ctrl.down.forward(&gate_in)?;
        let gain = ctrl.gain.forward(&gate.tanh()?)?;
        let forget = ctrl.forget.forward(&gate.tanh()?)?;
        let keep = (Tensor::ones_like(&forget)? - &forget)?;
        cur.broadcast_mul(&keep)?
            .broadcast_add(&msg.broadcast_mul(&gain)?)
    }

    fn convolve(
        &self,
        cur: &Tensor,             // [b, s, c]
        weight: &Tensor,          // [c, k]
        dynamic: Option<&Tensor>, // [b,s,k*n_heads]
        state: &Mutex<Tensor>,
    ) -> Result<Tensor> {
        let mut st = state.lock().unwrap();
        let new_state = (&*st).clone();
        let (res, new_state) = {
            let _st = &mut *st;
            self.convolve_impl(cur, weight, dynamic, _st)?
        };
        *st = new_state.clone();
        Ok(res)
    }

    fn convolve_impl(
        &self,
        cur: &Tensor,
        weight: &Tensor,
        dynamic: Option<&Tensor>,
        state: &mut Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, s, c) = cur.dims3()?;
        let k = weight.dim(1)?;
        let cur_t = cur.transpose(1, 2)?; // [b,c,s]
        let mut st = state;
        if st.dim(0)? != b || st.dim(1)? != c || st.dim(2)? != k.saturating_sub(1) {
            *st = Tensor::zeros((b, c, k.saturating_sub(1)), cur_t.dtype(), cur_t.device())?;
        }
        let st_ref = st.clone();
        let w = weight.to_dtype(cur_t.dtype())?;
        let _ = &st_ref;
        let (mut out, new_state, x_full_for_dyn) = if k > 1 && s == 1 {
            let x_full = Tensor::cat(&[&st, &cur_t], 2)?; // [b,c,k]
            let _ = ();
            let out = x_full
                .broadcast_mul(&w.unsqueeze(0)?)?
                .sum(D::Minus1)?
                .unsqueeze(2)?;
            let ns = x_full.narrow(2, 1, k - 1)?;
            (out, ns, Some(x_full))
        } else if k > 1 {
            let x_full = Tensor::cat(&[&st, &cur_t], 2)?;
            let total = x_full.dim(2)?;
            let _ = total;
            let ns = x_full.narrow(2, total - (k - 1), k - 1)?;
            let w3 = w.reshape((c, 1, k))?;
            let out = x_full.conv1d(&w3, 0, 1, 1, c)?;
            (out, ns, Some(x_full))
        } else {
            let out = cur_t.broadcast_mul(&w.unsqueeze(0)?)?;
            (out, st.clone(), None)
        };
        if let Some(dyn_t) = dynamic {
            let dyn_dim = dyn_t.dim(D::Minus1)?;
            if dyn_dim % k == 0 {
                let n_heads = dyn_dim / k;
                if n_heads > 0 && c % n_heads == 0 {
                    let cph = c / n_heads;
                    let dyn_r = dyn_t
                        .to_dtype(cur_t.dtype())?
                        .reshape((b, s, n_heads, k))?
                        .permute((0, 2, 3, 1))?; // [b,h,k,s]
                    let x_full = x_full_for_dyn
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| st.clone());
                    let x_full = x_full.contiguous()?;
                    if s == 1 {
                        let x_rev = x_full.flip(&[2])?; // [b,c,k], lag l -> window k-1-l
                        let x_rev = x_rev.reshape((b, n_heads, cph, k))?;
                        let tap = dyn_r.squeeze(3)?; // [b,h,k]
                        let term = x_rev.broadcast_mul(&tap.unsqueeze(2)?)?.sum(D::Minus1)?; // [b,h,cph]
                        out = (&out + term.reshape((b, c, 1))?)?;
                    } else {
                        for lag in 0..k {
                            let tap = dyn_r.narrow(2, lag, 1)?; // [b,h,1,s]
                            let window = x_full.narrow(2, k - 1 - lag, s)?; // [b,c,s]
                            let window = window.reshape((b, n_heads, cph, s))?;
                            let term = window.broadcast_mul(&tap)?.reshape((b, c, s))?;
                            out = (&out + term)?;
                        }
                    }
                }
            }
        }
        let scale = 1.0f64 / (((if dynamic.is_some() { 2 } else { 1 } * k) + 1) as f64).sqrt();
        let res = ((&cur_t + &out)? * scale)?;
        Ok((res.transpose(1, 2)?, new_state))
    }

    fn forward_attn(
        &self,
        layer: &DecoderLayer,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        _flash_params: &FlashParams,
        phys: usize,
        _logical: usize,
        msgs: Option<&[Tensor; 4]>,
        loop_layer: Option<&LoopLayer>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let attn = &layer.attn;
        let mut inputs = [xs.clone(), xs.clone(), xs.clone()];
        if let (Some(msgs), Some(ll)) = (msgs, loop_layer) {
            inputs[0] = self.apply_loop_ctrl(&ll.q, &inputs[0], &msgs[0])?;
            inputs[1] = self.apply_loop_ctrl(&ll.k, &inputs[1], &msgs[1])?;
            inputs[2] = self.apply_loop_ctrl(&ll.v, &inputs[2], &msgs[2])?;
        }
        let (q_in, k_in, v_in) = if attn.is_swa && msgs.is_some() {
            let q = if let Some(n) = &attn.attn_norm {
                n.forward(&inputs[0])?
            } else {
                inputs[0].clone()
            };
            let k = if let Some(n) = &attn.attn_norm {
                n.forward(&inputs[1])?
            } else {
                inputs[1].clone()
            };
            let v = if let Some(n) = &attn.attn_norm {
                n.forward(&inputs[2])?
            } else {
                inputs[2].clone()
            };
            (q, k, v)
        } else if attn.is_swa {
            let nrm = if let Some(n) = &attn.attn_norm {
                n.forward(&inputs[0])?
            } else {
                inputs[0].clone()
            };
            (nrm.clone(), nrm.clone(), nrm)
        } else {
            let q = if let Some(n) = &attn.q_inp_norm {
                n.forward(&inputs[0])?
            } else {
                inputs[0].clone()
            };
            let k = if let Some(n) = &attn.k_inp_norm {
                n.forward(&inputs[1])?
            } else {
                inputs[1].clone()
            };
            let v = if let Some(n) = &attn.v_inp_norm {
                n.forward(&inputs[2])?
            } else {
                inputs[2].clone()
            };
            (q, k, v)
        };

        let q_pre = attn.q_proj.forward(&q_in)?;
        let k_pre = attn.k_proj.forward(&k_in)?;
        let v_pre = attn.v_proj.forward(&v_in)?;
        let mut q = q_pre;
        let mut k = k_pre;
        let mut v = v_pre;
        let k_dyn = if !attn.is_swa {
            attn.k_conv_proj
                .as_ref()
                .map(|p| p.forward(&k_in))
                .transpose()?
        } else {
            None
        };
        let v_dyn = if !attn.is_swa {
            attn.v_conv_proj
                .as_ref()
                .map(|p| p.forward(&v_in))
                .transpose()?
        } else {
            None
        };
        k = self.convolve(&k, &attn.k_conv, k_dyn.as_ref(), &layer.conv_k)?;
        v = self.convolve(&v, &attn.v_conv, v_dyn.as_ref(), &layer.conv_v)?;
        let mut k = k;
        let mut v = v;
        let _ = (&k, &v);

        q = q
            .reshape((b_sz, q_len, attn.num_heads, attn.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, attn.num_kv_heads, attn.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, attn.num_kv_heads, attn.head_dim))?
            .transpose(1, 2)?;

        // qk clip
        q = self.clip_head_dim(&q, &attn.q_clip, self.qk_clip)?;
        k = self
            .clip_head_dim(
                &k,
                &Tensor::zeros((1,), DType::F32, q.device())?,
                self.qk_clip,
            )
            .unwrap_or(k);
        v = rms_norm(&v.contiguous()?, self.rms_eps)?;

        if !attn.is_swa {
            let pos: Vec<f32> = (0..q_len)
                .map(|i| (seqlen_offsets.first().copied().unwrap_or(0) + i + 1) as f32)
                .collect();
            let tau = Tensor::from_vec(pos, (1, 1, q_len, 1), q.device())?.to_dtype(DType::F32)?;
            let tau = ((tau / (self.log_scale_floor as f64))?
                .clamp(1.0f64, f64::INFINITY)?
                .log()?
                * (self.log_scale_alpha as f64))?;
            let tau = (tau + 1.0f64)?;
            let tau = tau.to_dtype(q.dtype())?;
            q = q.broadcast_mul(&tau)?;
        }

        let (k, v) = kv_cache.append(&k, &v)?;

        let n_rep = attn.num_heads / attn.num_kv_heads;
        let k_rep = if n_rep > 1 {
            let (b, h, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, h, n_rep, s, d))?
                .reshape((b, h * n_rep, s, d))?
        } else {
            k.clone()
        };
        let v_rep = if n_rep > 1 {
            let (b, h, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, h, n_rep, s, d))?
                .reshape((b, h * n_rep, s, d))?
        } else {
            v.clone()
        };

        let mut att = (q
            .to_dtype(DType::F32)?
            .contiguous()?
            .matmul(&k_rep.to_dtype(DType::F32)?.contiguous()?.t()?)?
            / (attn.head_dim as f64).sqrt())?;

        // relative bias: R = rel(q_in), table = rel_bias^T @ R, subtract sinks, gather by distance
        let mut r = attn.rel.forward(&q_in)?;
        r = r
            .reshape((b_sz, q_len, attn.num_heads, attn.d_rel))?
            .transpose(1, 2)?
            .contiguous()?;
        r = self.clip_head_dim(&r, &attn.rel_clip, self.rel_clip)?;
        if !attn.is_swa {
            let pos: Vec<f32> = (0..q_len)
                .map(|i| (seqlen_offsets.first().copied().unwrap_or(0) + i + 1) as f32)
                .collect();
            let tau = Tensor::from_vec(pos, (1, 1, q_len, 1), q.device())?.to_dtype(DType::F32)?;
            let tau = ((tau / (self.log_scale_floor as f64))?
                .clamp(1.0f64, f64::INFINITY)?
                .log()?
                * (self.log_scale_alpha as f64))?;
            let tau = (tau + 1.0f64)?.to_dtype(r.dtype())?;
            r = r.broadcast_mul(&tau)?;
        }
        let sinks_logits = attn
            .sink_proj
            .forward(&q_in)?
            .reshape((b_sz, q_len, attn.num_heads))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?; // [b,h,q]
                                    // table: [b,h,q,rel_extent] = R[b,h,q,d_rel] @ rel_bias[d_rel,rel_extent]
        let rb = attn.rel_bias.to_dtype(DType::F32)?;
        let rb = if rb.dim(0)? == attn.d_rel {
            rb
        } else {
            rb.t()?
        }; // [d_rel, extent]
        let r_f = r.to_dtype(DType::F32)?; // [b,h,q,d_rel]
        let rb3 =
            rb.unsqueeze(0)?
                .broadcast_as((b_sz * attn.num_heads, attn.d_rel, attn.rel_extent))?;
        let table = r_f
            .reshape((b_sz * attn.num_heads, q_len, attn.d_rel))?
            .matmul(&rb3)?
            .reshape((b_sz, attn.num_heads, q_len, attn.rel_extent))?;
        // ref: subtract sink from table, pass zero sink to build_attn => equivalent to sink logit
        let sink4 = sinks_logits.unsqueeze(D::Minus1)?;
        let table = table.broadcast_sub(&sink4)?;
        att = att.broadcast_sub(&sink4)?;

        let kv_len = k_rep.dim(2)?;
        let q_off = seqlen_offsets.first().copied().unwrap_or(0);
        let extent = attn.rel_extent;
        let key = (q_len, kv_len, q_off, extent);
        let dist_t = {
            let mut cache = dist_cache_mut().lock().unwrap();
            if let Some(t) = cache.get(&key) {
                t.clone()
            } else {
                let mut dist: Vec<u32> = Vec::with_capacity(q_len * kv_len);
                for qi in 0..q_len {
                    let qpos = (q_off + qi) as i64;
                    for kj in 0..kv_len {
                        let kpos = kj as i64;
                        let d = (qpos - kpos).abs().min(extent as i64 - 1).max(0) as u32;
                        dist.push(d);
                    }
                }
                let t = Tensor::from_vec(dist, (q_len, kv_len), q.device())?;
                cache.insert(key, t.clone());
                t
            }
        }; // [q,kv]
           // gather table[b,h,q,extent] by dist -> [b,h,q,kv]
        let table_flat = table.reshape((b_sz * attn.num_heads, q_len, attn.rel_extent))?;
        let idx = dist_t
            .unsqueeze(0)?
            .broadcast_as((b_sz * attn.num_heads, q_len, kv_len))?
            .contiguous()?;
        let rel =
            table_flat
                .gather(&idx, D::Minus1)?
                .reshape((b_sz, attn.num_heads, q_len, kv_len))?;
        att = (att + rel)?;

        let zero_sink = Tensor::zeros((b_sz, attn.num_heads, q_len, 1), DType::F32, att.device())?;
        if let Some(mask) = attention_mask {
            let mask = mask.to_dtype(DType::F32)?;
            let zeros = Tensor::zeros((b_sz, 1, q_len, 1), DType::F32, mask.device())?;
            let mask = Tensor::cat(
                &[
                    &mask.broadcast_as((b_sz, attn.num_heads, q_len, kv_len))?,
                    &zeros.broadcast_as((b_sz, attn.num_heads, q_len, 1))?,
                ],
                D::Minus1,
            )?;
            att = Tensor::cat(&[&att, &zero_sink], D::Minus1)?;
            att = att.broadcast_add(&mask)?;
        } else if q_len > 1 {
            let mask: Vec<u8> = (0..q_len)
                .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > (kv_len - q_len + i))))
                .collect();
            let mask = Tensor::from_vec(mask, (1, 1, q_len, kv_len), att.device())?;
            let att_kv = crate::layers_masker::masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            att = Tensor::cat(&[&att_kv, &zero_sink], D::Minus1)?;
        } else {
            att = Tensor::cat(&[&att, &zero_sink], D::Minus1)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let att = att.narrow(D::Minus1, 0, kv_len)?;
        let mut y = att
            .matmul(&v_rep.to_dtype(DType::F32)?.contiguous()?)?
            .to_dtype(xs.dtype())?;
        y = y
            .transpose(1, 2)?
            .reshape((b_sz, q_len, attn.num_heads * attn.head_dim))?;
        let gate = candle_nn::ops::sigmoid(&attn.gate_proj.forward(&q_in)?)?;
        y = y.broadcast_mul(&gate)?;
        let y = attn.o_proj.forward(&y)?;
        self.convolve(&y, &attn.attn_post_conv, None, &layer.conv_attn_post)
    }

    fn clip_head_dim(&self, x: &Tensor, clip_w: &Tensor, threshold: f32) -> Result<Tensor> {
        if threshold <= 0.0 {
            return Ok(x.clone());
        }
        let x_f = x.to_dtype(DType::F32)?;
        let norm = x_f.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?; // [b,h,s,1]
        let cw = clip_w.to_dtype(DType::F32)?;
        let cw = if cw.dims().len() == 2 && cw.dim(0)? == x.dim(3)? && cw.dim(1)? == x.dim(1)? {
            cw.t()?.unsqueeze(0)?.unsqueeze(2)? // [1,h,1,d]
        } else if cw.dims().len() == 2 && cw.dim(0)? == x.dim(1)? && cw.dim(1)? == x.dim(3)? {
            cw.unsqueeze(0)?.unsqueeze(2)?
        } else {
            return Ok(x.clone());
        };
        let scale = (cw.broadcast_mul(&norm)? / (threshold as f64))?;
        let scale = (scale + 1.0)?.recip()?;
        x_f.broadcast_mul(&scale)?.to_dtype(x.dtype())
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
            idx.sort_by(|&a, &c| {
                row[c]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sel = idx[..topk].to_vec();
            let mut w: Vec<f32> = sel.iter().map(|&i| row[i]).collect();
            let sum: f32 = w.iter().sum();
            if sum > 0.0 {
                for x in &mut w {
                    *x /= sum;
                }
            }
            for (i, wv) in sel.iter().zip(w.iter()) {
                topk_idx.push(*i as u32);
                topk_vals.push(*wv);
            }
        }
        let topk_ids = Tensor::from_vec(topk_idx, (b * s, topk), xs.device())?;
        let topk_weights =
            Tensor::from_vec(topk_vals, (b * s, topk), xs.device())?.to_dtype(xs.dtype())?;
        layer.moe.experts.forward(&normed, topk_weights, &topk_ids)
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
        let this = self;
        let flash_params = ctx.flash_params().clone();
        let mut xs = self
            .embed_tokens
            .embedding_forward(input_ids, candle_core::DType::F32.into())?;
        if xs.rank() == 2 {
            xs = xs.unsqueeze(1)?;
        }
        xs = self.embedding_norm.forward(&xs)?;

        let cache = &mut self.cache.normal().0;

        let kv_caches: &mut [KvCache] = cache;
        let start = self.loop_config.core_start;
        let end = self.loop_config.core_end;
        let n_passes = self.loop_config.num_passes;
        let mut cur = xs;
        let mut blocks: Vec<Tensor> = Vec::new();

        let build_layer = |this: &Self,
                           phys: usize,
                           logical: usize,
                           cur: &mut Tensor,
                           blocks: &mut Vec<Tensor>,
                           msgs: Option<&[Tensor; 4]>,
                           ll: Option<&LoopLayer>,
                           kv_caches: &mut [KvCache],
                           flash_params: &FlashParams,
                           attention_mask: Option<&Tensor>|
         -> Result<()> {
            let layer = &this.layers[phys];
            let kv_cache: &mut KvCache = &mut kv_caches[phys];
            let mut attn_in = cur.clone();
            if let (Some(msgs), Some(ll)) = (msgs, ll) {
                attn_in = this.apply_loop_ctrl(&ll.q, &attn_in, &msgs[0])?;
            }
            if let Some(rq) = &layer.attn.res_query {
                attn_in = this.attn_res_mix(blocks, &attn_in, rq)?;
            }
            let attn_out = this.forward_attn(
                layer,
                &attn_in,
                attention_mask,
                &[],
                kv_cache,
                flash_params,
                phys,
                logical,
                msgs,
                ll,
            )?;
            let _ = kv_cache;
            *cur = (&*cur + attn_out)?;
            if phys % this.attn_res_block_size == 0 {
                blocks.push(cur.clone());
            }

            let residual = cur.clone();
            let mut moe_in = this.attn_res_mix(blocks, cur, &layer.moe.res_query)?;
            if let (Some(msgs), Some(ll)) = (msgs, ll) {
                moe_in = this.apply_loop_ctrl(&ll.ffn, &moe_in, &msgs[3])?;
            }
            let moe_out = this.forward_moe(layer, &moe_in)?;
            let moe_out =
                this.convolve(&moe_out, &layer.moe.post_conv, None, &layer.conv_ffn_post)?;
            *cur = (&residual + moe_out)?;
            Ok(())
        };

        let mut logical = 0usize;
        for phys in 0..start {
            build_layer(
                self,
                phys,
                logical,
                &mut cur,
                &mut blocks,
                None,
                None,
                kv_caches,
                &flash_params,
                None,
            )?;
            logical += 1;
        }

        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..n_passes {
            let (bank, bank_out) = if pass > 0 {
                let bank = if blocks.len() > prelude_blocks.len() + 1 {
                    blocks[prelude_blocks.len() + 1..].to_vec()
                } else {
                    Vec::new()
                };
                let bo = cur.clone();
                cur = prelude.clone();
                blocks = prelude_blocks.clone();
                (bank, Some(bo))
            } else {
                (Vec::new(), None)
            };
            for phys in start..end {
                let layer = &self.layers[phys];
                let msgs = if pass > 0 {
                    let bo = bank_out.as_ref().unwrap();
                    let ll = layer.loop_layer.as_ref().unwrap();
                    let mut m = Vec::with_capacity(4);
                    for i in 0..4 {
                        let q = ll
                            .msg_query
                            .narrow(0, i.min(ll.msg_query.dim(0)? - 1), 1)?
                            .squeeze(0)?;
                        m.push(self.attn_res_mix(&bank, bo, &q)?);
                    }
                    Some([m[0].clone(), m[1].clone(), m[2].clone(), m[3].clone()])
                } else {
                    None
                };
                let ll = layer.loop_layer.as_ref();
                build_layer(
                    self,
                    phys,
                    logical,
                    &mut cur,
                    &mut blocks,
                    msgs.as_ref(),
                    ll,
                    kv_caches,
                    &flash_params,
                    None,
                )?;
                logical += 1;
            }
        }
        for phys in end..self.layers.len() {
            build_layer(
                self,
                phys,
                logical,
                &mut cur,
                &mut blocks,
                None,
                None,
                kv_caches,
                &flash_params,
                None,
            )?;
            logical += 1;
        }

        let cur = self.attn_res_mix(&blocks, &cur, &self.output_res_query)?;
        let cur = self.norm.forward(&cur)?;
        let logits = self.lm_head.forward(&cur)?;
        let logits = softcap(&logits, self.final_logit_softcapping)?;
        let logits = extract_logits(&logits, ctx.context_lens().to_vec())?;
        Ok(logits)
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

    fn is_xlora(&self) -> bool {
        false
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}
