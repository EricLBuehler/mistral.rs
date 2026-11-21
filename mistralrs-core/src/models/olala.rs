#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::{
    apply_immediate_isq, forward_group, weight_source_load_device, GroupKind, QuantMethod,
    QuantMethodConfig, ReplicatedLayer, Shard, ShardedVarBuilder, UnquantLinear,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod olala_cpu;
#[cfg(test)]
mod olala_prefill;

use crate::{
    paged_attention::ModelConfigMetadata,
    pipeline::{EitherCache, IsqModel, NormalModel},
    serde_default_fn,
};

serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(f64, default_softcap, 0.0);
serde_default_fn!(f64, default_gate_bias, 0.0);
serde_default_fn!(bool, default_tie, false);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: usize,
    #[serde(default = "default_softcap")]
    pub attn_logit_softcapping: f64,
    pub num_signal_heads: usize,
    pub num_noise_heads: usize,
    pub tpa_rank: usize,
    #[serde(default = "default_gate_bias")]
    pub gate_bias: f64,
    pub num_rope_angles: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub mamba_mimo_dim: usize,
    pub mamba_headdim: usize,
    pub expert_count: usize,
    pub expert_used_count: usize,
    pub expert_feed_forward_length: usize,
    pub expert_shared_feed_forward_length: usize,
    pub moe_latent_size: usize,
    pub expert_weights_scale: f64,
    pub layer_types: String,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<mistralrs_quant::QuantizedConfig>,
}

impl Config {
    fn layer_types(&self) -> Vec<bool> {
        self.layer_types.chars().map(|c| c == 'V').collect()
    }
}

const PI: f32 = std::f32::consts::PI;
const PI4: f32 = std::f32::consts::FRAC_PI_4;
const A_FLOOR: f32 = 1e-4;
const SUM_CLAMP: f32 = 6.103515625e-5;

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

fn wrap_angle(a: &Tensor) -> Result<Tensor> {
    let turns = a.affine(1.0 / (2.0 * f64::from(PI)), 0.0)?.round()?;
    a.sub(&turns.affine(2.0 * f64::from(PI), 0.0)?)
}

fn rms_inv(x: &Tensor, eps: f32) -> Result<Tensor> {
    x.sqr()?
        .mean(D::Minus1)?
        .affine(1.0, f64::from(eps))?
        .sqrt()?
        .recip()
}

fn geodesic(x: &Tensor, g: &Tensor, scale: f32, bias: f32, layer: usize) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let g = g.to_dtype(DType::F32)?.contiguous()?;
    let xx = x.sqr()?.sum(D::Minus1)?.clamp(1e-12, f32::MAX)?;
    let xg = x.mul(&g)?.sum(D::Minus1)?;
    let proj = xg.broadcast_div(&xx)?;
    let gp = g.sub(&x.broadcast_mul(&proj.unsqueeze(D::Minus1)?)?)?;
    let tan = gp.sqr()?.sum(D::Minus1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
    let r = xx.sqrt()?;
    let rmin = r.clamp(1e-6, f32::MAX)?;
    let theta0 = tan.broadcast_div(&rmin)?.clamp(0.0, PI4)?;
    let theta = theta0
        .affine(f64::from(scale), f64::from(bias))?
        .affine(1.0 / (layer + 1) as f64, 0.0)?
        .clamp(f32::MIN, PI4)?;
    let unit = gp.broadcast_div(&tan.unsqueeze(D::Minus1)?)?;
    let tangent = unit.broadcast_mul(&theta.sin()?.mul(&r)?.unsqueeze(D::Minus1)?)?;
    x.broadcast_mul(&theta.cos()?.unsqueeze(D::Minus1)?)?
        .add(&tangent)
}

struct MambaWeights {
    ssm_in: Arc<dyn QuantMethod>,
    ssm_in_dyn: Arc<dyn QuantMethod>,
    bias: Tensor,
    mxz: Tensor,
    norm_weight: Tensor,
    dt_bias: Tensor,
    d_skip: Tensor,
    mimo_o: Tensor,
    wo: Arc<dyn QuantMethod>,
}

struct AttentionWeights {
    wq: Arc<dyn QuantMethod>,
    wa_k: Arc<dyn QuantMethod>,
    wa_v: Arc<dyn QuantMethod>,
    wb_k: Arc<dyn QuantMethod>,
    wb_v: Arc<dyn QuantMethod>,
    ws_k: Arc<dyn QuantMethod>,
    ws_v: Arc<dyn QuantMethod>,
    wlambda: Arc<dyn QuantMethod>,
    wgate: Arc<dyn QuantMethod>,
    q_norm: Tensor,
    k_norm: Tensor,
    scaler: Tensor,
    wo: Arc<dyn QuantMethod>,
}

struct MoeWeights {
    gate_inp: Arc<dyn QuantMethod>,
    exp_probs_b: Vec<f32>,
    latent_down: Arc<dyn QuantMethod>,
    latent_up: Arc<dyn QuantMethod>,
    up_exps: Arc<dyn QuantMethod>,
    down_exps: Arc<dyn QuantMethod>,
    up_shexp: Arc<dyn QuantMethod>,
    down_shexp: Arc<dyn QuantMethod>,
}

struct LayerWeights {
    is_v: bool,
    mamba: Option<MambaWeights>,
    attn: Option<AttentionWeights>,
    moe: MoeWeights,
    geo_mixer_scale: f32,
    geo_mixer_bias: f32,
    geo_mlp_scale: f32,
    geo_mlp_bias: f32,
}

enum LayerState {
    Mamba {
        s: Tensor,
        k: Tensor,
        v: Tensor,
        ang: Tensor,
    },
    Attention {
        shift: Tensor,
        k_rows: Vec<Tensor>,
        v_rows: Vec<Tensor>,
    },
}

struct OlalaState {
    layers: Vec<LayerState>,
}

pub struct Model {
    config: Config,
    embed: Tensor,
    lm_head: Arc<dyn QuantMethod>,
    layers: Vec<LayerWeights>,
    state: Mutex<OlalaState>,
    cache: EitherCache,
    device: Device,
    dtype: DType,
    cfg: ModelConfigMetadata,
}

fn scalar(v: &Tensor) -> Result<f32> {
    v.reshape(())?.to_scalar::<f32>()
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let neg_abs = x.abs()?.neg()?;
    let soft = neg_abs.exp()?.affine(1.0, 1.0)?.log()?;
    x.clamp(0.0, f32::MAX)?.add(&soft)
}

fn group_forward(x: &Tensor, members: &[&Arc<dyn QuantMethod>]) -> Result<Option<Vec<Tensor>>> {
    let refs: Vec<&dyn QuantMethod> = members.iter().map(|m| m.as_ref()).collect();
    forward_group(x, &refs, GroupKind::Dense)
}

fn load_expert_projection(
    vb: &ShardedVarBuilder,
    name: &str,
    shape: (usize, usize, usize),
) -> Result<Arc<dyn QuantMethod>> {
    let projection_vb = vb.pp(name);
    if let Some(source) = vb.weight_source() {
        let load_device = weight_source_load_device(&projection_vb);
        if let Some(layer) =
            source.load_linear(&projection_vb.prefix(), &load_device, Shard::default())?
        {
            return apply_immediate_isq(layer, projection_vb);
        }
    }
    let weight = projection_vb.get(shape, "weight")?;
    let layer = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
        candle_nn::Linear::new(weight, None),
    ))?) as Arc<dyn QuantMethod>;
    apply_immediate_isq(layer, projection_vb)
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: Config,
        vb: ShardedVarBuilder,
        cfg: ModelConfigMetadata,
        dtype: DType,
        max_seq_len: usize,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let qcfg = &config.quantization_config;
        let load_vec =
            |name: &str, n: usize| -> Result<Tensor> { vb.get(n, name)?.to_dtype(DType::F32) };
        let load_mat3 = |name: &str, d0: usize, d1: usize, d2: usize| -> Result<Tensor> {
            vb.get((d0, d1, d2), name)?.to_dtype(DType::F32)
        };
        let load_scalar =
            |name: &str| -> Result<f32> { Ok(scalar(&vb.get(1, name)?.to_dtype(DType::F32)?)?) };
        let linear = |in_dim: usize, out_dim: usize, vb: ShardedVarBuilder| {
            ReplicatedLayer::new(in_dim, out_dim, qcfg, false, vb)
        };

        let embed = vb
            .get(
                (config.vocab_size, config.hidden_size),
                "model.embed_tokens.weight",
            )?
            .to_dtype(DType::F32)?;
        let lm_head = linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let h = config.hidden_size;
        let v = config.vocab_size;
        let n_mamba_h = config.ssm_inner_size / config.ssm_time_step_rank;
        let d_v = config.ssm_time_step_rank;
        let d_qk = config.ssm_state_size;
        let r = config.mamba_mimo_dim;
        let n_ang = config.num_rope_angles;
        let n_hn = config.num_noise_heads;
        let n_hs = config.num_signal_heads;
        let dh = config.head_dim;
        let latent = config.moe_latent_size;
        let n_exp = config.expert_count;
        let ff_exp = config.expert_feed_forward_length;
        let ff_sh = config.expert_shared_feed_forward_length;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut states = Vec::with_capacity(config.num_hidden_layers);
        for (i, &is_v) in config.layer_types().iter().enumerate() {
            let p = format!("model.layers.{i}");
            let moe = MoeWeights {
                gate_inp: linear(h, n_exp, vb.pp(format!("{p}.moe.gate_inp")))?,
                exp_probs_b: load_vec(&format!("{p}.moe.exp_probs_b"), n_exp)?
                    .reshape((n_exp,))?
                    .contiguous()?
                    .to_vec1::<f32>()?,
                latent_down: linear(h, latent, vb.pp(format!("{p}.moe.latent_down")))?,
                latent_up: linear(latent, h, vb.pp(format!("{p}.moe.latent_up")))?,
                up_exps: load_expert_projection(
                    &vb,
                    &format!("{p}.moe.up_exps"),
                    (n_exp, ff_exp, latent),
                )?,
                down_exps: load_expert_projection(
                    &vb,
                    &format!("{p}.moe.down_exps"),
                    (n_exp, latent, ff_exp),
                )?,
                up_shexp: linear(h, ff_sh, vb.pp(format!("{p}.moe.up_shexp")))?,
                down_shexp: linear(ff_sh, h, vb.pp(format!("{p}.moe.down_shexp")))?,
            };

            let (mamba, attn) = if is_v {
                let attn = AttentionWeights {
                    wq: linear(
                        h,
                        config.num_attention_heads * dh,
                        vb.pp(format!("{p}.mixer.attn_q")),
                    )?,
                    wa_k: linear(h, n_hn * r, vb.pp(format!("{p}.mixer.attn_wa_k")))?,
                    wa_v: linear(h, n_hn * r, vb.pp(format!("{p}.mixer.attn_wa_v")))?,
                    wb_k: linear(h, r * dh, vb.pp(format!("{p}.mixer.attn_wb_k")))?,
                    wb_v: linear(h, r * dh, vb.pp(format!("{p}.mixer.attn_wb_v")))?,
                    ws_k: linear(h, n_hn, vb.pp(format!("{p}.mixer.attn_shift_k")))?,
                    ws_v: linear(h, n_hn, vb.pp(format!("{p}.mixer.attn_shift_v")))?,
                    wlambda: linear(h, n_hn, vb.pp(format!("{p}.mixer.attn_lambda")))?,
                    wgate: linear(h, n_hs * dh, vb.pp(format!("{p}.mixer.attn_gate")))?,
                    q_norm: load_vec(&format!("{p}.mixer.attn_q_norm.weight"), dh)?,
                    k_norm: load_vec(&format!("{p}.mixer.attn_k_norm.weight"), dh)?,
                    scaler: load_vec(
                        &format!("{p}.mixer.attn_softmax_scaler"),
                        config.num_attention_heads,
                    )?,
                    wo: linear(n_hs * dh, h, vb.pp(format!("{p}.mixer.attn_output")))?,
                };
                let shift = Tensor::zeros(2 * n_hn * dh, DType::F32, &device)?;
                states.push(LayerState::Attention {
                    shift,
                    k_rows: Vec::new(),
                    v_rows: Vec::new(),
                });
                (None, Some(attn))
            } else {
                let mxz = Tensor::cat(
                    &[
                        load_mat3(&format!("{p}.mixer.ssm_mimo_x"), n_mamba_h, r, d_v)?,
                        load_mat3(&format!("{p}.mixer.ssm_mimo_z"), n_mamba_h, r, d_v)?,
                    ],
                    1,
                )?;
                let norms = Tensor::cat(
                    &[
                        load_vec(&format!("{p}.mixer.ssm_b_norm.weight"), d_qk)?
                            .reshape((1, d_qk))?,
                        load_vec(&format!("{p}.mixer.ssm_c_norm.weight"), d_qk)?
                            .reshape((1, d_qk))?,
                    ],
                    0,
                )?;
                let dt_bias = load_vec(&format!("{p}.mixer.ssm_dt_bias"), n_mamba_h)?
                    .reshape((n_mamba_h, 1))?;
                let d_skip =
                    load_vec(&format!("{p}.mixer.ssm_d"), n_mamba_h)?.reshape((n_mamba_h, 1))?;
                let mamba = MambaWeights {
                    ssm_in: linear(
                        h,
                        n_mamba_h * (2 * d_v + 3),
                        vb.pp(format!("{p}.mixer.ssm_in")),
                    )?,
                    ssm_in_dyn: linear(
                        h,
                        2 * config.ssm_group_count * d_qk * r + n_ang,
                        vb.pp(format!("{p}.mixer.ssm_in_dyn")),
                    )?,
                    bias: Tensor::cat(
                        &[
                            load_mat3(&format!("{p}.mixer.ssm_b_bias"), n_mamba_h, r, d_qk)?,
                            load_mat3(&format!("{p}.mixer.ssm_c_bias"), n_mamba_h, r, d_qk)?,
                        ],
                        1,
                    )?,
                    mxz: mxz.permute((0, 2, 1))?.contiguous()?,
                    norm_weight: Tensor::cat(
                        &[
                            norms.get(0)?.reshape((1, d_qk))?.expand((r, d_qk))?,
                            norms.get(1)?.reshape((1, d_qk))?.expand((r, d_qk))?,
                        ],
                        0,
                    )?
                    .contiguous()?,
                    dt_bias,
                    d_skip,
                    mimo_o: load_mat3(&format!("{p}.mixer.ssm_mimo_o"), n_mamba_h, r, d_v)?
                        .permute((0, 2, 1))?
                        .contiguous()?,
                    wo: linear(n_mamba_h * d_v, h, vb.pp(format!("{p}.mixer.attn_output")))?,
                };
                states.push(LayerState::Mamba {
                    s: Tensor::zeros((n_mamba_h, d_v, d_qk), DType::F32, &device)?,
                    k: Tensor::zeros((n_mamba_h, r, d_qk), DType::F32, &device)?,
                    v: Tensor::zeros((n_mamba_h, r, d_v), DType::F32, &device)?,
                    ang: Tensor::zeros((n_mamba_h, n_ang), DType::F32, &device)?,
                });
                (Some(mamba), None)
            };
            layers.push(LayerWeights {
                is_v,
                mamba,
                attn,
                moe,
                geo_mixer_scale: load_scalar(&format!("{p}.geo_mixer_scale"))?,
                geo_mixer_bias: load_scalar(&format!("{p}.geo_mixer_bias"))?,
                geo_mlp_scale: load_scalar(&format!("{p}.geo_mlp_scale"))?,
                geo_mlp_bias: load_scalar(&format!("{p}.geo_mlp_bias"))?,
            });
        }

        let _ = v;
        let num_layers = config.num_hidden_layers;
        Ok(Self {
            config,
            embed,
            lm_head,
            layers,
            state: Mutex::new(OlalaState { layers: states }),
            cache: EitherCache::Normal(crate::kv_cache::NormalCache::new(num_layers, max_seq_len)),
            device,
            dtype,
            cfg,
        })
    }

    fn mamba_forward(
        &self,
        x: &Tensor,
        w: &MambaWeights,
        s_state: &mut Tensor,
        k_state: &mut Tensor,
        v_state: &mut Tensor,
        ang_state: &mut Tensor,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.ssm_inner_size / cfg.ssm_time_step_rank;
        let d_v = cfg.ssm_time_step_rank;
        let d_qk = cfg.ssm_state_size;
        let r = cfg.mamba_mimo_dim;
        let n_ang = cfg.num_rope_angles;
        let eps = cfg.rms_norm_eps as f32;

        let (ps, pd) = match group_forward(x, &[&w.ssm_in, &w.ssm_in_dyn])? {
            Some(v) => (v[0].clone(), v[1].clone()),
            None => (w.ssm_in.forward(x)?, w.ssm_in_dyn.forward(x)?),
        };
        let ps = ps.reshape((h, 2 * d_v + 3))?;
        let z_in = ps.narrow(1, 0, d_v)?;
        let x_in = ps.narrow(1, d_v, d_v)?;
        let dt_raw = ps.narrow(1, 2 * d_v, 1)?;
        let a_raw = ps.narrow(1, 2 * d_v + 1, 1)?;
        let trap_raw = ps.narrow(1, 2 * d_v + 2, 1)?;

        let bc = pd
            .narrow(1, 0, 2 * cfg.ssm_group_count * d_qk * r)?
            .reshape((2 * r, d_qk))?;
        let ang_raw = pd
            .narrow(1, 2 * cfg.ssm_group_count * d_qk * r, n_ang)?
            .reshape((n_ang,))?;

        let inv = rms_inv(&bc, eps)?.reshape((2 * r, 1))?;
        let normed = bc
            .broadcast_mul(&inv)?
            .mul(&w.norm_weight)?
            .reshape((1, 2 * r, d_qk))?;

        let dt = softplus(&dt_raw.add(&w.dt_bias)?)?;
        let a_neg = softplus(&a_raw)?.neg()?.clamp(f32::MIN, -A_FLOOR)?;
        let alpha = a_neg.mul(&dt)?.exp()?;
        let trap = sigmoid(&trap_raw)?;
        let beta = trap.affine(-1.0, 1.0)?.mul(&dt)?.mul(&alpha)?;
        let gamma = trap.mul(&dt)?;

        let delta_ang = ang_raw
            .reshape((1, n_ang))?
            .tanh()?
            .affine(f64::from(PI), 0.0)?
            .broadcast_mul(&dt)?;
        let ang_new = wrap_angle(&ang_state.add(&delta_ang)?)?;
        let cos_a = ang_new.cos()?.reshape((h, 1, n_ang))?;
        let sin_a = ang_new.sin()?.reshape((h, 1, n_ang))?;

        let mut k_b = normed
            .narrow(1, 0, r)?
            .broadcast_add(&w.bias.narrow(1, 0, r)?)?;
        let mut q_b = normed
            .narrow(1, r, r)?
            .broadcast_add(&w.bias.narrow(1, r, r)?)?;

        let quarter = n_ang;
        let half = d_qk / 2;
        for mat in [&mut k_b, &mut q_b] {
            let g0 = mat.narrow(2, 0, quarter)?;
            let g1 = mat.narrow(2, quarter, quarter)?;
            let g2 = mat.narrow(2, half, quarter)?;
            let g3 = mat.narrow(2, half + quarter, d_qk - half - quarter)?;
            let n0 = g0.broadcast_mul(&cos_a)?.sub(&g2.broadcast_mul(&sin_a)?)?;
            let n2 = g0.broadcast_mul(&sin_a)?.add(&g2.broadcast_mul(&cos_a)?)?;
            *mat = Tensor::cat(&[n0, g1, n2, g3], 2)?.contiguous()?;
        }

        let v_proj = x_in
            .reshape((h, d_v, 1))?
            .broadcast_mul(&w.mxz.narrow(2, 0, r)?)?;
        let z_proj = z_in
            .reshape((h, d_v, 1))?
            .broadcast_mul(&w.mxz.narrow(2, r, r)?)?;

        let prev_kv = k_state
            .permute((0, 2, 1))?
            .contiguous()?
            .matmul(&v_state.contiguous()?)?
            .permute((0, 2, 1))?
            .contiguous()?;
        let curr_kv = k_b
            .permute((0, 2, 1))?
            .contiguous()?
            .matmul(&v_proj.permute((0, 2, 1))?.contiguous()?)?
            .permute((0, 2, 1))?
            .contiguous()?;

        let alpha_b = alpha.reshape((h, 1, 1))?;
        let beta_b = beta.reshape((h, 1, 1))?;
        let gamma_b = gamma.reshape((h, 1, 1))?;
        let s_new = s_state
            .broadcast_mul(&alpha_b)?
            .add(&prev_kv.broadcast_mul(&beta_b)?)?
            .add(&curr_kv.broadcast_mul(&gamma_b)?)?
            .contiguous()?;

        let qstate = s_new.matmul(&q_b.permute((0, 2, 1))?.contiguous()?)?;
        let o = qstate.add(&v_proj.broadcast_mul(&w.d_skip.reshape((h, 1, 1))?)?)?;
        let o = o.mul(&z_proj.silu()?)?;
        let y = o.mul(&w.mimo_o)?.sum(D::Minus1)?.reshape((h * d_v,))?;

        *s_state = s_new;
        *k_state = k_b.contiguous()?;
        *v_state = v_proj.permute((0, 2, 1))?.contiguous()?;
        *ang_state = ang_new.contiguous()?;

        w.wo.forward(&y.reshape((1, h * d_v))?)
    }

    fn attention_forward(
        &self,
        x: &Tensor,
        w: &AttentionWeights,
        pos: usize,
        state: &mut LayerState,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let h = cfg.num_attention_heads;
        let hs = cfg.num_signal_heads;
        let hn = cfg.num_noise_heads;
        let r = cfg.tpa_rank;
        let dh = cfg.head_dim;
        let g = h / hn;
        let snr = hs / hn;
        let eps = cfg.rms_norm_eps as f32;

        let members: [&Arc<dyn QuantMethod>; 9] = [
            &w.wq, &w.wa_k, &w.wa_v, &w.wb_k, &w.wb_v, &w.ws_k, &w.ws_v, &w.wlambda, &w.wgate,
        ];
        let outs = group_forward(x, &members)?;
        let proj = |i: usize, layer: &Arc<dyn QuantMethod>| -> Result<Tensor> {
            match &outs {
                Some(v) => Ok(v[i].clone()),
                None => layer.forward(x),
            }
        };
        let q = proj(0, &w.wq)?.reshape((h, dh))?;
        let ak = proj(1, &w.wa_k)?.reshape((hn, r))?;
        let av = proj(2, &w.wa_v)?.reshape((hn, r))?;
        let bk = proj(3, &w.wb_k)?.reshape((r, dh))?;
        let bv = proj(4, &w.wb_v)?.reshape((r, dh))?;
        let sk = proj(5, &w.ws_k)?.reshape((hn,))?;
        let sv = proj(6, &w.ws_v)?.reshape((hn,))?;
        let ell = proj(7, &w.wlambda)?.reshape((hn,))?;
        let gate = proj(8, &w.wgate)?.reshape((hs, dh))?;

        let k0 = ak.matmul(&bk)?.affine(1.0 / r as f64, 0.0)?;
        let v0 = av.matmul(&bv)?.affine(1.0 / r as f64, 0.0)?;

        let (shift, k_rows, v_rows) = match state {
            LayerState::Attention {
                shift,
                k_rows,
                v_rows,
            } => (shift, k_rows, v_rows),
            _ => candle_core::bail!("Olala attention layer has wrong state kind"),
        };

        let pmask = if pos == 0 { 0.0f32 } else { 1.0f32 };
        let a_k = sigmoid(&sk)?.affine(f64::from(pmask), 0.0)?;
        let a_v = sigmoid(&sv)?.affine(f64::from(pmask), 0.0)?;
        let k_prev = shift.narrow(0, 0, hn * dh)?.reshape((hn, dh))?;
        let v_prev = shift.narrow(0, hn * dh, hn * dh)?.reshape((hn, dh))?;
        let ktil = k_prev
            .broadcast_mul(&a_k.reshape((hn, 1))?)?
            .add(&k0.broadcast_mul(&a_k.affine(-1.0, 1.0)?.reshape((hn, 1))?)?)?;
        let vtil = v_prev
            .broadcast_mul(&a_v.reshape((hn, 1))?)?
            .add(&v0.broadcast_mul(&a_v.affine(-1.0, 1.0)?.reshape((hn, 1))?)?)?;
        *shift = Tensor::cat(&[k0.reshape((hn * dh,))?, v0.reshape((hn * dh,))?], 0)?;

        let rsk = rms_inv(&ktil, eps)?;
        let rsq = rms_inv(&q, eps)?;
        let k_cell = ktil.broadcast_mul(&rsk.reshape((hn, 1))?)?;

        let lp = ((pos + 1).min(cfg.sliding_window.max(1)) as f32).ln();
        let qs = q
            .broadcast_mul(&rsq.reshape((h, 1))?)?
            .broadcast_mul(&w.q_norm.reshape((1, dh))?)?
            .broadcast_mul(&w.k_norm.reshape((1, dh))?)?
            .broadcast_mul(&w.scaler.affine(f64::from(lp), 0.0)?.reshape((h, 1))?)?;

        k_rows.push(k_cell.reshape((hn * dh,))?.contiguous()?);
        v_rows.push(vtil.reshape((hn * dh,))?.contiguous()?);

        let n_cells = k_rows.len();
        let k_all = Tensor::stack(k_rows, 0)?.reshape((n_cells, hn, dh))?;
        let v_all = Tensor::stack(v_rows, 0)?.reshape((n_cells, hn, dh))?;

        let softcap = cfg.attn_logit_softcapping as f32;
        let qg = qs.reshape((hn, g, dh))?;
        let kk = k_all.permute((1, 2, 0))?.contiguous()?;
        let vv = v_all.permute((1, 0, 2))?.contiguous()?;
        let mut s = qg.matmul(&kk)?.affine(1.0 / dh as f64, 0.0)?;
        if softcap > 0.0 {
            s = s
                .affine(1.0 / f64::from(softcap), 0.0)?
                .tanh()?
                .affine(f64::from(softcap), 0.0)?;
        }
        let s = candle_nn::ops::softmax_last_dim(&s)?;
        let o_full = s.matmul(&vv)?.reshape((h, dh))?;

        let lam = sigmoid(&ell)?;
        let gate = gate.affine(1.0, cfg.gate_bias)?.silu()?;
        let o_grouped = o_full.reshape((hn, g, dh))?;
        let sig = o_grouped.narrow(1, 0, snr)?;
        let noi = o_grouped.narrow(1, snr, 1)?;
        let comb = sig.broadcast_sub(&noi.broadcast_mul(&lam.reshape((hn, 1, 1))?)?)?;
        let y = comb.reshape((hs, dh))?.mul(&gate)?;
        w.wo.forward(&y.reshape((1, hs * dh))?)
    }

    fn moe_forward(&self, x: &Tensor, w: &MoeWeights) -> Result<Tensor> {
        let cfg = &self.config;
        let k = cfg.expert_used_count;
        let n_exp = cfg.expert_count;
        let (latent, logits) = match group_forward(x, &[&w.latent_down, &w.gate_inp])? {
            Some(v) => (v[0].clone(), v[1].clone()),
            None => (w.latent_down.forward(x)?, w.gate_inp.forward(x)?),
        };

        // Router math stays in f32 (mandatory for top-k parity). Compute the bias-adjusted
        // scores in the same loop that selects the experts, avoiding a second full
        // device->host materialization.
        let logits_v = logits.reshape((n_exp,))?.to_vec1::<f32>()?;
        let bias_v = &w.exp_probs_b;
        let mut selected = Vec::with_capacity(k);
        let mut used = vec![false; n_exp];
        for _ in 0..k {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for e in 0..n_exp {
                if !used[e] {
                    let score = 1.0 / (1.0 + (-logits_v[e]).exp()) + bias_v[e];
                    if score > best_v {
                        best_v = score;
                        best = e;
                    }
                }
            }
            used[best] = true;
            selected.push(best);
        }

        let mut weights: Vec<f32> = selected
            .iter()
            .map(|&e| 1.0 / (1.0 + (-logits_v[e]).exp()))
            .collect();
        let sum: f32 = weights.iter().sum();
        let sum = sum.max(SUM_CLAMP);
        for weight in &mut weights {
            *weight = *weight / sum * cfg.expert_weights_scale as f32;
        }

        let latent_dim = latent.dim(1)?;
        let indices = Tensor::from_vec(
            selected.iter().map(|&e| e as u32).collect::<Vec<_>>(),
            (1, k),
            &self.device,
        )?;
        let weights_t = Tensor::from_vec(weights, (1, k, 1), &self.device)?;
        let x3 = latent.reshape((1, 1, latent_dim))?;
        let up = w.up_exps.gather_forward(&x3, &indices)?; // (1, k, ff_exp)
        let h = up.relu()?.sqr()?;
        let out = w.down_exps.gather_forward(&h, &indices)?; // (1, k, latent)
        let routed = out.broadcast_mul(&weights_t)?.sum(1)?; // (1, latent)
        let routed = w.latent_up.forward(&routed)?;

        let sh = w.up_shexp.forward(x)?.relu()?.sqr()?;
        let shared = w.down_shexp.forward(&sh)?;
        routed.add(&shared)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        ctx: &crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let offsets = ctx.seqlen_offsets();
        let (batch, seq_len) = input_ids.dims2()?;
        if batch != 1 {
            candle_core::bail!("Olala currently supports batch size 1 only");
        }
        let start_pos = offsets.first().copied().unwrap_or(0);
        let mut state = self.state.lock().unwrap();
        if start_pos == 0 {
            for layer in state.layers.iter_mut() {
                match layer {
                    LayerState::Mamba { s, k, v, ang } => {
                        *s = s.zeros_like()?;
                        *k = k.zeros_like()?;
                        *v = v.zeros_like()?;
                        *ang = ang.zeros_like()?;
                    }
                    LayerState::Attention {
                        shift,
                        k_rows,
                        v_rows,
                    } => {
                        *shift = shift.zeros_like()?;
                        k_rows.clear();
                        v_rows.clear();
                    }
                }
            }
        }

        let tokens = input_ids.reshape((seq_len,))?.to_vec1::<u32>()?;
        let ids = Tensor::from_vec(tokens, (seq_len,), &self.device)?;
        let embedded = self.embed.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        let norm = embedded.sqr()?.sum(D::Minus1)?.affine(1.0, 1e-12)?.sqrt()?;
        let inputs = embedded.broadcast_div(&norm.unsqueeze(D::Minus1)?)?;

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let pos = start_pos + t;
            let x = inputs.narrow(0, t, 1)?;
            let mut residual = x.clone();
            for (i, layer) in self.layers.iter().enumerate() {
                let mixer = if layer.is_v {
                    self.attention_forward(
                        &residual,
                        layer.attn.as_ref().unwrap(),
                        pos,
                        &mut state.layers[i],
                    )?
                } else {
                    let (s_st, k_st, v_st, ang_st) = match &mut state.layers[i] {
                        LayerState::Mamba { s, k, v, ang } => (s, k, v, ang),
                        _ => candle_core::bail!("Olala mamba layer has wrong state kind"),
                    };
                    self.mamba_forward(
                        &residual,
                        layer.mamba.as_ref().unwrap(),
                        s_st,
                        k_st,
                        v_st,
                        ang_st,
                    )?
                };
                let cur = geodesic(
                    &residual,
                    &mixer,
                    layer.geo_mixer_scale,
                    layer.geo_mixer_bias,
                    i,
                )?;
                let mlp = self.moe_forward(&cur, &layer.moe)?;
                residual = geodesic(&cur, &mlp, layer.geo_mlp_scale, layer.geo_mlp_bias, i)?;
            }
            outputs.push(residual);
        }
        let last = Tensor::cat(&outputs, 0)?;
        let logits = self.lm_head.forward(&last)?;
        Ok(logits)
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
        self.forward_inner(input_ids, ctx)
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
        _flash_params: &crate::pipeline::text_models_inputs_processor::FlashParams,
        _flash_params_full: &crate::pipeline::text_models_inputs_processor::FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("Olala does not support X-LoRA")
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
        self.config.max_position_embeddings
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for Model {}
