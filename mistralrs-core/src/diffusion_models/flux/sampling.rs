#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use serde::Deserialize;

use super::stepper::FluxTransformer;

/// Generate noise tensor for diffusion.
///
/// `latent_channels`: Number of latent channels (16 for FLUX.1, 32 for FLUX.2)
pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    latent_channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let height = height.div_ceil(16) * 2;
    let width = width.div_ceil(16) * 2;
    Tensor::randn(0f32, 1., (num_samples, latent_channels, height, width), device)
}

pub fn get_noise_flux2(
    num_samples: usize,
    height: usize,
    width: usize,
    latent_channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let height = (height / 16) * 2;
    let width = (width / 16) * 2;
    Tensor::randn(0f32, 1., (num_samples, latent_channels, height, width), device)
}

#[derive(Debug, Clone)]
pub struct State {
    pub img: Tensor,
    pub img_ids: Tensor,
    pub txt: Tensor,
    pub txt_ids: Tensor,
    pub vec: Tensor,
}

impl State {
    /// Create state for FLUX.1 (requires both T5 and CLIP embeddings)
    pub fn new(
        t5_emb: &Tensor,
        clip_emb: &Tensor,
        img: &Tensor,
        n_axes: usize,
    ) -> Result<Self> {
        let (img, img_ids, txt, txt_ids) = Self::prepare_common(t5_emb, img, n_axes)?;
        let bs = img.dim(0)?;
        let vec = clip_emb.repeat(bs)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }

    /// Create state for FLUX.2 (only T5 embeddings, no CLIP)
    pub fn new_flux2(t5_emb: &Tensor, img: &Tensor, n_axes: usize) -> Result<Self> {
        let (img, img_ids, txt, txt_ids) = Self::prepare_common(t5_emb, img, n_axes)?;
        let bs = img.dim(0)?;
        let dev = img.device();
        let dtype = img.dtype();
        // FLUX.2 doesn't use vec (CLIP pooled embedding), create empty placeholder
        let vec = Tensor::zeros((bs, 0), dtype, dev)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }

    /// Common preparation for both FLUX.1 and FLUX.2
    fn prepare_common(
        t5_emb: &Tensor,
        img: &Tensor,
        n_axes: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        if n_axes != 3 && n_axes != 4 {
            candle_core::bail!("unsupported rope axes count {n_axes}")
        }
        let dtype = img.dtype();
        let (bs, c, h, w) = img.dims4()?;
        let dev = img.device();
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?; // (b, c, h, ph, w, pw)
        let img = img.permute((0, 2, 4, 1, 3, 5))?; // (b, h, w, c, ph, pw)
        let img = img.reshape((bs, h / 2 * w / 2, c * 4))?;
        let t_ids = Tensor::full(0u32, (h / 2, w / 2), dev)?;
        let h_ids = Tensor::arange(0u32, h as u32 / 2, dev)?
            .reshape(((), 1))?
            .broadcast_as((h / 2, w / 2))?;
        let w_ids = Tensor::arange(0u32, w as u32 / 2, dev)?
            .reshape((1, ()))?
            .broadcast_as((h / 2, w / 2))?;
        let img_ids = if n_axes == 3 {
            Tensor::stack(&[t_ids.clone(), h_ids, w_ids], 2)?
        } else {
            let l_ids = Tensor::full(0u32, (h / 2, w / 2), dev)?;
            Tensor::stack(&[t_ids.clone(), h_ids, w_ids, l_ids], 2)?
        }
        .to_dtype(dtype)?;
        let img_ids = img_ids.reshape((1, h / 2 * w / 2, n_axes))?;
        let img_ids = img_ids.repeat((bs, 1, 1))?;
        let txt = t5_emb.repeat(bs)?;
        let txt_len = txt.dim(1)?;
        let txt_ids = if n_axes == 3 {
            Tensor::zeros((bs, txt_len, n_axes), dtype, dev)?
        } else {
            let l_ids = Tensor::arange(0u32, txt_len as u32, dev)?;
            let zeros = Tensor::zeros_like(&l_ids)?;
            let txt_ids = Tensor::stack(&[zeros.clone(), zeros.clone(), zeros.clone(), l_ids], 1)?;
            txt_ids
                .reshape((1, txt_len, n_axes))?
                .repeat((bs, 1, 1))?
                .to_dtype(dtype)?
        };
        Ok((img, img_ids, txt, txt_ids))
    }
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.73809524e-05, 1.89833333);
    let (a2, b2) = (0.00016927, 0.45666666);

    if image_seq_len > 4300 {
        return a2 * image_seq_len as f64 + b2;
    }

    let m_200 = a2 * image_seq_len as f64 + b2;
    let m_10 = a1 * image_seq_len as f64 + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

#[derive(Debug, Clone, Deserialize)]
pub struct FlowMatchScheduleConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    pub shift_terminal: Option<f64>,
    #[serde(default = "default_time_shift_type")]
    pub time_shift_type: String,
}

fn default_num_train_timesteps() -> usize {
    1000
}

fn default_shift() -> f64 {
    1.0
}

fn default_time_shift_type() -> String {
    "exponential".to_string()
}

impl Default for FlowMatchScheduleConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: default_num_train_timesteps(),
            shift: default_shift(),
            use_dynamic_shifting: false,
            shift_terminal: None,
            time_shift_type: default_time_shift_type(),
        }
    }
}

fn time_shift_kind(cfg: &FlowMatchScheduleConfig, mu: f64, sigma: f64, t: f64) -> f64 {
    match cfg.time_shift_type.as_str() {
        "linear" => mu / (mu + (1. / t - 1.).powf(sigma)),
        _ => time_shift(mu, sigma, t),
    }
}

fn stretch_shift_to_terminal(sigmas: &mut [f64], shift_terminal: f64) {
    if sigmas.is_empty() {
        return;
    }
    let one_minus_last = 1.0 - sigmas[sigmas.len() - 1];
    let scale_factor = one_minus_last / (1.0 - shift_terminal);
    for s in sigmas.iter_mut() {
        let one_minus = 1.0 - *s;
        *s = 1.0 - (one_minus / scale_factor);
    }
}

/// `shift` is a triple `(image_seq_len, base_shift, max_shift)`.
pub fn get_schedule(num_steps: usize, shift: Option<(usize, f64, f64)>) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .rev()
        .collect();
    match shift {
        None => timesteps,
        Some((image_seq_len, y1, y2)) => {
            let (x1, x2) = (256., 4096.);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            timesteps
                .into_iter()
                .map(|v| time_shift(mu, 1., v))
                .collect()
        }
    }
}

/// FlowMatch Euler schedule for FLUX.2 (sigmas, with dynamic shifting and terminal zero).
pub fn get_schedule_flux2(
    num_steps: usize,
    image_seq_len: usize,
    cfg: &FlowMatchScheduleConfig,
) -> Vec<f64> {
    let sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            if num_steps <= 1 {
                1.0
            } else {
                let t = i as f64 / (num_steps as f64 - 1.0);
                1.0 + (1.0 / cfg.num_train_timesteps as f64 - 1.0) * t
            }
        })
        .collect();

    let mut shifted: Vec<f64> = if cfg.use_dynamic_shifting {
        let mu = compute_empirical_mu(image_seq_len, num_steps);
        sigmas
            .into_iter()
            .map(|s| time_shift_kind(cfg, mu, 1.0, s))
            .collect()
    } else {
        sigmas
            .into_iter()
            .map(|s| cfg.shift * s / (1.0 + (cfg.shift - 1.0) * s))
            .collect()
    };

    if let Some(shift_terminal) = cfg.shift_terminal {
        stretch_shift_to_terminal(&mut shifted, shift_terminal);
    }

    shifted.push(0.0);
    shifted
}

pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _h_w, c_ph_pw) = xs.dims3()?;
    let height = height.div_ceil(16);
    let width = width.div_ceil(16);
    xs.reshape((b, height, width, c_ph_pw / 4, 2, 2))? // (b, h, w, c, ph, pw)
        .permute((0, 3, 1, 4, 2, 5))? // (b, c, h, ph, w, pw)
        .reshape((b, c_ph_pw / 4, height * 2, width * 2))
}

pub fn unpack_packed(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let (b, _h_w, c) = xs.dims3()?;
    let height = height / 16;
    let width = width / 16;
    xs.reshape((b, height, width, c))?
        .permute((0, 3, 1, 2))
}

pub fn unpatchify_packed(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    let c_out = c / 4;
    xs.reshape((b, c_out, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b, c_out, h * 2, w * 2))
}

pub fn patchify_latents(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c, h / 2, 2, w / 2, 2))?
        .permute((0, 1, 3, 5, 2, 4))?
        .reshape((b, c * 4, h / 2, w / 2))
}

pub fn pack_latents(xs: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = xs.dims4()?;
    xs.reshape((b, c, h * w))?.permute((0, 2, 1))
}

/// Denoise using FluxTransformer (supports both FLUX.1 and FLUX.2)
#[allow(clippy::too_many_arguments)]
fn denoise_inner(
    model: &mut FluxTransformer,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: Option<f64>,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance_tensor = if let Some(guidance) = guidance {
        Some(Tensor::full(guidance as f32, b_sz, dev)?)
    } else {
        None
    };
    let mut img = img.clone();
    for window in timesteps.windows(2) {
        let (t_curr, t_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
        let pred = model.forward(&img, img_ids, txt, txt_ids, &t_vec, vec_, guidance_tensor.as_ref())?;
        img = (img + pred * (t_prev - t_curr))?
    }
    Ok(img)
}

/// Denoise with guidance (for FLUX.1-dev style models)
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &mut FluxTransformer,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
) -> Result<Tensor> {
    denoise_inner(
        model,
        img,
        img_ids,
        txt,
        txt_ids,
        vec_,
        timesteps,
        Some(guidance),
    )
}

/// Denoise without guidance (for FLUX.1-schnell and FLUX.2 style models)
#[allow(clippy::too_many_arguments)]
pub fn denoise_no_guidance(
    model: &mut FluxTransformer,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
) -> Result<Tensor> {
    denoise_inner(model, img, img_ids, txt, txt_ids, vec_, timesteps, None)
}
