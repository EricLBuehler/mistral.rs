use candle_core::{DType, Device, Result, Tensor, D};
use serde::Deserialize;

use super::DiffusionGemmaModel;
use crate::pipeline::text_positions_tensor;

const DEFAULT_MAX_DENOISING_STEPS: usize = 48;
const DEFAULT_ENTROPY_BOUND: f64 = 0.1;
const DEFAULT_T_MIN: f64 = 0.4;
const DEFAULT_T_MAX: f64 = 0.8;
const DEFAULT_STABILITY_THRESHOLD: usize = 1;
const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.005;
const GUMBEL_EPS: f64 = 1e-20;

/// Diffusion sampling parameters; the checkpoint's `generation_config.json` is the source
/// of truth (request-level temperature/top_p do not apply to block diffusion).
#[derive(Debug, Clone)]
pub struct DiffusionParams {
    pub max_denoising_steps: usize,
    pub entropy_bound: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub stability_threshold: usize,
    pub confidence_threshold: f64,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            max_denoising_steps: DEFAULT_MAX_DENOISING_STEPS,
            entropy_bound: DEFAULT_ENTROPY_BOUND,
            t_min: DEFAULT_T_MIN,
            t_max: DEFAULT_T_MAX,
            stability_threshold: DEFAULT_STABILITY_THRESHOLD,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        }
    }
}

#[derive(Deserialize)]
struct RawSamplerConfig {
    entropy_bound: Option<f64>,
}

#[derive(Deserialize)]
struct RawGenerationConfig {
    max_denoising_steps: Option<usize>,
    sampler_config: Option<RawSamplerConfig>,
    t_min: Option<f64>,
    t_max: Option<f64>,
    stability_threshold: Option<usize>,
    confidence_threshold: Option<f64>,
}

impl DiffusionParams {
    pub fn from_generation_config(raw: &str) -> Self {
        let defaults = Self::default();
        let Ok(parsed) = serde_json::from_str::<RawGenerationConfig>(raw) else {
            return defaults;
        };
        Self {
            max_denoising_steps: parsed
                .max_denoising_steps
                .unwrap_or(defaults.max_denoising_steps),
            entropy_bound: parsed
                .sampler_config
                .and_then(|sc| sc.entropy_bound)
                .unwrap_or(defaults.entropy_bound),
            t_min: parsed.t_min.unwrap_or(defaults.t_min),
            t_max: parsed.t_max.unwrap_or(defaults.t_max),
            stability_threshold: parsed
                .stability_threshold
                .unwrap_or(defaults.stability_threshold),
            confidence_threshold: parsed
                .confidence_threshold
                .unwrap_or(defaults.confidence_threshold),
        }
    }
}

fn random_canvas(
    num_seqs: usize,
    canvas_length: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let uniform = Tensor::rand(0f32, 1f32, (num_seqs, canvas_length), device)?;
    (uniform * vocab_size as f64)?.floor()?.to_dtype(DType::U32)
}

/// Denoise one canvas per sequence, lockstep across the batch. All sequences share the
/// same context length (`cache_offsets` may still differ only by prefix-cache trims, so
/// each gets its own rope offset). Returns each sequence's committed argmax canvas (what
/// HF commits, NOT the renoised stochastic canvas). Sequences that converge early are
/// frozen in place; the extra passes they ride along for are nearly free since pass cost
/// is dominated by streaming the expert weights.
pub fn generate_canvas(
    model: &DiffusionGemmaModel,
    params: &DiffusionParams,
    canvas_kv: &[(Tensor, Tensor)],
    cache_offsets: &[usize],
    device: &Device,
) -> Result<(Vec<Vec<u32>>, std::time::Duration)> {
    let canvas_length = model.canvas_length();
    let vocab_size = model.config().text_config.vocab_size;
    let num_seqs = cache_offsets.len();
    let positions = text_positions_tensor(cache_offsets, canvas_length, device)?;

    // Validation hook: deterministic canvas, one raw denoise pass, dump logits, exit.
    // Engine warmup uses a 1-token dummy prompt; skip it and fire on the real prompt.
    if let (Ok(dump_path), true) = (
        std::env::var("MISTRALRS_DIFFUSION_DEBUG_DUMP"),
        num_seqs == 1 && cache_offsets[0] > 1,
    ) {
        let ids: Vec<u32> = (0..canvas_length as u32)
            .map(|i| i * 977 % vocab_size as u32)
            .collect();
        let fixed = Tensor::from_vec(ids, (1, canvas_length), device)?;
        let logits = model.denoise_step(&fixed, None, &positions, canvas_kv)?;
        logits.to_dtype(DType::F32)?.write_npy(&dump_path)?;
        tracing::info!(
            "Wrote diffusion debug logits (offset {}) to {dump_path}",
            cache_offsets[0]
        );
        std::process::exit(0);
    }

    let mut canvas = random_canvas(num_seqs, canvas_length, vocab_size, device)?;
    let mut argmax_canvas: Option<Tensor> = None;
    let mut sc_soft: Option<Tensor> = None;
    let mut history: Vec<Tensor> = Vec::with_capacity(params.stability_threshold);
    let mut finished = vec![false; num_seqs];
    let mut finished_mask: Option<Tensor> = None;
    let block_start = std::time::Instant::now();
    let mut passes = 0usize;

    for cur_step in (1..=params.max_denoising_steps).rev() {
        passes += 1;
        let logits = model.denoise_step(&canvas, sc_soft.as_ref(), &positions, canvas_kv)?;

        // Linear temperature schedule: cur_step counts down, so the first pass is hottest.
        let temperature = params.t_min
            + (params.t_max - params.t_min) * (cur_step as f64 / params.max_denoising_steps as f64);
        let scaled = (logits.to_dtype(DType::F32)? / temperature)?;

        // Gumbel-max: argmax(logits/T - ln(-ln(u))) ~ softmax(logits/T) sample.
        let u = scaled.rand_like(GUMBEL_EPS, 1.0)?;
        let gumbel = u.log()?.neg()?.log()?.neg()?;
        let denoiser_canvas = (&scaled + gumbel)?
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?;
        let mut new_argmax = scaled.argmax(D::Minus1)?.to_dtype(DType::U32)?;

        let log_probs = candle_nn::ops::log_softmax(&scaled, D::Minus1)?;
        let probs = log_probs.exp()?;
        let token_entropy = (&probs * &log_probs)?.sum(D::Minus1)?.neg()?;

        // Entropy-bound acceptance per sequence: take the k lowest-entropy tokens such that
        // sum(entropy_1..k) - max(entropy_1..k) <= bound; ascending sort makes the running
        // max the current element, so no cummax is needed.
        let (sorted_entropy, sorted_indices) = token_entropy.sort_last_dim(true)?;
        let cumulative = sorted_entropy.cumsum(D::Minus1)?;
        let sorted_mask = (cumulative - &sorted_entropy)?
            .le(params.entropy_bound)?
            .to_dtype(DType::U8)?;
        let accept_mask = Tensor::zeros((num_seqs, canvas_length), DType::U8, device)?.scatter(
            &sorted_indices,
            &sorted_mask,
            D::Minus1,
        )?;

        // Accepted positions take the denoiser sample; the rest are renoised uniformly.
        let renoised = random_canvas(num_seqs, canvas_length, vocab_size, device)?;
        let mut new_canvas = accept_mask.where_cond(&denoiser_canvas, &renoised)?;

        // Freeze converged sequences: keep their previous canvas/argmax rows.
        if let (Some(mask), Some(prev_argmax)) = (&finished_mask, &argmax_canvas) {
            new_canvas = mask.where_cond(&canvas, &new_canvas)?;
            new_argmax = mask.where_cond(prev_argmax, &new_argmax)?;
        }
        canvas = new_canvas;

        // Stop a sequence when its argmax canvas is stable across `stability_threshold`
        // steps AND its mean token entropy is confidently low. One host read per signal.
        let mean_entropy = token_entropy.mean(D::Minus1)?.to_vec1::<f32>()?;
        let stable: Vec<bool> = if history.len() == params.stability_threshold {
            let mut all_eq = vec![true; num_seqs];
            for prev in &history {
                let mismatches = prev
                    .ne(&new_argmax)?
                    .to_dtype(DType::F32)?
                    .sum(D::Minus1)?
                    .to_vec1::<f32>()?;
                for (eq, m) in all_eq.iter_mut().zip(mismatches) {
                    *eq &= m == 0.0;
                }
            }
            all_eq
        } else {
            vec![params.stability_threshold == 0; num_seqs]
        };
        if params.stability_threshold > 0 {
            if history.len() == params.stability_threshold {
                history.remove(0);
            }
            history.push(new_argmax.clone());
        }
        argmax_canvas = Some(new_argmax);

        let mut changed = false;
        for i in 0..num_seqs {
            if !finished[i] && stable[i] && mean_entropy[i] < params.confidence_threshold as f32 {
                finished[i] = true;
                changed = true;
            }
        }
        if finished.iter().all(|&f| f) {
            break;
        }
        if changed || (finished_mask.is_none() && finished.iter().any(|&f| f)) {
            let mask: Vec<u8> = finished.iter().map(|&f| f as u8).collect();
            finished_mask = Some(
                Tensor::from_vec(mask, (num_seqs, 1), device)?
                    .broadcast_as((num_seqs, canvas_length))?,
            );
        }

        sc_soft = Some(model.soft_embed(&probs)?);
    }

    let elapsed = block_start.elapsed().as_secs_f64();
    tracing::debug!(
        "{num_seqs} canvas(es) at offsets {cache_offsets:?}: {passes} denoising passes in {elapsed:.2}s \
         ({:.1} tok/s effective, {:.0}ms/pass)",
        (num_seqs * canvas_length) as f64 / elapsed,
        elapsed * 1000.0 / passes as f64,
    );

    let blocks = argmax_canvas
        .expect("max_denoising_steps >= 1 guarantees at least one pass")
        .to_vec2::<u32>()?;
    Ok((blocks, block_start.elapsed()))
}
