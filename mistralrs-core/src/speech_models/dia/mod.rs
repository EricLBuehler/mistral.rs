#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{sync::Arc, time::Instant};

use audio::{apply_audio_delay, build_delay_indices, build_revert_indices, revert_audio_delay};
use cache::DiaKvCache;
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use mistralrs_quant::{BitWiseOp, ShardedVarBuilder};
use model::DiaModel;
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    SeedableRng,
};
use rand_isaac::Isaac64Rng;

pub use config::DiaConfig;
use tracing::info;

use crate::ops::apply_triangular;

use super::{utils::normalize_loudness, SpeechGenerationConfig, SpeechGenerationOutput};

/// Aggregated outputs for generation preparation.
pub struct PrepareGenerationOutput {
    pub generated_tokens: Tensor,
    pub decoder_attn_mask: Tensor,
    pub encoder_out: Tensor,
    pub encoder_positions: Tensor,
    pub cross_cache: Vec<Option<DiaKvCache>>,
    pub self_cache: Vec<Option<DiaKvCache>>,
}

mod audio;
mod cache;
mod config;
mod dac;
mod model;

const RATE: usize = 44100;
const CHANNELS: usize = 1;
const TOKENS_PER_SECOND: usize = 86;

fn create_attn_mask(q_padding_mask_1d: &Tensor, k_padding_mask_1d: &Tensor) -> Result<Tensor> {
    let (b1, _tq) = q_padding_mask_1d.dims2()?;
    let (b2, _tk) = k_padding_mask_1d.dims2()?;
    assert_eq!(b1, b2);

    let p_mask_q = q_padding_mask_1d.unsqueeze(2)?;
    let p_mask_k = k_padding_mask_1d.unsqueeze(1)?;

    // # Condition A: Non-padding query attends to non-padding key
    // non_pad_attends_non_pad = p_mask_q & p_mask_k  # Shape [B, Tq, Tk]

    // # Condition B: Padding query attends to padding key
    // pad_attends_pad = (~p_mask_q) & (~p_mask_k)  # Shape [B, Tq, Tk]

    // # Combine: True if padding status is compatible (both non-pad OR both pad)
    // mask = non_pad_attends_non_pad | pad_attends_pad  # Shape [B, Tq, Tk]

    // let np_att_np = p_mask_q.bitwise_and(&p_mask_k)?;
    // let p_att_p = p_mask_q.bitwise_not()?.bitwise_and(&p_mask_k.bitwise_not()?)?;
    // let mask = np_att_np.bitwise_or(&p_att_p)?;

    let mask = p_mask_q.broadcast_eq(&p_mask_k)?;

    mask.unsqueeze(1)
}

pub struct DiaPipeline {
    model: DiaModel,
    cfg: DiaConfig,
    device: Device,
    dtype: DType,
    dac: dac::Model,
}

impl DiaPipeline {
    pub fn new(cfg: &DiaConfig, vb: ShardedVarBuilder, dac_vb: VarBuilder) -> Result<Self> {
        // https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth
        // https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/conf/final/44khz.yml
        let dac = dac::Model::new(&dac::Config::dia(), dac_vb.set_dtype(DType::F32))?;

        // Dia suffers accuracy issues with cublaslt.
        #[cfg(feature = "cuda")]
        mistralrs_quant::cublaslt::CUBLASLT_CONTROLLER.set_inhibit(true);

        Ok(Self {
            dtype: vb.dtype(),
            device: vb.device().clone(),
            model: DiaModel::new(cfg, vb)?,
            cfg: cfg.clone(),
            dac,
        })
    }

    fn prepare_audio_prompt(&self) -> Result<Tensor> {
        let num_channels = self.cfg.data.channels;
        let audio_pad_value = self.cfg.data.audio_pad_value;
        let audio_bos_value = self.cfg.data.audio_bos_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let prefill =
            (Tensor::ones((1, num_channels), DType::F32, &self.device)? * audio_bos_value as f64)?;

        let delay_pad_tensor = (Tensor::ones(
            (max_delay_pattern - 1, num_channels),
            DType::F32,
            &self.device,
        )? * -1f64)?;
        let prefill = Tensor::cat(&[prefill, delay_pad_tensor], 0)?;

        let delay_precomp = build_delay_indices(
            1,
            prefill.dim(0)?,
            num_channels,
            &delay_pattern.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            &self.device,
        )?;

        let prefill = apply_audio_delay(
            &prefill.unsqueeze(0)?,
            audio_pad_value as i64,
            audio_bos_value as i64,
            &delay_precomp,
        )?
        .squeeze(0)?;

        Ok(prefill)
    }

    fn prepare_text_prompt(&self, text: &str) -> Result<Tensor> {
        let text_pad_value = self.cfg.data.text_pad_value;
        let max_len = self.cfg.data.text_length;

        let text = text.replace("[S1]", "\x01").replace("[S2]", "\x02");
        let text_tokens = text.as_bytes();

        let current_len = text_tokens.len();
        let padding_needed = max_len - current_len;
        let padded_text_np = if max_len <= current_len {
            let text_tokens = &text_tokens[..max_len];
            Tensor::new(text_tokens, &self.device)?
        } else {
            let text = Tensor::new(text_tokens, &self.device)?;
            let pad =
                (Tensor::ones(padding_needed, DType::U8, &self.device)? * text_pad_value as f64)?;
            Tensor::cat(&[text, pad], 0)?
        };

        padded_text_np.to_dtype(DType::U32)?.unsqueeze(0)
    }

    /// Returns:
    /// - generated tokens
    /// - decoder attn mask
    /// - encoder out
    /// - encoder positions
    /// - cross cache
    /// - self cache
    fn prepare_generation(&self, text: &str) -> Result<PrepareGenerationOutput> {
        let enc_input_cond = self.prepare_text_prompt(text)?;
        let enc_input_uncond = enc_input_cond.zeros_like()?;
        let enc_input = Tensor::cat(&[&enc_input_uncond, &enc_input_cond], 0)?;

        let prefill = self.prepare_audio_prompt()?;

        let encoder_positions =
            Tensor::arange(0f32, self.cfg.data.text_length as f32, &self.device)?
                .unsqueeze(0)?
                .repeat((2, 1))?;
        let encoder_padding_mask = enc_input_cond
            .ne(self.cfg.data.text_pad_value)?
            .repeat((2, 1))?;
        let encoder_attn_mask = create_attn_mask(&encoder_padding_mask, &encoder_padding_mask)?;
        let encoder_out =
            self.model
                .encoder
                .forward(&enc_input, &encoder_positions, Some(&encoder_attn_mask))?;

        let decoder_cross_attn_cache = self
            .model
            .decoder
            .precompute_cross_attn_cache(&encoder_out, &encoder_positions)?;
        let decoder_padding_mask = Tensor::ones((2, 1), DType::U8, &self.device)?;
        let decoder_attn_mask = create_attn_mask(&decoder_padding_mask, &encoder_padding_mask)?;

        let max_audio_length = self.cfg.data.audio_length;
        let generated_tokens = Tensor::zeros(
            (max_audio_length, self.cfg.data.channels),
            DType::F32,
            &self.device,
        )?;

        generated_tokens.slice_set(&prefill.to_dtype(DType::F32)?, 0, 0)?;

        let mut decoder_self_attn_cache = Vec::new();
        for _ in 0..self.cfg.model.decoder.n_layer {
            decoder_self_attn_cache.push(Some(DiaKvCache::new(
                (
                    2,
                    self.cfg.model.decoder.kv_heads,
                    max_audio_length,
                    self.cfg.model.decoder.gqa_head_dim,
                ),
                self.dtype,
                &self.device,
            )?));
        }

        Ok(PrepareGenerationOutput {
            generated_tokens,
            decoder_attn_mask,
            encoder_out,
            encoder_positions,
            cross_cache: decoder_cross_attn_cache,
            self_cache: decoder_self_attn_cache,
        })
    }

    fn sample_next_token(
        &self,
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
        cfg_filter_top_k: Option<usize>,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        if temperature == 0. {
            return logits.argmax(D::Minus1)?.to_vec1();
        }

        let logits = candle_nn::ops::softmax_last_dim(
            &(logits.to_dtype(DType::F32)? / temperature as f64)?,
        )?;
        let batch_logits: Vec<Vec<f32>> = logits.to_vec2::<f32>()?;

        let mut sampled = Vec::with_capacity(batch_logits.len());
        let audio_eos_value = self.cfg.data.audio_eos_value as usize;

        for mut probs in batch_logits {
            let mut argsort_indices: Vec<usize> = (0..probs.len()).collect();
            argsort_indices.sort_unstable_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

            // ---- Mask out EOS unless it is the highest ----
            if !argsort_indices.is_empty() && argsort_indices[0] != audio_eos_value {
                probs[audio_eos_value] = 0.0;
            }

            // Top-k
            if let Some(cfg_filter_top_k) = cfg_filter_top_k {
                // Clamp smaller probabilities to zero.
                for (index, val) in argsort_indices.iter().enumerate() {
                    if index >= cfg_filter_top_k {
                        probs[*val] = 0.0;
                    }
                }
            }

            // Top-p
            let mut cumsum = 0.;
            for index in &argsort_indices {
                if cumsum >= top_p {
                    probs[*index] = 0.0;
                } else {
                    cumsum += probs[*index];
                }
            }

            let distr = WeightedIndex::new(&probs).map_err(candle_core::Error::msg)?;
            sampled.push(distr.sample(rng) as u32);
        }
        Ok(sampled)
    }

    #[allow(clippy::too_many_arguments)]
    fn decoder_step(
        &self,
        tokens: &Tensor,
        encoder_out: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        self_attn_cache: &mut [Option<DiaKvCache>],
        cross_attn_cache: &mut [Option<DiaKvCache>],
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        cfg_filter_top_k: Option<usize>,
        rng: &mut Isaac64Rng,
        current_idx: usize,
    ) -> Result<Vec<u32>> {
        let audio_eos_value = self.cfg.data.audio_eos_value as usize;

        // println!("{tokens}");
        let mut logits = self.model.decoder.decode_step(
            tokens,
            encoder_out,
            self_attn_mask,
            cross_attn_mask,
            encoder_positions,
            decoder_positions,
            self_attn_cache,
            cross_attn_cache,
            current_idx,
        )?;

        let logits_last = logits.i((.., logits.dim(1)? - 1.., .., ..))?.squeeze(1)?;
        let uncond_logits = logits_last.i((0, .., ..))?.squeeze(0)?;
        let cond_logits = logits_last.i((1, .., ..))?.squeeze(0)?;

        logits = (&cond_logits + (cfg_scale as f64 * (&cond_logits - uncond_logits)?)?)?;
        // logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
        // logits_CxV[1:, audio_eos_value:] = -torch.inf
        logits = logits.slice_assign(
            &[&.., &(audio_eos_value + 1..)],
            &(Tensor::ones(
                (logits.dim(0)?, logits.dim(1)? - (audio_eos_value + 1)),
                logits.dtype(),
                logits.device(),
            )? * f64::NEG_INFINITY)?,
        )?;
        logits = logits.slice_assign(
            &[&(1..), &(audio_eos_value..)],
            &(Tensor::ones(
                (logits.dim(0)? - 1, logits.dim(1)? - audio_eos_value),
                logits.dtype(),
                logits.device(),
            )? * f64::NEG_INFINITY)?,
        )?;

        let next = self.sample_next_token(&logits, temperature, top_p, cfg_filter_top_k, rng)?;
        Ok(next)
    }

    fn generate_output(&self, generated_codes: &Tensor) -> Result<Vec<f32>> {
        let num_channels = self.cfg.data.channels;
        let seq_length = generated_codes.dim(0)?;
        let audio_pad_value = self.cfg.data.audio_pad_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let revert_precomp = build_revert_indices(
            1,
            seq_length,
            num_channels,
            &delay_pattern.iter().map(|x| *x as i64).collect::<Vec<_>>(),
            &self.device,
        )?;

        let mut codebook = revert_audio_delay(
            &generated_codes.unsqueeze(0)?,
            audio_pad_value as i64,
            &revert_precomp,
            seq_length,
        )?;
        codebook = codebook.i((.., ..codebook.dim(1)? - max_delay_pattern, ..))?;

        let min_valid_index = 0f64;
        let max_valid_index = 1023f64;

        // Original code scatters values where the below `invalid_mask` is true to 0.
        // We do the opposite inverse.
        let invalid_mask = codebook
            .lt(min_valid_index)?
            .bitwise_or(&codebook.gt(max_valid_index)?)?;
        codebook = invalid_mask.where_cond(&codebook.zeros_like()?, &codebook)?;

        let codes = codebook.transpose(1, 2)?;
        let pcm = self.dac.decode_codes(&codes.to_dtype(DType::U32)?)?;
        let pcm = pcm.i((0, 0))?;
        let pcm = normalize_loudness(&pcm, RATE as u32, true)?;
        let pcm = pcm.to_vec1::<f32>()?;

        Ok(pcm)
    }

    pub fn generate(
        &self,
        text: &str,
        cfg: &SpeechGenerationConfig,
    ) -> Result<SpeechGenerationOutput> {
        let SpeechGenerationConfig::Dia {
            max_tokens,
            cfg_scale,
            temperature,
            top_p,
            top_k,
        } = cfg;

        let audio_pad_value = self.cfg.data.audio_pad_value as u32;
        let audio_eos_value = self.cfg.data.audio_eos_value as u32;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_tokens = max_tokens.unwrap_or(self.cfg.data.audio_length);
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let PrepareGenerationOutput {
            mut generated_tokens,
            decoder_attn_mask,
            encoder_out,
            encoder_positions,
            cross_cache: mut decoder_cross_attn_cache,
            self_cache: mut decoder_self_attn_cache,
        } = self.prepare_generation(text)?;

        let self_attn_mask = apply_triangular(
            &Tensor::ones(
                (self.cfg.data.audio_length, self.cfg.data.audio_length),
                DType::U8,
                decoder_attn_mask.device(),
            )?,
            0,
            false,
        )?;

        let mut dec_step = 0;

        let mut bos_countdown = max_delay_pattern;
        let mut eos_detected = false;
        let mut eos_countdown: Option<usize> = None;

        let mut rng = Isaac64Rng::seed_from_u64(0);

        let mut start = Instant::now();
        while dec_step < max_tokens {
            let dec_positions = Tensor::full(dec_step as f32, (2, 1), &self.device)?;
            let current_tokens = generated_tokens
                .i((dec_step..dec_step + 1, ..))?
                .unsqueeze(0)?
                .repeat((2, 1, 1))?;

            let mut pred_c = self.decoder_step(
                &current_tokens.to_dtype(DType::U32)?,
                &encoder_out,
                Some(&self_attn_mask),
                Some(&decoder_attn_mask),
                &encoder_positions,
                &dec_positions,
                &mut decoder_self_attn_cache,
                &mut decoder_cross_attn_cache,
                *cfg_scale,
                *temperature,
                *top_p,
                *top_k,
                &mut rng,
                dec_step,
            )?;

            if (!eos_detected && pred_c[0] == audio_eos_value)
                || dec_step == max_tokens - max_delay_pattern - 1
            {
                eos_detected = true;
                eos_countdown = Some(max_delay_pattern);
            }

            if let Some(eos_countdown) = &mut eos_countdown {
                let step_after_eos = max_delay_pattern - *eos_countdown;
                for (i, d) in delay_pattern.iter().enumerate() {
                    match step_after_eos.cmp(&(*d as usize)) {
                        std::cmp::Ordering::Equal => pred_c[i] = audio_eos_value,
                        std::cmp::Ordering::Greater => pred_c[i] = audio_pad_value,
                        std::cmp::Ordering::Less => {}
                    }
                }
                *eos_countdown -= 1;
            }

            bos_countdown = bos_countdown.saturating_sub(1);

            let apply_mask = bos_countdown > 0;
            if apply_mask {
                let len = pred_c.len();
                let dec_out = Tensor::from_vec(pred_c, len, &self.device)?.to_dtype(DType::F32)?;

                let mask = generated_tokens
                    .i((dec_step + 1..dec_step + 2, ..))?
                    .eq(-1.)?;
                generated_tokens = generated_tokens.slice_assign(
                    &[&(dec_step + 1..dec_step + 2), &..],
                    &mask.where_cond(
                        &dec_out.unsqueeze(0)?,
                        &generated_tokens.i((dec_step + 1..dec_step + 2, ..))?,
                    )?,
                )?;
            } else {
                let len = pred_c.len();
                generated_tokens.slice_set(
                    &Tensor::from_vec(pred_c, len, &self.device)?
                        .unsqueeze(0)?
                        .to_dtype(DType::F32)?,
                    0,
                    dec_step + 1,
                )?;
            }

            if eos_countdown.is_some_and(|eos_countdown| eos_countdown == 0) {
                break;
            }

            dec_step += 1;

            let end = Instant::now();
            if dec_step % TOKENS_PER_SECOND == 0 {
                info!(
                    "Generated {}s of audio, {dec_step} tokens at {:.2} tokens/second.",
                    dec_step / TOKENS_PER_SECOND,
                    TOKENS_PER_SECOND as f32 / (end - start).as_secs_f32()
                );
                start = end;
            }
        }

        let generated_codes = generated_tokens.i((0..dec_step + 1, ..))?;
        let pcm = self.generate_output(&generated_codes)?;
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: RATE,
            channels: CHANNELS,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
