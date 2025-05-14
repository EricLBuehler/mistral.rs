use std::io::Write;

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

use crate::{
    layers_masker::masked_fill,
    ops::{TopKLastDimOp, TopKOutput},
};

pub use config::DiaConfig;

use super::bs1770;

mod audio;
mod cache;
mod config;
mod dac;
mod model;

fn normalize_loudness(wav: &Tensor, sample_rate: u32, loudness_compressor: bool) -> Result<Tensor> {
    let energy = wav.sqr()?.mean_all()?.sqrt()?.to_vec0::<f32>()?;
    if energy < 2e-3 {
        return Ok(wav.clone());
    }
    let wav_array = wav.to_vec1::<f32>()?;
    let mut meter = bs1770::ChannelLoudnessMeter::new(sample_rate);
    meter.push(wav_array.into_iter());
    let power = meter.as_100ms_windows();
    let loudness = match bs1770::gated_mean(power) {
        None => return Ok(wav.clone()),
        Some(gp) => gp.loudness_lkfs() as f64,
    };
    let delta_loudness = -14. - loudness;
    let gain = 10f64.powf(delta_loudness / 20.);
    let wav = (wav * gain)?;
    if loudness_compressor {
        wav.tanh()
    } else {
        Ok(wav)
    }
}

pub trait Sample {
    fn to_i16(&self) -> i16;
}

impl Sample for f32 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for f64 {
    fn to_i16(&self) -> i16 {
        (self.clamp(-1.0, 1.0) * 32767.0) as i16
    }
}

impl Sample for i16 {
    fn to_i16(&self) -> i16 {
        *self
    }
}

fn write_pcm_as_wav<W: Write, S: Sample>(
    w: &mut W,
    samples: &[S],
    sample_rate: u32,
) -> std::io::Result<()> {
    let len = 12u32; // header
    let len = len + 24u32; // fmt
    let len = len + samples.len() as u32 * 2 + 8; // data
    let n_channels = 1u16;
    let bytes_per_second = sample_rate * 2 * n_channels as u32;
    w.write_all(b"RIFF")?;
    w.write_all(&(len - 8).to_le_bytes())?; // total length minus 8 bytes
    w.write_all(b"WAVE")?;

    // Format block
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // block len minus 8 bytes
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&n_channels.to_le_bytes())?; // one channel
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&bytes_per_second.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // 2 bytes of data per sample
    w.write_all(&16u16.to_le_bytes())?; // bits per sample

    // Data block
    w.write_all(b"data")?;
    w.write_all(&(samples.len() as u32 * 2).to_le_bytes())?;
    for sample in samples.iter() {
        w.write_all(&sample.to_i16().to_le_bytes())?
    }
    Ok(())
}

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
        let dac = dac::Model::new(&dac::Config::dia(), dac_vb)?;

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

        let delay_pad_tensor =
            (Tensor::ones((max_delay_pattern, num_channels), DType::F32, &self.device)? * 1024f64)?;
        let prefill = Tensor::cat(&[prefill, delay_pad_tensor], 0)?;

        let delay_precomp = build_delay_indices(
            1,
            prefill.dim(0)?,
            num_channels,
            delay_pattern,
            &self.device,
        )?;

        let prefill = apply_audio_delay(
            &prefill.unsqueeze(0)?,
            audio_pad_value,
            audio_bos_value,
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

        println!("{padded_text_np}");
        padded_text_np.to_dtype(DType::U32)?.unsqueeze(0)
    }

    /// Returns:
    /// - generated tokens
    /// - decoder attn mask
    /// - encoder out
    /// - encoder positions
    /// - cross cache
    /// - self cache
    fn prepare_generation(
        &self,
        text: &str,
    ) -> Result<(
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Vec<Option<DiaKvCache>>,
        Vec<Option<DiaKvCache>>,
    )> {
        let enc_input_cond = self.prepare_text_prompt(text)?;
        let enc_input_uncond = enc_input_cond.zeros_like()?;
        let enc_input = Tensor::cat(&[&enc_input_cond, &enc_input_uncond], 0)?;

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
        println!("{encoder_out}");

        let decoder_cross_attn_cache = self
            .model
            .decoder
            .precompute_cross_attn_cache(&encoder_out, &encoder_positions)?;
        let decoder_padding_mask = Tensor::ones((2, 1), DType::U8, &self.device)?;
        let decoder_attn_mask = create_attn_mask(&decoder_padding_mask, &encoder_padding_mask)?;

        let max_audio_length = self.cfg.data.audio_length;
        let generated_tokens = Tensor::zeros(
            (max_audio_length, self.cfg.data.channels),
            DType::U32,
            &self.device,
        )?;

        generated_tokens.slice_set(&prefill.to_dtype(DType::U32)?, 0, 0)?;

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

        Ok((
            generated_tokens,
            decoder_attn_mask,
            encoder_out,
            encoder_positions,
            decoder_cross_attn_cache,
            decoder_self_attn_cache,
        ))
    }

    fn sample_next_token(
        &self,
        logits: &Tensor,
        temperature: f32,
        top_p: f32,
        cfg_filter_top_k: Option<usize>,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        assert_eq!(logits.rank(), 2);

        // if temperature == 0. {
        return logits.argmax(D::Minus1)?.to_vec1::<u32>();
        // }

        let mut logits = (logits / temperature as f64)?;
        if let Some(cfg_filter_top_k) = cfg_filter_top_k {
            let TopKOutput {
                values: _,
                indices: top_k_indices,
            } = logits.topk(cfg_filter_top_k)?;
            let mut mask = logits.ones_like()?;
            mask = mask.scatter_add(&top_k_indices, &mask.gather(&top_k_indices, 0)?.neg()?, 0)?;
            logits = masked_fill(&logits, &mask, f32::NEG_INFINITY)?;
        }

        if top_p < 1. {
            let probs = candle_nn::ops::softmax_last_dim(&logits)?;
            let (sorted_probs, sorted_indices) = probs.sort_last_dim(false)?;
            let cumulative_probs = sorted_probs.cumsum(D::Minus1)?;

            let mut sorted_indices_to_remove = cumulative_probs.ge(top_p as f64)?;
            sorted_indices_to_remove = sorted_indices_to_remove.slice_assign(
                &[&.., &(1..)],
                &sorted_indices_to_remove
                    .i((.., ..sorted_indices_to_remove.dim(D::Minus1)? - 1))?,
            )?;
            sorted_indices_to_remove = sorted_indices_to_remove.slice_assign(
                &[&.., &0],
                &sorted_indices_to_remove.i((.., 0))?.zeros_like()?,
            )?;

            let mut indices_to_remove = sorted_indices_to_remove.zeros_like()?;
            indices_to_remove = indices_to_remove.scatter_add(
                &sorted_indices_to_remove,
                &sorted_indices,
                D::Minus1,
            )?;
            logits = masked_fill(&logits, &indices_to_remove, f32::NEG_INFINITY)?;
        }

        logits = candle_nn::ops::softmax_last_dim(&logits)?;

        let mut sampled = Vec::new();
        for mut logits in logits.chunk(logits.dim(0)?, 0)? {
            logits = logits.squeeze(0)?;

            let probs = logits.to_vec1::<f32>()?;

            let distr = WeightedIndex::new(&probs).map_err(candle_core::Error::msg)?;

            let next_token = distr.sample(rng);
            sampled.push(next_token as u32);
        }
        Ok(sampled)
    }

    fn decoder_step(
        &self,
        tokens: &Tensor,
        encoder_out: &Tensor,
        cross_attn_mask: Option<&Tensor>,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        self_attn_cache: &mut Vec<Option<DiaKvCache>>,
        cross_attn_cache: &mut Vec<Option<DiaKvCache>>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        cfg_filter_top_k: Option<usize>,
        rng: &mut Isaac64Rng,
    ) -> Result<Vec<u32>> {
        let audio_eos_value = self.cfg.data.audio_eos_value as usize;

        // println!("{tokens}");
        let mut logits = self.model.decoder.decode_step(
            tokens,
            encoder_out,
            cross_attn_mask,
            encoder_positions,
            decoder_positions,
            self_attn_cache,
            cross_attn_cache,
        )?;
        // println!("{logits}");
        // todo!();

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

        self.sample_next_token(&logits, temperature, top_p, cfg_filter_top_k, rng)
    }

    fn generate_output(&self, generated_codes: &Tensor) -> Result<Tensor> {
        let num_channels = self.cfg.data.channels;
        let seq_length = generated_codes.dim(0)?;
        let audio_pad_value = self.cfg.data.audio_pad_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let revert_precomp =
            build_revert_indices(1, seq_length, num_channels, delay_pattern, &self.device)?;

        let mut codebook = revert_audio_delay(
            &generated_codes.unsqueeze(0)?,
            audio_pad_value,
            &revert_precomp,
            seq_length,
        )?;
        codebook = codebook.i((.., codebook.dim(1)? - max_delay_pattern.., ..))?;

        let min_valid_index = 0f64;
        let max_valid_index = 1023f64;

        // Original code scatters values where the below `invalid_mask` is true to 0.
        // We do the opposite inverse.
        let invalid_mask = codebook
            .lt(min_valid_index)?
            .bitwise_or(&codebook.gt(max_valid_index)?)?;
        codebook = invalid_mask.where_cond(&codebook.zeros_like()?, &codebook)?;

        let codes = codebook.transpose(1, 2)?;
        let pcm = self.dac.decode_codes(&codes)?;
        let pcm = pcm.i((0, 0))?;
        let pcm = normalize_loudness(&pcm, 44_100, true)?;
        let pcm = pcm.to_vec1::<f32>()?;
        let mut output = std::fs::File::create(&"out.wav")?;
        write_pcm_as_wav(&mut output, &pcm, 44_100)?;

        todo!()
    }

    pub fn generate(&self, text: &str) -> Result<()> {
        let max_tokens: Option<usize> = None;
        let cfg_scale = 3.0f32;
        let temperature = 1.3f32;
        let top_p = 0.95f32;
        let cfg_filter_top_k = Some(35usize);

        let audio_pad_value = self.cfg.data.audio_pad_value as u32;
        let audio_eos_value = self.cfg.data.audio_eos_value as u32;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_tokens = max_tokens.unwrap_or(self.cfg.data.audio_length);
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let (
            mut generated_tokens,
            decoder_attn_mask,
            encoder_out,
            encoder_positions,
            mut decoder_cross_attn_cache,
            mut decoder_self_attn_cache,
        ) = self.prepare_generation(text)?;

        let mut dec_step = 0;

        let mut bos_countdown = max_delay_pattern;
        let mut eos_detected = false;
        let mut eos_countdown: Option<usize> = None;

        let mut rng = Isaac64Rng::seed_from_u64(0);

        while dec_step < max_tokens {
            let dec_positions = Tensor::full(dec_step as f32, (2, 1), &self.device)?;
            let current_tokens = generated_tokens
                .i((dec_step..dec_step + 1, ..))?
                .unsqueeze(0)?
                .repeat((2, 1, 1))?;

            let mut pred_c = self.decoder_step(
                &current_tokens,
                &encoder_out,
                Some(&decoder_attn_mask),
                &encoder_positions,
                &dec_positions,
                &mut decoder_self_attn_cache,
                &mut decoder_cross_attn_cache,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                &mut rng,
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
                    if step_after_eos == *d as usize {
                        pred_c[i] = audio_eos_value;
                    } else if step_after_eos > *d as usize {
                        pred_c[i] = audio_pad_value;
                    }
                }
                *eos_countdown -= 1;
            }

            bos_countdown = bos_countdown.saturating_sub(1);

            let apply_mask = bos_countdown > 0;
            if apply_mask {
                let len = pred_c.len();
                let dec_out = Tensor::from_vec(pred_c, len, &self.device)?;

                let mask = generated_tokens
                    .i((dec_step + 1..dec_step + 2, ..))?
                    .eq(1024.)?;
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
                    &Tensor::from_vec(pred_c, len, &self.device)?.unsqueeze(0)?,
                    0,
                    dec_step + 1,
                )?;
            }

            if eos_countdown.is_some_and(|eos_countdown| eos_countdown == 0) {
                break;
            }

            dec_step += 1;

            if dec_step % 86 == 0 {
                println!("Generated 1s")
            }
        }

        let generated_codes = generated_tokens.i((0..dec_step + 1, ..))?;
        self.generate_output(&generated_codes)?;
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
