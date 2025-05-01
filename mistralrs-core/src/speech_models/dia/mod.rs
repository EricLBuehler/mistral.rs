use audio::{apply_audio_delay, build_delay_indices};
use cache::DiaKvCache;
use candle_core::{DType, Device, Result, Tensor};
use config::DiaConfig;
use mistralrs_quant::{BitWiseOp, ShardedVarBuilder};
use model::DiaModel;

mod audio;
mod cache;
mod config;
mod model;

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
}

impl DiaPipeline {
    pub fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            dtype: vb.dtype(),
            device: vb.device().clone(),
            model: DiaModel::new(cfg, vb)?,
            cfg: cfg.clone(),
        })
    }

    fn prepare_audio_prompt(&self) -> Result<Tensor> {
        let num_channels = self.cfg.data.channels;
        let audio_pad_value = self.cfg.data.audio_pad_value;
        let audio_bos_value = self.cfg.data.audio_bos_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_delay_pattern = *delay_pattern.iter().max().unwrap() as usize;

        let prefill =
            (Tensor::ones((1, num_channels), DType::I32, &self.device)? * audio_bos_value as f64)?;

        let delay_pad_tensor =
            Tensor::ones((max_delay_pattern, num_channels), DType::I32, &self.device)?.neg()?;
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
        )?;

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

        padded_text_np.to_dtype(DType::I64)?.unsqueeze(0)
    }

    /// Returns:
    /// - generated tokens
    /// - decoder attn mask
    /// - encoder out
    /// - cross cache
    /// - self cache
    fn prepare_generation(
        &self,
        text: &str,
    ) -> Result<(
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

        let encoder_positions = &[self.cfg.data.text_length];
        let encoder_padding_mask =
            Tensor::arange(0f32, self.cfg.data.text_length as f32, &self.device)?
                .unsqueeze(2)?
                .broadcast_as((2, self.cfg.data.text_length))?
                .ne(&enc_input_cond)?;
        let encoder_attn_mask = create_attn_mask(&encoder_padding_mask, &encoder_padding_mask)?;
        let encoder_out =
            self.model
                .encoder
                .forward(&enc_input, encoder_positions, Some(&encoder_attn_mask))?;

        let decoder_cross_attn_cache = self
            .model
            .decoder
            .precompute_cross_attn_cache(&encoder_out, encoder_positions)?;
        let decoder_padding_mask = Tensor::ones((2, 1), DType::U8, &self.device)?;
        let decoder_attn_mask = create_attn_mask(&decoder_padding_mask, &encoder_padding_mask)?;

        let max_audio_length = self.cfg.data.audio_length;
        let generated_tokens = Tensor::ones(
            (max_audio_length, self.cfg.data.channels),
            DType::I32,
            &self.device,
        )?
        .neg()?;

        generated_tokens.slice_set(&prefill, 0, 0)?;

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
            decoder_cross_attn_cache,
            decoder_self_attn_cache,
        ))
    }

    pub fn generate(&self, text: &str) -> Result<()> {
        let max_tokens: Option<usize> = None;

        let audio_pad_value = self.cfg.data.audio_pad_value;
        let audio_eos_value = self.cfg.data.audio_eos_value;
        let delay_pattern = &self.cfg.data.delay_pattern;
        let max_tokens = max_tokens.unwrap_or(self.cfg.data.audio_length);
        let max_delay_pattern = delay_pattern.iter().max().unwrap();

        let (
            generated_tokens,
            decoder_attn_mask,
            encoder_out,
            decoder_cross_attn_cache,
            decoder_self_attn_cache,
        ) = self.prepare_generation(text)?;

        let mut dec_step = 0;

        let mut bos_countdown = max_delay_pattern;
        let mut eos_detected = false;
        let mut eos_countdown = -1;

        while dec_step < max_tokens {
            let dec_positions = &[dec_step + 1];
        }
        Ok(())
    }
}
