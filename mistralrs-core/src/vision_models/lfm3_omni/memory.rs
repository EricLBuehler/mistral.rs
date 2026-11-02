use candle_core::{DType, Device, Result};

use super::{
    audio::AudioEncoderConfig,
    audio_processing::{AudioProcessor, ENCODER_STRIDE},
};

const DOWNSAMPLE_STAGES: usize = 3;
const DOWNSAMPLE_STRIDE: usize = 2;
const CONV_KERNEL_AREA: usize = 9;
const CONV_INPUT_BUFFERS: usize = 2;
const CONV_OUTPUT_BUFFERS: usize = 3;
const ENCODER_HIDDEN_BUFFERS: usize = 12;
const ENCODER_INTERMEDIATE_BUFFERS: usize = 3;
const PROJECTOR_HIDDEN_BUFFERS: usize = 3;
const FLOAT32_HIDDEN_BUFFERS: usize = 4;
const ATTENTION_FLOAT32_BUFFERS: usize = 2;
const MASK_FLOAT32_BUFFERS: usize = 2;
const METAL_ATTENTION_HEAD_DIMS: &[usize] = &[32, 64, 72, 80, 96, 128, 256, 512];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationMemory {
    pub retained_bytes: usize,
    pub transient_bytes: usize,
}

impl ActivationMemory {
    pub(crate) fn total_bytes(self) -> Result<usize> {
        self.retained_bytes
            .checked_add(self.transient_bytes)
            .ok_or_else(overflow)
    }

    pub(crate) fn combine_serial(self, other: Self) -> Result<Self> {
        Ok(Self {
            retained_bytes: self
                .retained_bytes
                .checked_add(other.retained_bytes)
                .ok_or_else(overflow)?,
            transient_bytes: self.transient_bytes.max(other.transient_bytes),
        })
    }
}

pub(crate) struct AudioMemoryEstimate {
    mel_bins: usize,
    conv_channels: usize,
    dtype_bytes: usize,
    bytes_per_token: usize,
    retained_bytes_per_token: usize,
    attention_bytes_per_pair: usize,
}

impl AudioMemoryEstimate {
    pub(crate) fn new(
        config: &AudioEncoderConfig,
        language_hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let text = &config.text_config;
        let float_bytes = DType::F32.size_in_bytes();
        let dtype_bytes = dtype.size_in_bytes();
        let flash = device.is_cuda()
            && matches!(dtype, DType::F16 | DType::BF16)
            && crate::using_flash_attn()
            && crate::attention::flash_backend_supports_sdpa(text.head_dim(), false, true);
        let fused = flash
            || device.is_cpu()
            || device.is_metal() && METAL_ATTENTION_HEAD_DIMS.contains(&text.head_dim());
        let mask_bytes = if flash {
            0
        } else {
            MASK_FLOAT32_BUFFERS * float_bytes + dtype_bytes
        };
        let score_bytes = if fused {
            0
        } else {
            text.num_attention_heads * (ATTENTION_FLOAT32_BUFFERS * float_bytes + dtype_bytes)
        };
        Self {
            mel_bins: config.num_mel_bins,
            conv_channels: config.conv_channels,
            dtype_bytes,
            bytes_per_token: (ENCODER_HIDDEN_BUFFERS * text.hidden_size
                + ENCODER_INTERMEDIATE_BUFFERS * text.intermediate_size()
                + PROJECTOR_HIDDEN_BUFFERS * language_hidden_size)
                * dtype_bytes
                + FLOAT32_HIDDEN_BUFFERS * text.hidden_size * float_bytes,
            retained_bytes_per_token: config.num_mel_bins * ENCODER_STRIDE * float_bytes
                + language_hidden_size * dtype_bytes,
            attention_bytes_per_pair: mask_bytes + score_bytes,
        }
    }

    pub(crate) fn estimate_tokens(&self, tokens: usize) -> Result<usize> {
        let frames = tokens.checked_mul(ENCODER_STRIDE).ok_or_else(overflow)?;
        self.estimate_mel_frames(frames)
    }

    pub(crate) fn estimate_batch(
        &self,
        tokens_per_sequence: usize,
        batch_size: usize,
    ) -> Result<ActivationMemory> {
        // Clips run serially, but their mel inputs and projected cache entries remain resident.
        let retained_per_sequence = tokens_per_sequence
            .checked_mul(self.retained_bytes_per_token)
            .ok_or_else(overflow)?;
        Ok(ActivationMemory {
            retained_bytes: retained_per_sequence
                .checked_mul(batch_size)
                .ok_or_else(overflow)?,
            transient_bytes: self
                .estimate_tokens(tokens_per_sequence)?
                .checked_sub(retained_per_sequence)
                .ok_or_else(overflow)?,
        })
    }

    fn estimate_mel_frames(&self, frames: usize) -> Result<usize> {
        if frames == 0 {
            return Ok(0);
        }
        let tokens = frames.div_ceil(ENCODER_STRIDE);
        let input = self.mel_bins.checked_mul(frames).ok_or_else(overflow)?;
        let input_bytes = input
            .checked_mul(DType::F32.size_in_bytes() + self.dtype_bytes)
            .ok_or_else(overflow)?;
        let mut channels = 1usize;
        let mut frequency = self.mel_bins;
        let mut time = frames;
        let mut downsample = 0;
        for _ in 0..DOWNSAMPLE_STAGES {
            let input = channels
                .checked_mul(frequency)
                .and_then(|size| size.checked_mul(time))
                .ok_or_else(overflow)?;
            frequency = frequency.div_ceil(DOWNSAMPLE_STRIDE);
            time = time.div_ceil(DOWNSAMPLE_STRIDE);
            let spatial = frequency.checked_mul(time).ok_or_else(overflow)?;
            let output = self
                .conv_channels
                .checked_mul(spatial)
                .ok_or_else(overflow)?;
            // Depthwise convolution can still allocate im2col for every input channel.
            let columns = channels
                .checked_mul(CONV_KERNEL_AREA)
                .and_then(|size| size.checked_mul(spatial))
                .ok_or_else(overflow)?;
            let workspace = input
                .checked_mul(CONV_INPUT_BUFFERS)
                .and_then(|size| {
                    output
                        .checked_mul(CONV_OUTPUT_BUFFERS)
                        .and_then(|output| size.checked_add(output))
                })
                .and_then(|size| size.checked_add(columns))
                .and_then(|size| size.checked_mul(self.dtype_bytes))
                .ok_or_else(overflow)?;
            downsample = downsample.max(workspace);
            channels = self.conv_channels;
        }
        let encoder = tokens
            .checked_mul(self.bytes_per_token)
            .and_then(|size| {
                tokens
                    .checked_mul(tokens)
                    .and_then(|pairs| pairs.checked_mul(self.attention_bytes_per_pair))
                    .and_then(|attention| size.checked_add(attention))
            })
            .ok_or_else(overflow)?;
        input_bytes
            .checked_add(downsample.max(encoder))
            .ok_or_else(overflow)
    }

    pub(crate) fn for_sequence(&self, seq: &crate::sequence::Sequence) -> Result<usize> {
        seq.audios()
            .unwrap_or_default()
            .iter()
            .try_fold(0usize, |total, audio| {
                let tokens =
                    AudioProcessor::num_tokens_for_audio(audio).map_err(candle_core::Error::msg)?;
                total
                    .checked_add(self.estimate_tokens(tokens)?)
                    .ok_or_else(overflow)
            })
    }
}

fn overflow() -> candle_core::Error {
    candle_core::Error::msg("LFM3-Omni activation memory size overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_covers_convolution_columns() -> Result<()> {
        let estimate = AudioMemoryEstimate {
            mel_bins: 128,
            conv_channels: 256,
            dtype_bytes: 2,
            bytes_per_token: 100_000,
            retained_bytes_per_token: 128 * ENCODER_STRIDE * 4 + 2048 * 2,
            attention_bytes_per_pair: 0,
        };
        let frames = 800;
        let tokens = frames / ENCODER_STRIDE;
        let convolution_columns = CONV_KERNEL_AREA * 256 * 32 * (frames / 4) * 2;
        assert!(estimate.estimate_tokens(tokens)? > convolution_columns);
        Ok(())
    }
}
