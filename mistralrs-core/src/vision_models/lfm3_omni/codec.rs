use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};
use mistralrs_quant::ShardedVarBuilder;
use rustfft::{num_complex::Complex32, FftPlanner};
use serde::{Deserialize, Serialize};

use super::generation::{AuxiliaryConfig, AuxiliaryLfm3};
use crate::{
    layers::linear,
    models::lfm3::{LayerType, RopeParameters},
    utils::unvarbuilder::UnVarBuilder,
};

const CODEC_INTERMEDIATE_SIZE: usize = 2304;
const CODEC_LAYERS: usize = 8;
const CODEC_HEADS: usize = 8;
const CODEC_KV_HEADS: usize = 2;
const CODEC_HEAD_DIM: usize = 64;
const CODEC_CONV_WIDTH: usize = 7;
const CODEC_NORM_EPS: f64 = 1e-5;
const CODEC_ROPE_THETA: f64 = 1_000_000.0;
const CODEC_ATTENTION_LAYERS: [usize; 3] = [2, 4, 6];
const CODEC_DECODE_CHUNK_FRAMES: usize = 32;
const DECODER_HIDDEN_BUFFERS: usize = 16;
const DECODER_INTERMEDIATE_BUFFERS: usize = 3;
const ATTENTION_FLOAT32_BUFFERS: usize = 2;
const PCM_BUFFERS: usize = 3;
const CODE_BUFFERS: usize = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct CodecConfig {
    pub encoder_dim: usize,
    pub stft_n_fft: usize,
    pub stft_hop_length: usize,
    pub stft_win_length: usize,
    pub stft_frame_stack: usize,
    pub quantizer_dim: usize,
    pub fsq_levels: Vec<u32>,
    pub n_q: usize,
    pub decoder_window: usize,
    pub sampling_rate: u32,
}

impl CodecConfig {
    fn context_frames(&self) -> usize {
        let convolution = (CODEC_LAYERS - CODEC_ATTENTION_LAYERS.len()) * (CODEC_CONV_WIDTH - 1);
        let attention = CODEC_ATTENTION_LAYERS.len() * self.decoder_window;
        (convolution + attention).div_ceil(self.stft_frame_stack)
    }

    pub(crate) fn activation_memory_bytes(&self, frames: usize, dtype: DType) -> usize {
        let steps = frames * self.stft_frame_stack;
        let chunk =
            (CODEC_DECODE_CHUNK_FRAMES + self.context_frames()).min(frames) * self.stft_frame_stack;
        let neural = frames * (self.encoder_dim + self.quantizer_dim) * dtype.size_in_bytes()
            + frames
                * self.n_q
                * (CODE_BUFFERS * DType::U32.size_in_bytes()
                    + self.fsq_levels.len() * DType::F32.size_in_bytes())
            + chunk
                * (DECODER_HIDDEN_BUFFERS * self.encoder_dim
                    + DECODER_INTERMEDIATE_BUFFERS * CODEC_INTERMEDIATE_SIZE)
                * dtype.size_in_bytes()
            + chunk
                * chunk
                * CODEC_HEADS
                * (ATTENTION_FLOAT32_BUFFERS * DType::F32.size_in_bytes() + dtype.size_in_bytes());
        let spectrum = steps * 2 * (self.stft_n_fft / 2 + 1) * DType::F32.size_in_bytes();
        let pcm = PCM_BUFFERS * steps * self.stft_hop_length * DType::F32.size_in_bytes();
        neural + spectrum + pcm
    }

    pub(crate) fn parameter_count(&self) -> usize {
        let effective = self.n_q * self.fsq_levels.len();
        let project =
            usize::from(effective != self.quantizer_dim) * (effective + 1) * self.quantizer_dim;
        project
            + (self.quantizer_dim + 1) * self.encoder_dim
            + self.decoder_config().parameter_count()
            + (self.encoder_dim + 1) * 2 * (self.stft_n_fft / 2 + 1)
            + self.stft_win_length
    }

    fn decoder_config(&self) -> AuxiliaryConfig {
        AuxiliaryConfig {
            hidden_size: self.encoder_dim,
            intermediate_size: CODEC_INTERMEDIATE_SIZE,
            num_hidden_layers: CODEC_LAYERS,
            num_attention_heads: CODEC_HEADS,
            num_key_value_heads: CODEC_KV_HEADS,
            head_dim: CODEC_HEAD_DIM,
            norm_eps: CODEC_NORM_EPS,
            conv_l_cache: CODEC_CONV_WIDTH,
            conv_bias: false,
            use_sandwich_norm: true,
            layer_types: (0..CODEC_LAYERS)
                .map(|index| {
                    if CODEC_ATTENTION_LAYERS.contains(&index) {
                        LayerType::SlidingAttention
                    } else {
                        LayerType::Conv
                    }
                })
                .collect(),
            sliding_window: self.decoder_window,
            rope_parameters: HashMap::from([(
                "sliding_attention".to_string(),
                RopeParameters {
                    rope_theta: CODEC_ROPE_THETA,
                    rope_type: "default".to_string(),
                    partial_rotary_factor: 1.0,
                },
            )]),
        }
    }
}

pub(crate) struct LiquidCodec {
    config: CodecConfig,
    project_out: Option<Linear>,
    from_quantizer: Linear,
    decoder: AuxiliaryLfm3,
    spectrum: Linear,
    window: Tensor,
}

impl LiquidCodec {
    pub(crate) fn new(config: CodecConfig, vb: ShardedVarBuilder) -> Result<Self> {
        if config.stft_win_length != config.stft_n_fft
            || config.stft_hop_length == 0
            || config.stft_hop_length > config.stft_win_length
            || config.fsq_levels.iter().any(|&level| level < 2)
        {
            candle_core::bail!("Unsupported LiquidCodec synthesis configuration");
        }
        let effective = config.n_q * config.fsq_levels.len();
        Ok(Self {
            project_out: (effective != config.quantizer_dim)
                .then(|| {
                    linear(
                        effective,
                        config.quantizer_dim,
                        vb.pp("quantizer").pp("project_out"),
                    )
                })
                .transpose()?,
            from_quantizer: linear(
                config.quantizer_dim,
                config.encoder_dim,
                vb.pp("from_quantizer"),
            )?,
            decoder: AuxiliaryLfm3::new(
                &config.decoder_config(),
                vb.pp("decoder").pp("lfm"),
                true,
                true,
            )?,
            spectrum: linear(
                config.encoder_dim,
                2 * (config.stft_n_fft / 2 + 1),
                vb.pp("decoder").pp("lin"),
            )?,
            window: vb.pp("decoder.istft").get_with_hints_dtype(
                config.stft_win_length,
                "window",
                Default::default(),
                DType::F32,
            )?,
            config,
        })
    }

    pub(crate) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        if let Some(project) = &self.project_out {
            uvb.pp("quantizer.project_out").add(project);
        }
        uvb.pp("from_quantizer").add(&self.from_quantizer);
        self.decoder.residual_tensors(uvb.pp("decoder.lfm"));
        uvb.pp("decoder.lin").add(&self.spectrum);
        uvb.pp("decoder.istft")
            .add_tensor("window", self.window.clone());
        uvb.to_safetensors()
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        self.config.sampling_rate
    }

    pub(crate) fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (batch, books, frames) = codes.dims3()?;
        if books != self.config.n_q {
            candle_core::bail!(
                "LiquidCodec expected {} codebooks, got {books}",
                self.config.n_q
            );
        }
        if frames == 0 {
            return Tensor::zeros((batch, 0), DType::F32, &Device::Cpu);
        }
        let codes = codes
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .to_vec3::<u32>()?;
        let latents = decode_fsq(&codes, &self.config.fsq_levels)?;
        let weight = self.from_quantizer.weight();
        let mut latents = Tensor::from_vec(
            latents,
            (batch, frames, books * self.config.fsq_levels.len()),
            weight.device(),
        )?
        .to_dtype(weight.dtype())?;
        if let Some(project) = &self.project_out {
            latents = project.forward(&latents)?;
        }
        let embeddings = self.from_quantizer.forward(&latents)?;
        let spectrum = self.synthesize(&embeddings, CODEC_DECODE_CHUNK_FRAMES)?;
        let window = self.window.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let values = inverse_stft(&spectrum, &self.config, &window)?;
        Tensor::from_vec(
            values,
            (
                batch,
                frames * self.config.stft_frame_stack * self.config.stft_hop_length,
            ),
            &Device::Cpu,
        )
    }

    fn synthesize(&self, embeddings: &Tensor, chunk_frames: usize) -> Result<Vec<Vec<Vec<f32>>>> {
        let (batch, frames, _) = embeddings.dims3()?;
        let stack = self.config.stft_frame_stack;
        let mut spectrum = vec![Vec::with_capacity(frames * stack); batch];
        for start in (0..frames).step_by(chunk_frames) {
            let end = (start + chunk_frames).min(frames);
            // Summed causal receptive fields make the retained chunk independent of earlier input.
            let context = start.saturating_sub(self.config.context_frames());
            let count = end - context;
            let upsampled = embeddings
                .narrow(1, context, count)?
                .unsqueeze(2)?
                .broadcast_as((batch, count, stack, self.config.encoder_dim))?
                .reshape((batch, count * stack, self.config.encoder_dim))?;
            let states = self.decoder.forward(&upsampled)?;
            let chunk = self
                .spectrum
                .forward(&states.narrow(1, (start - context) * stack, (end - start) * stack)?)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec3::<f32>()?;
            for (output, chunk) in spectrum.iter_mut().zip(chunk) {
                output.extend(chunk);
            }
        }
        Ok(spectrum)
    }
}

fn decode_fsq(codes: &[Vec<Vec<u32>>], levels: &[u32]) -> Result<Vec<f32>> {
    let codebook_size = levels.iter().product::<u32>();
    let mut result = Vec::new();
    for row in codes {
        for frame in 0..row[0].len() {
            for book in row {
                let code = book[frame];
                if code >= codebook_size {
                    candle_core::bail!("LiquidCodec code {code} exceeds its vocabulary; remove terminal speech frames before decoding");
                }
                let mut basis = 1;
                for &level in levels {
                    let half = level / 2;
                    result.push(((code / basis % level) as f32 - half as f32) / half as f32);
                    basis *= level;
                }
            }
        }
    }
    Ok(result)
}

fn inverse_stft(
    spectrum: &[Vec<Vec<f32>>],
    config: &CodecConfig,
    window: &[f32],
) -> Result<Vec<f32>> {
    let n_fft = config.stft_n_fft;
    let bins = n_fft / 2 + 1;
    let frames = spectrum[0].len();
    let full_len = (frames - 1) * config.stft_hop_length + config.stft_win_length;
    let trim = (config.stft_win_length - config.stft_hop_length) / 2;
    let mut envelope = vec![0.0; full_len];
    for frame in 0..frames {
        for (i, &value) in window.iter().enumerate() {
            envelope[frame * config.stft_hop_length + i] += value * value;
        }
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_inverse(n_fft);
    let mut result = Vec::with_capacity(spectrum.len() * (full_len - 2 * trim));
    for row in spectrum {
        let mut audio = vec![0.0; full_len];
        for (frame, values) in row.iter().enumerate() {
            let mut complex = vec![Complex32::new(0.0, 0.0); n_fft];
            for k in 0..bins {
                complex[k] = Complex32::from_polar(values[k].exp(), values[bins + k]);
            }
            complex[0].im = 0.0;
            if n_fft % 2 == 0 {
                complex[n_fft / 2].im = 0.0;
            }
            for k in 1..(n_fft + 1) / 2 {
                complex[n_fft - k] = complex[k].conj();
            }
            fft.process(&mut complex);
            for (i, value) in complex.iter().enumerate() {
                audio[frame * config.stft_hop_length + i] += value.re / n_fft as f32 * window[i];
            }
        }
        result.extend((trim..full_len - trim).map(|i| audio[i] / envelope[i]));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_mixed_radix_and_terminal_code() -> Result<()> {
        let decoded = decode_fsq(&[vec![vec![0, 2015], vec![1, 8]]], &[8, 7, 6, 6])?;
        assert_eq!(
            &decoded[..8],
            &[-1.0, -1.0, -1.0, -1.0, -0.75, -1.0, -1.0, -1.0]
        );
        assert_eq!(&decoded[8..12], &[0.75, 1.0, 2.0 / 3.0, 2.0 / 3.0]);
        assert!(decode_fsq(&[vec![vec![2016]]], &[8, 7, 6, 6]).is_err());
        Ok(())
    }

    #[test]
    fn inverse_stft_matches_constant_waveform() -> Result<()> {
        let config = CodecConfig {
            encoder_dim: 1,
            stft_n_fft: 8,
            stft_hop_length: 2,
            stft_win_length: 8,
            stft_frame_stack: 1,
            quantizer_dim: 1,
            fsq_levels: vec![2],
            n_q: 1,
            decoder_window: 1,
            sampling_rate: 24_000,
        };
        // A periodic Hann window has DFT coefficients [N/2, -N/4, 0, ...].
        let mut frame = vec![f32::NEG_INFINITY; 10];
        frame[0] = 4.0f32.ln();
        frame[1] = 2.0f32.ln();
        frame[5..].fill(0.0);
        frame[6] = std::f32::consts::PI;
        let window = (0..config.stft_win_length)
            .map(|i| {
                0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / config.stft_win_length as f32).cos()
            })
            .collect::<Vec<_>>();
        let result = inverse_stft(&[vec![frame; 4]], &config, &window)?;
        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|sample| (*sample - 1.0).abs() < 1e-5));
        Ok(())
    }

    #[test]
    #[ignore = "requires downloaded Omni checkpoint and PyTorch parity tensors"]
    fn codec_matches_reference() -> anyhow::Result<()> {
        let directory = std::path::PathBuf::from(std::env::var("MISTRALRS_LFM3_OMNI_CHECKPOINT")?)
            .join("audio_tokenizer");
        let fixture = std::env::var("MISTRALRS_LFM3_OMNI_REFERENCE")?;
        let config: CodecConfig =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let vb = unsafe {
            mistralrs_quant::ShardedSafeTensors::sharded(
                &[directory.join("model.safetensors")],
                DType::F32,
                &Device::Cpu,
                None,
                std::sync::Arc::new(|_| true),
            )?
        };
        let parameter_count = config.parameter_count();
        let model = LiquidCodec::new(config, vb)?;
        assert_eq!(
            model
                .residual_tensors()
                .iter()
                .map(|(_, tensor)| tensor.elem_count())
                .sum::<usize>(),
            parameter_count
        );
        let fixture = candle_core::safetensors::load(fixture, &Device::Cpu)?;
        let output = model.decode(&fixture["codec_codes"])?;
        let error = (&output - &fixture["codec_pcm"])?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 2e-4, "maximum codec sample error: {error}");
        let frames = 2 * CODEC_DECODE_CHUNK_FRAMES + 3;
        let embeddings = (Tensor::arange(
            0u32,
            (frames * model.config.encoder_dim) as u32,
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?
        .sin()?
            * 0.2)?
            .reshape((1, frames, model.config.encoder_dim))?;
        let full = model.synthesize(&embeddings, frames)?;
        let chunked = model.synthesize(&embeddings, CODEC_DECODE_CHUNK_FRAMES)?;
        let error = full
            .into_iter()
            .flatten()
            .flatten()
            .zip(chunked.into_iter().flatten().flatten())
            .map(|(full, chunked)| (full - chunked).abs())
            .fold(0.0f32, f32::max);
        assert!(
            error < 2e-4,
            "maximum chunked codec spectrum error: {error}"
        );
        Ok(())
    }
}
